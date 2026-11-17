"""The longest run of output a schema fixes, and its cap (architecture.md, "Structured output"; operations.md has the
number).

Under the guidance backend's canonical forcing, llguidance scans the bytes a grammar fixes ahead of the current state at
every mask and in the engine's forced prefix, as far as they go (guidance-ai/llguidance#440). Reused definitions and
counted repetition make that text long from a few bytes of schema, and every step of the request's engine waits for the
scan. So each schema a grammar is built from (`response_format`, each tool of a call grammar) is measured first, and one
that fixes more than `MAX_FORCED_RUN` bytes in a row is a 400.

**The count** bounds llguidance's forced bytes at every state, in one pass, each definition once (`Runs`). A piece of
output is summed up as a `Run`: the fixed text at its start and at its end, the longest run inside it, and its whole
length when all of it is fixed. Joining two pieces joins the end of one to the start of the other, so a definition
fixed throughout adds its whole length at each use: the multiplication. A choice (an optional key, a free value,
several values) ends the run that enters it at its alternatives' common prefix, and each alternative's text runs on
into what follows. Objects write their keys in the grammar's order, with its separators.

**Patterns** are read as regular expressions (`_Regex`), fixed from a leading `^`; a choice in one (an alternation, a
repetition's optional copies) ends a run only where the next bytes differ. A length beside a pattern can leave one of
its choices, so it counts the pattern's fixed characters up to that length. A pattern beside a `format` may have
any of its choices ruled out by it, so it counts as fixed throughout, as long as its longest string (`_Regex.beside`).
A pattern in syntax this reader does not take is refused.

**Recursion** is unfolded `MAX_NESTING` times. Past that depth the closing text of each level a document nests adds to
the run, no faster than the model writes levels.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .bounds import MAX_NESTING, BoundError, components, definition

MAX_FORCED_RUN = 4_096  # bytes in a row: the first mask after a choice that enters such a run takes under 3 ms
SAT = 1 << 30  # counts saturate here (minItems and counted repetition multiply)
FORMAT_RUN = 8  # the most a `format` fixes in a row, its quotes included (llguidance 1.9.1's formats fix 1 or 2)
# llguidance 1.9.1's formats: their strings' fewest characters, and their most where they are bounded (each character
# of them one byte: ASCII, never escaped)
SHORTEST = {"date": 10, "time": 9, "date-time": 20, "duration": 3, "email": 3, "hostname": 1, "ipv4": 7, "ipv6": 2,
            "uuid": 36, "uri": 2}
FIXED_FORMATS = {"date": 10, "uuid": 36, "ipv4": 15, "ipv6": 39}
NUMBER_RUN = 8  # the most a fraction's bounds or `multipleOf` fix of its digits (measured: 5), but where they fix one
_I64 = (-(1 << 63), (1 << 63) - 1)  # llguidance's integers
MAX_GROUPS = 128  # a pattern's groups nested at most (llguidance's own parser refuses about 130)


def _sat(n: int) -> int:
    return n if n < SAT else SAT


class Run:
    """The fixed text of a piece of output: at its `start` and its `end`, the longest run inside it (`inner`, at JSON
    pointer `at`), and its `whole` length when all of it is fixed (else None). A piece that can produce nothing is
    None, not a Run; `maybe`: one that perhaps produces nothing (a length beside a pattern), whose parents count both
    cases. `exact`: a whole length that is the text's own, not a bound of it (a bound cannot tell two alternatives
    apart)."""

    __slots__ = ("at", "end", "exact", "inner", "maybe", "start", "whole")

    def __init__(self, start: int, end: int, inner: int, at: str, whole: int | None = None, maybe: bool = False,
                 exact: bool = True):
        self.start, self.end, self.whole, self.at, self.maybe, self.exact = start, end, whole, at, maybe, exact
        self.inner = max(inner, start, end)

    @staticmethod
    def text(n: int, at: str, exact: bool = True) -> Run:
        return Run(n, n, n, at, n, False, exact)

    def sure(self, maybe: bool = False) -> Run:
        """The same run, with `maybe` set."""
        if self.maybe == maybe:
            return self
        return Run(self.start, self.end, self.inner, self.at, self.whole, maybe, self.exact)

    def key(self) -> tuple:
        return self.start, self.end, self.inner, self.whole

    def __repr__(self) -> str:
        return f"Run({self.start}, {self.end}, {self.inner}, {self.at!r}, {self.whole})"


def free(at: str, end: int = 0) -> Run:
    """A piece that fixes nothing at its start; `end`: what it can fix at its end, after a choice (`true`'s `rue`)."""
    return Run(0, end, end, at)


def seq(parts: Iterable[Run | None], at: str) -> Run | None:
    """The pieces one after the other (None if one produces nothing)."""
    acc, maybe = Run.text(0, at), False
    for p in parts:
        if p is None:
            return None
        maybe |= p.maybe
        if acc.whole is not None and p.whole is not None:
            acc = Run.text(_sat(acc.whole + p.whole), at, acc.exact and p.exact)
            continue
        start = _sat(acc.whole + p.start) if acc.whole is not None else acc.start
        end = _sat(acc.end + p.whole) if p.whole is not None else p.end
        mid = _sat(acc.end + p.start)
        inner, where = max((acc.inner, acc.at), (p.inner, p.at), (mid, at), (start, at), (end, at),
                           key=lambda r: r[0])
        acc = Run(start, end, inner, where)
    return acc.sure(maybe)


def choice(alts: Iterable[Run | None], at: str) -> Run | None:
    """One of the alternatives (JSON values, which are prefix-free): the run that enters ends within the shortest
    alternative's start, and each alternative's text runs on into what follows. Alternatives that produce nothing drop
    out; alternatives fixed throughout and of one length count as one text."""
    alts = [a for a in alts if a is not None]
    sure = [a for a in alts if not a.maybe]
    if sure and len(sure) < len(alts):  # those that perhaps produce nothing: with them, and without
        return either(_choice([a.sure() for a in alts], at), _choice(sure, at), at).sure()
    got = _choice([a.sure() for a in alts], at)
    return got.sure(bool(alts)) if got is not None and not sure else got


def _choice(alts: list[Run], at: str) -> Run | None:
    if not alts:
        return None
    if len(alts) == 1:
        return alts[0]
    wholes = [a for a in alts if a.whole is not None]
    if len(wholes) == len(alts) and (len({a.whole for a in alts}) == 1 or not all(a.exact for a in alts)):
        return Run.text(max(a.inner for a in alts), at, False)  # all fixed, and perhaps one text
    best = max(alts, key=lambda a: a.inner)
    starts = [a.start for a in alts if a.whole is None or a.exact]  # a bound is no text to differ from
    return Run(min(starts), max(a.end for a in alts), best.inner, best.at)


def either(a: Run | None, b: Run | None, at: str) -> Run | None:
    """A bound of both readings of a piece (llguidance may infer a type or take any value; a piece that perhaps
    produces nothing, with it and without)."""
    if a is None or b is None:
        return a or b
    maybe = a.maybe or b.maybe
    if a.whole is not None or b.whole is not None:
        return Run(*(max(a.inner, b.inner),) * 3, at, max(a.inner, b.inner), maybe, False)
    best = a if a.inner >= b.inner else b
    return Run(max(a.start, b.start), max(a.end, b.end), best.inner, best.at, None, maybe)


def prefixed(texts: list[bytes]) -> bool:
    """Whether one text is a proper prefix of another: llguidance's lexer, which takes the longest match, may then never
    end the shorter one where the longer goes on, so the choice between them can be no choice."""
    texts = sorted(set(texts))
    return any(b.startswith(a) for a, b in itertools.pairwise(texts))


def lcp(texts: list[bytes]) -> int:
    if not texts:
        return 0
    lo, hi = min(texts), max(texts)
    n = 0
    for x, y in zip(lo, hi, strict=False):
        if x != y:
            break
        n += 1
    return n


def literals(texts: list[bytes], at: str) -> Run | None:
    """A choice of fixed texts (enum values), counted exactly: the common prefix, then each value's rest after the
    first byte that differs."""
    texts = sorted(set(texts))
    if not texts:
        return None
    if len(texts) == 1:
        return Run.text(len(texts[0]), at)
    common = lcp(texts)
    tail = max(len(t) for t in texts) - common - 1
    return Run(common, tail, tail, at)


# -- JSON text ------------------------------------------------------------------------------------------------------

def char_bytes(c: str) -> bytes:
    """A character inside a JSON string as llguidance writes it at its longest (a control character as `\\u00XX`)."""
    if c in '"\\':
        return b"\\" + c.encode()
    if c < " " or c == "\x7f":
        return f"\\u{ord(c):04x}".encode()
    return c.encode("utf-8", "surrogatepass")


def json_string(s: str) -> bytes:
    return b'"' + b"".join(map(char_bytes, s)) + b'"'


class Syntax:
    """How a grammar writes JSON: the key and item separators, and for a tool's Gemma grammar (`quote`: its `<|"|>`
    token as llguidance forces it) its own rules: keys in the template's order and unquoted, facets not enforced, no
    more keys than the named ones unless `additionalProperties` says so, `nullable` taken."""

    def __init__(self, key_sep: str = ":", item_sep: str = ",", quote: bytes | None = None, closed: bool = False):
        self.key_sep, self.item_sep, self.quote = key_sep.encode(), item_sep.encode(), quote
        self.json = quote is None
        self.closed = closed  # Qwen's `%json` items: an object with `properties` takes no other key

    def order(self, names: list[str]) -> list[str]:
        return names if self.json else sorted(names, key=lambda k: str(k).lower())

    def key(self, name: str) -> bytes:
        if self.json:
            return json_string(name) + self.key_sep
        return str(name).encode("utf-8", "surrogatepass") + b":"

    def value(self, v: Any) -> bytes:
        """A const or enum value's text (as Gemma's template writes it, for Gemma). llguidance writes a JSON number
        through its f64, a fraction as all its digits (1e300 as 301): here a float is its whole expansion and 2 bytes
        more, an integer past 2**53 its digits and 1 more, a bound of a spelling the count cannot know (`_inexact`)."""
        if isinstance(v, str):
            return json_string(v) if self.json else self.quote + v.encode("utf-8", "surrogatepass") + self.quote
        if v is None or isinstance(v, bool):
            return b"null" if v is None else b"true" if v else b"false"
        if isinstance(v, float) and self.json and math.isfinite(v):
            return _decimal(Fraction(repr(v))).encode() + b"00"
        if isinstance(v, int) and self.json and abs(v) >= 1 << 53:
            return str(v).encode() + b"0"
        if isinstance(v, int | float):
            return (repr(v) if isinstance(v, float) else str(v)).encode()
        if isinstance(v, list):
            return b"[" + self.item_sep.join(map(self.value, v)) + b"]"
        if isinstance(v, dict):
            return b"{" + self.item_sep.join(self.key(k) + self.value(v[k]) for k in self.order(list(v))) + b"}"
        raise BoundError(f"an enum or const value of type {type(v).__name__}")


JSON = Syntax()


def json_syntax(schema: dict, closed: bool = False) -> Syntax:
    """`schema`'s own `x-guidance` separators: llguidance writes them in compact and flexible JSON alike."""
    own = schema.get("x-guidance")
    if not isinstance(own, dict):
        return Syntax(closed=closed) if closed else JSON
    k, i = own.get("key_separator", ":"), own.get("item_separator", ",")
    return Syntax(k if isinstance(k, str) else ":", i if isinstance(i, str) else ",", closed=closed)


def members(open_: bytes, items: list[tuple[bytes, Run | None, bool]], sep: bytes, close: bytes, at: str,
            more: tuple[bytes, bytes, Run] | None = None) -> Run | None:
    """An object's text (or Qwen's parameter blocks'): `open_`, the named members in order (their key's text with its
    separator, their value's run, whether required) with `sep` before each but the first written, then `close`.
    `more`: the text that starts a member past the named ones after one is written and before any, and that member's
    run past its free key (None: none). At each gap the candidates are the members up to the next required one, and
    the end once none is left: one candidate is forced, several end the run at their texts' common prefix, and each
    starts a run of its own. Linear in the members (each gap's common prefix from suffix minima)."""
    n = len(items)
    if any(v is None for _, v, req in items if req):
        return None
    if any(v is not None and v.maybe and not req for _, v, req in items) or (more and more[2].maybe):
        full = members(open_, [(k, v and v.sure(), req) for k, v, req in items], sep, close, at,
                       more and (more[0], more[1], more[2].sure()))
        bare = members(open_, [(k, None if v is not None and v.maybe and not req else v, req) for k, v, req in items],
                       sep, close, at, None if more and more[2].maybe else more)
        return either(full, bare, at)
    if prefixed([k for k, v, _ in items if v is not None]):  # a key that starts another (Gemma's `a:` and `a:b:`)
        return _counted(items, at, sep, more, open_, close)
    maybe = any(v.maybe for _, v, req in items if req)
    nxt = [n] * (n + 1)  # the first required member at or after each index
    for i in range(n - 1, -1, -1):
        nxt[i] = i if items[i][2] else nxt[i + 1]
    # the common prefix of a gap's candidates: their least prefix shared with the gap's last text (the required
    # member's, or `close`), as suffix minima over each run of optional members (`first`: before anything is written)
    big = SAT
    low = [big] * (n + 1)
    low_first = [big] * (n + 1)
    for i in range(n - 1, -1, -1):
        low[i], low_first[i] = big, big
        if items[i][2]:
            continue
        r = nxt[i]
        ref = sep + items[r][0] if r < n else close
        ref_first = items[r][0] if r < n else close
        if items[i][1] is not None:
            low[i] = lcp([sep + items[i][0], ref])
            low_first[i] = lcp([items[i][0], ref_first])
        low[i], low_first[i] = min(low[i], low[i + 1]), min(low_first[i], low_first[i + 1])
    others = 0 if more is None else lcp([more[0], close])
    others_first = 0 if more is None else lcp([more[1], close])
    arrive: list[int | None] = [None] * n  # the longest run that reaches each member's end
    best, where, end_run = len(open_), at, 0
    on: int | None = -1  # the gap the run from the opening has reached (None once it ended)
    lead, start = len(open_), 0  # that run's length so far; where it ended
    live_opt = [0] * (n + 1)  # optional members that produce something, before each index
    for i in range(n):
        live_opt[i + 1] = live_opt[i] + (not items[i][2] and items[i][1] is not None)
    # each member's run once a choice took it (the same whichever gap: a member past an optional one is a choice's),
    # and the runs inside its value
    for k, (key, v, req) in enumerate(items):
        if v is None:
            continue
        if not req or (k > 0 and not items[k - 1][2]):
            text = len(key) + (len(sep) if k > 0 else 0)
            got = _sat(text + v.whole) if v.whole is not None else v.end
            if v.whole is None and text + v.start > best:
                best, where = _sat(text + v.start), at
            arrive[k] = got
        if v.inner > best:
            best, where = v.inner, v.at
    for g in range(-1, n):  # the gap after member g (-1: after the opening)
        run = len(open_) if g < 0 else arrive[g]
        if run is None:
            continue
        r = nxt[g + 1]
        ends = r >= n
        options = live_opt[min(r, n)] - live_opt[g + 1] + (1 if not ends else 1 + (more is not None))
        forced = options == 1
        here = on == g
        if forced and not ends:  # the next required member, its key fixed
            text = (sep if g >= 0 else b"") + items[r][0]
            v = items[r][1]
            got = _sat(run + len(text) + v.whole) if v.whole is not None else v.end
            if v.whole is None and run + len(text) + v.start > best:
                best, where = _sat(run + len(text) + v.start), at
            arrive[r] = got if arrive[r] is None else max(arrive[r], got)
            if here:
                if v.whole is not None:
                    lead, on = _sat(lead + len(text) + v.whole), r
                else:
                    start, on = _sat(lead + len(text) + v.start), None
            continue
        if forced:  # the close alone
            end_run = max(end_run, _sat(run + len(close)))
            if here:
                exact = all(v.exact for _, v, _ in items if v is not None)
                return Run.text(_sat(lead + len(close)), at, exact).sure(maybe)
            continue
        common = min(low_first[g + 1] if g < 0 else low[g + 1],
                     (others_first if g < 0 else others) if ends and more else big)
        common = 0 if common == big else common
        if run + common > best:
            best, where = _sat(run + common), at
        if here:
            start, on = _sat(lead + common), None
        if ends:
            end_run = max(end_run, len(close))
    if more:
        v = more[2]
        r = _sat(len(more[0]) + (v.whole if v.whole is not None else v.start))
        if r > best:
            best, where = r, at
        if v.inner > best:
            best, where = v.inner, v.at
    if end_run > best:
        best, where = end_run, at
    if start > best:
        best, where = start, at
    return Run(start, end_run, best, where, None, maybe)


def _array(items: list[Run | None], tail: Run | None, lo: int, hi: int | None, sep: int, sep_text: bytes, at: str,
           cut: bool) -> Run | None:
    """An array's text: its first `lo` items fixed (the tuple's, then `tail` copies), then a choice of the bracket or
    another item, the bracket forced after the last item `maxItems` or the tuple allows (after any, under `cut`)."""
    parts: list[Run | None] = [Run.text(1, at)]
    n = 0
    for item in items[:lo]:
        parts += [Run.text(sep, at), item] if n else [item]
        n += 1
    if lo > n:
        if tail is None:
            return None
        parts += ([Run.text(sep, at)] if n else []) + [repeat(tail, lo - n, sep, at)]
        n = lo
    nexts, stopped = [], False
    for x in items[n:]:
        if x is None:  # no array goes past an item that produces nothing
            stopped = True
            break
        nexts.append(x)
    if not stopped and tail is not None:
        nexts.append(tail)
    if (hi is not None and n >= hi) or not nexts:
        return seq([*parts, Run.text(1, at)], at)
    last = cut or hi is not None or tail is None or stopped
    gap = lcp([b"]", sep_text])
    runs = [seq([Run.text(sep, at), x], at) for x in nexts] + ([] if n else nexts)
    runs = [r for r in runs if r is not None]
    ends = [r.whole if r.whole is not None else r.end for r in runs]
    top = max(runs, key=lambda r: r.inner)
    inner = max(top.inner, max(ends) + max(gap, 1 if last else 0))
    end = max(ends) + 1 if last else 1 + gap
    return seq([*parts, Run(gap if n else 0, end, inner, top.at if top.inner >= inner else at)], at)


def repeat(item: Run, n: int, sep: int, at: str) -> Run:
    """`n` (at least 1) copies of `item`, `sep` bytes between them, in closed form."""
    if n == 1:
        return item
    if item.whole is not None:
        return Run.text(_sat(n * item.whole + (n - 1) * sep), at, item.exact).sure(item.maybe)
    mid = _sat(item.end + sep + item.start)
    inner, where = (item.inner, item.at) if item.inner >= mid else (mid, at)
    return Run(item.start, item.end, inner, where, None, item.maybe)


# -- schemas ----------------------------------------------------------------------------------------------------------

class Runs:
    """The runs of a document's schemas under `syntax`: `of(schema, at)` for one the document holds. Its definitions
    (those `$ref`s reach from where the count starts, `definitions`) are computed first, each once, in the order of the
    `$ref` graph's strongly connected components, a recursive group unfolded (module docstring). `resolve`: the
    definition a `$ref` names, given it and where it is (None when it does not resolve: llguidance's own 400)."""

    def __init__(self, root: dict, syntax: Syntax = JSON, resolve: Callable[[Any, str], Any] | None = None):
        self.root, self.syntax = root, syntax
        self.resolve = resolve or (lambda ref, at: definition(root, ref, at))
        self.memo: dict[int, Run | None] = {}

    def check(self, s: Any, at: str = "#", what: str = "the JSON schema") -> Run:
        """`s`'s run; a `BoundError` past the cap, or when `s` or a definition it reaches can produce no value."""
        self.definitions(s, at, what)
        got = self.of(s, at)
        if got is None:
            raise BoundError(f"{what} can produce no value")
        cap(got, what)
        return got

    def definitions(self, s: Any, at: str, what: str | None = "the JSON schema") -> None:
        """The runs of the definitions `s` reaches; `what`: the name a `BoundError` gives a definition that can produce
        no value (None: no error, as for a tool's, which its grammar may write as raw text)."""
        edges: dict[int, list[int]] = {}
        nodes: dict[int, tuple[Any, str]] = {}
        todo = [(s, at)]
        while todo:
            d, where = todo.pop()
            if id(d) in edges:
                continue
            nodes[id(d)] = (d, where)
            out = edges[id(d)] = []
            for ref, ref_at in _refs(d, where):
                t = self.resolve(ref, ref_at)
                if isinstance(t, dict):
                    out.append(id(t))
                    if id(t) not in edges:
                        todo.append((t, self._pointer(ref, t)))
        for comp in components(id(s), lambda v: edges.get(v, ())):
            group = [nodes[i] for i in comp]
            if len(comp) == 1 and comp[0] not in edges[comp[0]]:
                d, where = group[0]
                self.memo[id(d)] = self.of(d, where, top=True)
            else:
                self._unfold(group)
            for d, where in group:
                if what is not None and self.memo[id(d)] is None and d is not s:
                    raise BoundError(f"{what} can produce no value at {where}")

    def _pointer(self, ref: Any, t: Any) -> str:
        return "#" if t is self.root or not isinstance(ref, str) else ref

    def _unfold(self, group: list[tuple[Any, str]]) -> None:
        """A recursive group: Kleene's iteration from nothing, each round's members from the round before, for
        `MAX_NESTING` + 1 rounds. Its rounds can repeat with a period of 1 (it settled) or 2 (its members take turns),
        or grow by the same each period: then the larger state of the period, carried to the last round, bounds every
        round after."""
        for d, _ in group:
            self.memo[id(d)] = None
        rounds = MAX_NESTING + 1
        seen: list[list[tuple | None]] = []
        states: list[list[Run | None]] = []
        for k in range(1, rounds + 1):
            got = [self.of(d, where, top=True) for d, where in group]
            for (d, _), g in zip(group, got, strict=True):
                self.memo[id(d)] = g
            seen.append([None if g is None else g.key() for g in got])
            states.append(got)
            for period in (1, 2):
                if len(seen) < 3 * period:
                    continue
                steps = []  # each phase of the period: its states two periods apart, growing by the same
                for j in range(period):
                    step = _delta(seen[-1 - j - period], seen[-1 - j])
                    if step is None or step != _delta(seen[-1 - j - 2 * period], seen[-1 - j - period]):
                        break
                    steps.append(step)
                else:
                    left = -(-(rounds - k) // period)  # the periods still to come
                    for i, (d, where) in enumerate(group):
                        last = [_grow(states[-1 - j][i], steps[j][i], left, where) for j in range(period)]
                        self.memo[id(d)] = last[0] if period == 1 else either(*last, where)
                    return

    def of(self, s: Any, at: str, top: bool = False) -> Run | None:
        """The run of schema `s`, at JSON pointer `at`; `top`: `s` is the definition being computed."""
        if not top and id(s) in self.memo:
            return self.memo[id(s)]
        if s is False:
            return None
        if not isinstance(s, dict) or not s:
            return free(at, 4)
        while isinstance(s.get("allOf"), list) and s["allOf"]:  # a one-member allOf (the subset's rule)
            s = s["allOf"][0]
            if not isinstance(s, dict):
                return self.of(s, at)
        if "$ref" in s:
            t = self.resolve(s["$ref"], at)
            if isinstance(t, dict) and id(t) in self.memo:
                return self.memo[id(t)]
            return self.of(t, self._pointer(s["$ref"], t)) if isinstance(t, dict | bool) else free(at, 4)
        for k in ("anyOf", "oneOf"):
            if isinstance(s.get(k), list):
                return choice([self.of(x, f"{at}/{k}/{i}") for i, x in enumerate(s[k])], at)
        if "const" in s:
            return Run.text(len(self.syntax.value(s["const"])), at, not _inexact(s["const"]))
        if isinstance(s.get("enum"), list):
            return self._enum(s["enum"], at)
        t = s.get("type")
        types = [t] if isinstance(t, str) else [x for x in t if isinstance(x, str)] if isinstance(t, list) else []
        if not self.syntax.json and s.get("nullable") is True and types and "null" not in types:
            types.append("null")  # the tools' grammars take `nullable`; llguidance's JSON ignores it
        if types:
            return choice([self._typed(x, s, at) for x in dict.fromkeys(types)], at)
        if any(k in s for k in ("properties", "required", "additionalProperties")):
            guess = "object"
        elif any(k in s for k in ("items", "prefixItems")):
            guess = "array"
        elif self.syntax.json and any(k in s for k in ("pattern", "format", "minLength", "maxLength")):
            guess = "string"
        else:
            return free(at, 4)
        return either(self._typed(guess, s, at), free(at, 4), at) if self.syntax.json else self._typed(guess, s, at)

    def _enum(self, values: list, at: str) -> Run | None:
        texts = [self.syntax.value(v) for v in values]
        if any(_inexact(v) for v in values):  # spelled perhaps as llguidance does not: the shortest bounds it
            texts = sorted(set(texts))
            return choice([Run.text(len(x), at, False) for x in texts], at)
        return literals(texts, at)

    def _typed(self, t: str, s: dict, at: str) -> Run | None:
        syn = self.syntax
        if t == "null":
            return Run.text(4, at)
        if t == "boolean":
            return literals([b"true", b"false"], at)
        if t in ("integer", "number"):
            if not syn.json:
                return free(at)
            try:
                return _number(s, t == "integer", at)
            except (TypeError, ValueError, ArithmeticError):
                return free(at, 24)
        if t == "string":
            if not syn.json:
                q = len(syn.quote)
                return Run(q, q, q, at)
            return self._string(s, at)
        if t == "object":
            return self._object(s, at)
        if t == "array":
            return self._array(s, at)
        return free(at, 4)

    def _object(self, s: dict, at: str) -> Run | None:
        syn = self.syntax
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        req = [r for r in s["required"] if isinstance(r, str)] if isinstance(s.get("required"), list) else []
        if not syn.json and "properties" not in s and "additionalProperties" not in s:
            return Run(1, 1, 1, at)  # Gemma's dict: `{}`, or `{` and any keys
        addl = s.get("additionalProperties", syn.json and not (syn.closed and "properties" in s))
        required = set(req)
        items = [(syn.key(k), self.of(props[k], f"{at}/properties/{_key(k)}"), k in required)
                 for k in syn.order(list(props))]
        extra = addl if isinstance(addl, dict) else True
        if syn.json:  # llguidance writes a required name `properties` lacks after them
            items += [(syn.key(k), self.of(extra, f"{at}/additionalProperties"), True)
                      for k in dict.fromkeys(req) if k not in props]
        more = None
        value = None if addl is False else seq([Run.text(len(syn.key_sep) if syn.json else 0, at),
                                                 self.of(extra, f"{at}/additionalProperties")], at)
        if value is not None:  # more keys, unless their value can produce nothing
            more = (syn.item_sep + (b'"' if syn.json else b""), b'"' if syn.json else b"", value)
        if syn.json and ("minProperties" in s or "maxProperties" in s):  # a count can force any choice
            lo = s["minProperties"] if _count(s.get("minProperties")) else 0
            hi = s["maxProperties"] if _count(s.get("maxProperties")) else None
            live = [(k, v, req) for k, v, req in items if v is not None or req]
            need = sum(req for _, _, req in live)
            if (more is None and lo > len(live)) or (hi is not None and hi < need):
                return None
            if more is None and lo >= len(live):  # every member written
                return members(b"{", [(k, v, True) for k, v, _ in live], syn.item_sep, b"}", at)
            if hi is not None and hi == need:  # the required ones alone
                return members(b"{", [x for x in live if x[2]], syn.item_sep, b"}", at)
            return _counted(items, at, syn.item_sep, more, takes=1)
        return members(b"{", items, syn.item_sep, b"}", at, more)

    def _array(self, s: dict, at: str) -> Run | None:
        syn = self.syntax
        prefix = s["prefixItems"] if isinstance(s.get("prefixItems"), list) else \
            s["items"] if isinstance(s.get("items"), list) else []
        rest = s.get("additionalItems", True) if isinstance(s.get("items"), list) else s.get("items", True)
        lo = s["minItems"] if syn.json and _count(s.get("minItems")) else 0
        hi = s["maxItems"] if syn.json and _count(s.get("maxItems")) else None
        items = [self.of(x, f"{at}/prefixItems/{i}") for i, x in enumerate(prefix)]
        tail = self.of(rest, f"{at}/items")
        if not any(x is not None and x.maybe for x in [*items, tail]):
            return _array(items, tail, lo, hi, len(syn.item_sep), syn.item_sep, at, False)
        # items that perhaps produce nothing: with them, and without (and the bracket then forced after any item)
        full = _array([x and x.sure() for x in items], tail and tail.sure(), lo, hi, len(syn.item_sep), syn.item_sep,
                      at, True)
        bare = _array([None if x is not None and x.maybe else x for x in items], None if tail and tail.maybe else tail,
                      lo, hi, len(syn.item_sep), syn.item_sep, at, False)
        if bare is None:  # without them, no array: it perhaps produces nothing
            return full and full.sure(True)
        return either(full, bare, at)

    def _string(self, s: dict, at: str) -> Run | None:
        lo = s["minLength"] if _count(s.get("minLength")) else 0
        hi = s["maxLength"] if _count(s.get("maxLength")) else None
        if hi is not None and lo > hi:
            return None
        pattern, fmt = s.get("pattern"), s.get("format")
        if isinstance(pattern, str) and isinstance(fmt, str) and fmt in SHORTEST:
            return _Regex(pattern, at).beside(fmt, lo, hi)
        if isinstance(pattern, str):
            return _Regex(pattern, at).string(lo, hi)
        if fmt is not None:  # a length leaves a format's few fixed bytes as they are; a tight one, perhaps no string
            fixed = FIXED_FORMATS.get(fmt) if isinstance(fmt, str) else None
            if hi is not None and hi + 2 <= FORMAT_RUN:  # so short it may leave one string, fixed (ipv6's `::`)
                return Run.text(hi + 2, at, False).sure(True)
            empty = (hi is not None and hi < 64) or (fixed is not None and lo > fixed)
            return Run(FORMAT_RUN, FORMAT_RUN, FORMAT_RUN, at, None, empty)
        if hi == 0:
            return Run.text(2, at)
        return Run(1, 1, 2, at)  # a free string: `\u` takes only a control character, so `00` follows it


def _counted(items: list[tuple[bytes, Run | None, bool]], at: str, sep: bytes, more: tuple[bytes, bytes, Run] | None,
             open_: bytes = b"{", close: bytes = b"}", takes: int | None = None) -> Run | None:
    """An object whose choices may be forced (under `minProperties` or `maxProperties`, or keys that start one another):
    its runs over the sets of members it may write, in order. A run goes on through the members written in a row while
    their values are fixed throughout; a value that is not ends it (at its start) and starts the next (at its end);
    `more` (as `members` takes it) comes after the named members, its free key ending a run. `takes`: the most optional
    members one run can hold (llguidance's count forces one at most: its options each make one required, or leave it
    alone; None: any). Only maxima are kept, so a recursive member adds its own runs once a level, as an object's do.
    Fixed throughout only when it can write one set of members (no optional one, no `more`), its values fixed."""
    if any(v is None for _, v, req in items if req):
        return None
    if any(v is not None and v.maybe and not req for _, v, req in items):  # with them, and without (`members`)
        full = _counted([(k, v and v.sure(), req) for k, v, req in items], at, sep, more, open_, close, takes)
        bare = _counted([(k, None if v is not None and v.maybe and not req else v, req) for k, v, req in items], at,
                        sep, more, open_, close, takes)
        return either(full, bare, at)
    k = 1 if takes is None else takes + 1  # runs by the optional members they hold (the last: any more, unbounded)
    none = -1
    cur, lead = [len(open_)] + [none] * (k - 1), [len(open_)] + [none] * (k - 1)
    best, first, optional = len(open_), 0, False

    def add(xs: list[int], n: int) -> list[int]:
        return [_sat(x + n) if x >= 0 else none for x in xs]

    for key, v, req in items:
        if v is None:
            continue
        skip = not req or v.maybe
        optional |= skip
        text = len(key) + len(sep)
        if v.whole is not None:
            took = add(cur, text + v.whole)
            took_lead = add(lead, text + v.whole)
            if not skip:
                cur, lead = took, took_lead
            elif takes is None:  # written: never shorter than left out
                cur, lead = [max(a, b) for a, b in zip(cur, took, strict=True)], took_lead
            else:  # one optional member more in the run
                cur = [cur[0]] + [max(cur[j], took[j - 1]) for j in range(1, k)]
                lead = [lead[0]] + [max(lead[j], took_lead[j - 1]) for j in range(1, k)]
        else:
            best = max(best, _sat(max(cur) + text + v.start), v.inner)
            if max(lead) >= 0:
                first = max(first, _sat(max(lead) + text + v.start))
            if skip:
                cur[0] = max(cur[0], v.end)
            else:
                cur, lead = [v.end] + [none] * (k - 1), [none] * k
        best = max(best, max(cur))
    if more is not None:
        v = more[2]
        best = max(best, _sat(max(cur) + len(more[0])), _sat(len(more[0]) + (v.whole if v.whole is not None else
                                                                                v.start)), v.inner)
        if max(lead) >= 0:
            first = max(first, _sat(max(lead) + len(more[1])))
        after = _sat(1 + (v.whole if v.whole is not None else v.end))  # the free key's closing quote, then its value
        best = max(best, _sat(after + len(more[0])))
        cur[0] = max(cur[0], after)
    end = _sat(max(cur) + len(close))
    if max(lead) >= 0:
        first = max(first, _sat(max(lead) + len(close)))
        if not optional and more is None:
            exact = all(v.exact for _, v, _ in items if v is not None)
            return Run.text(first, at, exact).sure(any(v.maybe for _, v, _ in items if v is not None))
    return Run(first, end, max(best, end, first), at)


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _inexact(v: Any) -> bool:
    """Whether a value holds a number llguidance may spell otherwise (a float, an integer past 2**53)."""
    if isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1 << 53):
        return True
    if isinstance(v, list):
        return any(map(_inexact, v))
    if isinstance(v, dict):
        return any(map(_inexact, v.values()))
    return False


def _key(name: str) -> str:
    return str(name).replace("~", "~0").replace("/", "~1")


def _decimal(x: Fraction) -> str:
    """A number's digits as llguidance writes a bound it fixes (1e100 as 101 digits)."""
    if x.denominator == 1:
        return str(x.numerator)
    s = format(Decimal(x.numerator) / Decimal(x.denominator), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _number(s: dict, integer: bool, at: str) -> Run | None:
    """A number's fixed digits, its bounds read as llguidance reads them (doubles; the larger lower bound and the
    smaller upper one, an exclusive one on a tie): all of them when they and `multipleOf` leave one value; else an
    integer's least and greatest values' common prefix (`_int_range`), a fraction's longest bound and 2. Where
    llguidance's own reading may leave one value the count's does not, fixed throughout at the longest: a float bound
    its parser may read a double off (`_loose`), a few multiples between fractional bounds
    (its digit ranges leave some out), bounds past 10**15 less than 2 apart. None when the bounds leave no value."""
    def num(k: str) -> Fraction | None:  # a bound as llguidance reads it: a double (10**17 + 5 is 10**17)
        v = s.get(k)
        if isinstance(v, bool) or not isinstance(v, int | float):
            return None
        try:
            v = float(v)
        except OverflowError:
            return None
        return Fraction(repr(v)) if math.isfinite(v) else None
    lo, hi, xlo, xhi, step = (num(k) for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                                                "multipleOf"))
    has_lo, has_hi = lo is not None or xlo is not None, hi is not None or xhi is not None
    loose = any(isinstance(v, float) and _loose(v)
                for v in (s.get(k) for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")))
    texts = [_decimal(x).encode() for x in (lo, hi, xlo, xhi) if x is not None]
    longest = max(map(len, texts), default=0) + 2  # llguidance's range can fix more than the bounds' common digits
    if integer:
        got = _int_range(s)
        if got is None:
            return None
        a, b = got
        has_lo, has_hi = a is not None, b is not None
        a, b = _I64[0] if a is None else a, _I64[1] if b is None else b
        widest = max(len(str(a)), len(str(b))) + 1
        if loose and has_lo and has_hi:
            return Run.text(widest, at, False).sure(True)
        if step is not None and step > 0:
            unit = step.numerator // math.gcd(step.numerator, step.denominator)  # the integers it divides
            m1, m2 = -((-a) // unit) * unit, (b // unit) * unit
            if m1 > m2:  # past 2**53 a double's rounding may be what leaves no multiple
                return Run.text(widest, at, False).sure(True) if max(abs(a), abs(b)) >= 1 << 53 else None
            a, b = m1, m2
        lo_t, hi_t = str(a).encode(), str(b).encode()
        if a > b:
            return None
        if a == b:
            return Run.text(len(lo_t), at)
        common = lcp([lo_t, hi_t])
        tail = max(len(lo_t), len(hi_t)) - common - 1 if has_lo and has_hi else 0  # -1000 is longer than -999
        if has_lo and not has_hi and a < 0:  # the negatives down to `a`: its digits after `-` (`-1`)
            tail = max(tail, len(lo_t) - 1)
        if step is not None:  # a multiple's own digits: `-0` at the start, its zeros at the end (`000` of 1000)
            zeros = len(str(unit)) - len(str(unit).rstrip("0"))
            common, tail = common + 1, tail + 1 + zeros
        return Run(common, tail, max(common, tail), at)
    low = max((x, x is xlo) for x in (lo, xlo) if x is not None) if has_lo else None
    high = min((x, -(x is xhi)) for x in (hi, xhi) if x is not None) if has_hi else None
    if low is not None and high is not None and (low[0] > high[0] or (low[0] == high[0] and (low[1] or high[1]))):
        return None
    if step is not None and step > 0 and low is not None and high is not None:
        k1, k2 = math.ceil(low[0] / step), math.floor(high[0] / step)
        k1 += low[1] and k1 * step <= low[0]
        k2 -= bool(high[1]) and k2 * step >= high[0]
        if k1 > k2:
            big = loose or max(abs(low[0]), abs(high[0])) >= 2 ** 53
            return Run.text(longest, at, False).sure(True) if big else None
        if k2 - k1 < 3:  # one of a few multiples: perhaps the one llguidance's digit ranges leave
            return Run.text(max(longest, *(len(_decimal(k * step)) + 2 for k in (k1, k2))), at, False)
    if low is not None and high is not None and low[0] == high[0]:
        sign = low[0] == 0 and math.copysign(1.0, s["minimum"]) < 0  # a minimum of -0.0 writes `-0`
        return Run.text(len(_decimal(low[0])) + sign + 2 * loose, at, not loose)
    if not texts:
        return free(at) if step is None else Run(NUMBER_RUN, NUMBER_RUN, NUMBER_RUN, at)
    if low is not None and high is not None and max(abs(low[0]), abs(high[0])) >= 10 ** 15 and \
            high[0] - low[0] < 2:  # digits past a double's: perhaps one value
        return Run.text(longest, at, False)
    if has_lo and has_hi:
        return Run(max(longest, NUMBER_RUN), max(longest, NUMBER_RUN), max(longest, NUMBER_RUN), at)
    return Run(NUMBER_RUN, NUMBER_RUN, NUMBER_RUN, at)  # one bound: llguidance's range fixes a few digits too


def _loose(v: float) -> bool:
    """Whether llguidance's JSON parser (serde_json, without exact float parsing) may read a float literal as a double
    next to Python's: 15 significant digits or more, or several digits past exact powers of 10 (beyond 10**22)."""
    if not math.isfinite(v) or v == 0:
        return False
    digits = len(repr(abs(v)).split("e")[0].replace(".", "").strip("0"))
    return digits >= 15 or (digits > 1 and not 1e-22 <= abs(v) <= 1e22)


def _int_range(s: dict) -> tuple[int | None, int | None] | None:
    """An integer's least and greatest values as llguidance 1.9.1 reads its bounds: each a double (so 10**17 + 5 is
    10**17), the larger lower bound and the smaller upper one, an exclusive one moved by 1.0 in doubles (no move past
    2**53) or rounded inward, then cast to 64 bits, saturating (None: no bound). None when its check of the doubles
    finds no value: a lower bound above the upper one, or equal to it with either exclusive."""
    def double(k: str) -> float | None:
        v = s.get(k)
        if v is True and k.startswith("exclusive"):  # draft-04's: the plain bound, exclusive
            v = s.get("minimum" if k == "exclusiveMinimum" else "maximum")
        if isinstance(v, bool) or not isinstance(v, int | float):
            return None
        try:
            return float(v)
        except OverflowError:
            return math.copysign(math.inf, v)

    def pick(plain: float | None, excl: float | None, lower: bool) -> tuple[float, bool] | None:
        if plain is not None and excl is not None:
            return (excl, True) if (excl >= plain if lower else excl <= plain) else (plain, False)
        if plain is None and excl is None:
            return None
        return (plain, False) if plain is not None else (excl, True)

    def bound(got: tuple[float, bool] | None, lower: bool) -> int | None:
        if got is None or math.isnan(got[0]):
            return None
        f, excl = got
        if math.isfinite(f) and not f.is_integer():
            f = float(math.ceil(f) if lower else math.floor(f))
        elif excl:
            f = f + 1.0 if lower else f - 1.0
        if f >= 2.0 ** 63:
            return _I64[1]
        return _I64[0] if f <= -2.0 ** 63 else int(f)
    lo = pick(double("minimum"), double("exclusiveMinimum"), True)
    hi = pick(double("maximum"), double("exclusiveMaximum"), False)
    if lo is not None and hi is not None and (lo[0] > hi[0] or (lo[0] == hi[0] and (lo[1] or hi[1]))):
        return None
    return bound(lo, True), bound(hi, False)


def _refs(s: Any, at: str) -> list[tuple[Any, str]]:
    """The `$ref`s in schema `s`'s positions (not in values, nor in definitions it holds)."""
    found: list[tuple[Any, str]] = []
    todo = [(s, at)]
    while todo:
        x, where = todo.pop()
        if not isinstance(x, dict):
            continue
        if "$ref" in x:
            found.append((x["$ref"], where))
        if isinstance(x.get("properties"), dict):
            todo += [(v, f"{where}/properties/{_key(n)}") for n, v in x["properties"].items()]
        for k in ("items", "additionalProperties", "additionalItems"):
            if isinstance(x.get(k), dict):
                todo.append((x[k], f"{where}/{k}"))
        for k in ("anyOf", "oneOf", "prefixItems", "items", "allOf"):
            if isinstance(x.get(k), list):
                todo += [(v, f"{where}/{k}/{i}") for i, v in enumerate(x[k])]
    return found


def _delta(a: list[tuple | None], b: list[tuple | None]) -> list[tuple] | None:
    """How much each member's run grew in a round (None unless each keeps its kind)."""
    out = []
    for x, y in zip(a, b, strict=True):
        if x is None or y is None or (x[3] is None) != (y[3] is None):
            return None
        out.append(tuple((q or 0) - (p or 0) for p, q in zip(x, y, strict=True)))
    return out


def _grow(r: Run, step: tuple, rounds: int, at: str) -> Run:
    s, e, i, w = (None if v is None else _sat(max(v, v + d * rounds)) for v, d in zip(r.key(), step, strict=True))
    return Run.text(w, at, False) if w is not None else Run(s, e, i, r.at)


def special(token: int) -> int:
    """A special token's length in llguidance's forced bytes (`\\xff[ID]`)."""
    return 3 + len(str(token))


def gemma_call(runs: Runs | None, root: dict | None, name: str, ids: dict[str, int], last: bool) -> Run | None:
    """A Gemma call's run: `<|tool_call>call:NAME{...}<tool_call|>`, then the end token when no call can follow."""
    body = Run.text(2, "#") if root is None or runs is None else runs.of(root, "#")
    return seq([Run.text(special(ids["opener"]) + len(f"call:{name}".encode()), "#"), body,
                Run.text(special(ids["closer"]) + (special(ids["end"]) if last else 0), "#")], "#")


def qwen_call(params: list[tuple[bytes, Run | None, bool]], more: bool, name: str, ids: dict[str, int]) -> Run | None:
    """A Qwen call's run: `<tool_call>`, `\\n<function=NAME>\\n`, the parameter blocks (each its head and its value
    with the block's end), `</function>\\n` and `</tool_call>`."""
    extra = (b"<parameter=", b"<parameter=", Run(2, len(QWEN_TAIL), len(QWEN_TAIL), "#")) if more else None
    body = members(b"", params, b"", b"</function>\n", "#", extra)
    return seq([Run.text(special(ids["opener"]) + len(f"\n<function={name}>\n".encode()), "#"), body,
                Run.text(special(ids["closer"]), "#")], "#")


QWEN_TAIL = b"\n</parameter>\n"


def qwen_value(raw: bool, texts: list[str], body: Run | None, at: str) -> Run | None:
    """A Qwen parameter's value and its block's end: raw text, one of the literal texts, or JSON."""
    tail = len(QWEN_TAIL)
    if raw:  # any text: the others are some of it
        return Run(0, tail, tail, at)
    alts: list[Run | None] = []
    if texts:
        raws = [t.encode("utf-8", "surrogatepass") for t in texts]
        if prefixed(raws) or any(QWEN_TAIL in t for t in raws):  # raw texts: one may start another
            fixed = Run.text(max(map(len, raws)) + tail, at, False)
            return fixed if body is None else either(fixed, seq([body, Run.text(tail, at)], at), at)
        alts.append(literals([t + QWEN_TAIL for t in raws], at))
    if body is not None:
        alts.append(seq([body, Run.text(tail, at)], at))
    return choice(alts, at)


def cap(r: Run, what: str) -> None:
    if r.inner > MAX_FORCED_RUN:
        n = f"{r.inner:,}" if r.inner < SAT else f"more than {SAT:,}"
        raise BoundError(f"{what} fixes {n} bytes of output in a row at {r.at}, more than {MAX_FORCED_RUN:,}")


def check_json_schema(root: Any, what: str = "the JSON schema") -> None:
    """A `BoundError` when `root`, a `response_format` schema in the subset, fixes more than `MAX_FORCED_RUN` bytes in
    a row, or can produce no value, or its `x-guidance` whitespace fixes text (llguidance puts it between tokens)."""
    if not isinstance(root, dict):
        return
    own = root.get("x-guidance")
    pattern = own.get("whitespace_pattern") if isinstance(own, dict) else None
    try:
        if isinstance(pattern, str) and _Regex(f"^(?:{pattern})$", "#/x-guidance/whitespace_pattern", True).fixes():
            raise BoundError(f"{what}'s x-guidance whitespace_pattern fixes text between tokens: not supported")
        Runs(root, json_syntax(root)).check(root, "#", what)
    except RecursionError:
        raise BoundError(f"{what} is nested too deeply") from None


# -- patterns ---------------------------------------------------------------------------------------------------------

_DOT = ("cls", 0, frozenset(b"ab"), 6)
_FLAGS = re.compile(r"\(\?[isUu-]+\)")
_ANY = ("rep", _DOT, 0, None)


class _Regex:
    """A `pattern`'s fixed text in its JSON string. llguidance (derivre over regex-syntax) reads a pattern as a search,
    anchored only by a leading `^` and a trailing `$` of its alternatives; the syntax read here is a subset of
    regex-syntax's, and a pattern outside it is refused (one without `^`, a search throughout, fixes nothing).

    The tree: ("lit", bytes, 1), ("cls", lead, firsts, longest), ("unknown",) (a class this reader does not take
    apart), ("none",), ("cat", parts), ("alt", parts), ("rep", part, n, m or None). It is evaluated right to left
    (`_eval`), each piece given some of the first bytes of what follows it: a choice ends a run only when two of the
    next bytes differ."""

    def __init__(self, text: str, at: str, raw: bool = False):
        self.text, self.at, self.i, self.depth = text, at, 0, 0
        self.enc = (lambda c: c.encode("utf-8", "surrogatepass")) if raw else char_bytes  # raw: not in a string

    def string(self, lo: int, hi: int | None) -> Run | None:
        """The string's run: its quotes and content; a length (`lo`, `hi`) that leaves out some of the pattern's
        strings can leave one of its choices, up to that length."""
        at = self.at
        if "^" not in self.text and "\\A" not in self.text and hi is None:
            return Run(1, 1, 2, at)  # a search: anything before it, so it fixes nothing
        content = self._parse()
        quote = ("lit", b'"', 1)
        s = _eval(("cat", [quote, content, quote]), frozenset())
        if s.empty:
            return None
        got = Run.text(s.lc, at, False) if s.lp else Run(s.lc, s.bt if s.bq else 0, max(s.bb, s.bt), at)
        shortest, longest = _lengths(content)
        if (longest is not None and lo > longest) or (hi is not None and hi < shortest):
            return None
        if hi == 0:
            return Run.text(2, at)
        if lo > shortest or (hi is not None and (longest is None or hi < longest)):  # a length that leaves out some
            # of its strings can leave any choice between lengths one (a class stays a choice), or no string at all
            quote = Run.text(1, at)
            bound = seq([quote, _by_length(content, hi if hi is not None else lo, at), quote], at)
            if bound is None:
                return None
            return either(got, bound, at).sure(True)
        return got

    def beside(self, fmt: str, lo: int, hi: int | None) -> Run:
        """The string's run beside a `format`, which llguidance intersects with the pattern: the format may rule out
        any of the pattern's choices, so the whole string may be fixed. Its characters are the format's, a byte each,
        so it fixes at most as many as the pattern's longest string where a repetition without a maximum takes its
        minimum and as many copies as the format's and the length's shortest strings have characters (`_reach`), and
        `FORMAT_RUN` more; at most the format's longest, at most `hi`; and its quotes."""
        n = _sat(_reach(self._parse(), SHORTEST[fmt] + lo) + FORMAT_RUN)
        n = min([n, *([FIXED_FORMATS[fmt]] if fmt in FIXED_FORMATS else []), *([hi] if hi is not None else [])])
        return Run.text(n + 2, self.at, False).sure(True)

    def fixes(self) -> bool:
        """Whether the pattern, as the whitespace between JSON tokens, fixes text (`x-guidance`'s whitespace_pattern:
        the next token starts with one of `"{}[],:`, a digit, `-`, `t`, `f` or `n`): a byte at every gap's start,
        which joins the tokens' runs, or more than `FORMAT_RUN` bytes after a choice (`\\s*` fixes the rest of a
        character's bytes)."""
        s = _eval(self._parse(), frozenset(b'"{}[],:-0123456789tfn'))
        return not s.empty and (s.lc > 0 or max(s.bb, s.bt) > FORMAT_RUN)

    def _fail(self, why: str) -> BoundError:
        return BoundError(f"the pattern at {self.at} is not supported: {why}, at character {self.i}")

    def _parse(self) -> tuple:
        alts = [("cat", ([] if start else [_ANY]) + [body] + ([] if end else [_ANY]))
                for start, body, end in self._alternatives(False)]
        if self.i < len(self.text):
            raise self._fail("an unmatched ')'")
        return alts[0] if len(alts) == 1 else ("alt", alts)

    def _alternatives(self, closing: bool) -> list[tuple[bool, tuple, bool]]:
        """The pattern's alternatives (or a group's, `closing`), each with whether a `^` starts it and a `$` ends it.
        One that is a group alone, between its own anchors, stands for that group's alternatives with them (`(^a)|^(b$)`
        reads as `^a|^b$`)."""
        t, alts = self.text, []
        while True:
            while (m := _FLAGS.match(t, self.i)) is not None:  # `(?i)` before the anchor fixes nothing
                self.i = m.end()
            parts: list[tuple] = []
            start = self._anchor(("^", "\\A"))
            if t.startswith("(", self.i):
                inner = self._group()
                save = self.i
                end = self._anchor(("$", "\\z"))
                if self._ends(closing):
                    group = inner if isinstance(inner, list) else [(False, inner, False)]
                    alts += [(a or start, b, z or end) for a, b, z in group]
                    if not self._next(closing):
                        return alts
                    continue
                self.i = save
                parts.append(self._quantified(self._plain(inner)))
            body = self._seq(closing, parts)
            end = self._anchor(("$", "\\z"))
            alts.append((start, body, end))
            if not self._next(closing):
                return alts

    def _ends(self, closing: bool) -> bool:
        """Whether an alternative ends here."""
        t = self.text
        return self.i >= len(t) or t[self.i] == "|" or (closing and t[self.i] == ")")

    def _next(self, closing: bool) -> bool:
        """Past a `|` (True), or at the alternatives' end (False)."""
        t = self.text
        if self.i < len(t) and t[self.i] == "|":
            self.i += 1
            return True
        if self.i < len(t) and not (closing and t[self.i] == ")"):
            raise self._fail("an unmatched ')'")
        return False

    def _plain(self, inner: tuple | list) -> tuple:
        """A group read as one piece: an anchor inside one is refused."""
        if not isinstance(inner, list):
            return inner
        if any(a or z for a, _, z in inner):
            raise self._fail("an anchor inside a group that is not an alternative of its own")
        bodies = [b for _, b, _ in inner]
        return bodies[0] if len(bodies) == 1 else ("alt", bodies)

    def _anchor(self, marks: tuple[str, ...]) -> bool:
        found = False
        while True:
            m = next((m for m in marks if self.text.startswith(m, self.i)), None)
            if m is None:
                return found
            self.i += len(m)
            found = True

    def _seq(self, closing: bool = False, parts: list[tuple] | None = None) -> tuple:
        """A sequence, up to a `|` or a `)`; a `$` ends it only at its alternative's end."""
        parts, t = parts if parts is not None else [], self.text
        while self.i < len(t) and t[self.i] not in "|)":
            if t.startswith("$", self.i) or t.startswith("\\z", self.i):
                save = self.i
                self._anchor(("$", "\\z"))
                end = self._ends(closing)
                self.i = save
                if end:
                    break
                raise self._fail("'$' before the end")
            parts.append(self._quantified(self._atom()))
        return ("cat", parts)

    def _atom(self) -> tuple:
        c = self.text[self.i]
        if c == "(":
            return self._plain(self._group())
        if c == "[":
            return self._class()
        if c == ".":
            self.i += 1
            return _DOT
        if c == "\\":
            return self._escape()
        if c in "*+?{":
            raise self._fail(f"{c!r} with nothing to repeat")
        if c in "^$":
            raise self._fail(f"{c!r} inside the pattern")
        self.i += 1
        return ("lit", self.enc(c), 1)

    def _group(self) -> tuple | list[tuple[bool, tuple, bool]]:
        """A group: its alternatives (`_alternatives`), or a piece that fixes nothing for flags alone (`(?i)`)."""
        t = self.text
        self.i += 1
        self.depth += 1
        if self.depth > MAX_GROUPS:
            raise self._fail(f"groups nested more than {MAX_GROUPS} deep")
        if t.startswith("?", self.i):
            self.i += 1
            if t.startswith(("=", "!", "<=", "<!"), self.i):
                raise self._fail("a lookaround")
            if t.startswith(":", self.i):
                self.i += 1
            elif t.startswith("P<", self.i) or t.startswith("<", self.i):
                j = t.find(">", self.i)
                name = t[self.i:j].removeprefix("P").removeprefix("<") if j > 0 else ""
                if not name.replace("_", "a").isalnum() or name == "stop":
                    raise self._fail("a group name")
                self.i = j + 1
            else:  # flags: case-insensitive and the others fix less or the same; verbose and multi-line are refused
                j = self.i
                while j < len(t) and (t[j].isalpha() or t[j] == "-"):
                    j += 1
                flags = t[self.i:j]
                if not flags or j >= len(t) or t[j] not in ":)" or set(flags) - set("isUu-"):
                    raise self._fail(f"the flags {flags!r}")
                self.i = j + 1
                if t[j] == ")":
                    self.depth -= 1
                    return ("cat", [])
        alts = self._alternatives(True)
        if not t.startswith(")", self.i):
            raise self._fail("an unclosed group")
        self.i += 1
        self.depth -= 1
        return alts

    def _quantified(self, x: tuple) -> tuple:
        t = self.text
        while self.i < len(t) and t[self.i] in "*+?{":
            c = t[self.i]
            if c == "{":
                j = t.find("}", self.i)
                body = t[self.i + 1:j] if j > 0 else ""
                lo, comma, hi = body.partition(",")
                if j < 0 or not (lo.isascii() and lo.isdigit()) or (hi and not (hi.isascii() and hi.isdigit())):
                    raise self._fail("a counted repetition other than {n}, {n,} and {n,m}")
                n, m = int(lo), (int(hi) if hi else None) if comma else int(lo)
                if m is not None and m < n:
                    raise self._fail("a counted repetition's range")
                self.i = j + 1
            else:
                n, m = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
                self.i += 1
            if t.startswith("?", self.i):  # lazy: the same strings
                self.i += 1
            x = ("rep", x, n, m)
        return x

    def _code(self, c: str) -> int | None:
        """The code point of `\\xHH`, `\\x{...}`, `\\uHHHH`, `\\u{...}` or `\\UHHHHHHHH` (None for another escape)."""
        t, width = self.text, {"x": 2, "u": 4, "U": 8}.get(c)
        if width is None:
            return None
        if t.startswith("{", self.i):
            j = t.find("}", self.i)
            digits, self.i = (t[self.i + 1:j], j + 1) if j > 0 else ("", self.i)
        else:
            digits, self.i = t[self.i:self.i + width], self.i + width
        try:
            cp = int(digits, 16)
        except ValueError:
            raise self._fail(f"the escape \\{c}{digits}") from None
        if not 0 <= cp <= 0x10FFFF or 0xD800 <= cp <= 0xDFFF:
            raise self._fail(f"the code point {digits}")
        return cp

    def _escaped(self, in_class: bool) -> int | str:
        """The character an escape names, or the letter of a class escape (`d`, `w`, `s`, `p`)."""
        t = self.text
        if self.i >= len(t):
            raise self._fail("a trailing backslash")
        c = t[self.i]
        self.i += 1
        if c in "dDwWsS":
            return c.lower()
        if c in "pP":
            if t.startswith("{", self.i):
                j = t.find("}", self.i)
                if j < 0:
                    raise self._fail("an unclosed Unicode class")
                self.i = j + 1
            else:
                self.i += 1
            return "p"
        cp = self._code(c)
        if cp is not None:
            return cp
        simple = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "a": 7}
        if c in simple:
            return simple[c]
        if c.isascii() and not c.isalnum():
            return ord(c)
        raise self._fail(f"the escape \\{c}{' in a class' if in_class else ''}")

    def _escape(self) -> tuple:
        self.i += 1
        got = self._escaped(False)
        if isinstance(got, int):
            return ("lit", self.enc(chr(got)), 1)
        if got == "p":
            return ("cls", 5, frozenset(), 6)  # a Unicode class: many members, perhaps a first byte in common
        return ("cls", 0, frozenset(b"01" if got == "d" else b"ab"), 1 if got in "dw" else 6)

    def _class(self) -> tuple:
        """`[...]`, as ranges of code points; one member is a literal. Nested classes, set operations and POSIX names
        are read as a class that may hold a single member (`unknown`)."""
        t = self.text
        self.i += 1
        neg = t.startswith("^", self.i)
        self.i += neg
        ranges: list[tuple[int, int]] = []
        unknown, first = False, True
        while True:
            if self.i >= len(t):
                raise self._fail("an unclosed class")
            c = t[self.i]
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "[" or any(t.startswith(op, self.i) for op in ("&&", "--", "~~")):
                unknown = True
                self._skip_nested()
                continue
            if c == "\\":
                self.i += 1
                lo = self._escaped(True)
            else:
                lo = ord(c)
                self.i += 1
            if isinstance(lo, str):  # a class escape: many members
                ranges += [(0x30, 0x39)] if lo == "d" else [(0x61, 0x62), (0x100, 0x101)]
                continue
            if t.startswith("-", self.i) and not t.startswith("-]", self.i):
                self.i += 1
                if self.i >= len(t):
                    raise self._fail("an unclosed class")
                if t.startswith("\\", self.i):
                    self.i += 1
                    hi = self._escaped(True)
                else:
                    hi = ord(t[self.i])
                    self.i += 1
                if isinstance(hi, str) or hi < lo:
                    raise self._fail("a class range")
                ranges.append((lo, hi))
            else:
                ranges.append((lo, lo))
        if unknown:
            return ("unknown",)
        ranges = _merged(ranges)  # [aa] and [a-ca-c] are one member, and three
        if neg:
            ranges = _complement(ranges)
        count = sum(b - a + 1 for a, b in ranges)
        if count == 0:
            return ("none",)
        if count == 1:
            return ("lit", self.enc(chr(ranges[0][0])), 1)
        ends = [chr(cp) for a, b in ranges for cp in (a, b) if not 0xD800 <= cp <= 0xDFFF]
        encs = [self.enc(c) for c in ends]
        firsts = frozenset(e[0] for e in encs)
        escapes = self.enc is char_bytes and any(a < 0x20 or b >= 0x7F or (a <= 0x22 <= b) or (a <= 0x5C <= b)
                                                 for a, b in ranges)
        return ("cls", 0 if len(firsts) >= 2 else 5, firsts, 6 if escapes else max(map(len, encs)))

    def _skip_nested(self) -> None:
        t, depth = self.text, 0
        while self.i < len(t):
            if t[self.i] == "\\":
                self.i += 2
                continue
            if t[self.i] == "[":
                depth += 1
            elif t[self.i] == "]":
                if depth == 0:
                    return
                depth -= 1
            self.i += 1


class _Sum:
    """A pattern piece's runs as functions of the run `r` that follows it: from its start `lc + lp * r`, from any of its
    positions at most `max(bb, bt + bq * r)`. `first`: some of the bytes it can start with (what follows included when
    it can be empty); `own`: some of the first bytes of its own non-empty strings; `empty`: it matches nothing."""

    __slots__ = ("bb", "bq", "bt", "empty", "first", "lc", "lp", "nullable", "own")

    def __init__(self, lc: int, lp: int, bb: int, bt: int, bq: int, first: frozenset, own: frozenset,
                 nullable: bool, empty: bool = False):
        lc, bb, bt = _sat(lc), _sat(bb), _sat(bt)
        if lp:
            bt, bq = max(bt, lc), 1
        else:
            bb = max(bb, lc)
        if not bq:
            bb, bt = max(bb, bt), 0
        self.lc, self.lp, self.bb, self.bt, self.bq = lc, lp, bb, bt, bq
        self.first, self.own, self.nullable, self.empty = first, own, nullable, empty


_NONE = _Sum(0, 0, 0, 0, 0, frozenset(), frozenset(), False, empty=True)


def _empty(follow: frozenset) -> _Sum:
    return _Sum(0, 1, 0, 0, 1, follow, frozenset(), True)


def _merge(*terms: tuple[int, int]) -> tuple[int, int, int]:
    """`max` of constants and of runs `t + r`: (the largest constant, the largest `t`, whether any has `r`)."""
    bb = max([t for t, q in terms if not q], default=0)
    slopes = [t for t, q in terms if q]
    return (bb, max(slopes), 1) if slopes else (bb, 0, 0)


def _then(a: _Sum, b: _Sum) -> _Sum:
    """`a` followed by `b` (`a` evaluated with `b`'s first bytes as what follows it)."""
    if a.empty or b.empty:
        return _NONE
    lc, lp = a.lc + a.lp * b.lc, a.lp * b.lp
    bb, bt, bq = _merge((a.bb, 0), (a.bt + a.bq * b.lc, a.bq * b.lp), (b.bb, 0), (b.bt, b.bq))
    own = a.own | b.own if a.nullable else a.own
    return _Sum(lc, lp, bb, bt, bq, a.first, own, a.nullable and b.nullable)


def _alt(subs: list[_Sum]) -> _Sum:
    subs = [s for s in subs if not s.empty]
    if not subs:
        return _NONE
    first = frozenset().union(*(s.first for s in subs))
    fixed = [s.lc for s in subs if not s.lp]
    if len(first) >= 2:  # two different next bytes: nothing forced here
        lc, lp = 0, 0
    else:
        lc, lp = (min(fixed), 0) if fixed else (min(s.lc for s in subs), 1)
    terms = [(s.bb, 0) for s in subs] + [(s.bt, s.bq) for s in subs] + [(lc, lp)]
    bb, bt, bq = _merge(*terms)
    own = frozenset().union(*(s.own for s in subs))
    return _Sum(lc, lp, bb, bt, bq, first, own, any(s.nullable for s in subs))


def _shift(s: _Sum, c: int, p: int) -> tuple[int, int, int]:
    """`s`'s runs from any position when the run after it is `c + p * r`."""
    return _merge((s.bb, 0), (s.bt + s.bq * c, s.bq * p))


def _eval(x: tuple, follow: frozenset, memo: dict | None = None) -> _Sum:
    """Piece `x`'s runs, given some of the first bytes of what follows it; `memo`: each piece once per follow."""
    memo = {} if memo is None else memo
    key = (id(x), follow)
    got = memo.get(key)
    if got is None:
        got = memo[key] = _eval_one(x, follow, memo)
    return got


def _eval_one(x: tuple, follow: frozenset, memo: dict) -> _Sum:
    kind = x[0]
    if kind == "lit":
        n = len(x[1])
        return _Sum(n, 1, 0, n, 1, frozenset(x[1][:1]), frozenset(x[1][:1]), False)
    if kind == "cls":
        lead, firsts, longest = x[1], x[2], x[3]
        return _Sum(lead, 0, lead, longest - 1, 1, firsts, firsts, False)
    if kind == "unknown":  # perhaps one member: fixed, its first byte unknown
        return _Sum(6, 1, 0, 6, 1, frozenset(), frozenset(), False)
    if kind == "none":
        return _NONE
    if kind == "cat":
        acc = _empty(follow)
        for part in reversed(x[1]):
            acc = _then(_eval(part, acc.first, memo), acc)
        return acc
    if kind == "alt":
        return _alt([_eval(m, follow, memo) for m in x[1]])
    item, n, m = x[1], x[2], x[3]
    # the optional copies after the n required ones: each a choice between stopping and another copy
    if m == n:
        opt = _empty(follow)
    else:
        copy = _eval(item, follow, memo)  # what follows a copy is another copy or `follow`: `follow` bounds both
        if copy.empty:
            opt = _empty(follow)
        else:
            nexts = copy.own | follow
            if len(nexts) >= 2:
                lc, lp = 0, 0
            else:
                lc, lp = (copy.lc, 0) if not copy.lp else (0, 1)
            cx = lc if not lp else 0
            bb, bt, bq = _merge((0, 1), *_terms(_shift(copy, cx, 1)), (lc, lp))
            opt = _Sum(lc, lp, bb, bt, bq, nexts, copy.own, True)
    if n == 0:
        return opt
    last = _then(_eval(item, opt.first, memo), opt)
    if last.empty or n == 1:
        return last
    chain = _eval(item, _eval(item, frozenset(), memo).own, memo)  # a copy followed by another
    if chain.empty:
        return _NONE
    k = n - 1
    if chain.lp:  # each copy's start run goes on into the next
        lc, lp = _sat(chain.lc * min(k, SAT) + last.lc), last.lp
        c = _sat(chain.lc * min(k - 1, SAT) + last.lc)
        bb, bt, bq = _merge((last.bb, 0), (last.bt, last.bq), *_terms(_shift(chain, c, last.lp)))
    else:
        lc, lp = chain.lc, 0
        bb, bt, bq = _merge((last.bb, 0), (last.bt, last.bq), *_terms(_shift(chain, last.lc, last.lp)),
                            *_terms(_shift(chain, chain.lc, 0)))
    own = chain.own | last.own if chain.nullable else chain.own
    return _Sum(lc, lp, bb, bt, bq, chain.first, own, chain.nullable and last.nullable)


def _terms(m: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    bb, bt, bq = m
    return ((bb, 0), (bt, bq))


def _merged(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _complement(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, at = [], 0
    for a, b in sorted(ranges):
        if a > at:
            out.append((at, a - 1))
        at = max(at, b + 1)
    if at <= 0x10FFFF:
        out.append((at, 0x10FFFF))
    return out


def _reach(x: tuple, more: int) -> int:
    """A pattern piece's most characters, a repetition without a maximum taken at its minimum and `more` copies."""
    kind = x[0]
    if kind in ("lit", "cls", "unknown"):
        return 1
    if kind == "none":
        return 0
    if kind == "cat":
        return _sat(sum(_reach(p, more) for p in x[1]))
    if kind == "alt":
        return max(_reach(m, more) for m in x[1])
    return _sat(_reach(x[1], more) * min(x[3] if x[3] is not None else x[2] + more, SAT))


def _lengths(x: tuple) -> tuple[int, int | None]:
    """A pattern piece's shortest and longest match in characters (None: unbounded)."""
    kind = x[0]
    if kind in ("lit", "cls", "unknown"):
        return 1, 1
    if kind == "none":
        return 0, 0
    if kind == "cat":
        lo, hi = 0, 0
        for p in x[1]:
            a, b = _lengths(p)
            lo, hi = lo + a, None if hi is None or b is None else hi + b
        return lo, hi
    if kind == "alt":
        got = [_lengths(m) for m in x[1]]
        his = [b for _, b in got]
        return min(a for a, _ in got), None if None in his else max(his)
    a, b = _lengths(x[1])
    return a * x[2], None if x[3] is None or b is None else b * x[3]


def _by_length(x: tuple, budget: int, at: str) -> Run | None:
    """Piece `x`'s runs where a length (of `budget` characters at most) can leave any of its choices between lengths
    one: an alternation forced to one member, a repetition to any count up to the budget; a class stays a choice of
    one character among several."""
    kind = x[0]
    if kind == "lit":
        return Run.text(len(x[1]), at, False)
    if kind == "cls":
        return Run(x[1], x[3] - 1, max(x[1], x[3] - 1), at)
    if kind == "unknown":
        return Run.text(6, at, False)
    if kind == "none":
        return None
    if kind == "cat":
        return seq([_by_length(p, budget, at) for p in x[1]], at)
    if kind == "alt":
        subs = [r for r in (_by_length(m, budget, at) for m in x[1]) if r is not None]
        if not subs:
            return None
        if any(r.whole is not None for r in subs):
            return Run.text(max(r.inner for r in subs), at, False)
        return Run(max(r.start for r in subs), max(r.end for r in subs), max(r.inner for r in subs), at)
    item, n, m = _by_length(x[1], budget, at), x[2], x[3]
    if item is None:
        return Run.text(0, at, False) if n == 0 else None
    k = n + min(budget + 1, (m - n) if m is not None else budget + 1)  # the most copies a length can leave
    copies = repeat(item, k, 0, at) if k else Run.text(0, at)
    return either(Run.text(0, at), copies, at) if n == 0 else copies

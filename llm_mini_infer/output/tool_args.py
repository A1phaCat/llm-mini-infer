# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/parser/gemma4.py, vllm/parser/qwen3.py, vllm/parser/engine/parser_engine.py and
# vllm/tool_parsers/utils.py of vLLM v0.33.0 (https://github.com/vllm-project/vllm), and changed for llm-mini-infer.
"""A tool call's arguments: the model's syntax to a JSON object, typed by the tool's schema.

- **Gemma 4** (`gemma4_arguments`): `key:value` pairs with unquoted keys, `<|"|>`-delimited strings taken raw, bare
  values kept as strings (marked bare, `_Bare`), nested `{}` and `[]`; the last of two equal keys wins, except at a
  call's top level (below). One trailing `}` (or `)`, when `)` outnumbers `(`) is the call's own closer and is dropped
  first. Every level's closer comes from one scan of the body (`_closer`), so a body nested deeper than the cap (a
  model repeating `{a:`) gives way in time linear in it.
- **Qwen3.8** (`qwen_arguments`): `<parameter=K>V</parameter>` pairs; a missing `</parameter>` is closed by the next
  `<parameter=`; each value is a string with one leading and one trailing newline trimmed (the template's markup). The
  scan (`qwen_parameters`) finds what vLLM's qwen3 parameter patterns find.
- **A call's top level** (ours, since `ArgStream` may have sent its members): a key written twice keeps its first
  value; a Qwen parameter left open when its call ends (the output cut it, or no `</parameter>` came) keeps its value
  so far, as Gemma's open value does (qwen3.py drops it).
- **Typing** (`fix_types`): only the keys under the tool's `properties` are typed, recursively through `properties` and
  `items`; a string value takes the schema's types in the priority null, integer, number, boolean, object, array,
  string (`int()` tried before `float()`, so large integers keep their digits), then a `json.loads` fallback. A key
  without a `type` stays a string. A `$ref` into the tool's `$defs` or `definitions` is followed, the root's too
  (ours: parser_engine.py's `_fix_arg_types` leaves a value under a `$ref` a string, and types nothing under a root
  `$ref`), each definition resolved and typed once a call (`_Typing`), a cycle of refs left unresolved where it closes;
  schemas that reach each other through `anyOf`, `oneOf` or `allOf` type alike, with every type of their cycle.
- **Open values** (ours: parser_engine.py keeps them strings): inside an object schema without `properties` (whose
  `additionalProperties` is absent or true), Gemma's bare values take their literal types, nested ones too: an integer,
  a number, `true`, `false`, `null` (`_literal`); a `<|"|>` string stays a string. Qwen's values carry no such syntax.
- **Each top-level member stands alone:** a value nested deeper than `MAX_NESTING` levels is kept as written, a
  string (Gemma's `_Written`; Qwen's value is not decoded); the other members as usual. The interpreter's own limits
  stay a guard: a member deeper than the converter or the typing recurses is kept as written, or as converted,
  untyped.
- The JSON is `json.dumps(..., ensure_ascii=False)` with the default separators.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..grammar.bounds import def_key

# A call's argument value converts and types to this many levels: its own bracket is level 1. A value nested deeper
# stays as written, a string (vllm-compatibility.md#tool-calls). The typing takes up to 3 frames a level. At 256
# levels it peaks at 795 of the interpreter's 1,000 frames (Gemma, a recursive `$ref` to an object), about 205 to
# spare. Past the interpreter's limit a member is kept as written, or as converted, untyped (a RecursionError).
MAX_NESTING = 256
STRING_DELIM = '<|"|>'
_DELIM_LEN = len(STRING_DELIM)
_PARTIAL_DELIM_SUFFIXES = tuple(STRING_DELIM[:k] for k in range(len(STRING_DELIM), 0, -1))
# What the closer scans stop at: a string's delimiter, or a bracket
_NEST = re.compile(r'<\|"\|>|[{}\[\]]')


class _Bare(str):
    """A value the model wrote bare (not between `<|"|>`): open values take its literal type (`_literal`)."""

    __slots__ = ()


class _Written(str):
    """A top-level value nested deeper than `MAX_NESTING`, kept as written: no typing changes it."""

    __slots__ = ()


class _TooDeep(RecursionError):
    """A value nested deeper than `MAX_NESTING`: met where the interpreter's own limit would be."""


class _FirstWins(dict):
    """The top level of a call's final arguments: a key written twice keeps its first value, which `ArgStream` may
    have sent before the second began."""

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            super().__setitem__(key, value)


def _closer(text: str, i: int, opener: str, at: tuple[dict, int]) -> tuple[int, int]:
    """From just after an `opener` at `i - 1`: the index after its closer and 0, or the text's end and a depth still
    open; gemma4.py's character scan (strings skipped whole, brackets of the other kind ignored), by regex. `at`: the
    closers found so far, by position in the call's whole body, and where `text` starts in it. A scan records the
    closer of every opener inside it (each kind apart), so the levels inside take theirs from the table: the body is
    scanned once, however deep it nests."""
    table, off = at
    n = len(text)
    got = table.get(off + i - 1)
    if got is not None:
        end, depth = got
        if end is None:
            return n, depth
        if end - off <= n:
            return end - off, 0
    open_at: dict[str, list[int]] = {"{": [], "[": []}
    while True:
        m = _NEST.search(text, i)
        if m is None:
            break
        tok = m.group()
        if tok == STRING_DELIM:
            end = text.find(STRING_DELIM, m.end())
            if end == -1:
                break
            i = end + _DELIM_LEN
            continue
        i = m.end()
        if tok in "{[":
            open_at[tok].append(m.start())
            continue
        kind = "{" if tok == "}" else "["
        if open_at[kind]:
            table[off + open_at[kind].pop()] = (off + i, 0)
        elif kind == opener:  # its own closer
            for q in open_at["[" if opener == "{" else "{"]:  # open past it: open in every level inside
                table[off + q] = (None, 1)
            return i, 0
    for q in open_at["{"] + open_at["["]:
        table[off + q] = (None, 1)
    return n, 1


# -- Gemma 4 ---------------------------------------------------------------------------------------------------------

def _strip_partial_delim(value: str) -> str:
    for suffix in _PARTIAL_DELIM_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _without_closer(s: str, more_closes: bool) -> str:
    """`s` without the call's closer: a `}`, or a `)` when the body has more closes than opens (`more_closes`)."""
    if s and (s[-1] == "}" or (more_closes and s[-1] == ")")):
        return s[:-1]
    return s


def _parse_gemma4_args(args_str: str, *, partial: bool = False, _at: tuple[dict, int] | None = None,
                       trace: list | None = None, per_member: bool = False, _depth: int = 0) -> dict:
    """Gemma 4's `key:value` pairs as a dict; `partial`: an unterminated tail (a nested object or array the text
    cut), whose bare values at its end are left out. `_at`: the closers found so far (`_closer`). `trace`: each pair
    added, in order, as (key, where its key starts, its value's kind: str, open_str, obj, open_obj, arr, open_arr or
    bare, where the value starts, where the pair ends) (`ArgStream`). `per_member` (the top level of a call's final
    arguments): a key written twice keeps its first value, and a value nested deeper than `MAX_NESTING` (or than the
    converter recurses) is kept as written (`_Written`), the pairs after it parsed. `_depth`: this object's level (the
    call's body is 0); deeper than `MAX_NESTING`, `_TooDeep`."""
    if _depth > MAX_NESTING:
        raise _TooDeep(f"a value nested deeper than {MAX_NESTING} levels")
    if not args_str or not args_str.strip():
        return {}
    at = ({}, 0) if _at is None else _at
    result: dict = _FirstWins() if per_member else {}
    i = 0
    n = len(args_str)
    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        if key.startswith(STRING_DELIM) and key.endswith(STRING_DELIM):
            key = key[_DELIM_LEN:-_DELIM_LEN]
        i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break
        if args_str.startswith(STRING_DELIM, i):
            i += _DELIM_LEN
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                value = args_str[val_start:]
                if partial:
                    value = _strip_partial_delim(value)
                result[key] = value
                if trace is not None:
                    trace.append((key, key_start, "open_str", val_start, n))
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + _DELIM_LEN
            if trace is not None:
                trace.append((key, key_start, "str", val_start, i))
        elif args_str[i] == "{":
            obj_start = i + 1
            i, depth = _closer(args_str, i + 1, "{", at)
            inner = (at[0], at[1] + obj_start)
            try:
                if depth > 0:
                    result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True, _at=inner, _depth=_depth + 1)
                else:
                    result[key] = _parse_gemma4_args(args_str[obj_start: i - 1], _at=inner, _depth=_depth + 1)
            except RecursionError:
                if not per_member:
                    raise
                result[key] = _Written(args_str[obj_start - 1:i])
            if trace is not None:
                trace.append((key, key_start, "open_obj" if depth > 0 else "obj", obj_start - 1, i))
        elif args_str[i] == "[":
            arr_start = i + 1
            i, depth = _closer(args_str, i + 1, "[", at)
            inner = (at[0], at[1] + arr_start)
            try:
                if depth > 0:
                    result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True, _at=inner, _depth=_depth + 1)
                else:
                    result[key] = _parse_gemma4_array(args_str[arr_start: i - 1], _at=inner, _depth=_depth + 1)
            except RecursionError:
                if not per_member:
                    raise
                result[key] = _Written(args_str[arr_start - 1:i])
            if trace is not None:
                trace.append((key, key_start, "open_arr" if depth > 0 else "arr", arr_start - 1, i))
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:  # no progress: malformed input
                break
            raw_val = args_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                break
            result[key] = _Bare(raw_val)
            if trace is not None:
                trace.append((key, key_start, "bare", val_start, i))
    return dict(result) if per_member else result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False, _at: tuple[dict, int] | None = None,
                        _depth: int = 1) -> list:
    if _depth > MAX_NESTING:
        raise _TooDeep(f"a value nested deeper than {MAX_NESTING} levels")
    at = ({}, 0) if _at is None else _at
    items: list = []
    i = 0
    n = len(arr_str)
    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        if arr_str.startswith(STRING_DELIM, i):
            i += _DELIM_LEN
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + _DELIM_LEN
        elif arr_str[i] == "{":
            obj_start = i + 1
            i, depth = _closer(arr_str, i + 1, "{", at)
            inner = (at[0], at[1] + obj_start)
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True, _at=inner, _depth=_depth + 1))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start: i - 1], _at=inner, _depth=_depth + 1))
        elif arr_str[i] == "[":
            sub_start = i + 1
            i, depth = _closer(arr_str, i + 1, "[", at)
            inner = (at[0], at[1] + sub_start)
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True, _at=inner, _depth=_depth + 1))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start: i - 1], _at=inner, _depth=_depth + 1))
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                break
            raw_val = arr_str[val_start:i].strip()
            if partial and raw_val.endswith("."):
                break
            items.append(_Bare(raw_val))
    return items


def fallback_arguments(raw_args: str) -> str:
    """A call's arguments when its converter fails (parser_engine.py's `_extract_args_json`, with a nesting the
    converters cannot hold failing the same way): the stripped body when it is not a JSON object, else its
    `arguments` or `parameters`, else the object without `name`."""
    body = raw_args.strip()
    if not body:
        return "{}"
    try:
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            return body
        for key in ("arguments", "parameters"):
            if key in parsed:
                value = parsed[key]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return json.dumps({k: v for k, v in parsed.items() if k != "name"}, ensure_ascii=False)
    except (ValueError, RecursionError):
        return body


def gemma4_parse(raw_args: str, per_member: bool = False) -> dict:
    """A complete (or truncated) Gemma 4 call body, after its `{` or `(`, as its dict (bare values `_Bare`); a
    RecursionError when a value nests deeper than `MAX_NESTING` (`_TooDeep`; one frame a level), unless `per_member` (a
    call's final arguments: `_parse_gemma4_args`)."""
    text = raw_args.strip()
    text = _without_closer(text, text.endswith(")") and text.count("(") < text.count(")"))
    return _parse_gemma4_args(text, per_member=per_member)


def gemma4_arguments(raw_args: str) -> str:
    """A complete (or truncated) Gemma 4 call body, after its `{` or `(`, as JSON."""
    return json.dumps(gemma4_parse(raw_args), ensure_ascii=False)


# -- Qwen3.8 ---------------------------------------------------------------------------------------------------------

# What vLLM's qwen3 parameter pattern matches: an opener, its name up to the first `>`, its value up to a closer or the
# next opener
_OPEN = re.compile(r"<\s*parameter\s*=\s*")
_CLOSE = re.compile(r"<\s*/\s*parameter\s*>")


def qwen_parameters(text: str, at: list[int] | None = None, i: int = 0) -> list[tuple[str, str]]:
    """The (name, raw value) pairs vLLM's qwen3 parameter pattern finds: from each `<parameter=` (a `<` whose
    opener matches), the name runs to the first `>`, and the value to the first `</parameter>` or `<parameter=` after
    it. With no `>`, or no end, after an opener, no later one matches either. The scan starts at `i`.
    `at`: each pair's opener's offset, and after them where the last pair ended (`i` without one)."""
    out: list[tuple[str, str]] = []
    ended = i
    while True:
        s = text.find("<", i)
        if s < 0:
            break
        head = _OPEN.match(text, s)
        if head is None:
            i = s + 1
            continue
        gt = text.find(">", head.end())
        if gt < 0:
            break
        j = gt + 1
        found = False
        while True:
            q = text.find("<", j)
            if q < 0:
                break
            close = _CLOSE.match(text, q)
            if close is not None:
                out.append((text[head.end():gt], text[gt + 1:q]))
                found = True
                i = ended = close.end()
                break
            if _OPEN.match(text, q) is not None:
                out.append((text[head.end():gt], text[gt + 1:q]))
                found = True
                i = ended = q
                break
            j = q + 1
        if not found:
            break
        if at is not None:
            at.append(s)
    if at is not None:
        at.append(ended)
    return out


def open_parameter(text: str, i: int) -> tuple[int, str, int] | None:
    """The parameter a scan (`qwen_parameters`) stopped in, from `i` on: (its opener's offset, its name, where its value
    starts), as vLLM's qwen3 parser finds an open parameter."""
    while True:
        s = text.find("<", i)
        if s < 0:
            return None
        head = _OPEN.match(text, s)
        if head is not None:
            gt = text.find(">", head.end())
            if gt < 0:
                return None  # no `>` after this opener, nor after a later one
            name_at = head.end()
            if name_at == gt and text[name_at - 1] != "=":
                name_at -= 1  # a name of whitespace alone: its last character, as vLLM's parser names it
            if name_at < gt:
                return s, text[name_at:gt], gt + 1
        i = s + 1


def _open_value(value: str) -> str:
    """An open parameter's value so far: without a tail that may still grow into its end (a cut `</parameter>`), its
    wrapping newlines trimmed."""
    return _trim_wrapping_newlines(value[: len(value) - len(_value_end_window(value))])


def _trim_wrapping_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def qwen_arguments(raw_args: str) -> str:
    """A Qwen3.8 call body, after `<function=NAME>`, as JSON. A parameter left open (no closer and no parameter after
    it: the output cut it, or no `</parameter>` came) keeps its value so far, as Gemma's does; a parameter written twice
    keeps its first value. Both because `ArgStream` may have sent them (ours: qwen3.py drops the open one and keeps the
    last)."""
    params: dict[str, object] = {}
    at: list[int] = []
    for name, value in qwen_parameters(raw_args, at):
        params.setdefault(name, _trim_wrapping_newlines(value))
    cut = open_parameter(raw_args, at[-1])
    if cut is not None:
        params.setdefault(cut[1], _open_value(raw_args[cut[2]:]))
    return json.dumps(params, ensure_ascii=False)


# -- Typing by the schema --------------------------------------------------------------------------------------------

# The keywords the typing reads: a resolved schema keeps these alone, so a chain's merges stay small
_TYPING = ("type", "enum", "anyOf", "oneOf", "allOf", "properties", "items", "additionalProperties", "$ref")
_OWN_TYPES = ("type", "enum", "anyOf", "oneOf", "allOf")


def _typing_view(schema: dict) -> dict:
    return {k: schema[k] for k in _TYPING if k in schema}


class _Typing:
    """One call's typing over its tool's definitions (`defs`), in time linear in what it reads:
    - **`$ref` chains** (`chain`): a definition's own keywords under those of the definition it refers to (as JSON
      Schema's `$ref` beside keywords is merged here), each chain resolved once for every definition on it. A chain
      that closes a cycle keeps its last `$ref`, unresolved. A chain that re-enters a definition an enclosing schema
      followed, without a cycle, is followed, as the tests' oracle (the reference vLLM's typing over the refs
      inlined) follows it.
    - **Types** (`types_of`): a schema's `type` and `enum`, and the types of what its `anyOf`, `oneOf` and `allOf`
      hold, each schema and each list typed once. Schemas that reach each other through those lists (a cycle through
      alternatives) all take the cycle's every type, and each `$ref` on the cycle adds its own `type` and `enum`, or
      string with neither, as the oracle's `$ref` left unresolved where its inlining stops; so the types do not depend
      on which schema the call types first. Type names are kept as `coerce` reads them (an alias as its type, any other
      name as "")."""

    __slots__ = ("chains", "defs", "keep", "types")

    def __init__(self, defs: dict | None):
        self.defs = defs or {}
        self.chains: dict[str, dict] = {}  # a definition -> its chain resolved
        self.types: dict[tuple, frozenset[str]] = {}  # a schema (`_node`) or a list -> its types, before the default
        self.keep: list[Any] = []  # what the nodes' ids name, alive while they are

    def target(self, schema: Any) -> str | None:
        """The definition a schema's `$ref` names, when it is one of `defs`' objects, its name read as the grammar
        reads it (`grammar.bounds.def_key`: percent-decoded, then JSON Pointer's escapes)."""
        if isinstance(schema, dict) and self.defs:
            key = def_key(schema.get("$ref"), self.defs)
            if key is not None and isinstance(self.defs[key], dict):
                return key
        return None

    def resolve(self, schema: Any) -> Any:
        """`schema` with its `$ref` chain followed (the keywords the typing reads)."""
        key = self.target(schema)
        if key is None:
            return schema
        own = {k: schema[k] for k in _TYPING if k in schema and k != "$ref"}
        chain = self.chain(key)
        return {**own, **chain} if own else chain

    def chain(self, key: str) -> dict:
        """Definition `key` with its chain of `$ref`s followed, until one names a definition already on it (a cycle:
        that `$ref` stays); the result kept for every definition the walk passed (on a cycle, each member's own, from
        it around to the member before it)."""
        got = self.chains.get(key)
        if got is not None:
            return got
        path, at = [key], {key: 0}
        while True:
            nxt = self.target(self.defs[path[-1]])
            if nxt is None or nxt in self.chains or nxt in at:
                break
            at[nxt] = len(path)
            path.append(nxt)
        views = [_typing_view(self.defs[k]) for k in path]
        res: dict | None = None
        upto = len(path)
        if nxt is not None and nxt in at:  # path[upto:] is a cycle
            upto = at[nxt]
            for k, r in zip(path[upto:], _rotations(views[upto:]), strict=True):
                self.chains[k] = r
            res = self.chains[path[upto]]
        elif nxt is not None:  # joins a chain that ends
            res = self.chains[nxt]
        for j in range(upto - 1, -1, -1):
            own = views[j] if res is None else {a: b for a, b in views[j].items() if a != "$ref"}
            res = own if res is None else {**own, **res}
            self.chains[path[j]] = res
        return self.chains[key]

    def types_of(self, schema: Any) -> list[str]:
        """Every type a schema allows (`schema_types`)."""
        if not isinstance(schema, dict):
            return ["string"]
        node = self._node(schema)
        got = self.types.get(node)
        if got is None:
            self._solve(node, schema)
            got = self.types[node]
        return _ordered(got or _STRING)

    def _node(self, x: dict) -> tuple:
        """A schema's node: a `$ref` by its definition and its own type keywords' values, any other by identity."""
        key = self.target(x)
        if key is None:
            return ("s", id(x))
        return ("r", key, *((k, id(x[k])) for k in _OWN_TYPES if k in x))

    def _view(self, node: tuple, x: Any) -> Any:
        """What a node types: a `$ref`'s type keywords under its chain's, a list or another schema as it is."""
        if node[0] != "r":
            self.keep.append(x)
            return x
        chain = self.chain(node[1])
        return {**{k: x[k] for k in _OWN_TYPES if k in x}, **{k: chain[k] for k in _OWN_TYPES if k in chain}}

    def _edges(self, node: tuple, view: Any) -> list[tuple[tuple, Any]]:
        """A node's successors, with what each types as written: a schema's lists, a list's schemas."""
        if node[0] == "l":
            return [(self._node(e), e) for e in view if isinstance(e, dict)]
        return [(("l", id(v)), v) for k in ("anyOf", "oneOf", "allOf") if isinstance(v := view.get(k), list)]

    def _solve(self, start: tuple, schema: dict) -> None:
        """The types of `start` and of every node it reaches that has none yet, by strongly connected components
        (Tarjan's algorithm, iterative, successors first): a component of one node takes its own types and its
        successors'; a larger one, a cycle through alternatives, takes the class docstring's."""
        index: dict[tuple, int] = {}
        low: dict[tuple, int] = {}
        views: dict[tuple, Any] = {}
        path: list[tuple] = []
        on: set[tuple] = set()
        work: list[tuple[tuple, Any]] = []

        def push(node: tuple, x: Any) -> None:
            index[node] = low[node] = len(index)
            views[node] = v = self._view(node, x)
            path.append(node)
            on.add(node)
            work.append((node, iter(self._edges(node, v))))

        push(start, schema)
        while work:
            node, it = work[-1]
            for nxt, x in it:
                if nxt in self.types:
                    continue
                if nxt not in index:
                    push(nxt, x)
                    break
                if nxt in on:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    up = work[-1][0]
                    low[up] = min(low[up], low[node])
                if low[node] == index[node]:
                    members = []
                    while True:
                        m = path.pop()
                        on.discard(m)
                        members.append(m)
                        if m == node:
                            break
                    self._settle(members, views)

    def _settle(self, members: list[tuple], views: dict[tuple, Any]) -> None:
        inside = set(members)
        got: set[str] = set()
        for m in members:
            v = views[m]
            if m[0] == "l":
                if any(not isinstance(e, dict) for e in v):
                    got |= _STRING
                for nxt, e in self._edges(m, v):
                    if nxt not in inside:
                        got |= self.types[nxt] or _STRING
                    elif nxt[0] == "r":  # a `$ref` on the cycle: as written, where the cycle closes
                        got |= _written(e) or _STRING
            else:
                got |= _written(v)
                got.update(*(self.types[nxt] for nxt, _ in self._edges(m, v) if nxt not in inside))
        fixed = frozenset(got)
        for m in members:
            self.types[m] = fixed


_STRING = frozenset(("string",))


def _type_name(t: str) -> str:
    """A type name as `coerce` reads it: an alias as its type, a name it does not know as ""."""
    k = t.strip().lower()
    k = _TYPE_ALIASES.get(k, k)
    return k if k in _TYPE_PRIORITY else ""


def _written(schema: dict) -> frozenset[str]:
    """The types a schema's own `type` and `enum` name."""
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(_type_name(t))
    elif isinstance(t, list):
        types.update(_type_name(x) for x in t if isinstance(x, str))
    values = schema.get("enum")
    if isinstance(values, list):
        for value in values:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")
    return frozenset(types)


def _ordered(types: frozenset[str]) -> list[str]:
    return sorted(types, key=lambda t: _TYPE_PRIORITY.index(t) if t in _TYPE_PRIORITY else len(_TYPE_PRIORITY))


def _rotations(views: list[dict]) -> list[dict]:
    """For a cycle of definitions' views (each one's `$ref` names the next, the last's the first): each member's chain,
    from it around to the member before it, later keywords winning and the last `$ref` kept; in time linear in the
    cycle (for each keyword, the nearest member before each that has it)."""
    n = len(views)
    out: list[dict] = [{} for _ in range(n)]
    for k in {k for v in views for k in v if k != "$ref"}:
        last = -n - 1
        for t in range(2 * n):
            if t >= n and last >= t - n:
                out[t - n][k] = views[last % n][k]
            if k in views[t % n]:
                last = t
    for m in range(n):
        out[m]["$ref"] = views[m - 1]["$ref"]
    return out


def schema_types(schema: Any, defs: dict | None = None) -> list[str]:
    """Every type a schema allows: `type` (a string or a list), the types of `enum`'s values, and `anyOf`, `oneOf` and
    `allOf`'s, recursively, a `$ref` into `defs` followed; ["string"] when it names none."""
    return _Typing(defs).types_of(schema)


_LITERALS = {"true": True, "false": False, "null": None}
_INTEGER = re.compile(r"-?[0-9]+\Z")
_NUMBER = re.compile(r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")  # the call grammar's number


def _literal(value: str) -> Any:
    """A bare Gemma value as its literal: `true`, `false`, `null`, an integer (every digit kept) or a finite number;
    else the string."""
    if value in _LITERALS:
        return _LITERALS[value]
    try:
        if _INTEGER.match(value):
            return int(value)
        if _NUMBER.match(value):
            f = float(value)
            return f if math.isfinite(f) else value
    except ValueError:  # more digits than int() takes
        pass
    return value


def _is_open(schema: Any) -> bool:
    """An object schema without `properties` whose `additionalProperties` is absent or true: any key, any value."""
    if not isinstance(schema, dict) or "properties" in schema or schema.get("additionalProperties", True) is not True:
        return False
    t = schema.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t)


def _literals(value: dict | list) -> bool:
    """Every bare value in an open value, nested ones too, as its literal (in place); whether any changed."""
    changed = False
    for k, v in list(value.items() if isinstance(value, dict) else enumerate(value)):
        if isinstance(v, _Bare):
            typed = _literal(v)
            if typed is not v:
                value[k] = typed
                changed = True
        elif isinstance(v, dict | list):
            changed = _literals(v) or changed
    return changed


_TYPE_ALIASES: dict[str, str] = {
    "str": "string", "text": "string", "varchar": "string", "char": "string", "enum": "string",
    "int": "integer", "int32": "integer", "int64": "integer", "uint": "integer", "uint32": "integer",
    "uint64": "integer", "long": "integer", "short": "integer", "unsigned": "integer",
    "float": "number", "float32": "number", "float64": "number", "double": "number",
    "bool": "boolean", "dict": "object", "arr": "array", "list": "array", "sequence": "array",
}
_TYPE_PRIORITY = ("null", "integer", "number", "boolean", "object", "array", "string")


def _finite(obj: Any) -> bool:
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


_JSON_NEST = re.compile(r'"(?:[^"\\]|\\.)*"?|[\[{]|[\]}]')  # a JSON string (skipped whole), or a bracket


def _too_deep(text: str) -> bool:
    """Whether JSON text nests its brackets deeper than `MAX_NESTING` (strings skipped), without recursion."""
    if text.count("{") + text.count("[") <= MAX_NESTING:
        return False
    depth = 0
    for m in _JSON_NEST.finditer(text):
        c = text[m.start()]
        if c in "[{":
            depth += 1
            if depth > MAX_NESTING:
                return True
        elif c in "]}":
            depth -= 1
    return False


def coerce(value: str, schema_type: str | list[str]) -> Any:
    """A string value as the first of the schema's types it parses as (module docstring), else `json.loads`'s value,
    else the string; JSON nested deeper than `MAX_NESTING` stays the string."""
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    normalized = {_TYPE_ALIASES.get(key, key) for t in schema_type for key in [t.strip().lower()]}
    for candidate in _TYPE_PRIORITY:
        if candidate not in normalized:
            continue
        if candidate == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate == "string":
            return value
        if candidate == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate == "number":
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            return val if val != int(val) else int(val)
        if candidate == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate in ("object", "array"):
            if _too_deep(value):
                return value
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _finite(parsed):
                return parsed
            continue
    if _too_deep(value):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if not _finite(parsed):
        return value
    return parsed


def _coerce_value(value: object, schema: dict, typing: _Typing) -> tuple[object, bool]:
    if isinstance(value, _Written):
        return value, False
    schema = typing.resolve(schema)
    if isinstance(value, str):
        coerced = coerce(value, typing.types_of(schema))
        if coerced is not value:
            return coerced, True
        return value, False
    if isinstance(value, dict):
        if _is_open(schema):
            return value, _literals(value)
        nested = schema.get("properties")
        if isinstance(nested, dict):
            _, changed = _coerce_dict(value, nested, typing)
            return value, changed
        return value, False
    if isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            changed = False
            for i, item in enumerate(value):
                coerced, item_changed = _coerce_value(item, items_schema, typing)
                if item_changed:
                    value[i] = coerced
                    changed = True
            return value, changed
        return value, False
    as_str = json.dumps(value, ensure_ascii=False)
    coerced = coerce(as_str, typing.types_of(schema))
    if type(coerced) is not type(value) or coerced != value:
        return coerced, True
    return value, False


def _coerce_dict(args: dict, properties: dict, typing: _Typing) -> tuple[dict, bool]:
    changed = False
    for key, value in args.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        coerced, val_changed = _coerce_value(value, prop, typing)
        if val_changed:
            args[key] = coerced
            changed = True
    return args, changed


def fix_types(args_json: str, parameters: dict | None, parsed: dict | None = None) -> str:
    """The arguments typed by the tool's `parameters` (module docstring), a root `$ref` followed; unchanged without
    `properties` (or an open root). `parsed`: the arguments as the converter gave them (Gemma's, with its bare values
    marked), typed in place of `args_json`'s decoding. Each top-level member is typed alone: one nested deeper than the
    typing recurses keeps `args_json`'s value."""
    if not isinstance(parameters, dict):
        return args_json
    defs = {}
    for key in ("definitions", "$defs"):  # the grammar's merge: `$defs` wins a name both have
        if isinstance(parameters.get(key), dict):
            defs.update(parameters[key])
    typing = _Typing(defs)
    root = typing.resolve(parameters)
    properties = root.get("properties") if isinstance(root, dict) else None
    is_open = _is_open(root)
    if not is_open and (not properties or not isinstance(properties, dict)):
        return args_json
    args = parsed
    if args is None:
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, ValueError):
            return args_json
    if not isinstance(args, dict):
        return args_json
    changed = False
    failed = []
    for key, value in args.items():  # each member typed alone, as ArgStream types and sends it
        try:
            if is_open:
                if isinstance(value, _Bare):
                    typed = _literal(value)
                    if typed is not value:
                        args[key], changed = typed, True
                elif isinstance(value, dict | list):
                    changed = _literals(value) or changed
            elif isinstance(prop := properties.get(key), dict):
                typed, member_changed = _coerce_value(value, prop, typing)
                if member_changed:
                    args[key], changed = typed, True
        except RecursionError:  # nested deeper than the typing recurses: this member as converted
            failed.append(key)
    if failed:
        untyped = json.loads(args_json)  # the typing changes nested values in place
        for key in failed:
            args[key] = untyped[key]
        changed = True
    return json.dumps(args, ensure_ascii=False) if changed else args_json


# -- Arguments in pieces ---------------------------------------------------------------------------------------------

_STRUCTURAL = frozenset(",:{}[]<")  # gemma4.py's arg_structural_chars: a Gemma delta without one changes nothing
_NEST_OBJ = re.compile(r'<\|"\|>|[{}]')
_NEST_ARR = re.compile(r'<\|"\|>|[\[\]]')
_VALUE_END = re.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")  # qwen3.py's _PARAM_VALUE_END_RE
_WS = " \t\n\r\f\v"


def _may_end_value(tail: str, at: int = 0) -> bool:
    """Whether the tail from the `<` at `at` may still grow into a Qwen value's end (`_VALUE_END`): qwen3.py's
    `_PARAM_VALUE_END_RE.fullmatch(tail, partial=True)`."""
    i, n = at + 1, len(tail)
    while i < n and tail[i].isspace():
        i += 1
    if i < n and tail[i] == "/":
        i += 1
        while i < n and tail[i].isspace():
            i += 1
        closing = True
    else:
        closing = False
    word = "parameter"
    k = 0
    while i < n and k < len(word) and tail[i] == word[k]:
        i += 1
        k += 1
    if i == n:
        return True
    if k < len(word):
        return False
    while i < n and tail[i].isspace():
        i += 1
    if i == n:
        return True
    return tail[i] == (">" if closing else "=") and i == n - 1


def _value_end_window(text: str) -> str:
    """The suffix of an open Qwen value that may still grow into its end (qwen3.py's `_param_value_end_window`)."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _may_end_value(tail) else ""


# Where a Qwen scan of a text that grows at its end may resume (`ArgStream._qwen_members`). Each pattern has one `<`,
# its first character. So of the `<`s that match none yet, only the text's last may still grow into a match.

def _may_open(text: str, s: int) -> bool:
    """Whether the `<` at `s` opens a parameter (`_OPEN`), or may still. It may while the text after it is
    whitespace, a part of `parameter` (or all of it and whitespace), then the text's end."""
    if _OPEN.match(text, s) is not None:
        return True
    i, n = s + 1, len(text)
    while i < n and text[i].isspace():
        i += 1
    word = "parameter"
    k = 0
    while i < n and k < len(word) and text[i] == word[k]:
        i += 1
        k += 1
    if k == len(word):
        while i < n and text[i].isspace():
            i += 1
    return i == n


def _open_from(text: str, i: int) -> int:
    """The first `<` from `i` that opens a parameter or may still (`_may_open`), else the text's end: no `<` before it
    opens one, whatever follows."""
    s = text.find("<", i)
    while s >= 0 and not _may_open(text, s):
        s = text.find("<", s + 1)
    return len(text) if s < 0 else s


def _value_end(text: str, j: int) -> int:
    """The first `<` from `j` that ends a value (`_CLOSE`, or an opener: qwen_parameters' end of a value), else -1."""
    q = text.find("<", j)
    while q >= 0 and _CLOSE.match(text, q) is None and _OPEN.match(text, q) is None:
        q = text.find("<", q + 1)
    return q


def _end_from(text: str, j: int) -> int:
    """Where a search for a value's end resumes, none found from `j`. That is the text's last `<` if it may still grow
    into one, else the text's end."""
    q = text.rfind("<", j)
    return q if q >= 0 and _may_end_value(text, q) else len(text)


def _trailing_ws(text: str) -> int:
    return len(text) - len(text.rstrip())


class ArgStream:
    """One call's arguments sent in pieces as it streams (vllm-compatibility.md#tool-calls): the reference vLLM's
    `_compute_arg_delta`, `_safe_arg_prefix` and value extenders (parser_engine.py, gemma4.py, qwen3.py) over this
    module's converters and typing, so that every piece is a prefix of the call's final arguments. Each piece extends
    the JSON sent so far up to its last top-level member's `"key": `, then that member's string as far as it goes when
    the key's type is exactly string (or the tool has no `properties`); every other value is held until a member after
    it begins, or the call ends (`flush`). A Gemma delta without a structural character changes nothing, and so does a
    Qwen delta without `<` or `>` (no parameter can begin or end).

    Linear in the call's text, but for Gemma's text after the last member (below). The members before the last are
    converted and typed once, and their JSON sent once. The parse starts at the last member while it is open, else
    after it, and reads only the text from there (`raw`). An open string grows by its new text alone (`_extend`). An
    open object or array is scanned by its new text alone (`_nest_closed`), and parsed again once it closes. Qwen's
    scan of the text after the last member resumes where no `<` before can still begin a parameter (`_resume`).
    Gemma's partial parse reads that text again at each piece with a structural character, so `MAX_GAP` bounds it.
    A key written twice stops the pieces (`broken`). So do a nesting deeper than `MAX_NESTING`, and `MAX_GAP`
    characters that begin no member, after the last member or from the call's start. `flush` then sends the rest of
    the final arguments, which extend what was sent (the module docstring's top-level rules)."""

    MAX_GAP = 4096

    __slots__ = ("_base", "_closes", "_extend", "_keys", "_last", "_len", "_nest", "_open_root", "_opens", "_out",
                 "_props", "_resume", "_seg", "_tail", "_typing", "broken", "gemma", "raw", "streams", "string_keys")

    def __init__(self, gemma: bool, parameters: Any):
        self.gemma = gemma
        # The call's argument text (Gemma: after its `{`; Qwen: after `<function=NAME>`): `raw` from offset `_base` on
        # (what a parse reads: nothing before it is read again), the text appended since (`_tail`), and its length
        self.raw = ""
        self._base = 0
        self._tail: list[str] = []
        self._len = 0
        self._out: list[str] = []  # the pieces sent
        self._seg: list[str] = []  # the last member's text sent: its key, and its string as far as it went
        self.broken = False
        self._keys: set[str] = set()  # the members' keys before the last
        # the last member: (key, value, kind, its key's offset, its value's, its end) in the text; None before the first
        self._last: tuple | None = None
        self._extend: Any = None  # the open string's extender (`_gemma_extender`, `_qwen_extender`)
        self.streams = False  # the open string is a streamable key's
        # Gemma's open object or array: [the text not scanned yet, depth, in a string, kind]
        self._nest: list | None = None
        self._opens = self._closes = 0  # Gemma: "(" and ")" in the body (whether a trailing ")" is its closer)
        # Qwen: (the window's start, where its scan resumes, where a nameless parameter's value-end search resumes
        # or -1), while the window holds no parameter
        self._resume: tuple[int, int, int] | None = None
        defs: dict = {}
        if isinstance(parameters, dict):
            for key in ("definitions", "$defs"):
                if isinstance(parameters.get(key), dict):
                    defs.update(parameters[key])
        self._typing = _Typing(defs)
        root = self._typing.resolve(parameters) if isinstance(parameters, dict) else None
        props = root.get("properties") if isinstance(root, dict) else None
        self._open_root = _is_open(root)
        self._props = props if isinstance(props, dict) and props and not self._open_root else None
        # the keys whose strings stream: parser_engine.py's `_streamable_string_keys` through this module's typing
        self.string_keys: set[str] | None = None
        if self._props is not None:
            self.string_keys = {k for k, v in self._props.items()
                                if isinstance(v, dict) and self._typing.types_of(self._typing.resolve(v)) == ["string"]}

    @property
    def sent(self) -> str:
        """The JSON sent so far."""
        return "".join(self._out)

    # -- input --------------------------------------------------------------------------------------------------

    def append(self, text: str) -> None:
        """Argument text that sends nothing (the call's first: its name goes out instead)."""
        self._tail.append(text)
        self._len += len(text)
        if self.gemma:
            self._opens += text.count("(")
            self._closes += text.count(")")

    def piece(self, text: str) -> str | None:
        """Argument text appended to the call: the piece it sends, or None. Once the pieces stop, the text is not
        kept: `flush` sends the rest from the final arguments."""
        if self.broken:
            return None
        self.append(text)
        if self._extend is not None:
            diff = self._extend(text)
            if diff is not None:
                if not diff or not self.streams:
                    return None
                self._seg.append(diff)
                self._out.append(diff)
                return diff
            self._extend = None
        elif self._nest is not None:
            if not self._nest_closed(text):
                return None  # the open object or array: vLLM's prefix stops at its key
        elif (self.gemma and _STRUCTURAL.isdisjoint(text)) or (not self.gemma and "<" not in text and ">" not in text):
            return None
        return self._recompute()

    def flush(self, final: str) -> str | None:
        """The call ended with these final arguments: the rest to send, when they extend what was sent."""
        sent = "".join(self._out)
        if final and len(final) > len(sent) and (not sent or final.startswith(sent)):
            self._out = [final]
            return final[len(sent):]
        return None

    # -- the prefix ----------------------------------------------------------------------------------------------

    def _window(self, start: int) -> str:
        """The text from `start` on (at or after `_base`, which moves there: no parse reads before it again)."""
        tail = "".join(self._tail) if self._tail else ""
        self._tail.clear()
        cut = start - self._base
        raw = self.raw[cut:] + tail if cut <= len(self.raw) else tail[cut - len(self.raw):]
        self.raw, self._base = raw, start
        return raw

    def _recompute(self) -> str | None:
        try:
            return self._advance()
        except RecursionError:
            # nested deeper than `MAX_NESTING` (`_TooDeep`), or than the interpreter recurses: the final arguments
            # (`flush`) keep the members sent, and give each member that fails its own fallback (`gemma4_parse`'s
            # `per_member`, `fix_types`)
            self.broken = True
            return None

    def _advance(self) -> str | None:
        last = self._last
        complete = last is not None and not last[2].startswith("open")
        start = last[5] if complete else (last[3] if last is not None else 0)
        if (complete or last is None) and self._len - start > self.MAX_GAP:
            self.broken = True  # text after the last member, or from the call's start, that begins none
            return None
        text = self._window(start)
        members = self._gemma_members(start, text) if self.gemma else self._qwen_members(start, text)
        if complete:
            if not members:
                return None  # no member began after the last: nothing changes
            members = [last, *members]
        elif not members:
            return None
        keys = [m[0] for m in members]
        if len(set(keys)) < len(keys) or self._keys.intersection(keys):
            self.broken = True  # a key written twice: the sent arguments are no longer the final's prefix
            return None
        self._nest = None
        fixed = "".join(self._member(m[0], m[1]) + ", " for m in members[:-1])
        self._keys.update(keys[:-1])
        key, value, kind, _, value_at, _ = self._last = members[-1]
        typed = self._typed(key, value)
        streams = isinstance(typed, str) and (self.string_keys is None or key in self.string_keys)
        seg = json.dumps(key, ensure_ascii=False) + ": " + (json.dumps(typed, ensure_ascii=False)[:-1] if streams
                                                             else "")
        # what follows the members sent before the last: the last member's text sent must lead it
        safe = ("" if self._out else "{") + fixed + seg
        sent = "".join(self._seg)
        if not safe.startswith(sent):
            self.broken = True  # not the final arguments' prefix: the call's end sends the rest
            return None
        piece = safe[len(sent):]
        self._seg = [seg]
        if piece:
            self._out.append(piece)
        self.streams = streams
        if kind == "open_str":
            self._extend = self._gemma_extender(value_at) if self.gemma else self._qwen_extender(value_at)
        elif kind in ("open_obj", "open_arr"):
            self._nest = [text[value_at + 1 - start:], 1, False, kind]
            if self._nest_closed(""):
                self._nest = None  # open only by its last `}`, which may be the call's closer: nothing to scan
        return piece or None

    def _member(self, key: str, value: Any) -> str:
        return json.dumps(key, ensure_ascii=False) + ": " + json.dumps(self._typed(key, value), ensure_ascii=False)

    def _typed(self, key: str, value: Any) -> Any:
        """A member's value as `fix_types` types it (per member, as it types each)."""
        if self._open_root:
            if isinstance(value, _Bare):
                return _literal(value)
            if isinstance(value, dict | list):
                _literals(value)
            return value
        if self._props is None:
            return value
        prop = self._props.get(key)
        if not isinstance(prop, dict):
            return value
        return _coerce_value(value, prop, self._typing)[0]

    # -- Gemma ---------------------------------------------------------------------------------------------------

    def _gemma_members(self, start: int, text: str) -> list[tuple]:
        """The members of `text`, from `start` on, as the converter parses the whole body (stripped, its closer
        dropped, partial): [(key, value, kind, its key's offset, its value's, its end)]; a key written twice keeps both
        (its values None)."""
        stripped = _without_closer(text.rstrip(), self._opens < self._closes)
        trace: list = []
        parsed = _parse_gemma4_args(stripped, partial=True, trace=trace)
        values = list(parsed.values())
        if len(values) != len(trace):
            values = [None] * len(trace)
        return [(k, v, kind, start + at, start + vat, start + end)
                for (k, at, kind, vat, end), v in zip(trace, values, strict=True)]

    def _gemma_extender(self, value_at: int) -> Any:
        """gemma4.py's `_gemma4_arg_value_extender` for the open string whose value starts at `value_at`: it sends the
        new text of the string, holding back what the converter holds (trailing whitespace, the call's closer, a
        partial `<|"|>`), until a `<|"|>` closes it (None)."""
        raw = self.raw  # from `_base`, as `_advance` just read it
        stripped = raw[value_at - self._base:].rstrip()
        body = _without_closer(stripped, self._opens < self._closes)
        held_len = _trailing_ws(raw) + len(stripped) - len(_strip_partial_delim(body))
        held = [raw[len(raw) - held_len:]] if held_len else []  # its parts: a whitespace run kept apart, not copied

        def extend(delta: str) -> str | None:
            if held and delta.isspace():  # more whitespace after what is held: still held, nothing sent
                held.append(delta)
                return ""
            text = "".join(held) + delta
            if STRING_DELIM in text:
                return None
            visible = _strip_partial_delim(_without_closer(text.rstrip(), self._opens < self._closes))
            held[:] = [text[len(visible):]] if len(text) > len(visible) else []
            return json.dumps(visible, ensure_ascii=False)[1:-1]

        return extend

    def _nest_closed(self, text: str) -> bool:
        """Whether the open object or array closed with `text` (its scan resumed where it stopped, over what it had
        not scanned: the tail that may begin a `<|"|>`, and the new text; linear in the text)."""
        nest = self._nest
        s = nest[0] + text
        _, depth, in_str, kind = nest
        opener = "{" if kind == "open_obj" else "["
        pattern = _NEST_OBJ if kind == "open_obj" else _NEST_ARR
        i = 0
        while True:
            if in_str:
                j = s.find(STRING_DELIM, i)
                if j < 0:
                    nest[0], nest[1], nest[2] = s[max(i, len(s) - _DELIM_LEN + 1):], depth, True
                    return False
                i, in_str = j + _DELIM_LEN, False
                continue
            m = pattern.search(s, i)
            if m is None:
                nest[0], nest[1], nest[2] = s[max(i, len(s) - _DELIM_LEN + 1):], depth, False
                return False
            tok = m.group()
            i = m.end()
            if tok == STRING_DELIM:
                in_str = True
            elif tok == opener:
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return True

    # -- Qwen ----------------------------------------------------------------------------------------------------

    def _qwen_members(self, start: int, text: str) -> list[tuple]:
        """The parameters of `text`, from `start` on, as qwen3.py's converter takes them (partial): the complete ones,
        then the open one (`open_parameter`), unless its name is whitespace alone (vLLM's parser names an open one by
        its last whitespace character, a complete one ""). While the text holds none, each scan resumes where the last
        one left nothing that may change (`_quiet`)."""
        r = 0
        if self._resume is not None and self._resume[0] == start:
            _, r, hold = self._resume
            if hold >= 0 and _value_end(text, hold) < 0:  # still in the value of a parameter with no name
                self._resume = (start, r, _end_from(text, hold))
                return []
        at: list[int] = []
        pairs = qwen_parameters(text, at, r)
        ends = [*at[1:len(pairs)], at[-1]] if pairs else []
        out = [(name, _trim_wrapping_newlines(value), "param", start + a, 0, start + e)
               for (name, value), a, e in zip(pairs, at, ends, strict=False)]
        cut = open_parameter(text, at[-1])
        if cut is not None and cut[1].strip():  # a name of whitespace alone is "" once the parameter closes: held
            out.append((cut[1], _open_value(text[cut[2]:]), "open_str", start + cut[0], start + cut[2], self._len))
        self._resume = None if out else self._quiet(start, text, r)
        return out

    @staticmethod
    def _quiet(start: int, text: str, r: int) -> tuple[int, int, int]:
        """Where the next scan of the window at `start` resumes, when its text from `r` on holds no parameter. It
        resumes at the first `<` that opens one or may still (`_open_from`). An opener there with a `>` names none
        ("" or whitespace). Nothing changes until its value ends, so the search for that end resumes too
        (`_end_from`)."""
        s = _open_from(text, r)
        head = _OPEN.match(text, s)
        gt = -1 if head is None else text.find(">", head.end())
        return start, s, -1 if gt < 0 else _end_from(text, gt + 1)

    def _qwen_extender(self, value_at: int) -> Any:
        """qwen3.py's `_param_value_extender` (newlines trimmed) for the open parameter whose value starts at
        `value_at`: it sends the value's new text, without its first newline, until its end comes (None). It holds back
        a trailing newline and a tail that may still grow into the end (ours: qwen3.py's lexer holds such a tail, but
        a cut output flushes it into the value, `_open_value`)."""
        raw_value = self.raw[value_at - self._base:]  # from `_base`, as `_advance` just read it
        started = bool(raw_value)
        body = raw_value.removeprefix("\n")
        # received, not sent: a trailing newline and a tail, in parts (a whitespace run kept apart, not copied); and
        # whether more whitespace keeps all of it held (a tail that may still end the value, `<` then whitespace)
        pending = [body[len(_open_value(raw_value)):]]
        window = _value_end_window(pending[0])
        spaced = bool(window) and _may_end_value(window + " ")

        def extend(delta: str) -> str | None:
            nonlocal started, spaced
            if spaced and started and delta.isspace():
                pending.append(delta)
                return ""
            text = "".join(pending) + delta
            if _VALUE_END.search(text):
                return None
            if not started and text:
                started = True
                text = text.removeprefix("\n")
            window = _value_end_window(text)
            body = text[: len(text) - len(window)]
            send = body.removesuffix("\n")
            pending[:] = [text[len(send):]]
            spaced = bool(window) and _may_end_value(window + " ")
            return json.dumps(send, ensure_ascii=False)[1:-1]

        return extend

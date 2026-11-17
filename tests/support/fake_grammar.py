"""A deterministic grammar for tests on small vocabularies (the fake runner's), behind the same backend interface as
llguidance (`grammar.engine`).

The grammar text is `fake:<seed>:<lo>:<hi>`. A matcher's state is the number of tokens it consumed (n) and their rolling
hash; the tokens it allows next are a deterministic two thirds of the vocabulary (a hash of the state, the seed and the
token), never an EOS id, plus every EOS id once n >= lo (accepting), and only the EOS ids once n >= hi (stopped), after
an EOS it consumed, or in its error state, as llguidance's masks are. A rejected token puts it in the error state, and
so does any token given to a stopped matcher, an EOS included: llguidance takes an EOS before the
grammar stops, or in the very call whose tokens stopped it, never in a later call. (llguidance records that last EOS, so
a rollback counts it; the fake does not, which nothing relies on: the scheduler gives a grammar's EOS in a call of its
own, `advance_matcher`.) So a JSON document's EOS, allowed only once the document is complete, can be consumed together
with its last token but not after it. `fill` follows llguidance's draft rule: a mask at the row's index, then one after
each draft the grammar accepts (an EOS included), stopping at the first one it rejects or once the grammar has stopped,
and the matcher rolled back.

**Sparse masks** (for runner tests of the grammar-sparse head and the draft universe on tiny vocabularies):
`fake:<seed>:<lo>:<hi>:<one_in>:<universe>` (`spec(..., one_in, universe)`; both 0 is the grammar above). With
`one_in` > 0 a non-EOS token is allowed iff the same hash is 0 modulo `one_in` (about one token in `one_in`: a JSON
schema's grammar allows a handful of a large vocabulary's tokens at a position, on average), else by the two-thirds
rule; with `universe` > 0 only ids below it are (with the EOS ids, the grammar's universe: no mask allows a token
outside it). So that such a mask is never empty before `lo` (and a state that has not stopped always allows a non-EOS
token, as llguidance's do), a state whose rule allows no non-EOS token allows one fallback token instead: its rolling
hash modulo `universe` (or the vocabulary size), stepped up (modulo the same) past the EOS ids. The two-thirds grammar
keeps its rule bit for bit, without the fallback (a state there allows none of the n non-EOS tokens of a vocabulary
with probability 3^-n).

**Forced runs** (for tests of the forced prefix at admission and the EOS-free finish on tiny vocabularies):
`fake:<seed>:<lo>:<hi>:<one_in>:<universe>:<lead>:<tail>` (`spec(..., lead=, tail=)`). The first `lead` positions
(n < lead) and the `tail` before the end (hi − tail ≤ n < hi) are forced, as compact JSON's opening `{"` and first key
and its closing brace are: the mask allows one token, never an EOS (an accepting state's EOS included), a
deterministic id from the state's hash (stepped past the EOS ids, within `universe`). `compute_ff_tokens` returns the
run of forced positions from the matcher's state, as llguidance's fast-forward tokens (as on a compact JSON grammar,
where masks of popcount 1 and fast-forward tokens coincide before the closing brace), so a singleton that the sparse
rule's fallback makes is no fast-forward token: that one the scheduler may not force (`FakeGrammar.only`: the mask
allows one token).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.runner.fake import HASH_BASE, MASK64, splitmix64_np

_U64 = np.uint64
_K = _U64(0x9E3779B97F4A7C15)


def spec(seed: int, lo: int, hi: int, one_in: int = 0, universe: int = 0, lead: int = 0, tail: int = 0) -> str:
    """The grammar text (module docstring): EOS allowed from `lo` tokens on and forced at `hi`; `one_in` and
    `universe` make sparse masks, `lead` and `tail` forced runs."""
    if lead or tail:
        return f"fake:{seed}:{lo}:{hi}:{one_in}:{universe}:{lead}:{tail}"
    if one_in or universe:
        return f"fake:{seed}:{lo}:{hi}:{one_in}:{universe}"
    return f"fake:{seed}:{lo}:{hi}"


@dataclass(frozen=True)
class FakeCompiled:
    seed: int
    lo: int
    hi: int
    # Sparse masks (module docstring): a non-EOS token is allowed iff its hash % one_in == 0 (0: the two-thirds rule),
    # and only below `universe` (0: anywhere); either one set adds the fallback token
    one_in: int = 0
    universe: int = 0
    # Forced runs (module docstring): the first `lead` positions and the `tail` before `hi` allow one token
    lead: int = 0
    tail: int = 0

    @property
    def sparse(self) -> bool:
        return bool(self.one_in or self.universe)

    def forced_at(self, n: int) -> bool:
        """Whether the state after n tokens (not stopped) is a forced position."""
        return n < self.lead or self.hi - self.tail <= n < self.hi


class FakeMatcher:
    __slots__ = ("_g", "c", "err", "h", "hist", "n", "term")

    def __init__(self, g: FakeGrammar, c: FakeCompiled):
        self._g, self.c = g, c
        self.n, self.h, self.term, self.err = 0, 0, False, ""
        self.hist: list[tuple[int, int, bool]] = []

    def deep_copy(self) -> FakeMatcher:
        m = FakeMatcher(self._g, self.c)
        m.n, m.h, m.term, m.err, m.hist = self.n, self.h, self.term, self.err, list(self.hist)
        return m

    def is_error(self) -> bool:
        return bool(self.err)

    def get_error(self) -> str:
        return self.err

    def is_stopped(self) -> bool:
        return bool(self.err) or self.term or self.n >= self.c.hi

    def allowed(self) -> np.ndarray:
        """bool [vocab]: the tokens this state allows."""
        g, c = self._g, self.c
        if not self.is_stopped() and c.forced_at(self.n):
            a = np.zeros(g.vocab_size, bool)
            a[self.forced_token()] = True
            return a
        if self.is_stopped():
            a = np.zeros(g.vocab_size, bool)
        else:
            with np.errstate(over="ignore"):
                x = splitmix64_np(_U64((self.h ^ c.seed) & MASK64) + g.ids * _K)
            a = x % _U64(c.one_in) == 0 if c.one_in else x % _U64(3) != 0
            if c.sparse:
                self._sparse(a)
        a[g.eos_arr] = self.is_stopped() or self.n >= c.lo
        return a

    def _sparse(self, a: np.ndarray) -> None:
        """Sparse masks (module docstring), in place: only ids below the universe, and the fallback token when the rule
        allows no non-EOS token."""
        g, c = self._g, self.c
        if c.universe:
            a[c.universe:] = False
        a[g.eos_arr] = False
        if not a.any():
            span = c.universe or g.vocab_size
            f = self.h % span
            while f in g.eos:  # compile() made sure the span holds a token that is not an EOS
                f = (f + 1) % span
            a[f] = True

    def forced_token(self) -> int:
        """The token a forced position allows (module docstring, forced runs): from the state's hash, within the
        universe, stepped past the EOS ids."""
        g, c = self._g, self.c
        span = c.universe or g.vocab_size
        f = int(splitmix64_np(np.asarray([(self.h ^ (c.seed * 3 + 1)) & MASK64], _U64))[0] % _U64(span))
        while f in g.eos:  # compile() made sure the span holds a token that is not an EOS
            f = (f + 1) % span
        return f

    def compute_ff_tokens(self) -> list[int]:
        """llguidance's fast-forward tokens: the run of forced positions from this state (module docstring)."""
        out: list[int] = []
        m = self.deep_copy()
        while not m.is_stopped() and m.c.forced_at(m.n):
            t = m.forced_token()
            m.consume_tokens([t])
            out.append(t)
        return out

    def words(self) -> np.ndarray:
        """The mask as int32 words (bit t % 32 of word t // 32)."""
        g = self._g
        bits = np.zeros(g.words * 32, bool)
        bits[:g.vocab_size] = self.allowed()
        return np.packbits(bits, bitorder="little").view(np.int32)

    def consume_tokens(self, tokens: list[int]) -> bool:
        if tokens and self.is_stopped() and not self.err:  # llguidance's: nothing after the end, an EOS included
            self.err = f"token {tokens[0]} after the end ({self.n} tokens)"
        for t in tokens:
            if self.err:
                return False
            if not (0 <= t < self._g.vocab_size) or not self.allowed()[t]:
                self.err = f"token {t} rejected after {self.n} tokens"
                return False
            if self.is_stopped():  # an EOS (all a stopped state allows) in the call that stopped it: not recorded
                continue
            self.hist.append((self.n, self.h, self.term))
            self.term = self.term or t in self._g.eos
            self.n += 1
            self.h = (self.h * HASH_BASE + t + 1) & MASK64
        return True

    def try_consume_tokens(self, tokens: list[int]) -> int:
        """llguidance's: consume the tokens up to the first one the grammar rejects, or until it has stopped (no error
        state); their count."""
        n = 0
        for t in tokens:
            if self.is_stopped() or not (0 <= t < self._g.vocab_size) or not self.allowed()[t]:
                break
            self.consume_tokens([t])
            n += 1
        return n

    def validate_tokens(self, tokens: list[int]) -> int:
        """llguidance's: how many of the tokens the grammar would take in a row, up to and with an EOS (none once it
        has stopped), without advancing."""
        if self.is_stopped():
            return 0
        n = taken = 0
        for t in tokens:
            if not (0 <= t < self._g.vocab_size) or not self.allowed()[t]:
                break
            taken += 1
            if t in self._g.eos:
                break
            self.consume_tokens([t])
            n += 1
        self.rollback(n)
        return taken

    def rollback(self, k: int) -> None:
        for _ in range(k):
            self.n, self.h, self.term = self.hist.pop()


class FakeGrammar:
    """The backend (`grammar.engine.Backend`)."""

    name = "fake"

    def __init__(self, vocab_size: int, eos: Sequence[int]):
        self.vocab_size = vocab_size
        self.words = (vocab_size + 31) // 32
        self.eos = tuple(eos)
        self.eos_arr = np.asarray(self.eos, np.int64)
        self.ids = np.arange(vocab_size, dtype=_U64)

    def compile(self, text: str) -> FakeCompiled:
        parts = text.split(":")
        try:
            if parts[0] != "fake" or len(parts) not in (4, 6, 8):
                raise ValueError
            seed, lo, hi = (int(x) for x in parts[1:4])
            one_in, universe = (int(x) for x in parts[4:6]) if len(parts) >= 6 else (0, 0)
            lead, tail = (int(x) for x in parts[6:8]) if len(parts) == 8 else (0, 0)
        except ValueError:
            raise GrammarError(f"not a fake grammar: {text[:40]!r}") from None
        if not 0 <= lo <= hi:
            raise GrammarError(f"fake grammar: need 0 <= lo <= hi, got {lo}, {hi}")
        if one_in < 0 or not 0 <= universe <= self.vocab_size:
            raise GrammarError(f"fake grammar: need one_in >= 0 and 0 <= universe <= {self.vocab_size}, got {one_in}, "
                               f"{universe}")
        if lead < 0 or not 0 <= tail <= hi:
            raise GrammarError(f"fake grammar: need lead >= 0 and 0 <= tail <= hi, got {lead}, {tail}")
        span = universe or self.vocab_size
        if (one_in or universe or lead or tail) and len({t for t in self.eos if t < span}) >= span:
            raise GrammarError(f"fake grammar: no token below {span} but EOS ids for a fallback or forced token")
        return FakeCompiled(seed, lo, hi, one_in, universe, lead, tail)

    def instance(self, compiled: FakeCompiled) -> FakeMatcher:
        return FakeMatcher(self, compiled)

    def only(self, m: FakeMatcher, token: int, own: bool = False) -> bool:  # `own`: no state shared either way
        a = m.allowed()
        return 0 <= token < self.vocab_size and bool(a[token]) and int(a.sum()) == 1

    def fill(self, singles: list[tuple[FakeMatcher, int]], drafts: list[tuple[FakeMatcher, int, list[int]]],
             buf: np.ndarray) -> None:
        for m, i in singles:
            buf[i] = m.words()
        for m, i, ds in drafts:
            if not ds:
                raise ValueError("Draft tokens must not be empty")  # as llguidance's
            buf[i] = m.words()
            k = 0
            for t in ds:
                if m.is_stopped() or not (0 <= t < self.vocab_size) or not m.allowed()[t]:
                    break
                m.consume_tokens([t])
                k += 1
                buf[i + k] = m.words()
            m.rollback(k)

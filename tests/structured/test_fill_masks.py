"""Every mask `Scheduler.fill_masks` returns is defined, and the fake grammar's sparse masks.

The grammar-sparse head samples a split step over the union of its mask rows [0, n_masks), so no row may hold what an
earlier step left in the buffer, and the rows nobody reads should allow as little as they can. Each check fills the
buffer with a sentinel before every call and compares each row returned with an oracle computed apart from the
scheduler's code, from a deep copy of the row's matcher advanced along its drafts (`expected`):
- `mask`: the grammar's mask at the matcher's state, or after the drafts before it while each is one the grammar
  takes (and not an EOS);
- `unread`: after a draft the grammar rejects, an EOS (without ignore_eos) or the grammar's end, which verification
  never reads: the lowest EOS id alone (written by the scheduler: llguidance writes none of them on a JSON grammar,
  so the buffer would hold a stale row);
- `past_eos`: after a drafted EOS under ignore_eos, where the grammar terminates: all ones (a finished grammar applies
  no mask);
- `dead`: each mask of a row whose request finished, failed or left since the plan: the lowest EOS id alone (not all
  ones, which would make the sparse head's union the whole vocabulary);
- `terminated`: each mask of a request whose grammar terminated (an EOS under ignore_eos): all ones.
`Inflight.invalid` is unchanged: k minus the drafts the grammar takes in a row, up to and with an EOS (all once it has
stopped). Hand-set slots (`Rig`) cover every case on the fake grammar, its sparse masks and llguidance over Gemma 4's
tokenizer (guidance, compact JSON, on the catalog schema; drafts from a valid document, perturbed), with the case of
each mask written out by hand; runs through the engine loop and the validating fake runner cover them as they happen
(a finish between a plan and its masks, ignore_eos past a grammar's end, drafts the grammar rejects), and so do
`test_structured`'s random workloads, with sparse grammars among them.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.fake import MASK64, masked_pick, splitmix64_np, unpack_mask
from llm_mini_infer.sched.scheduler import Inflight, Req, Scheduler, eos_only

from ..support.catalog_schema import DOC
from ..support.catalog_schema import PATH as CATALOG_SCHEMA
from ..support.fake_grammar import FakeGrammar, spec
from ..support.fuzzing import event, fuzz_settings
from ..support.harness import EOS, Spec
from ..support.paths import MODELS_DIR
from ..support.structured_cases import run as run_structured
from ..support.structured_cases import run_structured_workload, structured_workloads

SENTINEL = 0x5A5A5A5A  # what the buffer holds before each fill: no mask, no singleton, not all ones
K = 5  # Gemma 4's assistant's default k
GEMMA = MODELS_DIR / "gemma-4-31b-it"
SYNTHETIC = CATALOG_SCHEMA  # the synthetic schema (tests/support/catalog_schema.py), and DOC a compact document of it


# -- the oracle --------------------------------------------------------------------------------------------------------


def bits(words: int, ids: list[int]) -> np.ndarray:
    """int32 mask words that allow exactly `ids`."""
    b = np.zeros(words * 32, bool)
    b[ids] = True
    return np.packbits(b, bitorder="little").view(np.int32)


def mask_at(backend: Any, m: Any, words: int) -> np.ndarray:
    """The backend's mask at matcher `m`'s state (a fill of one row)."""
    out = np.full((1, words), SENTINEL, np.int32)
    backend.fill([(m, 0)], [], out)
    return out[0]


def expected(sched: Scheduler, inf: Inflight, words: int) -> tuple[list[tuple[str, np.ndarray]], np.ndarray | None]:
    """The masks `fill_masks` must return for a plan, in order, each with its case (module docstring), and
    `Inflight.invalid`: from the scheduler's state between the plan and its masks, without its code."""
    k, backend, eos = sched.k, sched.grammars.backend, sched.grammars.eos
    vocab = backend.vocab_size
    only, ones = bits(words, [min(eos)]), np.full(words, -1, np.int32)
    rows: list[tuple[str, np.ndarray]] = []
    invalid = np.zeros(len(inf.slots), np.int64) if k else None
    for r in inf.masked.tolist():
        slot = int(inf.slots[r])
        n = 1 + k if k and inf.device[r] else 1
        req = sched.slot_req[slot]
        if req is None or sched.aid[slot] != inf.aids[r]:
            rows += [("dead", only)] * n
            continue
        if req.matcher is None:
            rows += [("terminated", ones)] * n
            continue
        m = req.matcher.deep_copy()
        mask = mask_at(backend, m, words)
        rows.append(("mask", mask))
        if n == 1:
            continue
        d = sched.drafts[slot].tolist()
        invalid[r] = k - (0 if m.is_stopped() else m.deep_copy().validate_tokens(d))
        read = True  # verification may read the next mask: every draft so far is one the grammar took, not an EOS
        for j, t in enumerate(d):
            if t in eos and req.ignore_eos:
                rows += [("past_eos", ones)] * (k - j)
                break
            read = read and t not in eos and not m.is_stopped() and 0 <= t < vocab and bool(unpack_mask(mask, vocab)[t])
            if read:
                assert m.consume_tokens([t])
                mask = mask_at(backend, m, words)
                rows.append(("mask", mask))
            else:
                rows.append(("unread", only))
    return rows, invalid


def check_fill(sched: Scheduler, inf: Inflight, buf: np.ndarray, fill: Any = None,
               seen: Counter | None = None, universe: int = 0) -> tuple[int, list[str]]:
    """`fill_masks` (or `fill`, the method itself) into `buf`, prefilled with the sentinel; every mask it returns is the
    oracle's, and none past them is written. `universe`: no mask but an all-ones one allows a token at or above it
    other than an EOS id (sparse fake grammars). Returns the masks' count and cases."""
    rows, invalid = expected(sched, inf, buf.shape[1])
    buf[:] = SENTINEL
    n = (fill or Scheduler.fill_masks)(sched, inf, buf)
    assert n == len(rows)
    for i, (case, row) in enumerate(rows):
        assert np.array_equal(buf[i], row), f"mask {i} of {n} is not its {case} mask"
    assert (buf[n:] == SENTINEL).all(), "a row past the plan's masks was written"
    if invalid is None:
        assert inf.invalid is None
    else:
        assert np.array_equal(inf.invalid, invalid), (inf.invalid, invalid)
    cases = [case for case, _ in rows]
    if universe and n and "terminated" not in cases and "past_eos" not in cases:  # the sparse head's union
        union = np.bitwise_or.reduce(buf[:n], axis=0)
        outside = np.flatnonzero(unpack_mask(union, sched.grammars.backend.vocab_size)[universe:]) + universe
        assert set(outside.tolist()) <= sched.grammars.eos, outside
    if seen is not None:
        seen.update(cases)
    return n, cases


@contextmanager
def checked_fills(seen: Counter, universe: int = 0) -> Iterator[None]:
    """Every `fill_masks` call meanwhile goes through `check_fill`."""
    orig = Scheduler.fill_masks

    def fill(self: Scheduler, inf: Inflight, buf: np.ndarray) -> int:
        return check_fill(self, inf, buf, orig, seen, universe)[0]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Scheduler, "fill_masks", fill)
        yield


class Rig:
    """`fill_masks` over hand-set slots: what it reads of the scheduler (each slot's request, admission and drafts)
    as `process` leaves it between a plan and its masks, and the plan's rows."""

    def __init__(self, grammars: Grammars, k: int = K, slots: int = 48):
        self.sched = Scheduler(max_num_seqs=slots, max_num_batched_tokens=1024, max_model_len=1024, page_size=16,
                               num_pages=64, num_state_slots=slots + 1, eos_token_ids=tuple(sorted(grammars.eos)),
                               mtp_k=k, grammars=grammars)
        self.k = k
        self.rows: list[tuple[int, int, bool, bool]] = []  # (slot, aid, device, masked)
        self.cases: list[str] = []  # each row's masks, by hand: m(ask), u(nread), p(ast_eos), d(ead), t(erminated)

    def row(self, cases: str, matcher: Any = None, drafts: Sequence[int] = (), *, device: bool = True,
            ignore_eos: bool = False, left: str = "", masked: bool = True) -> None:
        """A row of the plan: a verify row (`device`, under MTP) or a final prefill chunk; `left`: its request left
        since the plan, its slot freed ("freed") or holding another admission ("other"); `matcher` None: its grammar
        terminated (ignore_eos)."""
        s = self.sched
        slot = len(self.rows) + 1
        aid = 1000 + slot
        msg = AddRequest(rid=slot, frontend=0, prompt=np.zeros(4, np.int32), max_tokens=64, temperature=0.0,
                         top_p=1.0, top_k=0, min_p=0.0, seed=slot, stop_token_ids=[], ignore_eos=ignore_eos)
        req = Req(msg, slot, slot, 1024)
        req.matcher = matcher
        s.slot_req[slot], s.aid[slot] = req, aid
        if drafts:
            assert len(drafts) == self.k
            s.drafts[slot] = drafts
        if left == "freed":
            s.slot_req[slot], s.aid[slot] = None, 0
        elif left == "other":
            s.aid[slot] = aid + 1
        self.rows.append((slot, aid, device, masked))
        self.cases.append(cases)

    def fill(self) -> list[str]:
        """`fill_masks` through `check_fill`; each row's cases are the hand-written ones."""
        slots, aids, device, masked = (np.asarray(x) for x in zip(*self.rows, strict=True))
        inf = Inflight(step_id=1, slots=slots.astype(np.int64), aids=aids.astype(np.int64),
                       sample=np.ones(len(slots), bool), device=device.astype(bool), n_tokens=0,
                       masked=np.flatnonzero(masked))
        backend = self.sched.grammars.backend
        buf = np.zeros((len(slots) * (1 + self.k) + 3, (backend.vocab_size + 31) // 32), np.int32)
        _, cases = check_fill(self.sched, inf, buf)
        letters = "".join(c[0] for c in cases)
        want = "".join(c for c, mk in zip(self.cases, masked, strict=True) if mk)
        assert letters == want, (letters, want)
        return cases


# -- the fake grammar's masks ------------------------------------------------------------------------------------------


def state_hash(c: Any, h: int, vocab: int) -> np.ndarray:
    """uint64 [vocab]: the fake grammar's hash of a state (its rolling hash h), the seed and each token."""
    with np.errstate(over="ignore"):
        return splitmix64_np(np.uint64((h ^ c.seed) & MASK64) +
                             np.arange(vocab, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))


def two_thirds_allowed(c: Any, vocab: int, eos: tuple[int, ...], n: int, h: int, stopped: bool) -> np.ndarray:
    """The fake grammar's default (dense) rule, recomputed: two thirds of the vocabulary by the hash, the EOS ids from
    `lo` on, only those once stopped."""
    a = np.zeros(vocab, bool) if stopped else state_hash(c, h, vocab) % np.uint64(3) != 0
    a[list(eos)] = stopped or n >= c.lo
    return a


def test_default_fake_masks_are_the_two_thirds_rule() -> None:
    """The four-field grammar (and six fields with both 0) gives the two-thirds rule's masks bit for bit, without the
    sparse mode's fallback: on a two-token vocabulary the rule leaves states with no non-EOS token."""
    assert spec(5, 2, 9) == spec(5, 2, 9, 0, 0) == "fake:5:2:9"
    rng = np.random.default_rng(7)
    empty = 0
    for vocab, eos in ((1000, (999, 991)), (64, (63,)), (2, (1,))):
        g = FakeGrammar(vocab, eos)
        assert g.compile("fake:5:2:9:0:0") == g.compile("fake:5:2:9")
        for seed, lo, hi in ((5, 2, 9), (1 << 40, 0, 30), (123456789, 12, 12), (77, 3, 25)):
            m = g.instance(g.compile(spec(seed, lo, hi)))
            while True:
                a = m.allowed()
                assert np.array_equal(a, two_thirds_allowed(m.c, vocab, eos, m.n, m.h, m.is_stopped()))
                assert np.array_equal(unpack_mask(m.words(), vocab), a)
                if m.is_stopped():
                    break
                ok = a.copy()
                ok[list(eos)] = False
                if not ok.any():  # the two-thirds rule allowed no non-EOS token (no fallback here)
                    empty += 1
                    break
                t = masked_pick(int(rng.integers(vocab)), ok)
                assert m.consume_tokens([t])
    assert empty > 0


def test_sparse_fake_masks_density_universe_and_fallback() -> None:
    eos = (4095,)
    g = FakeGrammar(4096, eos)
    assert spec(3, 4, 40, one_in=64) == "fake:3:4:40:64:0" and spec(3, 4, 40, universe=512) == "fake:3:4:40:0:512"
    c = g.compile(spec(3, 4, 40, one_in=64))
    assert (c.one_in, c.universe) == (64, 0) and c.sparse
    # about one non-EOS token in 64, those whose hash (the default rule's) is 0 modulo 64
    m, counts = g.instance(c), []
    while not m.is_stopped():
        a = m.allowed()
        nz = np.flatnonzero(a[:4095])
        rule = np.flatnonzero(state_hash(c, m.h, 4096)[:4095] % np.uint64(64) == 0)
        assert nz.tolist() == rule.tolist() and len(nz) and a[4095] == (m.n >= 4)
        counts.append(len(nz))
        assert m.consume_tokens([int(nz[m.n % len(nz)])])
    assert len(counts) == 40 and 50 < np.mean(counts) < 80, np.mean(counts)
    # ... below the universe only: every state's mask, so their union, inside it (with the EOS)
    c = g.compile(spec(3, 4, 40, one_in=8, universe=512))
    m, union, counts = g.instance(c), np.zeros(4096, bool), []
    while not m.is_stopped():
        a = m.allowed()
        union |= a
        nz = np.flatnonzero(a[:4095])
        rule = np.flatnonzero(state_hash(c, m.h, 4096)[:512] % np.uint64(8) == 0)
        assert nz.tolist() == rule.tolist() and len(nz)
        counts.append(len(nz))
        assert m.consume_tokens([int(nz[(m.n * 7) % len(nz)])])
    assert not union[512:4095].any() and union[4095] and 50 < np.mean(counts) < 80, np.mean(counts)
    # the two-thirds rule within a universe
    c = g.compile(spec(3, 4, 40, universe=30))
    m = g.instance(c)
    a = m.allowed()
    assert np.flatnonzero(a).tolist() == np.flatnonzero(state_hash(c, 0, 4096)[:30] % np.uint64(3) != 0).tolist()
    # the fallback: a rule that allows nothing (one_in 2^62) allows its state's hash modulo the universe (or the
    # vocabulary), stepped up past the EOS ids; never an empty mask before lo, only the EOS ids once stopped
    g2 = FakeGrammar(64, (0, 1))
    for universe in (4, 0):
        c = g2.compile(spec(9, 2, 6, one_in=1 << 62, universe=universe))
        m, walked = g2.instance(c), []
        while not m.is_stopped():
            f = m.h % (universe or 64)
            while f in (0, 1):
                f = (f + 1) % (universe or 64)
            a = m.allowed()
            assert (set(np.flatnonzero(a).tolist()) - {0, 1}) == {f} and a[[0, 1]].all() == (m.n >= 2)
            walked.append(f)
            assert m.consume_tokens([f])
        assert walked[0] == 2 and len(walked) == 6 and np.flatnonzero(m.allowed()).tolist() == [0, 1]
        if universe:
            assert max(walked) < universe
    assert masked_pick(40, g2.instance(g2.compile(spec(9, 2, 6, one_in=1 << 62))).allowed()) == 2  # hash 0: 2
    for bad in ("fake:1:2:3:4", "fake:1:2:3:4:5:6", "fake:1:2:3:-1:0", "fake:1:2:3:0:-5", "fake:1:2:3:0:65",
                "fake:1:2:3:x:0", "fake:1:5:3:8:0"):
        with pytest.raises(GrammarError):
            g2.compile(bad)
    with pytest.raises(GrammarError, match="EOS"):
        g2.compile(spec(1, 2, 3, universe=2))  # ids 0 and 1 are both EOS ids: no fallback token


def test_eos_only_sets_the_sign_bit_through_a_uint32_view() -> None:
    for eos, words in ((1, 8192), (31, 2), (991, 32), (248044, 7760), (63, 2)):
        row = eos_only(words, eos)
        assert np.array_equal(row, bits(words, [eos])) and not row.flags.writeable
        assert eos_only(words, eos) is row  # built once per word count
    assert eos_only(32, 991)[30] == np.int32(-(1 << 31))


# -- hand-set slots: the fake grammar ----------------------------------------------------------------------------------


def walk(m: Any, n: int, eos: tuple[int, ...], salt: int = 0) -> list[int]:
    """Advance `m` by n of its grammar's non-EOS tokens (a hash-chosen one each); returns them."""
    out = []
    for i in range(n):
        a = m.allowed()
        a[list(eos)] = False
        t = masked_pick((salt * 131 + i * 37 + 11) % len(a), a)
        assert t >= 0 and m.consume_tokens([t])
        out.append(t)
    return out


def rejected(m: Any, eos: tuple[int, ...]) -> int:
    """A token `m`'s state rejects, other than an EOS id."""
    a = m.allowed()
    a[list(eos)] = True
    return int(np.flatnonzero(~a)[len(np.flatnonzero(~a)) // 2])


@pytest.mark.parametrize("sparse", [False, True])
def test_every_case_on_the_fake_grammar(sparse) -> None:
    """Every case of the spec on hand-set slots of one plan (k = 5): EOS ids 999 and 991, so a mask that allows the
    lowest alone sets bit 31 of its word (int32's sign bit); EOS allowed from 6 tokens on, forced at 12."""
    eos = (999, 991)
    g = Grammars(FakeGrammar(1000, eos))
    c = g.compile(spec(7, 6, 12, *((8, 600) if sparse else (0, 0))))

    def at(n: int) -> Any:  # a matcher after n of its grammar's tokens
        m = g.matcher(c)
        walk(m, n, eos)
        return m

    def ahead(m: Any, n: int, salt: int = 1) -> list[int]:  # the next n tokens it takes (m unmoved)
        return walk(m.deep_copy(), n, eos, salt)

    def bad_at(n: int, d: list[int], j: int) -> list[int]:  # d with draft j replaced by one the grammar rejects there
        m = at(n)
        for t in d[:j]:
            assert m.consume_tokens([t])
        return [*d[:j], rejected(m, eos), *d[j + 1:]]

    rig = Rig(g)
    rig.row("mmmmmm", at(0), ahead(at(0), K))  # every draft taken
    for j in range(K):  # a draft the grammar rejects, at each position: the masks after it unread
        rig.row("m" * (j + 1) + "u" * (K - j), at(2), bad_at(2, ahead(at(2), K), j))
    p = ahead(at(6), K)
    rig.row("mmuuuu", at(6), [p[0], 999, *p[2:]])  # an EOS while it accepts: the row ends there
    p = ahead(at(0), K)
    rig.row("mmuuuu", at(0), [p[0], 991, *p[2:]])  # ... before it accepts: rejected
    p = ahead(at(6), K)
    rig.row("mmmppp", at(6), [p[0], p[1], 999, p[3], p[4]], ignore_eos=True)  # ignore_eos: all ones past the EOS
    p = ahead(at(0), K)
    rig.row("mmuupp", at(0), [p[0], bad_at(0, p, 1)[1], p[2], 991, p[4]], ignore_eos=True)
    rig.row("mppppp", at(6), [991, *ahead(at(6), K - 1)], ignore_eos=True)
    junk = ahead(at(3), K)
    rig.row("muuuuu", at(12), [999, *junk[1:]])  # a stopped grammar: its mask (EOS ids), then nothing read
    rig.row("muuuuu", at(12), [junk[0], 991, *junk[2:]])
    rig.row("mppppp", at(12), [999, *junk[1:]], ignore_eos=True)
    p = ahead(at(10), 2)
    rig.row("mmmuuu", at(10), [*p, 991, *junk[:2]])  # an EOS drafted right after the token that completes it
    rig.row("mmmppp", at(10), [*p, 999, *junk[:2]], ignore_eos=True)
    rig.row("mmmuuu", at(10), [*p, *junk[:3]])  # ... and drafts past the end
    rig.row("dddddd", at(4), ahead(at(4), K), left="freed")  # finished since the plan, its slot freed
    rig.row("dddddd", at(4), ahead(at(4), K), left="other")  # ... its slot holding another admission
    rig.row("d", at(4), device=False, left="freed")  # a final prefill chunk
    rig.row("tttttt", None, ahead(at(4), K), ignore_eos=True)  # its grammar terminated (ignore_eos)
    rig.row("t", None, device=False, ignore_eos=True)
    rig.row("m", at(5), device=False)  # a final prefill chunk under MTP: one mask
    rig.row("", at(5), ahead(at(5), K), masked=False)  # a row without a mask (an unconstrained request)
    rig.row("mmmmmm", at(1), ahead(at(1), K, salt=3))
    seen = Counter(rig.fill())
    assert set(seen) == {"mask", "unread", "past_eos", "dead", "terminated"}


@pytest.mark.parametrize("left", ["freed", "other", ""])
def test_rows_without_mtp(left) -> None:
    """k = 0: one mask a row, the dead row's the lowest EOS alone, a terminated grammar's all ones."""
    eos = (999, 991)
    g = Grammars(FakeGrammar(1000, eos))
    c = g.compile(spec(3, 2, 8))
    rig = Rig(g, k=0)
    m = g.matcher(c)
    walk(m, 3, eos)
    rig.row("d" if left else "m", m, device=True, left=left)
    rig.row("t", None, device=True, ignore_eos=True)
    rig.row("m", g.matcher(c), device=False)
    rig.fill()


# -- hand-set slots: llguidance over Gemma 4's tokenizer ---------------------------------------------------------------


@pytest.fixture(scope="module")
def gemma():
    if not (GEMMA / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import LLGuidance
    from llm_mini_infer.grammar.spec import json_schema_grammar

    g = Grammars(LLGuidance(GEMMA / "tokenizer.json", 262144, "<eos>"))
    assert g.eos == {1}
    compiled = g.compile(json_schema_grammar(json.loads(SYNTHETIC.read_text()), "guidance", True))
    ids = g.backend.tokenizer.tokenize_bytes(DOC)
    m = g.matcher(compiled)
    assert m.consume_tokens(ids) and m.is_stopped()
    return g, compiled, ids


def test_every_case_on_llguidance(gemma) -> None:
    """The same cases with llguidance (1.9.1) over Gemma 4's tokenizer, guidance with compact JSON on the catalog
    schema: drafts are a valid document's next tokens, perturbed. Its fill writes the masks the oracle reads, and the
    scheduler the rest."""
    g, compiled, ids = gemma
    words = (262144 + 31) // 32
    rng = np.random.default_rng(11)

    def at(n: int) -> Any:
        m = g.matcher(compiled)
        assert m.consume_tokens(ids[:n])
        return m

    def no(m: Any) -> int:  # a token the state rejects (not the EOS)
        allowed = unpack_mask(mask_at(g.backend, m, words), 262144)
        while True:
            t = int(rng.integers(2, 262144))
            if not allowed[t]:
                return t

    def bad(n: int, j: int) -> list[int]:  # the document's next k tokens after n, draft j one the grammar rejects
        d = ids[n:n + K]
        return [*d[:j], no(at(n + j)), *d[j + 1:]]

    end = len(ids)
    rig = Rig(g)
    for n in (0, 1, 2, 10, 33, end - K):
        rig.row("mmmmmm", at(n), ids[n:n + K])
    for j in range(K):
        rig.row("m" * (j + 1) + "u" * (K - j), at(20), bad(20, j))
    rig.row("mmuuuu", at(30), [ids[30], 1, *ids[32:35]])  # an EOS mid-document: rejected
    rig.row("mmmppp", at(30), [*ids[30:32], 1, *ids[33:35]], ignore_eos=True)
    rig.row("mmuupp", at(20), [ids[20], bad(20, 1)[1], ids[22], 1, ids[24]], ignore_eos=True)
    junk = ids[:K]
    rig.row("mmmuuu", at(end - 2), [*ids[end - 2:], 1, *junk[:2]])  # an EOS right after the completing token
    rig.row("mmmppp", at(end - 2), [*ids[end - 2:], 1, *junk[:2]], ignore_eos=True)
    rig.row("mmmuuu", at(end - 2), [*ids[end - 2:], *junk[:3]])  # drafts past the end
    rig.row("muuuuu", at(end), [1, *junk[:4]])  # the document complete: its mask (the EOS), nothing read after
    rig.row("muuuuu", at(end), junk)
    rig.row("mppppp", at(end), [1, *junk[:4]], ignore_eos=True)
    rig.row("dddddd", at(12), ids[12:12 + K], left="freed")
    rig.row("dddddd", at(12), ids[12:12 + K], left="other")
    rig.row("d", at(12), device=False, left="freed")
    rig.row("tttttt", None, ids[12:12 + K], ignore_eos=True)
    rig.row("m", at(0), device=False)
    rig.row("", at(3), ids[3:3 + K], masked=False)
    seen = Counter(rig.fill())
    assert set(seen) == {"mask", "unread", "past_eos", "dead", "terminated"}


def test_llguidance_writes_the_masks_along_drafts_and_no_others(gemma) -> None:
    """The rule the scheduler's EOS-only masks rely on (llguidance 1.9.1, probed, as 1.7.6): along drafts, fill writes
    the mask at the state and one after each of the first w = min(v, e) drafts, none after: v from `validate_tokens` (0
    once stopped), e the first EOS's index. (A grammar that takes an EOS while it may go on gets one more, after the
    EOS; a JSON document's grammar stops at its end. The scheduler writes over every mask past w either way.)"""
    g, compiled, ids = gemma
    words = (262144 + 31) // 32
    end = len(ids)
    cases = [(10, ids[10:15]), (20, [ids[20], ids[21], 12345, *ids[23:25]]), (20, [99999, *ids[21:25]]),
             (30, [ids[30], 1, *ids[32:35]]), (end - 2, [*ids[end - 2:], 1, 5, 6]), (end - 2, [*ids[end - 2:], 7, 5]),
             (end, [1, 5, 6]), (end, [5, 1, 6])]
    for n, d in cases:
        m = g.matcher(compiled)
        assert m.consume_tokens(ids[:n])
        buf = np.full((len(d) + 2, words), SENTINEL, np.int32)
        g.fill([], [(m, 1, d)], buf, len(d) + 1)
        v = 0 if m.is_stopped() else m.validate_tokens(d)
        e = next((j for j, t in enumerate(d) if t == 1), len(d))
        written = [i - 1 for i in range(len(buf)) if not (buf[i] == SENTINEL).all()]
        assert written == list(range(min(v, e) + 1)), (n, d, written, v, e)
        assert m.consume_tokens(ids[n:])  # rolled back
    lark = g.compile('%llguidance {}\nstart: "x" ("y")*')  # accepts after "x", and may go on
    (x,), (y,) = (g.backend.tokenizer.tokenize_bytes(s) for s in (b"x", b"y"))
    m = g.matcher(lark)
    assert m.consume_tokens([x]) and not m.is_stopped() and m.validate_tokens([y, 1, y]) == 2
    buf = np.full((5, words), SENTINEL, np.int32)
    g.fill([], [(m, 1, [y, 1, y])], buf, 4)
    written = [i - 1 for i in range(len(buf)) if not (buf[i] == SENTINEL).all()]
    assert written == [0, 1, 2]  # w = min(2, 1) = 1, and the mask after the EOS


# -- through the engine loop and the validating fake runner ------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("sparse", [False, True])
def test_every_mask_of_a_run_is_defined(k, sparse) -> None:
    """Each case as it happens: a request ending by its grammar while another decodes (its row planned before the EOS
    came back is dead when its masks are filled), ignore_eos past a grammar's end (that row is unconstrained), drafts
    the grammar rejects (the fake drafts unmasked: a third of its tokens are rejected, nearly all of a sparse
    grammar's), stop ids. With sparse grammars every step's union stays in their universe (300 tokens) but where a
    mask allows everything."""
    universe = 300 if sparse else 0

    def g(seed: int, lo: int, hi: int) -> str:
        return spec(seed, lo, hi, *((16, universe) if sparse else (0, 0)))

    seen: Counter = Counter()
    with checked_fills(seen, universe):
        run_structured([Spec(rid=0, prompt_len=12, max_tokens=60, seed=0xE4E0, grammar=g(7, 3, 8)),
                        Spec(rid=1, prompt_len=12, max_tokens=60, seed=0xE4E1, ignore_eos=True)],
                       mtp_k=k, accept_p=0.8, page_size=4, num_pages=200)
        for i in range(2):
            run_structured([Spec(rid=0, prompt_len=16 + i, max_tokens=40, seed=0xE05 + i, grammar=g(9 + i, 3, 8),
                                 ignore_eos=True)], mtp_k=k, accept_p=0.9, page_size=4, num_pages=200)
        specs = [Spec(rid=i, prompt_len=10 + 7 * i, max_tokens=60, seed=0x5EED_0000 + i, arrive=i,
                      grammar=g(100 + i, 5 + i, 12 + 3 * i) if i % 3 else "",
                      stop_ids=[EOS - 2] if i == 4 else []) for i in range(6)]
        run_structured(specs, mtp_k=k, accept_p=0.9, max_num_seqs=4, max_tokens=128, page_size=4, num_pages=600)
    assert seen["dead"] and seen["terminated"] and seen["mask"], seen
    if k:
        assert seen["unread"], seen


@st.composite
def workloads(draw):
    """`test_structured`'s random structured workloads, about a third of their grammars with sparse masks (a density
    of 1 to 2^-40, a universe of 16 or 400 tokens or none)."""
    w = draw(structured_workloads())
    for s in w["specs"]:
        if s.grammar and not draw(st.integers(0, 2)):
            seed, lo, hi = (int(x) for x in s.grammar.split(":")[1:])
            s.grammar = spec(seed, lo, hi, draw(st.sampled_from([0, 1, 4, 64, 1 << 40])),
                             draw(st.sampled_from([0, 16, 400])))
    return w


@fuzz_settings(100)
@given(workloads())
def test_random_workloads_define_every_mask(w) -> None:
    seen: Counter = Counter()
    with checked_fills(seen):
        run_structured_workload(w)
    for case in ("dead", "terminated", "unread", "past_eos"):
        if seen[case]:
            event(f"fill_masks: {case} masks")

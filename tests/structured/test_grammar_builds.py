"""The engine's grammar builds beside the step loop (`grammar.engine.Grammars.submit`, `settle`): a build's future is
done only once the engine thread settles it, one build runs per (text, prefix), the caches and counters fill as
`compile` fills them, and a build job's forced prefix shares no buffer or executor call with the step loop's masks."""

from __future__ import annotations

import json
import logging
import threading

import numpy as np
import pytest

from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.grammar.engine import FORCED_MAX, Grammars

from ..support.compile_jobs import manual
from ..support.fake_grammar import FakeGrammar, spec
from ..support.fixtures import QWEN
from ..support.harness import EOS
from ..support.sched_cases import VOCAB

FORCED = spec(7, 4, 12, lead=3)  # a grammar that forces its first three tokens


def fake() -> FakeGrammar:
    return FakeGrammar(VOCAB, (EOS,))


def test_without_threads_submit_is_compile_and_done_at_once() -> None:
    """No compile threads (the harness, the plan-diff, the fuzz suites): `submit` is `compile`, call for call: a done
    future, the compiled grammar cached and counted once, an error as a value (counted, cached), the forced prefix left
    to `forced_prefix`."""
    g = Grammars(fake())
    f = g.submit(FORCED, True)
    compiled, forced = f.result()
    assert f.done() and forced is None and compiled is g._cache[FORCED] and (g.compiles, g.compile_errors) == (1, 0)
    assert g.submit(FORCED, False).result()[0] is compiled and g.compiles == 1  # a hit
    bad = g.submit("fake:not-a-grammar", True)
    assert bad.done() and isinstance(bad.result(), GrammarError) and (g.compiles, g.compile_errors) == (2, 1)
    again = g.submit("fake:not-a-grammar", False).result()
    assert isinstance(again, GrammarError) and again is not bad.result() and g.compiles == 2
    assert not g._inflight and not g._ready and g.building == 0


def test_a_build_is_done_only_once_settled(caplog) -> None:
    """A miss: the build's future stays pending after its job ends (only the done-callback ran, queuing it and waking
    the loop) until `settle`, which fills both caches, counts the compile once and logs it. The prefix is the one the
    synchronous path computes."""
    g = Grammars(fake())
    jobs = manual(g)
    woke: list[int] = []
    g.wake = lambda: woke.append(1)
    f = g.submit(FORCED, True)
    assert not f.done() and len(jobs) == 1 and g.building == 1 and g.compiles == 0
    jobs.complete()
    assert woke == [1] and not f.done() and g.compiles == 0 and len(g._ready) == 1
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.grammar.engine"):
        assert g.settle() == 1
    compiled, forced = f.result()
    assert forced == Grammars(fake()).forced_prefix(FORCED) and len(forced) == 3
    assert g._cache[FORCED] is compiled and g._forced[FORCED] == forced
    assert (g.compiles, g.compile_errors, g.building) == (1, 0, 0) and g.compile_ns == g.last_compile_ns > 0
    assert [r.getMessage().startswith(f"compiled a grammar of {len(FORCED)} characters") for r in caplog.records] == [
        True]
    assert g.settle() == 0  # nothing ready: at once
    hit = g.submit(FORCED, True)
    assert hit.done() and hit.result() == (compiled, None) and len(jobs) == 0  # forced: `forced_prefix`, a hit


def test_requests_of_one_grammar_share_its_build() -> None:
    """One build per (text, prefix) in flight: a second request for it gets the same future; another `prefix` is a
    build of its own. A cached grammar whose prefix is missing gets a job for the prefix alone, which compiles
    nothing."""
    g = Grammars(fake())
    jobs = manual(g)
    a, b, c = g.submit(FORCED, True), g.submit(FORCED, True), g.submit(FORCED, False)
    assert a is b and c is not a and len(jobs) == 2 and g.building == 2
    jobs.complete(1)
    assert g.settle() == 1 and c.done() and c.result()[1] is None and not a.done() and g.compiles == 1
    jobs.complete()
    g.settle()
    assert a.result()[1] == b.result()[1] and len(a.result()[1]) == 3 and g.compiles == 2  # two compiles of one text
    other = spec(8, 4, 12, lead=2)
    g.submit(other, False)
    jobs.complete()
    g.settle()
    assert other not in g._forced
    p = g.submit(other, True)
    assert not p.done() and len(jobs) == 1
    jobs.complete()
    g.settle()
    assert len(p.result()[1]) == 2 and g.compiles == 3 and g._forced[other] == p.result()[1]


def test_a_failed_build() -> None:
    """A GrammarError is a value: cached, counted as a compile and an error, never logged as compiled; the build's
    future gets a fresh one. Another exception is the future's, and nothing is cached or counted: the next request
    builds again."""
    g = Grammars(fake())
    jobs = manual(g)
    bad = g.submit("fake:not-a-grammar", True)
    jobs.complete()
    g.settle()
    assert isinstance(bad.result(), GrammarError) and (g.compiles, g.compile_errors) == (1, 1)
    assert isinstance(g._cache["fake:not-a-grammar"], GrammarError) and "fake:not-a-grammar" not in g._forced
    hit = g.submit("fake:not-a-grammar", True)
    assert hit.done() and isinstance(hit.result(), GrammarError) and hit.result() is not bad.result()
    boom = g.submit(FORCED, True)
    jobs.fail(MemoryError())
    assert g.settle() == 1 and isinstance(boom.exception(), MemoryError)
    assert FORCED not in g._cache and g.compiles == 1 and g.building == 0
    assert not g.submit(FORCED, True).done() and len(jobs) == 1


def test_the_caches_keep_their_lru_bound() -> None:
    g = Grammars(fake(), cache_size=2)
    jobs = manual(g)
    texts = [spec(s, 4, 12, lead=1) for s in range(4)]
    for t in texts:
        g.submit(t, True)
    jobs.complete_all()
    assert g.settle() == 4 and list(g._cache) == texts[2:] and list(g._forced) == texts[2:]


SCHEMAS = [
    {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
     "required": ["name", "age"], "additionalProperties": False},
    {"type": "object", "properties": {"kind": {"enum": ["alpha", "beta", "gamma"]}, "n": {"type": "integer",
                                                                                          "minimum": 0}},
     "required": ["kind", "n"], "additionalProperties": False},
    {"type": "object", "properties": {"identifier": {"type": "object", "properties": {"x": {"type": "integer",
                                                                                            "minimum": 0}},
                                                     "required": ["x"], "additionalProperties": False}},
     "required": ["identifier"], "additionalProperties": False},
    {"type": "object", "properties": {"answer": {"enum": ["yes", "no", "maybe"]}, "why": {"type": "string"}},
     "required": ["answer", "why"], "additionalProperties": False},
]


def test_build_jobs_prefixes_beside_the_step_loops_masks() -> None:
    """llguidance 1.9.1 documents no concurrent use of an executor, and `only` on the engine thread writes one shared
    buffer: a build job's prefix (`only`'s `own`) masks into its thread's buffer through the matcher's own call. Two
    threads computing prefixes 200 times each on different grammars, while a third fills masks of clones of the same
    grammars through the executor, get the single-thread prefixes."""
    if not (QWEN / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/qwen3.8-27b is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import LLGuidance, words
    from llm_mini_infer.grammar.spec import json_schema_grammar

    cfg = json.loads((QWEN / "config.json").read_text())
    vocab = int(cfg.get("text_config", cfg).get("vocab_size", cfg.get("vocab_size")))
    be = LLGuidance(QWEN / "tokenizer.json", vocab, "<|im_end|>")
    g = Grammars(be)
    compiled = [be.compile(json_schema_grammar(s, "guidance", True)) for s in SCHEMAS]
    want = [g._prefix(c) for c in compiled]
    assert all(0 < len(w) < FORCED_MAX for w in want)
    bad: list[object] = []
    stop = threading.Event()

    def prefixes(tid: int) -> None:
        try:
            for i in range(200):
                k = (2 * tid + i) % len(compiled)
                got = g._prefix(compiled[k], own=True)
                if got != want[k]:
                    bad.append((tid, k, got))
        except BaseException as exc:  # reported below
            bad.append(exc)

    def masks() -> None:
        buf = np.zeros((len(compiled), words(vocab)), np.int32)
        try:
            while not stop.is_set():
                g.fill([(g.matcher(c), i) for i, c in enumerate(compiled)], [], buf, len(compiled))
        except BaseException as exc:
            bad.append(exc)

    filler = threading.Thread(target=masks)
    filler.start()
    workers = [threading.Thread(target=prefixes, args=(t,)) for t in range(2)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(60)
    stop.set()
    filler.join(60)
    assert not any(t.is_alive() for t in [filler, *workers]), "a thread is still running"
    assert not bad and g.mask_rows > 0, bad[:3]


def test_own_masks_touch_neither_the_shared_buffer_nor_the_executor() -> None:
    """`only(..., own=True)`, and the `_prefix(..., own=True)` walk over it, mask into the calling thread's buffer: the
    engine thread's `_one` and the executor are never used, and two threads have two buffers."""
    if not (QWEN / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/qwen3.8-27b is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import LLGuidance
    from llm_mini_infer.grammar.spec import json_schema_grammar

    cfg = json.loads((QWEN / "config.json").read_text())
    vocab = int(cfg.get("text_config", cfg).get("vocab_size", cfg.get("vocab_size")))
    be = LLGuidance(QWEN / "tokenizer.json", vocab, "<|im_end|>")
    g = Grammars(be)
    compiled = be.compile(json_schema_grammar(SCHEMAS[0], "guidance", True))
    want = g._prefix(compiled)
    assert want
    shared = be._one
    shared_before = shared.copy()

    class Spy:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"the executor's {name} was used")

    be.executor = Spy()
    del be._one  # a read of it raises
    assert g._prefix(compiled, own=True) == want
    m = g.matcher(compiled)
    assert be.only(m, want[0], own=True) and not be.only(m, want[0] + 1, own=True)
    assert (shared == shared_before).all()
    bufs: list[object] = []

    def own() -> None:
        be.only(g.matcher(compiled), want[0], own=True)
        bufs.append(be._own.one)

    threads = [threading.Thread(target=own) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads) and len(bufs) == 2
    assert bufs[0] is not bufs[1] and all(b is not shared for b in bufs)

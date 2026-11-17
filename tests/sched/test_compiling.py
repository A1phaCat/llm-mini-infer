"""Requests whose grammar is being built (the scheduler's COMPILING; `grammar.engine.Grammars.submit`): they wait
outside the waiting heap with their arrival number while ready requests overtake them, and `Scheduler.collect` makes
them wait once built, or fails them (a GrammarError, another exception, the time limit) with an ERROR finish. Their
builds end at turns the tests choose (`ManualThreads`)."""

from __future__ import annotations

import logging

import pytest

from llm_mini_infer.engine.messages import AddRequest, FinishReason
from llm_mini_infer.sched.scheduler import COMPILING, RUNNING, WAITING

from ..support.compile_jobs import ManualThreads, manual
from ..support.fake_grammar import spec
from ..support.harness import Harness, Spec, prompt_of
from ..support.sched_cases import VOCAB
from ..support.workloads import image_prompt, text

G = spec(3, 2, 30)
F = spec(5, 4, 30, lead=3)  # forces its first three tokens


def harness(**kw) -> tuple[Harness, ManualThreads]:
    h = Harness(specs=[], vocab=VOCAB, **kw)
    return h, manual(h.sched.grammars)


def add(h: Harness, s: Spec) -> None:
    h.specs[s.rid] = s
    h.add(s)


def turn(h: Harness) -> None:
    rep = h.core.step()
    if rep is not None:
        h.collect(rep)
    h.sched.check_accounting(full=True)


def finish(h: Harness, jobs: ManualThreads, turns: int = 500) -> None:
    for _ in range(turns):
        jobs.complete_all()
        turn(h)
        if not h.core.busy:
            return
    raise AssertionError("not idle")


def status(h: Harness, rid: int) -> int:
    return h.sched.reqs[(0, rid)].status


def test_a_ready_request_overtakes_one_whose_grammar_builds() -> None:
    """Under fcfs: the first request waits for its grammar outside the heap, the second runs meanwhile; once built, the
    first waits with its own arrival number and runs too. Both outputs are the fake's."""
    h, jobs = harness(max_num_seqs=2)
    add(h, Spec(rid=0, prompt_len=10, max_tokens=6, seed=1, grammar=G))
    add(h, Spec(rid=1, prompt_len=10, max_tokens=6, seed=2))
    s = h.sched
    a, b = s.reqs[(0, 0)], s.reqs[(0, 1)]
    assert (a.status, b.status, a.arrival, b.arrival) == (COMPILING, WAITING, 0, 1)
    assert (s.num_compiling, s.num_waiting, h.core.busy, h.core.compiling_only) == (1, 1, True, False)
    for _ in range(3):
        turn(h)
        assert a.status == COMPILING and b.status == RUNNING and s.structured_requests == 0
    jobs.complete()
    turn(h)
    assert a.status == RUNNING and a.compiled is not None and s.structured_requests == 1 and s.num_compiling == 0
    finish(h, jobs)
    h.check()
    assert [h.results[r].finish for r in (0, 1)] == [FinishReason.LENGTH] * 2


def test_admission_follows_arrival_among_the_ready() -> None:
    """One slot: the compiling request's build ends while two later requests wait; it is admitted before the
    second of them, by its arrival number."""
    h, jobs = harness(max_num_seqs=1)
    for rid, g in enumerate(("", G, "", "")):
        add(h, Spec(rid=rid, prompt_len=8, max_tokens=3, seed=10 + rid, grammar=g))
    order: list[int] = []
    for _ in range(200):
        turn(h)
        running = [r.rid for r in h.sched.slot_req if r is not None]
        if running and running[0] not in order:
            order.append(running[0])
            if order == [0]:
                jobs.complete()  # built while 2 and 3 wait
        if not h.core.busy:
            break
    assert order == [0, 1, 2, 3]
    h.check()


@pytest.mark.parametrize("how", ["grammar", "exception", "timeout"])
def test_a_failed_build_fails_its_request_once_and_frees_its_images_once(how, caplog) -> None:
    """A GrammarError, another exception, or the time limit: an ERROR finish, once; its pixels released once (by
    `_to_finished`, not also the reject tail); counted in grammar_rejects except the exception, which is logged as
    `add`'s is. A ready request beside it runs on."""
    prompt = image_prompt(text(1, 20), 48, text(2, 15))
    h, jobs = harness(encoder_rows=1024, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=16)
    grammar = "fake:not-a-grammar" if how == "grammar" else G
    add(h, Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=5, images=[(20, 12, 16, 7)], grammar=grammar))
    add(h, Spec(rid=1, prompt_len=12, max_tokens=4, seed=6))
    turn(h)
    assert status(h, 0) == COMPILING
    with caplog.at_level(logging.WARNING, logger="llm_mini_infer.sched.scheduler"):
        if how == "grammar":
            jobs.complete()
        elif how == "exception":
            jobs.fail(RuntimeError("quoted request content"))
        else:
            h.sched.compile_timeout_ns = 0
        turn(h)
    assert (0, 0) not in h.sched.reqs and h.results[0].finish == FinishReason.ERROR and h.results[0].tokens == []
    assert h.released == {0} and h.sched.num_compiling == 0
    assert h.sched.grammar_rejects == (0 if how == "exception" else 1)
    msgs = [r.getMessage() for r in caplog.records]
    want = {"grammar": "rejecting request 0:0: its grammar does not compile with this tokenizer",
            "exception": "rejecting request 0:0: admission failed (exception messages left out)",
            "timeout": "rejecting request 0:0: its grammar did not compile within dev.structured_output."
                       "compile_timeout_s (0 s)"}[how]
    assert len(msgs) == 1 and msgs[0].startswith(want) and "quoted" not in msgs[0], msgs
    if how == "timeout":
        assert h.sched.grammars.building == 1  # the build goes on
    h.sched.compile_timeout_ns = 10**12
    finish(h, jobs)
    assert h.results[1].finish == FinishReason.LENGTH and h.results[1].tokens == h.expected(h.specs[1])
    assert not h.sched.grammars.building


def test_the_time_limit_leaves_the_build_for_a_later_request() -> None:
    """Past the limit the request fails, but its build stays in flight (an llguidance compile cannot be stopped): a
    later request of the same grammar joins it, and the cache fills when it ends (not cached as invalid)."""
    h, jobs = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    h.sched.compile_timeout_ns = 0
    turn(h)
    assert h.results[0].finish == FinishReason.ERROR and h.sched.grammars.building == 1
    h.sched.compile_timeout_ns = 10**12
    add(h, Spec(rid=1, prompt_len=8, max_tokens=3, seed=2, grammar=G))
    assert status(h, 1) == COMPILING and len(jobs) == 1 and jobs.submitted == 1
    jobs.complete()
    turn(h)
    assert status(h, 1) == RUNNING and h.sched.grammars.compiles == 1 and G in h.sched.grammars._cache
    finish(h, jobs)
    h.check(failed_ok=True)
    assert h.results[1].finish == FinishReason.LENGTH


def test_abort_and_abort_all_while_compiling() -> None:
    """An abort finishes a COMPILING request as a waiting one (counted, no finish row of the engine's), abort_all with
    a finish row; neither cancels the build, which may be shared."""
    h, jobs = harness()
    for rid in range(3):
        add(h, Spec(rid=rid, prompt_len=8, max_tokens=3, seed=rid + 1, grammar=G))
    s = h.sched
    fut = s.reqs[(0, 0)].future
    assert s.num_compiling == 3 and len(jobs) == 1 and s.reqs[(0, 2)].future is fut
    s.abort(0, [0])
    assert (0, 0) not in s.reqs and s.num_compiling == 2 and s.num_aborts == 1 and not fut.cancelled()
    s.check_accounting(full=True)
    s.abort_all()
    assert not s.reqs and s.num_compiling == 0 and not fut.cancelled() and s.grammars.building == 1
    assert sorted(rid for _, rid, _ in s._extra) == [1, 2]
    s.check_accounting(full=True)
    turn(h)
    assert h.results[1].finish == h.results[2].finish == FinishReason.ABORT


def test_a_duplicate_id_while_compiling_is_rejected() -> None:
    h, jobs = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    h.sched.add(AddRequest(rid=0, frontend=0, prompt=prompt_of(h.specs[0], h.vocab), max_tokens=3, temperature=0.0,
                           top_p=1.0, top_k=0, min_p=0.0, seed=9, stop_token_ids=[], grammar=G))
    assert h.sched._extra == [(0, 0, FinishReason.ERROR)] and status(h, 0) == COMPILING and len(jobs) == 1
    h.sched._extra.clear()  # the duplicate's row, which the harness would read as the original's finish
    finish(h, jobs)
    h.check()


def test_an_add_while_stopping_is_aborted_and_the_loop_idles() -> None:
    """The stopping loop adds then aborts everything (`EngineLoop._poll`): a request whose grammar builds is aborted
    at once, and once its finish row is sent nothing keeps the loop busy, though the build runs on."""
    h, jobs = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    h.sched.abort_all()
    assert h.core.busy and h.sched.num_compiling == 0  # the finish row, unsent
    turn(h)
    assert not h.core.busy and h.results[0].finish == FinishReason.ABORT and len(jobs) == 1


def test_a_paused_frontends_request_waits_once_built() -> None:
    """Built while its frontend is paused (backpressure): it waits in the heap, deferred by `_admit`, and runs once
    the frontend resumes."""
    h, jobs = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    h.sched.paused_frontends.add(0)
    jobs.complete()
    for _ in range(3):
        turn(h)
        assert status(h, 0) == WAITING and h.sched.num_waiting == 1 and h.core.paused_only
    h.sched.paused_frontends.clear()
    finish(h, jobs)
    h.check()


def test_requests_sharing_a_build_count_one_compile_and_take_its_prefix() -> None:
    """Two requests of one grammar: one job, one compile, the same forced prefix for both (cut at each one's stops);
    a stop-string request's build is another (it wants no prefix)."""
    h, jobs = harness(forced_prefix=True)
    add(h, Spec(rid=0, prompt_len=8, max_tokens=8, seed=1, grammar=F))
    add(h, Spec(rid=1, prompt_len=9, max_tokens=8, seed=2, grammar=F))
    add(h, Spec(rid=2, prompt_len=9, max_tokens=8, seed=3, grammar=F, stop_strings=True))
    assert len(jobs) == 2 and h.sched.reqs[(0, 0)].future is h.sched.reqs[(0, 1)].future
    jobs.complete_all()
    turn(h)
    g = h.sched.grammars
    a, b, c = (h.sched.reqs[(0, r)] for r in range(3))
    assert g.compiles == 2 and len(a.forced) == len(b.forced) == 3 and c.forced == () and g.building == 0
    finish(h, jobs)
    h.check()


def test_a_build_without_waiters_settles_at_the_next_collect() -> None:
    """Its only request aborted, the build ends later: the next collect fills the cache, counts it once and empties
    the builds in flight."""
    h, jobs = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    h.sched.abort(0, [0])
    turn(h)
    g = h.sched.grammars
    assert g.building == 1 and g.compiles == 0
    jobs.complete()
    assert h.core.collect_due and not h.core.busy
    turn(h)
    assert g.building == 0 and g.compiles == 1 and G in g._cache
    h.sched.check_accounting(full=True)


def test_a_preempted_request_is_readmitted_without_a_new_build() -> None:
    sp = Spec(rid=0, prompt_len=8, max_tokens=12, seed=1, grammar=F, preempt_at=5)
    h = Harness(specs=[sp], vocab=VOCAB, max_num_seqs=1, forced_prefix=True)  # its preemption hook needs the spec
    jobs = manual(h.sched.grammars)
    h.add(sp)
    jobs.complete()
    for _ in range(10):
        h.turn += 1  # the hook reads the harness's turn
        turn(h)
    finish(h, jobs)
    h.check()
    assert h.sched.num_preemptions == 1 and h.sched.grammars.compiles == 1 and jobs.submitted == 1


def test_the_accounting_checks_catch_a_broken_compiling_set() -> None:
    """`check_accounting` with COMPILING requests: a request missing from the set, or one also in the heap."""
    h, _ = harness()
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    s = h.sched
    req = s.reqs[(0, 0)]
    s.check_accounting(full=True)
    s._compiling.clear()
    with pytest.raises(AssertionError, match="compiling count out of sync"):
        s.check_accounting()
    s._compiling[req] = None
    req.future = None
    with pytest.raises(AssertionError, match="its build is not in flight"):
        s.check_accounting()
    req.future = s.grammars._inflight[(G, False)]
    s._waiting.append((0, req.arrival, req))  # only the heap holds it: status and counts stay consistent
    with pytest.raises(AssertionError, match="a compiling request in the waiting heap"):
        s.check_accounting(full=True)

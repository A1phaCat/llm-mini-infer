"""The randomized mode of the engine's grammar builds (`grammar.engine.Grammars.submit`): a structured workload run
twice, once with its grammars compiled at `add` (no compile threads, as every other suite runs) and once with its
builds ending at random turns (`ManualThreads`), `check_accounting(full=True)` after each of those turns. Every request
ends alike in both, with the same tokens: a build's wait changes when a request runs, never what it computes."""

from __future__ import annotations

import random
from dataclasses import replace

from hypothesis import strategies as st

from llm_mini_infer.engine.messages import FinishReason

from .compile_jobs import manual
from .fuzzing import event
from .harness import Harness
from .sched_cases import VOCAB, check_plans
from .structured_cases import structured_workloads


@st.composite
def compile_async_workloads(draw):
    """`structured_workloads` without aborts and NaNs: their turn, and a NaN's cascade through shared cache writes,
    depend on when requests run. An empty mask injected at a position fires only where the position is sampled, not
    where a final chunk took it as a forced token (when the step has room, best-effort), so it stays only where no
    lead is forced: no lead, or the forced prefix off (`run_async_workload` compares its request's tokens loosely
    under MTP). It never fires on a mask that allows every token (a grammar that ended under `ignore_eos`), whichever
    row samples the position. With grammars that force their first tokens, grammars shared by several requests (one
    build), the forced prefix on or off, and how the builds end: each queued job ends at a turn with probability `p`,
    drawn by `jobs_seed`."""
    w = draw(structured_workloads())
    forced_prefix = draw(st.booleans())
    texts: list[str] = []
    specs = []
    for s in w["specs"]:
        g = s.grammar
        if g and texts and draw(st.integers(0, 2)) == 0:
            g = draw(st.sampled_from(texts))  # a build in flight, or cached
        elif g:
            _, seed, lo, hi = g.split(":")
            lead = draw(st.sampled_from([0, 0, 1, 3]))
            g = f"fake:{seed}:{lo}:{hi}:0:0:{lead}:0" if lead and int(hi) > lead else g
            texts.append(g)
        forced = forced_prefix and g is not None and len(g.split(":")) > 6 and int(g.split(":")[6]) > 0
        keep = not forced
        specs.append(replace(s, grammar=g, abort_at=None, nonfinite_after=None,
                             mask_empty_after=s.mask_empty_after if keep else None))
    return dict(w, specs=specs, forced_prefix=forced_prefix, jobs_seed=draw(st.integers(0, 1 << 30)),
                p=draw(st.sampled_from([0.05, 0.3, 0.7, 1.0])))


def run_async_workload(ex: dict) -> None:
    ex = dict(ex)
    rng, p = random.Random(ex.pop("jobs_seed")), ex.pop("p")
    sync = Harness(vocab=VOCAB, **ex)
    sync.run(check_every_turn=True)
    h = Harness(vocab=VOCAB, **ex)
    jobs = manual(h.sched.grammars)
    step, s = h.core.step, h.sched
    waits = [0]
    pending: list[int] = []  # the builds' futures `submit` handed out pending, one entry a request
    submit = s.grammars.submit

    def submitting(text: str, prefix: bool):
        fut = submit(text, prefix)
        if not fut.done():
            pending.append(id(fut))
        return fut

    s.grammars.submit = submitting  # type: ignore[method-assign]

    def stepping(emit=None):
        jobs.complete_some(rng, p)
        out = step(emit)
        s.check_accounting(full=True)
        waits[0] += s.num_compiling
        return out

    h.core.step = stepping  # type: ignore[method-assign]
    h.run()
    check_plans(h)
    h.check(failed_ok=True)
    k = ex.get("mtp_k") or 0
    for rid, r in h.results.items():
        want = sync.results[rid]
        if k and h.specs[rid].mask_empty_after is not None and r.finish == want.finish == FinishReason.ERROR:
            # a verify step that meets the empty mask fails its request with the drafts it accepted before the
            # position (scheduler: the request fails alone), so how many it delivered depends on how its steps fell
            a, b = sorted((r.tokens, want.tokens), key=len)
            assert b[:len(a)] == a and len(b) - len(a) <= k, f"request {rid}: {r} != {want} (synchronous)"
            continue
        assert (r.finish, r.tokens) == (want.finish, want.tokens), f"request {rid}: {r} != {want} (synchronous)"
    assert s.num_compiling == 0 and not s.grammars.building and not s.reqs
    assert s.structured_requests == sync.sched.structured_requests
    assert s.grammar_rejects == sync.sched.grammar_rejects
    if waits[0]:
        event("compile_async: a request waited for its build")
    if len(set(pending)) < len(pending):
        event("compile_async: a build shared")
    if s.forced_requests:
        event("compile_async: forced prefix")

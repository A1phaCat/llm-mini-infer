"""The NaN-safety suite's workloads and driver (tests/sched/test_nan_safety.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.sched.plan import reserve_positions

from .fuzzing import event, st
from .harness import Harness, Spec, check_nan_invariants, check_precise
from .sched_cases import VOCAB, check_plans, nan_events, stem
from .window_mtp import bound


@st.composite
def nan_workloads(draw):
    """Small workloads aimed at what NaN that a prefill chunk writes unseen can leave in the cache: 2-5 requests on one
    stem, each cut at a multiple of the (full) page with a hint at its cut, so hits land on page boundaries (under MTP
    with a drafter that writes KV, an admission copies the page below its hit) and branch points on rows still being
    planned; early NaN writes, half of them transient; aborts and priorities; a precise namespace (bf16, or refine on
    reuse at the first reuse), its PRECISE chunks at most 1, 2 or 4 pages (window pages in window mode) in three
    workloads of five (the harness's `precise_cap`: a refine's piece size); MTP k 1, 3 or 5, the drafter writing KV in
    three workloads of four; GDN, attention-only or window mode. In the cache suites' campaigns the copy below a hit
    showed up once in 18,586 window_precise workloads, and a branch checkpoint that an unverified writer's row took was
    never left behind, though such checkpoints are taken in about 2% of their workloads."""
    mode = draw(st.sampled_from(["gdn", "attention", "window"]))
    unit = draw(st.sampled_from([1, 2, 4, 8]))  # the page (window mode: the window page)
    k = draw(st.sampled_from([1, 3, 5]))
    max_model_len, max_tokens = 96, draw(st.integers(4 * (1 + k), 64))
    w = dict(mtp_k=k, draft_kv=draw(st.sampled_from([True, True, True, False])),
             accept_p=draw(st.sampled_from([0.0, 0.5, 1.0])), precise=draw(st.sampled_from(["bf16", "reuse"])),
             refine_after=0, max_num_seqs=draw(st.integers(1, 3)), max_tokens=max_tokens,
             max_model_len=max_model_len, policy=draw(st.sampled_from(["fcfs", "priority"])),
             chunk_cap=unit * draw(st.sampled_from([0, 1, 2, 4])), grid=unit * draw(st.sampled_from([1, 2])),
             precise_cap=unit * draw(st.sampled_from([0, 0, 1, 2, 4])))
    if mode == "window":
        page, window = unit * draw(st.sampled_from([1, 2])), unit * draw(st.sampled_from([1, 2, 4]))
        w.update(window=window, window_page_size=unit, page_size=page)
        need = bound(window, unit, page, max_tokens, max_model_len, k) + 1
    else:
        page = unit
        w.update(page_size=page, hybrid=mode == "gdn", checkpoint_slots=draw(st.integers(1, 6)))
        need = pages_for_one_request(max_model_len, page, reserve_positions(2, k), True) + 1
    w["num_pages"] = draw(st.integers(need, 2 * need))  # prompts of at most 40 tokens: one fits beside another's lock
    base = stem(draw(st.integers(0, 9)), 40)
    n = draw(st.integers(2, 5))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs = []
    for rid in range(n):
        cut = page * draw(st.integers(1, max(2, 32 // page)))
        extra = draw(st.integers(1, 8))
        prompt = base[:cut] + (stem(10 + rid, extra) if draw(st.booleans()) else base[cut:cut + extra])
        nan = {}
        if not draw(st.integers(0, 2)):
            nan = dict(nonfinite_write=draw(st.integers(0, min(len(prompt) - 1, 2 * page))),
                       nonfinite_once=draw(st.booleans()))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 4)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 8)), priority=draw(st.integers(0, 2)),
                          abort_at=draw(st.integers(1, 12)) if not draw(st.integers(0, 3)) else None, hints=[cut],
                          **nan))
    w["specs"] = specs
    return w


def run_nan_workload(w: dict) -> Harness:
    """Run to the end, checking the accounting and the precise namespace after every loop turn, then the plans, the
    NaN invariants and every output."""
    h = Harness(vocab=VOCAB, **w)
    orig = h.core.step

    def step(*a, **kw):
        out = orig(*a, **kw)
        check_precise(h)
        return out

    h.core.step = step
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    nan_events(h)
    event("window mode" if s.window else "GDN" if s.hybrid else "attention-only")
    if any(p["kv_copy"] and any(n and not n % s.ps for n in p["seq_len"]) for p in h.plans):
        event("a page-aligned hit copied")
    if s.ckpts_taken["branch"]:
        event("branch checkpoint")
    if s.precise_hits:
        event("precise hit")
    if s.refines:
        event("refined")
    if s.num_preemptions:
        event("preempted")
    if s.num_failed and s.num_aborts:
        event("failed and aborted")
    return h

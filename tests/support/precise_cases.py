"""The precise-prefix suite's refine checks, workloads and drivers (tests/sched/test_precise_prefix.py)."""

from __future__ import annotations

import numpy as np

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.sched.plan import PRECISE, reserve_positions

from .fuzzing import event, nan_report, nan_write, st
from .harness import EOS, Harness, Spec, check_nan_invariants, check_precise
from .sched_cases import VOCAB, check_plans, nan_events, output_of, stem
from .window_mtp import with_mtp


def precise_rows(h: Harness, slot: int | None = None) -> list[tuple[int, int, int]]:
    """(step, pos0, n_query) of every PRECISE row the plans carried (of one slot, when given)."""
    out = []
    for p in h.plans:
        for s, q, p0, pr in zip(p["slots"], p["n_query"], p["pos0"], p["precise"]):
            if pr and (slot is None or s == slot):
                out.append((p["step"], p0, q))
    return out


def watch_refines(h: Harness) -> None:
    """Before a run, for `check_refine_plans`: record every PRECISE row's refine end, hints and image spans
    (`pieces`)."""
    s = h.sched
    h.pieces = {}
    schedule = s.schedule

    def watched():
        out = schedule()
        if out is not None:
            plan = out[0]
            for i in np.flatnonzero(plan.sched_flags & PRECISE).tolist():
                req = s.slot_req[int(plan.sched_slot[i])]
                h.pieces[plan.step_id, i] = (req.refine, list(req.hints), req.sp_start.tolist())
        return out

    s.schedule = watched


def check_refine_plans(h: Harness) -> None:
    """How a refine was stepped (reuse mode; `watch_refines` before the run): a PRECISE row ends where its piece does
    (the next multiple of the piece size from position 0, or the refine's end), whatever the load; only a hint or an
    image span inside the piece cuts it first, and a row that is one whole image span may pass that end."""
    s = h.sched
    if s.precise != "reuse":
        return
    piece = min(s.precise_cap or s.piece, s.max_tokens)
    for p in h.plans:
        for i in np.flatnonzero(p["precise"]).tolist():
            lo, hi = p["pos0"][i], p["pos0"][i] + p["n_query"][i]
            if (i, lo, hi) in p["spans"]:
                continue
            refine, hints, spans = h.pieces[p["step"], i]
            end = min(refine, (lo // piece + 1) * piece)
            assert hi <= end, f"plan {p['step']}: the PRECISE row [{lo}, {hi}) crosses its piece's end {end}"
            assert hi == end or hi in hints or any(lo < x <= end for x in spans), \
                f"plan {p['step']}: the PRECISE row [{lo}, {hi}) stops short of its piece's end {end}"


def refine_options(draw, grid: int, max_tokens: int, max_num_seqs: int, k: int) -> dict:
    """How a refine is stepped: the piece size (0: derived, as is a drawn piece that does not fit next to a full batch
    of decode rows), and in two workloads of five the harness's cap on PRECISE chunks in its place (`precise_cap`: one
    or two grids within max_num_batched_tokens, not held to that room), so that short prompts still refine in several
    pieces."""
    piece = min(grid * draw(st.sampled_from([0, 0, 1, 2, 5])), max_tokens // grid * grid)
    cap = min(grid * draw(st.sampled_from([0, 0, 0, 1, 2])), max_tokens // grid * grid)
    room = max_tokens - max_num_seqs * (1 + k)
    return dict(refine_chunk_tokens=piece if piece <= room else 0, precise_cap=cap)


@st.composite
def precise_workloads(draw):
    """The cache suite's workloads (shared stems, follow-up turns, hints, NaNs, aborts, MTP, priorities, pressure)
    under a precise prefix policy, with prompts drawn from fewer stems so that prefixes are reused often enough to be
    refined, and the refine threshold, the PRECISE cost and how a refine is stepped drawn too; half of the `bf16`
    workloads attention-only (a precise hit there locks the node holding hit - 1, which may run past the hit)."""
    page_size = draw(st.sampled_from([1, 2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
    max_model_len = 160
    longest = draw(st.sampled_from([100, 150]))
    stems = [stem(i, draw(st.integers(8, longest - 20))) for i in range(draw(st.integers(1, 2)))]
    n = draw(st.integers(2, 10))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["stem", "stem", "stem", "turn", "random"]))
        if kind == "turn" and specs:
            prev = draw(st.sampled_from(specs))
            out = output_of(prev)
            prompt = list(prev.prompt) + out[:draw(st.integers(0, len(out)))]
        elif kind in ("stem", "turn"):
            s = draw(st.sampled_from(stems))
            prompt = s[:draw(st.integers(len(s) // 2, len(s)))]
        else:
            prompt = []
        prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=12))
        prompt = [t if t != EOS else 0 for t in prompt][:longest]
        eos = draw(st.one_of(st.none(), st.integers(0, 30)))
        nan = draw(st.sampled_from([None] * 10 + [0, 3]))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 30)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 40)), eos_after=eos,
                          ignore_eos=draw(st.booleans()) if eos is not None else False,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 70))), priority=draw(st.integers(0, 2)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          salt=draw(st.sampled_from([None, None, None, "s"])), nonfinite_after=nan,
                          **nan_write(draw, len(prompt))))
    k = draw(st.sampled_from([0, 0, 1, 3]))
    min_pages = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True) + 1
    max_tokens, max_num_seqs = draw(st.integers(16, 128)), draw(st.integers(1, 5))
    precise = draw(st.sampled_from(["reuse", "reuse", "bf16"]))
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                report_unsampled=nan_report(draw),
                accept_p=draw(st.sampled_from([0.0, 0.5, 0.8, 1.0])),
                max_num_seqs=max_num_seqs, max_tokens=max_tokens,
                num_pages=draw(st.integers(min_pages, min_pages * 3)), checkpoint_slots=draw(st.integers(1, 8)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 13])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])),
                precise=precise, hybrid=precise != "bf16" or draw(st.booleans()),
                refine_after=draw(st.sampled_from([0, 1, 1, 2])), refine_min=draw(st.sampled_from([0, 0, 0, 8, 40])),
                precise_cost=draw(st.sampled_from([1.0, 2.5])),
                **refine_options(draw, grid, max_tokens, max_num_seqs, k))


def run_precise_workload(w: dict) -> Harness:
    h = Harness(vocab=VOCAB, **w)
    watch_refines(h)
    orig = h.core.step

    def step(*a, **kw):
        r = orig(*a, **kw)
        check_precise(h)
        return r

    h.core.step = step
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    check_refine_plans(h)
    h.check(failed_ok=True)
    s = h.sched
    nan_events(h)
    if not s.hybrid and s.precise_hits:
        event("attention-only precise hit")
    if s.refines:
        event("refined")
    if s.refines and s.k:
        event("refined under MTP")
    if s.precise == "reuse" and precise_rows(h):
        event("fixed pieces")
    if s.precise_hits:
        event("precise hit")
    if precise_rows(h):
        event("PRECISE rows")
    if s.num_preemptions and precise_rows(h):
        event("preempted with PRECISE rows")
    if s.precise == "reuse" and any(p["precise"][row] for p in h.plans for row, _, _ in p["spans"]):
        event("an image span in a refine's PRECISE row")
    if s.num_failed and precise_rows(h):
        event("failed (NaN) with PRECISE rows")
    return h


@st.composite
def window_precise_workloads(draw):
    """The window suite's workloads (Gemma 4's cache: windows as checkpoints in one arena), a third of them the Gemma
    image suite's (bidirectional image spans, encoder budgets and caches), under a precise prefix policy; half of them
    with MTP too (Gemma 4 with its assistant, refining on reuse)."""
    from .window_workloads import gemma_image_workloads, window_workloads

    w = draw(window_workloads()) if draw(st.integers(0, 2)) else dict(draw(gemma_image_workloads()), bidirectional=True)
    w.update(precise=draw(st.sampled_from(["reuse", "reuse", "bf16"])), refine_after=draw(st.sampled_from([0, 1, 2])),
             refine_min=draw(st.sampled_from([0, 0, 0, 8, 40])), precise_cost=draw(st.sampled_from([1.0, 2.5])))
    if draw(st.booleans()):
        w = with_mtp(draw, w, stops=draw(st.booleans()))
    w.update(refine_options(draw, w["grid"], w["max_tokens"], w["max_num_seqs"], w.get("mtp_k", 0)))
    return w

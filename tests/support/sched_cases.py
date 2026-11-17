"""The scheduler suites' shared pieces: the toy vocabulary and prompts, NaN events, and the plan checks every suite runs
(tests/sched/test_cache_sched.py, tests/sched/test_scheduler.py)."""

from __future__ import annotations

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.runner.fake import fake_generate

from .fuzzing import event
from .harness import EOS, Harness, Spec, prompt_of

VOCAB = 1000


def stem(i: int, n: int) -> list[int]:
    return [(j * 31 + i * 17 + 5) % (VOCAB - 1) for j in range(n)]


def output_of(spec: Spec) -> list[int]:
    eos = {spec.prompt_len + spec.eos_after: EOS} if spec.eos_after is not None else {}
    return fake_generate(prompt_of(spec, VOCAB), spec.seed, spec.max_tokens, VOCAB, (EOS,), eos)


def nan_events(h: Harness) -> None:
    """The NaN dimension's events (`tests/sched/test_nan_safety.py`), shared by the cache suites."""
    s = h.sched
    if any(sp.nonfinite_write is not None and h.results[sp.rid].finish == FinishReason.ERROR
           for sp in h.specs.values()):
        event("NaN written by a prefill chunk")
    if any(sp.nonfinite_once and sp.seed in h.runner.fired for sp in h.specs.values()):
        event("a transient NaN written (not repeated by a recomputation)")
    if s.num_failed and s.cache is not None and s.cache.rolled_back:
        event("rolled back")
    if h.runner.report_unsampled:
        event("unsampled rows report NaN")
    if h.cascaded:
        event("failed for a rolled-back write it used")


def shared(n_prefix: int, n_suffix: int, i: int) -> list[int]:
    return stem(0, n_prefix) + [(i * 13 + j * 7 + 1) % (VOCAB - 1) for j in range(n_suffix)]


def check_plans(h: Harness) -> None:
    budget, row_tokens, grid = h.sched.max_tokens, h.sched.row_tokens, h.sched.grid
    for p in h.plans:
        assert len(set(p["slots"])) == len(p["slots"]), f"slot twice in plan {p['step']}"
        assert p["n_tokens"] <= budget, f"plan {p['step']} over the token budget"
        assert len(p["slots"]) <= h.sched.max_num_seqs
        rows = sum(p["device"])
        dec = sum(n for n, d in zip(p["n_query"], p["device"]) if d)
        if row_tokens and rows:  # a step with decode rows: its prefill within rows x prefill tokens
            # A bidirectional image span that starts a chunk goes whole: it may pass the bound (by at most its own
            # length, the largest of the step's spans)
            extra = max((e - s for _, s, e in p.get("spans", ())), default=0)
            if h.sched.precise == "reuse":  # so do a refine's fixed PRECISE pieces
                extra += sum(q for q, pr in zip(p["n_query"], p["precise"]) if pr)
            if h.sched.forced_whole:  # and, under batch invariance, a final chunk's whole forced prefix
                extra += (h.sched.reserve - h.sched.chain) * sum(
                    sm and not d for sm, d in zip(p["sample"], p["device"]))
            assert p["n_tokens"] - dec <= max(grid, row_tokens // rows) + extra, \
                f"plan {p['step']}: {p['n_tokens'] - dec} prefill tokens next to {rows} decode rows"

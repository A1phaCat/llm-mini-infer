"""The logprobs fuzz suite's workload run (tests/runner/test_logprobs.py, the fuzz campaign and the plan diff)."""

from __future__ import annotations

import copy

from .forced_cases import run_switches_workload
from .fuzzing import event
from .harness import Harness
from .workloads import run_cache_workload


def run_both(w: dict, run, k_of) -> tuple[Harness, Harness]:
    """A workload twice: its requests with the logprobs `k_of(spec)` asks (checked against the fake's prediction),
    and without any; the plans and every request's tokens and finish must be the same."""
    plain = copy.deepcopy(w)
    for s in plain["specs"]:
        s.logprobs = -1
    lp = copy.deepcopy(w)
    for s in lp["specs"]:
        s.logprobs = k_of(s)
    a, b = run(plain), run(lp)
    assert a.plans == b.plans, "a request's logprobs changed the plans"
    for rid in a.results:
        assert (a.results[rid].tokens, a.results[rid].finish) == (b.results[rid].tokens, b.results[rid].finish), rid
    return a, b


def run_logprobs_workload(w: dict) -> None:
    tops = dict(zip((s.rid for s in w["workload"]["specs"]), w["tops"]))
    run = run_switches_workload if w["kind"].startswith("structured") else run_cache_workload
    _, h = run_both(w["workload"], run, lambda s: tops[s.rid])
    if any(r.logprobs for r in h.results.values()):
        event("logprobs: values delivered")
    if h.sched.num_preemptions and any(t >= 0 for t in w["tops"]):
        event("logprobs: preempted")
    if h.sched.forced_tokens and any(t >= 0 for t in w["tops"]):
        event("logprobs: forced tokens")

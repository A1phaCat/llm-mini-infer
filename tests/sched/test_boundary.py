"""Planning at the forward/sampling boundary (`EngineCore.await_boundary`; architecture.md, "Planning at the
forward/sampling boundary"): on a runner whose steps split into a forward and a sampling part, a request that arrives
during step N's forward joins step N+1 instead of N+2; for the arrivals it is given, the scheduler's every plan and
output stay the same; and the setting is off unless set, a set one checked against the runner's lone decode step
(`derive_plan_at_boundary`), Gemma 4 only on CUDA."""

from __future__ import annotations

import json
import time

import pytest

from llm_mini_infer import config
from llm_mini_infer.engine.core import BOUNDARY_PLAN_MS, P_BOUNDARY, derive_plan_at_boundary

from ..support.harness import Harness, Spec


def _timed(h: Harness, step_ms: float, sample_ms: float) -> Harness:
    """The harness's fake on a device clock: forwards of `step_ms`, sampling parts of `sample_ms`."""
    h.runner.step_s, h.runner.sample_s = step_ms / 1e3, sample_ms / 1e3
    return h


ATTEMPTS = 20


def _arrival_attempt(boundary: bool) -> bool | None:
    """One run of the scenario. Returns None when the host was too slow for its timing (the process was preempted
    before the arrival, or the arrival came after the forward ended): that run says nothing about the engine. Asserts
    the engine's behavior on any other run."""
    a = Spec(rid=1, prompt_len=8, max_tokens=8, seed=0xB0D1_0001)
    b = Spec(rid=2, prompt_len=8, max_tokens=2, seed=0xB0D1_0002)
    h = _timed(Harness(specs=[a, b], hybrid=False), 40.0, 20.0)
    h.add(a)
    for _ in range(3):  # A's prefill, then its decode steps: one in flight at each turn's top
        rep = h.core.step()
        if rep is not None:
            h.collect(rep)
    inf, hd = h.core._inflight[0], h.core._inflight[1]
    n = inf.step_id
    arrive = hd.fwd_end - 0.5 * hd.fwd_dur
    if time.perf_counter() >= arrive - 0.005:
        return None
    arrived, polled = [], []

    def poll() -> int:
        now = time.perf_counter()
        polled.append(now)
        if not arrived and now >= arrive:
            h.add(b)
            arrived.append(now)
        return 0

    first = None
    while h.core.busy or not arrived:
        if boundary:
            h.core.await_boundary(poll)
        poll()
        rep = h.core.step()
        if rep is None:
            continue
        for out in rep.outputs.values():
            if first is None and 2 in out.rids[out.n_new > 0].tolist():
                first = rep.step_id
        h.collect(rep)
    h.check()
    # The arrival must come while N's forward runs, and (off) the turn's top poll before it.
    if polled[0] >= arrive or (boundary and arrived[0] >= hd.fwd_end):
        return None
    assert first == n + (1 if boundary else 2)
    assert (h.core.phase_ns[P_BOUNDARY] > 0) == boundary
    if boundary:  # the spin ended at N's forward's end, after the arrival
        assert h.core.phase_ns[P_BOUNDARY] >= 0.5 * hd.fwd_dur * 1e9 * 0.5
    # The next launch came before N's sampling ended. A preemption between turns breaks that without the engine
    # being wrong, so a late launch also makes the run void.
    return True if h.core.gpu_idle_ns < 2e6 else None


@pytest.mark.parametrize("boundary", [False, True])
def test_an_arrival_during_a_forward_joins_the_next_step(boundary) -> None:
    """A request decodes on 40 ms forwards and 20 ms sampling parts; a second one arrives halfway through step N's
    forward. Polled at the top of the turn only, it is planned into N+2 (N+1 was planned before it arrived); polled
    until N's forward ends, into N+1, and the device never waits for the next step (its launch overlaps N's
    sampling). The test works on the real clock, so a run whose timing preconditions the host missed (preempted
    before the arrival, a late launch) is repeated; the asserts on the engine's behavior stay exact."""
    for _ in range(ATTEMPTS):
        if _arrival_attempt(boundary):
            return
    pytest.fail(f"no run in {ATTEMPTS} met the timing preconditions: the host is too slow for this test")


def _run(specs: list[Spec], boundary: bool, **kw) -> Harness:
    h = _timed(Harness(specs=specs, **kw), 2.0, 1.0)
    pending = sorted(specs, key=lambda s: s.arrive)
    while pending or h.core.busy:
        if boundary:
            h.core.await_boundary(lambda: 0)
        while pending and pending[0].arrive <= h.turn:
            h.add(pending.pop(0))
        rep = h.core.step()
        if rep is not None:
            h.collect(rep)
        h.turn += 1
    h.check()
    return h


@pytest.mark.parametrize("k", [0, 3])
def test_the_boundary_changes_no_plan_for_the_same_arrivals(k) -> None:
    """The same arrivals per turn give the same plans and outputs with the boundary wait as without: it changes when a
    turn starts, not what the scheduler decides (the plan-diff harness's claim, on a clocked fake)."""
    specs = [Spec(rid=i, prompt_len=5 + 7 * i, max_tokens=6 + i, seed=0xB0D2_0000 + i, arrive=2 * i) for i in range(5)]
    off = _run(specs, False, mtp_k=k, max_num_seqs=3)
    on = _run(specs, True, mtp_k=k, max_num_seqs=3)
    assert json.dumps(on.plans) == json.dumps(off.plans)
    assert {r: v.tokens for r, v in on.results.items()} == {r: v.tokens for r, v in off.results.items()}
    assert on.core.phase_ns[P_BOUNDARY] > 0 and off.core.phase_ns[P_BOUNDARY] == 0


def _poll_work_attempt() -> bool:
    """One run: True when the forward was pending after the poll's work and the clock was tight."""
    spec = Spec(rid=1, prompt_len=8, max_tokens=6, seed=0xB0D3_0001)
    h = _timed(Harness(specs=[spec], hybrid=False), 30.0, 10.0)
    h.add(h.specs[1])
    h.core.step()  # the prefill in flight: its forward runs for 30 ms
    calls = []

    def poll() -> int:  # the first poll handles a message for 10 ms
        calls.append(1)
        if len(calls) == 1:
            time.sleep(0.010)
            return 10_000_000
        return 0

    t0 = time.perf_counter_ns()
    waited = h.core.await_boundary(poll)
    spun = time.perf_counter_ns() - t0
    assert waited == h.core.phase_ns[P_BOUNDARY]
    if len(calls) < 2:  # the process stalled until the forward ended: nothing was waited
        return False
    assert waited <= spun - 10_000_000
    return spun - 10_000_000 - 1_000_000 < waited


def test_the_polls_own_work_is_not_boundary_wait() -> None:
    """Messages the polls inside the wait handle (a request's admission, a grammar compile) are the engine's work, not
    the wait: `await_boundary` returns, and books, the spin's time less what the polls report. The clock is real: a
    preemption before the wait, or between the outer and inner clock reads, adds time to `spun` that `waited` cannot
    hold, so the lower bound (1 ms of call overhead) is retried; the upper bound holds on every run."""
    if not any(_poll_work_attempt() for _ in range(ATTEMPTS)):
        pytest.fail(f"no run in {ATTEMPTS} had the forward pending after the poll's work and a tight clock")


def test_no_wait_without_a_forward_end() -> None:
    """A runner that cannot tell (the fake without a sampling part, the Qwen runner) never makes the engine wait."""
    h = Harness(specs=[Spec(rid=1, prompt_len=6, max_tokens=4, seed=7)])
    h.add(h.specs[1])
    while h.core.busy:
        assert h.core.await_boundary(lambda: pytest.fail("polled")) == 0
        rep = h.core.step()
        if rep is not None:
            h.collect(rep)
    assert h.core.phase_ns[P_BOUNDARY] == 0


def test_derive_plan_at_boundary() -> None:
    off, why = derive_plan_at_boundary(forward_end=False, sample_ms=5.0, launch_ms=0.1)
    assert not off and "no forward's end" in why
    off, why = derive_plan_at_boundary(forward_end=True, sample_ms=0.0, launch_ms=0.0)  # memory_gb: no profile
    assert not off and "timed no split step" in why
    need = BOUNDARY_PLAN_MS + 1.4
    on, why = derive_plan_at_boundary(forward_end=True, sample_ms=need, launch_ms=1.4)
    assert on and why.startswith("on:") and f"{need:.2f} ms" in why
    off, why = derive_plan_at_boundary(forward_end=True, sample_ms=need - 0.01, launch_ms=1.4)
    assert not off and why.startswith("off:") and "would wait" in why
    # measured on a 600 W card: a lone step's sampling part 2.14-2.19 ms, its launch 1.16-1.66 ms
    assert derive_plan_at_boundary(forward_end=True, sample_ms=2.14, launch_ms=1.66)[0]


def test_the_switches_settings(tmp_path) -> None:
    """Always-split graphs default on with structured output, Gemma 4 and Qwen alike, off without it, which refuses
    them; boundary planning is resolved at startup, off unless set (unset here), and refused on Qwen (its breakable
    launches' host time) and without structured output."""
    from ..support.config_cases import _gemma, _model

    g, q = str(_gemma(tmp_path / "g")), str(_model(tmp_path, 248_320, 1))
    so, no_so = {"structured_output.enabled": True}, {"structured_output.enabled": False}
    split = "dev.runner.cuda.always_split"

    def derive(path: str, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": path, **kw}))

    for path in (g, q):
        cfg = derive(path, **no_so)
        assert cfg.dev.runner.cuda.always_split is False and cfg.reasons[split] == "off: structured output is off"
        assert cfg.engine.plan_at_boundary is None  # resolved at startup: off unless set
        with pytest.raises(ValueError, match="needs structured output"):
            derive(path, **no_so, **{split: True})
        cfg = derive(path, **so)
        assert cfg.dev.runner.cuda.always_split is True and cfg.reasons[split].startswith("on: structured output;")
        assert "forward graph and a sampling graph" in cfg.reasons[split]
        assert derive(path, **so, **{split: False}).dev.runner.cuda.always_split is False
        assert derive(path, **so, **{split: True}).dev.runner.cuda.always_split is True
    for kw in (so, {}):
        with pytest.raises(ValueError, match=r"Gemma 4 only on CUDA \(.*breakable graphs"):
            derive(q, **kw, **{"engine.plan_at_boundary": True})
    with pytest.raises(ValueError, match="needs structured output on CUDA"):  # at load, not after the weights
        derive(g, **no_so, **{"engine.plan_at_boundary": True})

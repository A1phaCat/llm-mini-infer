"""The grammar-aware heads' counters on /metrics (operations.md, "Metrics"). A runner counts the split steps (those
with MASKED rows) it sampled on the sparse head or on the dense one and why, the sizes of their masks' unions, and the
draft universe's tokens, steps and rebuilds (`Runner.head_counters`: host ints); the engine reads them into every
step's stats, and every frontend's /metrics exports them (llm_mini_infer:grammar_head_steps_total{path,reason},
llm_mini_infer:grammar_head_union_tokens, llm_mini_infer:draft_universe_tokens,
llm_mini_infer:draft_universe_steps_total, llm_mini_infer:draft_universe_rebuilds_total). No CUDA: a stand-in replaces
the fake runner's counters (GemmaRunner keeps the real ones); the fake-runner server's zeros are checked in
tests/api/test_api.py."""

from __future__ import annotations

import asyncio
import shutil
from bisect import bisect_left

import numpy as np

from llm_mini_infer.api.engine_client import EngineClient
from llm_mini_infer.config import Config
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import EngineStats, StepOutputs
from llm_mini_infer.engine.process import EngineLoop
from llm_mini_infer.metrics import Histogram, Metrics
from llm_mini_infer.runner.base import UNION_BUCKETS, HeadCounters
from llm_mini_infer.runner.fake import FakeRunner

from ..support.harness import Harness

BASE = 'model_name="m",engine="0"'
PATHS = (("sparse", "union"), ("dense", "off"), ("dense", "unconstrained"), ("dense", "over_cap"))  # path, reason
UNIONS = (16, 20, 40, 60, 100, 130, 200, 1500)  # 16 sits on a bound (le="16"); 1,500 is over the default cap


def bucket_counts(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """`HeadCounters.union_counts` of unions of these sizes: a union counts at bisect_left(UNION_BUCKETS, size)."""
    counts = [0] * (len(UNION_BUCKETS) + 1)
    for s in sizes:
        counts[bisect_left(UNION_BUCKETS, s)] += 1
    return tuple(counts)


def through_the_engine(monkeypatch, counters: HeadCounters | None) -> tuple[EngineStats, str, str]:
    """A runner's head counters (the fake's own when None) through a step's stats (`EngineLoop._stats`, across the IPC
    encoding) to a frontend's /metrics: the stats it decoded, its page before them and its page after."""
    h = Harness(specs=[])
    if counters is not None:
        monkeypatch.setattr(h.runner, "head_counters", lambda: counters)
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), EngineCore(h.sched, h.runner), path)
    try:
        e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)
        msg = StepOutputs(step=0, rids=e64, n_new=e32, tokens=e32, finish=np.zeros(0, np.uint8), first_rids=e64,
                          first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64, first_runner_ns=e64,
                          t_sent_ns=0, stats=loop._stats(None))
        stats = ipc.Decoder().decode(ipc.Encoder().encode(msg)).stats
        metrics = Metrics("m", 4096)
        client = EngineClient(path, 0, metrics)
        try:
            before = metrics.render().decode()
            client._on_stats(stats)
            after = metrics.render().decode()
        finally:
            asyncio.run(client.close())
    finally:
        loop.close()
        shutil.rmtree(path, ignore_errors=True)
    return stats, before, after


def test_a_runner_without_the_heads_has_none() -> None:
    """The fake keeps `Runner.head_counters`' default: nothing counted, no union buckets."""
    c = FakeRunner(vocab_size=1000, max_positions=256, mtp_k=3).head_counters()
    assert c == HeadCounters() and c.union_counts == () and c.universe == 0


def test_the_engine_carries_a_runners_head_counters_to_metrics(monkeypatch) -> None:
    """The steps per path and reason as one labelled counter, the unions' bucket counts as a histogram that renders
    exactly as observing the union sizes would, the universe's tokens as a gauge and its steps and rebuilds as
    counters; the unlabelled ones read 0 before the first stats, the others appear with them."""
    counts = bucket_counts(UNIONS)
    counters = HeadCounters(sparse=7, dense_off=0, dense_unconstrained=2, dense_over_cap=1, union_counts=counts,
                            union_sum=sum(UNIONS), universe=290, universe_steps=6, universe_rebuilds=4)
    stats, before, after = through_the_engine(monkeypatch, counters)
    assert (stats.grammar_head_sparse, stats.grammar_head_dense_off, stats.grammar_head_dense_unconstrained,
            stats.grammar_head_dense_over_cap) == (7, 0, 2, 1)
    assert stats.grammar_head_union_counts == list(counts) and stats.grammar_head_union_sum == 2066
    assert (stats.draft_universe_tokens, stats.draft_universe_steps, stats.draft_universe_rebuilds) == (290, 6, 4)
    for name, kind in (("grammar_head_steps_total", "counter"), ("grammar_head_union_tokens", "histogram"),
                       ("draft_universe_tokens", "gauge"), ("draft_universe_steps_total", "counter"),
                       ("draft_universe_rebuilds_total", "counter")):
        assert f"# TYPE llm_mini_infer:{name} {kind}" in before and f"# TYPE llm_mini_infer:{name} {kind}" in after
    for name in ("draft_universe_tokens", "draft_universe_steps_total", "draft_universe_rebuilds_total"):
        assert f"llm_mini_infer:{name}{{{BASE}}} 0.0" in before
    assert ("llm_mini_infer:grammar_head_steps_total{" not in before and
            "llm_mini_infer:grammar_head_union_tokens_" not in before)
    for (path, reason), n in zip(PATHS, (7, 0, 2, 1)):
        assert f'llm_mini_infer:grammar_head_steps_total{{{BASE},path="{path}",reason="{reason}"}} {float(n)}' in after
    ref = Histogram("llm_mini_infer:grammar_head_union_tokens", "", BASE, UNION_BUCKETS)
    for s in UNIONS:
        ref.observe(s)
    lines = ref.render()[2:]  # its samples, without HELP and TYPE
    assert len(lines) == len(UNION_BUCKETS) + 3 and all(line in after for line in lines)
    for line in (f'llm_mini_infer:grammar_head_union_tokens_bucket{{{BASE},le="16"}} 1',
                 f'llm_mini_infer:grammar_head_union_tokens_bucket{{{BASE},le="1024"}} 7',
                 f'llm_mini_infer:grammar_head_union_tokens_bucket{{{BASE},le="+Inf"}} 8',
                 f"llm_mini_infer:grammar_head_union_tokens_sum{{{BASE}}} 2066.0",
                 f"llm_mini_infer:grammar_head_union_tokens_count{{{BASE}}} 8",
                 f"llm_mini_infer:draft_universe_tokens{{{BASE}}} 290.0",
                 f"llm_mini_infer:draft_universe_steps_total{{{BASE}}} 6.0",
                 f"llm_mini_infer:draft_universe_rebuilds_total{{{BASE}}} 4.0"):
        assert line in after, line


def test_a_runner_without_the_heads_exports_zeros(monkeypatch) -> None:
    """Every series is there with zeros (the engine fills the union buckets of a runner that has none), so a
    dashboard's rates exist from the start; stats without the heads' fields leave the histogram without samples."""
    stats, _, after = through_the_engine(monkeypatch, None)
    assert stats.grammar_head_union_counts == [0] * (len(UNION_BUCKETS) + 1)
    for path, reason in PATHS:
        assert f'llm_mini_infer:grammar_head_steps_total{{{BASE},path="{path}",reason="{reason}"}} 0.0' in after
    for b in (*UNION_BUCKETS, "+Inf"):
        assert f'llm_mini_infer:grammar_head_union_tokens_bucket{{{BASE},le="{b}"}} 0' in after
    for line in (f"llm_mini_infer:grammar_head_union_tokens_sum{{{BASE}}} 0.0",
                 f"llm_mini_infer:grammar_head_union_tokens_count{{{BASE}}} 0",
                 f"llm_mini_infer:draft_universe_tokens{{{BASE}}} 0.0",
                 f"llm_mini_infer:draft_universe_steps_total{{{BASE}}} 0.0",
                 f"llm_mini_infer:draft_universe_rebuilds_total{{{BASE}}} 0.0"):
        assert line in after, line
    bare = EngineStats(running=0, waiting=0, kv_usage=0.0, state_usage=0.0, num_preemptions=0, num_aborts=0,
                       prompt_tokens=0, generation_tokens=0, step_tokens=0, step_cpu_ns=0, step_runner_calls_ns=0,
                       step_runner_ns=0, gpu_busy_ns=0, gpu_idle_ns=0)
    metrics = Metrics("m", 4096)
    path = ipc.ipc_dir()
    client = EngineClient(path, 0, metrics)
    try:
        client._on_stats(bare)
        page = metrics.render().decode()
    finally:
        asyncio.run(client.close())
        shutil.rmtree(path, ignore_errors=True)
    assert ("llm_mini_infer:grammar_head_union_tokens_" not in page and
            "llm_mini_infer:grammar_head_steps_total{" in page)

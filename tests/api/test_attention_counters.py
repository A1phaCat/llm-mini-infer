"""FlashInfer's attention-plan counters on /metrics (a split plan that a true KV length breaks gives rows wrong but
finite, which would show otherwise only as falling draft acceptance). `kernels.flashinfer.FlashInferAttention` counts
the plans it made again for a split whose chunk count an MTP row's true KV length could change (`replans`), the
decode-graph plans it kept although no safe split fit (`kept`) and those it made without a split (`unsplit`); a runner
sums them over its instances (`Runner.attention_counters`), the engine's stats carry them to every frontend's /metrics
(llm_mini_infer:attention_replans_total, llm_mini_infer:attention_plans_kept_total,
llm_mini_infer:attention_plans_unsplit_total), with the vision tower's plan waits
(llm_mini_infer:vision_plan_waits_total, `kernels.flashinfer.FlashInferVisionAttention.waits`). No CUDA: stand-ins hold
the counters of FlashInfer's instances; the fake-runner server's zeros are checked in tests/api/test_api.py."""

from __future__ import annotations

import asyncio
import shutil
from types import SimpleNamespace

import numpy as np

from llm_mini_infer.api.engine_client import EngineClient
from llm_mini_infer.config import Config
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import StepOutputs
from llm_mini_infer.engine.process import EngineLoop
from llm_mini_infer.metrics import Metrics
from llm_mini_infer.runner.base import AttentionCounters, sum_attention_counters
from llm_mini_infer.runner.cuda.gemma import GemmaRunner
from llm_mini_infer.runner.cuda.runner import CudaRunner
from llm_mini_infer.runner.fake import FakeRunner

from ..support.harness import Harness


def plans(replans: int, kept: int, unsplit: int) -> SimpleNamespace:
    """A FlashInfer attention instance's counters, as `FlashInferAttention` keeps them."""
    return SimpleNamespace(replans=replans, kept=kept, unsplit=unsplit)


def test_the_sum_counts_each_flashinfer_instance_once() -> None:
    a, b = plans(3, 1, 0), plans(4, 0, 2)
    torch_ref = SimpleNamespace(pools=None)  # a torch reference or attn512: no counters
    assert sum_attention_counters([a, None, b, a, torch_ref]) == AttentionCounters(replans=7, kept=1, unsplit=2)
    assert sum_attention_counters(iter([b])) == AttentionCounters(replans=4, unsplit=2)  # a runner passes a generator
    assert sum_attention_counters([]) == sum_attention_counters([None, torch_ref]) == AttentionCounters()


def test_the_cuda_runners_sum_the_instances_they_own() -> None:
    """Gemma 4's sliding layers (the target's, the drafter's and the image spans' instances; attn512, the full layers',
    has no counters) and Qwen's target and drafter's chain, read from stand-ins for the runners' attributes (the
    runners need CUDA). Before `configure` and after `shutdown` a runner holds none."""
    w, dw, wb = plans(2, 0, 0), plans(5, 1, 0), plans(0, 0, 1)
    gemma = SimpleNamespace(attn_w=w, attn_dw=dw, attn_wb=wb, attn_f=SimpleNamespace(), attn_df=SimpleNamespace())
    assert GemmaRunner.attention_counters(gemma) == AttentionCounters(replans=7, kept=1, unsplit=1)
    assert GemmaRunner.attention_counters(SimpleNamespace(attn_w=w, attn_wb=None)) == AttentionCounters(replans=2)
    ref = SimpleNamespace()  # the torch references: the drafter shares the target's
    assert GemmaRunner.attention_counters(SimpleNamespace(attn_w=ref, attn_dw=ref, attn_wb=None)) == \
        AttentionCounters()
    assert GemmaRunner.attention_counters(SimpleNamespace()) == AttentionCounters()
    attn, chain = plans(1, 0, 3), plans(4, 2, 0)
    assert CudaRunner.attention_counters(SimpleNamespace(attn=attn, chain_attn=chain)) == \
        AttentionCounters(replans=5, kept=2, unsplit=3)
    assert CudaRunner.attention_counters(SimpleNamespace(attn=attn, chain_attn=None)) == \
        AttentionCounters(replans=1, unsplit=3)  # MTP off
    assert CudaRunner.attention_counters(SimpleNamespace()) == AttentionCounters()
    # the vision tower's plan waits (`FlashInferVisionAttention.waits`), from either runner's image encoder; its torch
    # attention has none
    tower = SimpleNamespace(attn=SimpleNamespace(waits=3))
    assert GemmaRunner.attention_counters(SimpleNamespace(attn_w=w, img=tower)) == \
        AttentionCounters(replans=2, vision_waits=3)
    assert CudaRunner.attention_counters(SimpleNamespace(img=tower)) == AttentionCounters(vision_waits=3)
    assert CudaRunner.attention_counters(SimpleNamespace(img=SimpleNamespace(attn=SimpleNamespace()))) == \
        AttentionCounters()


def test_the_fake_runner_has_none() -> None:
    assert FakeRunner(vocab_size=1000, max_positions=256, mtp_k=3).attention_counters() == AttentionCounters()


def test_the_engine_carries_a_runners_counters_to_metrics(monkeypatch) -> None:
    """The engine reads the runner's counters into a step's stats (`EngineLoop._stats`), which cross the IPC encoding
    to a frontend; its /metrics exports them as cumulative counters."""
    h = Harness(specs=[])
    monkeypatch.setattr(h.runner, "attention_counters",
                        lambda: AttentionCounters(replans=5, kept=1, unsplit=2, vision_waits=4))
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), EngineCore(h.sched, h.runner), path)
    try:
        e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)
        msg = StepOutputs(step=0, rids=e64, n_new=e32, tokens=e32, finish=np.zeros(0, np.uint8), first_rids=e64,
                          first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64, first_runner_ns=e64,
                          t_sent_ns=0, stats=loop._stats(None))
        stats = ipc.Decoder().decode(ipc.Encoder().encode(msg)).stats
        assert (stats.attention_replans, stats.attention_plans_kept, stats.attention_plans_unsplit,
                stats.vision_plan_waits) == (5, 1, 2, 4)
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
    for name, n in (("attention_replans", 5), ("attention_plans_kept", 1), ("attention_plans_unsplit", 2),
                    ("vision_plan_waits", 4)):
        assert f"# TYPE llm_mini_infer:{name}_total counter" in after
        assert f'llm_mini_infer:{name}_total{{model_name="m",engine="0"}} 0.0' in before
        assert f'llm_mini_infer:{name}_total{{model_name="m",engine="0"}} {float(n)}' in after


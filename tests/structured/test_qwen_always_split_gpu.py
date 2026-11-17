"""Always-split graphs in the Qwen runner on CUDA (`CudaRunner`, `dev.runner.cuda.always_split`; architecture.md, "CUDA
graphs"): BF16 with our kernels and CUDA graphs on `qwen_gpu`'s tiny Qwen with its MTP layer, and the block-FP8 one
with a block-FP8 MTP layer (the real FP8 checkpoints' shape), through the scheduler with its grammar switches (the
forced prefix, the EOS-free finish) and the engine loop, the grammar-aware heads on.

- **Bit for bit:** with always-split graphs every step with rows runs a forward graph and a sampling graph, a prompt's
  chunk without sample rows included, and no fused graph is captured; every step's tokens, counts and drafts (accepted
  or not), the finishes and the drafts accepted are the fused graphs' exactly (the same kernels at the same shapes: the
  sampling graph reads the forward's hidden states from a copy), greedy and seeded, with and without grammar masks,
  MTP on and off.
- **No forward end:** a split step's sampling part is timed (`StepResult.gpu_sample_ns`; a fused step has none), but
  the Qwen runner reports no forward's end (`Capabilities.forward_end`: `engine.plan_at_boundary` stays Gemma 4's).

Gemma 4's twin is test_always_split_gpu.py. CUDA only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar
from ..support.mtp_common import EOS, seq
from ..support.qwen_gpu import VOCAB, gpu_run
from ..support.qwen_gpu import tiny as tiny  # the fixture: the tiny Qwen with its MTP layer, kernel-sized, in BF16
from ..support.qwen_mtp import BUDGET, Run, requests
from ..support.tiny_qwen import make_tiny_fp8_model

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
LEAD = 200_000_000  # a device sleep of about 70-100 ms ahead of a step (runner.cuda.runner._LEAD_CYCLES)


@pytest.fixture(scope="module")
def fp8(tmp_path_factory):
    """The tiny Qwen in block-FP8 with its block-FP8 MTP layer, kernel-sized."""
    return make_tiny_fp8_model(tmp_path_factory.mktemp("qwenfp8gpu"), seed=5, mtp=True, gpu=True)[0]


def run(model, k: int, reqs: list[dict], *, split: bool, temperature: float = 0.0) -> Run:
    """The requests through the engine loop; `steps`: per launched step, (split, rows, sample rows); `results`: per
    waited step, its `StepResult`."""
    r = gpu_run(model, k, budget=BUDGET, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), forced_prefix=True,
                eos_free_finish=True, sparse_head=True, draft_universe=k > 0, always_split=split)
    r.steps, r.results = [], []
    launch, wait = r.r.launch_forward, r.r.wait

    def spy(plan, *a, **kw):
        h = launch(plan, *a, **kw)
        r.steps.append((h.split is not None, h.n_sched, h.n_sample))
        return h

    def waited(h):
        res = wait(h)
        r.results.append(res)
        return res

    r.r.launch_forward, r.r.wait = spy, waited
    for q in reqs:
        r.add(q["rid"], q["prompt"], max_tokens=q["max_tokens"], temperature=temperature, ignore_eos=False,
              grammar=q.get("grammar", ""))
    r.run()
    assert r.sched.grammars.errors == 0
    return r


def same(fused: Run, split: Run, constrained: bool) -> None:
    """The split run is the fused run's, bit for bit (module docstring); both runners shut down after."""
    assert split.out == fused.out and split.finish == fused.finish
    assert (split.sched.spec_drafts, split.sched.spec_accepted) == (fused.sched.spec_drafts, fused.sched.spec_accepted)
    assert [s[1:] for s in split.steps] == [s[1:] for s in fused.steps]  # the same plans
    assert len(split.results) == len(fused.results)
    for a, b in zip(split.results, fused.results):  # every step's tokens, counts and drafts
        assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.n_out, b.n_out)
        assert (a.drafts is None) == (b.drafts is None) and (a.drafts is None or np.array_equal(a.drafts, b.drafts))
    assert all(s for s, n, _ in split.steps if n) and any(n and not ns for _, n, ns in split.steps)
    assert any(s for s, _, _ in fused.steps) == constrained  # fused, only the steps with masks split
    assert split.sizes == fused.sizes  # the same graphs' kinds and buckets
    assert {"decode", "break"} <= set(split.modes) and "eager" not in split.modes, split.modes
    assert split.r.head_counters() == fused.r.head_counters()
    assert not [key for key in split.r._graphs if key[0] in ("decode", "break")], "a fused graph was captured"
    gs, gf = split.r.stats["graphs"], fused.r.stats["graphs"]
    assert gs["fused"] == 0 and gf["fused"] > 0 and gs["decode"] == gf["decode"] and gs["break"] == gf["break"]
    assert gs["captured"] == gf["captured"] - gf["fused"]
    assert gs["sampling_variants"] == gf["sampling_variants"] == split.r.gs.variants
    assert all(res.gpu_sample_ns > 0 for res in split.results if res.n_out.any())
    if not constrained:  # no masks: the fused run's steps are fused graphs, with no sampling part
        assert all(res.gpu_sample_ns == 0 for res in fused.results)
    split.r.shutdown()
    fused.r.shutdown()


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("constrained", [False, True])
def test_always_split_graphs_give_the_fused_graphs_tokens(tiny, k, constrained) -> None:
    for temperature in (0.0, 0.8):
        reqs = requests(constrained)
        fused = run(tiny, k, reqs, split=False, temperature=temperature)
        split = run(tiny, k, reqs, split=True, temperature=temperature)
        same(fused, split, constrained)
        assert split.r.gs.variants == (["s", "q", "p"] if k else ["s", "q"])
        if constrained:  # forced prefixes, and split steps on the sparse heads (`_q`, under MTP `_p`)
            c = split.r.head_counters()
            assert split.sched.forced_requests > 0 and c.sparse > 0 and (c.universe_steps > 0) == (k > 0), c


@pytest.mark.gpu
@pytest.mark.cutlass(reason="needs the CUTLASS library (block-FP8 GEMMs)")
@pytest.mark.parametrize("constrained", [False, True])
def test_always_split_graphs_on_fp8_give_the_fused_graphs_tokens(fp8, constrained) -> None:
    """The real FP8 checkpoints' shape: the FP8 GEMMs and the quantizing producers in both parts' graphs, k = 3."""
    for temperature in (0.0, 0.8):
        reqs = requests(constrained)
        fused = run(fp8, 3, reqs, split=False, temperature=temperature)
        split = run(fp8, 3, reqs, split=True, temperature=temperature)
        same(fused, split, constrained)
        assert split.r.stats["fp8_projections"] > 0


def test_a_split_step_reports_no_forward_end(tiny) -> None:
    """A split step queued behind a device sleep: its forward has not run when its turn ends, yet the Qwen runner
    reports nothing to wait for (`forward_pending`; `engine.plan_at_boundary` stays Gemma 4's: a breakable graph's
    eager GDN and attention breaks take host time the step in flight's outputs would wait for) and no split's costs
    (`Capabilities.sample_step_ms`, `launch_step_ms`); its sampling part is timed all the same."""
    r = gpu_run(tiny, 3, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), always_split=True, check=False)
    caps = r.r.capabilities()
    assert not caps.forward_end and caps.sample_step_ms == 0 and caps.launch_step_ms == 0
    r.add(0, seq(20, 1), max_tokens=40)
    for _ in range(3):
        r.core.step()
    torch.cuda.synchronize()
    torch.cuda._sleep(LEAD)  # the next step's forward waits behind it
    r.core.step()
    h = r.core._inflight[1]
    assert h.fwd_end is not None and not h.fwd_end.query()  # a split step whose forward is still to run
    assert r.r.forward_pending(h) is False
    torch.cuda.synchronize()
    r.run()
    assert r.core.gpu_sample_ns > 0
    r.r.shutdown()

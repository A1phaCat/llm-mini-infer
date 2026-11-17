"""Always-split graphs and planning at the boundary on CUDA (architecture.md, "CUDA graphs", "Planning at the
forward/sampling boundary"): GemmaRunner in BF16 with our kernels and CUDA graphs (`test_gemma4_mtp_gpu`'s tiny pair),
through the scheduler in window mode and the engine loop.

- **Bit for bit:** with always-split graphs every step runs a forward graph and a sampling graph, and no fused graph is
  captured; the tokens, finishes and accepted drafts are the fused graphs' exactly (the same kernels at the same
  shapes: the sampling graph reads the forward's hidden states from a copy), greedy and seeded, with and without
  grammar masks, MTP on and off.
- **The forward's end:** a split step reports its forward running until its forward-end event (`forward_pending`), and
  its sampling part's device time (`StepResult.gpu_sample_ns`); a fused step reports neither.
- **The memory profile** times a lone masked decode step's split for the engine's derivation
  (`Capabilities.sample_step_ms`, `launch_step_ms`)."""

from __future__ import annotations

import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar, spec
from ..support.gemma_gpu import W, cuda_runner
from ..support.gemma_gpu import pair as pair  # the fixture: the tiny pair in BF16
from ..support.gemma_mtp import Run
from ..support.mtp_common import EOS, seq

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
VOCAB = 512
LEAD = 200_000_000  # a device sleep of about 70-100 ms ahead of a step (runner.cuda.runner._LEAD_CYCLES)


def requests(constrained: bool, n: int = 6) -> list[dict]:
    out = []
    for i in range(n):
        q = {"rid": i, "prompt": seq(9 + 13 * i, i + 1), "max_tokens": 24}
        if constrained and i % 2 == 0:  # half under a JSON-like grammar, half free, in the same steps
            q["grammar"] = spec(i, 4, 10, one_in=16, universe=128)
        out.append(q)
    return out


def run(pair, k: int, reqs: list[dict], *, split: bool, temperature: float = 0.0) -> Run:
    grammar = FakeGrammar(VOCAB, (EOS,))
    r = Run(pair, k, make=lambda t, pc, **rk: cuda_runner(t, pc, True, **rk), window=W, wps=16,
            grammars=Grammars(grammar), caching=False, sparse_head=True, draft_universe=k > 0, always_split=split)
    r.modes, r.results = [], []
    mode, wait = r.r._mode, r.r.wait

    def spy_mode(lay, split=False):
        m = mode(lay, split)
        r.modes.append((m[0], split))
        return m

    def spy_wait(h):
        res = wait(h)
        r.results.append(res)
        return res

    r.r._mode, r.r.wait = spy_mode, spy_wait
    for q in reqs:
        r.add(q["rid"], q["prompt"], max_tokens=q["max_tokens"], temperature=temperature, ignore_eos=False,
              grammar=q.get("grammar", ""))
    r.run()
    assert r.sched.grammars.errors == 0
    return r


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("constrained", [False, True])
def test_always_split_graphs_give_the_fused_graphs_tokens(pair, k, constrained) -> None:
    for temperature in (0.0, 0.8):
        fused = run(pair, k, requests(constrained), split=False, temperature=temperature)
        split = run(pair, k, requests(constrained), split=True, temperature=temperature)
        assert split.out == fused.out and split.finish == fused.finish, (k, constrained, temperature)
        assert split.sched.spec_accepted == fused.sched.spec_accepted and split.sched.spec_drafts == \
            fused.sched.spec_drafts
        assert [m for m, _ in split.modes] == [m for m, _ in fused.modes]  # the same graphs' kinds and buckets
        assert all(s for _, s in split.modes) and "decode" in {m for m, _ in split.modes}
        assert split.r.head_counters() == fused.r.head_counters()
        g = split.r._graphs
        assert not [key for key in g if key[0] in ("decode", "break")], "a fused graph was captured"
        assert split.r.stats["graphs"]["fused"] == 0 and fused.r.stats["graphs"]["fused"] > 0
        assert split.r.stats["graphs"]["decode"] == fused.r.stats["graphs"]["decode"]
        assert split.r.stats["graphs"]["captured"] == fused.r.stats["graphs"]["captured"] - \
            fused.r.stats["graphs"]["fused"]
        assert all(res.gpu_sample_ns > 0 for res in split.results if res.n_out.any())
        if not constrained:  # no masks: the fused run's steps are fused graphs, with no sampling part
            assert all(res.gpu_sample_ns == 0 for res in fused.results)
        split.r.shutdown()
        fused.r.shutdown()


def test_a_split_step_reports_its_forward_running_until_it_ends(pair) -> None:
    """A step queued behind a device sleep: after its turn its forward has not started, so it reports it running; once
    the device is done, not. A fused step (always-split off, no masks) reports nothing to wait for."""
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    def unchecked(model_dir, pc, **kw):  # `check` synchronizes the device at every launch_sample (the host mirror)
        r = GemmaRunner(model_dir, device="cuda", max_num_seqs=pc.max_num_seqs,
                        max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len,
                        memory_gb=2.0, **kw)
        r.configure(pc)
        return r

    for split in (True, False):
        r = Run(pair, 3, make=unchecked, window=W, wps=16, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))),
                caching=False, always_split=split)
        assert r.r.capabilities().forward_end
        r.add(0, seq(20, 1), max_tokens=40)
        for _ in range(3):
            r.core.step()
        torch.cuda.synchronize()
        torch.cuda._sleep(LEAD)  # the next step's forward waits behind it
        r.core.step()
        h = r.core._inflight[1]
        assert r.r.forward_pending(h) == split
        polls = 0
        while r.r.forward_pending(h):  # until the forward ends: well before the step's results
            polls += 1
        assert (polls > 0) == split and not h.events[3].query()  # the sampling part still to run
        torch.cuda.synchronize()
        assert not r.r.forward_pending(h)
        r.run()
        r.r.shutdown()


def test_the_memory_profile_times_a_lone_decode_steps_split(pair) -> None:
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    target, assistant = pair
    r = GemmaRunner(target, device="cuda", max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, mtp_k=3,
                    assistant=assistant, sparse_head=True, draft_universe=True, always_split=True)
    caps = r.capabilities()
    assert caps.forward_end and caps.sample_step_ms > 0 and caps.launch_step_ms > 0, (caps.sample_step_ms,
                                                                                      caps.launch_step_ms)
    assert r.stats["sample_step_ms"] == caps.sample_step_ms
    r.shutdown()

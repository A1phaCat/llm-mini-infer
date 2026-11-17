"""The forced prefix in the Qwen runner on CUDA (`dev.structured_output.forced_prefix`; architecture.md, "Grammar-aware
heads"): BF16 with our kernels and CUDA graphs on `qwen_gpu`'s tiny Qwen with its MTP layer, through the scheduler and
the engine loop, in structured output's configuration (the heads, always-split graphs, the EOS-free finish). A grammar's
forced tokens, and the sample after them, run in the request's final prefill row (FlashInfer's chunked GDN prefill and
its attention plan) instead of decode and verify rows (the GDN decode and verify kernels), so their bits move as any
change of a position's row kind moves them: the tokens agree with the sampled path's up to a near tie of transformers'
fp32 logits (`qwen_gpu.agree`), where on a CPU they are equal (test_cuda_structured.py). CUDA only."""

from __future__ import annotations

import pytest

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar
from ..support.gemma_heads_gpu import forced_json
from ..support.mtp_common import EOS
from ..support.qwen_gpu import MANY, VOCAB, agree, gpu_run, reference
from ..support.qwen_gpu import tiny as tiny  # the fixture: the tiny Qwen with its MTP layer, kernel-sized, in BF16

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def run(model, k: int, forced: bool):
    """24 constrained requests on one cached prefix (`qwen_gpu.MANY`): the first computes it, the others hit its
    checkpoint, their final chunks taking their grammars' forced tokens with `forced`."""
    r = gpu_run(model, k, max_num_seqs=32, budget=256, num_pages=2048, caching=True, ckpt_slots=8,
                grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), forced_prefix=forced, eos_free_finish=True,
                sparse_head=True, draft_universe=k > 0, always_split=True)
    r.add(0, MANY[0], max_tokens=40, grammar=forced_json(0), ignore_eos=False)
    r.run()
    for i, p in enumerate(MANY[1:24], 1):
        r.add(i, p, max_tokens=40, grammar=forced_json(i), ignore_eos=False)
    r.run()
    r.r.shutdown()
    assert r.sched.grammars.errors == 0
    return r


@pytest.mark.parametrize("k", [0, 3])
def test_the_forced_prefix_gives_the_sampled_paths_tokens_up_to_near_ties(tiny, k) -> None:
    """Forced and sampled, in graphs: every request's tokens agree up to a near tie of the reference (the forced tokens
    included: they are the grammar's alone either way), the finishes too where the tokens agree."""
    sampled = run(tiny, k, False)
    forced = run(tiny, k, True)
    f = forced.sched
    assert f.forced_requests >= 20 and f.forced_tokens == 2 * f.forced_requests and sampled.sched.forced_tokens == 0
    assert "decode" in forced.modes and "break" in forced.modes and "eager" not in forced.modes
    grammars = {i: forced_json(i) for i in sampled.out}
    ref = reference(tiny, sampled.out, sampled.prompts)
    n = sum(len(o) for o in sampled.out.values())
    assert agree(forced.out, sampled.out, ref, sampled.prompts, grammars) > 0.8 * n
    assert all(forced.finish[i] == sampled.finish[i] for i in sampled.out if forced.out[i] == sampled.out[i])

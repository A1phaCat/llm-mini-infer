"""The Qwen runner (`CudaRunner`) on CUDA for its GPU tests: BF16 with our kernels, FlashInfer's attention and GDN
prefill, the MTP layer and CUDA graphs (decode graphs, breakable prefill graphs, a split step's forward and sampling
graphs), through the scheduler and the engine loop (`qwen_mtp.Run`) under grammars (`fake_grammar`), on a tiny Qwen
with the dims our kernels take (`tiny_qwen.GPU_DIMS`). Not a test module: the GPU tests import it (Gemma 4's twins are
gemma_gpu's and gemma_heads_gpu's helpers).

The weights are saved in BF16, so transformers in fp32 runs exactly the weights we run. BF16 rounds differently with the
row shapes (a step's rows and buckets, the sparse head's columns), so two runs agree token for token up to a near tie:
where they part, transformers' fp32 logits put the two best tokens there (of those the request's grammar allowed) within
twice a BF16 run's error (`NEAR`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .fake_grammar import FakeGrammar
from .mtp_common import EOS, seq
from .qwen_mtp import Run
from .tiny_qwen import cuda_runner, hf_logits, make_tiny_model

VOCAB = 512
NEAR = 0.3  # twice the BF16 runs' largest logit error against transformers in fp32 (`test_qwen_heads_gpu`: < 0.15)
SHARED = seq(72, 99)  # a shared system prompt: the first request's run leaves a checkpoint at 64 the others hit
MANY = [SHARED + seq(3 + (i * 7) % 20, 10 + i) for i in range(24)]  # each with its own tail


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    """The tiny Qwen with its MTP layer, kernel-sized, in BF16."""
    return make_tiny_model(tmp_path_factory.mktemp("qwengpu"), seed=4, mtp=True, gpu=True, bf16=True)


def gpu_run(model, k: int, graphs: bool = True, **kw) -> Run:
    """`Run` over a CUDA runner (`tiny_qwen.cuda_runner`); `modes`: each launched step's body (`CudaRunner._mode`: a
    decode graph, a breakable graph or eager) and `sizes` its (kind, bucket)."""
    r = Run(model, k, make=lambda m, pc, **rk: cuda_runner(m, pc, graphs, **rk), **kw)
    r.modes, r.sizes = [], []
    mode = r.r._mode

    def spy(lay, split=False):
        m = mode(lay, split)
        r.modes.append(m[0])
        r.sizes.append(m)
        return m

    r.r._mode = spy
    return r


def reference(model, outs: dict[int, list[int]], prompts: dict[int, list[int]]) -> dict[int, torch.Tensor]:
    """Transformers' fp32 logits over each request's prompt and tokens (rid -> [T, V])."""
    return {rid: hf_logits(model, prompts[rid] + o) for rid, o in outs.items()}


def agree(got: dict[int, list[int]], want: dict[int, list[int]], ref: dict[int, torch.Tensor],
          prompts: dict[int, list[int]], grammars: dict[int, str] | None = None) -> int:
    """`got` follows `want` (rid -> tokens) up to a near tie of the reference logits (`want`'s sequences, teacher
    forced): where they part, the reference's two best tokens (among those the request's grammar in `grammars`
    allowed there) are within `NEAR`. Returns how many tokens agree before the partings, every request together."""
    fake = FakeGrammar(VOCAB, (EOS,))
    same = 0
    for rid, w in want.items():
        g, start = got[rid], len(prompts[rid])
        i = next((j for j, (x, y) in enumerate(zip(g, w)) if x != y), None)
        if i is None:
            assert len(g) == len(w), rid
            same += len(w)
            continue
        logits = ref[rid][start + i - 1].clone()
        if grammars is not None and grammars.get(rid):
            m = fake.instance(fake.compile(grammars[rid]))
            assert m.consume_tokens(w[:i]), rid
            logits[~torch.as_tensor(m.allowed())] = float("-inf")
        top = logits.topk(2).values
        assert (top[0] - top[1]).item() < NEAR, (rid, i, g[i], w[i], (top[0] - top[1]).item())
        same += i
    return same

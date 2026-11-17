"""GemmaRunner on CUDA: BF16, our kernels, FlashInfer's sliding window and CUDA graphs, against transformers in fp32
on the same BF16 weights, on a tiny model with the real head dims (sliding 256, full 512 with k_eq_v); NONFINITE on
rows that do not sample, eagerly and in the graphs. CUDA only."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from llm_mini_infer.runner.base import NONFINITE

from ..support.gemma_gpu import cuda_runner
from ..support.tiny_gemma import GREEDY, GDriver, gemma_pools, hf_logits, make_tiny_gemma, poisoned_rope, spy_modes
from ..support.token_ids import ids

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
W = 40
POISON = 30  # the position whose sliding-layer RoPE the NaN test makes NaN (`poisoned_rope`)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """BF16 weights (saved as such), so transformers in fp32 runs exactly the weights we run. The q and k norms' weights
    are scaled to a quarter, as a trained model keeps its attention logits moderate (the scale is 1.0): near 1, the
    random model's q·k over 256-512 dims has a spread of ~16, and the BF16 rounding of q and k alone moves its logits
    by ~1 (measured on the RTX PRO 6000: transformers' own ops in BF16 as far off as ours)."""
    from safetensors.torch import load_file, save_file

    path = make_tiny_gemma(tmp_path_factory.mktemp("g4gpu"), window=W, gpu=True, seed=3)
    f = path / "model.safetensors"
    sd = load_file(f)
    sd = {k: v * 0.25 if k.endswith(("q_norm.weight", "k_norm.weight")) else v for k, v in sd.items()}
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}, f)
    return path


def cuda_logits(tiny, steps, pc, **kw) -> tuple[torch.Tensor, list[int]]:
    """Our logits at every position of one slot's steps (prompt, then the sampled tokens) and the tokens."""
    drv = GDriver(cuda_runner(tiny, pc, graphs=False, **kw), pc, window=W)
    res = drv.run(steps)
    gen = [int(r.tokens[0, 0]) for r in res if len(r.n_out) and r.n_out[0]]
    n = max(p for (_, p) in drv.logits) + 1
    drv.r.shutdown()
    return torch.stack([drv.logits[(1, p)] for p in range(n)]), gen


def chunked_steps(seq: list[int], n_dec: int) -> list[dict]:
    """Prefill in chunks of 50, 63 and 37 (rows start mid-window and mid-page), then `n_dec` decode steps."""
    steps = [dict(rows=[(1, seq[:50], 0, False)], admit=[(1, GREEDY)]),
             dict(rows=[(1, seq[50:113], 50, False)]), dict(rows=[(1, seq[113:], 113, True)])]
    return steps + [dict(rows=[(1, None, len(seq) + i, True)]) for i in range(n_dec)]


def test_fp32_matches_transformers_exactly(tiny) -> None:
    """The runner's plumbing on CUDA (tables, both pools, window release, decode from device input) in fp32 with the
    torch reference ops: transformers' logits to fp32 noise."""
    seq = ids(150, 1)
    pc = gemma_pools(budget=64, page_size=16, num_pages=40)
    ours, gen = cuda_logits(tiny, chunked_steps(seq, 6), pc, attention="torch", fused_ops=False, dtype=torch.float32)
    ref = hf_logits(tiny, seq + gen)
    assert (ours - ref[:len(ours)]).abs().max().item() < 2e-3


def test_eager_logits_match_transformers(tiny) -> None:
    """Our kernels (fused ops, FlashInfer's window, the head-512 kernel) in BF16 against transformers in fp32: within
    the BF16 noise of the torch reference path in BF16 (the same model, transformers' ops), the argmax equal where
    transformers' top two are apart."""
    from llm_mini_infer.kernels import ops

    seq = ids(150, 1)
    pc = gemma_pools(budget=64, page_size=16, num_pages=40)
    ours, gen = cuda_logits(tiny, chunked_steps(seq, 10), pc)
    ref = hf_logits(tiny, seq + gen)[:len(ours)]
    base, _ = cuda_logits(tiny, chunked_steps(seq, 10), pc, attention="torch", fused_ops=False)
    ops.enabled(True)
    err, noise = (ours - ref).abs(), (base - ref).abs()
    assert err.max().item() < 0.15 and err.mean().item() < 1.2 * noise.mean().item(), \
        (err.max().item(), err.mean().item(), noise.mean().item())
    top2 = ref.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 0.2
    assert torch.equal(ours.argmax(-1)[clear], ref.argmax(-1)[clear])


def test_graphs_equal_eager(tiny) -> None:
    """The same steps with CUDA graphs (decode graphs, breakable prefill graphs) and eagerly: the same tokens."""
    seq, other = ids(90, 2), ids(37, 3)
    toks = []
    for graphs in (False, True):
        pc = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(tiny, pc, graphs), pc, window=W)
        drv.r.logits_hook = None
        steps = [dict(rows=[(1, seq, 0, True), (2, other, 0, True)], admit=[(1, GREEDY), (2, GREEDY)])]
        steps += [dict(rows=[(1, None, 90 + i, True), (2, None, 37 + i, True)]) for i in range(20)]
        res = drv.run(steps)
        toks.append(np.stack([r.tokens[:, 0] for r in res]))
    assert np.array_equal(toks[0], toks[1])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_rows_that_do_not_sample_report_nan(tiny, graphs, masked) -> None:
    """tests/models/test_gemma4_runner.py's test of the same name with our kernels in BF16, eagerly or in the breakable
    graphs, which compute the flags (every step runs one: the mode spied), with `masked` their split graphs. The
    sliding layers' RoPE is NaN at position 30. The first step samples nothing: slot 1's chunk [0, 50) computes
    position 30 and reports it, slot 3's [0, 20) and slot 5's one token report nothing. The second: slot 1's next chunk
    reads the NaN and slot 4's [0, 40) computes it, both report it; slot 2's prompt of 25 tokens samples as without the
    NaN, and slot 3's next token (a row the step orders first) reports nothing. With the table restored, the next
    step's rows report nothing, and a runner that never saw the NaN reports nothing at all."""
    a, b, c, d, e = ids(80, 11), ids(25, 12), ids(40, 13), ids(40, 14), ids(45, 15)
    steps = [dict(rows=[(1, a[:50], 0, False), (3, c[:20], 0, False), (5, e[:1], 0, False)],
                  admit=[(1, GREEDY), (3, GREEDY), (5, GREEDY)]),
             dict(rows=[(1, a[50:], 50, False), (4, d, 0, False), (2, b, 0, True), (3, c[20:21], 20, False)],
                  admit=[(2, GREEDY), (4, GREEDY)])]
    after = [dict(rows=[(2, None, 25, True), (5, e[1:], 1, False), (3, c[21:], 21, True)])]

    def run(poison: bool) -> tuple[list, list[str]]:
        pc = gemma_pools(max_num_seqs=6, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(tiny, pc, graphs), pc, window=W)
        drv.r.logits_hook = None
        modes = spy_modes(drv.r)
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            res = drv.run(steps, masked=masked)
        res += drv.run(after, masked=masked)
        drv.r.shutdown()
        return res, modes

    (s1, s2, s3), modes = run(True)
    assert modes == ["break" if graphs else "eager"] * 3
    assert s1.status.tolist() == [NONFINITE, 0, 0] and not s1.n_out.any()
    assert s2.status.tolist() == [NONFINITE, NONFINITE, 0, 0] and s2.n_out.tolist() == [0, 0, 1, 0]
    assert s3.status is None
    clean, _ = run(False)
    assert all(r.status is None for r in clean)
    for x, y in zip((s1, s2, s3), clean):  # every token that samples
        assert np.array_equal(x.tokens, y.tokens) and np.array_equal(x.n_out, y.n_out)

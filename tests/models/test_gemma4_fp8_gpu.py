"""GemmaRunner on CUDA with a block-FP8 checkpoint: the CUTLASS GEMMs and the quantizing producers in BF16 and inside
CUDA graphs, on a tiny model with the real head dims in RedHat's compressed-tensors format, against transformers in
fp32 on the dequantized weights; NONFINITE on a refine's PRECISE pieces, none on FP8 rows. CUDA and the CUTLASS
library only."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import fp8, ops
from llm_mini_infer.runner.base import NONFINITE

from ..support.gemma_gpu import cuda_runner
from ..support.tiny_gemma import GREEDY, GDriver, gemma_pools, hf_logits, make_tiny_gemma_fp8, poisoned_rope, spy_modes
from ..support.token_ids import ids

pytestmark = [pytest.mark.gpu,
              pytest.mark.cutlass(reason="needs a CUDA GPU, the kernel library and the CUTLASS library")]
W = 40
POISON = 30  # the position whose sliding-layer RoPE the NaN test makes NaN (`poisoned_rope`)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma_fp8(tmp_path_factory.mktemp("g4fp8gpu"), seed=5, gpu=True)


def steps_of(seq: list[int], n_dec: int) -> list[dict]:
    steps = [dict(rows=[(1, seq[:50], 0, False)], admit=[(1, GREEDY)]),
             dict(rows=[(1, seq[50:113], 50, False)]), dict(rows=[(1, seq[113:], 113, True)])]
    return steps + [dict(rows=[(1, None, len(seq) + i, True)]) for i in range(n_dec)]


def logits(model_dir, steps, pc, **kw) -> tuple[torch.Tensor, list[int]]:
    drv = GDriver(cuda_runner(model_dir, pc, graphs=False, **kw), pc, window=W)
    res = drv.run(steps)
    gen = [int(r.tokens[0, 0]) for r in res if len(r.n_out) and r.n_out[0]]
    n = max(p for (_, p) in drv.logits) + 1
    drv.r.shutdown()
    return torch.stack([drv.logits[(1, p)] for p in range(n)]), gen


def test_fp8_logits_within_the_torch_paths_noise(tiny) -> None:
    """Our FP8 path (the fused producers, the CUTLASS GEMMs, our attention kernels) against transformers in fp32 on the
    dequantized weights: as close as the same runner on the torch references (BF16 ops, `quant_ref`, the dequantized
    product in fp32), which differ from ours only in rounding order."""
    fp8_dir, deq_dir = tiny
    seq = ids(150, 1)
    pc = gemma_pools(budget=64, page_size=16, num_pages=40)
    ours, gen = logits(fp8_dir, steps_of(seq, 10), pc)
    ref = hf_logits(deq_dir, seq + gen)[:len(ours)]
    base, _ = logits(fp8_dir, steps_of(seq, 10), pc, attention="torch", fused_ops=False)
    ops.enabled(True)
    err, noise = (ours - ref).abs(), (base - ref).abs()
    assert err.mean().item() < 1.3 * noise.mean().item(), (err.mean().item(), noise.mean().item())
    top2 = ref.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 1.0
    agree = (ours.argmax(-1)[clear] == ref.argmax(-1)[clear]).float().mean().item()
    assert agree > 0.95, agree


def test_fp8_graphs_equal_eager(tiny) -> None:
    """The FP8 GEMMs and the quantizing producers inside the decode graphs and the breakable prefill graphs give the
    eager tokens. The attention's key splits are fixed, as batch invariance fixes them (the sliding layers on our
    kernel, not FlashInfer's): without it, the wave split can split a graph's padded batch otherwise than the eager
    one, which moves the attention's rounding, not the FP8 path's, and flips greedy near-ties."""
    fp8_dir, _ = tiny
    seq, other = ids(90, 2), ids(37, 3)
    toks = []
    for graphs in (False, True):
        pc = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(fp8_dir, pc, graphs, attn512_split_keys=64, window_split_keys=16), pc, window=W)
        drv.r.logits_hook = None
        steps = [dict(rows=[(1, seq, 0, True), (2, other, 0, True)], admit=[(1, GREEDY), (2, GREEDY)])]
        steps += [dict(rows=[(1, None, 90 + i, True), (2, None, 37 + i, True)]) for i in range(20)]
        res = drv.run(steps)
        toks.append(np.stack([r.tokens[:, 0] for r in res]))
        drv.r.shutdown()
    assert np.array_equal(toks[0], toks[1])


def test_precise_rows_on_cuda(tiny) -> None:
    """PRECISE rows (the precise prefix): the chunk runs eagerly with BF16 activations (its logits as close to
    transformers on the dequantized weights as a BF16 model's, far closer than the FP8 path's), the chunk after it on
    the FP8 path again; the GEMM and dequantization kernels leave the activation mode as they found it."""
    fp8_dir, deq_dir = tiny
    seq = ids(110, 4)
    pc = gemma_pools(budget=64, page_size=16, num_pages=40)

    def steps(precise: bool) -> list[dict]:
        return [dict(rows=[(1, seq[:50], 0, False, precise)], admit=[(1, GREEDY)]),
                dict(rows=[(1, seq[50:], 50, True)])]

    ref = hf_logits(deq_dir, seq)
    pre, _ = logits(fp8_dir, steps(True), pc)
    quant, _ = logits(fp8_dir, steps(False), pc)
    assert fp8.activation_quant()
    e_pre, e_quant = (pre[:50] - ref[:50]).abs().mean().item(), (quant[:50] - ref[:50]).abs().mean().item()
    assert e_pre < 0.5 * e_quant, (e_pre, e_quant)
    e_next = (pre[50:] - ref[50:]).abs().mean().item()
    assert e_next > 2 * e_pre, (e_next, e_pre)


def test_a_mixed_steps_fp8_rows_keep_their_bits(tiny) -> None:
    """A step holding an FP8 row and a PRECISE row gives the FP8 row's logits bit for bit as the same step without the
    PRECISE flag (per-row precision: the decode rows and other prefills of a refine's steps keep the FP8 path)."""
    fp8_dir, _ = tiny
    a, b = ids(60, 5), ids(45, 6)

    def run(precise: bool) -> GDriver:
        pc = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(fp8_dir, pc, graphs=False), pc, window=W)
        drv.run([dict(rows=[(1, a, 0, True), (2, b, 0, True, precise)], admit=[(1, GREEDY), (2, GREEDY)])])
        drv.r.shutdown()
        return drv

    mixed, plain = run(True), run(False)
    assert all(torch.equal(mixed.logits[(1, p)], plain.logits[(1, p)]) for p in range(60))


def test_pdl_is_bit_for_bit(tiny) -> None:
    """Programmatic dependent launch changes when kernels start, never what they compute: eager logits with it on are
    the logits with it off bit for bit (prefill chunks of 50, 63 and 37 tokens and decode rows), and so are the graphs'
    greedy tokens; every path leaves the library's flag off for the next runner."""
    fp8_dir, _ = tiny
    seq = ids(150, 2)
    pc = gemma_pools(budget=64, page_size=16, num_pages=40)
    off, gen_off = logits(fp8_dir, steps_of(seq, 10), pc, pdl=False)
    on, gen_on = logits(fp8_dir, steps_of(seq, 10), pc, pdl=True)
    assert torch.equal(off, on) and gen_off == gen_on
    toks = []
    other = ids(37, 3)
    for pdl in (False, True):
        pcg = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(fp8_dir, pcg, True, pdl=pdl), pcg, window=W)
        drv.r.logits_hook = None
        assert ops.pdl() == pdl
        steps = [dict(rows=[(1, seq[:90], 0, True), (2, other, 0, True)], admit=[(1, GREEDY), (2, GREEDY)])]
        steps += [dict(rows=[(1, None, 90 + i, True), (2, None, 37 + i, True)]) for i in range(20)]
        toks.append(np.stack([r.tokens[:, 0] for r in drv.run(steps)]))
        drv.r.shutdown()
    assert np.array_equal(toks[0], toks[1])
    ops.set_pdl(False)


def test_a_precise_piece_reports_nan(tiny) -> None:
    """tests/models/test_gemma4_runner.py's test of the same name on CUDA with graphs on: the PRECISE steps run eagerly
    (the mode spied), and with the sliding layers' RoPE NaN at position 30 each piece (BF16 activations) reports its
    NaN, the first computing it, the second reading it; the FP8 rows beside them, short of it, sample as without the
    NaN. FP8 rows report nothing (the next test)."""
    fp8_dir, _ = tiny
    a, b = ids(80, 16), ids(25, 17)
    steps = [dict(rows=[(1, a[:50], 0, False, True), (2, b, 0, True)], admit=[(1, GREEDY), (2, GREEDY)]),
             dict(rows=[(2, None, 25, True), (1, a[50:], 50, False, True)])]

    def run(poison: bool) -> tuple[list, list[str]]:
        pc = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(fp8_dir, pc, graphs=True), pc, window=W)
        drv.r.logits_hook = None
        modes = spy_modes(drv.r)
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            res = drv.run(steps)
        drv.r.shutdown()
        return res, modes

    (nan, modes), (clean, _) = run(True), run(False)
    assert modes == ["eager", "eager"]
    assert [r.status.tolist() for r in nan] == [[NONFINITE, 0], [0, NONFINITE]]
    assert all(r.status is None for r in clean)
    assert all(np.array_equal(x.tokens, y.tokens) for x, y in zip(nan, clean))


def test_fp8_rows_turn_nan_finite(tiny) -> None:
    """tests/models/test_gemma4_runner.py's test of the same name on CUDA with graphs on: the activation quantizer
    clamps a NaN that attention computes to −448 (`runner.cuda.gemma`'s module docstring), so slot 1's FP8 chunk
    [0, 32), which computes position 30, and its final chunk [32, 40), which reads it in the window, report nothing; the
    second samples, and slot 2 beside them samples as without the NaN."""
    fp8_dir, _ = tiny
    a, b = ids(40, 18), ids(25, 19)
    steps = [dict(rows=[(1, a[:32], 0, False), (2, b, 0, True)], admit=[(1, GREEDY), (2, GREEDY)]),
             dict(rows=[(2, None, 25, True), (1, a[32:], 32, True)])]

    def run(poison: bool) -> list:
        pc = gemma_pools(max_num_seqs=4, budget=128, page_size=16, num_pages=64)
        drv = GDriver(cuda_runner(fp8_dir, pc, graphs=True), pc, window=W)
        drv.r.logits_hook = None
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            res = drv.run(steps)
        drv.r.shutdown()
        return res

    nan, clean = run(True), run(False)
    assert all(r.status is None for r in nan + clean) and nan[1].n_out.tolist() == [1, 1]
    assert np.array_equal(nan[0].tokens, clean[0].tokens) and np.array_equal(nan[1].tokens[0], clean[1].tokens[0])


def test_the_memory_profile_captures_without_cublas_warm(tiny) -> None:
    """The startup path, in a fresh process (no cuBLAS handle yet; the memory profile captures the graphs before any
    warm-up): with FP8 every projection is a CUTLASS GEMM and the head at one row our small-M kernel, so cuBLAS's first
    call would come inside the first capture, which then fails (`cublasCreate`). `graphs.warm_libraries` creates the
    handle first."""
    import os
    import subprocess
    import sys

    fp8_dir, _ = tiny
    code = ("from llm_mini_infer.runner.cuda.gemma import GemmaRunner\n"
            f"r = GemmaRunner({str(fp8_dir)!r}, device='cuda', max_num_seqs=8, max_num_batched_tokens=256, "
            "max_model_len=1024, memory_utilization=0.5, graphs=True)\n"
            "caps = r.capabilities()\n"
            "assert caps.memory_free_bytes > 0 and r.stats['graph_pool_bytes'] > 0\n"
            "print('captured', r.stats['profile_capture_s'])\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0 and "captured" in res.stdout, res.stderr[-3000:]

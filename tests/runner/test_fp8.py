"""Block-FP8 checkpoints on the torch references: the quantization numerics, the checkpoint's quantization config,
loading into fused projections, and a tiny FP8 model end to end. The CUDA kernels are checked against these references
in tests/kernels/test_kernels.py."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from llm_mini_infer.kernels import build, fp8, ops
from llm_mini_infer.models.qwen3_5.model import Fp8Linear, Qwen35Config, build_model, map_weight_name
from llm_mini_infer.runner.base import PlanError

from ..support.fixtures import fresh_kernel_lookup, require_qwen, simulated_cuda
from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cpu_runner, hf_logits, make_tiny_fp8_model, pools


@pytest.fixture(scope="module")
def tiny_fp8(tmp_path_factory):
    return make_tiny_fp8_model(tmp_path_factory.mktemp("tiny_fp8"), seed=5)


def test_quant_ref_numerics() -> None:
    """Per-token-group FP8 quantization: scale = max(absmax, 1e-10) / 448, q = e4m3(x · (1 / scale)), RNE,
    saturating."""
    x = torch.zeros(3, 256)
    x[0, :128] = torch.linspace(-2.0, 2.0, 128)
    x[0, 128:] = 0.5
    x[1, 5] = -7.0  # the rest of row 1 is zero
    a = fp8.quant_ref(x)
    s = a.s
    assert a.q.dtype == torch.float8_e4m3fn and a.q.shape == (3, 256) and s.shape == (2, 3)
    assert s[0, 0] == torch.tensor(2.0) / 448 and s[1, 0] == torch.tensor(0.5) / 448
    assert s[0, 1] == torch.tensor(7.0) / 448 and s[1, 1] == torch.tensor(1e-10) / 448  # an all-zero group
    assert s[0, 2] == s[1, 2] == torch.tensor(1e-10) / 448
    assert a.q[0, 0].float() == -448 and a.q[0, 127].float() == 448 and (a.q[0, 128:].float() == 448).all()
    assert a.q[1, 5].float() == -448 and (a.q[2].float() == 0).all()
    # round to nearest even: 448 * 0.3 = 134.4 lies between e4m3 values 128 and 144 (16 apart), nearer 128
    y = torch.zeros(1, 128)
    y[0, 0], y[0, 1] = 1.0, 0.3
    assert fp8.quant_ref(y).q[0, 1].float() == 128
    # dequantized values are within half a step of the input: 2^-4 relative, or half the smallest subnormal step
    # (2^-10 of the scale) below e4m3's normal range
    xr = torch.randn(8, 384) * torch.logspace(-2, 2, 8)[:, None]
    a = fp8.quant_ref(xr)
    tol = xr.abs() * 2 ** -4 + a.s.T.repeat_interleave(128, dim=1) * 2 ** -10
    assert ((fp8.dequant_act(a) - xr).abs() <= tol).all()


@pytest.mark.parametrize("divide", [False, True], ids=["reciprocal", "division"])
def test_quant_ref_non_finite(divide) -> None:
    """NaN and ±Inf as the kernels take them (`fmaxf` and `fminf`, kernels/csrc/quant.cuh; tests/kernels/test_kernels.py
    compares the two), in both flavours: a NaN among finite values is left out of the absmax and quantizes to −448
    (byte 0xFE); an all-NaN group has the floor's scale, whose reciprocal is finite (about 4.5e12), and every byte
    0xFE; an infinite value makes the scale Inf and its reciprocal 0, its own byte 0xFE (Inf · 0 or Inf / Inf: NaN) and
    the finite values' ±0. The other groups are untouched."""
    torch.manual_seed(3)
    x = torch.randn(3, 384)
    clean = fp8.quant_ref(x, divide=divide)
    x[0, 5] = float("nan")
    x[1, 128:256] = float("nan")
    x[2, 300], x[2, 301] = float("inf"), float("-inf")
    a = fp8.quant_ref(x, divide=divide)
    q, s = a.q.view(torch.uint8), a.s
    without = x[:1, :128].clone()
    without[0, 5] = 0
    w = fp8.quant_ref(without, divide=divide)
    wq = w.q.view(torch.uint8)[0]
    assert s[0, 0] == w.s[0, 0] and q[0, 5] == 0xFE
    assert torch.equal(q[0, :5], wq[:5]) and torch.equal(q[0, 6:128], wq[6:])
    assert s[1, 1] == torch.tensor(1e-10) / 448 and (q[1, 128:256] == 0xFE).all()
    floor_inv = torch.ones(()) / s[1, 1]
    assert torch.isfinite(floor_inv) and 4.4e12 < floor_inv < 4.5e12
    assert torch.ones(()) / s[2, 2] == 0
    assert s[2, 2] == float("inf") and q[2, 300] == q[2, 301] == 0xFE
    finite = torch.cat([x[2, 256:300], x[2, 302:]])
    assert torch.equal(torch.cat([q[2, 256:300], q[2, 302:]]), (finite < 0).to(torch.uint8) * 0x80)  # ±0
    hit = torch.zeros(3, 3, dtype=torch.bool)
    hit[0, 0] = hit[1, 1] = hit[2, 2] = True
    assert torch.equal(s[~hit.T], clean.s[~hit.T])
    keep = ~hit.repeat_interleave(128, dim=1)
    assert torch.equal(q[keep], clean.q.view(torch.uint8)[keep])


def test_the_reciprocal_and_the_division_round_apart() -> None:
    """The two flavours differ: a group of absmax 6.125 has the exact scale 7/512, so 1.5859375 divided by it is 116
    exactly, an e4m3 tie that rounds to even, 112 (byte 0x6E); times the rounded reciprocal (512/7) it lands just above
    116 and rounds to 120 (0x6F). vLLM's standalone and fused-norm quantizers take the reciprocal, its fused
    SiLU-and-mul the division."""
    x = torch.zeros(1, 128)
    x[0, 0], x[0, 1] = 6.125, 1.5859375
    rec, div = fp8.quant_ref(x), fp8.quant_ref(x, divide=True)
    assert rec.s[0, 0] == div.s[0, 0] == 7 / 512
    assert (x[0, 1] / rec.s[0, 0]).item() == 116 and (x[0, 1] * (torch.ones(()) / rec.s[0, 0])).item() > 116
    assert (div.q.view(torch.uint8)[0, 1].item(), div.q[0, 1].float().item()) == (0x6E, 112)
    assert (rec.q.view(torch.uint8)[0, 1].item(), rec.q[0, 1].float().item()) == (0x6F, 120)
    assert torch.equal(rec.q.view(torch.uint8)[0, [0, *range(2, 128)]], div.q.view(torch.uint8)[0, [0, *range(2, 128)]])


def test_quant_ref_writes_strided_views() -> None:
    """Producers that fill row ranges (the GDN decode and prefill rows) write views of one buffer."""
    x = torch.randn(10, 256)
    full = fp8.quant_ref(x)
    buf = fp8.empty_act(10, 256, torch.device("cpu"))
    fp8.quant_ref(x[:4], fp8.QuantAct(buf.q[:4], buf.s[:, :4]))
    fp8.quant_ref(x[4:], fp8.QuantAct(buf.q[4:], buf.s[:, 4:]))
    assert torch.equal(buf.q.view(torch.uint8), full.q.view(torch.uint8)) and torch.equal(buf.s, full.s)


def test_gemm_ref_is_the_dequantized_product() -> None:
    torch.manual_seed(0)
    w = torch.randn(256, 384) * 0.05
    q, sc = fp8.quantize_weight(w)
    assert q.shape == (256, 384) and sc.shape == (2, 3)
    tol = w.abs() * 2 ** -4 + sc.repeat_interleave(128, 0).repeat_interleave(128, 1) * 2 ** -10
    assert ((fp8.dequant_weight(q, sc) - w).abs() <= tol).all()
    x = torch.randn(5, 384)
    a = fp8.quant_ref(x)
    ref = (fp8.dequant_act(a).double() @ fp8.dequant_weight(q, sc).double().T)
    out = fp8.gemm(a, q, sc, out_dtype=torch.float32)
    assert torch.allclose(out.double(), ref, rtol=1e-5, atol=1e-5)
    assert fp8.gemm(a, q, sc).dtype == torch.bfloat16


def test_producers_equal_their_bf16_op_then_quantization() -> None:
    """The fused producers' references: the BF16 op rounded, then quantized by the reciprocal (SiLU-and-mul: the fp32
    product, by division, which its fallback asks `quant_ref` for: an fp32 product almost never lands on an e4m3 tie,
    so the bytes alone rarely tell the flavours apart)."""
    torch.manual_seed(1)
    bf = torch.bfloat16
    x, res, w = torch.randn(7, 256, dtype=bf), torch.randn(7, 256, dtype=bf), torch.randn(256, dtype=bf) * 0.1
    a, r = ops.rmsnorm_quant(x, w, 1e-6, res, bf16=True)
    y, r2 = ops.rmsnorm(x, w, 1e-6, res)
    assert torch.equal(r, r2) and torch.equal(a.bf16, y)
    assert torch.equal(a.q.view(torch.uint8), fp8.quant_ref(y).q.view(torch.uint8))
    gu = torch.randn(7, 512, dtype=bf)
    g, u = gu.float().chunk(2, dim=-1)
    asked, ref = [], fp8.quant_ref

    def recorded(x, out=None, divide=False):
        asked.append(divide)
        return ref(x, out, divide)

    fp8.quant_ref = recorded
    try:
        got = ops.silu_mul_quant(gu)
    finally:
        fp8.quant_ref = ref
    div = fp8.quant_ref(torch.nn.functional.silu(g) * u, divide=True)
    assert asked == [True] and torch.equal(got.s, div.s)
    assert torch.equal(got.q.view(torch.uint8), div.q.view(torch.uint8))
    o, gate = torch.randn(7, 256, dtype=bf), torch.randn(7, 2, 256, dtype=bf)[..., 128:]
    got = ops.sigmoid_gate_quant(o, gate)
    want = fp8.quant_ref(ops.sigmoid_gate(o, gate))
    assert torch.equal(got.s, want.s) and torch.equal(got.q.view(torch.uint8), want.q.view(torch.uint8))


def test_quantization_config_of_the_real_checkpoint() -> None:
    """Qwen3.8-27B-FP8: block [128, 128], dynamic; every projection FP8 except in_proj_ba; BF16 embeddings/lm_head."""
    cfg = Qwen35Config.from_hf(json.loads((require_qwen() / "config.json").read_text()))
    assert cfg.quant is not None and cfg.quant.block == (128, 128)
    for name in ("layers.3.self_attn.qkv_proj", "layers.3.self_attn.o_proj", "layers.0.linear_attn.in_proj_qkvz",
                 "layers.0.linear_attn.out_proj", "layers.0.mlp.gate_up_proj", "layers.63.mlp.down_proj"):
        assert cfg.fp8_module(name), name
    assert not cfg.fp8_module("layers.0.linear_attn.in_proj_ba")
    assert not cfg.fp8_module("lm_head") and not cfg.fp8_module("embed_tokens")
    # weight_scale_inv of a fused shard lands at its first row / 128
    assert map_weight_name(cfg, "model.language_model.layers.3.self_attn.k_proj.weight_scale_inv") == \
        ("layers.3.self_attn.qkv_proj.weight_scale", 96)
    assert map_weight_name(cfg, "model.language_model.layers.0.linear_attn.in_proj_z.weight_scale_inv") == \
        ("layers.0.linear_attn.in_proj_qkvz.weight_scale", 80)
    assert map_weight_name(cfg, "model.language_model.layers.5.mlp.up_proj.weight_scale_inv") == \
        ("layers.5.mlp.gate_up_proj.weight_scale", 136)
    assert map_weight_name(cfg, "model.language_model.layers.5.mlp.down_proj.weight_scale_inv") == \
        ("layers.5.mlp.down_proj.weight_scale", 0)
    assert map_weight_name(cfg, "mtp.layers.0.mlp.up_proj.weight_scale_inv") is None


def test_unsupported_quantization_is_refused() -> None:
    hf = json.loads((require_qwen() / "config.json").read_text())
    for change, match in (({"weight_block_size": None}, "per-tensor"), ({"activation_scheme": "static"}, "dynamic"),
                          ({"quant_method": "awq"}, "awq")):
        bad = dict(hf, quantization_config={**hf["quantization_config"], **change})
        with pytest.raises(NotImplementedError, match=match):
            Qwen35Config.from_hf(bad)


def test_loading_keeps_weights_and_scales_as_stored(tiny_fp8) -> None:
    fp8_dir, _ = tiny_fp8
    model = build_model(fp8_dir, torch.device("cpu"), torch.float32)
    sd = load_file(fp8_dir / "model.safetensors")
    lay0, lay1 = model.layers[0], model.layers[1]
    assert isinstance(lay0.linear_attn.in_proj_qkvz, Fp8Linear) and not isinstance(lay0.linear_attn.in_proj_ba,
                                                                                    Fp8Linear)
    assert isinstance(lay1.self_attn.qkv_proj, Fp8Linear) and isinstance(lay1.mlp.down_proj, Fp8Linear)
    assert lay0.mixer_quant == "fp8+bf16" and lay1.mixer_quant == "fp8" and lay0.mlp.input_quant == "fp8"
    w = lay1.self_attn.qkv_proj
    q = sd["model.layers.1.self_attn.q_proj.weight"]
    k = sd["model.layers.1.self_attn.k_proj.weight"]
    assert torch.equal(w.weight[:256].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(w.weight[256:384].view(torch.uint8), k.view(torch.uint8))
    assert torch.equal(w.weight_scale[2:3], sd["model.layers.1.self_attn.k_proj.weight_scale_inv"])
    up = sd["model.layers.0.mlp.up_proj.weight_scale_inv"]
    assert torch.equal(lay0.mlp.gate_up_proj.weight_scale[2:], up)
    assert model.lm_head.weight.dtype == torch.float32  # BF16 in the checkpoint, the model dtype here


def test_mismatched_config_and_tensors_are_refused(tiny_fp8, tmp_path) -> None:
    import shutil

    fp8_dir, _ = tiny_fp8
    bad = tmp_path / "bad"
    shutil.copytree(fp8_dir, bad)
    conf = json.loads((bad / "config.json").read_text())
    conf["quantization_config"]["modules_to_not_convert"].append("model.layers.1.mlp.down_proj")
    (bad / "config.json").write_text(json.dumps(conf))
    with pytest.raises(ValueError, match="BF16 modules disagree with the tensors"):
        build_model(bad, torch.device("cpu"), torch.float32)


def test_ue8m0_scales_are_refused(tiny_fp8, tmp_path) -> None:
    """Block scales in UE8M0 are refused at load (`fp8.check_tensor`, Gemma 4's loader's check too), not widened into
    the FP32 scales."""
    import shutil

    from safetensors.torch import save_file

    fp8_dir, _ = tiny_fp8
    bad = tmp_path / "bad"
    shutil.copytree(fp8_dir, bad)
    name = "model.layers.0.mlp.up_proj.weight_scale_inv"
    path = next(p for p in bad.glob("*.safetensors") if name in load_file(p))
    sd = load_file(path)
    sd[name] = sd[name].to(torch.float8_e8m0fnu)
    save_file(sd, path)
    with pytest.raises(NotImplementedError, match=rf"^{name}: block scales in torch\.float8_e8m0fnu"):
        build_model(bad, torch.device("cpu"), torch.float32)


def _ids(n: int, seed: int = 0) -> list[int]:
    return [(i * 37 + 11 + seed * 101) % 500 for i in range(n)]


def _prefill_logits(model_dir, seq: list[int]) -> torch.Tensor:
    pc = pools(budget=128)
    drv = Driver(cpu_runner(model_dir, pc), pc)
    drv.run([Step(rows=[Row(1, seq, 0)], admit=[(1, 1, GREEDY)])])
    return torch.stack([drv.logits[(1, p)] for p in range(len(seq))])


def test_tiny_fp8_model_equals_transformers_on_the_dequantized_weights(tiny_fp8, monkeypatch) -> None:
    """With the activation quantization replaced by the identity (q = x in fp32, scale 1), our FP8 model is
    transformers on the dequantized weights: the loading, the fused shards and every producer's wiring are exact."""
    fp8_dir, deq_dir = tiny_fp8
    seq = _ids(70)
    ref = hf_logits(deq_dir, seq)

    def identity(x: torch.Tensor, out: fp8.QuantAct | None = None, divide: bool = False) -> fp8.QuantAct:
        m, k = x.shape
        if out is None:
            return fp8.QuantAct(x.float().clone(), torch.ones((k // 128, m)))
        out.q.copy_(x)
        out.s[:, :m] = 1.0
        return out

    monkeypatch.setattr(fp8, "quant_ref", identity)
    monkeypatch.setattr(fp8, "empty_act", lambda m, k, device: fp8.QuantAct(torch.empty((m, k)),
                                                                           torch.ones((k // 128, m))))
    exact = _prefill_logits(fp8_dir, seq)
    assert (exact - ref).abs().max().item() < 1e-4


def test_tiny_fp8_model_with_quantized_activations(tiny_fp8) -> None:
    """The quantized activations move this random tiny model's logits by about 13% RMS (e4m3 keeps 3 mantissa bits;
    nothing in random weights damps the error): a sanity bound only. Real-model accuracy is the parity comparison's
    (architecture.md, "Testing")."""
    fp8_dir, deq_dir = tiny_fp8
    seq = _ids(70)
    ref = hf_logits(deq_dir, seq)
    ours = _prefill_logits(fp8_dir, seq)
    rel = ((ours - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel < 0.25, rel
    assert (ours.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.6


def test_tiny_fp8_chunked_prefill_and_decode_equal_single_shot(tiny_fp8) -> None:
    """Quantization is per token, so chunking and decoding change nothing but fp32 summation order."""
    fp8_dir, _ = tiny_fp8
    seq = _ids(60, seed=2)
    pc = pools(budget=64)
    one = Driver(cpu_runner(fp8_dir, pc), pc)
    one.run([Step(rows=[Row(1, seq, 0)], admit=[(1, 1, GREEDY)])])
    chunked = Driver(cpu_runner(fp8_dir, pc), pc)
    res = chunked.run([Step(rows=[Row(1, seq[:21], 0, sample=False)], admit=[(1, 1, GREEDY)]),
                       Step(rows=[Row(1, seq[21:22], 21, sample=False)]),
                       Step(rows=[Row(1, seq[22:], 22)])] +
                      [Step(rows=[Row(1, None, 60 + i)]) for i in range(5)])
    a = torch.stack([one.logits[(1, p)] for p in range(60)])
    b = torch.stack([chunked.logits[(1, p)] for p in range(60)])
    assert (a - b).abs().max().item() < 1e-4
    gen = [int(r.tokens[0, 0]) for r in res[2:]]
    full = Driver(cpu_runner(fp8_dir, pools(budget=128)), pools(budget=128))
    full.run([Step(rows=[Row(1, seq + gen[:-1], 0)], admit=[(1, 1, GREEDY)])])
    assert [int(full.logits[(1, 59 + i)].argmax()) for i in range(len(gen))] == gen


def test_precise_rows_run_without_activation_quantization(tiny_fp8) -> None:
    """PRECISE rows (the precise prefix, per row): such a row multiplies BF16 activations with the dequantized
    weights, so its logits are transformers' on the dequantized weights; the next chunk, not PRECISE, is quantized
    again; a runner with the precise prefix off (fp8_cold_prefill fp8) refuses PRECISE rows."""
    fp8_dir, deq_dir = tiny_fp8
    seq = _ids(70, seed=3)
    ref = hf_logits(deq_dir, seq)

    def run(precise: bool, mode: str = "reuse") -> torch.Tensor:
        pc = pools(budget=128)
        drv = Driver(cpu_runner(fp8_dir, pc, fp8_cold_prefill=mode), pc)
        drv.run([Step(rows=[Row(1, seq[:40], 0, sample=False, precise=precise)], admit=[(1, 1, GREEDY)]),
                 Step(rows=[Row(1, seq[40:], 40)])])
        assert fp8.activation_quant()  # the mode ends with the step
        return torch.stack([drv.logits[(1, p)] for p in range(len(seq))])

    pre, quant = run(True), run(False)
    assert (pre[:40] - ref[:40]).abs().max().item() < 1e-4  # the PRECISE chunk: no activation quantization
    assert (quant[:40] - ref[:40]).abs().max().item() > 1e-2  # an FP8 row quantizes it
    assert (pre[40:] - ref[40:]).abs().max().item() > 1e-2  # the next chunk is quantized again
    with pytest.raises(PlanError, match="PRECISE"):
        run(True, mode="fp8")


def test_a_step_mixes_precise_and_fp8_rows(tiny_fp8) -> None:
    """Per-row precision: in one step, a PRECISE row (planned first; the runner orders it last) takes BF16 activations
    and the FP8 rows the FP8 path, each as it would without the other: the FP8 row's logits are those of the same step
    without the PRECISE flag (SiLU-and-mul's fused quantization included, `fp8.SplitAct`), the PRECISE row's are
    transformers' on the dequantized weights."""
    fp8_dir, deq_dir = tiny_fp8
    a, b = _ids(40, seed=4), _ids(30, seed=6)

    def run(precise: bool) -> Driver:
        pc = pools(budget=128)
        drv = Driver(cpu_runner(fp8_dir, pc), pc)
        drv.run([Step(rows=[Row(2, b, 0, precise=precise), Row(1, a, 0)], admit=[(1, 1, GREEDY), (2, 2, GREEDY)])])
        return drv

    mixed, plain = run(True), run(False)
    fa_mixed = torch.stack([mixed.logits[(1, p)] for p in range(40)])
    fa_plain = torch.stack([plain.logits[(1, p)] for p in range(40)])
    assert (fa_mixed - fa_plain).abs().max().item() < 1e-5
    pb = torch.stack([mixed.logits[(2, p)] for p in range(30)])
    assert (pb - hf_logits(deq_dir, b)).abs().max().item() < 1e-4
    qb = torch.stack([plain.logits[(2, p)] for p in range(30)])
    assert (qb - hf_logits(deq_dir, b)).abs().max().item() > 1e-2


def test_split_activations_reach_each_rows_path(tiny_fp8) -> None:
    """`fp8.precise_rows(t, first)`: an Fp8Linear given t rows sends rows [0, first) through the FP8 path (quantized
    there) and [first, t) through the BF16 path; activations of other sizes keep the FP8 path."""
    fp8_dir, _ = tiny_fp8
    model = build_model(fp8_dir, torch.device("cpu"), torch.float32)
    lin = next(m for m in model.modules() if isinstance(m, Fp8Linear))
    torch.manual_seed(2)
    x = torch.randn(7, lin.in_features)
    # each path's reference on the same rows: a GEMM's sums can depend on M (x86 BLAS blocks by it)
    fp8_first, fp8_five = lin(x[:4]), lin(x[:5])
    with fp8.bf16_activations():
        bf16_last = lin(x[4:])
    assert not torch.equal(lin(x[4:]), bf16_last)  # the two paths differ
    with fp8.precise_rows(7, 4):
        assert not fp8.activation_quant() and fp8.bf16_rows(7) == 3 and fp8.bf16_rows(5) == 0
        y = lin(x)
        z = lin(x[:5])  # another size: FP8
        with pytest.raises(ValueError, match="rows that take BF16"):
            lin(fp8.quant_ref(x))
    assert fp8.activation_quant()
    assert torch.equal(y[:4], fp8_first) and torch.equal(y[4:], bf16_last)
    assert torch.equal(z, fp8_five)


def test_bf16_activations_switch_restores_and_refuses_quantized_inputs(tiny_fp8) -> None:
    fp8_dir, _ = tiny_fp8
    model = build_model(fp8_dir, torch.device("cpu"), torch.float32)
    lin = next(m for m in model.modules() if isinstance(m, Fp8Linear))
    x = torch.randn(3, lin.in_features)
    with fp8.bf16_activations():
        assert not fp8.activation_quant()
        y = lin(x)
        with pytest.raises(ValueError, match="quantized activation"):
            lin(fp8.quant_ref(x))
    assert fp8.activation_quant()
    want = (x @ fp8.dequant_weight(lin.weight, lin.weight_scale).T)
    assert (y - want).abs().max().item() < 1e-5
    assert (ops.dequant_fp8(lin.weight, lin.weight_scale).float() -
            fp8.dequant_weight(lin.weight, lin.weight_scale).to(torch.bfloat16).float()).abs().max().item() == 0


def test_dispatch_table_keeps_qwens_choices_and_takes_gemmas() -> None:
    """The table's rows for Gemma 4's N (gate_up 43,008, the full layers' qkv 18,432) change no Qwen3.8-27B shape's
    configuration: each keeps the Qwen-only table's (`old`) at every M (the table is a speed choice: every
    configuration gives the same bits)."""
    old = [(96, fp8.BIG, "cutlass", 0), (128, 8192, "cutlass", 0), (128, 20480, "cutlass", 2),
           (128, fp8.BIG, "cutlass", 0), (256, fp8.BIG, "cutlass", 1), (fp8.BIG, fp8.BIG, "cutlass", 2)]

    def pick(table, m, n):
        return next((b, c) for mm, nn, b, c in table if m <= mm and n <= nn)

    fp8.use_backend("auto")
    for n in (5120, 14336, 16384, 34816):  # Qwen3.8-27B's projections
        for m in range(1, 2049):
            assert fp8._pick(m, n) == pick(old, m, n), (m, n)
    assert fp8._pick(128, 43008) == ("cutlass", 2) and fp8._pick(256, 43008) == ("cutlass", 2)
    assert fp8._pick(256, 18432) == ("cutlass", 2) and fp8._pick(64, 43008) == ("cutlass", 0)
    assert fp8._pick(128, 5376) == ("cutlass", 0) and fp8._pick(256, 5376) == ("cutlass", 1)


@pytest.mark.parametrize("fused_ops", [True, False])
def test_a_cuda_runner_without_the_kernel_library_fails_before_loading(tiny_fp8, monkeypatch, fused_ops) -> None:
    """The Qwen runner's side of tests/models/test_gemma4_fp8.py's test: without the kernel library it refuses to start,
    naming why, before any weight loads, with dev.runner.cuda.fused_ops off too (the sampler needs the library all the
    same). CUDA simulated on the CPU, nvcc and the prebuilt directories (whose record of nvcc's version would key the
    lookup) hidden."""
    from llm_mini_infer.runner.cuda import runner

    simulated_cuda(monkeypatch)
    fresh_kernel_lookup(monkeypatch)
    monkeypatch.setattr(build, "find_nvcc", lambda: None)
    monkeypatch.delenv("LLM_MINI_INFER_KERNEL_PREBUILT", raising=False)
    monkeypatch.setattr(runner, "build_model", lambda *args, **kw: pytest.fail("the weights started loading"))
    with pytest.raises(RuntimeError, match=r"needs llm-mini-infer's CUDA kernel library.*nvcc not found"):
        runner.CudaRunner(tiny_fp8[0], device="cuda", memory_gb=1.0, fused_ops=fused_ops)


def test_a_cuda_runner_refuses_fp32_with_our_kernels(tiny_fp8, monkeypatch) -> None:
    """The Qwen runner's side of tests/models/test_gemma4_fp8.py's test: fp32 with dev.runner.cuda.fused_ops on is
    refused before any weight loads (the Qwen wrappers would hand BF16 kernels fp32 buffers). CUDA and the kernel
    library simulated on the CPU."""
    from llm_mini_infer.runner.cuda import runner

    simulated_cuda(monkeypatch)
    monkeypatch.setattr(build, "require", lambda what: None)
    monkeypatch.setattr(runner, "build_model", lambda *args, **kw: pytest.fail("the weights started loading"))
    with pytest.raises(ValueError, match=r"needs dev\.runner\.cuda\.fused_ops false"):
        runner.CudaRunner(tiny_fp8[0], device="cuda", memory_gb=1.0, dtype=torch.float32)


@pytest.mark.gpu
@pytest.mark.cutlass(reason="needs a CUDA GPU, the kernel library and the CUTLASS library")
def test_a_mixed_steps_fp8_rows_keep_their_bits_on_cuda(tiny_fp8) -> None:
    """On CUDA (eager), a step holding an FP8 row and a PRECISE row gives the FP8 row's logits bit for bit as the same
    step without the PRECISE flag (its quantization, GEMMs and attention are its own), and the PRECISE row's logits are
    closer to transformers on the dequantized weights than the FP8 path's."""
    from llm_mini_infer.runner.cuda.runner import CudaRunner

    fp8_dir, deq_dir = tiny_fp8
    a, b = _ids(40, seed=4), _ids(30, seed=6)

    def run(precise: bool) -> Driver:
        pc = pools(budget=128)
        r = CudaRunner(fp8_dir, device="cuda", max_num_seqs=pc.max_num_seqs,
                       max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len,
                       memory_gb=1.0, check=True, graphs=False)
        r.configure(pc)
        drv = Driver(r, pc)
        drv.run([Step(rows=[Row(1, a, 0), Row(2, b, 0, precise=precise)], admit=[(1, 1, GREEDY), (2, 2, GREEDY)])])
        r.shutdown()
        return drv

    mixed, plain = run(True), run(False)
    assert all(torch.equal(mixed.logits[(1, p)], plain.logits[(1, p)]) for p in range(40))
    ref = hf_logits(deq_dir, b)
    pb = torch.stack([mixed.logits[(2, p)] for p in range(30)]).float()
    qb = torch.stack([plain.logits[(2, p)] for p in range(30)]).float()
    assert (pb - ref).abs().mean().item() < 0.5 * (qb - ref).abs().mean().item()

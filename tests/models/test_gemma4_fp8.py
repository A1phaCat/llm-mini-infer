"""Gemma 4 in block-FP8 (RedHatAI/gemma-4-31B-it-FP8-block, compressed-tensors FP8_BLOCK) on the torch references:
the quantization config (the published one), the configs we refuse, loading into fused projections with the BF16
scales widened exactly, the model against transformers on the dequantized weights, and PRECISE rows (the precise
prefix) through Gemma's runner (the Qwen runner's implementation, `runner.cuda.fp8`). The CUDA kernels are checked
against these references in tests/kernels/test_gemma_kernels.py and tests/models/test_gemma4_gpu.py."""

from __future__ import annotations

import copy
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from llm_mini_infer.kernels import fp8
from llm_mini_infer.kernels import gemma_ops as gops
from llm_mini_infer.models.fp8 import Fp8Linear, QuantConfig
from llm_mini_infer.models.gemma4.model import Gemma4Config, build_model, map_weight_name

from ..support.fixtures import fresh_kernel_lookup, simulated_cuda
from ..support.paths import DATA
from ..support.tiny_gemma import (
    GREEDY,
    REDHAT_FP8_BLOCK,
    Direct,
    GDriver,
    gemma_cpu_runner,
    gemma_pools,
    hf_logits,
    make_tiny_gemma_fp8,
)
from ..support.token_ids import ids

W = 8


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma_fp8(tmp_path_factory.mktemp("g4fp8"), seed=3)


def test_the_published_quantization_config() -> None:
    """RedHatAI/gemma-4-31B-it-FP8-block's config.json: every text projection of every layer block-FP8, the embeddings
    (the tied head) BF16, the scales `weight_scale` at their shard's first block row (the full layers' k after a q of
    32 heads of 512, no v_proj), the vision tower skipped and its modules matched by `re:.*vision.*`."""
    cfg = Gemma4Config.from_hf(json.loads((DATA / "gemma-4-31b-it-fp8-block.config.json").read_text()))
    q = cfg.quant
    assert q is not None and q.method == "compressed-tensors" and q.block == (128, 128)
    assert q.scale_suffix == "weight_scale" and q.patterns == ("re:.*vision.*", "lm_head", "re:.*embed_tokens.*")
    for layer in (0, 5, 59):
        for proj in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
            assert cfg.fp8_module(f"layers.{layer}.{proj}"), (layer, proj)
    assert not cfg.fp8_module("embed_tokens") and not cfg.fp8_module("lm_head")
    pre = "model.language_model."
    assert map_weight_name(cfg, pre + "layers.5.self_attn.k_proj.weight_scale") == \
        ("layers.5.self_attn.qkv_proj.weight_scale", 128)
    assert map_weight_name(cfg, pre + "layers.0.self_attn.k_proj.weight_scale") == \
        ("layers.0.self_attn.qkv_proj.weight_scale", 64)
    assert map_weight_name(cfg, pre + "layers.0.self_attn.v_proj.weight_scale") == \
        ("layers.0.self_attn.qkv_proj.weight_scale", 96)
    assert map_weight_name(cfg, pre + "layers.0.mlp.up_proj.weight_scale") == ("layers.0.mlp.gate_up_proj.weight_scale",
                                                                                  168)
    assert map_weight_name(cfg, pre + "layers.0.self_attn.o_proj.weight") == ("layers.0.self_attn.o_proj.weight", 0)
    assert map_weight_name(cfg, "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight") is None
    assert q.skips("-", ["model.vision_tower.encoder.layers.0.mlp.gate_proj"]) and q.skips("-", ["lm_head"])
    assert q.skips("-", [pre + "embed_tokens"]) and not q.skips("-", [pre + "layers.0.mlp.gate_proj"])
    assert not q.skips("-", ["model.lm_head_extra"])  # a name entry matches exactly, not as a prefix


@pytest.mark.parametrize("path, value, match", [
    (("config_groups", "FP8_BLOCK", "weights", "strategy"), "channel", "weights strategy"),
    (("config_groups", "FP8_BLOCK", "weights", "block_structure"), [64, 64], "weights strategy"),
    (("config_groups", "FP8_BLOCK", "input_activations", "dynamic"), False, "input activations"),
    (("config_groups", "FP8_BLOCK", "input_activations", "group_size"), 64, "input activations"),
    (("config_groups", "FP8_BLOCK", "input_activations", "strategy"), "token", "input activations"),
    (("config_groups", "FP8_BLOCK", "weights", "num_bits"), 4, "weights"),
    (("config_groups", "FP8_BLOCK", "weights", "type"), "int", "weights"),
    (("config_groups", "FP8_BLOCK", "weights", "symmetric"), False, "weights"),
    (("config_groups", "FP8_BLOCK", "targets"), ["re:.*q_proj"], "targets"),
    (("config_groups", "FP8_BLOCK", "format"), "pack-quantized", "format"),
    (("kv_cache_scheme",), {"num_bits": 8, "type": "float", "strategy": "tensor"}, "KV cache"),
    (("quantization_status",), "frozen", "quantization_status"),
    (("quant_method",), "awq", "awq"),
])
def test_other_quantizations_are_refused(path, value, match) -> None:
    q = copy.deepcopy(REDHAT_FP8_BLOCK)
    d = q
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(NotImplementedError, match=match):
        QuantConfig.from_hf(q)
    two = copy.deepcopy(REDHAT_FP8_BLOCK)
    two["config_groups"]["OTHER"] = two["config_groups"]["FP8_BLOCK"]
    with pytest.raises(NotImplementedError, match="2 config groups"):
        QuantConfig.from_hf(two)


def test_loading_keeps_weights_and_widens_bf16_scales_exactly(tiny) -> None:
    fp8_dir, _ = tiny
    model = build_model(fp8_dir, torch.device("cpu"), torch.float32)
    sd = load_file(fp8_dir / "model.safetensors")
    pre = "model.language_model.layers"
    s0, f2 = model.layers[0].self_attn, model.layers[2].self_attn  # sliding, full (S S F S S F)
    for m in (s0.qkv_proj, s0.o_proj, f2.qkv_proj, f2.o_proj, model.layers[0].mlp.gate_up_proj,
              model.layers[0].mlp.down_proj):
        assert isinstance(m, Fp8Linear)
    assert s0.qkv_proj.out_features == 4 * 64 + 2 * 2 * 64 and f2.qkv_proj.out_features == 4 * 128 + 128  # no v_proj
    assert s0.in_fp8 and s0.out_fp8 and model.layers[0].mlp.in_fp8 and model.layers[0].mlp.down_fp8
    k = sd[f"{pre}.2.self_attn.k_proj.weight"]
    assert torch.equal(f2.qkv_proj.weight[512:].view(torch.uint8), k.view(torch.uint8))
    ks = sd[f"{pre}.2.self_attn.k_proj.weight_scale"]
    assert ks.dtype == torch.bfloat16 and f2.qkv_proj.weight_scale.dtype == torch.float32
    assert torch.equal(f2.qkv_proj.weight_scale[4:], ks.float())  # widened without rounding, at block row 512 / 128
    v = sd[f"{pre}.0.self_attn.v_proj.weight_scale"]
    assert torch.equal(s0.qkv_proj.weight_scale[3:], v.float())
    up = sd[f"{pre}.1.mlp.up_proj.weight_scale"]
    assert torch.equal(model.layers[1].mlp.gate_up_proj.weight_scale[2:], up.float())
    assert not isinstance(model.embed_tokens, Fp8Linear) and model.embed_tokens.weight.dtype == torch.float32


def test_config_and_tensors_that_disagree_are_refused(tiny, tmp_path) -> None:
    fp8_dir, _ = tiny
    bad = tmp_path / "bad"
    shutil.copytree(fp8_dir, bad)
    conf = json.loads((bad / "config.json").read_text())
    conf["quantization_config"]["ignore"].append("re:.*down_proj")  # says BF16; the tensors are FP8
    (bad / "config.json").write_text(json.dumps(conf))
    with pytest.raises(ValueError, match="disagree"):
        build_model(bad, torch.device("cpu"), torch.float32)
    conf["quantization_config"]["ignore"][-1] = "re:.*layers.0.mlp.gate_proj"  # half of a fused projection
    (bad / "config.json").write_text(json.dumps(conf))
    with pytest.raises(ValueError, match="cannot be fused"):
        build_model(bad, torch.device("cpu"), torch.float32)


def _identity_quant(monkeypatch) -> None:
    """Activation quantization replaced by the identity (q = x in fp32, scale 1)."""
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


def test_tiny_fp8_gemma_equals_transformers_on_the_dequantized_weights(tiny, monkeypatch) -> None:
    """With the activation quantization replaced by the identity, our FP8 model is transformers on the dequantized
    weights: the loading, the fused shards (the full layers' q and k only), and every producer's wiring are exact."""
    fp8_dir, deq_dir = tiny
    seqs = [ids(70, 1), ids(23, 2)]
    _identity_quant(monkeypatch)
    ours = Direct(fp8_dir).logits(seqs)
    for i, s in enumerate(seqs):
        ref = hf_logits(deq_dir, s)
        start = sum(len(x) for x in seqs[:i])
        assert (ours[start:start + len(s)] - ref).abs().max().item() < 1e-4, i


def test_tiny_fp8_gemma_with_quantized_activations(tiny) -> None:
    """The quantized activations move this random tiny model's logits by a few percent RMS: a sanity bound only.
    Real-model fidelity is measured on the real checkpoint, not here (architecture.md, "Testing")."""
    fp8_dir, deq_dir = tiny
    s = ids(70, 1)
    ref = hf_logits(deq_dir, s)
    ours = Direct(fp8_dir).logits([s])
    rel = ((ours - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert 1e-4 < rel < 0.25, rel
    assert (ours.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.6


def test_producer_references_are_the_bf16_op_then_quant_ref() -> None:
    """The quantizing producers' definitions (the kernels must equal these bit for bit, on the GPU)."""
    torch.manual_seed(0)
    bf = torch.bfloat16
    x, res, w = torch.randn(5, 256, dtype=bf), torch.randn(5, 256, dtype=bf), torch.randn(256, dtype=bf)
    w2, sc = torch.randn(256, dtype=bf), torch.tensor([0.7], dtype=bf)
    a = gops.rmsnorm_quant(x, w, 1e-6)
    assert torch.equal(a.q.view(torch.uint8), fp8.quant_ref(gops.rmsnorm(x, w, 1e-6)).q.view(torch.uint8))
    r, q = gops.sandwich_quant(x, res, w, w2, 1e-6, sc)
    r2, x2 = gops.sandwich(x, res, w, w2, 1e-6, sc)
    assert torch.equal(r, r2) and torch.equal(q.s, fp8.quant_ref(x2).s)
    gu = torch.randn(5, 512, dtype=bf)
    assert torch.equal(gops.gelu_tanh_mul_quant(gu).q.view(torch.uint8),
                       fp8.quant_ref(gops.gelu_tanh_mul(gu)).q.view(torch.uint8))


def test_precise_rows_through_gemmas_runner(tiny) -> None:
    """PRECISE rows in GemmaRunner (the precise prefix, the shared `runner.cuda.fp8.Fp8Runtime`): such a row multiplies
    BF16 activations with the dequantized weights, so its logits are transformers' on the dequantized weights; the next
    chunk, not PRECISE, is quantized; with the precise prefix off the runner refuses PRECISE rows."""
    from llm_mini_infer.runner.base import PlanError

    fp8_dir, deq_dir = tiny
    seq = ids(70, 3)
    ref = hf_logits(deq_dir, seq)

    def run(precise: bool, mode: str = "reuse") -> torch.Tensor:
        pc = gemma_pools(budget=128, page_size=4)
        drv = GDriver(gemma_cpu_runner(fp8_dir, pc, fp8_cold_prefill=mode), pc, window=W)
        drv.run([dict(rows=[(1, seq[:40], 0, False, precise)], admit=[(1, GREEDY)]),
                 dict(rows=[(1, seq[40:], 40, True)])])
        assert fp8.activation_quant()  # the mode ends with the step
        return torch.stack([drv.logits[(1, p)] for p in range(len(seq))])

    pre, quant = run(True), run(False)
    assert (pre[:40] - ref[:40]).abs().max().item() < 1e-4  # the PRECISE chunk: no activation quantization
    assert (quant[:40] - ref[:40]).abs().max().item() > 1e-3  # an FP8 row quantizes it
    assert (pre[40:] - ref[40:]).abs().max().item() > 1e-3  # the next chunk is quantized again
    with pytest.raises(PlanError, match="PRECISE"):
        run(True, mode="fp8")
    with pytest.raises(ValueError, match="fp8_cold_prefill"):
        gemma_cpu_runner(fp8_dir, gemma_pools(), fp8_cold_prefill="int8")


def test_gemmas_runner_mixes_precise_and_fp8_rows(tiny) -> None:
    """Per-row precision in GemmaRunner: a step holding a PRECISE row (planned first) and an FP8 row computes the FP8
    row as the same step without the flag does, and the PRECISE row with BF16 activations."""
    fp8_dir, deq_dir = tiny
    a, b = ids(40, 4), ids(30, 6)

    def run(precise: bool) -> GDriver:
        pc = gemma_pools(budget=128, page_size=4)
        drv = GDriver(gemma_cpu_runner(fp8_dir, pc), pc, window=W)
        drv.run([dict(rows=[(2, b, 0, True, precise), (1, a, 0, True)], admit=[(1, GREEDY), (2, GREEDY)])])
        return drv

    mixed, plain = run(True), run(False)
    fa = torch.stack([mixed.logits[(1, p)] for p in range(40)])
    assert (fa - torch.stack([plain.logits[(1, p)] for p in range(40)])).abs().max().item() < 1e-5
    pb = torch.stack([mixed.logits[(2, p)] for p in range(30)])
    assert (pb - hf_logits(deq_dir, b)).abs().max().item() < 1e-4


@pytest.mark.parametrize("fused_ops", [True, False])
@pytest.mark.parametrize("attention", ["auto", "torch"])
def test_a_cuda_runner_without_the_kernel_library_fails_before_loading(tiny, monkeypatch, attention,
                                                                       fused_ops) -> None:
    """On CUDA the sampler has no torch fallback, so Gemma's runner refuses to start when the kernel library's lookup
    found nothing, naming why, before any weight loads (else it would load them, then fail at its first sampled step
    inside the memory profile; the tiny dims take torch attention, so the full-attention check does not fire). With
    dev.runner.cuda.fused_ops off too: the sampler needs the library all the same. CUDA simulated on the CPU (on a GPU
    host, the real device), nvcc and the prebuilt directories (whose record of nvcc's version would key the lookup)
    hidden, loading the weights a trap."""
    from llm_mini_infer.kernels import build as kbuild
    from llm_mini_infer.runner.cuda import gemma

    simulated_cuda(monkeypatch)
    fresh_kernel_lookup(monkeypatch)
    monkeypatch.setattr(kbuild, "find_nvcc", lambda: None)
    monkeypatch.delenv("LLM_MINI_INFER_KERNEL_PREBUILT", raising=False)
    monkeypatch.setattr(gemma, "build_model", lambda *args, **kw: pytest.fail("the weights started loading"))
    with pytest.raises(RuntimeError, match=r"needs llm-mini-infer's CUDA kernel library.*nvcc not found"):
        gemma.GemmaRunner(tiny[0], device="cuda", memory_gb=1.0, attention=attention, fused_ops=fused_ops)


def test_a_cuda_runner_refuses_fp32_with_our_kernels(tiny, monkeypatch) -> None:
    """Our kernels read and write BF16: Gemma's runner with an fp32 dtype and dev.runner.cuda.fused_ops on refuses to
    start, before any weight loads (the server refuses it at load, `config.FP32_KERNELS`). CUDA and the kernel
    library simulated on the CPU."""
    from llm_mini_infer.kernels import build as kbuild
    from llm_mini_infer.runner.cuda import gemma

    simulated_cuda(monkeypatch)
    monkeypatch.setattr(kbuild, "require", lambda what: None)
    monkeypatch.setattr(gemma, "build_model", lambda *args, **kw: pytest.fail("the weights started loading"))
    with pytest.raises(ValueError, match=r"needs dev\.runner\.cuda\.fused_ops false"):
        gemma.GemmaRunner(tiny[0], device="cuda", memory_gb=1.0, attention="torch", dtype=torch.float32)


@pytest.mark.parametrize("d", [96, 128])
def test_the_qkv_kernel_gets_only_the_head_dims_it_runs(monkeypatch, d) -> None:
    """lmk_g4_qkv runs head dims that are multiples of 64, up to 512, and returns -1 for others: `qkv_store` and
    `q_rope` send it only those, and take their torch references for the rest (96), where a -1 raised. The kernel
    library is a stand-in that answers as the kernel does."""
    calls = []
    lib = SimpleNamespace(lmk_g4_qkv=lambda *a: calls.append(a) or (-1 if a[-3] % 64 else 0))
    monkeypatch.setattr(gops.build, "load", lambda: lib)
    monkeypatch.setattr(gops, "_stream", lambda t: None)
    t, h, hkv, ps = 3, 2, 1, 4
    g = torch.Generator().manual_seed(0)
    qkv = torch.randn(t, (h + hkv) * d, generator=g).to(torch.bfloat16)
    w = torch.randn(d, generator=g).to(torch.bfloat16)
    pos = torch.arange(t)
    cos, sin = torch.randn(8, d // 2, generator=g), torch.randn(8, d // 2, generator=g)
    kv = torch.zeros(2, ps, hkv, d, dtype=torch.bfloat16)
    page, off = torch.zeros(t, dtype=torch.int64), pos.clone()

    def both():
        q = gops.q_rope(qkv[:, :h * d], heads=h, head_dim=d, q_w=w, eps=1e-6, positions=pos, cos_cache=cos,
                        sin_cache=sin)
        q2 = gops.qkv_store(qkv, heads=h, kv_heads=hkv, head_dim=d, k_eq_v=True, q_w=w, k_w=w, eps=1e-6,
                            positions=pos, cos_cache=cos, sin_cache=sin, kv_layer=kv, kv_page=page, kv_off=off)
        return q, q2

    want = both()  # use_kernels is false on the CPU: the references
    monkeypatch.setattr(gops, "use_kernels", lambda x: True)
    got = both()
    assert len(calls) == (0 if d % 64 else 2)
    if d % 64:
        assert all(torch.equal(a, b) for a, b in zip(got, want))

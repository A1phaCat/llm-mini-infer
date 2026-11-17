"""The Qwen3.5 model through the torch reference backends (CPU, fp32) against transformers' implementation."""

from __future__ import annotations

import json

import pytest
import torch

from llm_mini_infer.models.qwen3_5.model import Qwen35Config, map_weight_name

from ..support.fixtures import require_qwen
from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cpu_runner, hf_logits, make_tiny_model, pools
from ..support.token_ids import ids

TOL = 2e-5  # fp32 on both sides; logits are of order 1


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny"))


def logits_of(drv: Driver, slot: int, n: int) -> torch.Tensor:
    return torch.stack([drv.logits[(slot, p)] for p in range(n)])


@pytest.mark.parametrize(("prefix", "tie"), [(None, False), ("model.language_model.", False), (None, True)],
                         ids=["causal-lm", "conditional-generation-names", "tied-embeddings"])
def test_prefill_logits_match_transformers(tmp_path, prefix, tie) -> None:
    model = make_tiny_model(tmp_path / "m", prefix=prefix, tie=tie, seed=3)
    seq = ids(70)
    ref = hf_logits(model, seq) if prefix is None else None
    if prefix is not None:  # transformers loads the text-only names; compare against the same weights saved plainly
        ref = hf_logits(make_tiny_model(tmp_path / "plain", tie=tie, seed=3), seq)
    pc = pools(budget=128)
    drv = Driver(cpu_runner(model, pc), pc)
    res = drv.run([Step(rows=[Row(1, seq, 0)], admit=[(1, 1, GREEDY)])])
    ours = logits_of(drv, 1, len(seq))
    assert (ours - ref).abs().max().item() < TOL
    assert int(res[0].tokens[0, 0]) == int(ref[-1].argmax())


def test_chunked_prefill_and_decode_match_transformers(tiny) -> None:
    """Prefill in three uneven chunks (the GDN state and conv state carried across them, a 1-token chunk on the
    decode path), then 12 greedy decode steps from device input: every position matches one full forward."""
    seq = ids(70, seed=1)
    pc = pools(budget=64)
    drv = Driver(cpu_runner(tiny, pc), pc)
    steps = [Step(rows=[Row(1, seq[:13], 0, sample=False)], admit=[(1, 1, GREEDY)]),
             Step(rows=[Row(1, seq[13:14], 13, sample=False)]),
             Step(rows=[Row(1, seq[14:40], 14, sample=False)]),
             Step(rows=[Row(1, seq[40:], 40)])]
    steps += [Step(rows=[Row(1, None, 70 + i)]) for i in range(12)]
    res = drv.run(steps)
    gen = [int(r.tokens[0, 0]) for r in res[3:]]
    assert len(gen) == 13 and all(int(r.n_out[0]) == 0 for r in res[:3])
    full = seq + gen
    ref = hf_logits(tiny, full)
    ours = logits_of(drv, 1, len(full) - 1)
    assert (ours - ref[:-1]).abs().max().item() < TOL
    assert [int(x) for x in ref[69:82].argmax(-1)] == gen  # gen[i] was sampled from position 69 + i


def test_weight_name_mapping_and_geometry() -> None:
    hf = json.loads((require_qwen() / "config.json").read_text())
    assert Qwen35Config.from_hf(hf).quant is not None  # the local copy is the FP8 checkpoint's config (test_fp8.py)
    hf.pop("quantization_config")
    cfg = Qwen35Config.from_hf(hf)
    assert cfg.quant is None and not cfg.fp8_module("layers.3.self_attn.qkv_proj")
    assert (len(cfg.attn_layers), len(cfg.gdn_layers), cfg.rotary_dim, cfg.conv_dim) == (16, 48, 64, 10240)
    # without the MTP layer: KV bytes per token, 16 attention layers of 4 KV heads × 256 in BF16; then the GDN layers'
    # state bytes per state slot (the recurrent state in fp32, the conv state in BF16)
    assert len(cfg.attn_layers) * 2 * cfg.num_kv_heads * cfg.head_dim * 2 == 65_536
    rec = cfg.lin_v_heads * cfg.lin_v_dim * cfg.lin_k_dim * 4
    assert len(cfg.gdn_layers) * (rec + cfg.conv_dim * 3 * 2) == 153_944_064
    q = cfg.num_heads * cfg.head_dim * 2
    assert map_weight_name(cfg, "model.language_model.layers.3.self_attn.k_proj.weight") == \
        ("layers.3.self_attn.qkv_proj.weight", q)
    assert map_weight_name(cfg, "model.language_model.layers.0.linear_attn.in_proj_z.weight") == \
        ("layers.0.linear_attn.in_proj_qkvz.weight", cfg.conv_dim)
    assert map_weight_name(cfg, "model.language_model.layers.0.linear_attn.in_proj_a.weight") == \
        ("layers.0.linear_attn.in_proj_ba.weight", cfg.lin_v_heads)
    assert map_weight_name(cfg, "model.language_model.layers.5.mlp.up_proj.weight") == \
        ("layers.5.mlp.gate_up_proj.weight", cfg.intermediate_size)
    assert map_weight_name(cfg, "mtp.layers.0.mlp.up_proj.weight") is None
    assert map_weight_name(cfg, "model.visual.blocks.0.attn.qkv.weight") is None
    assert map_weight_name(cfg, "lm_head.weight") == ("lm_head.weight", 0)

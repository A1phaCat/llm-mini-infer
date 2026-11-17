"""The Gemma 4 model through the torch reference backends (CPU, fp32) against transformers' implementation, layer by
layer, on tiny random configs that keep the real structure: the 5 : 1-style layer pattern with full layers last, the
two head dims, k_eq_v, proportional RoPE, a (scaled-down) sliding window and the softcap."""

from __future__ import annotations

import json

import pytest
import torch

from llm_mini_infer.models.gemma4.model import FULL, SLIDING, Gemma4Config, inv_freq, map_weight_name

from ..support.paths import DATA
from ..support.tiny_gemma import Direct, hf_logits, hf_model, make_tiny_gemma

TOL = 5e-5  # fp32 on both sides; softcapped logits are of order 1-10


def ids(n: int, seed: int = 0, vocab: int = 500) -> list[int]:
    return [(i * 37 + 11 + seed * 101) % vocab for i in range(n)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma(tmp_path_factory.mktemp("g4"))


def test_logits_match_transformers_across_the_window(tiny) -> None:
    """Rows shorter than, equal to and several times the window (8), batched in one forward: every position's logits
    match HF's full forward of that sequence alone."""
    seqs = [ids(5, 1), ids(8, 2), ids(40, 3), ids(17, 4)]
    d = Direct(tiny)
    hidden, _ = d.forward(seqs)
    ours = d.model.logits(hidden)
    off = 0
    for s in seqs:
        ref = hf_logits(tiny, s)
        got = ours[off:off + len(s)]
        off += len(s)
        assert (got - ref).abs().max().item() < TOL, len(s)
        assert torch.equal(got.argmax(-1), ref.argmax(-1))


def test_every_layer_matches_transformers(tiny) -> None:
    """The residual stream after each decoder layer (layer scalar included) against HF's hidden states."""
    seq = ids(30, 5)
    d = Direct(tiny)
    _, layers = d.forward([seq])
    model = hf_model(tiny)
    with torch.no_grad():
        out = model(torch.tensor([seq]), output_hidden_states=True)
    ref = out.hidden_states  # embeddings, then one per layer (the last after the final norm in HF: compared apart)
    assert len(layers) == len(ref) - 1
    for i, ours in enumerate(layers[:-1]):
        err = (ours - ref[i + 1][0]).abs().max().item()
        assert err < 1e-4 * max(1.0, ref[i + 1].abs().max().item()), (i, err)


@pytest.mark.parametrize("window", [1, 3, 16])
def test_window_sizes(tmp_path, window) -> None:
    """A window of 1 (a key sees only its own position), of a few tokens and wider than the sequence."""
    m = make_tiny_gemma(tmp_path / "m", window=window, seed=window)
    seq = ids(12, window)
    got = Direct(m).logits([seq])
    assert (got - hf_logits(m, seq)).abs().max().item() < TOL


def test_conditional_generation_names_and_config(tmp_path) -> None:
    """The real checkpoint's layout (model.language_model.*, text_config with global_head_dim) loads the same model."""
    plain = make_tiny_gemma(tmp_path / "plain", seed=7)
    cg = make_tiny_gemma(tmp_path / "cg", seed=7, prefix="model.language_model.")
    seq = ids(20, 9)
    a, b = Direct(plain).logits([seq]), Direct(cg).logits([seq])
    assert torch.equal(a, b)
    assert (a - hf_logits(plain, seq)).abs().max().item() < TOL


def test_without_k_eq_v_and_softcap(tmp_path) -> None:
    m = make_tiny_gemma(tmp_path / "m", k_eq_v=False, softcap=None, seed=11)
    seq = ids(25, 2)
    assert (Direct(m).logits([seq]) - hf_logits(m, seq)).abs().max().item() < TOL


def test_real_config_geometry() -> None:
    hf = json.loads((DATA / "gemma-4-31b-it.config.json").read_text())
    cfg = Gemma4Config.from_hf(hf)
    assert (cfg.num_layers, len(cfg.sliding_layers), len(cfg.full_layers)) == (60, 50, 10)
    assert cfg.full_layers == list(range(5, 60, 6))
    assert (cfg.sliding_head_dim, cfg.sliding_kv_heads, cfg.full_head_dim, cfg.full_kv_heads) == (256, 16, 512, 4)
    assert (cfg.sliding_window, cfg.full_rope_angles, cfg.k_eq_v, cfg.final_logit_softcapping) == (1024, 64, True, 30.0)
    # KV per token in BF16: sliding 50 x 16 KB, full 10 x 4 KB (one plane: k_eq_v)
    assert cfg.kv_bytes_per_token(SLIDING, 2) == 50 * 16 * 1024
    assert cfg.kv_bytes_per_token(FULL, 2) == 10 * 4 * 1024
    f = inv_freq(cfg, FULL)
    assert f.shape == (256,) and bool((f[:64] > 0).all()) and bool((f[64:] == 0).all())
    assert torch.equal(f[:64], 1.0 / (1e6 ** (torch.arange(0, 128, 2).float() / 512)))
    assert map_weight_name(cfg, "model.language_model.layers.5.self_attn.k_proj.weight") == \
        ("layers.5.self_attn.qkv_proj.weight", 32 * 512)
    assert map_weight_name(cfg, "model.language_model.layers.4.self_attn.v_proj.weight") == \
        ("layers.4.self_attn.qkv_proj.weight", 32 * 256 + 16 * 256)
    with pytest.raises(ValueError, match="v_proj on a full-attention layer"):
        map_weight_name(cfg, "model.language_model.layers.5.self_attn.v_proj.weight")
    assert map_weight_name(cfg, "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight") is None
    assert map_weight_name(cfg, "model.language_model.layers.3.layer_scalar") == ("layers.3.layer_scalar", 0)
    assert map_weight_name(cfg, "model.language_model.layers.3.mlp.up_proj.weight") == \
        ("layers.3.mlp.gate_up_proj.weight", 21_504)


def test_unsupported_variants_are_refused() -> None:
    hf = json.loads((DATA / "gemma-4-31b-it.config.json").read_text())
    for key, value in (("enable_moe_block", True), ("hidden_size_per_layer_input", 256), ("num_kv_shared_layers", 4),
                       ("use_bidirectional_attention", "all")):
        bad = json.loads(json.dumps(hf))
        bad["text_config"][key] = value
        with pytest.raises(NotImplementedError):
            Gemma4Config.from_hf(bad)

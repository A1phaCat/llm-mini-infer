"""Gemma 4's MTP assistant (`models.gemma4.assistant`) against transformers' `Gemma4AssistantForCausalLM` on tiny
random models (CPU, fp32): the config, the weights, and one draft step and a chained one, at positions before and past
the sliding window, over the target's KV as our target model wrote it.

The reference is HF's assistant forward given the drafting semantics of `runner.cuda.gemma`: the query at the last
committed position p, the target's KV of [0, p] in the full layer and of [p − W + 1, p] in the sliding ones (HF's own
bidirectional sliding mask would let it see W + 1 keys); and, as `dev.speculative.draft_query_next` places it, the query
at p + 1 (HF's generation loop's position) over the same KV."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from llm_mini_infer.models.gemma4.assistant import AssistantConfig, build_assistant
from llm_mini_infer.models.gemma4.backend import Gemma4Context, Gemma4Meta
from llm_mini_infer.models.gemma4.model import FULL, SLIDING, Gemma4Config

from ..support.tiny_gemma import Direct, hf_assistant, hf_model, make_tiny_assistant, make_tiny_gemma

W = 8
TOL = 5e-5  # fp32 on both sides, relative to the larger magnitude; x86's BLAS reaches about 2.03e-5

# google/gemma-4-31B-it-assistant's config.json, its text_config trimmed to the keys that matter
REAL = {
    "architectures": ["Gemma4AssistantForCausalLM"], "backbone_hidden_size": 5376, "centroid_intermediate_top_k": 32,
    "dtype": "bfloat16", "model_type": "gemma4_assistant", "num_centroids": 2048, "tie_word_embeddings": True,
    "use_ordered_embeddings": False,
    "text_config": {
        "attention_bias": False, "attention_k_eq_v": True, "enable_moe_block": False, "eos_token_id": 1,
        "final_logit_softcapping": None, "global_head_dim": 512, "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1024, "hidden_size_per_layer_input": 0, "intermediate_size": 8192,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 262144, "model_type": "gemma4_text", "num_attention_heads": 32,
        "num_global_key_value_heads": 4, "num_hidden_layers": 4, "num_key_value_heads": 16, "num_kv_shared_layers": 4,
        "rms_norm_eps": 1e-06, "sliding_window": 1024, "tie_word_embeddings": True, "use_double_wide_mlp": False,
        "vocab_size": 262144, "vocab_size_per_layer_input": 0,
        "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                               "rope_type": "proportional"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}
# google/gemma-4-31B-it's text config (the target), as models.gemma4.model reads it
TARGET = {"model_type": "gemma4", "text_config": {
    "model_type": "gemma4_text", "vocab_size": 262144, "hidden_size": 5376, "intermediate_size": 21504,
    "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
    "global_head_dim": 512, "num_global_key_value_heads": 4, "attention_k_eq_v": True, "sliding_window": 1024,
    "final_logit_softcapping": 30.0, "rope_parameters": REAL["text_config"]["rope_parameters"]}}


def test_the_31b_assistant_config_fits_its_target() -> None:
    a = AssistantConfig.from_hf(REAL)
    t = Gemma4Config.from_hf(TARGET)
    a.check_target(t)
    assert a.layer_types == (SLIDING, SLIDING, SLIDING, FULL) and a.hidden_size == 1024 and a.num_heads == 32
    assert (a.sliding_head_dim, a.full_head_dim, a.backbone_hidden_size) == (256, 512, 5376)
    assert a.final_logit_softcapping is None and a.tie_word_embeddings and a.sliding_window == 1024
    # its layers read the target's last layer of each kind: 58 (the 50th sliding) and 59 (the 10th full)
    assert max(i for i, k in enumerate(t.layer_types) if k == SLIDING) == 58 and t.layer_types[59] == FULL
    with pytest.raises(NotImplementedError, match="centroid"):
        AssistantConfig.from_hf(dict(REAL, use_ordered_embeddings=True))
    bad = json.loads(json.dumps(TARGET))
    bad["text_config"]["hidden_size"] = 4096
    with pytest.raises(ValueError, match="backbone"):
        a.check_target(Gemma4Config.from_hf(bad))


def test_the_assistant_takes_the_targets_defaults() -> None:
    """Keys a config may leave out default as the target's (`Gemma4Config.from_hf`, HF's): rope_parameters without a
    full-attention entry (proportional, 0.25 of the head rotated) and no global_head_dim (512, unless a
    per_layer_config sets it)."""
    a, t = json.loads(json.dumps(REAL)), json.loads(json.dumps(TARGET))
    for c in (a, t):
        c["text_config"]["rope_parameters"] = {SLIDING: REAL["text_config"]["rope_parameters"][SLIDING]}
    AssistantConfig.from_hf(a).check_target(Gemma4Config.from_hf(t))
    for c in (a, t):  # an entry without partial_rotary_factor: 1.0 on both sides
        c["text_config"]["rope_parameters"][FULL] = {"rope_type": "default", "rope_theta": 1e6}
    AssistantConfig.from_hf(a).check_target(Gemma4Config.from_hf(t))
    del a["text_config"]["global_head_dim"]
    assert AssistantConfig.from_hf(a).full_head_dim == 512
    a["text_config"].update(global_head_dim=None, per_layer_config={FULL: {"head_dim": 384}})
    assert AssistantConfig.from_hf(a).full_head_dim == 384


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("g4a")
    target = make_tiny_gemma(d / "target", window=W, seed=3)
    return target, make_tiny_assistant(d / "assistant", target, seed=4)


def drafter_meta(direct: Direct, positions: list[int], shift: int = 0) -> Gemma4Meta:
    """One drafter query per position p of sequence 0 (Direct's slot 0): KV [0, p]; the window's pages from 0 (the
    reference masks keys older than W − 1 positions before p); RoPE at p + `shift`."""
    n = len(positions)
    p = np.asarray(positions, np.int64)
    return Gemma4Meta(num_tokens=n, positions=torch.as_tensor(p + shift), f_page=None, f_off=None, w_page=None,
                      w_off=None,
                      n_rows=n, row_start_host=np.arange(n), row_nq_host=np.ones(n, np.int64),
                      row_kv_len_host=p + 1, row_slot_host=np.zeros(n, np.int64), row_wlo_host=np.zeros(n, np.int64),
                      page_table=direct.table, wpage_table=direct.table)


@pytest.mark.parametrize("shift", [0, 1], ids=["p", "p+1"])
def test_draft_steps_match_transformers(pair, shift) -> None:
    """A 40-token context (5 windows): at positions before and past the window, our assistant's first step (the
    target's embedding of the token after p and its hidden at p) and a chained second step (the first draft and the
    first step's backbone hidden) give HF's logits and backbone hidden states, the queries at p or at p + 1 (both
    sides)."""
    target_dir, asst_dir = pair
    seq = [(i * 37 + 11) % 500 for i in range(40)]
    direct = Direct(target_dir)
    hidden, _ = direct.forward([seq])
    ours = build_assistant(asst_dir, direct.cfg, direct.dev, torch.float32)
    assert ours.share(direct.model.layers, direct.cfg) == [4, 4, 4, 5]  # the tiny target's last sliding and full
    ht, ha = hf_model(target_dir), hf_assistant(asst_dir)
    ids = torch.tensor([seq])
    with torch.no_grad():
        out = ht.model(ids, return_shared_kv_states=True)
    kv = out.shared_kv_states
    positions = [2, 7, 8, 15, 23, 38]
    nxt = torch.as_tensor([seq[p + 1] for p in positions])
    meta = drafter_meta(direct, positions, shift)
    ctx = Gemma4Context(meta, direct.sliding, direct.full, direct.rotary, direct.pools)
    with torch.no_grad():
        d1, b1 = ours(direct.model.embed(nxt), hidden[positions], ctx)
        lg1 = ours.logits(d1)
        draft = lg1.argmax(-1)
        d2, b2 = ours(direct.model.embed(draft), b1, ctx)
        lg2 = ours.logits(d2)
    for i, p in enumerate(positions):
        lo = max(0, p - W + 1)
        shared = {FULL: (kv[FULL][0][:, :, :p + 1], kv[FULL][1][:, :, :p + 1]),
                  SLIDING: (kv[SLIDING][0][:, :, lo:p + 1], kv[SLIDING][1][:, :, lo:p + 1])}
        pos = torch.tensor([[p + shift]])
        with torch.no_grad():
            e1 = ht.model.embed_tokens(torch.tensor([[seq[p + 1]]]))
            r1 = ha(inputs_embeds=torch.cat([e1, out.last_hidden_state[:, p:p + 1]], -1), position_ids=pos,
                    shared_kv_states=shared)
            e2 = ht.model.embed_tokens(draft[i].view(1, 1))
            r2 = ha(inputs_embeds=torch.cat([e2, r1.last_hidden_state], -1), position_ids=pos, shared_kv_states=shared)
        for got, want, what in ((lg1[i], r1.logits[0, 0], "logits"), (b1[i], r1.last_hidden_state[0, 0], "backbone"),
                                (lg2[i], r2.logits[0, 0], "chained logits"),
                                (b2[i], r2.last_hidden_state[0, 0], "chained backbone")):
            err = (got - want).abs().max().item() / max(1.0, want.abs().max().item())
            assert err < TOL, (p, what, err)
        assert int(r1.logits[0, 0].argmax()) == int(draft[i])

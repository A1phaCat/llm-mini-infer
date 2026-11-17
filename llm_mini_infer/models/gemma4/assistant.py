# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Team. All rights reserved.
# The proportional RoPE frequencies are adapted from src/transformers/modeling_rope_utils.py of transformers 5.17.0
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""Gemma 4's MTP assistant (`Gemma4AssistantForCausalLM`, `google/gemma-4-31B-it-assistant`; docs/architecture.md,
"Speculative decoding") as plain `nn.Module`s: a small decoder that drafts from the target's own hidden state and reads
the target's KV, writing none.

The 31B's assistant: 4 decoder layers (3 sliding, 1 full), hidden 1,024, a GELU-tanh MLP of 8,192, 32 query heads of
256 (sliding) and 512 (full), BF16, 0.87 GiB. HF's assistant forward, given positions and KV, is the reference
(tests/models/test_gemma4_assistant.py); the serving engines' drafting semantics are followed (`runner.cuda.gemma`).

**One draft step:** the input is the target's scaled embedding of the last token (the target's `embed_tokens` ×
bf16(sqrt(5,376)) = 73.5, as HF) concatenated with a backbone-width hidden state, `pre_projection` (10,752 →
1,024), the decoder layers, the final norm (the draft hidden, which the head scores), and `post_projection` (1,024 →
5,376: the backbone hidden the next step takes).

**Q-only layers, the target's KV** (`num_kv_shared_layers` = all 4). Each assistant layer has q_proj, q_norm, o_proj
and an MLP; its keys and values are those of the target's **last layer of the same type**: on Gemma 4 31B the sliding
layers read layer 58's window pages and the full layer reads layer 59's V plane, with layer 59's k_norm deriving K =
RoPE(V ⊙ k_norm.weight) as for the target itself (`attention_k_eq_v`). The query heads (32) divide among the target's
KV heads (16 sliding, 4 full) as the target's own do. Attention scale 1.0, the sliding window the target's.

**The head**: the assistant's own embedding table (or an untied `lm_head`) over the whole vocabulary, softcapped when
`final_logit_softcapping` is set. A checkpoint with the centroid head (`use_ordered_embeddings`, the E2B/E4B
assistants') is refused at load.

**Numerics** as the target's (models/gemma4/model.py): every norm in fp32, rounded once; the
sandwich (post norm, residual add, layer scalar) rounded once; q rounded once after its norm and RoPE; GELU-tanh × up
rounded once; the head's GEMM in the model dtype, widened to fp32 for the sampler.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...kernels import gemma_ops as gops
from ..common import Coverage
from .model import DEFAULT_ROPE, FULL, SLIDING, Gemma4Config, RMSNorm, inv_freq, linear

if TYPE_CHECKING:
    from .backend import Gemma4Context

log = logging.getLogger(__name__)

MODEL_TYPES = ("gemma4_assistant", "gemma4_unified_assistant")


@dataclass(frozen=True)
class AssistantConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    num_heads: int
    sliding_head_dim: int
    full_head_dim: int
    backbone_hidden_size: int
    sliding_window: int
    rms_norm_eps: float
    final_logit_softcapping: float | None
    tie_word_embeddings: bool
    rope: dict[str, dict[str, Any]]  # rope_parameters by layer type, as in config.json
    eos_token_ids: tuple[int, ...] = ()

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    def head_dim(self, kind: str) -> int:
        return self.sliding_head_dim if kind == SLIDING else self.full_head_dim

    @classmethod
    def from_hf(cls, hf: dict[str, Any]) -> AssistantConfig:
        """From the assistant's `config.json` (`Gemma4AssistantForCausalLM`, `text_config` of type gemma4_text). Raises
        NotImplementedError for what this build does not serve (the centroid head, per-layer inputs, MoE)."""
        if hf.get("model_type") not in MODEL_TYPES:
            raise NotImplementedError(f"model_type {hf.get('model_type')!r} is not a Gemma 4 assistant")
        if hf.get("use_ordered_embeddings"):
            raise NotImplementedError("the centroid head (use_ordered_embeddings: the E2B/E4B assistants')")
        tc = hf.get("text_config") or {}
        for key, what in (("enable_moe_block", "MoE"), ("use_double_wide_mlp", "the double-wide MLP"),
                          ("hidden_size_per_layer_input", "per-layer inputs"), ("attention_bias", "attention bias"),
                          ("attn_logit_softcapping", "attention logit softcapping")):
            if tc.get(key):
                raise NotImplementedError(f"a Gemma 4 assistant with {what} ({key})")
        if tc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"hidden_activation {tc['hidden_activation']!r}")
        n = int(tc["num_hidden_layers"])
        if int(tc.get("num_kv_shared_layers") or n) != n:
            raise NotImplementedError("an assistant whose layers do not all share the target's KV")
        types = tuple(tc.get("layer_types") or [])
        if len(types) != n or set(types) - {SLIDING, FULL}:
            raise ValueError(f"bad assistant layer_types {types}")
        eos = tc.get("eos_token_id")
        full_head = int(tc.get("global_head_dim") or 512)  # absent or null: HF's default, as the target's
        for k, v in (tc.get("per_layer_config") or {}).items():  # save_pretrained's form of global_head_dim
            kind = types[int(k)] if str(k).isdigit() else k
            if kind != FULL:
                raise NotImplementedError(f"per_layer_config for the assistant's {kind} layers")
            full_head = int(v.get("head_dim", full_head))
        return cls(vocab_size=int(tc["vocab_size"]), hidden_size=int(tc["hidden_size"]),
                   intermediate_size=int(tc["intermediate_size"]), layer_types=types,
                   num_heads=int(tc["num_attention_heads"]), sliding_head_dim=int(tc.get("head_dim", 256)),
                   full_head_dim=full_head,
                   backbone_hidden_size=int(hf["backbone_hidden_size"]),
                   sliding_window=int(tc.get("sliding_window", 512)), rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
                   final_logit_softcapping=tc.get("final_logit_softcapping"),
                   tie_word_embeddings=bool(hf.get("tie_word_embeddings", tc.get("tie_word_embeddings", True))),
                   rope=dict(tc.get("rope_parameters") or {}),
                   eos_token_ids=tuple(eos if isinstance(eos, list) else [eos]) if eos is not None else ())

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> AssistantConfig:
        return cls.from_hf(json.loads((Path(model_dir) / "config.json").read_text()))

    def check_target(self, t: Gemma4Config) -> None:
        """The assistant must fit its target: its backbone width and vocabulary, the target's head dims and window, its
        query heads a multiple of the target's KV heads, and the target's RoPE per layer type (its queries meet the
        target's keys)."""
        why = []
        if self.backbone_hidden_size != t.hidden_size:
            why.append(f"backbone_hidden_size {self.backbone_hidden_size} != the target's hidden {t.hidden_size}")
        if self.vocab_size != t.vocab_size:
            why.append(f"vocabulary {self.vocab_size} != the target's {t.vocab_size}")
        for kind in set(self.layer_types):
            if kind not in t.layer_types:
                why.append(f"the target has no {kind} layer to share")
            if self.head_dim(kind) != t.head_dim(kind):
                why.append(f"{kind} head dim {self.head_dim(kind)} != the target's {t.head_dim(kind)}")
            if self.num_heads % t.kv_heads(kind):
                why.append(f"{self.num_heads} query heads do not divide among the target's {t.kv_heads(kind)} "
                           f"{kind} KV heads")
        if SLIDING in self.layer_types and self.sliding_window != t.sliding_window:
            why.append(f"sliding window {self.sliding_window} != the target's {t.sliding_window}")
        for kind in set(self.layer_types):
            if not torch.equal(inv_freq_of(self, kind), inv_freq(t, kind)):
                why.append(f"{kind} RoPE {self.rope.get(kind)} differs from the target's")
        if why:
            raise ValueError("the Gemma 4 assistant does not fit its target: " + "; ".join(why))


def inv_freq_of(a: AssistantConfig, kind: str) -> torch.Tensor:
    """The assistant's RoPE inverse frequencies for a layer kind (models.gemma4.model.inv_freq on its own
    rope_parameters)."""
    rp = a.rope.get(kind) or DEFAULT_ROPE[kind]  # as the target's (`Gemma4Config.from_hf`)
    d = a.head_dim(kind)
    if kind == SLIDING:
        if rp.get("rope_type", "default") != "default" or rp.get("partial_rotary_factor", 1.0) != 1.0:
            raise NotImplementedError(f"assistant sliding RoPE {rp}")
        theta = float(rp.get("rope_theta", 10_000.0))
        return 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
    if rp.get("rope_type", "proportional") not in ("proportional", "default") or rp.get("factor", 1.0) != 1.0:
        raise NotImplementedError(f"assistant full-attention RoPE {rp}")
    theta = float(rp.get("rope_theta", 1_000_000.0))
    a_ = int(float(rp.get("partial_rotary_factor", 1.0)) * d // 2)
    rot = 1.0 / (theta ** (torch.arange(0, 2 * a_, 2, dtype=torch.int64).to(torch.float) / d))
    return torch.cat([rot, torch.zeros(d // 2 - a_, dtype=torch.float32)]) if d // 2 > a_ else rot


# -- layers ------------------------------------------------------------------------------------------------------


class SharedAttention(nn.Module):
    """A Q-only attention layer over one of the target's layers' KV: `index` is that layer's place among the target's
    layers of its kind (its KV pool layer), `k_norm` its k_norm (a full layer derives its keys with it), set by
    `Gemma4Assistant.share`."""

    def __init__(self, cfg: AssistantConfig, kind: str, hkv: int, dtype: torch.dtype):
        super().__init__()
        self.kind = kind
        self.h, self.d, self.hkv = cfg.num_heads, cfg.head_dim(kind), hkv
        self.k_eq_v = kind == FULL
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps, dtype)
        self.eps = cfg.rms_norm_eps
        self.index = -1
        self.target: list[nn.Module] = []  # the target layer's attention: not a submodule (its weights)

    @property
    def k_norm(self) -> nn.Module:
        return self.target[0].k_norm

    def forward(self, x: torch.Tensor, ctx: Gemma4Context) -> torch.Tensor:
        q = ctx.q_rope(self, linear(x, self.q_proj.weight))  # q_norm, RoPE at the drafter's positions; no KV write
        return linear(ctx.attention(self, q), self.o_proj.weight)


class AssistantMLP(nn.Module):
    def __init__(self, cfg: AssistantConfig, dtype: torch.dtype):
        super().__init__()
        i = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * i, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(i, cfg.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(gops.gelu_tanh_mul(linear(x, self.gate_up_proj.weight)), self.down_proj.weight)


class AssistantLayer(nn.Module):
    def __init__(self, cfg: AssistantConfig, kind: str, hkv: int, dtype: torch.dtype):
        super().__init__()
        self.self_attn = SharedAttention(cfg, kind, hkv, dtype)
        self.mlp = AssistantMLP(cfg, dtype)
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(h, eps, dtype)
        self.post_attention_layernorm = RMSNorm(h, eps, dtype)
        self.pre_feedforward_layernorm = RMSNorm(h, eps, dtype)
        self.post_feedforward_layernorm = RMSNorm(h, eps, dtype)
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.eps = eps


class Gemma4Assistant(nn.Module):
    def __init__(self, cfg: AssistantConfig, target: Gemma4Config, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.pre_projection = nn.Linear(2 * cfg.backbone_hidden_size, cfg.hidden_size, bias=False, dtype=dtype)
        self.layers = nn.ModuleList(AssistantLayer(cfg, kind, target.kv_heads(kind), dtype)
                                    for kind in cfg.layer_types)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.post_projection = nn.Linear(cfg.hidden_size, cfg.backbone_hidden_size, bias=False, dtype=dtype)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)  # the head (tied)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)

    @property
    def head_weight(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight

    def share(self, target_layers: nn.ModuleList, target_cfg: Gemma4Config) -> list[int]:
        """Point each layer at the target's last layer of its kind (module docstring); returns their indices."""
        last = {kind: max(i for i, t in enumerate(target_cfg.layer_types) if t == kind) for kind in (SLIDING, FULL)}
        out = []
        for layer in self.layers:
            a = layer.self_attn
            t = target_layers[last[a.kind]].self_attn
            a.index, a.target = t.index, [t]
            out.append(last[a.kind])
        return out

    def forward(self, emb: torch.Tensor, hidden: torch.Tensor, ctx: Gemma4Context) -> tuple[torch.Tensor, torch.Tensor]:
        """emb: the target's scaled embeddings of the input tokens [n, backbone]; hidden: the target's (or the previous
        step's backbone) hidden states [n, backbone]. Returns (the draft hidden [n, hidden], the backbone hidden
        [n, backbone])."""
        residual = linear(torch.cat([emb, hidden], dim=-1), self.pre_projection.weight)
        layers = self.layers
        x = layers[0].input_layernorm(residual)
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            a = layer.self_attn(x, ctx)
            residual, h = gops.sandwich(a, residual, layer.post_attention_layernorm.weight,
                                        layer.pre_feedforward_layernorm.weight, layer.eps)
            m = layer.mlp(h)
            nxt = self.norm.weight if i == last else layers[i + 1].input_layernorm.weight
            residual, x = gops.sandwich(m, residual, layer.post_feedforward_layernorm.weight, nxt, layer.eps,
                                        scalar=layer.layer_scalar)
        return x, linear(x, self.post_projection.weight)

    def logits(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        """[n, V] fp32: the head's GEMM in the model dtype (module docstring, "The head"), widened. The reference
        (tests): the CUDA runner scores drafts with `runner.cuda.draft_head.DraftHead`, cuBLAS at every M."""
        lg = linear(draft_hidden, self.head_weight)
        cap = self.cfg.final_logit_softcapping
        if cap:
            lg = torch.tanh(lg / cap) * cap
        return lg.float()


# -- weights -----------------------------------------------------------------------------------------------------

_FUSED = {"mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0), "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", 1)}


def map_weight_name(cfg: AssistantConfig, name: str) -> tuple[str, int] | None:
    """Checkpoint name -> (parameter name, which half of gate_up_proj or 0), or None for weights not loaded."""
    if name in ("lm_head.weight", "model.lm_head.weight"):
        return None if cfg.tie_word_embeddings else ("lm_head.weight", 0)
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("layers."):
        _, idx, rest = name.split(".", 2)
        if int(idx) >= cfg.num_layers:
            raise ValueError(f"{name}: layer {idx} beyond the assistant's {cfg.num_layers}")
        if rest in _FUSED:
            p, half = _FUSED[rest]
            return f"layers.{idx}.{p}", half
        return f"layers.{idx}.{rest}", 0
    return name, 0


@torch.inference_mode()
def load_weights(model: Gemma4Assistant, model_dir: Path, device: torch.device) -> int:
    from safetensors import safe_open

    from .model import checkpoint_files

    params = dict(model.named_parameters())
    cover = Coverage()
    i = model.cfg.intermediate_size
    unknown: list[str] = []
    dev = str(device) if device.type == "cpu" else f"cuda:{device.index or 0}"
    n = 0
    for path in checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt", device=dev) as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                target = map_weight_name(model.cfg, name)
                if target is None:
                    continue
                pname, half = target
                param = params.get(pname)
                if param is None:
                    unknown.append(name)
                    continue
                t = f.get_tensor(name)
                if t.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                    raise NotImplementedError(f"{name}: {t.dtype} (BF16, FP16 or FP32 only)")
                if pname.endswith("layer_scalar"):
                    t = t.reshape(1)
                dst = param.data.narrow(0, half * i, i) if pname.endswith("gate_up_proj.weight") else param.data
                if dst.shape != t.shape:
                    raise ValueError(f"{name} {tuple(t.shape)} does not fit {pname} {tuple(dst.shape)}")
                dst.copy_(t)
                cover.add(pname, half * i if pname.endswith("gate_up_proj.weight") else 0, t.shape[0] if t.dim() else 1)
                n += 1
    if unknown:
        raise ValueError(f"unexpected assistant weights: {unknown[:8]}{' ...' if len(unknown) > 8 else ''}")
    missing = cover.incomplete(params)
    if missing:
        raise ValueError(f"assistant weights missing in the checkpoint: {missing[:8]}")
    return n


def build_assistant(model_dir: str | Path, target_cfg: Gemma4Config, device: torch.device,
                    dtype: torch.dtype) -> Gemma4Assistant:
    """Construct on the meta device, allocate on `device`, load the checkpoint (BF16, as Google ships it)."""
    model_dir = Path(model_dir)
    cfg = AssistantConfig.from_dir(model_dir)
    cfg.check_target(target_cfg)
    with torch.device("meta"):
        model = Gemma4Assistant(cfg, target_cfg, dtype)
    model = model.to_empty(device=device).requires_grad_(False)
    n = load_weights(model, model_dir, device)
    log.info("loaded the Gemma 4 assistant from %s: %d tensors, %d layers (%s), hidden %d, %.2f GiB", model_dir, n,
             cfg.num_layers, " ".join("S" if t == SLIDING else "F" for t in cfg.layer_types), cfg.hidden_size,
             sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30)
    return model.eval()

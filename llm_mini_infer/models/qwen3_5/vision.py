# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# merge_order and interp_taps are adapted from src/transformers/vision_utils.py of transformers 5.17.0
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""The Qwen3.5 / Qwen3.8 vision tower (`model.visual.*`) as plain `nn.Module`s, numerics after transformers'
`Qwen3_5VisionModel`, which is the reference the embeddings are tested against.

- **Pixels to patches, on the device** (`patches`): the image pool's uint8 pixels [3, H, W] are rescaled and normalised
  with the processor's fused arithmetic (bit-identical to HF's `pixel_values`), then cut into 16 x 16 patches in the
  spatial-merge order (2 x 2 blocks, row-major), each duplicated over the temporal patch (2) and flattened as (channel,
  time, row, column): [N, 1536].
- **Patch embedding:** the checkpoint's Conv3d (kernel = stride = (2, 16, 16)) is the same map as a linear layer over
  those 1,536 values; it runs as one, on the patches rounded to BF16 (HF casts them to the weights' dtype).
- **Position embedding:** the learned 48 x 48 grid, resampled bilinearly to each image's patch grid as HF does
  (`interp_taps`), weighted and summed in fp32, rounded once, then added.
- **27 blocks:** pre-LayerNorm attention and MLP with residual adds in BF16. Attention is bidirectional within each
  image (a ragged batch, `cu_seqlens`); q and k take the axial 2-D RoPE in fp32, rounded once, in one kernel per block
  (`ops.vision_qkv_rope`, which also pads q, k and v to the attention kernel's head dim). The MLP is GELU-tanh.
- **The merger:** LayerNorm per patch, then each 2 x 2 block's four patches concatenated (4,608), a linear layer, exact
  GELU and a linear layer to the language model's width (5,120): one embedding per image token.

**Attention backends:** `TorchVisionAttention` runs SDPA per image (the reference: what HF's SDPA path computes on
CPU); `kernels.flashinfer.FlashInferVisionAttention` runs FlashInfer's ragged prefill kernel, non-causal, over the
encoder call's images. The projections, LayerNorms, GELU and the merger are torch ops on BF16 tensors, as in HF.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import ops
from ..common import rope_inv_freq, to_device_int32

GRID_CACHE = 256  # patch grids whose per-patch metadata stays on the device


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden: int
    heads: int
    intermediate: int
    patch: int
    temporal: int
    merge: int
    out_hidden: int
    num_pos: int
    in_channels: int
    act: str
    rope_theta: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    rescale: float

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal * self.patch * self.patch

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> VisionConfig:
        model_dir = Path(model_dir)
        hf = json.loads((model_dir / "config.json").read_text())
        vc = hf["vision_config"]
        pre_path = model_dir / "preprocessor_config.json"
        pre = json.loads(pre_path.read_text()) if pre_path.exists() else {}
        rope = vc.get("rope_parameters") or {}
        if vc.get("deepstack_visual_indexes"):
            raise NotImplementedError("deepstack vision features (Qwen3-VL) are not supported")
        return cls(depth=int(vc["depth"]), hidden=int(vc["hidden_size"]), heads=int(vc["num_heads"]),
                   intermediate=int(vc["intermediate_size"]), patch=int(vc.get("patch_size", 16)),
                   temporal=int(vc.get("temporal_patch_size", 2)), merge=int(vc.get("spatial_merge_size", 2)),
                   out_hidden=int(vc["out_hidden_size"]), num_pos=int(vc.get("num_position_embeddings", 2304)),
                   in_channels=int(vc.get("in_channels", 3)), act=vc.get("hidden_act", "gelu_pytorch_tanh"),
                   rope_theta=float(rope.get("rope_theta", 10000.0)),
                   mean=tuple(pre.get("image_mean", (0.5, 0.5, 0.5))), std=tuple(pre.get("image_std", (0.5, 0.5, 0.5))),
                   rescale=float(pre.get("rescale_factor", 1 / 255)))


# -- per-image metadata (host ints in, device tensors out) -------------------------------------------------------------


def merge_order(h: int, w: int, merge: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, column) of each patch of an h x w patch grid, in the spatial-merge order: 2 x 2 blocks row-major, each
    block's patches row-major (HF's `get_vision_position_ids`)."""
    hh, ww = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    return hh.reshape(shape).transpose(1, 2).reshape(-1), ww.reshape(shape).transpose(1, 2).reshape(-1)


def interp_taps(rows: torch.Tensor, cols: torch.Tensor, h: int, w: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (align_corners) taps into the side x side position grid and their fp32 weights, [N, 4] each: HF's
    `get_vision_interpolation_indices_and_weights` for one image."""

    def axis(index: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = index.to(torch.float32)
        # a tensor divisor, as HF's: on CUDA torch divides by a Python scalar through its reciprocal
        src = index * (side - 1) / torch.full_like(index, max(size - 1, 1))
        floor = torch.floor(src)
        offsets = torch.arange(0, 2, device=index.device)
        taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
        dist = (src[:, None] - floor[:, None] - offsets).abs()
        return taps, (1 - dist).clamp(min=0)

    ht, hw_ = axis(rows, h)
    wt, ww_ = axis(cols, w)
    idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
    wts = (hw_[:, :, None] * ww_[:, None, :]).reshape(-1, 4)
    return idx, wts


@dataclass
class EncodeMeta:
    """One encoder call: its images' patch offsets and the per-patch tensors the tower reads."""

    n: int  # patches
    cu: torch.Tensor  # int32 [images + 1]: patch offsets (cu_seqlens)
    cu_host: list[int]
    cos: torch.Tensor  # fp32 [N, head_dim]
    sin: torch.Tensor
    pos_idx: torch.Tensor  # int64 [N, 4]
    pos_w: torch.Tensor  # fp32 [N, 4]
    attn: Any = None  # the attention backend's plan

    PER_PATCH = ("cos", "sin", "pos_idx", "pos_w")  # what `runner.cuda.images.VisionGraphs` keeps in static buffers


# -- the modules -------------------------------------------------------------------------------------------------------


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig, dtype: torch.dtype):
        super().__init__()
        self.heads, self.d = cfg.heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=True, dtype=dtype)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor, meta: EncodeMeta, backend: Any) -> torch.Tensor:
        n = x.shape[0]
        pad = getattr(backend, "pad", self.d)
        q, k, v = ops.vision_qkv_rope(self.qkv(x), heads=self.heads, head_dim=self.d, cos=meta.cos, sin=meta.sin,
                                      pad_to=pad)
        o = backend.run(meta, q, k, v)
        if pad != self.d:
            o = o[..., :self.d]
        return self.proj(o.reshape(n, -1))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig, dtype: torch.dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden, eps=1e-6, dtype=dtype)
        self.norm2 = nn.LayerNorm(cfg.hidden, eps=1e-6, dtype=dtype)
        self.attn = VisionAttention(cfg, dtype)
        self.linear_fc1 = nn.Linear(cfg.hidden, cfg.intermediate, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(cfg.intermediate, cfg.hidden, bias=True, dtype=dtype)
        self.tanh = cfg.act == "gelu_pytorch_tanh"
        if cfg.act not in ("gelu_pytorch_tanh", "gelu"):
            raise NotImplementedError(f"vision hidden_act {cfg.act!r}")

    def forward(self, x: torch.Tensor, meta: EncodeMeta, backend: Any) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), meta, backend)
        h = F.gelu(self.linear_fc1(self.norm2(x)), approximate="tanh" if self.tanh else "none")
        return x + self.linear_fc2(h)


class Merger(nn.Module):
    def __init__(self, cfg: VisionConfig, dtype: torch.dtype):
        super().__init__()
        self.width = cfg.hidden * cfg.merge * cfg.merge
        self.norm = nn.LayerNorm(cfg.hidden, eps=1e-6, dtype=dtype)
        self.linear_fc1 = nn.Linear(self.width, self.width, bias=True, dtype=dtype)
        self.linear_fc2 = nn.Linear(self.width, cfg.out_hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.width)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


def vision_buffers(cfg: VisionConfig) -> dict[str, torch.Tensor]:
    """The rotary frequencies (HF's `compute_axial_rope_parameters`: head_dim / 4 of them, computed on the CPU) and the
    normalisation's mean and std, each times 1 / `rescale`."""
    inv = rope_inv_freq(cfg.rope_theta, cfg.head_dim // 2)
    mean = torch.tensor(cfg.mean, dtype=torch.float32) * (1.0 / cfg.rescale)
    std = torch.tensor(cfg.std, dtype=torch.float32) * (1.0 / cfg.rescale)
    return {"inv_freq": inv, "px_mean": mean.view(3, 1, 1), "px_std": std.view(3, 1, 1)}


class QwenVisionEncoder(nn.Module):
    """Pixels to image-token embeddings (module docstring): `patches`, `meta`, then `forward`."""

    def __init__(self, cfg: VisionConfig, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.hidden, bias=True, dtype=dtype)
        self.pos_embed = nn.Embedding(cfg.num_pos, cfg.hidden, dtype=dtype)
        self.blocks = nn.ModuleList([VisionBlock(cfg, dtype) for _ in range(cfg.depth)])
        self.merger = Merger(cfg, dtype)
        self.side = round(cfg.num_pos ** 0.5)
        for name, value in vision_buffers(cfg).items():
            self.register_buffer(name, value, persistent=False)
        self._grid_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    # -- pixels --------------------------------------------------------------------------------------------------

    def patches(self, px: torch.Tensor) -> torch.Tensor:
        """uint8 [3, H, W] (H, W multiples of patch x merge) -> fp32 [N, patch_dim] in the merge order, exactly HF's
        `pixel_values` (the fused normalisation, the temporal duplicate)."""
        c = self.cfg
        x = (px.float() - self.px_mean) / self.px_std
        p, m, t = c.patch, c.merge, c.temporal
        _, hgt, wid = x.shape
        gh, gw = hgt // p, wid // p
        x = x.reshape(3, gh // m, m, p, gw // m, m, p).permute(1, 4, 2, 5, 0, 3, 6)  # (bh, bw, mh, mw, c, p, p)
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, t, -1, -1)
        return x.reshape(gh * gw, 3 * t * p * p)

    # -- metadata ------------------------------------------------------------------------------------------------

    def grid_meta(self, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """(cos, sin, position taps, tap weights) of one h x w patch grid, cached by grid."""
        key = (h, w, str(device))
        got = self._grid_cache.get(key)
        if got is None:
            rows, cols = merge_order(h, w, self.cfg.merge, device)
            freqs = torch.cat([rows[:, None].float() * self.inv_freq, cols[:, None].float() * self.inv_freq], dim=-1)
            freqs = torch.cat([freqs, freqs], dim=-1)
            i, wt = interp_taps(rows, cols, h, w, self.side)
            got = (freqs.cos(), freqs.sin(), i, wt)
            if len(self._grid_cache) >= GRID_CACHE:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = got
        return got

    def meta(self, grids: list[tuple[int, int, int]], device: torch.device) -> EncodeMeta:
        """The per-patch rotary angles and position taps of a batch of images (grids (t, h, w); t must be 1)."""
        parts, cu = [], [0]
        for t, h, w in grids:
            if t != 1:
                raise NotImplementedError("video (temporal grids > 1) is not supported")
            parts.append(self.grid_meta(h, w, device))
            cu.append(cu[-1] + h * w)
        cos, sin, idx, wts = (x[0] if len(parts) == 1 else torch.cat(list(x)) for x in zip(*parts))
        return EncodeMeta(n=cu[-1], cu=to_device_int32(cu, device), cu_host=cu, cos=cos, sin=sin, pos_idx=idx,
                          pos_w=wts)

    # -- the tower -----------------------------------------------------------------------------------------------

    def forward(self, patches: torch.Tensor, meta: EncodeMeta, backend: Any) -> torch.Tensor:
        """fp32 patches [N, patch_dim] -> embeddings [N / merge^2, out_hidden]."""
        x = self.patch_embed(patches.to(self.dtype))
        pos = (self.pos_embed(meta.pos_idx) * meta.pos_w[:, :, None]).sum(1)
        x = x + pos.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, meta, backend)
        return self.merger(x)


class TorchVisionAttention:
    """Reference attention: SDPA per image, non-causal (HF's SDPA path)."""

    name = "torch"

    def __init__(self, head_dim: int | None = None, scale: float | None = None):
        self.head_dim = head_dim  # the softmax scale's dim (inputs are not padded here)
        self.scale = scale  # the softmax scale itself (Gemma 4's tower: 1.0); default 1 / sqrt(head_dim)

    def plan(self, meta: EncodeMeta) -> None:
        meta.attn = None

    def run(self, meta: EncodeMeta, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        scale = self.scale if self.scale is not None else (self.head_dim or q.shape[-1]) ** -0.5
        cu = meta.cu_host
        for a, b in itertools.pairwise(cu):
            o = F.scaled_dot_product_attention(q[a:b].transpose(0, 1)[None], k[a:b].transpose(0, 1)[None],
                                               v[a:b].transpose(0, 1)[None], scale=scale)
            out[a:b] = o[0].transpose(0, 1)
        return out


# -- weights -----------------------------------------------------------------------------------------------------------

_PREFIXES = ("model.visual.", "visual.")


def vision_weight_name(name: str) -> str | None:
    """A checkpoint name -> our parameter name, or None when it is not a vision-tower weight."""
    for p in _PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    else:
        return None
    if name.startswith("deepstack_merger_list."):
        return None
    name = name.replace(".mlp.linear_fc", ".linear_fc").replace("patch_embed.proj.", "patch_embed.")
    return name


@torch.inference_mode()
def load_vision(model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> QwenVisionEncoder:
    """Build on the meta device, allocate on `device`, load `model.visual.*` (the vision tower is BF16 in the FP8
    checkpoints: `modules_to_not_convert`)."""
    from safetensors import safe_open

    from .model import checkpoint_files

    cfg = VisionConfig.from_dir(model_dir)
    with torch.device("meta"):
        enc = QwenVisionEncoder(cfg, dtype)
    enc = enc.to_empty(device=device).requires_grad_(False)
    for name, value in vision_buffers(cfg).items():  # to_empty leaves the buffers uninitialised
        setattr(enc, name, value.to(device))
    params = dict(enc.named_parameters())
    seen: set[str] = set()
    dev = str(device) if device.type == "cpu" else f"cuda:{device.index or 0}"
    for path in checkpoint_files(Path(model_dir)):
        with safe_open(str(path), framework="pt", device=dev) as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                ours = vision_weight_name(name)
                if ours is None:
                    continue
                p = params.get(ours)
                if p is None:
                    raise ValueError(f"unexpected vision weight {name}")
                t = f.get_tensor(name)
                if ours == "patch_embed.weight":
                    t = t.reshape(t.shape[0], -1)
                if t.shape != p.shape:
                    raise ValueError(f"{name} {tuple(t.shape)} does not fit {ours} {tuple(p.shape)}")
                p.data.copy_(t.to(p.dtype))
                seen.add(ours)
    missing = sorted(set(params) - seen)
    if missing:
        raise ValueError(f"vision weights missing from the checkpoint: {missing[:6]}")
    return enc.eval()

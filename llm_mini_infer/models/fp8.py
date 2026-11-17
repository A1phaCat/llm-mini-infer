"""Block-FP8 checkpoints (docs/architecture.md, "FP8"): the quantization config, in the two formats that carry the
block-FP8 scheme (`kernels.fp8`), and the projection that runs it (`Fp8Linear`).

**The formats** (`QuantConfig.from_hf`):
- **Hugging Face's `fp8` method** (fine-grained FP8; Qwen3.8-27B-FP8): `weight_block_size` [128, 128],
  `activation_scheme` dynamic, `fmt` e4m3; the scales are FP32 tensors named `weight_scale_inv` (the name is DeepSeek's;
  the values multiply); the modules kept in BF16 are listed in `modules_to_not_convert` (or `ignored_layers`), matched
  without the model prefix.
- **compressed-tensors `float-quantized`** (llm-compressor's FP8_BLOCK scheme): one config group targeting `Linear`,
  static symmetric weights in 128 × 128 blocks and dynamic symmetric activations in groups of 128, no KV cache scheme
  (the KV stays BF16), no sparsity or transforms. The scales are named `weight_scale`; a checkpoint may store them in
  BF16, which loading widens to FP32 exactly. The modules kept in BF16
  are `ignore`'s entries, module names or `re:` regexes, matched from the start of the checkpoint's full module names.

Anything else (static activations, per-tensor or per-channel weights, other block sizes, UE8M0 scales, an FP8 KV cache)
is refused at startup with the reason.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..kernels import fp8, ops
from ..kernels.fp8 import QuantAct

# The scale dtypes a checkpoint may store (loaded into FP32 exactly); UE8M0 (float8_e8m0fnu) scales are not supported.
SCALE_DTYPES = (torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class QuantConfig:
    """A block-FP8 checkpoint's quantization (see the module docstring). `ignore`: exact module names without the model
    prefix (the `fp8` format, `modules_to_not_convert`); `patterns`: compressed-tensors' `ignore` entries as given,
    matched against the checkpoint's full module names; `scale_suffix`: the scale tensors' name after the weight's
    module."""

    method: str
    block: tuple[int, int]
    ignore: frozenset[str]
    patterns: tuple[str, ...] = ()
    scale_suffix: str = "weight_scale_inv"

    @classmethod
    def from_hf(cls, q: dict[str, Any], strip: tuple[str, ...] = ()) -> QuantConfig:
        """From config.json's `quantization_config`; `strip`: model prefixes to remove from `fp8`-format names."""
        method = q.get("quant_method")
        if method == "fp8":
            return cls._from_fp8(q, strip)
        if method == "compressed-tensors":
            return cls._from_compressed_tensors(q)
        raise NotImplementedError(f"quantization {method!r}: only block-FP8 checkpoints are supported (Hugging Face's "
                                  "fine-grained fp8 method, or compressed-tensors float-quantized with 128 x 128 "
                                  "weight blocks)")

    @classmethod
    def _from_fp8(cls, q: dict[str, Any], strip: tuple[str, ...]) -> QuantConfig:
        block = tuple(q.get("weight_block_size") or ())
        if block != (fp8.GROUP, fp8.GROUP):
            raise NotImplementedError(f"FP8 weight_block_size {list(block)}: only [128, 128] (per-tensor FP8 is not "
                                      "supported)")
        if q.get("activation_scheme", "dynamic") != "dynamic":
            raise NotImplementedError(f"FP8 activation_scheme {q.get('activation_scheme')!r}: only dynamic")
        if q.get("fmt", "e4m3") != "e4m3":
            raise NotImplementedError(f"FP8 fmt {q.get('fmt')!r}: only e4m3")
        ignore = set()
        for name in q.get("modules_to_not_convert") or q.get("ignored_layers") or ():
            for p in strip:
                if name.startswith(p):
                    name = name[len(p):]
                    break
            ignore.add(name)
        return cls(method="fp8", block=block, ignore=frozenset(ignore))

    @classmethod
    def _from_compressed_tensors(cls, q: dict[str, Any]) -> QuantConfig:
        def refuse(what: str) -> NotImplementedError:
            return NotImplementedError(f"compressed-tensors {what}: only the FP8_BLOCK scheme (128 x 128 static "
                                       "weight blocks, dynamic per-token group-128 activations) is supported")

        groups = q.get("config_groups") or {}
        if len(groups) != 1:
            raise refuse(f"with {len(groups)} config groups")
        (gname, g), = groups.items()
        fmt = g.get("format") or q.get("format")
        if fmt != "float-quantized":
            raise refuse(f"format {fmt!r}")
        if q.get("quantization_status", "compressed") != "compressed":
            raise refuse(f"quantization_status {q.get('quantization_status')!r} (the weights must be stored in FP8)")
        if list(g.get("targets") or ()) != ["Linear"]:
            raise refuse(f"group {gname} targets {g.get('targets')}")
        w, a = g.get("weights") or {}, g.get("input_activations") or {}
        for args, kind in ((w, "weights"), (a, "input activations")):
            if args.get("type") != "float" or args.get("num_bits") != 8 or not args.get("symmetric", False):
                raise refuse(f"{kind} {args.get('type')} {args.get('num_bits')}-bit "
                             f"{'symmetric' if args.get('symmetric') else 'asymmetric'}")
            if args.get("scale_dtype") not in (None, "float32", "bfloat16", "torch.float32", "torch.bfloat16"):
                raise refuse(f"{kind} scale_dtype {args.get('scale_dtype')!r}")
        if w.get("strategy") != "block" or list(w.get("block_structure") or ()) != [fp8.GROUP, fp8.GROUP] or \
                w.get("dynamic", False):
            raise refuse(f"weights strategy {w.get('strategy')!r} block {w.get('block_structure')} "
                         f"dynamic {w.get('dynamic')}")
        if a.get("strategy") != "group" or a.get("group_size") != fp8.GROUP or a.get("dynamic") is not True:
            raise refuse(f"input activations strategy {a.get('strategy')!r} group {a.get('group_size')} "
                         f"dynamic {a.get('dynamic')}")
        if g.get("output_activations"):
            raise refuse("output activation quantization")
        if q.get("kv_cache_scheme"):
            raise refuse("KV cache quantization (kv_cache_scheme; the KV stays BF16)")
        if q.get("sparsity_config") or q.get("transform_config"):
            raise refuse("sparsity or transforms")
        return cls(method="compressed-tensors", block=(fp8.GROUP, fp8.GROUP), ignore=frozenset(),
                   patterns=tuple(q.get("ignore") or ()), scale_suffix="weight_scale")

    def skips(self, name: str, full_names: Iterable[str] = ()) -> bool:
        """Whether the checkpoint module `name` (model prefix stripped) stays BF16: listed in `ignore`, or one of its
        full names (`full_names`: with the checkpoint's prefixes) matching an `ignore` entry of compressed-tensors."""
        if name in self.ignore:
            return True
        for p in self.patterns:
            for full in full_names:
                if (re.match(p[3:], full) is not None) if p.startswith("re:") else p == full:
                    return True
        return False


def fused_fp8(q: QuantConfig | None, name: str, parts: list[str], full: Iterable[str]) -> bool:
    """Whether our projection `name` is block-FP8: neither `name` nor any of its checkpoint modules `parts` is skipped
    (`full`: the checkpoint's prefixes for the full names). A fused projection must not mix FP8 and BF16 shards. Loading
    checks the result against the checkpoint's tensors."""
    if q is None:
        return False
    prefixes = tuple(full)
    skipped = [q.skips(p, [pre + p for pre in prefixes]) for p in parts]
    if q.skips(name, [pre + name for pre in prefixes]) or all(skipped):
        return False
    if any(skipped):
        raise ValueError(f"{name}: the checkpoint quantizes some of {parts} but not all; they cannot be fused")
    return True


class Fp8Linear(nn.Module):
    """A block-FP8 projection (no bias): `weight` [N, K] e4m3 and `weight_scale` [N / 128, K / 128] fp32 (the
    checkpoint's `weight_scale_inv` or `weight_scale`, widened to FP32 when stored in BF16). Takes a BF16 tensor
    (quantized here), a `QuantAct` from a quantizing producer, or a `SplitAct`; returns BF16. In a step with PRECISE
    rows, the last `fp8.bf16_rows` rows take the BF16 path (`kernels.fp8`, "BF16 activations")."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.out_dtype = dtype  # BF16 on CUDA; the CPU reference runs the model in fp32
        if in_features % fp8.GROUP or out_features % fp8.GROUP:
            raise ValueError(f"a block-FP8 projection needs both dims multiples of 128 ({out_features} x "
                             f"{in_features})")
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=fp8.FP8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(out_features // fp8.GROUP, in_features // fp8.GROUP,
                                                     dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor | QuantAct | fp8.SplitAct) -> torch.Tensor:
        w, ws = self.weight, self.weight_scale
        if isinstance(x, fp8.SplitAct):
            f = x.head.m
            out = torch.empty((x.m, self.out_features), dtype=self.out_dtype, device=x.tail.device)
            fp8.gemm(x.head, w, ws, out_dtype=self.out_dtype, out=out[:f])
            fp8.linear_bf16(x.tail, w, ws, self.out_dtype, out=out[f:])
            return out
        if isinstance(x, QuantAct):
            if fp8.bf16_rows(x.m):
                raise ValueError("a quantized activation for rows that take BF16")
            return fp8.gemm(x, w, ws, out_dtype=self.out_dtype)
        m = x.shape[0]
        nb = fp8.bf16_rows(m)
        if nb == m:
            return fp8.linear_bf16(x, w, ws, self.out_dtype)
        if not nb:
            return fp8.gemm(ops.quant_fp8(x), w, ws, out_dtype=self.out_dtype)
        f = m - nb  # the step's FP8 rows first, its PRECISE rows after them
        out = torch.empty((m, self.out_features), dtype=self.out_dtype, device=x.device)
        fp8.gemm(ops.quant_fp8(x[:f]), w, ws, out_dtype=self.out_dtype, out=out[:f])
        fp8.linear_bf16(x[f:], w, ws, self.out_dtype, out=out[f:])
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, block-FP8"


def check_tensor(name: str, param: torch.Tensor, t: torch.Tensor, row0: int) -> None:
    """A checkpoint tensor against the parameter it loads into: FP8 exactly where the parameter is (the quantization
    config agrees with the tensors), 128-row aligned FP8 shards, and scales in FP32 or BF16 (UE8M0 scales refused)."""
    if (param.dtype == fp8.FP8) != (t.dtype == fp8.FP8):
        raise ValueError(f"{name} is {t.dtype} in the checkpoint but its parameter is {param.dtype} (the "
                         "quantization_config's BF16 modules disagree with the tensors)")
    if param.dtype == fp8.FP8 and (row0 % fp8.GROUP or t.shape[0] % fp8.GROUP):
        raise ValueError(f"the FP8 shard {name} ({t.shape[0]} rows at {row0}) is not 128-row aligned")
    if name.endswith(("weight_scale", "weight_scale_inv")) and t.dtype not in SCALE_DTYPES:
        raise NotImplementedError(f"{name}: block scales in {t.dtype} (FP32 or BF16 only)")

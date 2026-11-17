"""Helpers both model families' backends and towers use."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def rope_inv_freq(theta: float, spatial: int) -> torch.Tensor:
    """The vision towers' RoPE inverse frequencies in fp32: HF's formula, with each power taken in float64 by
    `math.pow` and rounded once. ATen's vector and scalar `pow` routes round differently, so HF's own bits depend on
    the CPU; these are the scalar route's bits."""
    base = float(np.float32(theta))  # ATen casts the base to float32; 100, 1e4 and 1e6 are exact
    e = np.arange(0, spatial, 2, dtype=np.float32) / np.float32(spatial)
    p = np.array([math.pow(base, float(x)) for x in e], dtype=np.float64).astype(np.float32)
    return torch.from_numpy(np.float32(1.0) / p)


def weak_view(t: torch.Tensor) -> torch.Tensor:
    """A tensor on t's memory that does not keep it allocated. The eager segments of a
    breakable CUDA graph hold these: the graph's private pool owns the memory, and later captures into the pool may
    reuse it (graphs never run concurrently)."""
    s = t.untyped_storage()
    storage = torch._C._construct_storage_from_data_pointer(s.data_ptr(), t.device, s.nbytes())
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(storage, t.storage_offset(), t.size(), t.stride())


def to_device_int32(values: list[int], device: torch.device) -> torch.Tensor:
    """Host ints as int32 on `device`, queued without a wait: on CUDA copied from page-locked memory (the driver
    stages a pageable source itself, and may wait for the stream to do it)."""
    t = torch.tensor(values, dtype=torch.int32)
    if device.type == "cuda":
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


class Coverage:
    """The row ranges a weight loader has written, per parameter. A parameter is complete when its ranges tile
    [0, rows) exactly: a count of rows would pass a shard loaded twice beside one never loaded. A 0-dimensional
    parameter has one row (its tensor reshaped to [1])."""

    def __init__(self) -> None:
        self.ranges: dict[str, list[tuple[int, int]]] = {}

    def add(self, name: str, row0: int, rows: int) -> None:
        self.ranges.setdefault(name, []).append((row0, row0 + rows))

    def incomplete(self, params: dict[str, nn.Parameter]) -> list[str]:
        """The parameters whose ranges leave a gap, overlap or run past the end (in `params` order)."""
        bad = []
        for name, p in params.items():
            end = 0
            for lo, hi in sorted(self.ranges.get(name, ())):
                if lo != end:
                    break
                end = hi
            else:
                if end == (p.shape[0] if p.dim() else 1):
                    continue
            bad.append(name)
        return bad

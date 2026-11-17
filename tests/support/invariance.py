"""Batch invariance's logits comparison (tests/runner/test_batch_invariance_gpu.py)."""

from __future__ import annotations

import numpy as np

from .forced import Bits


class LogitsDiffer(AssertionError):
    """Logits of a request whose bits differ alone and beside others (the dense head's expected failure)."""


def logits_compared(alone: Bits, beside: Bits, common: list[int]) -> tuple[int, int, list[tuple[int, int]]]:
    """(logits both runs hold at `common`'s positions, how many of them sit in another column, (position, token) of
    those whose bits differ)."""
    n = moved = 0
    differ: list[tuple[int, int]] = []
    for p in common:
        (ia, la), (ib, lb) = alone.logits[p], beside.logits[p]
        both, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        n += len(both)
        moved += int((xa != xb).sum())
        differ += [(p, int(t)) for t, a, b in zip(both, la[xa], lb[xb]) if a != b]
    return n, moved, differ

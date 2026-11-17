"""FlashInfer's decode plans by length bucket (tests/kernels/test_flashinfer_selfcheck.py)."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

HQ, HKV, PS, W = 32, 16, 16, 1024  # Gemma 4's sliding layers: query and KV heads (of 256), page size, window
DEFAULT_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64]  # `default_decode_buckets(64)`: max_num_seqs 64
BUCKETS = sorted([*DEFAULT_BUCKETS, 12])  # and 12: the low edge of the window grid's two-chunk override


class GraphPlan(NamedTuple):
    """A CUDA-graph plan as FlashInfer makes it (`graph_plan`)."""

    split: bool  # the split flag (`_PLAN_SPLIT_KV`)
    padded: int  # the captured grid's work items (`_PLAN_PADDED`)
    tile: int  # the query tile (`_PLAN_CTA_TILE_Q`)
    chunk: int  # the chunk in pages (the chunk size the plan writes into the page-locked buffer, over the page size)
    counts: np.ndarray  # each row's chunks
    work: int  # the work items the plan fills: each row's query tiles times its chunks


def graph_plan(qo: Any, pages: Any, bound: int, sms: int, fixed: int = 0, *, window_left: int = W - 1, hq: int = HQ,
               hkv: int = HKV, ps: int = PS) -> GraphPlan:
    """FlashInfer's plan of a CUDA-graph paged prefill wrapper at head dim 256 on a GPU of `sms` SMs: rows of `qo`
    queries over `pages` KV pages, chunked by FlashInfer's own search or by `fixed` pages (`fixed_split_size`). Read
    from the 0.6.18 wheel, the same lines in 0.7.2's: `PrefillPlanImpl`, `PrefillSplitQOKVIndptr` and
    `PrefillBinarySearchKVChunkSize` in include/flashinfer/attention/scheduler.cuh, `FA2DetermineCtaTileQ` in
    include/flashinfer/utils.cuh, and the paged wrapper's plan() in prefill.py, which passes its row bound
    `_max_total_num_rows` (`bound`: the first plan's rows unless set before) as the total rows and uniform_q_len 0.

    The split work items are 2 × SMs / KV heads (two CTAs per SM, none colocated). The query tile is 16 where the
    longest row the bound allows packs at most 16 query rows with its group's heads, (bound − rows + 1) × group ≤ 16,
    else 64 (head dim 256 on SM 8.0+, whose shared memory holds the 16-row tile). A row's pages count up to
    ceil((window_left + tile) / page size). The search takes the smallest chunk from 128 positions whose work items fit
    the split work items, else the most pages a row counts (one chunk each), and a graph plan reports a split whatever
    it takes; a fixed chunk reports one only when some row gets two chunks or more. The grid is the larger of the split
    work items and the query tiles `bound` rows can take, ceil(bound × group / tile) + rows − 1, and FlashInfer refuses
    a plan whose work exceeds it (FLASHINFER_CHECK: `GraphWrapper` asserts it)."""
    qo, pages = np.asarray(qo, np.int64), np.asarray(pages, np.int64)
    group, rows, items = hq // hkv, len(qo), 2 * sms // hkv
    tile = 16 if (bound - rows + 1) * group <= 16 else 64
    tiles = -(-qo * group // tile)
    eff = np.maximum(np.minimum(pages, -(-(window_left + tile) // ps)) if window_left >= 0 else pages, 1)
    chunk = fixed
    if not fixed:
        lo, hi = max(128 // ps, 1), int(eff.max())
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if int((tiles * -(-eff // mid)).sum()) > items else (lo, mid)
        chunk = lo
    counts = -(-eff // chunk)
    work, padded = int((tiles * counts).sum()), max(items, -(-bound * group // tile) + rows - 1)
    return GraphPlan(not fixed or bool((counts > 1).any()), padded, tile, chunk, counts, work)

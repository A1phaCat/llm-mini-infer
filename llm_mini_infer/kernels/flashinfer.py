"""FlashInfer on CUDA: paged attention, the vision tower's ragged attention and GDN chunked prefill
(docs/architecture.md, "Kernels").

**Paged attention** uses `BatchPrefillWithPagedKVCacheWrapper` for every row, decode rows included (qo_len 1, or the
1 + k of an MTP verify row), with a causal mask aligned to the end of each row's KV. `plan()` needs host-side
`qo_indptr`, `kv_indptr`, `last_page_len` and the KV lengths: the runner passes pinned host tensors built from its host
mirror of `seq_len` (exact without MTP, each row's upper bound under it), so planning never reads device memory.
- **Eager steps and the eager breaks of breakable graphs** use two dynamic wrappers, alternating with the runner's
  staging parity, with page indices built on the device from the page table (`page_indices`).
- **Decode CUDA graphs** use one CUDA-graph wrapper per batch-size bucket (`graph_wrapper`): FlashInfer then fixes
  the batch size and always splits the KV (its CUDA-graph schedule), and the kernel reads the plan from fixed
  buffers that `plan_graph` refills before each replay. The buckets share those buffers (only one graph runs at a
  time). The page indices are built inside the graph (`graph_page_indices`) from the device page table, so the plan
  copies none. Padding rows attend to the null page.

**Planning ahead is the hazard.** `plan()` writes the plan into a page-locked host buffer and enqueues its copy to the
device. Step N+1 is planned while step N may still be queued, so one buffer would be overwritten before step N's copy
ran. Every wrapper therefore plans from one of two page-locked buffers chosen by the step's parity (step N+2 reuses
step N's only after `wait(N)`), and they share one device int workspace and one float workspace, which only
stream-ordered copies and kernels touch. The MTP drafter's attention (the runner's `chain_attn`, Gemma 4's `attn_dw`)
is a second instance with its own int workspace and buffers, planned in the same step.

**True lengths on the device (MTP).** The plan is made for each row's upper bound, and the kernels read the true lengths
at run time: the paged kernel derives each row's KV length, and its chunk bounds, from the device `kv_indptr` and
`last_page_len` buffers, so `fix_lengths` rewrites `last_page_len` on the device as `true_len − (ub_pages − 1) ×
page_size` (zero or negative when the true length ends a page earlier: FlashInfer's length arithmetic is modular, so it
comes out right) while `kv_indptr` and the page indices keep the upper bound's pages. FlashInfer launches with
programmatic dependent launch by its own default and reads those two buffers before its wait, so a run right after the
rewrite launches serialized (`run_fixed`).

**Chunk counts.** With split KV, FlashInfer's kernel counts a row's chunks from the true KV length, while the merge
reads the partials where the plan put them. When a chunk boundary lies between a row's lowest and highest possible
length, the true length can drop the plan's last chunk, and the row's queries merge from the wrong partials.
`_plan_safely` (behind `plan` and `plan_graph`) therefore reads the chunk size and query tile the plan chose and, when a
boundary falls there for some row (`unsafe_chunks`), plans again with a `fixed_split_size` that puts none there. An
empty chunk at a planned count merges harmlessly (its log-sum-exp is −inf, a zero weight).

**Decode graphs keep the captured split.** A CUDA-graph wrapper's kernel is fixed at capture, and FlashInfer's graph
schedule always splits; a re-plan without a split would make the captured merge read the plan's valid-block mask as its
indices. So a graph's re-plan keeps some row at two chunks or more and fits the captured grid (`safe_chunk_pages`);
when no chunk size does, the first plan stays (`kept`, logged once: its uncertain rows may attend wrongly). Where the
rows fit only with large chunks, the sliding layers' graph wrappers plan under MTP for a larger row bound
(`graph_wrapper`), which grows the grid to two chunks per row.

**The self-check** (`self_check`, a runner with MTP, once at startup) tests what this reads beyond FlashInfer's public
API: four entries of `PrefillPlanInfo::ToVector()` and the wrappers' private fields (`_plan_info`,
`_pin_memory_int_workspace_buffer`, `_paged_kv_indptr_buf`, `_paged_kv_last_page_len_buf`, `_max_total_num_rows`). If
the plan info reads otherwise, eager plans with uncertain lengths go without a split, and with decode graphs startup
fails; if the length rewrite does not take, startup fails.

**GDN prefill** (`flashinfer.gdn_prefill.chunk_gated_delta_rule`, CuTe DSL on SM120): q/k/v without the batch dim, `g`
passed as alpha = exp(g), fp32 beta, int64 `cu_seqlens`; q and k are L2-normalised by the caller
(`use_qk_l2norm_in_kernel=False`). It takes more value heads than q/k heads (16 and 48). The states stay in the layer's
view of the state pool `[S, H_v, V, K]` (fp32, strided across the layers): `state_indices` names each sequence's slot,
which the kernel reads at the start and writes at the end, in place. The kernel launch reads no device memory on the
host.

**The per-sequence route at every step** (`use_cp` off). FlashInfer's own choice takes its context-parallel route on
SM120 for few sequences (a call of one sequence on a 188-SM card). That route computes otherwise, as close to the fp32
recurrence, and is the slower one at the chunks the default budget makes: measured on an RTX PRO 6000 per GDN layer at
Qwen3.8 27B's heads, one sequence of 2,048 tokens takes 0.217 ms against 0.184, and only from 4,096 does it win.

**Checkpoints by cutting rows** (`_gdn_cut_rows`). FlashInfer 0.7.2's checkpointing variant (the state every 64 tokens
of every row) is not deterministic on SM120: between identical packed calls its outputs and states moved (fixed in
0.7.3, flashinfer-ai/flashinfer#7785). So a step's mid-chunk checkpoints cut their rows into consecutive calls
(`gdn_segments`), each starting from the states the call before left in the pool; a checkpoint is its segment's final
state. The cuts lie at multiples of 64 from the row's start, where the route computes the same chunks as one call.

**Batch invariance** (docs/architecture.md, "Batch invariance") takes the same calls:
tests/kernels/test_qwen_kernels_gpu.py fails on an upgrade that breaks the route's invariance. One hazard remains: a NaN
in a sequence's first rows of k or v reaches the sequence before it in the packed call when that one ends in a partial
chunk, which reads them as padding (multiplied by zeros, which keeps a NaN: flashinfer-ai/flashinfer#9372, still in
0.7.4). A zero-length sequence leaves its slot as it was (the runner plans none).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ..models.qwen3_5.backend import Pools, StepMeta

# 394 MiB: FlashInfer 0.7.2's plans need at most 80 MiB at 64 decode rows and 320 at 256, the most of the supported
# models (Gemma 4's sliding layers' decode graphs at k = 7, the largest k), measured with workspace_size() on SM120
WORKSPACE_BYTES = 394 << 20
INT_WORKSPACE_BYTES = 8 << 20  # FlashInfer's own


# The positions of PrefillPlanInfo::ToVector() (FlashInfer's attention/scheduler.cuh)
_PLAN_PADDED, _PLAN_CTA_TILE_Q, _PLAN_CHUNK_SIZE_OFFSET, _PLAN_SPLIT_KV, _PLAN_INFO_LEN = 0, 3, 9, 14, 15
_plan_info_ok = True  # `self_check`: whether the plan info reads as above (its first call's finding)
# The self-check's findings, once per process and kind of caller (with decode graphs or without: `self_check`): the
# result it returns, the plan info's and the length rewrite's (None when they passed)
_checked: dict[bool, tuple[dict[str, Any], str | None, str | None]] = {}


def chunk_counts(ub: np.ndarray, page_size: int, chunk_pages: int, window_left: int = -1,
                 cta_tile_q: int = 0) -> np.ndarray:
    """The KV chunks FlashInfer plans per row (`PrefillSplitQOKVIndptr`): ceil(pages / chunk_pages) of the row's pages
    for KV length ub, with a window at most ceil((window_left + cta_tile_q) / page_size) of them (`cta_tile_q` 0: not
    known, no cap, which counts at least as many)."""
    pages = np.maximum(-(-np.asarray(ub, np.int64) // page_size), 1)
    if window_left >= 0 and cta_tile_q:
        pages = np.minimum(pages, -(-(window_left + cta_tile_q) // page_size))
    return -(-pages // chunk_pages)


def unsafe_chunks(lb: np.ndarray, ub: np.ndarray, page_size: int, chunk_pages: int, *, window_left: int = -1,
                  cta_tile_q: int = 0) -> bool:
    """Whether a split-KV plan with chunks of `chunk_pages` pages, planned for KV lengths ub, can run with fewer chunks
    than it planned at some true length in [lb, ub] (module docstring): a chunk boundary in [min(lb, cap),
    min(ub, cap)) for some row, the cap being window_left + cta_tile_q with a window (none when the tile is not
    known)."""
    lb, ub = np.asarray(lb, np.int64), np.asarray(ub, np.int64)
    if window_left >= 0 and cta_tile_q:
        lb, ub = np.minimum(lb, window_left + cta_tile_q), np.minimum(ub, window_left + cta_tile_q)
    cs = chunk_pages * page_size
    return bool((-(-lb // cs) * cs < ub).any())


def safe_chunk_pages(lb: np.ndarray, ub: np.ndarray, page_size: int, chunk_pages: int, *, keep_split: bool = False,
                     max_work: int = 0, tiles: Any = 1, **layout: Any) -> int | None:
    """A chunk size whose count no true length can change (`unsafe_chunks`; `layout`: its window arguments): the
    smallest above `chunk_pages`, where one chunk per row (no split) always is one. `keep_split` (a CUDA-graph wrapper
    captured with the split kernel): some row must keep two chunks or more, and the rows' work items (`tiles` query
    tiles per row times their chunks) must fit the captured grid's `max_work`: the smallest above `chunk_pages` that
    does, else the largest below it, else None."""
    win, tile = layout.get("window_left", -1), layout.get("cta_tile_q", 0)
    top = int(chunk_counts(ub, page_size, 1, win, tile).max())  # the most pages a row's planned chunks cover

    def ok(c: int) -> bool:
        if keep_split:
            n = chunk_counts(ub, page_size, c, win, tile)
            if n.max() < 2 or (max_work and int((n * tiles).sum()) > max_work):
                return False
        return not unsafe_chunks(lb, ub, page_size, c, **layout)

    for c in range(chunk_pages + 1, top + 1):
        if ok(c):
            return c
    if keep_split:
        for c in range(min(chunk_pages, top) - 1, 0, -1):
            if ok(c):
                return c
    return None


class FlashInferAttention:
    name = "flashinfer"

    def __init__(self, pools: Pools, num_qo_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype,
                 device: torch.device, max_rows: int = 0, max_pages: int = 0, qo_per_row: int = 1,
                 workspace: torch.Tensor | None = None):
        """max_rows, max_pages: the largest decode bucket and pages per row, for the CUDA-graph buffers; qo_per_row: a
        decode row's query tokens (1 + k under MTP). `workspace`: the float workspace to share (a runner's FlashInfer
        instances share one: their kernels run one after the other)."""
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self._cls = BatchPrefillWithPagedKVCacheWrapper
        self.pools = pools
        self.hq, self.hkv, self.d = num_qo_heads, num_kv_heads, head_dim
        self.dtype = dtype
        self.workspace = workspace if workspace is not None else \
            torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.int_workspace = torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.pinned = [torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.wrappers = [self._share(self._cls(self.workspace, "NHD")) for _ in range(2)]
        # What the chunk-count hazard did: host ints, summed over a runner's instances (`runner.base.AttentionCounters`)
        self.replans = 0  # plans made again for a split whose chunk count the true lengths could change
        self.kept = 0  # graph plans kept although no safe split fit (`_plan_safely`)
        self.unsplit = 0  # plans made without a split instead (the self-check's fallback)
        # CUDA-graph decode: fixed buffers shared by every bucket's wrapper
        self.max_rows, self.max_pages, self.qo_per_row = max_rows, max_pages, qo_per_row
        if max_rows:
            self.g_qo = torch.arange(max_rows + 1, dtype=torch.int32, device=device) * qo_per_row
            self.g_kv_indptr = torch.zeros(max_rows + 1, dtype=torch.int32, device=device)
            self.g_kv_indices = torch.zeros(max_rows * max_pages + 1, dtype=torch.int32, device=device)  # + a dump
            self.g_last = torch.ones(max_rows, dtype=torch.int32, device=device)
            self.g_cols = torch.arange(max_pages, dtype=torch.int64, device=device)
        self.g_wrappers: dict[int, Any] = {}

    def _plan_safely(self, w: Any, parity: int, lb: np.ndarray | None, ub: np.ndarray, *args: Any,
                     graph: bool = False, **kw: Any) -> None:
        """`w.plan(*args, **kw)` (args from qo_indptr on); when rows' true KV lengths are only known within [lb, ub]
        and the plan's chunk count could change with them (`unsafe_chunks`), plan again with chunks whose count
        cannot, or, when the self-check found the plan info unreadable, without a split. `graph`: a CUDA-graph
        wrapper, whose captured split kernel needs a split that fits its grid (`safe_chunk_pages`); when none is safe,
        or FlashInfer reports none, its first plan stays (`kept`; module docstring)."""
        if lb is None or not bool((lb < ub).any()):
            w.plan(*args, **kw)
            return
        if not _plan_info_ok:  # eager plans go without a split; decode graphs refuse to start (`self_check`)
            if not graph:
                self.unsplit += 1
                kw = dict(kw, disable_split_kv=True)
            w.plan(*args, **kw)
            return
        w.plan(*args, **kw)
        info = w._plan_info
        if not int(info[_PLAN_SPLIT_KV]):
            return
        ps = self.pools.page_size
        off = int(info[_PLAN_CHUNK_SIZE_OFFSET])
        chunk = int(self.pinned[parity][off:off + 4].view(torch.int32)[0]) // ps
        tile = int(info[_PLAN_CTA_TILE_Q])
        layout = dict(window_left=kw.get("window_left", -1), cta_tile_q=tile)
        if not unsafe_chunks(lb, ub, ps, chunk, **layout):
            return
        tiles = np.maximum(-(-np.diff(args[0].numpy()) * (self.hq // self.hkv) // tile), 1) if graph and tile else 1
        c = safe_chunk_pages(lb, ub, ps, chunk, keep_split=graph, max_work=int(info[_PLAN_PADDED]), tiles=tiles,
                             **layout)
        if c is not None:
            self.replans += 1
            w.plan(*args, **kw, fixed_split_size=c)
            if not graph or int(w._plan_info[_PLAN_SPLIT_KV]):
                return
            w.plan(*args, **kw)  # never replay the split kernel over an unsplit plan
        self.kept += 1
        if self.kept == 1:
            import logging

            logging.getLogger(__name__).warning(
                "FlashInfer: a decode graph's attention plan kept chunks whose count the true KV lengths may change "
                "(no safe split fits its grid; rows %s, chunk %d pages): those rows may attend wrongly", len(ub), chunk)

    def _share(self, w: Any) -> Any:
        w._int_workspace_buffer = self.int_workspace
        w._pin_memory_int_workspace_buffer = self.pinned[0]
        return w

    # -- eager steps and breaks ---------------------------------------------------------------------------------

    def plan(self, meta: StepMeta | None, *, parity: int, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor,
             kv_indices: torch.Tensor, last_page_len: torch.Tensor, kv_lens: torch.Tensor, max_q: int,
             lb: np.ndarray | None = None) -> Any:
        """Host tensors (pinned): qo_indptr, kv_indptr, last_page_len, kv_lens (int32: the upper bounds). kv_indices
        on the device. `lb`: each row's lowest possible KV length (MTP; see the module docstring). Returns the wrapper
        (also as `meta.attn_plan`)."""
        w = self.wrappers[parity]
        w._pin_memory_int_workspace_buffer = self.pinned[parity]
        self._plan_safely(w, parity, lb, kv_lens.numpy(), qo_indptr, kv_indptr,
                          kv_indices, last_page_len, self.hq, self.hkv, self.d, self.pools.page_size, causal=True,
                          q_data_type=self.dtype, kv_data_type=self.dtype, seq_lens=kv_lens,
                          max_token_per_sequence=max_q, non_blocking=True)
        if meta is not None:
            meta.attn_plan = w
        return w

    def fix_lengths(self, w: Any, rows: int, true_len: torch.Tensor) -> None:
        """Rewrite the planned last-page lengths of rows [0, rows) from their true KV lengths on the device (int64
        [rows]; module docstring). Graph-safe for the CUDA-graph buffers."""
        indptr = w._paged_kv_indptr_buf[:rows + 1].long()
        ps = self.pools.page_size
        w._paged_kv_last_page_len_buf[:rows].copy_(true_len[:rows] - (indptr[1:] - indptr[:-1] - 1) * ps)

    def run_fixed(self, w: Any, q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor, rows: int,
                  true_len: torch.Tensor) -> None:
        """`fix_lengths` on w's rows [0, rows), then w's run without programmatic dependent launch, under which it
        could read the stale upper bound (module docstring)."""
        self.fix_lengths(w, rows, true_len)
        w.run(q, kv, out=out, enable_pdl=False)

    def run(self, meta: StepMeta, index: int, q: torch.Tensor, out: torch.Tensor) -> None:
        if meta.fix_len:  # MTP: the rows' true KV lengths, decided on the device (the drafter's chain)
            self.run_fixed(meta.attn_plan, q, self.pools.kv[index], out, meta.n_rows, meta.kv_len_dev)
        else:
            meta.attn_plan.run(q, self.pools.kv[index], out=out)

    # -- decode CUDA graphs -------------------------------------------------------------------------------------

    def graph_wrapper(self, rows: int) -> Any:
        w = self.g_wrappers.get(rows)
        if w is None:
            w = self._share(self._cls(self.workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=self.g_qo[:rows + 1],
                                      paged_kv_indptr_buf=self.g_kv_indptr[:rows + 1],
                                      paged_kv_indices_buf=self.g_kv_indices,
                                      paged_kv_last_page_len_buf=self.g_last[:rows]))
            self.g_wrappers[rows] = w
        return w

    def plan_graph(self, meta: StepMeta | None, *, rows: int, parity: int, qo_indptr: torch.Tensor,
                   kv_indptr: torch.Tensor, last_page_len: torch.Tensor, kv_lens: torch.Tensor, max_q: int,
                   lb: np.ndarray | None = None) -> Any:
        """Plan a decode bucket of `rows` rows (padding included) from pinned host arrays; the page indices are built
        inside the graph (`graph_page_indices`). `lb` as in `plan`."""
        w = self.graph_wrapper(rows)
        w._pin_memory_int_workspace_buffer = self.pinned[parity]
        self._plan_safely(w, parity, lb, kv_lens.numpy(), qo_indptr, kv_indptr,
                          self.g_kv_indices[:0], last_page_len, self.hq, self.hkv, self.d, self.pools.page_size,
                          graph=True, causal=True, q_data_type=self.dtype, kv_data_type=self.dtype, seq_lens=kv_lens,
                          max_token_per_sequence=max_q, non_blocking=True)
        if meta is not None:
            meta.attn_plan = w
        return w

    def graph_page_indices(self, rows: int, page_table: torch.Tensor, slots: torch.Tensor) -> None:
        """Inside a decode graph: each row's pages, packed at the plan's kv_indptr (copied to the device by
        `plan_graph` before the replay) with the plan's page counts, from the device page table rows `slots`
        [rows]."""
        indptr = self.g_kv_indptr[:rows + 1].long()
        n_pages = indptr[1:] - indptr[:-1]
        dest = torch.where(self.g_cols[None, :] < n_pages[:, None], indptr[:-1, None] + self.g_cols[None, :],
                           self.g_kv_indices.shape[0] - 1)
        self.g_kv_indices.scatter_(0, dest.reshape(-1), page_table.index_select(0, slots)[:, :self.max_pages]
                                   .reshape(-1))


class FlashInferWindowAttention(FlashInferAttention):
    """Gemma 4's sliding-window layers: FlashInfer's paged prefill with `window_left = window − 1` (a query at q sees
    keys [q − W + 1, q], HF's window) and scale 1.0, over the window pool. Each row's pages start at its
    first window page (`Gemma4Meta.row_wlo_host`): the plan's KV of a row is positions [wlo × page_size, kv_len), and
    FlashInfer aligns the causal mask and the window to the end of that KV, so the positions stay right while the
    released pages before it are never named. Decode graphs take the window's pages from the device window table at
    each row's first window page (`graph_window_indices`, a static per-row column).

    **Bidirectional image spans.** A second instance with `causal=False` plans the step's image spans as rows of their
    own: a span [a, e) is a row of e − a queries whose KV is positions [wlo(a) × page_size, e) of its request's window
    pages. Without the causal mask FlashInfer's window still bounds the keys from below (`kv_idx + qo_len + window_left
    >= kv_len + qo_idx`, aligned to the KV's end), so a span query at q sees [q − W + 1, e − 1]: HF's `(causal OR
    blockwise) AND sliding_window`. `run` of the causal instance then runs the span queries again under that plan
    (`Gemma4Meta.bidi_plan`, gathered by `span_q`) and writes them over the causal output; the causal pass's work on
    them is wasted (a span is a few hundred queries of a step's thousands)."""

    def __init__(self, pools: Any, num_qo_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype,
                 device: torch.device, window: int, max_rows: int = 0, max_pages: int = 0, scale: float = 1.0,
                 workspace: torch.Tensor | None = None, causal: bool = True, qo_per_row: int = 1,
                 upper_bounds: bool = False):
        """`upper_bounds`: decode graphs planned for uncertain lengths (MTP), whose grids may need room
        (`graph_wrapper`)."""
        super().__init__(pools, num_qo_heads, num_kv_heads, head_dim, dtype, device, max_rows=max_rows,
                         max_pages=max_pages, workspace=workspace, qo_per_row=qo_per_row)
        self.window = window
        self.scale = scale
        self.causal = causal
        if max_rows:
            self.g_first = torch.zeros(max_rows, dtype=torch.int64, device=device)  # each row's first window page
        # FlashInfer's split work items without colocated CTAs (`PrefillPlan`: 2 per SM, over the KV heads)
        self.split_items = 2 * torch.cuda.get_device_properties(device).multi_processor_count // num_kv_heads \
            if max_rows and upper_bounds else 0

    def _extra(self) -> dict[str, Any]:
        return dict(window_left=self.window - 1, sm_scale=self.scale)

    def graph_wrapper(self, rows: int) -> Any:
        """A decode bucket's wrapper; under MTP, where the bucket's rows exceed half of FlashInfer's split work items
        but not all of them, with a row bound that grows the captured grid to two chunks per row (module docstring):
        its query tiles, ceil(bound × group / tile) + rows − 1, then reach 2 × rows for any tile up to 128."""
        w = self.g_wrappers.get(rows)
        if w is None:
            w = super().graph_wrapper(rows)
            if rows < self.split_items < 2 * rows:
                w._max_total_num_rows = max(rows * self.qo_per_row, -(-(rows + 1) * 128 // (self.hq // self.hkv)))
        return w

    def plan(self, meta: Any, *, parity: int, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor,
             kv_indices: torch.Tensor, last_page_len: torch.Tensor, kv_lens: torch.Tensor, max_q: int,
             lb: np.ndarray | None = None) -> Any:
        w = self.wrappers[parity]
        w._pin_memory_int_workspace_buffer = self.pinned[parity]
        self._plan_safely(w, parity, lb, kv_lens.numpy(), qo_indptr, kv_indptr,
                          kv_indices, last_page_len, self.hq, self.hkv, self.d, self.pools.page_size,
                          causal=self.causal, q_data_type=self.dtype, kv_data_type=self.dtype, seq_lens=kv_lens,
                          max_token_per_sequence=max_q, non_blocking=True, **self._extra())
        if meta is not None:
            if self.causal:
                meta.sliding_plan = w
            else:
                meta.bidi_plan = w
        return w

    def plan_graph(self, meta: Any, *, rows: int, parity: int, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor,
                   last_page_len: torch.Tensor, kv_lens: torch.Tensor, max_q: int,
                   lb: np.ndarray | None = None) -> Any:
        w = self.graph_wrapper(rows)
        w._pin_memory_int_workspace_buffer = self.pinned[parity]
        self._plan_safely(w, parity, lb, kv_lens.numpy(), qo_indptr, kv_indptr,
                          self.g_kv_indices[:0], last_page_len, self.hq, self.hkv, self.d, self.pools.page_size,
                          graph=True, causal=True, q_data_type=self.dtype, kv_data_type=self.dtype, seq_lens=kv_lens,
                          max_token_per_sequence=max_q, non_blocking=True, **self._extra())
        if meta is not None:
            meta.sliding_plan = w
        return w

    def graph_window_indices(self, rows: int, wpage_table: torch.Tensor, slots: torch.Tensor) -> None:
        """Inside a decode graph: each row's window pages, packed at the plan's kv_indptr, from the device window table
        rows `slots` starting at the static column `g_first` (filled with the plan)."""
        indptr = self.g_kv_indptr[:rows + 1].long()
        n_pages = indptr[1:] - indptr[:-1]
        cols = (self.g_first[:rows, None] + self.g_cols[None, :]).clamp(max=wpage_table.shape[1] - 1)
        src = wpage_table.index_select(0, slots).gather(1, cols)
        dest = torch.where(self.g_cols[None, :] < n_pages[:, None], indptr[:-1, None] + self.g_cols[None, :],
                           self.g_kv_indices.shape[0] - 1)
        self.g_kv_indices.scatter_(0, dest.reshape(-1), src.reshape(-1))

    def run(self, meta: Any, layer: Any, q: torch.Tensor, out: torch.Tensor) -> None:  # type: ignore[override]
        kv = self.pools.wkv[layer.index]
        if meta.fix_len:  # MTP: the rows' true KV from their first window page, decided on the device
            self.run_fixed(meta.sliding_plan, q, kv, out, meta.n_rows, meta.kv_named_dev)
        else:
            meta.sliding_plan.run(q, kv, out=out)
        if meta.bidi_plan is not None:  # image spans: their queries again, without the causal mask (class docstring)
            out.index_copy_(0, meta.span_q, meta.bidi_plan.run(q.index_select(0, meta.span_q), kv))


def window_page_indices(table: torch.Tensor, slots: torch.Tensor, first: torch.Tensor, n_pages: torch.Tensor,
                        indptr: torch.Tensor, max_pages: int, total: int, arange: torch.Tensor) -> torch.Tensor:
    """`page_indices` from each row's first column `first` [n] (int64, device) instead of column 0."""
    col = arange[:max_pages]
    cols = (first[:, None] + col[None, :]).clamp(max=table.shape[1] - 1)
    tbl = table.index_select(0, slots).gather(1, cols)
    dest = torch.where(col[None, :] < n_pages[:, None], indptr[:-1, None] + col[None, :], total)
    out = torch.empty(total + 1, dtype=torch.int32, device=table.device)
    out.scatter_(0, dest.reshape(-1), tbl.reshape(-1))
    return out[:total]


class FlashInferVisionAttention:
    """Bidirectional attention within each image of an encoder call: FlashInfer's ragged prefill kernel, non-causal,
    q = k = v lengths per image (`cu_seqlens`). A head dim FlashInfer does not compile (72 in both towers) is
    zero-padded to the next one it does (`pad`; each tower's qkv op pads the inputs), with the softmax scale of the
    true dim unless `scale` is given: zeros add nothing to q·k and give zero output columns, which the caller cuts off.

    **Plans in flight.** The paged wrappers' hazard (module docstring, "Planning ahead is the hazard") holds here: an
    encoder call is planned when it is queued, and the stream may reach it much later. A step's ENCODE is queued while
    the step before it still runs, and with batch invariance it makes one call per image
    (`runner.cuda.images.ImageEncoder.per_image`), so the step's parity cannot key the buffers. Each plan takes the next
    of `PLANS` wrappers in turn, each with its own page-locked buffer and an event recorded right after its plan's copy.
    A plan into a buffer whose event has not completed waits for it first (`waits` counts those; the pattern of torch's
    caching host allocator). The wrappers share one device int workspace and the float workspace of the runner's
    FlashInfer instances (the tower runs before the step's forward, on the same stream), which only stream-ordered
    copies and kernels touch. So a call's attention must run before the next call is planned, as every caller does
    (`run` checks it)."""

    name = "flashinfer"
    HEAD_DIMS = (64, 128, 256)
    # The plans in flight without a wait: up to two tower calls in each of the two steps in flight (`runner.base`). A
    # step makes one call, or one per image under batch invariance, where the steps in flight of a loaded server can
    # encode more than four images: a plan then waits for its buffer, which costs some overlap with the step before,
    # never a wrong plan
    PLANS = 4

    def __init__(self, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device,
                 workspace: torch.Tensor | None = None, pad: int | None = None, scale: float | None = None,
                 event: Any = None):
        """`event`: makes the wrappers' events (`torch.cuda.Event`; tests pass stand-ins)."""
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        self.h, self.d, self.dtype = heads, head_dim, dtype
        self.scale = scale if scale is not None else head_dim ** -0.5  # Gemma 4's tower: 1.0
        self.pad = pad if pad is not None else (head_dim if head_dim in self.HEAD_DIMS else
                                                next(x for x in self.HEAD_DIMS if x >= head_dim))
        self.workspace = workspace if workspace is not None else \
            torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        # Each wrapper keeps the page-locked buffer FlashInfer gave it, as large as the int workspace it gave it (its
        # plans allocate within the int workspace), and they all plan into the first one's int workspace. FA2 by name:
        # "auto" gives it on SM120 for these plans, but picks other kernels on other GPUs or settings, which the image
        # does not hold and the tower's batch invariance was not proved on
        self.wrappers = [BatchPrefillWithRaggedKVCacheWrapper(self.workspace, "NHD", backend="fa2")
                         for _ in range(self.PLANS)]
        self.int_workspace = self.wrappers[0]._int_workspace_buffer
        for w in self.wrappers[1:]:
            w._int_workspace_buffer = self.int_workspace
        self.pinned = [w._pin_memory_int_workspace_buffer for w in self.wrappers]
        self.events = [(event or torch.cuda.Event)() for _ in range(self.PLANS)]
        self.turn = 0  # the wrapper the next plan takes
        self.planned: Any = None  # the last call planned, whose plan the int workspace holds once its copy runs
        self.waits = 0  # plans that waited for their buffer's last copy

    def plan(self, meta: Any) -> None:
        i = self.turn
        self.turn = (i + 1) % self.PLANS
        w, done = self.wrappers[i], self.events[i]
        if not done.query():  # the buffer's last copy is still queued: it must run before the buffer is written
            from ..runner.cuda import sync_check

            self.waits += 1
            sync_check.blocked("the vision tower's plan buffers are all in flight (FlashInferVisionAttention.plan)")
            done.synchronize()
        cu = _pinned32(meta.cu_host)
        w.plan(cu, cu, self.h, self.h, self.pad, causal=False, sm_scale=self.scale, q_data_type=self.dtype,
               kv_data_type=self.dtype, non_blocking=True)
        done.record()
        meta.attn, self.planned = w, meta

    def run(self, meta: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q, k, v [n, h, pad] (zero-padded past the head dim by the tower's qkv op) -> [n, h, pad]."""
        if meta is not self.planned:  # the shared int workspace holds a later call's plan (class docstring)
            raise RuntimeError("vision attention run after another encoder call was planned")
        return meta.attn.run(q, k, v)


def _pinned32(a: Any) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a, np.int32))
    return t.pin_memory() if torch.cuda.is_available() else t


def check_plan_info(attn: FlashInferAttention) -> str | None:
    """None if FlashInfer's plan info reads as `_plan_safely` expects, else what differs. Plans one row of a few
    queries over 64 pages split by `fixed_split_size` (4 and then 8 pages: its query tiles times 16 or 8 work items)
    and once with `disable_split_kv`."""
    import inspect

    w = attn.wrappers[0]
    params = inspect.signature(w.plan).parameters
    for name in ("fixed_split_size", "disable_split_kv", "seq_lens", "max_token_per_sequence", "non_blocking"):
        if name not in params:
            return f"plan() takes no {name!r}"
    ps = attn.pools.page_size
    n_pages, q = 64, max(1, attn.qo_per_row)
    kv_len = n_pages * ps
    dev = attn.int_workspace.device
    kv_indices = torch.arange(n_pages, dtype=torch.int32, device=dev)
    w._pin_memory_int_workspace_buffer = attn.pinned[0]
    args = (_pinned32([0, q]), _pinned32([0, n_pages]), kv_indices, _pinned32([ps]), attn.hq, attn.hkv, attn.d, ps)
    kw = dict(causal=True, q_data_type=attn.dtype, kv_data_type=attn.dtype, seq_lens=_pinned32([kv_len]),
              max_token_per_sequence=q, non_blocking=True)
    for chunk in (4, 8):
        w.plan(*args, **kw, fixed_split_size=chunk)
        info = getattr(w, "_plan_info", None)
        if info is None or len(info) != _PLAN_INFO_LEN:
            return f"the plan info has {None if info is None else len(info)} entries, not {_PLAN_INFO_LEN}"
        if int(info[_PLAN_SPLIT_KV]) != 1:
            return f"a plan split into chunks of {chunk} pages reports split_kv {int(info[_PLAN_SPLIT_KV])}"
        off = int(info[_PLAN_CHUNK_SIZE_OFFSET])
        buf = attn.pinned[0]
        if not 0 <= off <= buf.numel() - 4:
            return f"the chunk size's offset {off} is outside the page-locked buffer"
        got = int(buf[off:off + 4].view(torch.int32)[0])
        if got != chunk * ps:
            return f"the chunk size at offset {off} reads {got}, not {chunk * ps} ({chunk} pages)"
        tile = int(info[_PLAN_CTA_TILE_Q])
        if tile not in (16, 32, 64, 128):
            return f"the query tile reads {tile}"
        work = -(-q * (attn.hq // attn.hkv) // tile) * (n_pages // chunk)
        if int(info[_PLAN_PADDED]) != work:
            return f"a plan of {work} work items reports a padded batch of {int(info[_PLAN_PADDED])}"
    w.plan(*args, **kw, disable_split_kv=True)
    if int(w._plan_info[_PLAN_SPLIT_KV]) != 0:
        return "a plan with disable_split_kv reports a split"
    return None


def check_lengths(attn: FlashInferAttention, kv_dtype: torch.dtype) -> str | None:
    """None if an attention plan made for upper-bound KV lengths, whose last-page lengths `fix_lengths` rewrites to
    the true ones on the device, runs as the plan made for the true lengths; else what differs."""
    ps = attn.pools.page_size
    dev = attn.int_workspace.device
    q_rows = max(1, attn.qo_per_row)
    true_len = np.asarray([q_rows + 13, 2 * ps, 3 * ps + q_rows + 1], np.int64)
    ub = true_len + np.asarray([3, ps - 1, 6])
    rows = len(true_len)
    g = torch.Generator(device=dev).manual_seed(0)
    n_pages = int(sum(-(-ub // ps)))
    kv = torch.randn((n_pages + 1, 2, ps, attn.hkv, attn.d), generator=g, device=dev).to(kv_dtype)
    q = torch.randn((rows * q_rows, attn.hq, attn.d), generator=g, device=dev).to(attn.dtype)
    outs = []
    for lens, fix in ((ub, True), (true_len, False)):
        pages = -(-lens // ps)
        indptr = np.concatenate([[0], np.cumsum(-(-ub // ps))])  # every row keeps ub's pages
        used = np.concatenate([np.arange(indptr[i], indptr[i] + pages[i]) for i in range(rows)]) + 1
        kv_indptr = np.concatenate([[0], np.cumsum(pages)])
        w = attn.plan(None, parity=0, qo_indptr=_pinned32(np.arange(rows + 1) * q_rows), kv_indptr=_pinned32(kv_indptr),
                      kv_indices=torch.as_tensor(used, dtype=torch.int32, device=dev),
                      last_page_len=_pinned32(lens - (pages - 1) * ps), kv_lens=_pinned32(lens), max_q=q_rows,
                      lb=true_len if fix else None)
        out = torch.empty_like(q)
        if fix:  # serialized after the rewrite, as the serving path runs it
            attn.run_fixed(w, q, kv, out, rows, torch.as_tensor(true_len, device=dev))
        else:
            w.run(q, kv, out=out)
        outs.append(out.float())
    torch.cuda.synchronize(dev)
    if not torch.isfinite(outs[0]).all():
        return "the upper-bound plan's output has non-finite values"
    err = float((outs[0] - outs[1]).abs().max())
    if err > 2e-2:
        return f"the upper-bound plan with rewritten lengths differs from the exact plan by {err:.3g}"
    if attn.max_rows:  # the decode graphs' wrappers must read the length buffers they were given (`fix_lengths`)
        wg = attn.graph_wrapper(1)
        for name, mine in (("_paged_kv_indptr_buf", attn.g_kv_indptr), ("_paged_kv_last_page_len_buf", attn.g_last)):
            buf = getattr(wg, name, None)
            if not isinstance(buf, torch.Tensor) or buf.data_ptr() != mine.data_ptr():
                return f"the CUDA-graph wrapper's {name} is not the buffer it was given"
        if not hasattr(wg, "_max_total_num_rows"):  # the sliding layers' larger grids (`graph_wrapper`)
            return "the CUDA-graph wrapper has no _max_total_num_rows"
    return None


def self_check(attn: FlashInferAttention, kv_dtype: torch.dtype) -> dict[str, Any]:
    """The startup check of what upper-bound planning reads from FlashInfer (module docstring, "The self-check"),
    once per process for each kind of caller: with decode graphs (`attn.max_rows`) the length check also reads the
    graph wrappers' buffers. A later call of the same kind reuses the findings and fails as the first one did; the
    plan info is the first caller's (its `attn`), which alone sets `_plan_info_ok`. Returns a copy of the result."""
    global _plan_info_ok
    graphs = bool(attn.max_rows)
    found = _checked.get(graphs)
    first = found is None
    if first:
        try:
            info = check_plan_info(attn)
        except Exception as exc:  # an API that moved: the same fallback
            info = f"{type(exc).__name__}: {exc}"
        if not _checked:
            _plan_info_ok = info is None
        try:
            lengths = check_lengths(attn, kv_dtype)
        except Exception as exc:
            lengths = f"{type(exc).__name__}: {exc}"
        result = {"flashinfer": versions()["flashinfer"], "plan_info": info or "ok", "lengths": lengths or "ok"}
        found = _checked[graphs] = (result, info, lengths)
    result, info, lengths = found
    ver = result["flashinfer"]
    if lengths is not None:
        raise RuntimeError(f"FlashInfer {ver}: upper-bound attention planning (MTP) does not work with this version "
                           f"({lengths}); use flashinfer-python 0.7.2 or speculative.num_speculative_tokens=0")
    if info is not None and attn.max_rows:
        raise RuntimeError(f"FlashInfer {ver}: the plan info does not read as in 0.7.2 ({info}), and decode graphs "
                           f"need it to keep their KV split safe; use flashinfer-python 0.7.2 or "
                           f"dev.runner.cuda.graphs=false")
    if first:
        import logging

        log = logging.getLogger(__name__)
        if info is None:
            log.info("FlashInfer %s: the upper-bound planning self-check passed (plan info, length rewrite)", ver)
        else:
            log.warning("FlashInfer %s: the plan info does not read as in 0.7.2 (%s); plans with uncertain KV "
                        "lengths disable the KV split instead (MTP stays correct, long-context attention gets slower)",
                        ver, info)
    return dict(result)


def page_indices(page_table: torch.Tensor, slots: torch.Tensor, n_pages: torch.Tensor, indptr: torch.Tensor,
                 max_pages: int, total: int, arange: torch.Tensor) -> torch.Tensor:
    """Packed page ids of every row, gathered on the device from the page table (sizes from the host)."""
    tbl = page_table.index_select(0, slots)[:, :max_pages]
    col = arange[:max_pages]
    dest = torch.where(col[None, :] < n_pages[:, None], indptr[:-1, None] + col[None, :], total)
    out = torch.empty(total + 1, dtype=torch.int32, device=page_table.device)
    out.scatter_(0, dest.reshape(-1), tbl.reshape(-1))
    return out[:total]


def flashinfer_gdn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                           beta: torch.Tensor, pool: torch.Tensor, slots: torch.Tensor, meta: StepMeta) -> torch.Tensor:
    """q, k [Tp, HK, K] normalised (model dtype), v [Tp, HV, V] contiguous, the decay alpha = exp(g) and beta
    [Tp, HV] fp32; pool [S, HV, V, K] fp32, the layer's view of the recurrent state pool, and slots [n] (int64),
    each prefill row's: unique, or the slot's state is undefined after it (the runner's profile and capture steps
    share one scratch slot). Each sequence reads and writes its slot's state in place (`state_indices`). Returns
    o [Tp, HV, V]. FlashInfer's per-sequence route at every step (module docstring, "The per-sequence route at every
    step"). A step's mid-chunk checkpoints (`meta.ckpt`) cut their rows into consecutive calls (module docstring,
    "Checkpoints by cutting rows")."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    if meta.ckpt is not None:
        return _gdn_cut_rows(q, k, v, alpha, beta, pool, slots, meta.ckpt)
    o, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=alpha, beta=beta, scale=None, initial_state=pool,
                                  output_state=pool, state_indices=slots, output_final_state=True,
                                  cu_seqlens=meta.pf_cu, use_qk_l2norm_in_kernel=False, use_cp=False)
    return o


def _gdn_cut_rows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                  pool: torch.Tensor, slots: torch.Tensor, ck: Any) -> torch.Tensor:
    """Runs the calls `gdn_segments` made (`ck.segments`) in order (module docstring, "Checkpoints by cutting rows"),
    each on the state pool as `flashinfer_gdn_prefill`'s call; a checkpoint is its segment's final state, copied into
    the workspace where FlashInfer's every-64 checkpoint would lie before the next call writes its slot (the caller
    copies the requested ones out)."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    kw = dict(scale=None, initial_state=pool, output_state=pool, output_final_state=True,
              use_qk_l2norm_in_kernel=False, use_cp=False)
    o = torch.empty_like(v)
    for tok, cu, rows, ck_seq, ck_src in ck.segments:
        ids = slots.index_select(0, rows)
        oj, _ = chunk_gated_delta_rule(q=q.index_select(0, tok), k=k.index_select(0, tok), v=v.index_select(0, tok),
                                       g=alpha.index_select(0, tok), beta=beta.index_select(0, tok),
                                       state_indices=ids, cu_seqlens=cu, **kw)
        o.index_copy_(0, tok, oj)
        if ck_seq is not None:
            ck.workspace.index_copy_(0, ck_src, pool.index_select(0, ids.index_select(0, ck_seq)))
    return o


def gdn_segments(pf_cu: np.ndarray, rows: np.ndarray, offs: np.ndarray, src: np.ndarray,
                 put: Any) -> list[tuple[Any, Any, Any, Any, Any]]:
    """The calls `_gdn_cut_rows` makes for a step's mid-chunk checkpoints: prefill row r (of `pf_cu`'s,
    tokens [pf_cu[r], pf_cu[r + 1])) cut at its checkpoints' offsets `offs` (multiples of 64 from its start, where a cut
    replays the same chunks), its segment j in call j. Per call (device int64 views staged by `put`): the segments'
    token indices, their cu_seqlens, their rows, and which of its sequences end at a checkpoint with that checkpoint's
    workspace index (`src`), or None, None."""
    n = len(pf_cu) - 1
    cuts: list[dict[int, int]] = [{} for _ in range(n)]
    for r, o, x in zip(rows.tolist(), offs.tolist(), src.tolist()):
        cuts[r][int(o)] = int(x)
    bounds = [[0, *sorted(c), int(pf_cu[r + 1] - pf_cu[r])] for r, c in enumerate(cuts)]
    calls = []
    for j in range(max(len(b) for b in bounds) - 1):
        live = [r for r in range(n) if len(bounds[r]) - 1 > j]
        lo = np.asarray([pf_cu[r] + bounds[r][j] for r in live], np.int64)
        hi = np.asarray([pf_cu[r] + bounds[r][j + 1] for r in live], np.int64)
        lens = hi - lo
        tok = np.repeat(lo - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(int(lens.sum()))
        ends = [(i, cuts[r][bounds[r][j + 1]]) for i, r in enumerate(live) if j + 2 < len(bounds[r])]
        view = lambda a: torch.as_tensor(a) if put is None else put(np.asarray(a, np.int64))  # noqa: E731
        calls.append((view(tok), view(np.concatenate([[0], np.cumsum(lens)])), view(np.asarray(live, np.int64)),
                      view([i for i, _ in ends]) if ends else None, view([x for _, x in ends]) if ends else None))
    return calls


def versions() -> dict[str, Any]:
    import flashinfer

    return {"flashinfer": getattr(flashinfer, "__version__", "?")}

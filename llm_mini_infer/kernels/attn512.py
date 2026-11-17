"""Gemma 4's full-attention layers on CUDA: our paged attention kernel for head dim 512 with K derived from the stored
V plane (`csrc/attn512.cu`; the reference is `models.gemma4.backend.TorchFullAttention`).

**Planning** (`Attn512.plan`, on the host from the step's planned lengths: under MTP each decode row's upper bound,
below). A row is (slot, first token, n_query, KV length after the step); a work item is (row, tile of 16 query rows, KV
head, split), and a group the items of one (row, tile, KV head). The plan copies its row, item and group tables to the
device in stream order, from the runner's page-locked staging (two parities, as FlashInfer's plans: a step is planned
while the one before may still be queued).

**The split** (`choose_splits`: the wave split). A launch with fewer groups than the GPU holds CTAs at once splits each
group's keys, so more of them run: as many splits as fill one wave of resident CTAs, floor(resident / groups)
(`resident` from the CUDA occupancy API: 2 CTAs to an SM on SM120). A wave left under `WAVE_FILL` (60%) full takes
about 5/3 of a wave instead; rounding up past a wave is never taken, since a second wave of a few items nearly doubles
the kernel. At most 32 splits (the merge's lanes) and, on eager plans, each at least `MIN_SPLIT_KEYS` keys of the
longest row. In a decode graph the split count comes from the replay's real rows: a bucket's padding rows get one item
per group (their KV is the null page's 1 + k positions). The split count sets the merge order, so another count gives
other last bits.

**The fixed split** (`split_keys` > 0; docs/architecture.md, "Batch invariance"). The wave split makes a row's merge
order depend on the rows beside it, and its chunks on its tile's causal end. With a fixed split every split s holds the
keys [s × split_keys, (s + 1) × split_keys), whatever the batch, the tile or the row's planned length: a decode row's
group takes ceil(kv / split_keys) splits of its upper-bound KV length (a split past the tile's true keys is empty, and
the merge skips it). Prefill rows (explicit tokens) take one split, the whole of their keys in one CTA, whatever the
step (`plan`'s `fixed_rows`).

**CUDA graphs** (decode buckets): every row has `qo_per_row` queries (one, or the 1 + k of an MTP verify row). The
captured launches read static row, item and group tables of a bucket's capacity (padding items exit at once); before
each replay `plan_graph` rewrites the row table, and the item and group tables when they changed: its real rows under
the wave split, its rows' split counts under the fixed split.

**Lengths on the device (MTP).** The kernel reads each row's KV length from the device row table and computes its
splits' key ranges from it at run time (an empty split keeps m = −inf and the merge skips it), so a plan made for an
upper bound runs exactly once the table's lengths are rewritten on the device: `run` does it from `meta.kv_len_dev`
when `meta.fix_len` is set (the drafter's rows), the runner's step body otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import build
from .ops import _check, _p, _stream

ROWS = 16  # query rows per tile
KT = 16  # keys per KV tile
MIN_SPLIT_KEYS = 256  # an eager plan's splits hold at least this many keys of the longest row
MAX_SPLITS = 32  # the merge's lanes (csrc/attn512.cu MAX_MERGE_SPLITS)
WAVE_FILL = 0.6  # a wave left under this share full takes about 5/3 of a wave (module docstring; fitted on SM120)
SPLIT_KEYS = 256  # the fixed split's keys a split (module docstring), unless longer rows need more (`fixed_split_keys`)


def tiles(n_query: int, group: int) -> int:
    return -(-n_query * group // ROWS)


def plan_items(nq: np.ndarray, group: int, kv_heads: int, n_splits: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(items int32 [n_items, 4]: (row, tile, KV head, split | n << 16); groups int32 [n_groups, 2]: (first item, n)),
    grouped by (row, tile, KV head) with a group's n splits consecutive; `n_splits` per row, or one for all."""
    nq = np.asarray(nq, np.int64)
    ns = np.broadcast_to(np.asarray(n_splits, np.int64), nq.shape)
    per_row = (-(-nq * group // ROWS)) * kv_heads  # groups per row
    g_row = np.repeat(np.arange(len(nq)), per_row)
    g_local = np.arange(len(g_row)) - np.repeat(np.cumsum(per_row) - per_row, per_row)
    g_ns = ns[g_row]
    first = np.cumsum(g_ns) - g_ns
    i_grp = np.repeat(np.arange(len(g_row)), g_ns)
    split = np.arange(len(i_grp)) - first[i_grp]
    items = np.stack([g_row[i_grp], g_local[i_grp] // kv_heads, g_local[i_grp] % kv_heads,
                      split | (g_ns[i_grp] << 16)], axis=1).astype(np.int32)
    return items, np.stack([first, g_ns], axis=1).astype(np.int32)


def choose_splits(n_groups: int, max_kv: int, sms: int, resident: int = 0) -> int:
    """Splits per group (module docstring, the split). `resident`: CTAs the GPU holds at once (0: 2 × SMs)."""
    r = resident or 2 * sms
    if n_groups >= r:
        return 1
    s = r // n_groups  # one wave
    if n_groups * s < WAVE_FILL * r:  # it would leave CTAs idle: about 5/3 of a wave (module docstring)
        s = (10 * r + 3 * n_groups) // (6 * n_groups)  # 5r / 3g rounded: 3 (r / 2 < g < 0.6 r here)
    return max(1, min(s, MAX_SPLITS, max(1, max_kv // MIN_SPLIT_KEYS)))


def bucket_splits(real: int, per_row: int, sms: int, resident: int, max_items: int) -> int:
    """The splits of each real row's group in a decode graph's replay of `real` real rows, each `per_row` groups
    (padding rows one item a group: module docstring), within `max_items` split items."""
    s = choose_splits(real * per_row, 1 << 20, sms, resident)
    return max(1, min(s, max_items // (real * per_row)))


def fixed_split_keys(max_kv: int, keys: int = SPLIT_KEYS) -> int:
    """The fixed split's keys a split for rows of up to `max_kv` keys: `keys`, or the least multiple of 16 that keeps
    such a row within the merge's `MAX_SPLITS`."""
    need = -(-max_kv // MAX_SPLITS)
    return max(keys, -(-need // KT) * KT)


def fixed_splits(kv: np.ndarray | int, split_keys: int) -> np.ndarray:
    """Each decode row's split count under the fixed split: its KV length's splits of `split_keys` keys (at least
    one)."""
    return np.maximum(-(-np.asarray(kv, np.int64) // split_keys), 1)


def bucket_capacity(bucket: int, per_row: int, sms: int, resident: int, max_items: int) -> tuple[int, bool]:
    """The most items any real row count makes in a decode bucket, and whether any splits (the captured launch's
    capacity and kernel)."""
    cap, split = bucket * per_row, False
    for real in range(1, bucket + 1):
        s = bucket_splits(real, per_row, sms, resident, max_items)
        cap = max(cap, real * per_row * s + (bucket - real) * per_row)
        split |= s > 1
    return cap, split


class Attn512:
    """The full-attention backend (`plan`, `plan_graph`, `run`) for `models.gemma4.backend.Gemma4Context`."""

    name = "attn512"

    def __init__(self, pools: Any, rotary: Any, num_heads: int, num_kv_heads: int, rot: int, device: torch.device,
                 max_tokens: int, max_rows: int, scale: float = 1.0, qo_per_row: int = 1, share: Attn512 | None = None,
                 split_keys: int = 0, max_kv: int = 0):
        """`qo_per_row`: a decode graph row's queries (1 + k under MTP). `share`: an instance whose split partials this
        one uses when large enough (the MTP drafter's attention runs after the target's, on the same stream).
        `split_keys`: the fixed split's keys a split (module docstring; 0: the wave split), for rows of up to `max_kv`
        keys."""
        self.pools = pools
        self.cos, self.sin = rotary.cos["full_attention"], rotary.sin["full_attention"]
        self.h, self.hkv, self.rot, self.scale = num_heads, num_kv_heads, rot, scale
        self.group = num_heads // num_kv_heads
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        lib = build.load()
        occ = lib.lmk_attn512_occupancy() if lib is not None else -1
        self.resident = (occ if occ > 0 else 2) * self.sms  # CTAs at once (2 an SM on SM120)
        self.device = device
        self.qo_per_row = qo_per_row
        self.split_keys, self.max_splits = split_keys, 0
        if split_keys:
            if split_keys % KT or max_kv < 1:
                raise ValueError(f"attn512: a fixed split of {split_keys} keys (a multiple of {KT}) for rows of "
                                 f"{max_kv} keys")
            self.max_splits = int(fixed_splits(max_kv, split_keys))
            if self.max_splits > MAX_SPLITS:
                raise ValueError(f"attn512: rows of {max_kv} keys take {self.max_splits} splits of {split_keys} keys "
                                 f"(the merge takes {MAX_SPLITS}: `fixed_split_keys`)")
        per_row = max(1, max_rows * tiles(qo_per_row, self.group) * num_kv_heads)
        # Split partials for the most items a split launch can have: at most one wave of split items, or under
        # WAVE_FILL 3 splits of fewer than 0.6 × resident groups (at most 9 × resident // 5); every row of a
        # decode-sized step's tiles, times MAX_SPLITS, caps it for small steps. The fixed split: every decode row's
        # groups at the most splits a row takes (its rows come first in a step, so their items do)
        max_items = min(per_row * MAX_SPLITS, 9 * self.resident // 5 + 2 * MAX_SPLITS)
        if split_keys:
            max_items = max(max_items, per_row * self.max_splits)
        if share is not None and share.max_split_items >= max_items:
            self.max_split_items, self.o_part, self.ml_part = share.max_split_items, share.o_part, share.ml_part
        else:
            self.max_split_items = max_items
            self.o_part = torch.empty((max_items, ROWS, 512), dtype=torch.float32, device=device)
            self.ml_part = torch.empty((max_items, ROWS, 2), dtype=torch.float32, device=device)
        # Tables, per parity (module docstring)
        max_groups = (tiles(max_tokens, self.group) + max_rows) * num_kv_heads
        self.rows = [torch.zeros((max_rows + 1, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.items = [torch.zeros((max_groups + max_items, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.groups = [torch.zeros((max_groups, 2), dtype=torch.int32, device=device) for _ in range(2)]
        # decode buckets: static row, item and group tables, the launch's sizes (items, groups, split), the host
        # tables per real row count (the wave split) and what the bucket's last replay's tables were made for (its
        # real rows; the fixed split: every row's split count)
        self.g_rows: dict[int, torch.Tensor] = {}
        self.g_tabs: dict[int, tuple[torch.Tensor, torch.Tensor, int, int, int]] = {}
        self.g_host: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.g_last: dict[int, Any] = {}

    def _splits(self, n_groups: int, max_kv: int) -> int:
        s = choose_splits(n_groups, max_kv, self.sms, self.resident)
        if n_groups * s <= self.max_split_items:
            return s
        return max(1, self.max_split_items // n_groups)

    def plan(self, meta: Any, *, parity: int, rows: np.ndarray, stage: Any, fixed_rows: int = 0) -> None:
        """Eager steps: `rows` int32 [n, 4] (slot, first token, n_query, KV length after the step); `stage` puts an
        int32 array into the step's page-locked staging and returns its host view (the runner's `put32`). Under the
        fixed split, the first `fixed_rows` rows are decode rows (split by their own lengths) and the rest prefill rows
        (one split each: module docstring)."""
        nq, kv_len = rows[:, 2].astype(np.int64), rows[:, 3].astype(np.int64)
        if self.split_keys:
            splits = np.ones(len(rows), np.int64)
            splits[:fixed_rows] = fixed_splits(kv_len[:fixed_rows], self.split_keys)
            split = bool((splits > 1).any())
        else:
            groups = int(np.sum(-(-nq * self.group // ROWS))) * self.hkv
            splits = self._splits(groups, int(kv_len.max()))
            split = splits > 1
        items, grp = plan_items(nq, self.group, self.hkv, splits)
        if self.split_keys and split:  # the decode rows' items come first: the partials they write must fit
            last = int(np.flatnonzero(items[:, 3] >> 16 > 1)[-1]) + 1
            if last > self.max_split_items:
                raise RuntimeError(f"attn512: {last} split items over the partials' {self.max_split_items}")
        dr, di, dg = self.rows[parity], self.items[parity], self.groups[parity]
        dr[:len(rows)].copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        di[:len(items)].copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
        if split:
            dg[:len(grp)].copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
        meta.full_plan = (dr, di, len(items), dg, len(grp), int(split))

    def _graph_host(self, bucket: int, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A bucket's item and group tables for split counts `ns` per row (padding rows 1), padded to the bucket's
        capacity with padding items (row −1)."""
        items, grp = plan_items(np.full(bucket, self.qo_per_row, np.int64), self.group, self.hkv, ns)
        _, _, cap_i, _, _ = self.g_tabs[bucket]
        if len(items) > cap_i:
            raise RuntimeError(f"attn512: {len(items)} items for bucket {bucket} (capacity {cap_i})")
        return np.concatenate([items, np.full((cap_i - len(items), 4), -1, np.int32)]), grp

    def _wave_host(self, bucket: int, real: int) -> tuple[np.ndarray, np.ndarray]:
        """The wave split's tables for `real` real rows (the rest padding), built once per bucket and real rows."""
        key = (bucket, real)
        got = self.g_host.get(key)
        if got is None:
            per_row = tiles(self.qo_per_row, self.group) * self.hkv
            ns = np.ones(bucket, np.int64)
            ns[:real] = bucket_splits(real, per_row, self.sms, self.resident, self.max_split_items)
            got = self.g_host[key] = self._graph_host(bucket, ns)
        return got

    def graph_tables(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int, int]:
        """A decode bucket's static row table and its item and group tables (`qo_per_row` queries per row), with the
        sizes its captured launch takes: items, groups, split."""
        if bucket not in self.g_tabs:
            per_row = tiles(self.qo_per_row, self.group) * self.hkv
            n_groups = bucket * per_row
            if self.split_keys:  # every row at the most splits a row takes
                cap, any_split = n_groups * self.max_splits, self.max_splits > 1
            else:
                cap, any_split = bucket_capacity(bucket, per_row, self.sms, self.resident, self.max_split_items)
            self.g_tabs[bucket] = (torch.full((cap, 4), -1, dtype=torch.int32, device=self.device),
                                   torch.full((n_groups, 2), -1, dtype=torch.int32, device=self.device), cap,
                                   n_groups, int(any_split))
            self.g_last[bucket] = None
            self.g_rows[bucket] = torch.zeros((bucket, 4), dtype=torch.int32, device=self.device)
        it, gt, n_items, n_groups, split = self.g_tabs[bucket]
        return self.g_rows[bucket], it, n_items, gt, n_groups, split

    def plan_graph(self, meta: Any, *, bucket: int, rows: np.ndarray, stage: Any, real: int | None = None) -> None:
        """Before a decode graph's replay: its row table (int32 [bucket, 4], padding rows included) into the static
        buffer the captured launch reads; `real`, the real rows (the first ones; None: all of them), whose groups split
        to fill the wave, or by their own lengths under the fixed split (module docstring): the item and group tables
        are rewritten when they changed."""
        dr, it, n_items, gt, n_groups, split = self.graph_tables(bucket)
        dr.copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        real = bucket if real is None else real
        if self.split_keys:
            ns = np.ones(bucket, np.int64)
            ns[:real] = fixed_splits(rows[:real, 3], self.split_keys)
            key: Any = ns.tobytes()
        else:
            key = real
        if self.g_last[bucket] != key:  # stream order: replays already queued read the old tables first
            items, grp = self._graph_host(bucket, ns) if self.split_keys else self._wave_host(bucket, real)
            it.copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            gt.copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
            self.g_last[bucket] = key
        meta.full_plan = (dr, it, n_items, gt, n_groups, split)

    def run(self, meta: Any, layer: Any, q: torch.Tensor, out: torch.Tensor) -> None:
        rows, items, n_items, groups, n_groups, split = meta.full_plan
        if meta.fix_len:  # the true KV lengths, decided on the device (module docstring)
            rows[:meta.n_rows, 3].copy_(meta.kv_len_dev[:meta.n_rows])
        kv = self.pools.fkv[layer.index]  # [pages, page_size, Hkv, 512]
        pt = meta.page_table
        _check(build.load().lmk_attn512(
            _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.shape[1], _p(pt), pt.stride(0),
            _p(layer.k_norm.weight), _p(self.cos), _p(self.sin), self.rot, self.scale, _p(rows), _p(items), n_items,
            _p(groups), n_groups, split, self.split_keys, _p(out), _p(self.o_part) if split else _p(None),
            _p(self.ml_part) if split else _p(None), _stream(q)), "attn512")

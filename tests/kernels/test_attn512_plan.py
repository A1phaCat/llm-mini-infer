"""attn512's work planning (`kernels.attn512`; architecture.md, "Kernels"), on the CPU: the split, the item and group
tables, and the decode graphs' per-replay tables. The kernel's numerics against the reference are
tests/kernels/test_gemma_kernels.py's (GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_mini_infer.kernels import attn512 as am

SMS, RESIDENT = 188, 376  # the RTX PRO 6000's SMs, and attn512's CTAs at once (2 an SM: 45 KiB of shared memory each)
G, HKV = 8, 4  # Gemma 4 31B's full layers: 32 query heads over 4 KV heads
ITEMS = 740  # Attn512's split partials there: 9 × resident // 5 + 2 × MAX_SPLITS


def direct_items(nq: np.ndarray, group: int, kv_heads: int, n_splits: int) -> np.ndarray:
    """The items of one split count, enumerated directly: (row, tile, KV head, split), the split varying fastest."""
    t = -(-nq * group // am.ROWS)
    row = np.repeat(np.arange(len(nq)), t * kv_heads * n_splits)
    starts = np.concatenate([[0], np.cumsum(t)[:-1]])
    flat = np.arange(int(t.sum()) * kv_heads * n_splits) - np.repeat(starts * kv_heads * n_splits,
                                                                      t * kv_heads * n_splits)
    split = flat % n_splits
    kvh = (flat // n_splits) % kv_heads
    tile = flat // (n_splits * kv_heads)
    return np.stack([row, tile, kvh, split], axis=1).astype(np.int32)


@pytest.mark.parametrize("nq", [[1, 1, 1], [6, 6], [8, 3, 64, 1], [130]])
@pytest.mark.parametrize("splits", [1, 2, 5, 32])
def test_uniform_items_are_the_direct_enumeration(nq, splits) -> None:
    """One split count for every group (an eager plan's): the directly enumerated items (the split word's low 16 bits),
    each carrying its group's count, and groups at multiples of it."""
    nq = np.asarray(nq, np.int64)
    items, groups = am.plan_items(nq, G, HKV, splits)
    direct = direct_items(nq, G, HKV, splits)
    assert np.array_equal(items[:, :3], direct[:, :3])
    assert np.array_equal(items[:, 3] & 0xFFFF, direct[:, 3]) and (items[:, 3] >> 16 == splits).all()
    assert np.array_equal(groups[:, 0], np.arange(len(groups)) * splits) and (groups[:, 1] == splits).all()


def test_mixed_items_and_their_groups() -> None:
    """A split count per row: each group's items consecutive, its first item and count in the group table, the rows'
    groups in (row, tile, KV head) order."""
    nq = np.asarray([6, 6, 6], np.int64)
    items, groups = am.plan_items(nq, G, HKV, [7, 1, 3])
    per_row = am.tiles(6, G) * HKV
    assert len(groups) == 3 * per_row and len(items) == per_row * (7 + 1 + 3)
    for g, (first, n) in enumerate(groups.tolist()):
        row, local = divmod(g, per_row)
        assert n == (7, 1, 3)[row]
        span = items[first:first + n]
        assert (span[:, 0] == row).all() and (span[:, 1] == local // HKV).all() and (span[:, 2] == local % HKV).all()
        assert np.array_equal(span[:, 3] & 0xFFFF, np.arange(n)) and (span[:, 3] >> 16 == n).all()
    assert groups[-1, 0] + groups[-1, 1] == len(items)


def test_the_wave_split_at_gemma_4s_decode_shapes() -> None:
    """k = 5 (a verify row: 3 tiles of 16 rows per KV head, 12 groups) and the assistant's draft rows (one query: 4
    groups), every row real: one wave of the 376 resident CTAs unless it would be under 60% full (whole-KV groups at 16
    to 18 rows), then 5 × 376 / (3 × groups) splits rounded, about 5/3 of a wave (3 at 16-18 rows: 1.5-1.7 waves). A
    sweep of split counts on SM120 put these within 2 µs of the fastest at every shape it timed (7% at one draft row),
    within 3% from 8 rows on."""
    target, draft = am.tiles(6, G) * HKV, am.tiles(1, G) * HKV
    assert (target, draft) == (12, 4)
    got = {rows: am.choose_splits(rows * target, 1 << 20, SMS, RESIDENT)
           for rows in (1, 2, 4, 8, 9, 12, 15, 16, 17, 18, 19, 24, 32)}
    assert got == {1: 31, 2: 15, 4: 7, 8: 3, 9: 3, 12: 2, 15: 2, 16: 3, 17: 3, 18: 3, 19: 1, 24: 1, 32: 1}
    for rows, s in got.items():
        n = rows * target * s
        assert n <= RESIDENT or (16 <= rows <= 18 and n <= 9 * RESIDENT // 5) or s == 1, (rows, n)
    got = {rows: am.choose_splits(rows * draft, 1 << 20, SMS, RESIDENT) for rows in (1, 2, 4, 8, 16, 24, 32, 48)}
    assert got == {1: 32, 2: 32, 4: 23, 8: 11, 16: 5, 24: 3, 32: 2, 48: 3}
    # eager plans keep MIN_SPLIT_KEYS of the longest row per split
    assert am.choose_splits(12, 2700, SMS, RESIDENT) == 2700 // am.MIN_SPLIT_KEYS


def test_split_launches_fit_the_partials() -> None:
    """On any SM count and occupancy a split launch plans at most 9 × resident // 5 items (one wave, or 3 splits of
    fewer than 0.6 × resident groups): `Attn512`'s split partials hold that and 2 × MAX_SPLITS more."""
    for sms in (1, 2, 3, 46, 84, 132, 170, 188):
        for occupancy in (1, 2, 3, 4):
            r = occupancy * sms
            for groups in range(1, r + 1):
                s = am.choose_splits(groups, 1 << 20, sms, r)
                assert s == 1 or groups * s <= 9 * r // 5, (sms, occupancy, groups, s)


def test_graph_splits_follow_the_real_rows() -> None:
    """A decode graph's replay splits its real rows' groups by their own count, not the bucket's: its padding rows'
    items (one a group, on the null page's few keys) cost next to nothing. 3 real rows of bucket 4 split 10 ways
    (the bucket's 4 rows would split 7), 5 of bucket 8 split 6 ways; within the partials buffer (`ITEMS`)."""
    per_row = 12
    for real in range(1, 65):
        s = am.bucket_splits(real, per_row, SMS, RESIDENT, ITEMS)
        assert s == min(am.choose_splits(real * per_row, 1 << 20, SMS, RESIDENT), ITEMS // (real * per_row)) or s == 1
        assert real * per_row * s <= ITEMS or s == 1
    assert am.bucket_splits(3, per_row, SMS, RESIDENT, ITEMS) == 10
    assert am.bucket_splits(5, per_row, SMS, RESIDENT, ITEMS) == 6
    assert am.bucket_splits(9, per_row, SMS, RESIDENT, ITEMS) == 3
    assert am.bucket_splits(13, per_row, SMS, RESIDENT, ITEMS) == 2
    assert am.bucket_splits(16, per_row, SMS, RESIDENT, ITEMS) == 3
    assert am.bucket_splits(18, per_row, SMS, RESIDENT, ITEMS) == 3  # 648 items: 1.7 waves
    assert am.bucket_splits(32, per_row, SMS, RESIDENT, ITEMS) == 1


def test_graph_capacity_covers_every_real_count() -> None:
    """A bucket's captured launch has items for any real row count (`bucket_capacity`), and splits whenever some count
    does; buckets whose every count makes whole-KV groups run the kernel that writes its output directly."""
    for per_row in (4, 12, 16):
        for bucket in (1, 2, 4, 8, 16, 24, 32, 48, 64):
            cap, split = am.bucket_capacity(bucket, per_row, SMS, RESIDENT, ITEMS)
            for real in range(1, bucket + 1):
                s = am.bucket_splits(real, per_row, SMS, RESIDENT, ITEMS)
                assert real * per_row * s + (bucket - real) * per_row <= cap
                if per_row == 12:
                    items, groups = am.plan_items(np.full(bucket, 6), G, HKV, [s] * real + [1] * (bucket - real))
                    assert len(items) <= cap and len(groups) == bucket * per_row
                assert split or s == 1
            assert cap <= max(ITEMS + bucket * per_row, bucket * per_row)


def test_the_fixed_split_takes_each_rows_own_count() -> None:
    """The fixed split (answer reproducibility): a decode row's split count is its KV length's in `split_keys` keys,
    at least one; longer rows widen the split so a row keeps to the merge's lanes."""
    assert am.fixed_split_keys(4096 + 6) == am.SPLIT_KEYS == 256
    assert am.fixed_split_keys(9000) == 288  # ceil(9,000 / 32) = 282, rounded up to 16
    assert am.fixed_split_keys(131_072) == 4096
    assert am.fixed_splits(4096 + 6, am.fixed_split_keys(4096 + 6)) <= am.MAX_SPLITS
    np.testing.assert_array_equal(am.fixed_splits([1, 6, 256, 257, 2700, 4102], 256), [1, 1, 1, 2, 11, 17])
    # per-row counts make each row's groups (decode rows split by their own length, a prefill row whole)
    items, groups = am.plan_items(np.asarray([6, 6, 40]), G, HKV, am.fixed_splits([2700, 300, 1], 256))
    per = am.tiles(6, G) * HKV
    assert (groups[:per, 1] == 11).all() and (groups[per:2 * per, 1] == 2).all() and (groups[2 * per:, 1] == 1).all()
    assert len(items) == per * (11 + 2) + am.tiles(40, G) * HKV


def test_the_window_kernels_splits_cover_a_tile() -> None:
    """The sliding window on our kernel (answer reproducibility): a tile of 8 query positions sees at most window + 7
    keys, which meet at most ceil(that / split_keys) + 1 absolute splits (`kernels.attn_window`)."""
    from llm_mini_infer.kernels.attn_window import splits_per_tile

    assert splits_per_tile(1024, 8, 256) == 6 and splits_per_tile(1024, 8, 128) == 10
    assert splits_per_tile(64, 8, 16) == 6 and splits_per_tile(1024, 1, 1024) == 2
    for window, split in ((1024, 256), (64, 16), (1000, 48)):
        most = max(len(range(lo // split, (lo + window + 7 - 1) // split + 1)) for lo in range(0, 4 * split))
        assert most <= splits_per_tile(window, 8, split), (window, split, most)

"""Gemma's KV arena (runner/cuda/gemma.py `_Arena`) on the CPU: its page copy moves both overlaid views."""

from __future__ import annotations

import types

import torch

from llm_mini_infer.runner.cuda.gemma import _Arena


def arena_cfg() -> types.SimpleNamespace:
    """4 sliding layers and 2 full layers (groups = 2), a window page of 4 positions: the chunk is 128 elements and
    holds 8 positions of a full layer (sub = 8)."""
    return types.SimpleNamespace(sliding_layers=range(4), full_layers=range(2), sliding_kv_heads=2, sliding_head_dim=8,
                                 full_kv_heads=1, full_head_dim=8, k_eq_v=False)


def test_arena_copy_moves_both_views_of_a_page() -> None:
    """`_Arena.copy` copies whole pages of the one backing tensor. The full layers' sub-page view and the sliding
    layers' window-page view are both full views of it, so a copy through either moves every byte; what the test
    catches is a copy that misses part of a page (a slab, sub-pages indexed as pages, or one layer kind's region).
    One page is filled through both views with distinct values (the window view sets the sliding layers 0 and 1, the
    sub-page view the bytes of full layer 1, the slabs 2 and 3), copied onto another page; the destination reads the
    source's values through both views, and the source and every other page are as they were."""
    a = _Arena(arena_cfg(), 4, 6, torch.float32, torch.device("cpu"))
    assert (a.groups, a.sub, a.chunk) == (2, 8, 128)
    pages, src, dst = list(range(a.num_pages)), 1, 4
    for p in pages:  # every page holds something, so a stray write shows
        a.mem[:, p] = -float(p + 1)
    # The sub-page view as [full layer, group, page, ...]: sub-page j·num_pages + p is chunk p of slab j.
    fkv = a.fkv.reshape(2, a.groups, a.num_pages, *a.fkv.shape[2:])
    wval = torch.arange(a.wkv[:2, src].numel(), dtype=torch.float32).reshape(a.wkv[:2, src].shape) + 100.0
    fval = torch.arange(fkv[1, :, src].numel(), dtype=torch.float32).reshape(fkv[1, :, src].shape) + 1000.0
    a.wkv[:2, src] = wval  # window view: sliding layers 0 and 1, page src
    fkv[1, :, src] = fval  # sub-page view: full layer 1 (slabs 2 and 3), page src
    before = a.mem.clone()
    a.copy(torch.tensor([src]), torch.tensor([dst]))
    assert torch.equal(a.wkv[:2, dst], wval), "window view not copied"
    assert torch.equal(fkv[1, :, dst], fval), "sub-page view not copied"
    assert torch.equal(a.mem[:, dst], before[:, src]), "the page's bytes are not the source's"
    others = [p for p in pages if p != dst]
    assert torch.equal(a.mem[:, others], before[:, others]), "another page changed"

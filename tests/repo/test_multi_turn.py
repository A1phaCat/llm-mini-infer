"""tests/scripts/multi_turn.py on a small workload: a seed gives the same readout twice, every eviction order runs, and
the readout's sums agree with its parts."""

from __future__ import annotations

import pytest

from ..scripts import multi_turn as mt

WL = mt.Workload(sessions=4, turns=4, prefix=128, tool=96, reply=24, think=40.0, sigma=1.2, edits=0.2, retries=0.2)


@pytest.fixture(scope="module")
def mdir(tmp_path_factory):
    return mt.model_dir("hybrid", 0, tmp_path_factory.mktemp("multiturn"))


def test_a_seed_repeats_and_the_readout_adds_up(mdir) -> None:
    """Two runs of one seed give the same readout but for the wall time; the hit rates are their sums' ratios, every
    turn is counted once, and with the checkpoint budget below the conversations' needs the scheduler evicts and loses
    tokens to checkpoints."""
    eng = mt.Engine(max_num_seqs=2, slots=3, pages=2048)
    a, b = mt.run(WL, eng, mdir), mt.run(WL, eng, mdir)
    a.pop("wall_s"), b.pop("wall_s")
    assert a == b
    assert a["requests"] == WL.sessions * WL.turns
    assert a["hit_rate"] == round(a["cached"] / a["prompt"], 4)
    assert a["hit_rate_later"] == round(a["cached_later"] / a["prompt_later"], 4)
    assert a["edits"]["n"] + a["retries"]["n"] > 0
    assert sum(t["n"] for t in a["uncached_by_turn"].values()) == a["requests"]
    assert a["evicted_total"] > 0 and a["lost"]["checkpoint"] > 0


@pytest.mark.parametrize("order", mt.ORDERS)
def test_every_order_runs_the_same_workload(mdir, order: str) -> None:
    """The same turns under each eviction order (the workload does not depend on the timing): the prompts' tokens and
    the turns' kinds are equal, the hits are what differs: with the superseded checkpoints kept (`lru`) fewer tokens hit
    than with them demoted or the leaves kept last."""
    base = mt.run(WL, mt.Engine(max_num_seqs=2, slots=3, pages=2048, order="lru"), mdir)
    r = mt.run(WL, mt.Engine(max_num_seqs=2, slots=3, pages=2048, order=order), mdir)
    assert (r["prompt"], r["requests"], r["edits"]["n"], r["retries"]["n"]) == \
        (base["prompt"], base["requests"], base["edits"]["n"], base["retries"]["n"])
    if order != "lru":
        assert r["cached"] > base["cached"]


def test_leaf_last_takes_inner_checkpoints_then_idle_leaves() -> None:
    """The leaf-last comparator on a tree alone: checkpoints at 8, 16 and 32 on one path, the leaf at 32 the least
    recently used (the two above it hit since). LRU would take the leaf; leaf-last takes the inner ones first (a deeper
    checkpoint below, one child, no resume), least recently used first, and the leaf once it has idled past `idle` on
    the given clock: then every checkpoint is in the first tier, and recency alone orders them."""
    from types import SimpleNamespace

    import numpy as np

    from ..support.radix_model import Model

    mdl = Model(4, 64, 4, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [8, 16, 32])
    c, step, now = mdl.cache, mdl.step + 1, [0]
    ll = mt.LeafLast(SimpleNamespace(cache=c, step_id=step), idle=100, clock=lambda: now[0])
    ends = {n.end: n for n in mt.LeafLast._nodes(ll)}
    for limit in (8, 16):  # used now: the leaf at 32 is the least recently used
        assert c.match(None, a, step, limit).hit == limit
    assert [ll.tier(ends[e]) for e in (8, 16, 32)] == [0, 0, 1]
    assert c.evict_checkpoints(1, step) == 1 and not ends[8].ckpt and ends[16].ckpt and ends[32].ckpt
    now[0] = 101  # the leaf has idled past `idle`; the one left above it is inner
    assert ll.tier(ends[32]) == 0
    assert c.evict_checkpoints(1, step) == 1 and not ends[32].ckpt and ends[16].ckpt  # the older of the two
    assert ll.tier(ends[16]) == 0  # now a leaf (no checkpoint below), unused since 0
    mdl.check()

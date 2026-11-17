"""The parity bar against a reference engine (vLLM): ours not significantly below the lowest of its own greedy
agreements, by a paired bootstrap over prompts (`compare.bootstrap_min_gap`)."""

from __future__ import annotations

import numpy as np

from ..scripts import parity_compare as compare


def test_the_parity_bar_is_paired_against_the_lowest_agreement() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.random(n) < 0.55
    own = {"batched": base | (rng.random(n) < 0.1), "eager": base.copy()}
    within = base.copy()
    within[:4] = ~within[:4]  # a few prompts flip either way: within noise of the lowest (eager)
    r = compare.bootstrap_min_gap(within, own)
    assert r["lowest"] == "eager" and r["ci95"][0] < 0 <= r["ci95"][1]
    assert abs(r["gap"] - (within.mean() - base.mean())) < 1e-12
    far = base & (rng.random(n) < 0.6)  # loses ~40% of the identical prompts: significantly below
    assert compare.bootstrap_min_gap(far, own)["ci95"][1] < 0
    assert set(r["per_variant"]) == {"batched", "eager"}

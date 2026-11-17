"""The comparison against a reference engine's own variation: greedy outputs per prompt, ours against the reference's,
judged by the lowest agreement among the reference's own runs (batched, eager, other variants), since batch
composition and implementation choices move every engine's outputs."""

from __future__ import annotations

import numpy as np


def bootstrap_min_gap(ours: np.ndarray, own: dict[str, np.ndarray], n: int = 4000, seed: int = 0) -> dict:
    """Per prompt, 1 where our greedy output is identical to the reference's (`ours`), and where each of the
    reference's own runs is (`own`, by run): the mean of `ours` minus the lowest of `own`'s means, with a paired 95%
    interval from resampling prompts (the minimum taken in each resample), and the paired gap to each run. Ours is not
    significantly below the reference's own variation when the interval reaches 0."""
    rng = np.random.default_rng(seed)
    names = list(own)
    mat = np.stack([own[k].astype(np.float64) for k in names])  # [variants, prompts]
    o = ours.astype(np.float64)
    gap = float(o.mean() - mat.mean(1).min())
    idx = rng.integers(0, len(o), size=(n, len(o)))
    boot = o[idx].mean(1) - mat[:, idx].mean(2).min(0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per = {}
    for i, k in enumerate(names):
        d = o - mat[i]
        b = d[idx].mean(1)
        per[k] = {"gap": float(d.mean()), "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])]}
    return {"gap": gap, "ci95": [float(lo), float(hi)], "lowest": names[int(mat.mean(1).argmin())], "per_variant": per}

"""Grammar masks as the sampler takes them (tests/runner/test_logprobs.py, tests/runner/test_logprobs_gpu.py). Apart
from the logprobs runs: the logprobs fuzz suite imports this alone, and what a suite imports is what Hypothesis mines
for the constants it mixes into its draws."""

from __future__ import annotations

import numpy as np
import torch


def words_of(allowed: np.ndarray) -> torch.Tensor:
    """bool [m, V] -> int32 mask words (token t: bit t % 32 of word t // 32)."""
    return torch.from_numpy(np.packbits(allowed, axis=1, bitorder="little").view(np.int32).copy())

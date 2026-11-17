"""The least of several timings, for tests that bound a cost."""

from __future__ import annotations

import gc
import math
import time


def least_each(*fs, runs: int = 5) -> tuple[float, ...]:
    """The least of `runs` timings of each `f()`, the garbage collector held off: a full collection of a test process's
    heap can take longer than the runs timed. The runs interleave (f0, f1, f0, f1, ...), so a stretch of contention, or
    a move to a slower core, slows the functions alike and leaves their ratio alone."""
    gc.collect()
    gc.disable()
    try:
        best = [math.inf] * len(fs)
        for _ in range(runs):
            for i, f in enumerate(fs):
                t = time.perf_counter()
                f()
                best[i] = min(best[i], time.perf_counter() - t)
        return tuple(best)
    finally:
        gc.enable()


def least(f, runs: int = 3) -> float:
    """The least of `runs` timings of `f()`, the garbage collector held off (see `least_each`)."""
    return least_each(f, runs=runs)[0]


def under(f, bound: float, runs: int = 3) -> float:
    """The least timing of `f()`, run again while it is over `bound`, up to `runs` times. The bound keeps its meaning: a
    busy host slows one run, while a cost past the bound slows every run. `f` must do its whole work each run, with no
    cache from the run before."""
    best = math.inf
    for _ in range(runs):
        best = min(best, least(f, 1))
        if best < bound:
            break
    return best

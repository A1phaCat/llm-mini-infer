"""Garbage-collector pauses (`gc.callbacks`), per generation, in the engine and in each frontend:
llm_mini_infer:gc_seconds_total and llm_mini_infer:gc_collections_total{process,generation} on /metrics (the engine's
travel in `EngineStats`).

The callback runs inside the collector, so it only adds integers; the warning for a long pause is logged later, from
the process's own loop (`GcStats.maybe_warn`), at most once a `WARN_EVERY_S`.

**The warning's threshold, `SLOW_MS`.** The engine stays one step ahead of the device: a pause shorter than the step in
flight hides behind it. A 27-31B model's decode steps take 20 ms and more on an SM120 GPU (measured on an RTX PRO 6000:
about 24 ms for a k = 5 verify step at one row), and a smaller model's take less. A pause of 10 ms or more can idle the
device on shorter steps, and in a frontend it is five times the time to first token's host budget (2 ms): it shows in
the tail latency. Shorter pauses show in the counters (seconds over collections).
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

GENERATIONS = 3
SLOW_MS = 10
WARN_EVERY_S = 60.0


class GcStats:
    """One process's collections and their time, per generation (cumulative from `install`)."""

    def __init__(self, process: str):
        self.process = process  # the `process` label: engine or frontend
        self.ns = [0] * GENERATIONS
        self.collections = [0] * GENERATIONS
        self.slow = 0  # pauses of SLOW_MS or more since the last warning
        self.slow_max_ns = 0
        self._slow_ns = SLOW_MS * 1_000_000
        self._t0 = 0
        self._warned = 0  # time.monotonic_ns() of the last warning
        self._since = 0  # ... of the last warning, or of install

    def _callback(self, phase: str, info: dict[str, Any]) -> None:
        """Inside the collector: integers only (no logging, no allocation the collector could see)."""
        if phase == "start":
            self._t0 = time.perf_counter_ns()
            return
        dt = time.perf_counter_ns() - self._t0
        g = info["generation"]
        self.ns[g] += dt
        self.collections[g] += 1
        if dt >= self._slow_ns:
            self.slow += 1
            if dt > self.slow_max_ns:
                self.slow_max_ns = dt

    def install(self) -> GcStats:
        if self._callback not in gc.callbacks:
            gc.callbacks.append(self._callback)
        self._since = time.monotonic_ns()
        return self

    def uninstall(self) -> None:
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)

    def maybe_warn(self) -> None:
        """Log the long pauses since the last warning, at most once a WARN_EVERY_S (called from the process's loop)."""
        if not self.slow:
            return
        now = time.monotonic_ns()
        if self._warned and now - self._warned < WARN_EVERY_S * 1e9:
            return
        log.warning("%s: %d garbage collection%s of %d ms or more in the last %.0f s, the longest %.1f ms "
                    "(llm_mini_infer:gc_seconds_total)", self.process, self.slow, "" if self.slow == 1 else "s",
                    SLOW_MS, (now - self._since) / 1e9, self.slow_max_ns / 1e6)
        self.slow = self.slow_max_ns = 0
        self._warned = self._since = now

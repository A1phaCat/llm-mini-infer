# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/compilation/breakable_cudagraph.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm), and
# changed for llm-mini-infer.
"""CUDA-graph capture with eager breaks (docs/architecture.md, "CUDA graphs"), after vLLM's `breakable_cudagraph.py` and
SGLang's breakable CUDA-graph backend, without torch.compile.

One capture context drives a runner's step body (or one part of a split step), or a vision-tower call. Where it calls an
op that cannot live in a graph (attention, and Qwen's GDN mixer, on prefill and mixed steps; the tower's attention),
`eager()` ends the current graph segment, runs the op eagerly on the capture stream, records it, and begins the next
segment. The captured artifact is the list of segments; `replay()` runs them in order on the current stream. Eager
segments must work on the graph's static buffers (inputs produced by the previous segment, outputs allocated inside a
segment before the break), so the next segment reads the same addresses. The forward contexts
(`models.qwen3_5.backend.ForwardContext`, `models.gemma4.backend.Gemma4Context`) and the tower's `runner.cuda.images`
arrange that, and keep only weak references, so later captures into the shared pool can reuse the memory. A full
graph (decode steps) is the same object with no breaks.

A runner's graphs are named per bucket kind ("decode": a full graph per batch-size bucket; "break": a breakable one per
token bucket): `graph_names`, `graph_name`, `parse_graph_name` and `graph_summary` build and read them.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from enum import IntEnum
from typing import Any

import torch

from .heads import NO_HEADS, VARIANT, VARIANT_HEADS, Heads

log = logging.getLogger(__name__)


@contextmanager
def no_gc() -> Iterator[None]:
    """No garbage collection while capturing: a collection could release another object's CUDA resources (pinned
    memory, events, a module) in the middle of a capture, which invalidates it. Collect first, then keep the collector
    off until the captures end."""
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def warm_libraries(device: torch.device) -> None:
    """Create the cuBLAS handle (and its workspace on the current stream) eagerly, before any capture: cuBLAS cannot
    create a handle inside one (`CUBLAS_STATUS_NOT_INITIALIZED` from `cublasCreate`, which invalidates it). A BF16 model
    runs a cuBLAS GEMM in its first eager step; a block-FP8 one may not (Gemma 4's projections are CUTLASS, its head at
    one row our small-M kernel), and its first cuBLAS call would then be the head inside the first capture."""
    torch.cuda.current_blas_handle()
    x = torch.zeros((16, 64), dtype=torch.bfloat16, device=device)
    torch.nn.functional.linear(x, torch.zeros((64, 64), dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize(device)


def graph_names(kind: str, *, grammar: bool, always_split: bool, variants: Sequence[str]) -> tuple[str, ...]:
    """A bucket's graphs in capture order: the fused graph; with grammar masks also the forward and a sampling graph per
    heads variant (`variants`, in their order); always split, those alone (every step runs as a forward and a
    sampling)."""
    if not grammar:
        return (kind,)
    split = (f"{kind}_f", *(f"{kind}_{v}" for v in variants))
    return split if always_split else (kind, *split)


def graph_name(kind: str, part: str, heads: Heads = NO_HEADS) -> str:
    """The graph a step of `kind` replays for `part` of its body ("all": the fused graph; "fwd": the forward; "smp":
    the sampling on `heads`); `parse_graph_name`'s inverse."""
    return kind if part == "all" else f"{kind}_{'f' if part == 'fwd' else VARIANT[heads]}"


def parse_graph_name(name: str) -> tuple[str, str, Heads]:
    """(kind, part, heads) of a graph's name: "decode" (decode, all), "break_f" (break, fwd), "decode_q" (decode, smp,
    the sparse head); NO_HEADS but for a sampling graph."""
    kind, _, p = name.partition("_")
    part = {"": "all", "f": "fwd"}.get(p, "smp")
    return kind, part, VARIANT_HEADS[p] if part == "smp" else NO_HEADS


def graph_summary(keys: Collection[tuple[str, int]], *, variants: list[str], vision: list[int]) -> dict[str, Any]:
    """`stats["graphs"]` of the captured graphs' (name, size) keys: the decode and token buckets that have graphs
    (fused or split), the split steps' forward graphs, the fused graphs, every graph, the sampling variants (`variants`:
    none without grammar masks) and the vision tower's buckets (`vision`)."""
    return {"decode": sorted({n for k, n in keys if k in ("decode", "decode_f")}),
            "break": sorted({n for k, n in keys if k in ("break", "break_f")}),
            "split": sum(1 for k, _ in keys if k.endswith("_f")),
            "fused": sum(1 for k, _ in keys if k in ("decode", "break")),
            "captured": len(keys),
            "sampling_variants": variants,
            "vision": vision}


class Capture(IntEnum):
    """A `BreakableCapture`'s state."""

    NEW = 0
    RECORDING = 1  # a graph segment is being captured
    BREAK = 2  # `eager` runs its op between two segments
    CAPTURED = 3  # the `with` ended: it replays
    FAILED = 4  # an exception inside: it never replays


_CAPTURED = Capture.CAPTURED  # replay's one compare


class BreakableCapture:
    """One capture (module docstring), NEW until its `with` begins RECORDING a segment; `eager` ends the segment, is
    in BREAK while its op runs, and begins the next; the end of the `with` makes it CAPTURED, any exception inside
    FAILED. Only a CAPTURED one replays: its callers keep it only once the `with` has ended."""

    def __init__(self, pool: Any):
        self.pool = pool
        self.segments: list[Any] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        self.state = Capture.NEW
        self.graphs = 0
        self.breaks = 0

    def __enter__(self) -> BreakableCapture:
        if self.state is not Capture.NEW:
            raise RuntimeError(f"a capture entered when {self.state.name}, not NEW")
        self._begin()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        if exc_type is None:
            self._end()
            self.state = Capture.CAPTURED
            return
        if self.state is Capture.RECORDING:  # leave the stream's capture mode; the error inside is the one raised
            g, self._graph = self._graph, None
            try:
                g.capture_end()
            except Exception as end:
                log.warning("ending the capture after %s also failed: %s", exc_type.__name__, end)
        self.state = Capture.FAILED

    def _begin(self) -> None:
        g = torch.cuda.CUDAGraph()
        g.capture_begin(pool=self.pool)
        self._graph = g
        self.state = Capture.RECORDING

    def _end(self) -> None:
        """Close the segment being recorded: FAILED until it is closed (a `capture_end` that raises is not called
        again), then the caller sets the next state."""
        if self.state is not Capture.RECORDING:
            raise RuntimeError(f"a capture segment ended when {self.state.name}, not RECORDING")
        g, self._graph = self._graph, None
        self.state = Capture.FAILED
        g.capture_end()
        self.segments.append(g)
        self.graphs += 1

    def eager(self, fn: Callable[[], Any]) -> None:
        """End the segment, run `fn` now (on the capture stream) and at every replay, then begin the next segment."""
        self._end()
        self.state = Capture.BREAK
        try:
            fn()
        except BaseException:
            self.state = Capture.FAILED
            raise
        self.segments.append(fn)
        self.breaks += 1
        self._begin()

    def replay(self) -> None:
        if self.state is not _CAPTURED:
            raise RuntimeError(f"a replay of a capture that is {self.state.name}, not CAPTURED")
        for s in self.segments:
            if isinstance(s, torch.cuda.CUDAGraph):
                s.replay()
            else:
                s()

    def __repr__(self) -> str:
        return f"BreakableCapture(graphs={self.graphs}, breaks={self.breaks})"

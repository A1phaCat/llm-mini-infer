"""Block-FP8 checkpoints in the CUDA runners (docs/architecture.md, "FP8"): one implementation for the Qwen runner and
Gemma 4's.

- **The GEMMs** (`Fp8Runtime.__init__`, before the model loads): `dev.runner.cuda.fp8_gemm` picks the backend
  (`kernels.fp8.use_backend`; auto: the measured table), and the CUTLASS library must load when the table uses it, or
  startup fails with the reason.
- **PRECISE rows** (`runner.cuda.fp8_cold_prefill`; architecture.md, "The precise prefix"): a step holding any runs
  eagerly, those rows' FP8 projections taking BF16 activations against their weights dequantized to BF16 into one
  buffer reserved before the memory profile (`attach`), the step's other rows on the FP8 path bit for bit
  (`kernels.fp8.precise_rows`). The step costs the scheduler derives its budgets from are timed on the FP8 path
  (`timing`), plus one full-budget step of PRECISE rows (`Capabilities.precise_step_ms`).

A runner asks `precise_step(n)` when it picks a step's mode (a step with n > 0 PRECISE tokens runs eagerly, whatever
the graphs) and runs the eager body inside `activations(t, n)`, its PRECISE rows ordered last.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
from torch import nn

from ...kernels import build as kbuild
from ...kernels import fp8
from ...models.fp8 import Fp8Linear

log = logging.getLogger(__name__)

# runner.cuda.fp8_cold_prefill's values (config.py)
MODES = ("reuse", "bf16", "fp8")


class Fp8Runtime:
    """A runner's block-FP8 setup. Construct it before loading the model (`quantized`: the checkpoint has a
    quantization config), then `attach` the loaded model. Without FP8 projections it does nothing."""

    def __init__(self, quantized: bool, *, cuda: bool, kernels: bool, dtype: torch.dtype, fp8_gemm: str = "auto",
                 fp8_cold_prefill: str = "reuse", stats: dict[str, Any] | None = None):
        if fp8_cold_prefill not in MODES:
            raise ValueError(f"runner.cuda.fp8_cold_prefill {fp8_cold_prefill!r}: one of {', '.join(MODES)}")
        self.quantized, self.cuda, self.mode = quantized, cuda, fp8_cold_prefill
        self.stats = stats if stats is not None else {}
        self.n = 0
        self.precise = False  # PRECISE rows: FP8 projections, and the precise prefix on
        self._timing = False
        if not quantized:
            return
        if dtype != torch.bfloat16 and cuda:
            raise ValueError("block-FP8 checkpoints run with BF16 activations (dev.runner.cuda.dtype)")
        fp8.use_backend(fp8_gemm)
        if kernels and fp8_gemm in ("auto", "cutlass", "sglang"):
            kbuild.require_cutlass("dev.runner.cuda.fp8_gemm")
            self.stats["cutlass_load_s"] = kbuild.stats_cutlass["load_s"]

    def attach(self, model: nn.Module, device: torch.device) -> int:
        """After loading: count the FP8 projections, and reserve the BF16 rows' weight buffer (for the largest
        projection) before the memory profile. Returns the count."""
        lins = [m for m in model.modules() if isinstance(m, Fp8Linear)]
        self.n = len(lins)
        self.precise = bool(lins) and self.mode != "fp8"
        if self.precise and self.cuda:
            fp8.reserve_bf16_weight(max(m.weight.numel() for m in lins), device)
        if lins:
            self.stats.update(fp8_projections=self.n, fp8_gemm=fp8.describe()["table"], fp8_cold_prefill=self.mode)
        return self.n

    def precise_step(self, n_precise: int) -> bool:
        """Whether a step with `n_precise` tokens of PRECISE rows runs them with BF16 activations (eagerly): it holds
        some and the runner computes them (except while the profile times the FP8 path)."""
        return self.precise and n_precise > 0 and not self._timing

    def activations(self, t: int, n_precise: int) -> AbstractContextManager:
        """The context a step's eager body of `t` tokens runs in: its last `n_precise` tokens (its PRECISE rows) with
        BF16 activations (`precise_step`), else nothing."""
        return fp8.precise_rows(t, t - n_precise) if self.precise_step(n_precise) else nullcontext()

    @contextmanager
    def timing(self) -> Iterator[None]:
        """The memory profile's timed steps (the costs the scheduler derives its budgets from): on the FP8 path."""
        self._timing = True
        try:
            yield
        finally:
            self._timing = False

    def log_setup(self) -> None:
        if self.precise:
            log.info("FP8 activations: PRECISE rows (the precise prefix, fp8_cold_prefill %s) run with BF16 "
                     "activations against the weights dequantized to BF16 (eager steps)", self.mode)
        if self.n:
            log.info("FP8 GEMMs by M: %s", ", ".join(
                f"{f'M <= {m}' if m < fp8.BIG else 'larger M'}{f' N <= {n}' if n < fp8.BIG else ''}: "
                f"{b}{f' {c}' if b == 'cutlass' else ''}" for m, n, b, c in fp8.table()))

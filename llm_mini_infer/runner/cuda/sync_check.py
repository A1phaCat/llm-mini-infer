"""The CUDA sync check (`dev.runner.cuda.sync_check`: off, warn or error; docs/architecture.md, "The step loop": the
step path never waits for the device). Armed, a runner's `launch_forward` and `launch_sample` run under
`torch.cuda.set_sync_debug_mode`, saved, set and restored around each call, exceptions included: a synchronizing call
inside them warns or raises at its line. The GPU suite runs it at error (tests/conftest.py); a deployment leaves it
off, or at warn, since an exception in a launch ends the engine loop. One torch thread drives the GPU, so the mode,
which is process-wide, polices that thread alone.

**Never armed** (`make`): the runner's warm-up and capture calls (`_internal`), `dev.runner.cuda.check` and the torch
reference configurations (a CPU device, `fused_ops` off, torch attention or GDN, the torch FP8 GEMM), all of which sync
on purpose; nor inside `allowed()` (a test's logits hook).

**What torch's mode misses**, reported here while armed (`blocked`, `host_wait`, the copy check):
- a host wait on a CUDA event (`Event.synchronize`): the step path's on purpose, reported only when the event has not
  completed, i.e. when the host does wait: the draft universe's ids copied again (`draft_head.SubsetHead.install`), a
  staging buffer reused before its step's `wait()`, the vision tower's plan buffers all in flight
  (`kernels.flashinfer.FlashInferVisionAttention`);
- a `non_blocking` copy to the device from pageable memory, which CUDA performs synchronously: `Tensor.to`, `.cuda` and
  `.copy_` are checked while armed (installed once, by the first armed runner).
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

MODES = ("off", "warn", "error")
# The runners' default when they are not given one: off; the tests' suite sets it (tests/conftest.py: error)
DEFAULT = "off"

_armed: str | None = None  # the mode while a launch runs under the check


class SyncCheck:
    """One runner's check (`make`): `launch()` arms it around a step's launch."""

    def __init__(self, mode: str):
        import torch

        if mode not in ("warn", "error"):
            raise ValueError(f"dev.runner.cuda.sync_check: {mode!r} (off, warn or error)")
        self.mode = mode
        self._torch = torch
        _install_copy_check()

    @contextlib.contextmanager
    def launch(self) -> Iterator[None]:
        global _armed
        cuda = self._torch.cuda
        prev, outer = cuda.get_sync_debug_mode(), _armed
        cuda.set_sync_debug_mode(self.mode)
        _armed = self.mode
        try:
            yield
        finally:
            _armed = outer
            cuda.set_sync_debug_mode(prev)


def make(mode: str | None, *, cuda: bool, reference: str | None, check: bool) -> SyncCheck | None:
    """The runner's check, or None when off. `mode` None: `DEFAULT`. `reference`: why the runner runs a torch reference
    (None: our kernels); `check`: `dev.runner.cuda.check` (syncs every step)."""
    mode = DEFAULT if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"dev.runner.cuda.sync_check: {mode!r} (off, warn or error)")
    if mode == "off":
        return None
    why = "no CUDA device" if not cuda else reference if reference else "dev.runner.cuda.check syncs every step" \
        if check else None
    if why:
        log.info("dev.runner.cuda.sync_check %s: not armed (%s)", mode, why)
        return None
    log.info("dev.runner.cuda.sync_check %s: launch_forward and launch_sample %s on a sync", mode,
             "raise" if mode == "error" else "warn")
    return SyncCheck(mode)


def armed() -> str | None:
    return _armed


@contextlib.contextmanager
def allowed() -> Iterator[None]:
    """Inside a launch, a part that syncs on purpose (a test's logits hook): the check disarmed there."""
    global _armed
    if _armed is None:
        yield
        return
    import torch

    outer, prev = _armed, torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(0)
    _armed = None
    try:
        yield
    finally:
        _armed = outer
        torch.cuda.set_sync_debug_mode(prev)


def blocked(what: str) -> None:
    """A wait for the device on the step path that torch's mode does not see: reported while armed."""
    if _armed is not None:
        _report(what)


def host_wait(event: Any, what: str) -> None:
    """`event.synchronize()`, reported while armed when the event has not completed (the host then waits)."""
    if _armed is not None and not event.query():
        _report(what)
    event.synchronize()


def _report(what: str) -> None:
    msg = f"CUDA sync check: {what}"
    if _armed == "error":
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=3)


# -- non_blocking copies from pageable memory ----------------------------------------------------------------------

_originals: dict[str, Any] = {}


def pageable(src: torch.Tensor, device: Any, non_blocking: bool) -> bool:
    """Whether copying `src` to `device` is a non_blocking copy from pageable host memory (CUDA does it synchronously).
    An empty tensor copies nothing."""
    import torch

    return (bool(non_blocking) and device is not None and torch.device(device).type == "cuda" and
            src.device.type == "cpu" and src.numel() > 0 and not src.is_pinned())


def _to_args(args: tuple, kw: dict[str, Any]) -> tuple[Any, bool]:
    """`Tensor.to`'s target device and non_blocking, by torch's own parser, which takes every form but `copy` (a
    keyword, or the last positional): parsed without it."""
    import torch

    kw = {k: v for k, v in kw.items() if k != "copy"}
    for a in (args, args[:-1]):
        try:
            device, _, non_blocking, _ = torch._C._nn._parse_to(*a, **kw)
            return device, non_blocking
        except (RuntimeError, TypeError):
            continue
    return None, False


def _check_to(self: torch.Tensor, *args: Any, **kw: Any) -> torch.Tensor:
    if _armed is not None and pageable(self, *_to_args(args, kw)):
        _report(f"Tensor.to: a non_blocking copy of {tuple(self.shape)} from pageable memory")
    return _originals["to"](self, *args, **kw)


def _check_cuda(self: torch.Tensor, *args: Any, **kw: Any) -> torch.Tensor:
    """`Tensor.cuda(device=None, non_blocking=False, memory_format=...)`, any of them positional."""
    non_blocking = args[1] if len(args) > 1 else kw.get("non_blocking", False)
    if _armed is not None and pageable(self, "cuda", non_blocking):
        _report(f"Tensor.cuda: a non_blocking copy of {tuple(self.shape)} from pageable memory")
    return _originals["cuda"](self, *args, **kw)


def _check_copy(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    if _armed is not None and pageable(src, self.device, non_blocking):
        _report(f"Tensor.copy_: a non_blocking copy of {tuple(src.shape)} from pageable memory")
    return _originals["copy_"](self, src, non_blocking)


def _install_copy_check() -> None:
    """Wrap `Tensor.to`, `.cuda` and `.copy_` once; each checks only while a launch is armed."""
    import torch

    if _originals:
        return
    for name, check in (("to", _check_to), ("cuda", _check_cuda), ("copy_", _check_copy)):
        _originals[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, check)

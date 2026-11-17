"""The CUDA sync check (`runner.cuda.sync_check`, `dev.runner.cuda.sync_check`) on a CPU-only host: torch's sync debug
mode is stood in for (its get and set), so the tests see what the runner arms, when and what it restores;
the GPU suite runs the real check at error (tests/conftest.py).

- **Where it arms:** engine turns on the CPU Gemma runner with the check forced on: the mode is the check's inside every
  `launch_forward` and `launch_sample`, the saved one outside, after an exception too; the warm-up's and captures'
  internal calls never arm it.
- **When it is off:** off by default; not armed without CUDA, with `dev.runner.cuda.check` or on the torch references.
- **What torch's mode misses:** host waits on events (`host_wait`, `blocked`) and `non_blocking` copies from pageable
  memory (the copy check), reported while armed: a warning at warn, an exception at error.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.config import Config
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.runner.cuda import sync_check
from llm_mini_infer.runner.cuda.gemma import GemmaRunner
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.tiny_gemma import gemma_pools, make_tiny_gemma

MODES = {"default": 0, "warn": 1, "error": 2}


@pytest.fixture
def modes(monkeypatch) -> list:
    """torch.cuda's sync debug mode, stood in for: every set recorded; get returns the last one set (0 at first)."""
    log: list = []
    monkeypatch.setattr(torch.cuda, "get_sync_debug_mode", lambda: log[-1] if log else 0)
    monkeypatch.setattr(torch.cuda, "set_sync_debug_mode", lambda m: log.append(MODES.get(m, m)))
    return log


def torch_tensor_methods(monkeypatch) -> None:
    """Tensor.to, .cuda and .copy_ as torch has them until the test ends, and no copy check installed: an armed runner
    of an earlier test (the GPU suite's) installs it once a process, leaving its wrappers on torch.Tensor and torch's
    methods in `sync_check._originals`."""
    for name in ("to", "cuda", "copy_"):
        monkeypatch.setattr(torch.Tensor, name, sync_check._originals.get(name, getattr(torch.Tensor, name)))
    monkeypatch.setattr(sync_check, "_originals", {})


@pytest.fixture(autouse=True)
def _tensor_methods(monkeypatch) -> None:
    torch_tensor_methods(monkeypatch)


@pytest.fixture
def copy_check() -> dict:
    """The copy check installed over stand-ins for torch's Tensor.to, .cuda and .copy_ (no CUDA here), each recording
    its call."""
    calls: dict = {}
    sync_check._install_copy_check()
    for name in ("to", "cuda", "copy_"):
        sync_check._originals[name] = lambda *a, _n=name, **kw: calls.setdefault(_n, []).append((a, kw))
    return calls


def test_a_copy_check_an_earlier_test_installed_is_set_aside(monkeypatch) -> None:
    """The GPU suite's order: an armed runner installed the copy check before this module ran. The tests then run on
    torch's own methods, and a copy outside a launch still works."""
    real = torch.Tensor.copy_
    sync_check._install_copy_check()
    assert torch.Tensor.copy_ is not real and sync_check._originals["copy_"] is real
    torch_tensor_methods(monkeypatch)
    assert torch.Tensor.copy_ is real and sync_check._originals == {}
    torch.nn.Embedding(4, 4)  # its initialisation copies into its weight


def test_the_check_is_off_by_default_and_never_armed_where_syncs_are_meant(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger=sync_check.__name__)
    monkeypatch.setattr(sync_check, "DEFAULT", "off")  # the engine's (the tests' suite sets its own: conftest.py)
    assert Config().dev.runner.cuda.sync_check == "off"
    assert sync_check.make(None, cuda=True, reference=None, check=False) is None
    assert sync_check.make("off", cuda=True, reference=None, check=False) is None
    for kw in (dict(cuda=False, reference=None, check=False), dict(cuda=True, reference="torch attention", check=False),
               dict(cuda=True, reference=None, check=True)):
        assert sync_check.make("error", **kw) is None, kw
    assert "not armed (torch attention)" in caplog.text and "dev.runner.cuda.check syncs every step" in caplog.text
    with pytest.raises(ValueError, match="sync_check: 'loud'"):
        sync_check.make("loud", cuda=True, reference=None, check=False)


def test_a_launch_sets_the_mode_and_restores_the_saved_one(modes) -> None:
    modes.append(1)  # someone's warn, set before
    check = sync_check.SyncCheck("error")
    with check.launch():
        assert torch.cuda.get_sync_debug_mode() == 2 and sync_check.armed() == "error"
    assert modes[-1] == 1 and sync_check.armed() is None
    with pytest.raises(KeyError), check.launch():
        raise KeyError("a launch that fails")
    assert modes[-1] == 1 and sync_check.armed() is None
    with check.launch():
        with sync_check.allowed():  # a part that syncs on purpose: disarmed inside
            assert torch.cuda.get_sync_debug_mode() == 0 and sync_check.armed() is None
            sync_check.blocked("a wait the test makes")
        assert torch.cuda.get_sync_debug_mode() == 2 and sync_check.armed() == "error"
    with sync_check.allowed():  # not armed: nothing changes
        assert modes[-1] == 1


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma(tmp_path_factory.mktemp("tiny-gemma"))


def test_engine_turns_arm_every_launch_and_never_the_internal_calls(tiny, modes, monkeypatch) -> None:
    """The CPU Gemma runner (no CUDA: the check forced on, as a CUDA runner of our kernels arms it): its configure's
    warm-up runs internal launches, which arm nothing; then three requests through the engine loop, every
    launch_forward and launch_sample inside the check's mode and the saved one between them. A launch that raises
    restores the mode."""
    pc = gemma_pools(max_num_seqs=2, budget=64, page_size=4, num_pages=256, max_model_len=256)
    r = GemmaRunner(tiny, device="cpu", dtype=torch.float32, max_num_seqs=2, max_num_batched_tokens=64,
                    max_model_len=256, sync_check="error")
    assert r._sync is None  # no CUDA: not armed
    r._sync = sync_check.SyncCheck("error")
    r.configure(pc)
    assert modes == [], "the warm-up's internal launches armed the check"
    seen: list[tuple[str, int]] = []
    fwd, smp = r._launch_forward, r._launch_sample

    def _launch_forward(*a, **kw):
        seen.append(("forward", torch.cuda.get_sync_debug_mode()))
        return fwd(*a, **kw)

    def _launch_sample(*a, **kw):
        seen.append(("sample", torch.cuda.get_sync_debug_mode()))
        return smp(*a, **kw)

    monkeypatch.setattr(r, "_launch_forward", _launch_forward)
    monkeypatch.setattr(r, "_launch_sample", _launch_sample)
    sched = Scheduler(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=256, page_size=pc.page_size,
                      num_pages=pc.num_pages, num_state_slots=3, eos_token_ids=(511,), hybrid=False,
                      checkpoint_grid=4, window=8, window_page_size=4)
    core = EngineCore(sched, r)
    for rid in range(3):
        core.handle(AddRequest(rid=rid, frontend=0, prompt=np.arange(5 + 3 * rid, 40 + 3 * rid, dtype=np.int32),
                               max_tokens=6, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=rid,
                               stop_token_ids=[]))
    turns = 0
    while core.busy and turns < 200:
        core.step()
        turns += 1
        assert torch.cuda.get_sync_debug_mode() == 0  # between launches: the saved mode
    assert not core.busy and seen and {m for _, m in seen} == {2}, seen
    assert {k for k, _ in seen} == {"forward", "sample"} and modes.count(2) == len(seen)

    def broken(*a, **kw):
        raise RuntimeError("a launch that fails")

    monkeypatch.setattr(r, "_launch_forward", broken)
    with pytest.raises(RuntimeError, match="a launch that fails"):
        r.launch_forward(SimpleNamespace())
    assert torch.cuda.get_sync_debug_mode() == 0 and sync_check.armed() is None


def test_the_setting_takes_off_warn_or_error() -> None:
    import msgspec

    from llm_mini_infer.config import DevCudaConfig

    assert msgspec.convert({"sync_check": "warn"}, DevCudaConfig).sync_check == "warn"
    with pytest.raises(msgspec.ValidationError, match="sync_check"):
        msgspec.convert({"sync_check": "loud"}, DevCudaConfig)


class Event:
    def __init__(self, done: bool):
        self.done, self.waited = done, 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.waited += 1


def test_host_waits_are_reported_while_armed(modes) -> None:
    done, pending = Event(True), Event(False)
    sync_check.host_wait(pending, "a wait outside a launch")  # not armed: nothing reported
    assert pending.waited == 1
    with sync_check.SyncCheck("error").launch():
        sync_check.host_wait(done, "a completed event")  # no wait: nothing reported
        with pytest.raises(RuntimeError, match="CUDA sync check: the staging buffer"):
            sync_check.host_wait(pending, "the staging buffer")
        with pytest.raises(RuntimeError, match="CUDA sync check: a plan buffer"):
            sync_check.blocked("a plan buffer")
    with sync_check.SyncCheck("warn").launch(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sync_check.host_wait(pending, "the staging buffer")
        sync_check.blocked("a plan buffer")
    assert [str(x.message) for x in w] == ["CUDA sync check: the staging buffer", "CUDA sync check: a plan buffer"]
    assert done.waited == 1 and pending.waited == 2  # at error, the report raises before the wait


def test_non_blocking_copies_from_pageable_memory(modes, copy_check) -> None:
    """While armed, a non_blocking copy to the device from pageable memory is reported (Tensor.to, .cuda, .copy_); a
    blocking one, an empty tensor or a copy outside a launch is not, and every call reaches torch's method."""
    t, empty = torch.zeros(4), torch.zeros(0)
    assert not t.is_pinned() and sync_check.pageable(t, "cuda", True) and sync_check.pageable(t, "cuda:0", True)
    assert not sync_check.pageable(t, "cuda", False) and not sync_check.pageable(empty, "cuda", True)
    assert not sync_check.pageable(t, "cpu", True) and not sync_check.pageable(t, None, True)
    t.to("cuda", non_blocking=True)  # not armed
    with sync_check.SyncCheck("error").launch():
        t.to("cuda")
        t.to(torch.device("cuda"), torch.float16)
        empty.to("cuda", non_blocking=True)
        with pytest.raises(RuntimeError, match=r"Tensor\.to: a non_blocking copy of \(4,\) from pageable memory"):
            t.to("cuda", non_blocking=True)
        with pytest.raises(RuntimeError, match=r"Tensor\.to"):
            t.to(torch.device("cuda", 0), torch.float16, True)
        with pytest.raises(RuntimeError, match=r"Tensor\.to"):
            t.to("cuda", non_blocking=True, copy=True)  # torch's parser takes no copy: parsed without it
        with pytest.raises(RuntimeError, match=r"Tensor\.to"):
            t.to("cuda", torch.float16, True, False)
        t.to("cuda", copy=True)
        with pytest.raises(RuntimeError, match=r"Tensor\.cuda"):
            t.cuda(non_blocking=True)
        with pytest.raises(RuntimeError, match=r"Tensor\.cuda"):
            t.cuda(None, True, torch.preserve_format)
        t.cuda(0, False, torch.preserve_format)
        dst = SimpleNamespace(device=torch.device("cuda"))
        with pytest.raises(RuntimeError, match=r"Tensor\.copy_"):
            torch.Tensor.copy_(dst, t, non_blocking=True)
        torch.Tensor.copy_(dst, t)
    assert len(copy_check["to"]) == 5 and len(copy_check["cuda"]) == 1 and len(copy_check["copy_"]) == 1

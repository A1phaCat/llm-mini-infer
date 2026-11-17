"""The launcher's supervision on fake processes (`launcher`): the exit code after a drain says whether a process failed
during it, and while the image pool's workers and the frontends start, the engine and the pool are watched as the
supervision loop watches them."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from llm_mini_infer import config, launcher
from llm_mini_infer.launcher import Watchdog

from ..support.api_util import Server
from ..support.fixtures import require_qwen
from ..support.paths import ROOT


class Proc:
    """A process the launcher holds: alive until `exitcode` is set."""

    def __init__(self, name: str, exitcode: int | None = None) -> None:
        self.name, self.exitcode, self.pid = name, exitcode, 0

    def is_alive(self) -> bool:
        return self.exitcode is None

    def kill(self) -> None:
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        pass


def drain(frontends: list[Proc], engine: Proc, caplog) -> int:
    """`_supervise` after SIGTERM, with the frontends already gone."""
    wake_r, wake_w = socket.socketpair()
    try:
        with caplog.at_level(logging.INFO, logger="llm_mini_infer.launcher"):
            return launcher._supervise(config.Config(), [engine, *frontends], frontends, engine,
                                       Watchdog(SimpleNamespace(value=0), 0.0), [signal.SIGTERM], wake_r)
    finally:
        wake_r.close()
        wake_w.close()


def test_a_clean_drain_exits_0(caplog) -> None:
    engine = Proc("llm-mini-infer-engine")
    assert drain([Proc("llm-mini-infer-frontend-0", 0), Proc("llm-mini-infer-frontend-1", 0)], engine, caplog) == 0
    assert "frontends drained" in caplog.text and "exited during the drain" not in caplog.text


def test_a_frontend_that_failed_during_the_drain_exits_1(caplog) -> None:
    frontends = [Proc("llm-mini-infer-frontend-0", 0), Proc("llm-mini-infer-frontend-1", -11)]
    assert drain(frontends, Proc("llm-mini-infer-engine"), caplog) == 1
    assert "frontends drained, but llm-mini-infer-frontend-1 exited during the drain (SIGSEGV)" in caplog.text


def test_an_engine_that_died_during_the_drain_exits_1(caplog) -> None:
    assert drain([Proc("llm-mini-infer-frontend-0", 0)], Proc("llm-mini-infer-engine", 1), caplog) == 1
    assert "frontends drained, but llm-mini-infer-engine exited during the drain (code 1)" in caplog.text


class Never:
    """A ready pipe that never reports."""

    def poll(self, timeout: float) -> bool:
        time.sleep(min(timeout, 0.01))
        return False


def test_an_engine_that_dies_while_a_frontend_starts_ends_the_start() -> None:
    """The wait for a frontend's ready message (300 s) also watches the engine: its death ends the start at once,
    through the start's failure path (a RuntimeError)."""
    engine = Proc("llm-mini-infer-engine")
    watchdog = Watchdog(SimpleNamespace(value=0), 0.0)
    turns = []

    def check() -> None:
        turns.append(1)
        if len(turns) == 3:
            engine.exitcode = -9
        launcher._check_started(engine, watchdog)

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"llm-mini-infer-engine exited while the server started \(SIGKILL\)"):
        launcher._recv(Never(), Proc("llm-mini-infer-frontend-0"), 300, "frontend 0", [], check=check)
    assert len(turns) == 3 and time.monotonic() - t0 < 5


def test_a_hung_engine_is_stopped_while_the_server_starts_and_a_gap_only_rearms(monkeypatch) -> None:
    """The watchdog armed when the engine reported ready judges it at every turn of the start's waits: a check that
    comes long after the previous one (a slow spawn between two waits) re-arms it instead of judging; one period of
    staleness after that, the engine is stopped (SIGABRT for its stacks, then SIGKILL) and the start fails."""
    now, beat = [0], SimpleNamespace(value=0)
    watchdog = Watchdog(beat, 10.0, clock=lambda: now[0])
    engine = Proc("llm-mini-infer-engine")
    killed = []
    monkeypatch.setattr(launcher, "_kill_engine", killed.append)
    for s in (0.2, 0.4, 0.6):  # the engine stamps its heartbeat
        now[0] = beat.value = int(s * 1e9)
        launcher._check_started(engine, watchdog)
    now[0] = int(30e9)  # a 29.4 s gap, the heartbeat 29.4 s old: re-armed, not judged
    launcher._check_started(engine, watchdog)
    assert watchdog.rearmed == 1 and not killed
    t = 30.0
    with pytest.raises(RuntimeError, match=r"made no progress for 10 s \(engine.watchdog_s\) while the server started"):
        while t < 60:  # checks 0.2 s apart, the engine hung since the re-arm
            t += 0.2
            now[0] = int(t * 1e9)
            launcher._check_started(engine, watchdog)
    assert killed == [engine] and 40 < t < 41


def test_a_crash_loop_of_the_pool_while_the_frontends_start_ends_the_start() -> None:
    """Once its workers are ready, the pool is supervised while the frontends start: its crash loop ends the start."""
    calls = []
    pool = SimpleNamespace(supervise=lambda: calls.append(1) or len(calls) == 2)
    engine, watchdog = Proc("llm-mini-infer-engine"), Watchdog(SimpleNamespace(value=0), 0.0)
    launcher._check_started(engine, watchdog, pool)
    with pytest.raises(RuntimeError, match="image pool's workers kept dying while the server started"):
        launcher._check_started(engine, watchdog, pool)
    launcher._check_started(engine, watchdog)  # while the workers themselves start, the pool is not supervised
    assert len(calls) == 2


@pytest.mark.serial
@pytest.mark.parametrize("how", ["kill", "hang"])
def test_the_engine_is_watched_while_the_frontends_start(tmp_path, how) -> None:
    """The real launcher, its frontends starting a minute late (`tests.support.slow_start`): the engine killed, or
    stopped (a hang, past a 5 s engine.watchdog_s), once it reported ready ends the start within seconds, exit 1,
    not when the frontends report or their 300 s run out. The kill runs with the watchdog off: the death's own
    message is what ends it."""
    srv = Server(require_qwen(), tmp_path, extra={"engine": {"watchdog_s": 5.0 if how == "hang" else 0},
                                                   "observability": {"log_level": "info"}})
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["SLOW_START_S"] = "60"
    code = f"from tests.support.slow_start import serve; serve({str(srv.cfg_path)!r})"
    with srv.log_path.open("w") as log:
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)  # its own group: the finally kills the engine, pool, frontends
    pid = 0
    try:
        deadline = time.monotonic() + 60
        while not (m := re.search(r"engine ready in .*; pid (\d+)\)", srv.log_path.read_text())):
            assert time.monotonic() < deadline and proc.poll() is None, srv.log_path.read_text()[-3000:]
            time.sleep(0.1)
        pid = int(m.group(1))
        time.sleep(0.5)  # the pool's workers or the frontends are starting
        t0 = time.monotonic()
        os.kill(pid, signal.SIGKILL if how == "kill" else signal.SIGSTOP)
        assert proc.wait(40) == 1
        took = time.monotonic() - t0
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(10)
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    log = srv.log_path.read_text()
    assert took < (20 if how == "hang" else 10), log[-3000:]
    want = ("llm-mini-infer-engine exited while the server started" if how == "kill" else
            "made no progress for 5 s (engine.watchdog_s) while the server started")
    assert want in log, log[-3000:]

"""The engine loop while grammars build beside it (architecture.md, "Structured output"): running requests keep
stepping, a build's end wakes a blocked loop at once (a pipe beside the input socket), the loop blocks rather than
turns while builds are its only work, and the frontends see a compiling request in the waiting count."""

from __future__ import annotations

import shutil
import signal
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from llm_mini_infer.config import Config
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.messages import AddRequest, StepOutputs
from llm_mini_infer.engine.process import EngineLoop
from llm_mini_infer.grammar.threads import CompileThreads

from ..support.compile_jobs import manual
from ..support.fake_grammar import FakeGrammar, spec
from ..support.harness import EOS, Harness, Spec, prompt_of
from ..support.sched_cases import VOCAB

G = spec(3, 2, 30)


class SlowGrammar(FakeGrammar):
    """The fake grammar, whose compile takes `delay` seconds (on a compile thread) and notes when it ended."""

    def __init__(self, delay: float):
        super().__init__(VOCAB, (EOS,))
        self.delay = delay
        self.ended_ns = 0

    def compile(self, text: str):
        time.sleep(self.delay)
        out = super().compile(text)
        self.ended_ns = time.perf_counter_ns()
        return out


POOLS: list[CompileThreads] = []


@pytest.fixture(autouse=True)
def close_pools() -> Iterator[None]:
    """The tests' compile threads end with each test (daemons would idle on for the worker's life)."""
    yield
    while POOLS:
        POOLS.pop().close()


def slow_harness(delay: float, **kw) -> tuple[Harness, SlowGrammar]:
    h = Harness(specs=[], vocab=VOCAB, **kw)
    g = h.sched.grammars
    g.backend = slow = SlowGrammar(delay)
    g.threads = CompileThreads(2, "test-compile")
    POOLS.append(g.threads)
    return h, slow


def add(h: Harness, s: Spec) -> None:
    h.specs[s.rid] = s
    h.core.handle(AddRequest(rid=s.rid, frontend=0, prompt=prompt_of(s, h.vocab), max_tokens=s.max_tokens,
                             temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=s.seed, stop_token_ids=[],
                             grammar=s.grammar))


def run_loop(h: Harness, done: Callable[[], bool], before: Callable[[int], None] | None = None,
             limit_s: float = 20.0) -> tuple[list[int], list[tuple[int, StepOutputs]]]:
    """`EngineLoop.run` over the harness's scheduler with its real poll (the input socket and the wake-up pipe), no idle
    spin; `before(turn)` runs before each turn's step. Ends once `done()`. Returns each step's start (ns) and every
    StepOutputs sent with its time."""
    cfg = Config()
    cfg.dev.engine.idle_spin_ms = 0.0
    path = ipc.ipc_dir()
    loop = EngineLoop(cfg, h.core, path)
    sent: list[tuple[int, StepOutputs]] = []
    loop._send = lambda fe, msg: sent.append((time.perf_counter_ns(), msg))  # type: ignore[method-assign]
    starts: list[int] = []
    step = h.core.step
    end = time.monotonic() + limit_s

    def counted(emit=None):
        if before is not None:
            before(len(starts))
        starts.append(time.perf_counter_ns())
        out = step(emit)
        if done() or time.monotonic() > end:
            loop.stopping = True
        return out

    h.core.step = counted  # type: ignore[method-assign]
    term = signal.getsignal(signal.SIGTERM)
    try:
        loop.run()
    finally:
        signal.signal(signal.SIGTERM, term)
        loop.close()
        h.core.step = step  # type: ignore[method-assign]
        shutil.rmtree(path, ignore_errors=True)
    assert time.monotonic() <= end, "the loop did not finish in time"
    return starts, sent


@pytest.mark.serial
def test_a_builds_end_wakes_the_blocked_loop_and_it_does_not_spin() -> None:
    """The only work is a 200 ms build: the loop blocks (a handful of turns, not one per poll), and the build's end
    wakes it at once (not at its 1 s poll): the request is admitted within a few milliseconds."""
    h, slow = slow_harness(0.2)
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    req = h.sched.reqs[(0, 0)]
    starts, _ = run_loop(h, lambda: req.t_admit_ns > 0)
    latency_ms = (req.t_admit_ns - slow.ended_ns) / 1e6
    assert len(starts) <= 4 and 0 < latency_ms < 50, (len(starts), latency_ms)


@pytest.mark.serial
def test_running_requests_keep_stepping_while_a_grammar_builds() -> None:
    """A 200 ms build, added while two requests decode: their steps go on (no gap near 200 ms), and the new request
    runs once built. The fake waits for its steps asleep, the GIL free as CUDA's synchronize leaves it: its spin is a
    Python loop that holds the GIL, which starts a build late (by up to 200 ms here)."""
    h, slow = slow_harness(0.2, num_pages=300, max_model_len=512)
    for rid in (0, 1):  # decoding for 400 steps or more: past the build's end
        add(h, Spec(rid=rid, prompt_len=8, max_tokens=400, seed=rid + 1))
    h.runner.step_s = 1e-3  # 1 ms a step
    h.runner.wait_mode = "sleep"

    def before(turn: int) -> None:
        if turn == 20:
            add(h, Spec(rid=2, prompt_len=8, max_tokens=3, seed=3, grammar=G))

    starts, _ = run_loop(h, lambda: not h.core.busy, before)
    i = 19  # the gap that holds the add (`before` runs ahead of its turn's start)
    end = next(j for j, t in enumerate(starts) if t > slow.ended_ns)
    gaps = [(b - a) / 1e6 for a, b in zip(starts[i:end], starts[i + 1:end + 1])]
    assert gaps and max(gaps) < 50 and end < 400, (max(gaps), end)
    assert h.sched.grammars.compiles == 1 and not h.sched.reqs


def test_the_waiting_count_reaches_the_frontends_while_nothing_steps() -> None:
    """A lone compiling request: the next blocking turn sends idle stats counting it as waiting; once it ran and
    finished, the count is back to 0."""
    h, _ = slow_harness(0.3)
    add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
    _, sent = run_loop(h, lambda: not h.core.busy)
    waits = [(t, m.stats.waiting) for t, m in sent if m.stats is not None]
    assert waits and waits[0][1] == 1 and waits[0][0] < h.sched.grammars.backend.ended_ns and waits[-1][1] == 0


def test_the_boundary_poll_books_its_collect_as_poll_work() -> None:
    """`_poll_now` collects a build that ended (the request then waits, so it joins the next plan) and returns the ns
    it spent; a collect that does nothing (no build ended, the request still compiling) and no message give 0."""
    h = Harness(specs=[], vocab=VOCAB)
    jobs = manual(h.sched.grammars)
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), h.core, path)
    try:
        assert loop._poll_now() == 0
        add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
        assert h.core.collect_due and loop._poll_now() == 0 and h.sched.num_compiling == 1
        jobs.complete()
        assert h.core.collect_due and loop._poll_now() > 0
        assert h.sched.num_waiting == 1 and h.sched.num_compiling == 0 and not h.core.collect_due
        assert loop._poll_now() == 0
    finally:
        loop.close()
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.serial  # a wall-clock bound
def test_the_wake_up_pipe_is_drained_and_survives_its_reader() -> None:
    """A build's end writes one byte; the poll that wakes for it empties the pipe. After `close` a late build's write
    fails quietly."""
    h, _ = slow_harness(0.0)
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), h.core, path)
    try:
        g = h.sched.grammars
        assert g.wake == loop._wake and h.core.drain == loop._drain
        done = threading.Event()
        g.wake = lambda: (loop._wake(), done.set())
        add(h, Spec(rid=0, prompt_len=8, max_tokens=3, seed=1, grammar=G))
        assert done.wait(5)
        t = time.perf_counter()
        loop._poll(block=True)
        assert time.perf_counter() - t < 0.5
        with pytest.raises(BlockingIOError):
            import os

            os.read(loop._wake_r, 1)
    finally:
        loop.close()
        shutil.rmtree(path, ignore_errors=True)
    loop._wake()  # the read end is closed: no error

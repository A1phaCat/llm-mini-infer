"""A stand-in for the engine's compile threads (`grammar.threads.CompileThreads`) whose jobs run when a test says, so
the turn at which a grammar's build ends is the test's choice (`grammar.engine.Grammars.submit`)."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from llm_mini_infer.grammar.engine import Grammars


class ManualThreads:
    """`submit` queues a job and returns its future; `complete` runs one on the caller's thread (its done-callbacks
    with it), `fail` ends one with an exception instead. `submitted` counts the jobs ever queued. `run(fn)`, which
    `Grammars.compile` and `forced_prefix` wait on, runs `fn` at once."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Callable[[], Any], Future]] = []
        self.submitted = 0

    def submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        self.jobs.append((fn, fut))
        self.submitted += 1
        return fut

    def run(self, fn: Callable[[], Any]) -> Any:
        return fn()

    def complete(self, i: int = 0) -> None:
        fn, fut = self.jobs.pop(i)
        try:
            fut.set_result(fn())
        except BaseException as exc:  # handed to the future, as CompileThreads does
            fut.set_exception(exc)

    def complete_all(self) -> None:
        while self.jobs:
            self.complete()

    def fail(self, exc: BaseException, i: int = 0) -> None:
        self.jobs.pop(i)[1].set_exception(exc)

    def complete_some(self, rng: random.Random, p: float) -> int:
        """Each queued job completed with probability `p`, in a random order: how many."""
        picked = [j for j in self.jobs if rng.random() < p]
        rng.shuffle(picked)
        for job in picked:
            self.complete(self.jobs.index(job))
        return len(picked)

    def __len__(self) -> int:
        return len(self.jobs)


def manual(grammars: Grammars) -> ManualThreads:
    """`grammars`' builds moved onto a ManualThreads (a harness builds its Grammars without threads)."""
    threads = grammars.threads = ManualThreads()  # type: ignore[assignment]
    return threads

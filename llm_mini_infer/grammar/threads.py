"""Threads with a large stack for llguidance's compiles (architecture.md, "Structured output").

llguidance compiles a grammar by recursion, so every compile and check runs on a thread of its own with a large stack:
in the frontend a few threads that every build and check of a cache miss goes to, in the engine a few that its builds (a
compile, then the forced prefix) run on beside the step loop (`grammar.engine.Grammars.submit`). Under the caps
(`grammar.bounds`) the deepest compile measured takes a small part of `STACK_BYTES`. A stack is address space reserved
at the thread's start, committed only as it is touched: a strict address-space limit (`RLIMIT_AS`,
`vm.overcommit_memory=2`) can refuse it: the pool's start then fails with a message that says so.

The threads start with the pool, `threading.stack_size` set around their start and restored after it: a thread that
another thread starts meanwhile gets the same stack. llguidance's matchers, compiled on one thread, are used on any
(they are `Send`): the engine's step loop and llguidance's own mask threads take them.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

STACK_BYTES = 32 << 20  # several times what the deepest grammar at the caps takes
_START = threading.Lock()  # one pool's threads start at a time, so the stack size each sets is its own


class CompileThreads:
    """`n` daemon threads with a `stack` of their own, running what `submit` gives them in order of arrival."""

    def __init__(self, n: int = 1, name: str = "grammar", stack: int = STACK_BYTES) -> None:
        self._jobs: queue.SimpleQueue[tuple[Callable[[], Any], Future] | None] = queue.SimpleQueue()
        self.threads: list[threading.Thread] = []
        self.stack = stack
        self._lock = threading.Lock()  # `submit` and `close`: no job queued after the threads' end
        self.closed = False
        with _START:
            old = threading.stack_size(stack)
            try:
                for i in range(n):
                    t = threading.Thread(target=self._run, name=f"{name}-{i}", daemon=True)
                    t.start()
                    self.threads.append(t)
            except RuntimeError as exc:  # "can't start new thread"
                self.close()
                raise RuntimeError(f"cannot start {n} grammar compile threads of {stack >> 20} MiB of stack each: an "
                                   "address-space limit (RLIMIT_AS, ulimit -v) or vm.overcommit_memory=2 refuses them: "
                                   f"{exc}") from exc
            finally:
                threading.stack_size(old)

    def submit(self, fn: Callable[[], Any]) -> Future:
        """`fn` queued for a thread; a RuntimeError once the pool is closed."""
        fut: Future = Future()
        with self._lock:
            if self.closed:
                raise RuntimeError("the grammar compile threads are closed")
            self._jobs.put((fn, fut))
        return fut

    def run(self, fn: Callable[[], Any]) -> Any:
        """`fn()` on one of the threads, waited for; its exception raised here."""
        return self.submit(fn).result()

    def close(self) -> None:
        """The threads end after the jobs queued before."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            for _ in self.threads:
                self._jobs.put(None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, fut = job
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # handed to the waiter, which raises it
                fut.set_exception(exc)

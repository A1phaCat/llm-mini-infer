"""Grammars in the engine process: compile once, clone per request, masks in parallel, and each grammar's forced prefix.

**The backend** is llguidance (`LLGuidance`; docs/architecture.md, "Structured output"), under these conditions:
- its tokenizer is built from the checkpoint's `tokenizer.json` with the `tokenizers` library, as `llguidance.hf` does
  from a transformers tokenizer, so the serving path imports no transformers. On Gemma 4 it equals
  `llguidance.hf.from_tokenizer`'s on every id (tests/structured/test_grammar_parity.py);
- the vocabulary is the logits width (`n_vocab`; ids past the tokenizer's are never allowed; llguidance refuses a
  smaller one);
- `eos`: the EOS ids a completed grammar allows, and only where the grammar accepts, chosen per backend by
  `engine.process` (`_grammars`) as vLLM's backends choose them: with guidance the tokenizer's `eos_token`; with
  auto and xgrammar the model's stop ids (`generation_config.json`'s EOS ids). The grammar must end on a token the
  model ends its turn with (Qwen's turns end with `<|im_end|>`, transformers' eos_token for Qwen3.8), and the engine
  refuses to start when the model's stop ids miss the grammar's EOS;
- compiled grammars are cached by their text (`Grammars`), and each request gets a clone (`deep_copy`, far cheaper
  than a compile);
- a compile and a forced prefix run on a thread with a large stack (`grammar.threads`). The engine's builds run beside
  its step loop (`Grammars.submit`, `settle`): a request waits for its grammar outside the waiting heap (the scheduler's
  COMPILING).

**Masks** go straight into the runner's (page-locked) buffer through llguidance's parallel fill, which releases the
GIL: `fill` takes the rows that need one mask and the speculative rows that need 1 + k along their drafts (llguidance
advances a row's matcher along its drafts, writes a mask after each, stops at the first draft the grammar rejects and
rolls back; the rows after it are never read). Clones of one compiled grammar share its caches, so a step's masks take
longer on many threads than on one: `dev.structured_output.threads` is 1 by default.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np

from . import GrammarError
from .spec import clean_error
from .threads import CompileThreads

log = logging.getLogger(__name__)
CACHE_SIZE = 128
FORCED_MAX = 16  # the most tokens a grammar's forced prefix takes (`Grammars.forced_prefix`)


class Matcher(Protocol):
    """What the engine uses of a matcher (llguidance's `LLMatcher`, or the fake's)."""

    def consume_tokens(self, tokens: list[int]) -> bool: ...

    def try_consume_tokens(self, tokens: list[int]) -> int: ...  # up to the first rejected token, no error state

    def validate_tokens(self, tokens: list[int]) -> int: ...  # how many it would take in a row (to an EOS), unmoved

    def rollback(self, num_tokens: int) -> object: ...

    def is_stopped(self) -> bool: ...  # nothing but the EOS ids allowed next (a complete document)

    def is_error(self) -> bool: ...

    def get_error(self) -> str: ...

    def compute_ff_tokens(self) -> list[int]: ...  # the tokens the grammar forces from here (fast-forward), maybe none


class Backend(Protocol):
    name: str
    vocab_size: int
    eos: tuple[int, ...]

    def compile(self, text: str) -> Any: ...

    def instance(self, compiled: Any) -> Matcher: ...

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]],
             buf: np.ndarray) -> None: ...

    # The matcher's mask allows `token` and nothing else. `own`: called beside the step loop (a build job), through no
    # buffer or call that `fill` shares
    def only(self, m: Any, token: int, own: bool = False) -> bool: ...


def words(vocab_size: int) -> int:
    """int32 words of one mask."""
    return (vocab_size + 31) // 32


class LLGuidance:
    """llguidance over the checkpoint's tokenizer (module docstring). `eos`: the EOS ids a completed grammar allows, or
    a token's name (the tokenizer's eos_token: guidance), resolved by the tokenizer loaded here."""

    name = "llguidance"

    def __init__(self, tokenizer_json: str | Path, vocab_size: int, eos: Sequence[int] | str, threads: int = 1):
        import llguidance
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        tok = Tokenizer.from_file(str(tokenizer_json))
        tok.no_padding()
        tok.no_truncation()
        if isinstance(eos, str):
            eos_id = tok.token_to_id(eos)
            if eos_id is None:
                raise ValueError(f"{tokenizer_json} has no token {eos!r} (the tokenizer's eos_token)")
            eos = (eos_id,)
        self._llg, self.version = llguidance, llguidance.__version__
        self.tokenizer = llguidance.LLTokenizer(tok.to_str(), n_vocab=vocab_size, eos_token=list(eos))
        self.executor = llguidance.LLExecutor(num_threads=threads)
        self.vocab_size, self.eos, self.threads = vocab_size, tuple(eos), threads
        self._one = np.zeros((1, words(vocab_size)), np.int32)  # `only`'s mask, on the engine thread
        self._own = threading.local()  # `only`'s mask on another thread (a build job's): its `one`
        self.build_s = time.perf_counter() - t0

    def compile(self, text: str) -> Any:
        m = self._llg.LLMatcher(self.tokenizer, text, log_level=0)
        if m.is_error():
            raise GrammarError(clean_error(m.get_error()))
        return m

    @staticmethod
    def instance(compiled: Any) -> Any:
        return compiled.deep_copy()

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]], buf: np.ndarray) -> None:
        if buf.dtype != np.int32 or buf.ndim != 2 or not buf.flags.c_contiguous:
            raise ValueError("masks need a C-contiguous int32 [rows, words] buffer")
        row, n = buf.shape[1] * 4, buf.shape[0]
        if singles:
            self.executor.unsafe_compute_mask_ptr(singles, buf.ctypes.data, row, n)
        if drafts:
            self.executor.unsafe_compute_mask_ptr_with_draft_token(drafts, buf.ctypes.data, row, n)

    def only(self, m: Any, token: int, own: bool = False) -> bool:
        """Whether the mask at the matcher's state allows `token` alone (for `Grammars.forced_prefix`): one mask into a
        buffer kept for it, then its nonzero words, which is cheaper than going through `compute_bitmask`'s bytes.

        `own`: a build job's, beside the engine thread's `fill`. The mask goes into the calling thread's buffer through
        the matcher's own `unsafe_compute_mask_ptr`, not the executor: at 1.9.1 the executor computes a lone matcher's
        mask the same way (no pool), and llguidance documents no concurrent use of it. A matcher's mask releases the
        GIL; clones of one compiled grammar take turns on the lock their caches share."""
        if own:
            w = getattr(self._own, "one", None)
            if w is None:
                w = self._own.one = np.zeros((1, words(self.vocab_size)), np.int32)
            m.unsafe_compute_mask_ptr(w.ctypes.data, w.shape[1] * 4)
        else:
            w = self._one
            self.executor.unsafe_compute_mask_ptr([(m, 0)], w.ctypes.data, w.shape[1] * 4, 1)
        return 0 <= token < self.vocab_size and np.count_nonzero(w) == 1 and \
            int(w[0].view(np.uint32)[token >> 5]) == 1 << (token & 31)


class Built(NamedTuple):
    """What a build job (`Grammars.submit`) returns from its thread: the compiled grammar or its `GrammarError` (a
    value, as `Grammars._compile` returns it), the forced prefix (None: not wanted, or the compile failed), and the
    compile's ns (None: the grammar was cached, the job computed its prefix only)."""

    got: Any
    forced: tuple[int, ...] | None
    ns: int | None


class Grammars:
    """The engine's grammars: a backend, compiled grammars cached by their text (LRU, errors included), and the
    counters behind the compile, mask and error metrics (`llm_mini_infer:grammar_*`, EngineStats). `threads`: where a
    compile and a forced prefix run (the engine's: `engine.process`); without, on the calling thread.

    **Builds beside the step loop** (`submit`, `settle`; architecture.md, "Structured output"). With `threads`, a
    grammar the caches miss is built by a job on them: its compile, then its forced prefix when wanted. One build runs
    per (text, prefix) at a time (`_inflight`), shared by every request that needs it. Each build has two futures: the
    job's, which its thread completes, and the build's, which the requests hold and only `settle` completes, on the
    engine thread: so whether a request's grammar is ready is decided there, never by a race with a thread. A job's
    done-callback queues it on `_ready` and calls `wake`; `settle` then fills the caches, counts the compile and logs
    it, as `compile` does. Only the engine thread touches the caches and the counters (no lock); a job returns values.
    """

    def __init__(self, backend: Backend, cache_size: int = CACHE_SIZE, threads: CompileThreads | None = None,
                 wake: Callable[[], None] | None = None):
        self.backend = backend
        self.threads = threads
        self.wake = wake  # called on a job's thread when its build is done (the engine loop's wake-up)
        self.eos = frozenset(backend.eos)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._forced: OrderedDict[str, tuple[int, ...]] = OrderedDict()  # `forced_prefix`, by grammar text
        # The builds in flight, by (text, prefix): the build's future; and the builds whose jobs are done, for `settle`:
        # (key, the build's future, the job's)
        self._inflight: dict[tuple[str, bool], Future] = {}
        self._ready: deque[tuple[tuple[str, bool], Future, Future]] = deque()
        self.compiles = self.compile_errors = 0
        self.compile_ns = 0
        self.last_compile_ns = 0
        self.mask_rows = 0
        self.mask_ns = 0
        self.errors = 0  # requests failed by a token their grammar rejected (never expected)

    def compile(self, text: str) -> Any:
        """The compiled grammar, or its `GrammarError`: a fresh one each time, so that a cached error's traceback does
        not grow, and holds no compile's frames."""
        got = self._cache.get(text)
        if got is None:
            t0 = time.perf_counter_ns()
            got = self._on_thread(lambda: self._compile(text))
            self._store(text, got, time.perf_counter_ns() - t0)
        else:
            self._cache.move_to_end(text)
        if isinstance(got, GrammarError):
            raise GrammarError(*got.args)
        return got

    def _store(self, text: str, got: Any, ns: int) -> None:
        """A compile's outcome cached and counted (`compile`, `settle`)."""
        if isinstance(got, GrammarError):
            self.compile_errors += 1
        self.last_compile_ns = ns
        self.compile_ns += ns
        self.compiles += 1
        self._cache[text] = got
        self._cache.move_to_end(text)  # a second build of the text (another `prefix`) renews it
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        if not isinstance(got, GrammarError):
            log.info("compiled a grammar of %d characters in %.2f ms", len(text), ns / 1e6)

    def submit(self, text: str, prefix: bool) -> Future:
        """A request's grammar, never waited for: a future whose result is the compiled grammar and its forced prefix,
        `(compiled, forced)` (forced None: not wanted, or to be read with `forced_prefix`, a cache hit then), or the
        `GrammarError` as a value; another exception is the future's. `prefix`: the request wants the forced prefix.

        The future comes back done when the grammar is decided now: without `threads`, `compile` runs here (an
        exception other than GrammarError raised here, as `compile` raises it); with them, a cache hit (in `_forced`
        too when `prefix`). Otherwise it is the build's (class docstring), done at a later `settle`: a new job, or the
        build of the same (text, prefix) already in flight. A cached grammar whose prefix is missing gets a job for the
        prefix alone."""
        if self.threads is None:
            fut: Future = Future()
            try:
                fut.set_result((self.compile(text), None))
            except GrammarError as exc:
                fut.set_result(GrammarError(*exc.args))  # no traceback held (its frames hold this future)
            return fut
        key = (text, prefix)
        fut = self._inflight.get(key)
        if fut is not None:
            return fut
        got = self._cache.get(text)
        if got is not None and (isinstance(got, GrammarError) or not prefix or text in self._forced):
            self._cache.move_to_end(text)
            fut = Future()
            fut.set_result(GrammarError(*got.args) if isinstance(got, GrammarError) else (got, None))
            return fut
        job = self.threads.submit(lambda: self._build(text, got, prefix))
        fut = self._inflight[key] = Future()
        job.add_done_callback(lambda j, key=key, fut=fut: self._done(key, fut, j))
        return fut

    def _build(self, text: str, compiled: Any, prefix: bool) -> Built:
        """A build job, on a compile thread: `compiled` None compiles `text` first; the prefix when wanted."""
        ns = None
        if compiled is None:
            t0 = time.perf_counter_ns()
            compiled = self._compile(text)
            ns = time.perf_counter_ns() - t0
            if isinstance(compiled, GrammarError):
                return Built(compiled, None, ns)
        return Built(compiled, self._prefix(compiled, own=True) if prefix else None, ns)

    def _done(self, key: tuple[str, bool], fut: Future, job: Future) -> None:
        """A job's done-callback, on its thread (or on the caller's, if done before it was added)."""
        self._ready.append((key, fut, job))
        wake = self.wake
        if wake is not None:
            wake()

    def settle(self) -> int:
        """The builds whose jobs are done, settled (engine thread only): each leaves `_inflight`, fills the caches
        (`compile`'s eviction), counts its compile once and logs it, then completes its future. A build no request waits
        for any more settles too. An exception other than GrammarError is the build's future's, and caches nothing.
        Returns how many settled; returns at once when none is ready."""
        ready = self._ready
        n = 0
        while ready:
            key, fut, job = ready.popleft()
            n += 1
            if self._inflight.get(key) is fut:
                del self._inflight[key]
            exc = CancelledError() if job.cancelled() else job.exception()  # (never cancelled: no caller cancels)
            if exc is not None:
                fut.set_exception(exc)
                continue
            got, forced, ns = job.result()
            text = key[0]
            if ns is not None:
                self._store(text, got, ns)
            if isinstance(got, GrammarError):
                fut.set_result(GrammarError(*got.args))
                continue
            if forced is not None:
                self._forced[text] = forced
                self._forced.move_to_end(text)
                if len(self._forced) > self.cache_size:
                    self._forced.popitem(last=False)
            fut.set_result((got, forced))
        return n

    @property
    def building(self) -> int:
        """Builds in flight (`_inflight`), whether or not a request waits for them."""
        return len(self._inflight)

    def _compile(self, text: str) -> Any:
        try:
            return self.backend.compile(text)
        except GrammarError as exc:
            return GrammarError(*exc.args)  # its message alone, no traceback

    def _on_thread(self, fn: Any) -> Any:
        return fn() if self.threads is None else self.threads.run(fn)

    def matcher(self, compiled: Any) -> Matcher:
        return self.backend.instance(compiled)

    def only(self, m: Matcher, token: int) -> bool:
        return self.backend.only(m, token)

    def forced_prefix(self, text: str) -> tuple[int, ...]:
        """The tokens a grammar forces at its start, cached by its text: along a fresh matcher, llguidance's first
        fast-forward token while the mask allows it alone, one at a time, never an EOS, at most `FORCED_MAX`. Compact
        JSON with a fixed first key forces `{"` and the key's tokens (two for a one-token key). The `":` after the key
        is forced bytes, not a forced token, when the first value can start in a way that merges with it (`":-`,
        `":"...`); where the value's first byte is fixed (an enum, an integer with a minimum of 0, a boolean, an object)
        the prefix goes on with the `":`-led token. `no_forcing` grammars (auto, xgrammar) force none."""
        got = self._forced.get(text)
        if got is None:
            compiled = self.compile(text)
            got = self._forced[text] = self._on_thread(lambda: self._prefix(compiled))
            if len(self._forced) > self.cache_size:
                self._forced.popitem(last=False)
        else:
            self._forced.move_to_end(text)
        return got

    def _prefix(self, compiled: Any, own: bool = False) -> tuple[int, ...]:
        """`forced_prefix`'s walk. `own`: a build job's, beside the step loop: its masks go through no buffer or call
        the engine thread's `fill` shares (`Backend.only`)."""
        m = self.matcher(compiled)
        out: list[int] = []
        while len(out) < FORCED_MAX:
            ff = m.compute_ff_tokens()
            if not ff or ff[0] in self.eos or \
                    not (self.backend.only(m, ff[0], True) if own else self.only(m, ff[0])) or \
                    not m.consume_tokens([ff[0]]):
                break
            out.append(int(ff[0]))
        return tuple(out)

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]], buf: np.ndarray,
             n: int) -> None:
        t0 = time.perf_counter_ns()
        self.backend.fill(singles, drafts, buf)
        self.mask_ns += time.perf_counter_ns() - t0
        self.mask_rows += n

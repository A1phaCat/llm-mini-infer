"""The engine's step loop body (docs/architecture.md, "The step loop"), free of I/O so tests can drive it in-process.

One call of `step()` is one turn of the loop, one step ahead of the device: it collects the grammar builds that ended
(`collect`: their requests join the waiting), schedules and launches step N+1, waits
for step N's tokens, folds them in, emits step N's outputs per frontend (through `emit`, or returned to the caller),
fills step N+1's grammar masks (`Scheduler.fill_masks`) and launches its sampling. The process wrapper (`process.py`)
polls the frontends before each call and sends the outputs through `emit` before the masks are filled, so their
computation never delays a token on its way to the client.

**Planning at the forward/sampling boundary** (`engine.plan_at_boundary`; architecture.md, "Planning at the
forward/sampling boundary"). When step N is split into its forward and its sampling, the process wrapper first polls
until N's forward has ended on the device (`await_boundary`, `Runner.forward_pending`), then runs the turn unchanged.
The scheduler's decisions for the arrivals it is given stay the same (`tests/sched/test_boundary.py`). The device idles
when the poll, the planning and N+1's launch outlast N's sampling part, which `wait` counts as idle before N+1
(`StepResult.gpu_idle_before_ns`).
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass

from ..metrics import STEP_IMAGES, STEP_ROWS
from ..runner.base import Runner, StepResult
from ..sched.scheduler import FrontendOutputs, Inflight, Scheduler
from .messages import Abort, AddRequest

# Indices into EngineCore.phase_ns (cumulative ns per loop phase, for the overhead breakdown). "boundary": polling for
# the end of the step in flight's forward (`await_boundary`), a wait for the device like "wait"
PHASES = ("poll", "schedule", "launch_forward", "wait", "process", "send", "masks", "launch_sample", "boundary")
P_POLL, P_SCHED, P_LAUNCH, P_WAIT, P_PROCESS, P_SEND, P_MASKS, P_SAMPLE, P_BOUNDARY = range(len(PHASES))

# The engine's planning before it launches the next step (`derive_plan_at_boundary`): the scheduler's turn at a few
# verify rows and the poll, rounded up. One decode row is the tightest case: its sampling part is the shortest, while a
# launch costs about as much at any row count.
BOUNDARY_PLAN_MS = 0.3


def derive_plan_at_boundary(*, forward_end: bool, sample_ms: float, launch_ms: float) -> tuple[bool, str]:
    """Whether `engine.plan_at_boundary` pays on this runner, and why, from the runner's capabilities
    (`Capabilities.forward_end`, `sample_step_ms`, `launch_step_ms`: its memory profile's lone masked decode step, the
    tightest case): it does when the runner reports its forwards' ends and that step's sampling part outlasts launching
    the next step plus the engine's planning (`BOUNDARY_PLAN_MS`), so the next step is queued before the device runs
    dry."""
    if not forward_end:
        return False, ("off: the runner reports no forward's end (Gemma 4 on CUDA with structured output does, and "
                       "the fake runner with dev.runner.fake.sample_ms; the Qwen runner does not)")
    if sample_ms <= 0:
        return False, "off: the runner timed no split step (no memory profile: dev.runner.cuda.memory_gb set)"
    need = launch_ms + BOUNDARY_PLAN_MS
    why = (f"a lone decode step samples for {sample_ms:.2f} ms of the device; planning and launching the next takes "
           f"about {need:.2f} ms of the host ({launch_ms:.2f} ms to launch, {BOUNDARY_PLAN_MS:g} ms to plan)")
    if sample_ms >= need:
        return True, f"on: {why}; arrivals during a step's forward join the next step"
    return False, f"off: {why}: the device would wait for the next step"


@dataclass
class StepReport:
    """What one turn produced: outputs of the step that completed, and its accounting."""

    outputs: dict[int, FrontendOutputs]
    step_id: int  # the step whose results these are (0: none, only rejections, aborts or pixel releases)
    step_tokens: int
    runner_ns: int
    wait_ns: int


class EngineCore:
    def __init__(self, sched: Scheduler, runner: Runner):
        self.sched = sched
        self.runner = runner
        self._inflight: tuple[Inflight, object, bool] | None = None  # (meta, handle, pipelined at launch)
        self.phase_ns = [0] * len(PHASES)
        self.gpu_busy_ns = 0
        self.gpu_idle_ns = 0
        self.gpu_sample_ns = 0  # device time of split steps' sampling parts (`StepResult.gpu_sample_ns`)
        self.encoder_ns = 0  # device time of ENCODE sections (images)
        self.last_encoder_ns = 0  # of the step whose results came last
        self.steps = 0
        # Every completed step's decode (verify) rows and encoded images, as counts per bucket of STEP_ROWS and
        # STEP_IMAGES (llm_mini_infer:step_decode_rows, llm_mini_infer:step_encoder_images): kept here, not observed by
        # the frontends, so a step that sends no outputs (a non-final prefill chunk, a refine's piece alone) counts too
        self.rows_hist = [0] * (len(STEP_ROWS) + 1)
        self.rows_sum = 0
        self.images_hist = [0] * (len(STEP_IMAGES) + 1)
        self.images_sum = 0
        self.drain: Callable[[], None] | None = None  # empties the loop's wake-up pipe (`EngineLoop`), once builds end

    def handle(self, msg: AddRequest | Abort) -> None:
        if isinstance(msg, AddRequest):
            self.sched.add(msg)
        elif isinstance(msg, Abort):
            self.sched.abort(msg.frontend, msg.rids, msg.reason)

    @property
    def busy(self) -> bool:
        """Something to schedule or send, a step in flight, or a grammar being built (otherwise the loop may block on
        the poll; a stopping loop ends)."""
        s = self.sched
        return self._inflight is not None or s.has_work() or s.has_unsent() or s.num_compiling > 0

    @property
    def compiling_only(self) -> bool:
        """The only work is grammars being built: nothing in flight, to schedule or to send. The loop blocks on its
        poll then, which a build's end wakes (`EngineLoop`), instead of turning until it ends."""
        s = self.sched
        return self._inflight is None and not s.has_work() and not s.has_unsent() and s.num_compiling > 0

    @property
    def collect_due(self) -> bool:
        """`collect` has work: a build ended, or a request waits for one (its time limit)."""
        g = self.sched.grammars
        return g is not None and (bool(g._ready) or self.sched.num_compiling > 0)

    def collect(self) -> bool:
        """The grammar builds that ended, and their requests (`Scheduler.collect`). A build's end queues it, then
        writes the loop's wake-up pipe: emptied here once one is queued, so the next blocking poll does not return for
        it again (a byte written just after is one spare turn). Whether it did work: a build settled, or a request
        left COMPILING."""
        g = self.sched.grammars
        if g is None:
            return False
        settled = bool(g._ready)
        if settled and self.drain is not None:
            self.drain()
        return self.sched.collect() or settled

    @property
    def paused_only(self) -> bool:
        """Nothing in flight, and the scheduler's work waits only for paused frontends (`Scheduler.paused_only`)."""
        return self._inflight is None and self.sched.paused_only()

    def await_boundary(self, poll: Callable[[], int]) -> int:
        """Planning at the forward/sampling boundary (module docstring): while the step in flight still runs its
        forward on the device, `poll` the frontends (their arrivals join the next plan; it returns the ns it spent
        on handled work, messages and a collect that did work: the caller's poll time and engine CPU). Returns the ns
        spent waiting (P_BOUNDARY)."""
        inf = self._inflight
        if inf is None or not self.runner.forward_pending(inf[1]):
            return 0
        t0, handled = time.perf_counter_ns(), 0
        while self.runner.forward_pending(inf[1]):
            handled += poll()
        dt = time.perf_counter_ns() - t0 - handled
        self.phase_ns[P_BOUNDARY] += dt
        return dt

    def step(self, emit: Callable[[StepReport], None] | None = None) -> StepReport | None:
        """One turn. With `emit`, the completed step's outputs go to it before the next step's masks are filled and its
        sampling launched, and the turn returns None; without, they are returned."""
        sched, runner, ph = self.sched, self.runner, self.phase_ns
        clock = time.perf_counter_ns
        t0 = clock()
        self.collect()
        pipelined = self._inflight is not None
        planned = sched.schedule() if sched.has_work() else None
        t1 = clock()
        ph[P_SCHED] += t1 - t0
        handle = None
        if planned is not None:
            plan, inf = planned
            handle = runner.launch_forward(plan)
            t2 = clock()
            ph[P_LAUNCH] += t2 - t1
            sched.on_launched(inf, t2)
            t1 = t2
        report = None
        if self._inflight is not None:
            prev, prev_handle, prev_pipelined = self._inflight  # `pipelined` is the new step's (stored below)
            res: StepResult = runner.wait(prev_handle)
            t2 = clock()
            wait_ns = t2 - t1
            ph[P_WAIT] += wait_ns
            outputs = sched.process(prev, res.tokens, res.n_out, res.gpu_busy_ns, res.status, res.drafts,
                                    res.logprobs)
            t1 = clock()
            ph[P_PROCESS] += t1 - t2
            self.gpu_busy_ns += res.gpu_busy_ns
            self.gpu_sample_ns += res.gpu_sample_ns
            self.last_encoder_ns = res.encoder_busy_ns
            self.encoder_ns += res.encoder_busy_ns
            # Idle with work queued: always between a step's forward and its sampling; before the forward only
            # if a step was in flight when it was launched (otherwise the device was idle for lack of work).
            self.gpu_idle_ns += res.gpu_idle_inside_ns + (res.gpu_idle_before_ns if prev_pipelined else 0)
            self.steps += 1
            rows = int(prev.device.sum())
            self.rows_hist[bisect_left(STEP_ROWS, rows)] += 1
            self.rows_sum += rows
            self.images_hist[bisect_left(STEP_IMAGES, prev.n_encode)] += 1
            self.images_sum += prev.n_encode
            report = StepReport(outputs=outputs, step_id=prev.step_id, step_tokens=prev.n_tokens,
                                runner_ns=res.gpu_busy_ns, wait_ns=wait_ns)
        elif sched.has_unsent():
            report = StepReport(outputs=sched.take_extra(), step_id=0, step_tokens=0, runner_ns=0, wait_ns=0)
        if report is not None and emit is not None:
            emit(report)  # timed by the caller (P_SEND)
            report = None
            t1 = clock()
        if planned is not None:
            inf = planned[1]
            n_masks = 0
            if inf.masked is not None:
                n_masks = sched.fill_masks(inf, runner.mask_buffer(handle))
                t2 = clock()
                ph[P_MASKS] += t2 - t1
                t1 = t2
            runner.launch_sample(handle, n_masks)
            ph[P_SAMPLE] += clock() - t1
            self._inflight = (inf, handle, pipelined)
        else:
            self._inflight = None
        return report

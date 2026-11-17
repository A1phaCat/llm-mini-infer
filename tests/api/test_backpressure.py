"""Backpressure from a stalled frontend (architecture.md, "Frontend and engine messages"): the engine numbers each
frontend's StepOutputs, the frontend acknowledges them, and a frontend too far behind gets no new admissions until it
catches up."""

from __future__ import annotations

import shutil
import signal
import time
from collections.abc import Callable

import numpy as np
import zmq

from llm_mini_infer.api.engine_client import ACK_EVERY
from llm_mini_infer.config import Config
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import Ack, AddRequest, StepOutputs
from llm_mini_infer.engine.process import LAG_PAUSE, LAG_RESUME, EngineLoop
from llm_mini_infer.sched import scheduler

from ..support.harness import Harness, Spec, prompt_of


def test_a_paused_frontend_waits_and_resumes() -> None:
    specs = [Spec(rid=i, prompt_len=12, max_tokens=4, seed=10 + i) for i in range(3)]
    h = Harness(specs=[])
    h.specs = {s.rid: s for s in specs}
    h.sched.paused_frontends.add(0)
    for spec in specs:
        h.add(spec)
    for _ in range(5):
        assert h.core.step() is None
    assert h.sched.num_running == 0 and h.sched.num_waiting == 3  # nothing admitted while paused
    h.sched.paused_frontends.clear()
    while h.core.busy:
        rep = h.core.step()
        if rep is not None:
            h.collect(rep)
    h.check()


def _outputs() -> StepOutputs:
    e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)
    return StepOutputs(step=1, rids=e64, n_new=e32, tokens=e32, finish=np.zeros(0, np.uint8), first_rids=e64,
                       first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64, first_runner_ns=e64,
                       t_sent_ns=0)


def test_the_engine_pauses_a_frontend_that_lags_and_resumes_it_on_acks() -> None:
    h = Harness(specs=[])
    cfg = Config()
    cfg.server.num_frontends = 2
    path = ipc.ipc_dir()
    loop = EngineLoop(cfg, EngineCore(h.sched, h.runner), path)
    try:
        ctx = zmq.Context()
        pulls = []
        for i in range(2):
            s = ctx.socket(zmq.PULL)
            s.setsockopt(zmq.RCVTIMEO, 5000)
            s.connect(ipc.frontend_output_addr(path, i))
            pulls.append(s)
        time.sleep(0.3)  # the connections attach asynchronously (a send before would find no peer)
        for _ in range(LAG_PAUSE):
            loop._send(0, _outputs())
            loop._send(1, _outputs())
            loop.on_ack(1, loop.sent_seq[1])  # frontend 1 keeps up
        assert not loop.paused
        loop._send(0, _outputs())
        assert loop.paused == {0} and h.sched.paused_frontends == {0} and loop.pauses == 1
        loop.on_ack(0, loop.sent_seq[0] - LAG_RESUME - 1)
        assert loop.paused == {0}  # not yet within LAG_RESUME
        loop.on_ack(0, loop.sent_seq[0] - LAG_RESUME)
        assert not loop.paused
        dec = ipc.Decoder()
        seqs = [dec.decode(pulls[1].recv_multipart()).seq for _ in range(LAG_PAUSE)]
        assert seqs == list(range(1, LAG_PAUSE + 1))  # numbered from 1, per frontend
        for s in pulls:
            s.close(0)
        ctx.term()
    finally:
        loop.close()
        shutil.rmtree(path, ignore_errors=True)
    assert ACK_EVERY < LAG_RESUME  # a frontend that keeps up never pauses, whatever it has not acknowledged yet
    assert Ack(frontend=1, seq=5).seq == 5


def _add(h: Harness, spec: Spec, frontend: int) -> None:
    h.core.handle(AddRequest(rid=spec.rid, frontend=frontend, prompt=prompt_of(spec, h.vocab),
                             max_tokens=spec.max_tokens, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                             seed=spec.seed, stop_token_ids=[]))


def _run(h: Harness, poll: Callable[[EngineLoop, bool], None]) -> EngineLoop:
    """`EngineLoop.run` over the harness's scheduler, two frontends, no idle spin and no sockets in the way: each poll
    calls `poll(loop, block)` instead; the loop ends at its first blocking poll once nothing is left."""
    cfg = Config()
    cfg.server.num_frontends = 2
    cfg.dev.engine.idle_spin_ms = 0.0
    path = ipc.ipc_dir()
    loop = EngineLoop(cfg, EngineCore(h.sched, h.runner), path)
    loop._send = lambda fe, msg: None  # type: ignore[method-assign]

    def fake(block: bool) -> None:
        poll(loop, block)
        if block and not h.core.busy:
            loop.stopping = True

    loop._poll = fake  # type: ignore[method-assign]
    term = signal.getsignal(signal.SIGTERM)
    try:
        loop.run()
    finally:
        signal.signal(signal.SIGTERM, term)
        loop.close()
        shutil.rmtree(path, ignore_errors=True)
    return loop


def test_a_paused_frontend_blocks_the_idle_loop_until_its_ack() -> None:
    """Every waiting request is a paused frontend's and nothing runs: the loop blocks on its poll (counted), and the
    Ack it brings resumes the requests, which then run to their end."""
    h = Harness(specs=[])
    for i in range(3):
        _add(h, Spec(rid=i, prompt_len=12, max_tokens=4, seed=10 + i), 0)
    h.sched.paused_frontends.add(0)
    polls: list[bool] = []  # whether each poll while the frontend was paused blocked

    def poll(loop: EngineLoop, block: bool) -> None:
        if loop.paused:
            polls.append(block)
            if sum(polls) == 3 or len(polls) == 100:  # the frontend catches up (a spinning loop ends too)
                loop.on_ack(0, loop.sent_seq[0])

    loop = _run(h, poll)
    assert polls == [True, True, True] and loop.waits == 3
    assert not h.sched.reqs and h.sched.step_id > 0  # all three ran to their end
    assert loop._stats(None).backpressure_waits == 3


def test_a_request_deferred_for_a_hit_keeps_the_loop_polling_beside_a_paused_frontend(monkeypatch) -> None:
    """A paused frontend's request waits beside another frontend's request that admission defers for a better hit
    (`Admit.WAIT`): the next turn may admit the latter, so the loop never blocks while it waits."""
    h = Harness(specs=[])
    _add(h, Spec(rid=0, prompt_len=12, max_tokens=4, seed=10), 0)
    _add(h, Spec(rid=1, prompt_len=12, max_tokens=4, seed=11), 1)
    h.sched.paused_frontends.add(0)
    admit, deferred = h.sched._admit_one, []

    def admit_one(req):
        if req.rid == 1 and len(deferred) < 5:
            deferred.append(req.rid)
            return scheduler.Admit.WAIT
        return admit(req)

    monkeypatch.setattr(h.sched, "_admit_one", admit_one)
    polls: list[tuple[bool, bool]] = []  # each poll's block, and whether request 1 was still waiting then

    def poll(loop: EngineLoop, block: bool) -> None:
        waiting = (1, 1) in h.sched.reqs and h.sched.reqs[(1, 1)].status == scheduler.WAITING
        polls.append((block, waiting))
        if (1, 1) not in h.sched.reqs and loop.paused:  # request 1 ran to its end: frontend 0 catches up
            loop.on_ack(0, loop.sent_seq[0])

    loop = _run(h, poll)
    assert len(deferred) == 5 and not h.sched.reqs
    assert sum(1 for _, waiting in polls if waiting) >= 5 and not any(b for b, waiting in polls if waiting)
    assert loop.waits >= 1  # once request 1 was gone, request 0's paused frontend blocked the loop


def test_only_paused_frontends_waiting_requests_let_the_loop_block() -> None:
    """`Scheduler.paused_only`: true while every waiting request is a paused frontend's and nothing runs or is unsent;
    false with a request running, an output unsent, a held sample, another frontend's request, or no pause."""
    h = Harness(specs=[])
    _add(h, Spec(rid=0, prompt_len=12, max_tokens=4, seed=10), 0)
    s = h.sched
    assert not s.paused_only()  # no frontend paused
    s.paused_frontends.add(0)
    assert s.paused_only()
    s._extra.append((0, 9, 0))  # a finish row not sent yet
    assert not s.paused_only()
    s._extra.clear()
    req = s.reqs[(0, 0)]
    req.held = True
    assert not s.paused_only()
    req.held = False
    _add(h, Spec(rid=1, prompt_len=12, max_tokens=4, seed=11), 1)
    assert not s.paused_only()  # frontend 1's request may be admitted
    s.paused_frontends.add(1)
    assert s.paused_only()
    s.paused_frontends.discard(0)
    h.core.step()  # frontend 0's request runs
    assert s.num_running == 1 and s.num_waiting == 1
    assert not s.paused_only()

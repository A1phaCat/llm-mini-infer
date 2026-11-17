"""IPC round trips: every message type, inline and out-of-band array frames, over real ZMQ ipc sockets."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import zmq

from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.messages import Abort, AddRequest, EngineStats, FinishReason, Shutdown, StepOutputs


def make_add(n: int) -> AddRequest:
    return AddRequest(rid=7, frontend=1, prompt=np.arange(n, dtype=np.int32), max_tokens=64, temperature=0.7,
                      top_p=0.95, top_k=20, min_p=0.0, seed=None, stop_token_ids=[5, 6], ignore_eos=True,
                      priority=-1, cache_salt="s", t_recv_ns=123)


def make_outputs() -> StepOutputs:
    stats = EngineStats(running=3, waiting=1, kv_usage=0.5, state_usage=0.25, num_preemptions=2, num_aborts=1,
                        prompt_tokens=100, generation_tokens=20, step_tokens=3, step_cpu_ns=1000,
                        step_runner_calls_ns=10, step_runner_ns=5000, gpu_busy_ns=10, gpu_idle_ns=1)
    return StepOutputs(step=9, rids=np.array([1, 2, 3], np.int64), n_new=np.array([1, 1, 0], np.int32),
                       tokens=np.array([10, 11], np.int32),
                       finish=np.array([0, FinishReason.STOP, FinishReason.ABORT], np.uint8),
                       first_rids=np.array([2], np.int64), first_cached=np.zeros(1, np.int32),
                       first_t_admit_ns=np.array([5], np.int64), first_t_launch_ns=np.array([6], np.int64),
                       first_runner_ns=np.array([7], np.int64), t_sent_ns=42, stats=stats)


def assert_same(a, b) -> None:
    assert type(a) is type(b)
    for f in a.__struct_fields__:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), f
        elif f == "stats" and x is not None:
            assert_same(x, y)
        else:
            assert x == y, f


@pytest.fixture
def pair():
    ctx = zmq.Context()
    path = ipc.ipc_dir()  # pytest's tmp_path is longer than the 104-byte ipc path limit on macOS
    addr = f"ipc://{path}/t.sock"
    pull = ctx.socket(zmq.PULL)
    pull.bind(addr)
    push = ctx.socket(zmq.PUSH)
    push.connect(addr)
    yield push, pull
    push.close(0)
    pull.close(0)
    ctx.term()
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("msg", [make_add(3), make_add(1400), make_add(0), Abort(frontend=0, rids=[1, 2]),
                                 Shutdown(reason="test"), make_outputs()])
def test_round_trip_over_zmq(pair, msg) -> None:
    push, pull = pair
    enc, dec = ipc.Encoder(), ipc.Decoder()
    push.send_multipart(enc.encode(msg), copy=False)
    got = dec.decode(pull.recv_multipart(copy=False))
    assert_same(msg, got)


def test_large_arrays_travel_as_frames_and_decode_zero_copy(pair) -> None:
    push, pull = pair
    enc, dec = ipc.Encoder(inline_threshold=256), ipc.Decoder()
    frames = enc.encode(make_add(1400))
    assert len(frames) == 3  # type, header, prompt
    small = enc.encode(make_add(10))
    assert len(small) == 2  # 40 bytes are inlined
    push.send_multipart(frames, copy=False)
    got = dec.decode(pull.recv_multipart(copy=False))
    assert not got.prompt.flags.owndata  # a view of the received frame
    assert got.prompt[-1] == 1399


def test_non_contiguous_and_numpy_scalars() -> None:
    enc, dec = ipc.Encoder(inline_threshold=8), ipc.Decoder()
    base = np.arange(40, dtype=np.int32)
    msg = make_add(4)
    msg.prompt = base[::2]
    msg.max_tokens = np.int64(9)
    got = dec.decode([bytes(memoryview(f)) for f in enc.encode(msg)])
    assert got.prompt.tolist() == base[::2].tolist() and got.max_tokens == 9

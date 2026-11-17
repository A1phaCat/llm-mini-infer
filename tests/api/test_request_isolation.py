"""One request's failure stays its own (`api.engine_client`): an exception while the frontend processes one request's
outputs finishes that request with an `error` finish (logged once), the step's other requests get every token, and the
engine's message counts as processed (its acknowledgement). A deeply nested call argument raises no exception at all:
the parser keeps its value as written (`output.tool_args.gemma4_parse`'s `per_member`)."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zmq
from tokenizers import Tokenizer

from llm_mini_infer.api.engine_client import EngineClient, RequestState
from llm_mini_infer.api.sse import ChunkTemplate
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.messages import FinishReason, StepOutputs
from llm_mini_infer.metrics import Metrics
from llm_mini_infer.output.detokenizer import Detokenizer
from llm_mini_infer.output.stop import OutputText
from llm_mini_infer.output.tool_calls import GEMMA4, Family, ToolCallParser, ToolFeed

from ..support.fixtures import model_dir

STOP = int(FinishReason.STOP)
E64, E32 = np.zeros(0, np.int64), np.zeros(0, np.int32)


@pytest.fixture(scope="module")
def gemma() -> tuple[Tokenizer, Family]:
    path = model_dir("gemma-4-31b-it") / "tokenizer.json"
    if not path.exists():
        pytest.skip(f"{path} is absent: uv run tests/scripts/fetch_fixtures.py")
    tok = Tokenizer.from_file(str(path))
    return tok, Family(GEMMA4, tok)


class Sent:
    """The client's push socket, recording what it sends."""

    def __init__(self) -> None:
        self.frames: list[Any] = []

    def send_multipart(self, frames: Any, copy: bool = False) -> None:
        self.frames.append(frames)

    def close(self, linger: int = 0) -> None:
        self.closed = True


@pytest.fixture
def client():
    path = ipc.ipc_dir()
    c = EngineClient(path, 0, Metrics("m", 1 << 20))
    real = c.push
    c.push = Sent()
    yield c
    c.push = real
    asyncio.run(c.close())
    shutil.rmtree(path, ignore_errors=True)


def state(client: EngineClient, tok: Tokenizer, rid: int, stream: bool, parser: ToolCallParser | None = None
          ) -> RequestState:
    st = RequestState(client, rid, 4, Detokenizer(tok, [2], skip_special_tokens=parser is None), OutputText(None),
                      ChunkTemplate(f"chatcmpl-{rid}", 0, "m"), stream, True, False, [], 100_000, 0, 0, 0)
    if parser is not None:
        st.tools = ToolFeed(parser)
    client.reqs[rid] = st
    return st


def step(outputs: list[tuple[int, list[int], int]], seq: int = 1) -> StepOutputs:
    """One engine step's message: (rid, tokens, finish) per request, in that order."""
    return StepOutputs(step=seq, rids=np.asarray([r for r, _, _ in outputs], np.int64),
                       n_new=np.asarray([len(t) for _, t, _ in outputs], np.int32),
                       tokens=np.asarray([x for _, t, _ in outputs for x in t], np.int32),
                       finish=np.asarray([f for _, _, f in outputs], np.uint8), first_rids=E64, first_cached=E32,
                       first_t_admit_ns=E64, first_t_launch_ns=E64, first_runner_ns=E64, t_sent_ns=0, seq=seq)


def events(st: RequestState) -> list[Any]:
    return [json.loads(c[6:]) if c.startswith(b"data: {") else c for c in st.pending]


def deep(kind: str, depth: int) -> tuple[str, str]:
    """A Gemma call whose one argument nests `depth` levels deep (objects or arrays), and that argument."""
    value = "{a:" * depth + "1" + "}" * depth if kind == "object" else "[" * depth + "1" + "]" * depth
    return f"<|tool_call>call:f{{x:{value}}}<tool_call|>", value


@pytest.mark.parametrize("kind", ["object", "array"])
@pytest.mark.parametrize("stream", [False, True])
def test_a_deep_argument_and_another_request_in_one_step(gemma, client, kind, stream) -> None:
    """A degenerate call 3,000 levels deep, finishing in the same step as a plain answer behind it: the call ends with
    its argument as written, a string, and the plain answer gets every token."""
    tok, fam = gemma
    parser = ToolCallParser(fam, {"f": {"type": "object", "properties": {"x": {"type": "object"}}}}, "auto", True)
    a = state(client, tok, 1, stream, parser)
    b = state(client, tok, 2, stream)
    text, value = deep(kind, 3000)
    ids_a = [*tok.encode(text, add_special_tokens=False).ids, 50]
    ids_b = [*tok.encode("It is sunny.", add_special_tokens=False).ids, 106]
    client._on_step(step([(1, ids_a, STOP), (2, ids_b, STOP)]))
    assert a.done and b.done and a.finish_reason == b.finish_reason == "stop"
    calls = [(c.name, c.arguments) for c in parser.kept_calls()]
    assert calls == [("f", json.dumps({"x": value}))]
    assert b.n_out == len(ids_b)
    if stream:
        got = "".join(e["choices"][0]["delta"].get("content", "") for e in events(b)
                      if isinstance(e, dict) and e.get("choices"))
        assert got == "It is sunny." and events(b)[-1] == b"data: [DONE]\n\n"
        assert events(a)[-1] == b"data: [DONE]\n\n"
    else:
        assert b.text.take(final=True) == "It is sunny."


class Raises(ToolCallParser):
    """A parser that fails from its third token on."""

    __slots__ = ("n",)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.n = 0

    def feed(self, tid: int, piece: str) -> None:
        self.n += 1
        if self.n >= 3:
            raise RuntimeError("forced")
        super().feed(tid, piece)


@pytest.mark.parametrize("stream", [False, True])
def test_a_forced_exception_finishes_that_request_alone(gemma, client, stream, caplog) -> None:
    """An exception in one request's `on_output` (forced in its parser, which a non-streamed request feeds at its
    finish): that request finishes with `error` and is aborted in the engine, logged once; the request behind it in
    the same step gets every token, and the next step's tokens too."""
    tok, fam = gemma
    a = state(client, tok, 1, stream, Raises(fam, {}, "auto", True))
    b = state(client, tok, 2, stream)
    ids = tok.encode("One two three four five.", add_special_tokens=False).ids
    with caplog.at_level(logging.ERROR, logger="llm_mini_infer.api.engine_client"):
        client._on_step(step([(1, [*ids[:4], 106], STOP), (2, ids[:4], 0)]))
        client._on_step(step([(2, [*ids[4:], 106], STOP)], seq=2))
    assert a.done and a.finish_reason == "error" and 1 not in client.reqs
    assert len(client.push.frames) == 1  # its Abort
    assert len([r for r in caplog.records if "request" in r.getMessage()]) == 1
    assert b.done and b.finish_reason == "stop" and b.n_out == len(ids) + 1
    if stream:
        assert any(isinstance(e, dict) and "error" in e for e in events(a)) and events(a)[-1] == b"data: [DONE]\n\n"
        got = "".join(e["choices"][0]["delta"].get("content", "") for e in events(b)
                      if isinstance(e, dict) and e.get("choices"))
        assert got == "One two three four five."
    else:
        assert b.text.take(final=True) == "One two three four five."


class FinalTextFails(OutputText):
    """Output text whose final take raises (the stop strings' or the text's own state broken)."""

    def take(self, final: bool = False) -> str:
        if final:
            raise RuntimeError("forced final take")
        return super().take(final)


class FinishFails(RequestState):
    """A request whose `finish` always raises, the error finish included."""

    __slots__ = ()

    def finish(self, reason: str, stop_reason: str | int | None, now: int = 0) -> None:
        raise RuntimeError("forced finish")


def streamed_text(st: RequestState) -> str:
    return "".join(e["choices"][0]["delta"].get("content", "") for e in events(st)
                   if isinstance(e, dict) and e.get("choices"))


@pytest.mark.parametrize("probe", ["final-text", "finish"])
@pytest.mark.parametrize("stream", [False, True])
def test_the_crash_path_cannot_fail(gemma, client, probe, stream) -> None:
    """The request's own output state fails again in the crash path (a stream's final text take, or its whole
    `finish`): it still ends with an error, leaves `client.reqs`, and its waiter wakes (a stream gets its error event
    and `[DONE]`), and the request behind it in the same step gets every token."""
    tok, _ = gemma
    if probe == "final-text" and not stream:
        pytest.skip("a non-streamed finish takes no text here: the handler takes it")
    if probe == "finish":
        a = FinishFails(client, 1, 4, Detokenizer(tok, [2], skip_special_tokens=True), OutputText(None),
                        ChunkTemplate("chatcmpl-1", 0, "m"), stream, True, False, [], 100_000, 0, 0, 0)
        client.reqs[1] = a
    else:
        a = state(client, tok, 1, stream)
        a.text = FinalTextFails(None)
    b = state(client, tok, 2, stream)
    real, calls = client.on_finished, []
    client.on_finished = lambda st: (calls.append(st.rid), real(st))  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        a.waiter = loop.create_future()
        ids = tok.encode("One two three.", add_special_tokens=False).ids
        client._on_step(step([(1, [*ids, 106], STOP), (2, [*ids, 106], STOP)]))
        assert a.waiter.done()
    finally:
        loop.close()
    assert a.done and a.finish_reason == "error" and 1 not in client.reqs and a.error_status == 500
    assert b.done and b.finish_reason == "stop" and b.n_out == len(ids) + 1 and 2 not in client.reqs
    assert a.counted  # its error finish counted by `on_finished`, by the fallback when the finish failed before it
    assert calls == [1, 2]  # once each
    if stream:
        assert any(isinstance(e, dict) and "error" in e for e in events(a)) and events(a)[-1] == b"data: [DONE]\n\n"
        assert streamed_text(b) == "One two three." and events(b)[-1] == b"data: [DONE]\n\n"
    else:
        assert b.text.take(final=True) == "One two three."


class AppendFails(OutputText):
    """Output text that takes no piece (the stop strings' state broken)."""

    def append(self, piece: str) -> str | None:
        raise RuntimeError("forced append")


@pytest.mark.parametrize("stream", [False, True])
def test_the_crash_path_adds_only_what_the_finish_left(gemma, client, stream) -> None:
    """The request's output fails, and then its error finish (on a stream: after queueing the error event and
    `[DONE]`) fails in `on_finished`: the crash path neither queues them again nor runs `on_finished` again, and still
    ends the request, out of `client.reqs`, with its waiter woken."""
    tok, _ = gemma
    a, b = state(client, tok, 1, stream), state(client, tok, 2, stream)
    a.text = AppendFails(None)
    real, calls = client.on_finished, []

    def on_finished(st: RequestState) -> None:
        calls.append(st.rid)
        real(st)
        if st.rid == 1:
            raise RuntimeError("forced on_finished")

    client.on_finished = on_finished  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        a.waiter = loop.create_future()
        ids = tok.encode("One two three.", add_special_tokens=False).ids
        client._on_step(step([(1, [*ids, 106], STOP), (2, [*ids, 106], STOP)]))
        assert a.waiter.done()
    finally:
        loop.close()
    assert calls == [1, 2]
    assert a.done and a.finish_reason == "error" and a.error_status == 500 and 1 not in client.reqs
    assert b.done and b.finish_reason == "stop" and b.n_out == len(ids) + 1 and 2 not in client.reqs
    if not stream:
        assert not a.pending and b.text.take(final=True) == "One two three."
        return
    ev = events(a)
    assert len([e for e in ev if isinstance(e, dict) and "error" in e]) == 1
    assert ev.count(b"data: [DONE]\n\n") == 1 and ev[-1] == b"data: [DONE]\n\n"
    assert streamed_text(b) == "One two three." and events(b)[-1] == b"data: [DONE]\n\n"


@pytest.mark.parametrize("left", [True, False], ids=["client-left", "server-stops"])
def test_a_cancel_whose_on_finished_raises_still_wakes_the_request(gemma, client, left) -> None:
    """A cancel (the client left, or the server stops) whose `on_finished` raises: the request is done and its waiter
    still wakes, as a finish's."""
    tok, _ = gemma
    a = state(client, tok, 1, False)
    real = client.on_finished

    def on_finished(st: RequestState) -> None:
        real(st)
        raise RuntimeError("forced on_finished")

    client.on_finished = on_finished  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        a.waiter = loop.create_future()
        with pytest.raises(RuntimeError, match="forced on_finished"):
            a.cancel(left)
        assert a.waiter.done()
    finally:
        loop.close()
    assert a.done and a.cancelled and a.finish_reason == "abort"


@pytest.mark.parametrize("left", [True, False])
def test_a_cancel_counts_its_request_once(gemma, left) -> None:
    """A cancel (the client left, or the server stops) ends the request as `RequestState`'s table says: `on_finished`
    called once and the request counted, as a finish counts it; a second cancel (the watcher, then a stream's failed
    send) changes nothing."""
    tok, _ = gemma
    finished: list[RequestState] = []
    aborts: list[list[int]] = []
    client = SimpleNamespace(on_finished=finished.append, abort=lambda rids, reason: aborts.append(rids))
    st = RequestState(client, 1, 4, Detokenizer(tok, [2]), OutputText(None), ChunkTemplate("c", 0, "m"), True, False,
                      False, [], 100, 0, 0, 0)
    st.cancel(left)
    st.cancel(left)
    assert st.counted and finished == [st] and aborts == [[1]]
    assert (st.done, st.cancelled, st.left, st.finish_reason) == (True, True, left, "abort")


@pytest.mark.parametrize("stream", [False, True])
def test_a_finish_whose_on_finished_raises_still_wakes_the_request(gemma, client, stream) -> None:
    """A normal finish whose `on_finished` raises (its metrics, say): `on_finished` took the request out of
    `client.reqs` first, so the crash path does not run for it; its waiter still wakes, or a non-streamed response
    would never return."""
    tok, _ = gemma
    a = state(client, tok, 1, stream)
    real = client.on_finished

    def on_finished(st: RequestState) -> None:
        real(st)
        raise RuntimeError("forced on_finished")

    client.on_finished = on_finished  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        a.waiter = loop.create_future()
        ids = tok.encode("One two.", add_special_tokens=False).ids
        client._on_step(step([(1, [*ids, 106], STOP)]))
        assert a.waiter.done()
    finally:
        loop.close()
    assert a.done and a.finish_reason == "stop" and 1 not in client.reqs


class Pull:
    """The client's pull socket, holding decoded messages (`Dec` passes them through)."""

    def __init__(self, msgs: list[StepOutputs]) -> None:
        self.msgs = list(msgs)

    def get(self, option: int) -> int:
        return zmq.POLLIN if self.msgs else 0

    def recv_multipart(self, flags: int = 0, copy: bool = True) -> StepOutputs:
        return self.msgs.pop(0)


class Dec:
    @staticmethod
    def decode(frames: Any) -> Any:
        return frames


def test_the_message_counts_as_processed(gemma, client) -> None:
    """The message whose processing failed for one request is still acknowledged in sequence: `_seq` advances, and
    the eighth message sends the Ack."""
    tok, fam = gemma
    state(client, tok, 1, True, Raises(fam, {}, "auto", True))  # streamed: each step feeds the parser
    ids = tok.encode("One two three four five.", add_special_tokens=False).ids
    real_pull, real_dec = client.pull, client.dec
    client.pull, client.dec = Pull([step([(1, ids[:4], 0)], seq=s) for s in range(1, 9)]), Dec()
    try:
        client._on_readable()
    finally:
        client.pull, client.dec = real_pull, real_dec
    assert client._seq == 8 and client._acked == 8


@pytest.mark.parametrize("part", ["stats", "release"])
def test_a_failed_stats_or_release_loses_no_tokens(gemma, client, caplog, part) -> None:
    """The engine's statistics or the pixel release raising while the frontend processes a step: every request of
    the step gets its tokens and finish, the other part still runs, and the failure is logged and counted."""
    tok, _ = gemma
    a, b = state(client, tok, 1, False), state(client, tok, 2, True)
    released: list[list[int]] = []
    seen: list[object] = []

    def on_released(rids: list[int]) -> None:
        if part == "release":
            raise RuntimeError("forced release")
        released.append(list(rids))

    def on_stats(s: object) -> None:
        if part == "stats":
            raise RuntimeError("forced stats")
        seen.append(s)

    client.on_released = on_released
    client._on_stats = on_stats  # type: ignore[method-assign]
    ids = [*tok.encode("It is sunny.", add_special_tokens=False).ids, 106]
    msg = step([(1, ids, STOP), (2, ids, STOP)])
    msg.stats, msg.released = object(), [1, 2]
    pull, dec = client.pull, client.dec
    client.pull, client.dec = Pull([msg]), Dec()
    try:
        with caplog.at_level(logging.ERROR, logger="llm_mini_infer.api.engine_client"):
            client._on_readable()
    finally:
        client.pull, client.dec = pull, dec
    assert a.done and b.done and a.finish_reason == b.finish_reason == "stop" and not client.reqs
    assert a.text.take(final=True) == "It is sunny." and streamed_text(b) == "It is sunny."
    assert released == ([] if part == "release" else [[1, 2]])
    assert seen == ([] if part == "stats" else [msg.stats])
    assert client.metrics.step_output_errors.values == {(part,): 1.0}
    assert [r for r in caplog.records if r.exc_info and f"forced {part}" in str(r.exc_info[1])]
    assert client._seq == 1


def raising_on_finished(client: EngineClient, rid: int) -> None:
    """`client.on_finished` raises after doing its work, for request `rid`."""
    real = client.on_finished

    def on_finished(st: RequestState) -> None:
        real(st)
        if st.rid == rid:
            raise RuntimeError("forced on_finished")

    client.on_finished = on_finished  # type: ignore[method-assign]


def test_close_cancels_every_request_past_a_cancel_that_raises(gemma, caplog) -> None:
    """`close` cancels and wakes every request, though the first one's `on_finished` raises; the exception is logged,
    and the sockets close."""
    tok, _ = gemma
    path = ipc.ipc_dir()
    c = EngineClient(path, 0, Metrics("m", 1 << 20))
    c.push.close(0)
    c.push = Sent()
    try:
        a, b = state(c, tok, 1, False), state(c, tok, 2, False)
        raising_on_finished(c, 1)
        loop = asyncio.new_event_loop()
        try:
            a.waiter, b.waiter = loop.create_future(), loop.create_future()
            with caplog.at_level(logging.ERROR, logger="llm_mini_infer.api.engine_client"):
                loop.run_until_complete(c.close())
            assert a.waiter.done() and b.waiter.done()
        finally:
            loop.close()
        assert a.cancelled and b.cancelled and b.finish_reason == "abort" and not c.reqs
        assert c.push.closed and c.pull.closed and c._sctx.closed
        assert [r for r in caplog.records if r.exc_info and "forced on_finished" in str(r.exc_info[1])]
    finally:  # what a failed close left open (its context would hold the process at exit)
        if not c.pull.closed:
            c.pull.close(0)
        if not c._sctx.closed:
            c._sctx.term()
        shutil.rmtree(path, ignore_errors=True)


def test_close_closes_its_sockets_when_a_cancel_raises_past_exception(gemma) -> None:
    """A cancel that raises a `BaseException` (KeyboardInterrupt) ends `close`, but its sockets and context still
    close."""
    tok, _ = gemma
    path = ipc.ipc_dir()
    c = EngineClient(path, 0, Metrics("m", 1 << 20))
    c.push.close(0)
    c.push = Sent()
    try:
        state(c, tok, 1, False)
        real = c.on_finished

        def on_finished(st: RequestState) -> None:
            real(st)
            raise KeyboardInterrupt

        c.on_finished = on_finished  # type: ignore[method-assign]
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(c.close())
        assert c.push.closed and c.pull.closed and c._sctx.closed
    finally:
        if not c.pull.closed:
            c.pull.close(0)
        if not c._sctx.closed:
            c._sctx.term()
        shutil.rmtree(path, ignore_errors=True)


def test_a_disconnect_whose_cancel_raises_is_logged(gemma, client, caplog) -> None:
    """The client left and the cancel's `on_finished` raised: the request is cancelled and woken, and the exception
    `App.watch_disconnect` swallows is logged."""
    from llm_mini_infer.api.app import App, Connection

    tok, _ = gemma
    a = state(client, tok, 1, False)
    raising_on_finished(client, 1)
    conn = Connection()
    conn.st = a

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    loop = asyncio.new_event_loop()
    try:
        a.waiter = loop.create_future()
        with caplog.at_level(logging.ERROR, logger="llm_mini_infer.api.app"):
            loop.run_until_complete(App.watch_disconnect(receive, conn))
        assert a.waiter.done()
    finally:
        loop.close()
    assert a.done and a.cancelled
    assert [r for r in caplog.records if r.exc_info and "forced on_finished" in str(r.exc_info[1])]

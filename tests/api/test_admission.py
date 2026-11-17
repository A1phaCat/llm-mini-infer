"""vLLM's queue caps (`scheduler.max_num_queued_reqs`, `max_num_queued_tokens`; `App.admit`) through the ASGI
application in process, its engine client's socket stubbed so that nothing answers and a request stays in flight until
the test ends it: the 503s and their counter, the order of the checks, the request that never reaches the engine, the
frontend's backlog count (`EngineClient.queued_tokens`) after every way a request ends, the settings, and the gauge of
chat requests in flight."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from llm_mini_infer import cli, config
from llm_mini_infer.api.app import App
from llm_mini_infer.api.engine_client import RequestState
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.messages import ABORT_STOP, FinishReason, StepOutputs
from llm_mini_infer.grammar.spec import Pending

from ..support.fixtures import require_qwen

CHAT = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
METRICS = {"type": "http", "path": "/metrics", "method": "GET", "headers": []}
BODY = {"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 4}
REQS = ('{"error":{"message":"The engine is currently busy and cannot accept new requests. Please try again later or '
        'on a different instance.","type":"Service Unavailable","param":null,"code":503}}')
TOKENS = ('{"error":{"message":"The engine has reached its prefill token backlog limit. Please try again later or on '
          'a different instance.","type":"Service Unavailable","param":null,"code":503}}')
E64, E32 = np.zeros(0, np.int64), np.zeros(0, np.int32)


class Sent:
    """The client's push socket: nothing is sent anywhere."""

    def send_multipart(self, frames: Any, copy: bool = False) -> None:
        pass

    def close(self, linger: int = 0) -> None:
        pass


class Peer:
    """The HTTP client: its body at once, then nothing until it leaves."""

    def __init__(self, body: dict | bytes, scope: dict = CHAT) -> None:
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.scope = scope
        self.read = False
        self.gone = asyncio.Event()
        self.sent: list[dict] = []

    async def receive(self) -> dict:
        if not self.read:
            self.read = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def send(self, msg: dict) -> None:
        self.sent.append(msg)

    def status(self) -> int:
        return next(m["status"] for m in self.sent if m["type"] == "http.response.start")

    def raw(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.sent if m["type"] == "http.response.body")

    def content_type(self) -> bytes:
        start = next(m for m in self.sent if m["type"] == "http.response.start")
        return dict(start["headers"])[b"content-type"]


class Rig:
    """An application over a stubbed client, and the requests it was sent (`adds`)."""

    def __init__(self, cfg: config.Config) -> None:
        self.path = ipc.ipc_dir()
        self.app = App(cfg, 0, self.path)
        self.client = self.app.client
        self.client.push = Sent()  # type: ignore[assignment]
        self.adds: list[Any] = []
        add = self.client.add
        self.client.add = lambda st, msg: (self.adds.append(msg), add(st, msg))[1]  # type: ignore[method-assign]
        self.tasks: list[tuple[asyncio.Task, Peer]] = []

    async def post(self, body: dict | bytes, hold: bool = False) -> Peer:
        """A chat request to completion, or (`hold`) until it is in the client's `reqs` and in flight."""
        peer = Peer(body)
        task = asyncio.create_task(self.app.http(peer.scope, peer.receive, peer.send))
        if not hold:
            await asyncio.wait_for(task, 10)
            return peer
        n = len(self.client.reqs)
        async with asyncio.timeout(10):
            while len(self.client.reqs) == n and not task.done():
                await asyncio.sleep(0.005)
        assert not task.done() and len(self.client.reqs) == n + 1
        self.tasks.append((task, peer))
        return peer

    async def metrics(self) -> dict[str, float]:
        """The /metrics series of the two new names, by their text up to the value."""
        peer = Peer(b"", METRICS)
        await asyncio.wait_for(self.app.http(peer.scope, peer.receive, peer.send), 10)
        found = {}
        for line in peer.raw().decode().splitlines():
            if re.match(r"(vllm:admission_rejections_total|llm_mini_infer:chat_requests_inflight)\{", line):
                series, value = line.rsplit(" ", 1)
                found[series] = float(value)
        return found

    async def rejected(self) -> tuple[float, float]:
        got = await self.metrics()
        label = f'{{model_name="{self.app.model_name}",reason="%s"}}'
        return tuple(got["vllm:admission_rejections_total" + label % r]  # type: ignore[return-value]
                     for r in ("max_num_queued_reqs", "max_num_queued_tokens"))

    def check_sum(self) -> None:
        """The backlog count is the prompts of the requests still marked queued."""
        reqs = self.client.reqs.values()
        assert self.client.queued_tokens == sum(st.prompt_len for st in reqs if st.queued)

    async def end_all(self) -> None:
        """Every held request finishes; its task ends; the client is empty."""
        for st in list(self.client.reqs.values()):
            st.finish("stop", None)
        for task, peer in self.tasks:
            await asyncio.wait_for(task, 10)
            peer.gone.set()
        self.tasks.clear()
        assert not self.client.reqs and self.client.queued_tokens == 0

    def close(self) -> None:
        asyncio.run(self.client.close())
        if self.app.grammars is not None:  # its compile threads, which would outlive it
            self.app.grammars.threads.close()
        shutil.rmtree(self.path, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def one_renderer() -> Iterator[None]:
    """One renderer for the file's applications: each would load the tokenizer again (seconds, and hundreds of MB
    a worker keeps)."""
    from llm_mini_infer.api import app as app_module

    made: dict[tuple, Any] = {}
    cls = app_module.Renderer

    def renderer(*args: Any) -> Any:
        if args not in made:
            made[args] = cls(*args)
        return made[args]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "Renderer", renderer)
        yield


@pytest.fixture
def rig() -> Iterator[Callable[..., Rig]]:
    made: list[Rig] = []

    def make(**scheduler: Any) -> Rig:
        sets = {"model.path": str(require_qwen()), **{f"scheduler.{k}": v for k, v in scheduler.items()}}
        made.append(Rig(config.derive(config.load(None, sets))))
        return made[-1]

    yield make
    for r in made:
        r.close()


def err(peer: Peer) -> str:
    return peer.raw().decode()


def test_the_request_cap_refuses_the_request_past_it(rig) -> None:
    """With max_num_queued_reqs = 2 the third concurrent request gets vLLM's 503 (exact body, plain JSON), the counter's
    requests series moves by one and the tokens series stays 0; the engine got two requests only; a slot freed admits
    the next."""
    r = rig(max_num_queued_reqs=2)

    async def main() -> None:
        assert await r.rejected() == (0.0, 0.0)
        await r.post(BODY, hold=True)
        await r.post(BODY, hold=True)
        third = await r.post(BODY)
        assert third.status() == 503 and err(third) == REQS and third.content_type() == b"application/json"
        assert len(r.adds) == 2 and len(r.client.reqs) == 2 and await r.rejected() == (1.0, 0.0)
        next(iter(r.client.reqs.values())).finish("stop", None)
        await r.post(BODY, hold=True)  # admitted again
        assert len(r.adds) == 3 and await r.rejected() == (1.0, 0.0)
        await r.end_all()

    asyncio.run(main())


def test_the_token_cap_looks_at_the_backlog_before_the_new_prompt(rig) -> None:
    """With max_num_queued_tokens = P + 1 (P: a prompt's tokens): a request arriving at backlog 0 and one at backlog P
    (just below the cap) are admitted, though the second takes the backlog to 2P; the next, at 2P, gets the tokens 503
    (exact body, series moved by one) and never reaches the engine. A first output frees its prompt from the backlog,
    admitting a request again."""
    probe = rig()

    async def prompt_len() -> int:
        await probe.post(BODY, hold=True)
        n = probe.client.queued_tokens
        await probe.end_all()
        return n

    p = asyncio.run(prompt_len())
    assert p > 1
    r = rig(max_num_queued_tokens=p + 1)

    async def main() -> None:
        await r.post(BODY, hold=True)
        assert r.client.queued_tokens == p
        await r.post(BODY, hold=True)
        assert r.client.queued_tokens == 2 * p and len(r.adds) == 2
        refused = await r.post(BODY)
        assert refused.status() == 503 and err(refused) == TOKENS and refused.content_type() == b"application/json"
        assert len(r.adds) == 2 and await r.rejected() == (0.0, 1.0)
        first = next(iter(r.client.reqs.values()))
        r.client._on_step(output(first.rid, [], 0, first=True))  # its first output: p tokens out of the backlog
        assert r.client.queued_tokens == p and not first.queued
        await r.post(BODY, hold=True)  # backlog p < p + 1: admitted
        assert len(r.adds) == 3 and await r.rejected() == (0.0, 1.0)
        await r.end_all()

    asyncio.run(main())


def test_the_request_cap_is_checked_before_the_token_cap(rig) -> None:
    """Both caps exceeded at once (requests 1, tokens 1, one request held): the requests text and series move; the
    tokens series does not."""
    r = rig(max_num_queued_reqs=1, max_num_queued_tokens=1)

    async def main() -> None:
        await r.post(BODY, hold=True)
        refused = await r.post(BODY)
        assert refused.status() == 503 and err(refused) == REQS and await r.rejected() == (1.0, 0.0)
        await r.end_all()

    asyncio.run(main())


def test_the_caps_are_checked_again_once_a_request_is_prepared(rig, monkeypatch) -> None:
    """A request that passed the first check with room left, then waited in its preparation (a grammar build, held
    here) while another filled the cap, is refused at the second check, as vLLM's add refuses it: the same plain 503,
    the requests series moved by one, never sent to the engine. With room left at the second check, it is sent."""
    r = rig(max_num_queued_reqs=1)
    app = r.app

    async def main() -> None:
        gate, building = asyncio.Event(), asyncio.Event()

        def grammar(req: Any, tools: Any = None, case: str = "a", warn: bool = False) -> str:
            if req.max_tokens == 5:  # the request held in its preparation
                raise Pending(b"held", lambda: "")
            return ""

        async def grammar_built(p: Any, req: Any, tools: Any, case: str = "a") -> str:
            building.set()
            await gate.wait()
            return ""

        monkeypatch.setattr(app, "grammar", grammar)
        monkeypatch.setattr(app, "grammar_built", grammar_built)
        held = Peer({**BODY, "max_tokens": 5})
        task = asyncio.create_task(app.http(held.scope, held.receive, held.send))
        await asyncio.wait_for(building.wait(), 10)  # past the first check, preparing
        await r.post(BODY, hold=True)  # fills the cap meanwhile
        gate.set()
        await asyncio.wait_for(task, 10)
        assert held.status() == 503 and err(held) == REQS and held.content_type() == b"application/json"
        assert len(r.adds) == 1 and await r.rejected() == (1.0, 0.0)
        await r.end_all()
        await r.post({**BODY, "max_tokens": 5}, hold=True)  # room left at both checks: sent
        assert len(r.adds) == 2 and await r.rejected() == (1.0, 0.0)
        await r.end_all()

    asyncio.run(main())


def test_a_full_server_answers_503_before_the_chat_template_check_and_after_the_body_checks(rig) -> None:
    """With the server full: a request with its own chat_template gets the 503 (not the untrusted-template 400), a
    malformed body (no messages) still gets its 400, and a request to an unknown model its 404; only the first moves
    the counter."""
    r = rig(max_num_queued_reqs=1)

    async def main() -> None:
        await r.post(BODY, hold=True)
        templated = await r.post({**BODY, "chat_template": "{{ messages }}"})
        assert templated.status() == 503 and err(templated) == REQS
        malformed = await r.post({"messages": []})
        assert malformed.status() == 400
        unknown = await r.post({**BODY, "model": "no-such-model"})
        assert unknown.status() == 404
        assert await r.rejected() == (1.0, 0.0) and len(r.adds) == 1
        free = rig()  # the same template request on a server with room: the 400 this one would have got
        templated = await free.post({**BODY, "chat_template": "{{ messages }}"})
        assert templated.status() == 400
        await r.end_all()

    asyncio.run(main())


@pytest.mark.parametrize("cap", ["max_num_queued_reqs", "max_num_queued_tokens"])
def test_a_streaming_request_refused_by_a_cap_gets_a_plain_503(rig, cap) -> None:
    """A stream refused by either cap: one JSON 503 response, no event stream, no SSE byte, nothing sent on."""
    r = rig(**{cap: 0})

    async def main() -> None:
        peer = await r.post({**BODY, "stream": True})
        assert peer.status() == 503 and peer.content_type() == b"application/json"
        assert err(peer) == (REQS if cap == "max_num_queued_reqs" else TOKENS)
        assert b"data:" not in peer.raw() and len(peer.sent) == 2 and r.adds == []
        assert await r.rejected() == ((1.0, 0.0) if cap == "max_num_queued_reqs" else (0.0, 1.0))

    asyncio.run(main())


def test_a_cap_of_zero_refuses_every_request_and_unset_caps_refuse_none(rig) -> None:
    """Requests cap 0: every request is refused (as vLLM's), the counter counting each. Unset (the default): eight
    concurrent requests are all admitted, and /metrics shows both series at 0 with the model_name and reason labels
    only."""
    zero = rig(max_num_queued_reqs=0)
    unset = rig()

    async def main() -> None:
        for _ in range(3):
            assert (await zero.post(BODY)).status() == 503
        assert await zero.rejected() == (3.0, 0.0) and zero.adds == []
        for _ in range(8):
            await unset.post(BODY, hold=True)
        assert len(unset.adds) == 8 and len(unset.client.reqs) == 8
        series = {k: v for k, v in (await unset.metrics()).items() if k.startswith("vllm:admission")}
        name = unset.app.model_name
        assert series == {f'vllm:admission_rejections_total{{model_name="{name}",reason="max_num_queued_reqs"}}': 0.0,
                          f'vllm:admission_rejections_total{{model_name="{name}",reason="max_num_queued_tokens"}}': 0.0}
        await unset.end_all()

    asyncio.run(main())


def output(rid: int, tokens: list[int], finish: int, first: bool = False) -> StepOutputs:
    """A step's message for one request: its tokens and finish; with `first`, its first output."""
    one = np.asarray([0], np.int64)
    return StepOutputs(step=1, rids=np.asarray([rid], np.int64), n_new=np.asarray([len(tokens)], np.int32),
                       tokens=np.asarray(tokens, np.int32), finish=np.asarray([finish], np.uint8),
                       first_rids=np.asarray([rid], np.int64) if first else E64,
                       first_cached=np.zeros(1, np.int32) if first else E32,
                       first_t_admit_ns=one if first else E64, first_t_launch_ns=one if first else E64,
                       first_runner_ns=one if first else E64, t_sent_ns=0, seq=1)


def end_finish(r: Rig, st: RequestState) -> None:
    r.client._on_step(output(st.rid, [1000], int(FinishReason.STOP)))


def end_abort_then_finish(r: Rig, st: RequestState) -> None:
    r.client.abort([st.rid])
    st.finish("stop", None)


def end_stop_string(r: Rig, st: RequestState) -> None:
    r.client.abort([st.rid], ABORT_STOP)
    st.finish("stop", None)


def end_crash(r: Rig, st: RequestState) -> None:
    st.crash()


def end_cancel(r: Rig, st: RequestState) -> None:
    st.cancel()


def end_fail(r: Rig, st: RequestState) -> None:
    st.fail("gone")


ENDINGS = [end_finish, end_abort_then_finish, end_stop_string, end_crash, end_cancel, end_fail]


@pytest.mark.parametrize("first", [False, True], ids=["no_output", "after_first_output"])
@pytest.mark.parametrize("ending", ENDINGS, ids=lambda f: f.__name__)
def test_the_backlog_is_the_prompts_still_queued_after_every_ending(rig, ending, first) -> None:
    """Three requests in flight, one ended each way (a finish, an abort and its finish, a stop string's abort and its
    finish, a crash, a cancel, a fail), with or without a first output before: the backlog equals the sum of the prompts
    marked queued at every point, the others' prompts stay counted, and when all end it is 0 with `reqs` empty."""
    r = rig()

    async def main() -> None:
        for _ in range(3):
            await r.post(BODY, hold=True)
        p = r.client.queued_tokens // 3
        r.check_sum()
        assert r.client.queued_tokens == 3 * p
        st = next(iter(r.client.reqs.values()))
        if first:
            r.client._on_step(output(st.rid, [1000], 0, first=True))
            assert r.client.queued_tokens == 2 * p
            r.check_sum()
        ending(r, st)
        assert st.rid not in r.client.reqs and not st.queued
        assert r.client.queued_tokens == 2 * p and len(r.client.reqs) == 2
        r.check_sum()
        if ending is end_finish:  # a second finish of an ended request counts nothing twice
            st.finish("stop", None)
            r.client.abort([st.rid])
            assert r.client.queued_tokens == 2 * p
        await r.end_all()
        r.check_sum()

    asyncio.run(main())


def test_the_settings_refuse_a_negative_cap_and_the_flags_reach_them() -> None:
    """A negative cap is refused with its message; 0 and unset are valid; unset by default; the CLI flags set both."""
    for key in ("max_num_queued_reqs", "max_num_queued_tokens"):
        with pytest.raises(ValueError, match=re.escape(f"scheduler.{key} -1: 0 or more, or unset (no limit)")):
            config.derive(config.load(None, {"model.path": str(require_qwen()), f"scheduler.{key}": -1}))
        cfg = config.derive(config.load(None, {"model.path": str(require_qwen()), f"scheduler.{key}": 0}))
        assert getattr(cfg.scheduler, key) == 0
    cfg = config.derive(config.load(None, {"model.path": str(require_qwen())}))
    assert (cfg.scheduler.max_num_queued_reqs, cfg.scheduler.max_num_queued_tokens) == (None, None)
    parser = cli.serve_args(argparse.ArgumentParser())
    got = cli.overrides(parser.parse_args(["--max-num-queued-reqs", "7", "--max-num-queued-tokens", "9000"]))
    assert got == {"scheduler.max_num_queued_reqs": 7, "scheduler.max_num_queued_tokens": 9000}
    assert not any(k.startswith("scheduler.max_num_queued") for k in cli.overrides(parser.parse_args([])))


def test_the_inflight_gauge_counts_a_request_until_its_response_ends(rig) -> None:
    """A stream held mid-flight shows llm_mini_infer:chat_requests_inflight 1 with its model_name and engine labels;
    after it ends 0; a scrape and a request refused by a cap are not counted."""
    r = rig(max_num_queued_reqs=1)
    series = f'llm_mini_infer:chat_requests_inflight{{model_name="{r.app.model_name}",engine="0"}}'

    async def main() -> None:
        assert (await r.metrics())[series] == 0.0
        await r.post({**BODY, "stream": True}, hold=True)
        assert (await r.metrics())[series] == 1.0
        assert (await r.post(BODY)).status() == 503
        assert (await r.metrics())[series] == 1.0
        await r.end_all()
        assert (await r.metrics())[series] == 0.0

    asyncio.run(main())

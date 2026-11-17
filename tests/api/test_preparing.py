"""A chat request still being prepared (its body read, its grammar built, its images fetched) belongs to the frontend's
drain and stop, through the ASGI application in process with the engine stubbed: a drain waits for it, a drain's
timeout and `stop_now` fail it with a 503 at its next checkpoint, before it reaches the engine, and a client that
leaves meanwhile is never sent to the engine. A request leaving at a checkpoint discards its images. The frontend's mode
(`App.mode`): its edges, and the stop signals in each order, as the launcher sends them."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from PIL import Image

from llm_mini_infer import config
from llm_mini_infer.api.app import STOPPED, App, Mode
from llm_mini_infer.engine import ipc
from llm_mini_infer.grammar.spec import Pending

from ..support.fixtures import require_qwen

SCOPE = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}


@pytest.fixture
def make_app() -> Iterator[Callable[[float], App]]:
    made: list[tuple[App, str]] = []

    def make(drain_timeout_s: float) -> App:
        cfg = config.derive(config.load(None, {"model.path": str(require_qwen()),
                                               "server.drain_timeout_s": drain_timeout_s}))
        path = ipc.ipc_dir()
        app = App(cfg, 0, path)
        made.append((app, path))
        return app

    yield make
    for app, path in made:
        asyncio.run(app.client.close())
        shutil.rmtree(path, ignore_errors=True)


def png() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Gate:
    """An await that holds until `go`; `entered` once it was reached."""

    def __init__(self) -> None:
        self.entered, self.go = asyncio.Event(), asyncio.Event()

    async def __call__(self, *args: Any) -> str:
        self.entered.set()
        await self.go.wait()
        return ""  # a grammar build's text: none


class Peer:
    """The HTTP client: its body (held until `gate` opens, for "body"), then nothing until it leaves."""

    def __init__(self, body: dict, gate: Gate | None = None) -> None:
        self.body, self.gate = json.dumps(body).encode(), gate
        self.read = False
        self.gone = asyncio.Event()
        self.sent: list[dict] = []

    async def receive(self) -> dict:
        if not self.read:
            self.read = True
            if self.gate is not None:
                await self.gate()
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def send(self, msg: dict) -> None:
        self.sent.append(msg)

    def status(self) -> int | None:
        return next((m["status"] for m in self.sent if m["type"] == "http.response.start"), None)

    def json(self) -> dict:
        return json.loads(b"".join(m.get("body", b"") for m in self.sent if m["type"] == "http.response.body"))


def stub(app: App, where: str, stream: bool = False) -> tuple[Gate, Peer, list, list]:
    """The request held at `where`: its body (body), its grammar's build (grammar), the second build a thought in front
    takes (thinking), its images' fetch (images) or the pool (wait). Returns its gate, its client, the requests the
    engine got (each finished at once) and the image counts of each discarded `Prepared`."""
    gate = Gate()
    body: dict = {"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 4, "stream": stream,
                  "chat_template_kwargs": {"enable_thinking": where == "thinking"}}
    if where in ("grammar", "thinking"):
        def grammar(req: Any, tools: Any = None, case: str = "a", warn: bool = False) -> str:
            if where == "grammar" or case in ("b", "c"):
                raise Pending(b"key", lambda: "")
            return "the thinking-off grammar"  # built: the thought in front is built next

        app.grammar = grammar  # type: ignore[method-assign]
        app.grammar_built = gate  # type: ignore[method-assign]
    discarded: list[int] = []
    if where in ("images", "wait"):
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                                         {"type": "image_url", "image_url": {"url": png()}}]}]
        images = app.images
        prepare, wait, discard = images.prepare, images.wait, images.discard

        async def held(urls: list[str], submit: bool = True) -> Any:
            await gate()
            return await prepare(urls, submit)

        async def held_wait(prep: Any) -> None:
            await gate()
            await wait(prep)

        if where == "images":
            images.prepare = held  # type: ignore[method-assign]
        else:
            images.wait = held_wait  # type: ignore[method-assign]
        images.discard = lambda prep: (discarded.append(len(prep.images)), discard(prep))  # type: ignore
    added: list = []

    def add(st: Any, msg: Any) -> None:
        added.append(msg)
        app.client.reqs[st.rid] = st
        asyncio.get_running_loop().call_soon(st.finish, "stop", None)

    app.client.add = add  # type: ignore[method-assign]
    return gate, Peer(body, gate if where == "body" else None), added, discarded


async def until(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.005)


async def turns(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


WHERE = ["body", "grammar", "thinking", "images", "wait"]


@pytest.mark.parametrize("where", WHERE)
def test_a_drain_waits_for_a_request_being_prepared(make_app, where) -> None:
    """A drain that begins while a request reads its body, builds its grammar or prepares its images waits for it:
    the request is served, then the frontend reports itself drained."""
    app = make_app(30.0)

    async def main() -> None:
        gate, peer, added, _ = stub(app, where)
        drained = asyncio.Event()
        app.on_drained = drained.set
        task = asyncio.create_task(app.http(SCOPE, peer.receive, peer.send))
        await until(gate.entered.is_set)
        app.begin_drain(0.0)
        await turns()
        assert app.inflight == 1 and not drained.is_set() and not app.accepting
        gate.go.set()
        await asyncio.wait_for(task, 5)
        await asyncio.wait_for(drained.wait(), 5)
        assert peer.status() == 200 and len(added) == 1

    asyncio.run(main())


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("how", ["stop_now", "drain_timeout"])
def test_a_stop_fails_a_request_being_prepared_before_the_engine(make_app, where, how) -> None:
    """`stop_now`, or a drain whose timeout passes, while a request is still being prepared: the frontend waits for
    it to answer, it answers 503 at its next checkpoint, it never reaches the engine, and its images are discarded."""
    app = make_app(0.2)

    async def main() -> None:
        gate, peer, added, discarded = stub(app, where)
        drained = asyncio.Event()
        app.on_drained = drained.set
        failed: list[str] = []
        fail_all = app.fail_all
        app.fail_all = lambda message: (failed.append(message), fail_all(message))  # type: ignore[method-assign]
        task = asyncio.create_task(app.http(SCOPE, peer.receive, peer.send))
        await until(gate.entered.is_set)
        if how == "stop_now":
            app.stop_now()
        else:
            app.begin_drain(0.0)
            await turns()
            assert not drained.is_set()  # it waits for the request, until its timeout
            await until(lambda: bool(failed))
        await turns()
        assert app.inflight == 1 and not drained.is_set()
        gate.go.set()
        await asyncio.wait_for(task, 5)
        await asyncio.wait_for(drained.wait(), 5)
        assert peer.status() == 503 and peer.json()["error"]["message"] == STOPPED
        assert peer.json()["error"]["type"] == "ServiceUnavailableError"
        assert added == [] and discarded == ([1] if where in ("images", "wait") else [])

    asyncio.run(main())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("where", WHERE[1:])
def test_a_client_that_leaves_while_its_request_is_prepared_never_reaches_the_engine(make_app, where, stream) -> None:
    """The client disconnects while its request builds its grammar or prepares its images: the request is never
    sent to the engine, gets no response, discards its images and counts as a disconnect."""
    app = make_app(30.0)

    async def main() -> None:
        gate, peer, added, discarded = stub(app, where, stream)
        task = asyncio.create_task(app.http(SCOPE, peer.receive, peer.send))
        await until(gate.entered.is_set)
        peer.gone.set()
        await turns()
        gate.go.set()
        await asyncio.wait_for(task, 5)
        assert added == [] and peer.sent == [] and app.inflight == 0
        assert discarded == ([1] if where in ("images", "wait") else [])
        assert app.metrics.client_disconnects.values == {("/v1/chat/completions", "true" if stream else "false"): 1}

    asyncio.run(main())


# -- the frontend's mode ---------------------------------------------------------------------------------------------

APP_LOG = "llm_mini_infer.api.app"
FLAGS = {Mode.SERVING: (False, True, True), Mode.DRAINING: (True, True, True), Mode.REFUSING: (True, False, True),
         Mode.FAILING: (True, False, True), Mode.STOPPED: (True, False, False)}  # draining, accepting, engine_ok


def flags(app: App) -> tuple:
    return app.draining, app.accepting, app.engine_ok, app.failed


def lines(caplog) -> list[str]:
    """The application's log lines, the request lines and the startup line on the API key aside."""
    return [r.getMessage() for r in caplog.records
            if r.name == APP_LOG and not r.getMessage().startswith(("request ", "API key: "))]


@pytest.mark.parametrize("path", [(Mode.DRAINING, Mode.REFUSING, Mode.FAILING, Mode.STOPPED), (Mode.STOPPED,),
                                  (Mode.DRAINING, Mode.STOPPED), (Mode.DRAINING, Mode.REFUSING, Mode.STOPPED)])
def test_the_frontends_mode_takes_its_edges_and_no_other(make_app, path) -> None:
    """Every edge of `Mode` from SERVING sets its mode's flags; a step back, or out of STOPPED, raises and changes
    nothing."""
    app = make_app(30.0)
    for mode in path:
        app._set(mode)
        assert app.mode.phase is mode and flags(app)[:3] == FLAGS[mode]
        if mode is Mode.REFUSING:
            with pytest.raises(RuntimeError, match="the frontend's mode: no edge from REFUSING to DRAINING"):
                app._set(Mode.DRAINING)
            assert app.mode.phase is mode and flags(app)[:3] == FLAGS[mode]
    with pytest.raises(RuntimeError, match="the frontend's mode: no edge from STOPPED to SERVING"):
        app._set(Mode.SERVING)
    assert app.mode.phase is Mode.STOPPED and flags(app)[:3] == FLAGS[Mode.STOPPED]


def test_a_stop_during_the_grace(make_app, caplog) -> None:
    """SIGTERM, then SIGUSR1 before the grace ends: everything fails at once, and the drain, woken by the stop, logs
    its end without a step of its own; both report the frontend drained."""
    app = make_app(30.0)

    async def main() -> None:
        drained: list[int] = []
        app.on_drained = lambda: drained.append(1)
        with caplog.at_level(logging.INFO, logger=APP_LOG):
            app.begin_drain(30.0)
            await turns()
            assert flags(app) == (True, True, True, None)
            app.stop_now()
            await until(lambda: len(drained) == 2)
            await turns()
        assert flags(app) == (True, False, False, STOPPED) and len(drained) == 2
        assert lines(caplog) == ["draining: 0 requests in flight; new ones served for 30 s more (server.drain_grace_s)",
                                 "stopping now: failing 0 requests in flight", "drained"]
        assert app.mode.phase is Mode.STOPPED

    asyncio.run(main())


def test_a_stop_during_the_drains_wait(make_app, caplog) -> None:
    """SIGINT with a request in flight (its grammar building), then SIGUSR1 while the drain waits for it: the request
    fails with a 503 at its checkpoint, and the drain ends once it has."""
    app = make_app(30.0)

    async def main() -> None:
        gate, peer, added, _ = stub(app, "grammar")
        drained: list[int] = []
        app.on_drained = lambda: drained.append(1)
        task = asyncio.create_task(app.http(SCOPE, peer.receive, peer.send))
        await until(gate.entered.is_set)
        with caplog.at_level(logging.INFO, logger=APP_LOG):
            app.begin_drain(0.0)
            await turns()
            assert flags(app) == (True, False, True, None)
            app.stop_now()
            await turns()
            assert flags(app) == (True, False, False, STOPPED)
            gate.go.set()
            await asyncio.wait_for(task, 5)
            await until(lambda: len(drained) == 2)
            await turns()
        assert peer.status() == 503 and added == [] and len(drained) == 2
        assert flags(app) == (True, False, False, STOPPED)
        assert lines(caplog) == ["draining: 1 requests in flight; new ones served for 0 s more (server.drain_grace_s)",
                                 "refusing new chat requests; waiting for 1 in flight (up to 30 s)",
                                 "stopping now: failing 1 requests in flight", "drained"]
        assert app.mode.phase is Mode.STOPPED

    asyncio.run(main())


def test_a_second_sigint_changes_nothing(make_app, caplog) -> None:
    """`begin_drain(0.0)` twice: one drain, one log line for it, the grace already cut."""
    app = make_app(30.0)

    async def main() -> None:
        drained: list[int] = []
        app.on_drained = lambda: drained.append(1)
        with caplog.at_level(logging.INFO, logger=APP_LOG):
            app.begin_drain(0.0)
            app.begin_drain(0.0)
            await until(lambda: bool(drained))
            await turns()
        assert flags(app) == (True, False, True, None) and len(drained) == 1
        assert lines(caplog) == ["draining: 0 requests in flight; new ones served for 0 s more (server.drain_grace_s)",
                                 "drained"]
        assert app.mode.phase is Mode.REFUSING

    asyncio.run(main())


def test_a_drain_signal_after_a_stop_changes_nothing(make_app, caplog) -> None:
    """SIGUSR1, then SIGTERM and SIGINT: the stop alone runs; the drain signals start no drain and log nothing."""
    app = make_app(30.0)

    async def main() -> None:
        drained: list[int] = []
        app.on_drained = lambda: drained.append(1)
        with caplog.at_level(logging.INFO, logger=APP_LOG):
            app.stop_now()
            app.begin_drain(30.0)
            app.begin_drain(0.0)
            await until(lambda: bool(drained))
            await turns()
        assert flags(app) == (True, False, False, STOPPED) and len(drained) == 1
        assert lines(caplog) == ["stopping now: failing 0 requests in flight"]
        assert app.mode.phase is Mode.STOPPED

    asyncio.run(main())

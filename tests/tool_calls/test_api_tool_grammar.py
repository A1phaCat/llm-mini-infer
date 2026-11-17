"""The call grammars through the server (launcher, engine and frontend processes) on the fake runner, on Gemma 4 and
Qwen3.8 under the image's structured-output settings (guidance, compact JSON) with MTP: the 400s (structured output off,
the byte cap, a keyword or `$ref` the grammar cannot take, only where one is built), a named tool's forced prefix and
`max_tokens` below it, and the fake's masked samples, which can only spell calls under `required` and named."""

from __future__ import annotations

import json

import pytest

from ..support.api_util import Server, env_has_model
from ..support.fixtures import QWEN
from ..support.paths import MODELS_DIR

CHAT = "/v1/chat/completions"
GEMMA = MODELS_DIR / "gemma-4-31b-it"
WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time"}}
MSG = [{"role": "user", "content": "Weather in Paris?"}]
GUIDANCE = {"structured_output": {"backend": "guidance", "disable_any_whitespace": True}}
NAMED = {"type": "function", "function": {"name": "get_weather"}}


def start(model_dir, tmp, mtp=3, extra=None) -> Server:
    cfg = {"runner": {"kind": "fake"}, "model": {"chat_template_kwargs": {"enable_thinking": False}}, **GUIDANCE,
           **(extra or {})}
    if model_dir == GEMMA:
        cfg["images"] = {"enabled": False}
        if mtp:
            (tmp / "assistant").mkdir(exist_ok=True)
            (tmp / "assistant" / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant",
                                                                       "text_config": {"model_type": "gemma4_text"}}))
            cfg["speculative"] = {"model": str(tmp / "assistant")}
    return Server(model_dir, tmp, step_ms=0.2, mtp=mtp, extra=cfg).start(120)


@pytest.fixture(scope="module", params=["gemma", "qwen"])
def server(request, tmp_path_factory):
    model_dir = GEMMA if request.param == "gemma" else QWEN
    if not env_has_model(model_dir):
        pytest.skip(f"{model_dir} is absent: uv run tests/scripts/fetch_fixtures.py")
    srv = start(model_dir, tmp_path_factory.mktemp("srv"))
    srv.family = request.param
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def post(srv: Server, **extra) -> tuple[int, dict]:
    return srv.post(CHAT, {"messages": MSG, "tools": [WEATHER, TIME], "max_tokens": 48, **extra})


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("choice", ["required", "named"])
def test_masked_samples_spell_calls(server, choice, seed) -> None:
    """Whatever the fake samples, its masks make calls: of the tools (the named one only), typed, through MTP's
    drafts; a call the grammar completes ends the turn (`tool_calls` for required, `stop` for named)."""
    code, r = post(server, tool_choice=NAMED if choice == "named" else "required", seed=900 + seed, max_tokens=200)
    assert code == 200, r
    c = r["choices"][0]
    assert c["message"]["content"] == ""
    if c["finish_reason"] == "length":
        return
    assert c["finish_reason"] == ("stop" if choice == "named" else "tool_calls")
    assert c["stop_reason"] is None  # the EOS-free finish's id: the tokenizer's own EOS
    names = {x["function"]["name"] for x in c["message"]["tool_calls"]}
    assert names and names <= ({"get_weather"} if choice == "named" else {"get_weather", "get_time"})
    for x in c["message"]["tool_calls"]:
        args = json.loads(x["function"]["arguments"])
        if x["function"]["name"] == "get_weather":
            assert isinstance(args["city"], str) and set(args) <= {"city", "days"}
            assert isinstance(args.get("days", 0), int)
        else:
            assert args == {}


def test_max_tokens_below_the_forced_prefix(server) -> None:
    """A named tool's opening is forced (guidance): cut at 4 tokens, the call is truncated, by `length`."""
    for stream in (False, True):
        if stream:
            events = [e for e in server.stream({"messages": MSG, "tools": [WEATHER], "max_tokens": 4,
                                                "tool_choice": NAMED}) if isinstance(e, dict)]
            assert events[-1]["choices"][0]["finish_reason"] == "length"
        else:
            code, r = post(server, tool_choice=NAMED, max_tokens=4)
            assert code == 200 and r["choices"][0]["finish_reason"] == "length", r
            assert r["usage"]["completion_tokens"] == 4


def test_response_format_with_tools(server) -> None:
    """auto with a response_format: a call or the schema's compact JSON (the fake's masked samples take either)."""
    rf = {"type": "json_schema", "json_schema": {"name": "a", "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
        "additionalProperties": False}}}
    kinds = set()
    for seed in range(8):
        code, r = post(server, response_format=rf, seed=1300 + seed, max_tokens=120)
        assert code == 200, r
        c = r["choices"][0]
        if c["finish_reason"] == "length":
            continue
        if c["message"].get("tool_calls"):  # only with calls (vLLM's envelope)
            kinds.add("call")
        else:
            doc = json.loads(c["message"]["content"])
            assert set(doc) == {"answer"}
            kinds.add("json")
    assert kinds


def test_schema_400s_only_where_a_grammar_is_built(server) -> None:
    bad = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"$ref": "#/$defs/nowhere"}}}}}
    code, r = post(server, tools=[bad], tool_choice="required")
    assert code == 400 and r["error"]["param"] == "tools" and "does not resolve" in r["error"]["message"], r
    code, r = post(server, tools=[bad])  # auto: no grammar, served
    assert code == 200, r
    odd = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"type": "string", "not": {"const": "x"}}}}}}
    code, r = post(server, tools=[odd], tool_choice="required")
    assert code == 400 and r["error"]["param"] == "tools" and "'not'" in r["error"]["message"], r


def test_a_broken_response_format_is_its_own_400(server) -> None:
    """auto with tools and a `response_format` llguidance refuses: the `response_format`'s 400, its text and param as
    without tools (checked first, alone)."""
    rf = {"type": "json_schema", "json_schema": {"name": "a", "schema": {
        "type": "object", "properties": {"a": {"type": "nonsense"}}}}}
    alone = server.post(CHAT, {"messages": MSG, "max_tokens": 8, "response_format": rf})
    assert alone[0] == 400 and alone[1]["error"]["param"] == "response_format", alone
    assert alone[1]["error"]["message"].startswith("invalid JSON schema in 'response_format': ")
    assert post(server, response_format=rf) == alone


def test_a_grammar_above_its_bound_is_refused(server) -> None:
    """A `$defs` chain each Qwen parameter reaches would build a grammar 57 times its schema: a 400 (param tools), the
    schema well under the byte cap. Gemma's grammar names each definition once: served."""
    from ..support.tool_grammar_cases import chained_defs

    code, r = post(server, tools=[chained_defs(True)], tool_choice="required", max_tokens=4)
    if server.family == "qwen":
        assert code == 400 and r["error"]["param"] == "tools" and "bytes" in r["error"]["message"], r
    else:
        assert code == 200, r


def test_a_malformed_schema_value_is_a_400(server) -> None:
    """A wrong-typed `type`, `required`, `properties`, `items` or `additionalProperties`, at the root and nested: a 400,
    param `tools`, under `required` and a named tool, and the next request is served."""
    from ..support.tool_grammar_cases import MALFORMED

    for form in MALFORMED:
        for params in ({"type": "object", "properties": {"a": {}}, **form},
                       {"type": "object", "properties": {"x": form}, "required": ["x"]}):
            t = {"type": "function", "function": {"name": "f", "parameters": params}}
            for choice in ("required", {"type": "function", "function": {"name": "f"}}):
                code, r = post(server, tools=[t], tool_choice=choice, max_tokens=4)
                assert code == 400 and r["error"]["param"] == "tools", (form, params, r)
    assert post(server, tool_choice="required", max_tokens=4)[0] == 200


@pytest.mark.serial  # a server of its own beside the module's: three of Gemma's are 4 GB
def test_the_schema_byte_cap_counts_the_tools(tmp_path) -> None:
    srv = start(QWEN if env_has_model(QWEN) else GEMMA, tmp_path, mtp=0,
                extra={"structured_output": {**GUIDANCE["structured_output"], "max_schema_bytes": 50}})
    try:
        code, r = post(srv, tool_choice="required")
        assert code == 400 and r["error"]["param"] == "tools" and "max_schema_bytes" in r["error"]["message"], r
        code, r = post(srv)  # auto: no grammar
        assert code == 200, r
    finally:
        assert srv.stop() == 0


@pytest.mark.serial  # a server of its own beside the module's: three of Gemma's are 4 GB
def test_structured_output_off(tmp_path) -> None:
    srv = start(QWEN if env_has_model(QWEN) else GEMMA, tmp_path, mtp=0,
                extra={"structured_output": {"enabled": False}})
    try:
        for extra, param in (({"tool_choice": "required"}, "tool_choice"), ({"tool_choice": NAMED}, "tool_choice"),
                             ({"response_format": {"type": "json_object"}}, "response_format")):
            code, r = post(srv, **extra)
            assert code == 400 and r["error"]["param"] == param, r
            assert r["error"]["message"] == "structured output is off on this server (structured_output.enabled)"
        code, r = post(srv)  # auto, unconstrained
        assert code == 200, r
    finally:
        assert srv.stop() == 0

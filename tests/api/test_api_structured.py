"""`response_format` through the real server (launcher, engine and frontend processes) against the fake runner, with
llguidance's masks over the Qwen3.8 tokenizer.

The fake samples a deterministic token from the whole vocabulary and moves it to the next one the mask allows
(`runner.fake.masked_pick`), so every response that ends by stop must be a JSON document of its schema, whatever the
tokens; with MTP its drafts are unmasked and verified against the masked samples. Also: invalid schemas (400s with
llguidance's message), the frontend's limits (nesting, NUL bytes, the schema's bytes: a 400 or served, never a 500),
OpenAI's role-only first chunk, /detokenize and /server_info.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from ..support.api_util import Server, content_of, env_has_model
from ..support.fixtures import QWEN
from ..support.structured_cases import SCHEMA

pytestmark = pytest.mark.skipif(not env_has_model(QWEN), reason="tests/.fixtures/models/qwen3.8-27b is absent: "
                                                                "uv run tests/scripts/fetch_fixtures.py")
MSG = [{"role": "user", "content": "List the entries in the catalogue."}]
KW = {"enable_thinking": False}


@pytest.fixture(scope="module", params=[0, 3], ids=["mtp0", "mtp3"])
def server(request, tmp_path_factory):
    srv = Server(QWEN, tmp_path_factory.mktemp("srv"), mtp=request.param, step_ms=0.2).start()
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def schema_format(schema: dict, strict: bool | None = True) -> dict:
    js = {"name": "catalog", "schema": schema}
    if strict is not None:
        js["strict"] = strict
    return {"type": "json_schema", "json_schema": js}


def chat(**kw) -> dict:
    return {"messages": MSG, "chat_template_kwargs": KW, "max_tokens": 700, "temperature": 0.0, **kw}


def test_json_schema_responses_parse_and_validate(server) -> None:
    stops = 0
    for seed in range(12):
        status, r = server.post("/v1/chat/completions", chat(seed=seed, temperature=1.0,
                                                             response_format=schema_format(SCHEMA, seed % 2 == 0)))
        assert status == 200, r
        choice = r["choices"][0]
        if choice["finish_reason"] == "stop":
            stops += 1
            jsonschema.validate(json.loads(choice["message"]["content"]), SCHEMA)
        else:
            assert choice["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 700
    assert stops >= 8
    assert server.metric("llm_mini_infer:structured_requests_total") >= 12
    assert server.metric("llm_mini_infer:grammar_compiles_total") == 1  # one schema: compiled once, cloned per request
    assert server.metric("llm_mini_infer:grammar_errors_total") == 0


def test_streamed_json_and_the_role_only_first_chunk(server) -> None:
    events = server.stream(chat(seed=3, temperature=1.0, response_format=schema_format(SCHEMA)))
    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    last = events[-2]["choices"][0]
    if last["finish_reason"] == "stop":
        jsonschema.validate(json.loads(content_of(events)), SCHEMA)


def test_json_object_and_text(server) -> None:
    for seed in range(4):
        status, r = server.post("/v1/chat/completions",
                                chat(seed=100 + seed, temperature=1.0, response_format={"type": "json_object"}))
        assert status == 200, r
        if r["choices"][0]["finish_reason"] == "stop":
            assert isinstance(json.loads(r["choices"][0]["message"]["content"]), dict)
    status, r = server.post("/v1/chat/completions", chat(max_tokens=3, response_format={"type": "text"}))
    assert status == 200 and r["usage"]["completion_tokens"] == 3


def test_truncated_json_reports_length(server) -> None:
    status, r = server.post("/v1/chat/completions", chat(max_tokens=4, response_format=schema_format(SCHEMA)))
    assert status == 200 and r["choices"][0]["finish_reason"] == "length"
    assert r["choices"][0]["message"]["content"].startswith("{")


@pytest.mark.parametrize(("schema", "msg"), [
    ({"type": "object", "properties": {"a": {"type": "strin"}}}, "Invalid type: strin"),
    ({"type": "object", "properties": {"a": {"type": "string", "format": "whatever"}}}, "Unknown format"),
    ({"type": "array", "uniqueItems": True}, "Unimplemented keys"),
    ({"$ref": "#/$defs/nowhere"}, "does not exist"),
])
def test_invalid_schemas_are_400s_with_llguidance_s_message(server, schema, msg) -> None:
    status, r = server.post("/v1/chat/completions", chat(response_format=schema_format(schema)))
    assert status == 400, r
    assert msg in r["error"]["message"] and r["error"]["param"] == "response_format"


def nested(levels: int) -> dict:
    """A schema whose JSON nests `levels` deep (arrays of arrays of ... integers)."""
    schema: dict = {"type": "integer"}
    for _ in range(levels - 1):
        schema = {"type": "array", "items": schema}
    return schema


def padded(levels: int) -> dict:
    """A schema two levels deep whose JSON nests `levels` deep: a `const` of nested lists under the root, `properties`
    and `v`."""
    c: list = [1]
    for _ in range(levels - 4):
        c = [c]
    return {"type": "object", "properties": {"v": {"const": c}}}


TOO_DEEP = "The JSON schema is nested too deeply: objects and arrays may nest at most 64 levels"


@pytest.mark.parametrize(("schema", "status", "msg", "param"), [
    (nested(32), 200, None, None), (nested(33), 400, "nests 33 levels deep", "response_format"),
    (padded(64), 200, None, None), (padded(65), 400, TOO_DEEP, None), (nested(2000), 400, TOO_DEEP, None)],
    ids=["32-levels", "33-levels", "json-64", "json-65", "2000-levels"])
def test_nested_schemas(server, schema, status, msg, param) -> None:
    """A schema nests 32 levels at most (the subset's cap), and its JSON 64 levels at most: vLLM's cap, checked first,
    with vLLM's text and no param, as vLLM raises it."""
    code, r = server.post("/v1/chat/completions", chat(max_tokens=4, response_format=schema_format(schema)))
    assert code == status, r
    if status == 400:
        assert r["error"]["param"] == param
        assert msg in r["error"]["message"] and (param is not None or r["error"]["message"] == msg)


def test_bodies_nested_too_deeply_are_400s(server) -> None:
    """Deeper than msgspec decodes (about 10,000 levels) is a 400: inside a schema, as the schema, inside the chat
    template's kwargs; so is a tool call's `arguments` string nested deeper than json.loads parses, on
    /v1/chat/completions and /tokenize."""
    deep = "[" * 200_000 + "]" * 200_000
    for path, request in (
            ("/v1/chat/completions", chat(response_format=schema_format({"type": "object", "x": "DEEP"}))),
            ("/v1/chat/completions", chat(response_format={"type": "json_schema",
                                                           "json_schema": {"name": "catalog", "schema": "DEEP"}})),
            ("/v1/chat/completions", chat(chat_template_kwargs={**KW, "x": "DEEP"})),
            ("/tokenize", {"messages": MSG, "chat_template_kwargs": {**KW, "x": "DEEP"}})):
        code, r = server.post(path, None, raw=json.dumps(request).replace('"DEEP"', deep).encode())
        assert code == 400 and r["error"]["message"] == "invalid JSON: nested too deeply", (path, request, r)
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": deep}}
    messages = [*MSG, {"role": "assistant", "tool_calls": [call]}, {"role": "user", "content": "Again."}]
    for path, body in (("/v1/chat/completions", chat(messages=messages)),
                       ("/tokenize", {"messages": messages, "chat_template_kwargs": KW})):
        code, r = server.post(path, body)
        assert code == 400 and r["error"]["message"] == "the messages are nested too deeply", (path, r)


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"a": {"type": "string", "pattern": "^a\u0000b$"}}, "required": ["a"]},
    {"type": "object", "properties": {"a": {"enum": ["x\u0000y", "z"]}}, "required": ["a"]},
    {"type": "object", "properties": {"a\u0000b": {"type": "integer"}}, "required": ["a\u0000b"]},
    {"type": "object", "properties": {"a": {"const": "\u0000"}}, "required": ["a"]},
], ids=["pattern", "enum", "key", "const"])
def test_nul_bytes_in_a_schema(server, schema) -> None:
    """NUL bytes in a schema are characters like any other here."""
    code, r = server.post("/v1/chat/completions", chat(max_tokens=40, response_format=schema_format(schema)))
    assert code == 200, r
    if r["choices"][0]["finish_reason"] == "stop":
        jsonschema.validate(json.loads(r["choices"][0]["message"]["content"]), schema)


def test_the_schemas_bytes_are_capped(server) -> None:
    """structured_output.max_schema_bytes (1 MiB by default), as the request carries the schema: at the cap it is
    served, a byte over it is a 400 that names the cap."""
    cap = 1 << 20

    def padded(size: int) -> dict:  # a schema of `size` bytes as the request carries it (json.dumps)
        schema = {"type": "object", "description": ""}
        schema["description"] = "x" * (size - len(json.dumps(schema)))
        return schema

    code, r = server.post("/v1/chat/completions", chat(max_tokens=2, response_format=schema_format(padded(cap))))
    assert code == 200, r
    code, r = server.post("/v1/chat/completions", chat(max_tokens=2, response_format=schema_format(padded(cap + 1))))
    assert code == 400 and r["error"]["param"] == "response_format", r
    assert f"is {cap + 1} bytes" in r["error"]["message"]
    assert f"{cap} (structured_output.max_schema_bytes)" in r["error"]["message"]


def test_detokenize_and_server_info(server) -> None:
    status, tok = server.post("/tokenize", {"prompt": "Hello, structured world!"})
    assert status == 200
    status, r = server.post("/detokenize", {"tokens": tok["tokens"]})
    assert status == 200 and r == {"prompt": "Hello, structured world!"}
    assert server.post("/detokenize", {"tokens": [10 ** 9]})[0] == 400
    assert server.post("/detokenize", {"tokens": [1], "extra": 1})[0] == 400
    status, body = server.get("/server_info")
    info = json.loads(body)
    assert status == 200 and info["config"]["structured_output"]["enabled"] is True
    assert info["config"]["images"]["allow_urls"] is False
    assert "dev.structured_output.threads" in info["reasons"]
    assert info["runner"]["capabilities"]["grammar_masks"] is True
    assert info["runner"]["capabilities"]["mtp_depth"] in (0, 3)

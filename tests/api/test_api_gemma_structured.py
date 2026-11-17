"""`response_format` through the real server (launcher, engine and frontend processes) on google/gemma-4-31B-it's
tokenizer and config against the fake runner, configured with the guidance backend and compact JSON (vLLM's
`--structured-outputs-config`): `structured_output.backend: guidance` with `disable_any_whitespace`, without and
with MTP.

The fake picks its tokens under the masks (`runner.fake.masked_pick`), so every response that ends by stop is a compact
JSON document of its schema, whatever the tokens; /server_info names the backend and the grammar's EOS (the
tokenizer's `<eos>`, not the model's other stop ids). The masks themselves are proven equal to vLLM's reference
code in tests/structured/test_grammar_parity.py.
"""

from __future__ import annotations

import json
import re

import jsonschema
import pytest

from ..support.api_util import Server, env_has_model
from ..support.paths import MODELS_DIR
from ..support.structured_cases import SCHEMA

GEMMA = MODELS_DIR / "gemma-4-31b-it"
pytestmark = pytest.mark.skipif(not env_has_model(GEMMA), reason="tests/.fixtures/models/gemma-4-31b-it is absent: "
                                                                 "uv run tests/scripts/fetch_fixtures.py")
GUIDANCE = {"runner": {"kind": "fake"}, "images": {"enabled": False},
              "structured_output": {"backend": "guidance", "disable_any_whitespace": True}}
MSG = [{"role": "user", "content": "List the entries in the catalogue."}]


@pytest.fixture(scope="module", params=[0, 3], ids=["mtp0", "mtp3"])
def server(request, tmp_path_factory):
    """With MTP the fake drafts as the assistant would (a config is all it reads of one), k = 3: the verify rows take
    1 + 3 masks along the drafts."""
    tmp = tmp_path_factory.mktemp("srv")
    extra = dict(GUIDANCE)
    if request.param:
        (tmp / "assistant").mkdir()
        (tmp / "assistant" / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant",
                                                                   "text_config": {"model_type": "gemma4_text"}}))
        extra["speculative"] = {"model": str(tmp / "assistant")}
    srv = Server(GEMMA, tmp, step_ms=0.2, mtp=request.param, extra=extra).start(120)
    srv.mtp = request.param
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def schema_format(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "catalog", "schema": schema, "strict": True}}


def compact(text: str) -> bool:
    """No whitespace outside the string literals (not a round trip through json, which writes llguidance's `-0` as
    0)."""
    return not re.search(r"\s", re.sub(r'"(?:[^"\\]|\\.)*"', '""', text))


def test_guidance_compact_responses_parse_and_validate(server) -> None:
    stops = 0
    for seed in range(12):
        status, r = server.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 400, "temperature": 1.0,
                                                         "seed": seed, "response_format": schema_format(SCHEMA)})
        assert status == 200, r
        choice = r["choices"][0]
        if choice["finish_reason"] == "stop":
            stops += 1
            jsonschema.validate(json.loads(choice["message"]["content"]), SCHEMA)
            assert compact(choice["message"]["content"])
        else:
            assert choice["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 400
    assert stops >= 8
    for seed in range(100, 104):
        status, r = server.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 400, "temperature": 1.0,
                                                         "seed": seed, "response_format": {"type": "json_object"}})
        assert status == 200, r
        if r["choices"][0]["finish_reason"] == "stop":
            content = r["choices"][0]["message"]["content"]
            assert isinstance(json.loads(content), dict) and compact(content)
    assert server.metric("llm_mini_infer:grammar_errors_total") == 0
    if server.mtp:  # the fake drafts the unmasked chain: the grammar refuses most drafts, which are not counted
        drafts, tokens = (server.metric(f"vllm:spec_decode_num_{n}_total") for n in ("drafts", "draft_tokens"))
        assert drafts > 0 and server.metric("vllm:spec_decode_num_accepted_tokens_total") <= tokens < 3 * drafts


@pytest.mark.parametrize("stream", [False, True])
def test_a_finished_grammar_reports_no_stop_reason(server, stream) -> None:
    """The grammar's end: the EOS-free finish appends the tokenizer's `<eos>` (1), the id a guidance grammar ends on,
    so `stop_reason` stays null, streamed and not, while the model's other stop ids are reported."""
    stops = 0
    for seed in range(6):
        body = {"messages": MSG, "max_tokens": 400, "temperature": 1.0, "seed": seed, "return_token_ids": True,
                "response_format": schema_format(SCHEMA)}
        if stream:
            events = [e for e in server.stream(body) if isinstance(e, dict)]
            final = events[-1]["choices"][0]
            ids = [t for e in events for t in e["choices"][0].get("token_ids") or []]
            content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        else:
            status, r = server.post("/v1/chat/completions", body)
            assert status == 200, r
            final, ids, content = r["choices"][0], r["choices"][0]["token_ids"], r["choices"][0]["message"]["content"]
        if final["finish_reason"] == "stop":
            stops += 1
            assert final["stop_reason"] is None and ids[-1] == 1
            jsonschema.validate(json.loads(content), SCHEMA)
    assert stops >= 3


def test_ignore_eos_runs_on_past_the_document(server) -> None:
    """With ignore_eos the forced `<eos>` after the closing brace ends the grammar without being consumed (llguidance
    takes no EOS once the document is complete) and does not end the request:
    the rest is unconstrained, up to max_tokens."""
    errors = server.metric("llm_mini_infer:grammar_errors_total")
    for seed in range(4):
        status, r = server.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 400, "temperature": 1.0,
                                                         "seed": seed, "ignore_eos": True,
                                                         "response_format": schema_format(SCHEMA)})
        assert status == 200, r
        choice, content = r["choices"][0], r["choices"][0]["message"]["content"]
        assert choice["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 400
        doc, end = json.JSONDecoder().raw_decode(content)
        jsonschema.validate(doc, SCHEMA)
        assert compact(content[:end]) and end < len(content)
    assert server.metric("llm_mini_infer:grammar_errors_total") == errors


def test_truncated_compact_json_reports_length(server) -> None:
    status, r = server.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 5, "temperature": 1.0, "seed": 7,
                                                     "response_format": schema_format(SCHEMA)})
    assert status == 200 and r["choices"][0]["finish_reason"] == "length"
    assert r["choices"][0]["message"]["content"].startswith('{"total":')  # '{"', 'total' and '":' forced, no whitespace


@pytest.mark.parametrize(("schema", "msg"), [
    ({"type": "object", "properties": {"a": {"type": "strin"}}}, "Invalid type: strin"),
    ({"type": "object", "properties": {"a": {"type": "string", "format": "whatever"}}}, "Unknown format"),
    ({"type": "array", "uniqueItems": True}, "Unimplemented keys"),
    ({"$ref": "#/$defs/nowhere"}, "does not exist"),
])
def test_invalid_schemas_are_400s_with_llguidance_s_message(server, schema, msg) -> None:
    """The guidance language's errors are llguidance's JSON compiler's, as the emulation's (test_api_structured)."""
    status, r = server.post("/v1/chat/completions", {"messages": MSG, "response_format": schema_format(schema)})
    assert status == 400, r
    err = r["error"]
    assert msg in err["message"] and "\n" not in err["message"] and err["param"] == "response_format"


def test_server_info_names_the_backend_and_its_eos(server) -> None:
    status, body = server.get("/server_info")
    info = json.loads(body)
    assert status == 200
    so = info["config"]["structured_output"]
    assert so["enabled"] is True and so["backend"] == "guidance" and so["disable_any_whitespace"] is True
    assert info["config"]["grammar_eos_token_ids"] == [1] and info["config"]["eos_token_ids"] == [1, 50, 106]
    assert "<eos>" in info["reasons"]["grammar_eos_token_ids"]
    assert "GuidanceBackend" in info["reasons"]["structured_output.backend"]
    assert info["runner"]["capabilities"]["mtp_depth"] == server.mtp

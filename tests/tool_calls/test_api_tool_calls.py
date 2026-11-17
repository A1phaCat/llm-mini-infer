"""Tool calls end to end: the real server (launcher, engine and frontend processes) on the fake runner, whose
`dev.runner.fake.scripts` make a request (by seed) emit the call syntax a model writes, on Gemma 4's and Qwen3.8's
tokenizers and templates, without and with MTP. The responses: the non-streamed message and finish reason, the
streamed deltas (a call's id, type and name first, then its arguments; `index` on every one), `stop_reason`, usage,
logprobs, `return_token_ids`, `parallel_tool_calls`, every `tool_choice`, and stop strings in content, inside a call's
arguments and next to a marker; and no parser for a request without tools (the metric)."""

from __future__ import annotations

import json
import re

import pytest

from llm_mini_infer.api.tools import template_tool
from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.render.renderer import Renderer

from ..support.api_util import Server, env_has_model
from ..support.fixtures import QWEN
from ..support.paths import MODELS_DIR

CHAT = "/v1/chat/completions"
GEMMA = MODELS_DIR / "gemma-4-31b-it"
WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time"}}
TOOLS = [WEATHER, TIME]
MSG = [{"role": "user", "content": "Weather in Paris, and the time?"}]
KW = {"enable_thinking": False}
D = '<|"|>'
ID = re.compile(r"chatcmpl-tool-[0-9a-f]{16}")

# family -> case -> (output text, stop token or None)
OUTPUTS = {
    "gemma": {
        "single": (f"<|tool_call>call:get_weather{{city:{D}Paris{D},days:3}}<tool_call|>", 50),
        "around": ("Let me check. <|tool_call>call:get_time{}<tool_call|> Done.", 50),
        "two": (f"<|tool_call>call:get_weather{{city:{D}Paris{D}}}<tool_call|><|tool_call>call:get_time{{}}"
                "<tool_call|>", 50),
        "turn": (f"<|tool_call>call:get_weather{{city:{D}Rome{D}}}", 106),
        "text": ("It is sunny in Paris.", 106),
        "cut": (f"<|tool_call>call:get_weather{{city:{D}Par", None),
    },
    "qwen": {
        "single": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n"
                   "</parameter>\n</function>\n</tool_call>", 248046),
        "around": ("Let me check.\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>\n\nDone.", 248046),
        "two": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
                "</tool_call>\n"
                "<tool_call>\n<function=get_time>\n</function>\n</tool_call>", 248046),
        "turn": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>", 248046),
        "text": ("It is sunny in Paris.", 248046),
        "cut": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar", None),
    },
}
SEEDS = {name: 7100 + i for i, name in enumerate(OUTPUTS["gemma"])}
# Gemma calls nested 3,000 levels deep (their one argument): kept as written, a string
DEEP = {"deep_object": "{a:" * 3000 + "1" + "}" * 3000, "deep_array": "[" * 3000 + "1" + "]" * 3000}
DEEP_SEEDS = {name: 7200 + i for i, name in enumerate(DEEP)}
# The models' stop ids (generation_config.json's eos_token_id) -> the stop_reason a finish on one reports: each but
# the tokenizer's own EOS (Gemma's <eos>, Qwen's <|im_end|>), which is null
MODEL_STOPS = {"gemma": {1: None, 50: 50, 106: 106}, "qwen": {248044: 248044, 248046: None}}
PLAIN, PLAIN_SEED = "It is sunny.", 1_000_000
EXPECTED_CALLS = {
    "single": [("get_weather", '{"city": "Paris", "days": 3}')], "around": [("get_time", "{}")],
    "two": [("get_weather", '{"city": "Paris"}'), ("get_time", "{}")], "turn": [("get_weather", '{"city": "Rome"}')],
    "text": [], "cut": [("get_weather", '{"city": "Par"}')],
}
AROUND = {"gemma": "Let me check.  Done.", "qwen": "Let me check.\n\n\n\nDone."}  # the text before and after the call
EXPECTED_CONTENT = {"single": None, "around": AROUND, "two": None, "turn": None, "text": "It is sunny in Paris.",
                    "cut": None}


def script(model_dir, family: str) -> tuple[list, dict]:
    """The fake's scripts for one family's cases, and each case's output ids."""
    r = Renderer(model_dir)
    conv = to_template_messages(MSG, r.content_format, r.supports_developer_role)
    n = len(r.encode(r.render(conv, tools=[template_tool(t) for t in TOOLS], **KW)))
    scripts, ids = [], {}
    for name, (text, stop) in OUTPUTS[family].items():
        toks = r.tokenizer.encode(text, add_special_tokens=False).ids + ([stop] if stop is not None else [])
        ids[name] = toks
        scripts.append([SEEDS[name], n, toks])
        for k in (1, 2, 3):  # the stop-string cases' seeds (stop_args, stop_marker, stop_content)
            scripts.append([SEEDS[name] + 1000 * k, n, toks])
    if family == "gemma":
        for name, body in DEEP.items():
            ids[name] = [*r.tokenizer.encode(f"<|tool_call>call:get_weather{{x:{body}}}<tool_call|>",
                                             add_special_tokens=False).ids, 50]
            scripts.append([DEEP_SEEDS[name], n, ids[name]])
    plain = len(r.encode(r.render(conv, **KW)))  # a request without tools: its answer, ending on each stop id
    for stop in MODEL_STOPS[family]:
        scripts.append([PLAIN_SEED + stop, plain, [*r.tokenizer.encode(PLAIN, add_special_tokens=False).ids, stop]])
    return scripts, ids


def start(model_dir, family: str, tmp, mtp: int) -> Server:
    scripts, ids = script(model_dir, family)
    extra = {"runner": {"kind": "fake"}, "server": {"logprobs": True}, "model": {"chat_template_kwargs": KW},
             "dev": {"runner": {"fake": {"scripts": scripts}}}}
    if family == "gemma":
        extra["images"] = {"enabled": False}
        extra["model"]["max_model_len"] = 8192  # the deep calls' 6,760 tokens
        if mtp:
            (tmp / "assistant").mkdir()
            (tmp / "assistant" / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant",
                                                                       "text_config": {"model_type": "gemma4_text"}}))
            extra["speculative"] = {"model": str(tmp / "assistant")}
    srv = Server(model_dir, tmp, step_ms=0.2, mtp=mtp, extra=extra).start(120)
    srv.ids, srv.family = ids, family
    return srv


@pytest.fixture(scope="module", params=[("gemma", 0), ("gemma", 3), ("qwen", 0), ("qwen", 3)],
                ids=["gemma-mtp0", "gemma-mtp3", "qwen-mtp0", "qwen-mtp3"])
def server(request, tmp_path_factory):
    family, mtp = request.param
    model_dir = GEMMA if family == "gemma" else QWEN
    if not env_has_model(model_dir):
        pytest.skip(f"{model_dir} is absent: uv run tests/scripts/fetch_fixtures.py")
    srv = start(model_dir, family, tmp_path_factory.mktemp("srv"), mtp)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def ask(srv: Server, case: str, **extra) -> dict:
    body = {"messages": MSG, "tools": TOOLS, "seed": SEEDS[case], "max_tokens": len(srv.ids[case]) + 4, **extra}
    code, r = srv.post(CHAT, body)
    assert code == 200, r
    return r


def stream(srv: Server, case: str, **extra) -> list[dict]:
    body = {"messages": MSG, "tools": TOOLS, "seed": SEEDS[case], "max_tokens": len(srv.ids[case]) + 4, **extra}
    return [e for e in srv.stream(body) if isinstance(e, dict)]


def calls_of(message: dict) -> list[tuple[str, str]]:
    """A message's calls: `tool_calls` only when there are some (vLLM's envelope)."""
    assert message.get("tool_calls") != []
    return [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls", [])]


def expected_calls(srv: Server, case: str) -> list:
    got = EXPECTED_CALLS[case]
    return got[srv.family] if isinstance(got, dict) else got


def expected_content(srv: Server, case: str) -> str | None:
    got = EXPECTED_CONTENT[case]
    return got[srv.family] if isinstance(got, dict) else got


@pytest.mark.parametrize("case", list(OUTPUTS["gemma"]))
def test_non_streamed(server, case) -> None:
    max_tokens = len(server.ids[case]) if case == "cut" else len(server.ids[case]) + 4
    r = ask(server, case, max_tokens=max_tokens)
    choice = r["choices"][0]
    msg = choice["message"]
    assert calls_of(msg) == expected_calls(server, case)
    assert all(ID.fullmatch(c["id"]) and c["type"] == "function" for c in msg.get("tool_calls", []))
    assert msg["content"] == expected_content(server, case) and msg["reasoning"] is None
    finish = "length" if case == "cut" else ("tool_calls" if expected_calls(server, case) else "stop")
    assert choice["finish_reason"] == finish and choice["stop_reason"] == stop_reason(server, case)
    assert r["usage"]["completion_tokens"] == len(server.ids[case])


def stop_reason(srv: Server, case: str) -> int | None:
    stop = OUTPUTS[srv.family][case][1]
    return None if stop is None else MODEL_STOPS[srv.family][stop]


@pytest.mark.parametrize("case", list(OUTPUTS["gemma"]))
def test_streamed(server, case) -> None:
    """The deltas, accumulated as a client does, give the non-streamed message's calls; each call's first delta has
    its id, type, index and name (no arguments unless the call came whole), the later ones its index and arguments."""
    max_tokens = len(server.ids[case]) if case == "cut" else len(server.ids[case]) + 4
    events = stream(server, case, max_tokens=max_tokens, stream_options={"include_usage": True})
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content, calls, seen = "", {}, set()
    for e in events[1:-1]:
        d = e["choices"][0]["delta"]
        content += d.get("content", "")
        for tc in d.get("tool_calls", []):
            i = tc["index"]
            if i not in seen:
                seen.add(i)
                assert ID.fullmatch(tc["id"]) and tc["type"] == "function" and tc["function"]["name"]
                calls[i] = [tc["function"]["name"], tc["function"].get("arguments", "")]
            else:
                assert set(tc) == {"index", "function"} and set(tc["function"]) == {"arguments"}
                calls[i][1] += tc["function"]["arguments"]
    assert [tuple(calls[i]) for i in sorted(calls)] == expected_calls(server, case)
    final = events[-2]["choices"][0]
    finish = "length" if case == "cut" else ("tool_calls" if expected_calls(server, case) else "stop")
    assert final["finish_reason"] == finish and final["stop_reason"] == stop_reason(server, case)
    assert content == (expected_content(server, case) or "")  # here the raw spacing and the stripped text agree
    assert events[-1]["usage"]["completion_tokens"] == len(server.ids[case])


@pytest.mark.parametrize("stream", [False, True])
def test_stop_reason_reports_the_models_own_stop_ids(server, stream) -> None:
    """A plain answer (no tools) ending on each of the model's stop ids: `stop_reason` is the id, but the tokenizer's
    own EOS, which is null (the generation config's ids are stop ids of every request); the content and the finish do
    not move. The EOS listed in `stop_token_ids` stays null; with `ignore_eos` it is reported; and so with
    `include_stop_str_in_output` (a special token's text is none)."""
    for stop, want in MODEL_STOPS[server.family].items():
        for extra, reason in (({}, want), ({"stop_token_ids": [stop]}, want),
                              ({"stop_token_ids": [stop], "ignore_eos": True}, stop),
                              ({"include_stop_str_in_output": True}, want)):
            body = {"messages": MSG, "seed": PLAIN_SEED + stop, "max_tokens": 16, **extra}
            if stream:
                events = [e for e in server.stream(body) if isinstance(e, dict)]
                content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
                final = events[-1]["choices"][0]
            else:
                code, r = server.post(CHAT, body)
                assert code == 200, r
                final = r["choices"][0]
                content = final["message"]["content"]
            assert (content, final["finish_reason"], final["stop_reason"]) == (PLAIN, "stop", reason), (stop, extra)


@pytest.mark.parametrize("case", list(DEEP))
def test_a_deep_call_finishes(server, case) -> None:
    """A call nested 3,000 levels deep finishes, streamed and not, with that argument as written, a string, and the
    requests after it are served (Qwen's typed nesting: test_tool_parse.py)."""
    if server.family != "gemma":
        pytest.skip("Gemma's syntax")
    body = {"messages": MSG, "tools": TOOLS, "seed": DEEP_SEEDS[case], "max_tokens": len(server.ids[case]) + 4}
    code, r = server.post(CHAT, body)
    assert code == 200, r
    assert calls_of(r["choices"][0]["message"]) == [("get_weather", json.dumps({"x": DEEP[case]}))]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    events = [e for e in server.stream(body) if isinstance(e, dict)]
    args = "".join(tc["function"].get("arguments", "") for e in events[1:]
                   for tc in e["choices"][0]["delta"].get("tool_calls", []))
    assert args == json.dumps({"x": DEEP[case]}) and events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert calls_of(ask(server, "single")["choices"][0]["message"]) == expected_calls(server, "single")


def test_logprobs_cover_every_token(server) -> None:
    r = ask(server, "two", logprobs=True, top_logprobs=1)
    assert len(r["choices"][0]["logprobs"]["content"]) == len(server.ids["two"])
    events = stream(server, "two", logprobs=True, top_logprobs=1)
    n = sum(len(e["choices"][0]["logprobs"]["content"]) for e in events[1:] if e["choices"][0].get("logprobs"))
    assert n == len(server.ids["two"])


def test_return_token_ids_carry_the_call_and_the_stop_token(server) -> None:
    r = ask(server, "single", return_token_ids=True)
    assert r["choices"][0]["token_ids"] == server.ids["single"]
    events = stream(server, "single", return_token_ids=True)
    assert [t for e in events[1:] for t in e["choices"][0].get("token_ids", [])] == server.ids["single"]


def test_parallel_tool_calls_false_keeps_the_first(server) -> None:
    r = ask(server, "two", parallel_tool_calls=False)
    assert calls_of(r["choices"][0]["message"]) == expected_calls(server, "two")[:1]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    events = stream(server, "two", parallel_tool_calls=False)
    assert {tc["index"] for e in events[1:] for tc in e["choices"][0]["delta"].get("tool_calls", [])} == {0}


@pytest.mark.parametrize("choice", ["none", None])
def test_none_and_null_keep_no_call(server, choice) -> None:
    """The calls leave the content and none is returned; the engine's finish."""
    r = ask(server, "around", tool_choice=choice)
    msg = r["choices"][0]["message"]
    assert "tool_calls" not in msg and r["choices"][0]["finish_reason"] == "stop"
    assert msg["content"] == AROUND[server.family]
    events = stream(server, "around", tool_choice=choice)
    assert not any(e["choices"][0]["delta"].get("tool_calls") for e in events[1:])
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_named_and_required(server) -> None:
    """Under their grammar (`grammar.tools`) the model's own call comes back as it wrote it; prose cannot: the fake's
    samples, masked, become a call (cut by max_tokens here)."""
    named = {"type": "function", "function": {"name": "get_weather"}}
    r = ask(server, "single", tool_choice=named)
    assert calls_of(r["choices"][0]["message"]) == expected_calls(server, "single")
    assert r["choices"][0]["message"]["content"] == "" and r["choices"][0]["finish_reason"] == "stop"
    r = ask(server, "single", tool_choice="required")
    assert calls_of(r["choices"][0]["message"]) == expected_calls(server, "single")
    assert r["choices"][0]["finish_reason"] == "tool_calls" and r["choices"][0]["message"]["content"] == ""
    r = ask(server, "text", tool_choice="required", max_tokens=64)
    msg = r["choices"][0]["message"]
    assert msg["content"] == "" and r["choices"][0]["finish_reason"] in ("tool_calls", "length")
    assert msg["tool_calls"] and all(c["function"]["name"] in ("get_weather", "get_time") or r["choices"][0][
        "finish_reason"] == "length" for c in msg["tool_calls"])


def test_stop_strings(server) -> None:
    """A stop string in content, inside a call's arguments, and next to a marker: the parse sees the output the stop
    string leaves (what the client's text would hold, the markers included), and the finish is `stop` (with the
    truncated call, `tool_calls`)."""
    r = ask(server, "text", stop=["sunny"], seed=SEEDS["text"] + 3000)
    assert r["choices"][0]["message"]["content"] == "It is " and r["choices"][0]["finish_reason"] == "stop"
    r = ask(server, "single", stop=["Paris"], seed=SEEDS["single"] + 1000)
    assert calls_of(r["choices"][0]["message"]) == [("get_weather", '{"city": ""}')]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    marker = "call:" if server.family == "gemma" else "<function="
    r = ask(server, "single", stop=[marker], seed=SEEDS["single"] + 2000)
    assert "tool_calls" not in r["choices"][0]["message"] and r["choices"][0]["finish_reason"] == "stop"
    assert r["choices"][0]["message"]["content"] is None
    for case, stop, seed in (("text", ["sunny"], SEEDS["text"] + 3000), ("single", ["Paris"], SEEDS["single"] + 1000)):
        events = stream(server, case, stop=stop, seed=seed)
        streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:])
        assert streamed == (ask(server, case, stop=stop, seed=seed)["choices"][0]["message"]["content"] or "")


def test_no_parser_without_tools(server) -> None:
    """Requests without tools build no parser: the metric's counters stay at 0, while one with tools counts."""
    def counts() -> float:
        return server.metric("vllm:tool_call_parser_invocations_total")
    before = counts()
    for stream_ in (False, True):
        code, _ = server.post(CHAT, {"messages": MSG, "max_tokens": 3, "stream": stream_})
        assert code == 200
    assert counts() == before
    ask(server, "single")
    assert counts() == before + 1

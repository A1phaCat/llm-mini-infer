"""Reasoning end to end: the real server (launcher, engine and frontend processes) on the fake runner, Gemma 4's and
Qwen3.8's tokenizers and templates, without and with MTP, each with its family's reasoning parser on by default.

The fake replays the reasoning vectors' outputs (tests/data/reasoning/vectors.json: the reference vLLM's parse,
edited where this server deviates) after a prompt that opens the vector's case: the message's `reasoning`, `content`
and calls, the streamed deltas accumulated, each call's argument pieces one token a step, and `reasoning_tokens` in the
usage equal the vector's. Then the request's rules: `include_reasoning` false, `reasoning_effort` through the effort
map (the prompt it renders), its 400s and the template's, `thinking_token_budget` refused, history under both names,
/tokenize, a stop string inside a thought or across a marker (the count of the accepted ids parsed whole), the closer
inside one MTP step, logprobs over a thought, the schema and calls enforced from the first token with thinking on, and
an ignored `response_format`'s warning once a request."""

from __future__ import annotations

import json

import pytest

from llm_mini_infer.api.tools import template_tool
from llm_mini_infer.output.detokenizer import Detokenizer
from llm_mini_infer.output.reasoning import ReasoningFamily, ReasoningParser, thinking_flag
from llm_mini_infer.output.tool_calls import Family, ToolCallParser
from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.render.renderer import Renderer

from ..support.api_util import Server, env_has_model
from ..support.fixtures import QWEN
from ..support.paths import DATA as TESTS_DATA
from ..support.paths import MODELS_DIR

CHAT = "/v1/chat/completions"
GEMMA = MODELS_DIR / "gemma-4-31b-it"
DATA = json.loads((TESTS_DATA / "reasoning" / "vectors.json").read_text())
FAMILY = {"gemma": "gemma4", "qwen": "qwen3"}
USER = [{"role": "user", "content": "What is 2 + 2?"}]
AFTER_TOOL = [{"role": "user", "content": "Weather in Paris?"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                  "name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
STOPS = {"gemma4": {1, 106, 50}, "qwen3": {248044, 248046}}
FUZZED = 80  # the fuzz sample's first vectors replayed, with every written one


def replayed() -> list[dict]:
    """The vectors the server replays: both parsers (the server's defaults), no grammar (`tool_choice` absent, auto or
    none: the fake's scripted tokens are masked like its samples), an output the engine can give (a stop id last if
    any)."""
    out, fuzzed = [], 0
    for i, v in enumerate(DATA["vectors"]):
        if v["combo"] != "both" or v["choice"] not in (None, "auto", "none") or not v["ids"] or any(
                t in STOPS[v["fam"]] for t in v["ids"][:-1]):
            continue
        if v["name"].startswith("fuzz"):
            fuzzed += 1
            if fuzzed > FUZZED:
                continue
        out.append({**v, "seed": 30_000 + i})
    return out


VECTORS = replayed()


def messages_for(v: dict) -> list[dict]:
    """A conversation whose prompt opens the vector's case: Gemma's c and d follow a tool result."""
    return AFTER_TOOL if v["fam"] == "gemma4" and v["case"] in ("c", "d") else USER


def body_for(v: dict, **extra) -> dict:
    body = {"messages": messages_for(v), "seed": v["seed"], "chat_template_kwargs": v["kwargs"],
            "skip_special_tokens": v["skip_special"],
            "max_tokens": len(v["ids"]) + (4 if v["ids"] and v["ids"][-1] in STOPS[v["fam"]] else 0)}
    if v["tools"]:
        body["tools"] = DATA["tools"][v["tools"]]
    if v["choice"] is not None:
        body["tool_choice"] = v["choice"]
    return {**body, **extra}


def prompt_of(r: Renderer, msgs: list[dict], tools: list[dict] | None = None, **kw) -> list[int]:
    conv = to_template_messages(msgs, r.content_format, r.supports_developer_role)
    return r.encode(r.render(conv, tools=[template_tool(t) for t in tools] if tools else None, **kw)).tolist()


# The scripted outputs of the request tests below, after USER (by seed): thought then text, and a thought alone
GEMMA_THOUGHT = "<|channel>thought\nLet me think.<channel|>It is 4."
QWEN_THOUGHT = "Let me think.\n</think>\n\nIt is 4."
PLAN = {"gemma": "<|channel>thought\nPlan.<channel|>", "qwen": "Plan.\n</think>\n\n"}
SEED_THOUGHT, SEED_PLAN, SEED_SPAN, SEED_STEP = 40_000, 41_000, 42_000, 43_000
STEP_SEEDS = 12  # the thought scripted again, so that the fake's MTP drafts put the closer inside steps
DELIM = '<|"|>'
GEMMA_CALL = f"<|tool_call>call:get_weather{{city:{DELIM}Paris{DELIM}}}<tool_call|>"
QWEN_CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
# Stop strings across the markers, by family: (thinking, with TOOLS, the output, the stop strings).
# Thinking off without tools takes the lazy path (the parser built at the first marker)
SPANS = {
    "gemma": [(True, False, GEMMA_THOUGHT, ["<|channel>thought\nLet", "ought\nLet", "think.<channel|>It"]),
              (True, True, f"<|channel>thought\nPlan{GEMMA_CALL}", ["Plan<|tool_call>call"]),
              (True, False, "thought\nLet me think.<channel|>It is 4.", ["thought"]),
              (False, False, "Before <|channel>thought\nHm, ok<channel|>After", ["<|channel>thought\nHm"]),
              (False, False, "thought\nHm<channel|>After", ["thought"])],
    "qwen": [(True, False, QWEN_THOUGHT, ["\n</think>\n\nIt"]),
             (True, True, f"Let me think.\n{QWEN_CALL}", ["think.\n<tool_call>\n<function"]),
             (False, False, "Hmm</think>\n\nOk", ["m</think>\n"])],
}


def start(model_dir, family: str, tmp, mtp: int, extra: dict | None = None) -> Server:
    """The server, its fake scripted with the family's vectors whose case `messages_for` opens (`srv.vectors`)."""
    r = Renderer(model_dir)
    fam = FAMILY[family]
    rf = ReasoningFamily(fam, r.tokenizer)
    scripts, vectors = [], []
    for v in VECTORS:
        if v["fam"] == fam:
            prompt = prompt_of(r, messages_for(v), DATA["tools"][v["tools"]] if v["tools"] else None, **v["kwargs"])
            if rf.case(prompt, thinking_flag(fam, v["kwargs"])) == v["case"]:  # not a custom template's prompt
                scripts.append([v["seed"], len(prompt), v["ids"]])
                vectors.append(v)
    on = {"enable_thinking": True}
    stop = 106 if family == "gemma" else 248046
    text = GEMMA_THOUGHT if family == "gemma" else QWEN_THOUGHT
    toks = [*r.tokenizer.encode(text, add_special_tokens=False).ids, stop]
    scripts.append([SEED_THOUGHT, len(prompt_of(r, USER, **on)), toks])
    scripts.append([SEED_THOUGHT + 1, len(prompt_of(r, USER, TOOLS, **on)), toks])  # with the tools rendered
    plan = r.tokenizer.encode(PLAN[family], add_special_tokens=False).ids
    for k in range(8):  # the thinking-on grammars: a thought scripted, the rest the fake's masked samples
        scripts.append([SEED_PLAN + k, len(prompt_of(r, USER, **on)), plan])
        scripts.append([SEED_PLAN + 100 + k, len(prompt_of(r, USER, TOOLS, **on)), plan])
    for i, (think, tools, out, _) in enumerate(SPANS[family]):
        scripts.append([SEED_SPAN + i, len(prompt_of(r, USER, TOOLS if tools else None, enable_thinking=think)),
                        [*r.tokenizer.encode(out, add_special_tokens=False).ids, stop]])
    for k in range(STEP_SEEDS):
        scripts.append([SEED_STEP + k, len(prompt_of(r, USER, **on)), toks])
    cfg = {"runner": {"kind": "fake"}, "server": {"logprobs": True}, "dev": {"runner": {"fake": {"scripts": scripts}}},
           # compact JSON: the fake's masked samples would otherwise take whitespace without end
           "structured_output": {"backend": "guidance", "disable_any_whitespace": True}}
    if family == "gemma":
        cfg["images"] = {"enabled": False}
        if mtp:
            (tmp / "assistant").mkdir()
            (tmp / "assistant" / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant",
                                                                       "text_config": {"model_type": "gemma4_text"}}))
            cfg["speculative"] = {"model": str(tmp / "assistant")}
    for key, value in (extra or {}).items():
        cfg[key] = {**cfg.get(key, {}), **value}
    srv = Server(model_dir, tmp, step_ms=0.2, mtp=mtp, extra=cfg).start(120)
    srv.family, srv.fam, srv.r, srv.mtp, srv.toks, srv.vectors = family, fam, r, mtp, toks, vectors
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


def post(srv: Server, body: dict) -> dict:
    code, r = srv.post(CHAT, body)
    assert code == 200, r
    return r


def streamed(srv: Server, body: dict) -> dict:
    """The stream accumulated as a client does: reasoning, content, calls (name, arguments, pieces), the finish, the
    usage, the logprob entries and token ids, and the chunks."""
    events = [e for e in srv.stream({**body, "stream_options": {"include_usage": True}}) if isinstance(e, dict)]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    out = {"reasoning": "", "content": "", "calls": {}, "lp": 0, "ids": [], "chunks": events}
    for e in events[1:-1]:
        c = e["choices"][0]
        d = c["delta"]
        out["reasoning"] += d.get("reasoning") or ""
        out["content"] += d.get("content") or ""
        out["lp"] += len((c.get("logprobs") or {}).get("content") or [])
        out["ids"] += c.get("token_ids") or []
        for tc in d.get("tool_calls", []):
            call = out["calls"].setdefault(tc["index"], ["", "", []])
            call[0] += tc["function"].get("name", "")
            if "arguments" in tc["function"]:
                call[1] += tc["function"]["arguments"]
                call[2].append(tc["function"]["arguments"])
    out["finish"] = events[-2]["choices"][0]["finish_reason"]
    out["usage"] = events[-1]["usage"]
    return out


def test_the_replies_are_vllms(server) -> None:
    """Every vector of the family (the written ones and a fuzz sample), streamed and not."""
    assert len(server.vectors) >= 50
    failed = []
    for v in server.vectors:
        try:
            replay(server, v)
        except AssertionError as exc:
            failed.append(f"{v['name']}: {str(exc)[:300]}")
    assert not failed, "\n".join(failed)


def replay(server: Server, v: dict) -> None:
    r = post(server, body_for(v))
    msg = r["choices"][0]["message"]
    calls = [[c["function"]["name"], c["function"]["arguments"]] for c in msg.get("tool_calls", [])]
    assert (msg["reasoning"], msg["content"], calls) == (v["reasoning"], v["content"], v["calls"]), v["edits"]
    assert msg.get("tool_calls", None) != []  # only with calls
    assert r["usage"]["completion_tokens_details"] == {"reasoning_tokens": v["rt"]}
    s = streamed(server, body_for(v))
    got = [[c[0], c[1]] for _, c in sorted(s["calls"].items())]
    assert (s["reasoning"] or None, s["content"] or None, got) == (v["reasoning"] or None, v["stream_content"] or None,
                                                                    v["stream_calls"])
    assert s["usage"]["completion_tokens_details"] == {"reasoning_tokens": v["rt"]}
    if server.mtp == 0 and "1" in v["pieces"]:  # one token a step: vLLM's own pieces
        assert [c[2] for _, c in sorted(s["calls"].items())] == v["pieces"]["1"]


# -- the request -------------------------------------------------------------------------------------------

def thought_body(srv: Server, **extra) -> dict:
    return {"messages": USER, "seed": SEED_THOUGHT, "max_tokens": len(srv.toks) + 4,
            "chat_template_kwargs": {"enable_thinking": True}, **extra}


def test_a_thought_with_logprobs_and_token_ids(server) -> None:
    """The thought's tokens and its markers carry their logprob entries and ids, streamed and not, each once (a chunk
    with an empty delta where a step's text is held or a marker's)."""
    r = post(server, thought_body(server, logprobs=True, return_token_ids=True))
    c = r["choices"][0]
    assert c["message"]["reasoning"] == "Let me think." + ("" if server.family == "gemma" else "\n")
    assert c["message"]["content"] == ("It is 4." if server.family == "gemma" else "\n\nIt is 4.")
    assert len(c["logprobs"]["content"]) == len(server.toks) and c["token_ids"] == server.toks
    want_rt = len(server.r.tokenizer.encode("thought\nLet me think." if server.family == "gemma" else
                                            "Let me think.\n", add_special_tokens=False).ids)
    assert r["usage"]["completion_tokens_details"] == {"reasoning_tokens": want_rt}
    s = streamed(server, thought_body(server, logprobs=True, return_token_ids=True))
    assert (s["reasoning"], s["lp"], s["ids"]) == (c["message"]["reasoning"], len(server.toks), server.toks)
    assert s["usage"]["completion_tokens_details"] == {"reasoning_tokens": want_rt}


def test_qwen_thinks_by_default(server) -> None:
    """Qwen3.8 without kwargs: its template opens a thought, so the output is reasoning, and content stays null until
    `</think>`."""
    if server.family != "qwen":
        pytest.skip("Qwen3.8's default")
    body = {"messages": USER, "seed": SEED_THOUGHT, "max_tokens": 3}
    msg = post(server, body)["choices"][0]["message"]
    assert (msg["reasoning"], msg["content"]) == (server.r.tokenizer.decode(server.toks[:3]), None)
    s = streamed(server, body)
    assert (s["reasoning"], s["content"]) == (msg["reasoning"], "")
    msg = post(server, {**body, "max_tokens": len(server.toks) + 4})["choices"][0]["message"]
    assert (msg["reasoning"], msg["content"]) == ("Let me think.\n", "\n\nIt is 4.")


def test_include_reasoning_false(server) -> None:
    """No reasoning (null, and none streamed: content chunks carry `"reasoning": null`), logprobs and token ids hidden,
    the prompt's ids kept, the count still made (the reference vLLM's `hide_metadata`)."""
    extra = {"include_reasoning": False, "logprobs": True, "return_token_ids": True}
    r = post(server, thought_body(server, **extra))
    c = r["choices"][0]
    assert c["message"]["reasoning"] is None and c["message"]["content"]
    assert c["logprobs"] is None and "token_ids" not in c
    assert r["prompt_token_ids"] == prompt_of(server.r, USER, enable_thinking=True)
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0
    s = streamed(server, thought_body(server, **extra))
    assert s["reasoning"] == "" and s["content"] == (c["message"]["content"] or "") and s["lp"] == 0 and not s["ids"]
    content_chunks = [e["choices"][0]["delta"] for e in s["chunks"][1:-1] if e["choices"][0]["delta"].get("content")]
    assert content_chunks and all(d["reasoning"] is None for d in content_chunks)


@pytest.mark.parametrize("effort", ["none", "minimal", "medium", "high", "max", None])
def test_reasoning_effort_renders_through_the_map(server, effort) -> None:
    """The effort onto the template's values (Qwen3.8: minimal and low to low, high, xhigh and max to xhigh; Gemma's map
    empty), `enable_thinking` from it (`none` off), the request's chat_template_kwargs over both."""
    qwen = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
    body = {"messages": USER, "max_tokens": 1, "return_token_ids": True}
    if effort is not None:
        body["reasoning_effort"] = effort
    kw = {} if effort is None else {"reasoning_effort": qwen.get(effort, effort) if server.family == "qwen" else
                                    effort, "enable_thinking": effort != "none"}
    assert post(server, body)["prompt_token_ids"] == prompt_of(server.r, USER, **kw)
    if effort not in (None, "none"):
        body["chat_template_kwargs"] = {"enable_thinking": False}  # the request's own kwargs win
        assert post(server, body)["prompt_token_ids"] == prompt_of(server.r, USER, **{**kw, "enable_thinking": False})


@pytest.mark.parametrize("value", ["bogus", 5, True, ["high"], "HIGH"])
def test_an_effort_outside_the_set_is_vllms_400(server, value) -> None:
    code, r = server.post(CHAT, {"messages": USER, "max_tokens": 1, "reasoning_effort": value})
    shown = value if isinstance(value, str) else "<list of 1 items>" if isinstance(value, list) else str(value)
    efforts = "'none', 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max'"
    want = ("1 validation error:\n  {'type': 'literal_error', 'loc': 'body.reasoning_effort', 'msg': \"Input should be "
            f"{efforts}\", 'input': '{shown}', 'ctx': {{'expected': \"{efforts}\"}}}}")
    assert (code, r["error"]["message"], r["error"]["param"], r["error"]["type"]) == (400, want,
                                                                                     "body.reasoning_effort",
                                                                                     "Bad Request")


def test_thinking_token_budget_is_refused(server) -> None:
    code, r = server.post(CHAT, {"messages": USER, "max_tokens": 1, "thinking_token_budget": 64})
    assert code == 400 and "thinking_token_budget" in r["error"]["message"], r


def test_history_takes_both_names(server) -> None:
    """An assistant turn's thought under `reasoning` or `reasoning_content` renders alike; /tokenize gives the chat
    path's ids."""
    def history(name: str) -> list[dict]:
        turn = {**AFTER_TOOL[1], name: "I need the weather."}
        return [AFTER_TOOL[0], turn, AFTER_TOOL[2]]
    ids = []
    for name in ("reasoning", "reasoning_content"):
        body = {"messages": history(name), "max_tokens": 1, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True}}
        ids.append(post(server, body)["prompt_token_ids"])
        code, tok = server.post("/tokenize", {"messages": history(name),
                                              "chat_template_kwargs": {"enable_thinking": True}})
        assert code == 200 and tok["tokens"] == ids[-1]
    assert ids[0] == ids[1]
    if server.family == "qwen":  # Qwen3.8 renders the thought after the last user turn
        assert ids[0] != prompt_of(server.r, AFTER_TOOL, enable_thinking=True)


def test_a_stop_string_inside_the_thought(server) -> None:
    """The reply ends there: the reasoning before it, no content, `stop`; counted up to the token that completed it."""
    for stream in (False, True):
        body = thought_body(server, stop=["think."])
        if stream:
            s = streamed(server, body)
            reasoning, content, finish, usage = s["reasoning"], s["content"] or None, s["finish"], s["usage"]
        else:
            r = post(server, body)
            msg = r["choices"][0]["message"]
            reasoning, content, finish, usage = (msg["reasoning"], msg["content"], r["choices"][0]["finish_reason"],
                                                 r["usage"])
        cut = "Let me" if server.family == "gemma" else "Let me "  # Gemma's trailing whitespace dropped
        assert (reasoning, content, finish) == (cut, None, "stop"), stream
        assert usage["completion_tokens_details"]["reasoning_tokens"] == usage["completion_tokens"] - (
            1 if server.family == "gemma" else 0)  # Gemma's opener is not counted


def whole_count(srv: Server, kw: dict, tools: list[dict] | None, ids: list[int]) -> int:
    """`reasoning_tokens` of these output ids parsed whole and unstopped, after USER's prompt (`output.reasoning`)."""
    prompt = prompt_of(srv.r, USER, tools, **kw)
    tok = srv.r.tokenizer
    rf = ReasoningFamily(srv.fam, tok)
    inner = None
    if tools:
        tf = Family("gemma4" if srv.fam == "gemma4" else "qwen3_coder", tok)
        inner = ToolCallParser(tf, {t["function"]["name"]: t["function"]["parameters"] for t in tools}, "auto")
    p = ReasoningParser(rf, rf.case(prompt, thinking_flag(srv.fam, kw)), inner)
    d = Detokenizer(tok, prompt, skip_special_tokens=False)
    for t in ids:
        p.feed(t, d.step(t))
    p.finish()
    return p.tokens


def replied(srv: Server, body: dict, stream: bool) -> tuple:
    """The reasoning, the content, the finish, the usage and the output ids, streamed or not."""
    if stream:
        s = streamed(srv, body)
        return s["reasoning"] or None, s["content"] or None, s["finish"], s["usage"], s["ids"]
    r = post(srv, body)
    c = r["choices"][0]
    return c["message"]["reasoning"], c["message"]["content"], c["finish_reason"], r["usage"], c["token_ids"]


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_a_stop_string_across_a_marker_counts_the_accepted_ids(server, stream) -> None:
    """A stop string across the opener, the label, the closer or a call's opener, or cutting a first token `thought`,
    with the thought's case and on the lazy path: `reasoning_tokens` is the count of the ids the engine accepted, parsed
    whole, with the stop string kept or not."""
    failed = []
    for i, (think, with_tools, out, stops) in enumerate(SPANS[server.family]):
        kw, tools = {"enable_thinking": think}, TOOLS if with_tools else None
        whole = len(server.r.tokenizer.encode(out, add_special_tokens=False).ids) + 1  # the scripted ids, the stop id
        for stop in stops:
            for include in (False, True):
                body = {"messages": USER, "seed": SEED_SPAN + i, "max_tokens": 64, "chat_template_kwargs": kw,
                        "stop": [stop], "include_stop_str_in_output": include, "return_token_ids": True,
                        **({"tools": tools} if tools else {})}
                _, _, finish, usage, ids = replied(server, body, stream)
                got = (finish, usage["completion_tokens"], usage["completion_tokens_details"]["reasoning_tokens"],
                       len(ids) < whole)
                want = ("stop", len(ids), whole_count(server, kw, tools, ids), True)  # the stop string cut it
                if got != want:
                    failed.append(f"{stop!r} include {include}: {got} != {want}")
    assert not failed, failed


def test_the_closer_inside_one_step(server) -> None:
    """MTP: the fake's drafts accept the closer with other tokens in one step (seen in the stream's token ids, a step
    a chunk) for some of the seeds; every seed gives the same reasoning, content and count, streamed and not."""
    if not server.mtp:
        pytest.skip("one token a step")
    end = ReasoningFamily(server.fam, server.r.tokenizer).end
    want = (("Let me think.", "It is 4.") if server.family == "gemma" else ("Let me think.\n", "\n\nIt is 4."))
    rt = len(server.r.tokenizer.encode("thought\nLet me think." if server.family == "gemma" else "Let me think.\n",
                                       add_special_tokens=False).ids)
    held = 0
    for k in range(STEP_SEEDS):
        body = {"messages": USER, "seed": SEED_STEP + k, "max_tokens": len(server.toks) + 4,
                "chat_template_kwargs": {"enable_thinking": True}, "return_token_ids": True}
        s = streamed(server, body)
        steps = [c["choices"][0].get("token_ids") or [] for c in s["chunks"][1:-1]]
        held += any(end in step and len(step) > 1 for step in steps)
        assert [t for step in steps for t in step] == server.toks, k
        for stream in (False, True):
            reasoning, content, finish, usage, _ = replied(server, body, stream)
            assert (reasoning, content, finish) == (*want, "stop"), (k, stream)
            assert usage["completion_tokens_details"]["reasoning_tokens"] == rt, (k, stream)
    assert held, "no step held the closer with other tokens"


def test_a_stop_string_in_the_closers_step(server) -> None:
    """MTP: a stop string across the closer, completed in a step that also holds the closer (the stream's last chunk
    carries that step's ids) for some of the seeds: the thought before it, no content, `stop`, and the count of the
    accepted ids parsed whole, with the stop string kept or not."""
    if not server.mtp:
        pytest.skip("one token a step")
    end = ReasoningFamily(server.fam, server.r.tokenizer).end
    stop = "<channel|>It" if server.family == "gemma" else "</think>\n\nIt"
    kw = {"enable_thinking": True}
    held = 0
    for k in range(STEP_SEEDS):
        for include in (False, True):
            body = {"messages": USER, "seed": SEED_STEP + k, "max_tokens": len(server.toks) + 4,
                    "chat_template_kwargs": kw, "stop": [stop], "include_stop_str_in_output": include,
                    "return_token_ids": True}
            s = streamed(server, body)
            last = s["chunks"][-2]["choices"][0].get("token_ids") or []
            held += end in last and len(last) > 1
            for stream in (False, True):
                reasoning, content, finish, usage, ids = replied(server, body, stream)
                want_content = (None if not include else "It" if server.family == "gemma" else "\n\nIt")
                assert (reasoning, content, finish) == ("Let me think." + ("" if server.family == "gemma" else "\n"),
                                                         want_content, "stop"), (k, include, stream)
                assert usage["completion_tokens_details"]["reasoning_tokens"] == whole_count(server, kw, None, ids)
    assert held, "no stop string completed in a step that held the closer"


def test_an_ignored_response_format_warns_once_a_request(server) -> None:
    """`required` with a `response_format`, its grammars built on the compile threads (a miss each: new tools),
    thinking off and on: the server logs its warning once a request."""
    if server.mtp:
        pytest.skip("the frontend's log: one server a family")
    for think in (False, True):
        tag = f"warn-once-{think}"
        tools = [{"type": "function", "function": {"name": f"w_{int(think)}", "parameters": {
            "type": "object", "properties": {"a": {"enum": ["x"]}}, "required": ["a"], "additionalProperties": False}}}]
        post(server, {"messages": USER, "max_tokens": 4, "tools": tools, "tool_choice": "required", "user": tag,
                      "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": think}})
        line = f"request {tag}: tool_choice required ignores its response_format"
        assert server.log_path.read_text().count(line) == 1, think


# -- the grammars with thinking on ------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"k": {"enum": ["a", "b"]}, "n": {"type": "integer", "minimum": 0,
                                                                              "maximum": 9}},
          "required": ["k", "n"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"enum": ["Paris", "Rome"]}}, "required": ["city"],
    "additionalProperties": False}}}]


def test_the_schema_follows_the_thought(server) -> None:
    """A json_schema reply with thinking on: the scripted thought, its closer, then the fake's masked samples, which
    the grammar makes the schema's JSON (the thought free)."""
    done = 0
    for k in range(8):
        r = post(server, {"messages": USER, "seed": SEED_PLAN + k, "max_tokens": 64,
                          "chat_template_kwargs": {"enable_thinking": True},
                          "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}}})
        c = r["choices"][0]
        assert c["message"]["reasoning"] == ("Plan." if server.family == "gemma" else "Plan.\n")
        if c["finish_reason"] == "length":
            continue
        doc = json.loads(c["message"]["content"])
        assert doc["k"] in ("a", "b") and 0 <= doc["n"] <= 9 and set(doc) == {"k", "n"}
        done += 1
    assert done


@pytest.mark.parametrize("choice", ["required", "named"])
def test_a_call_follows_the_thought(server, choice) -> None:
    named = {"type": "function", "function": {"name": "get_weather"}}
    done = 0
    for k in range(8):
        r = post(server, {"messages": USER, "seed": SEED_PLAN + 100 + k, "max_tokens": 64, "tools": TOOLS,
                          "tool_choice": named if choice == "named" else "required",
                          "parallel_tool_calls": False,  # one call: the fake's masked samples would add more
                          "chat_template_kwargs": {"enable_thinking": True}})
        c = r["choices"][0]
        assert c["message"]["reasoning"] == ("Plan." if server.family == "gemma" else "Plan.\n")
        if c["finish_reason"] == "length":
            continue
        calls = c["message"]["tool_calls"]
        assert calls and all(json.loads(x["function"]["arguments"])["city"] in ("Paris", "Rome") for x in calls)
        done += 1
    assert done


@pytest.mark.serial  # a server of its own beside the module's: three of Gemma's are 4 GB
def test_the_template_refuses_an_effort_the_map_gives(tmp_path) -> None:
    """A map onto a value the template refuses: its 400, the template's own message."""
    if not env_has_model(QWEN):
        pytest.skip(f"{QWEN} is absent: uv run tests/scripts/fetch_fixtures.py")
    srv = start(QWEN, "qwen", tmp_path, 0, {"reasoning": {"effort_map": {"high": "bogus"}}})
    try:
        code, r = srv.post(CHAT, {"messages": USER, "max_tokens": 1, "reasoning_effort": "high"})
        assert code == 400 and "Unexpected reasoning effort bogus" in r["error"]["message"], r
        code, _ = srv.post(CHAT, {"messages": USER, "max_tokens": 1, "reasoning_effort": "low"})
        assert code == 200
    finally:
        assert srv.stop() == 0


@pytest.mark.serial  # a server of its own beside the module's: three of Gemma's are 4 GB
@pytest.mark.parametrize("family", ["gemma", "qwen"])
@pytest.mark.parametrize("parsers", [("none", None), (None, "none"), ("none", "none")],
                         ids=["no-reasoning", "no-tools", "neither"])
def test_each_combination_of_the_parsers(family, parsers, tmp_path) -> None:
    """`reasoning.parser` and `tool_calls.parser`, each on or off: without a reasoning parser the thought is content
    (Gemma's markers dropped as special tokens, Qwen's kept), `reasoning` null and the usage without details;
    without a tool parser the thought is still parsed, and a request with tools under `auto` is a 400."""
    model_dir = GEMMA if family == "gemma" else QWEN
    if not env_has_model(model_dir):
        pytest.skip(f"{model_dir} is absent: uv run tests/scripts/fetch_fixtures.py")
    extra = {}
    if parsers[0]:
        extra["reasoning"] = {"parser": parsers[0]}
    if parsers[1]:
        extra["tool_calls"] = {"parser": parsers[1]}
    srv = start(model_dir, family, tmp_path, 0, extra)
    try:
        r = post(srv, thought_body(srv))
        msg = r["choices"][0]["message"]
        s = streamed(srv, thought_body(srv))
        if parsers[0] == "none":
            text = srv.r.tokenizer.decode(srv.toks[:-1], skip_special_tokens=True)
            assert (msg["reasoning"], msg["content"], r["usage"]["completion_tokens_details"]) == (None, text, None)
            assert (s["reasoning"], s["content"]) == ("", text) and "completion_tokens_details" not in s["usage"]
        else:
            assert msg["reasoning"] == ("Let me think." if family == "gemma" else "Let me think.\n")
            assert s["reasoning"] == msg["reasoning"] and s["usage"]["completion_tokens_details"]["reasoning_tokens"]
        with_tools = {**thought_body(srv), "tools": TOOLS, "seed": SEED_THOUGHT + 1}
        code, err = srv.post(CHAT, with_tools)
        if parsers[1] == "none":
            assert code == 400 and "tool_calls.parser" in err["error"]["message"], err
            with_tools["tool_choice"] = "none"
        else:
            assert code == 200, err
        assert post(srv, with_tools)["choices"][0]["message"]["reasoning"] == msg["reasoning"]
    finally:
        assert srv.stop() == 0

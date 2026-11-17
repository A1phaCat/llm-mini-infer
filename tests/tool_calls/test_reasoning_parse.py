"""The reasoning parse (`output.reasoning` in front of `output.tool_calls`) and a call's arguments in pieces
(`output.tool_args.ArgStream`) against the reference vLLM's on vectors (tests/data/reasoning/vectors.json: the Gemma 4
and Qwen3.8 reasoning cases, ids G1-G7 and Q1-Q5, edge cases and a fuzz sample, run through the reference vLLM's own
parser engine and serving helpers on the real tokenizers, its stream at one token a step, edited where this server
deviates).
Each vector's ids go to the parser one token at a time, its deltas taken every 1, 2, 3, 5, 7, 11, 23 or 1,000 tokens:
the reasoning, content, calls and reasoning tokens equal the expected ones streamed and not, and so do the argument
pieces vLLM sends at each size. Then the cases on the real templates, the effort map, the count under a stop
string (across each marker too: the accepted ids as the parse counts them whole), whitespace around a call (the
reference's content, streamed and not), and the lazy path's chunks (the eager parse's)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

from llm_mini_infer.api.engine_client import RequestState
from llm_mini_infer.api.sse import ChunkTemplate
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.output import reasoning as reasoning_module
from llm_mini_infer.output import tool_args, tool_calls
from llm_mini_infer.output.detokenizer import Detokenizer
from llm_mini_infer.output.logprobs import RequestLogprobs, TokenText
from llm_mini_infer.output.reasoning import (
    EFFORT_MAPS,
    EFFORTS,
    GEMMA4,
    QWEN3,
    ReasoningFamily,
    ReasoningParser,
    check_effort_map,
    effort_kwargs,
    missing_tokens,
    template_format,
    thinking_flag,
)
from llm_mini_infer.output.stop import OutputText, StopStrings
from llm_mini_infer.output.tool_calls import QWEN3 as QWEN3_CODER
from llm_mini_infer.output.tool_calls import Family, ToolCallParser, ToolFeed
from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.render.renderer import Renderer

from ..support.fixtures import model_dir
from ..support.paths import DATA as TESTS_DATA
from ..support.timing import least_each

DATA = json.loads((TESTS_DATA / "reasoning" / "vectors.json").read_text())
MODELS = {GEMMA4: "gemma-4-31b-it", QWEN3: "qwen3.8-27b"}
TOOL_FAMILY = {GEMMA4: "gemma4", QWEN3: QWEN3_CODER}
STOPS = {GEMMA4: {1, 106, 50}, QWEN3: {248046, 248044}}
_fams: dict[str, tuple] = {}


def families(fam: str) -> tuple[Tokenizer, ReasoningFamily, Family]:
    if fam not in _fams:
        path = model_dir(MODELS[fam]) / "tokenizer.json"
        if not path.exists():
            pytest.skip(f"{path} is absent: uv run tests/scripts/fetch_fixtures.py")
        tok = Tokenizer.from_file(str(path))
        _fams[fam] = (tok, ReasoningFamily(fam, tok), Family(TOOL_FAMILY[fam], tok))
    return _fams[fam]


def pieces_of(fam: str, prompt: list[int], ids: list[int]) -> list[str]:
    """Each output token's text as the server detokenizes it: primed by the prompt, special tokens kept, the stop
    token's text none."""
    tok = families(fam)[0]
    d = Detokenizer(tok, prompt, skip_special_tokens=False)
    out = [d.step(t) for t in ids]
    if ids and ids[-1] in STOPS[fam]:
        out[-1] = ""
    return out


def parser_for(v: dict, prompt: list[int]):
    _, rf, tf = families(v["fam"])
    tools = DATA["tools"].get(v["tools"]) if v["tools"] else None
    choice = v["choice"] or ("auto" if tools else None)
    inner = None
    if tools and v["combo"] in ("both", "tool"):
        schemas = {t["function"]["name"]: t["function"].get("parameters") for t in tools}
        inner = ToolCallParser(tf, schemas, choice, True, v["skip_special"] if choice == "none" else False)
    return ReasoningParser(rf, rf.case(prompt, thinking_flag(v["fam"], v["kwargs"])), inner, v["skip_special"])


def run(v: dict, per_step: int | None) -> dict:
    """The vector through our parse, its deltas taken every `per_step` tokens (the last step's with the finish, as the
    server sends them), or the non-streamed message's fields (None)."""
    prompt, ids = v["prompt_tail"], v["ids"]
    p = parser_for(v, prompt)
    reasoning, content, calls = "", "", {}

    def take() -> None:
        nonlocal reasoning, content
        r, c, t = p.take()
        reasoning += r
        content += c
        for e in t:
            got = calls.setdefault(e["index"], ["", "", []])
            got[0] += e["function"].get("name", "")
            if "arguments" in e["function"]:
                got[1] += e["function"]["arguments"]
                got[2].append(e["function"]["arguments"])

    for k, (t, piece) in enumerate(zip(ids, pieces_of(v["fam"], prompt, ids), strict=True)):
        p.feed(t, piece)
        if per_step is not None and (k + 1) % per_step == 0 and k + 1 < len(ids):
            take()
    p.finish()
    if per_step is None:
        return {"reasoning": p.reasoning(), "content": p.content(), "rt": p.tokens,
                "calls": [[c.name.strip(), c.arguments] for c in p.kept_calls()]}
    take()
    return {"reasoning": reasoning or None, "content": content or None, "rt": p.tokens,
            "calls": [[calls[i][0].strip(), calls[i][1]] for i in sorted(calls)],
            "pieces": [calls[i][2] for i in sorted(calls)]}


def test_the_parse_reads_the_tool_parsers_own_states(monkeypatch) -> None:
    """Whether a call is open is the tool parser's state, compared with the tool parser's own constant: with its
    states renumbered, every vector with tools parses as before, streamed and not."""
    vectors = [v for v in DATA["vectors"] if v["tools"] and v["combo"] in ("both", "tool")]
    want = [(run(v, None), run(v, 3)) for v in vectors]
    monkeypatch.setattr(tool_calls, "CONTENT", 5)  # past BETWEEN (4): no other state's value
    assert ToolCallParser(families(GEMMA4)[2], {}, "auto", True).state == 5
    assert [(run(v, None), run(v, 3)) for v in vectors] == want


@pytest.mark.parametrize("v", DATA["vectors"], ids=[v["name"] for v in DATA["vectors"]])
def test_the_parse_is_vllms(v) -> None:
    got = run(v, None)
    want = {"reasoning": v["reasoning"], "content": v["content"], "rt": v["rt"], "calls": v["calls"]}
    assert got == want, v["edits"]
    for k in DATA["chunks"]:
        s = run(v, k)
        assert (s["reasoning"], s["content"], s["rt"], s["calls"]) == (
            v["reasoning"], v["stream_content"], v["rt"], v["stream_calls"]), k
        if str(k) in v["pieces"]:
            assert s["pieces"] == v["pieces"][str(k)], k


# -- where the output starts ------------------------------------------------------------------------------

USER = [{"role": "user", "content": "hi"}]
AFTER_TOOL = [{"role": "user", "content": "weather?"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                  "name": "get_w", "arguments": '{"c": "X"}'}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
TOOLS = [{"type": "function", "function": {"name": "get_w", "parameters": {"type": "object", "properties": {
    "c": {"type": "string"}}}}}]


@pytest.mark.parametrize("fam,msgs,kw,case", [
    (GEMMA4, USER, {}, "a"), (GEMMA4, USER, {"enable_thinking": False}, "a"), (GEMMA4, USER, {"enable_thinking": True},
                                                                                "b"),
    (GEMMA4, AFTER_TOOL, {"enable_thinking": True}, "c"), (GEMMA4, AFTER_TOOL, {}, "d"),
    (QWEN3, USER, {}, "c"), (QWEN3, USER, {"enable_thinking": True}, "c"), (QWEN3, USER, {"enable_thinking": False},
                                                                              "a"),
    (QWEN3, AFTER_TOOL, {}, "c"), (QWEN3, AFTER_TOOL, {"enable_thinking": False}, "a")])
def test_where_the_output_starts(fam, msgs, kw, case) -> None:
    """The prompt's end on the real templates (docs/architecture.md, "Reasoning"): the reference vLLM's
    `is_reasoning_end` and Gemma's `adjust_initial_state_from_prompt`."""
    r = Renderer(model_dir(MODELS[fam]))
    families(fam)
    conv = to_template_messages(msgs, r.content_format, r.supports_developer_role)
    tools = TOOLS if msgs is AFTER_TOOL else None
    ids = r.encode(r.render(conv, tools=tools, **kw)).tolist()
    assert families(fam)[1].case(ids, thinking_flag(fam, kw)) == case


def test_the_flag_and_the_prompt_disagree() -> None:
    """The prompt's end decides the case; the flag only where the prompt says nothing (Qwen: its opener opens nothing
    with thinking off)."""
    _, g, _ = families(GEMMA4)
    _, q, _ = families(QWEN3)
    assert q.case([248045, 77091, 198, q.start, 198], True) == "c"
    assert q.case([248045, 77091, 198, q.start, 198], False) == "a"  # a custom template's open thought, flag off
    assert q.case([248045, 77091, 198, q.start, 271, q.end, 271], True) == "a"  # closed, flag on
    assert q.case([248045, 77091, 198], True) == "c" and q.case([248045, 77091, 198], False) == "a"
    assert g.case([105, 4368, 107, g.start, 45518, 107, g.end], True) == "a"  # closed, flag on
    assert g.case([105, 4368, 107, g.start, 45518, 107], False) == "c"  # opened, flag off
    assert g.case([105, 4368, 107, g.tool_start], True) == "a"  # a call's opener ends it


def test_the_families_markers_and_templates() -> None:
    _, g, _ = families(GEMMA4)
    _, q, _ = families(QWEN3)
    assert (g.start, g.end, g.tool_start, g.quote, g.grammar_token_ids) == (100, 101, 48, 52, [100, 101])
    assert (q.start, q.end, q.tool_start, q.quote, q.grammar_token_ids) == (248068, 248069, 248058, -1,
                                                                            [248068, 248069])
    assert 98 in g.dropped and 52 not in g.dropped  # <|think|> dropped, <|"|> kept
    assert missing_tokens(GEMMA4, {"<|channel>": 1}) == ["<channel|>"] and missing_tokens(QWEN3, {}) == ["<think>",
                                                                                                     "</think>"]
    for fam in (GEMMA4, QWEN3):
        assert template_format((model_dir(MODELS[fam]) / "chat_template.jinja").read_text()) == fam
    assert template_format("{{ messages }}") is None


# -- reasoning_effort ----------------------------------------------------------------------------------

def test_the_effort_maps_and_kwargs() -> None:
    q = EFFORT_MAPS[QWEN3]
    assert {e: q.get(e, e) for e in EFFORTS} == {"none": "none", "minimal": "low", "low": "low", "medium": "medium",
                                                  "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
    assert EFFORT_MAPS[GEMMA4] == {}
    assert effort_kwargs(None, q, None) == {}
    assert effort_kwargs("high", q, None) == {"reasoning_effort": "xhigh", "enable_thinking": True}
    assert effort_kwargs("none", q, {}) == {"reasoning_effort": "none", "enable_thinking": False}
    assert effort_kwargs("low", q, {"enable_thinking": False}) == {"reasoning_effort": "low"}  # the request's wins
    assert effort_kwargs("max", {"max": "auto"}, None) == {"enable_thinking": True}  # "auto" sets nothing
    assert effort_kwargs("minimal", {}, None) == {"reasoning_effort": "minimal", "enable_thinking": True}
    assert thinking_flag(GEMMA4, {}) is False and thinking_flag(QWEN3, {}) is True
    assert thinking_flag(QWEN3, {"enable_thinking": 0}) is False  # read for its truth, as vLLM's parsers read it


@pytest.mark.parametrize("value,error", [
    (None, None), ({}, None), ({"high": "xhigh"}, None),
    ([1], "Error: reasoning.effort_map must be a JSON object mapping reasoning efforts to strings."),
    ({"high": 3}, "Error: reasoning.effort_map must be a JSON object mapping reasoning efforts to strings."),
    ({"ultra": "max", "high": "x"}, "Error: reasoning.effort_map has unknown efforts ['ultra']; keys must be one of "
                                    "['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']."),
])
def test_the_effort_map_is_checked_with_v033s_messages(value, error) -> None:
    if error is None:
        check_effort_map(value)
    else:
        with pytest.raises(ValueError) as e:
            check_effort_map(value)
        assert str(e.value) == error


# -- the count under a stop string ------------------------------------------------------------------

@pytest.mark.parametrize("stop,reasoning,count", [("think", "Let me", 5), ("<channel|>", "Let me think. Ok", 7),
                                                  ("Hello", "Let me think. Ok", 7)])
def test_a_stop_string_cuts_the_thought_and_counts_what_the_engine_accepted(stop, reasoning, count) -> None:
    """Gemma, thinking on: a stop string inside the thought ends the reply there (the reasoning before the cut,
    content null), one equal to the closer's text too, one after it cuts the content; the count is the thought's
    tokens the engine accepted up to the one that completed the stop string (` think` counts), not the cut text's;
    Gemma's trailing whitespace is dropped where the thought ends."""
    tok, rf, _ = families(GEMMA4)
    ids = tok.encode("<|channel>thought\nLet me think. Ok<channel|>Hello there", add_special_tokens=False).ids
    prompt = [105, 4368, 107]
    p = ReasoningParser(rf, "b", None, True)
    feed = ToolFeed(p)
    text = OutputText(StopStrings([stop]))
    for t, piece in zip(ids, pieces_of(GEMMA4, prompt, ids), strict=True):
        feed.push(t, piece, len(text.text) + len(piece))
        if piece and text.append(piece) is not None:
            feed.cut(len(text.text))
            break
    else:
        pytest.fail("the stop string never came")
    p.finish()
    assert (p.reasoning(), p.content(), p.tokens) == (reasoning, None, count)


# A call in each family's syntax (vectors.json's "weather" tool)
DELIM = '<|"|>'
GEMMA_CALL = f"<|tool_call>call:get_weather{{city:{DELIM}Paris{DELIM}}}<tool_call|>"
QWEN_CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
GEMMA_THOUGHT = "<|channel>thought\nLet me think<channel|>Answer"
# (family, case, the output, a stop string, with tools): a stop string across each marker
SPANS = [
    (QWEN3, "c", "Let me think\n</think>\n\nAnswer", "\n</think>\n\nAn", False),  # the closer
    (QWEN3, "c", f"Let me think\n{QWEN_CALL}", "think\n<tool_call>\n<function", True),  # a call's opener
    (QWEN3, "a", "Hmm</think>\n\nOk", "m</think>\n", False),  # thinking off: content
    (GEMMA4, "b", GEMMA_THOUGHT, "<|channel>thought\nLet", False),  # the opener
    (GEMMA4, "b", GEMMA_THOUGHT, "ought\nLet", False),  # the label, which counts
    (GEMMA4, "b", GEMMA_THOUGHT, "think<channel|>Ans", False),  # the closer
    (GEMMA4, "b", f"<|channel>thought\nLet me think{GEMMA_CALL}", "think<|tool_call>call", True),  # a call's opener
    (GEMMA4, "b", "thought\nLet me think<channel|>Answer", "thought", False),  # a first token `thought`
    (GEMMA4, "a", "Before <|channel>thought\nHm, ok<channel|>After", "<|channel>thought\nHm", False),  # after content
    (GEMMA4, "a", "thought\nHm<channel|>After", "thought", False),
]


def span_parser(fam: str, case: str, tools: bool, choice: str = "auto") -> ReasoningParser:
    _, rf, tf = families(fam)
    inner = None
    if tools:
        schemas = {t["function"]["name"]: t["function"].get("parameters") for t in DATA["tools"]["weather"]}
        inner = ToolCallParser(tf, schemas, choice, True, False)
    return ReasoningParser(rf, case, inner, False)


@pytest.mark.parametrize("fam,case,text,stop,tools", SPANS, ids=[f"{s[0]}-{s[1]}-{s[3]!r}" for s in SPANS])
def test_a_stop_string_across_a_marker_counts_the_ids_fed_whole(fam, case, text, stop, tools) -> None:
    """R is the count of the ids the engine accepted, up to the one completing the stop string, as the parse counts
    them fed whole, with the stop string kept or not, the text released a step at a time or at the end:
    the cut tokens' markers move the count's state as they would move the parse's."""
    tok = families(fam)[0]
    ids = tok.encode(text, add_special_tokens=False).ids
    pieces = pieces_of(fam, [], ids)
    for include in (False, True):
        for per_step in (None, 1, 3):
            p = span_parser(fam, case, tools)
            feed = ToolFeed(p)
            out = OutputText(StopStrings([stop], include))
            accepted = 0
            for t, piece in zip(ids, pieces, strict=True):
                accepted += 1
                feed.push(t, piece, len(out.text) + len(piece))
                if piece and out.append(piece) is not None:
                    feed.cut(len(out.text))
                    break
                if per_step and accepted % per_step == 0:  # a stream's step: what the stop strings release
                    out.take(final=False)
                    feed.release(out.sent)
            else:
                pytest.fail("the stop string never came")
            p.finish()
            whole = span_parser(fam, case, tools)
            for t, piece in zip(ids[:accepted], pieces[:accepted], strict=True):
                whole.feed(t, piece)
            whole.finish()
            assert p.tokens == whole.tokens, (include, per_step, p.tokens, whole.tokens)


@pytest.mark.parametrize("fam,case,head,closer", [
    (QWEN3, "c", "Let me think.\n", ["</", "think", ">"]),
    (GEMMA4, "b", "<|channel>thought\nLet me think.", ["<", "channel", "|>"])])
def test_a_marker_spelled_with_ordinary_tokens_is_text(fam, case, head, closer) -> None:
    """Markers count by id: a closer the model spells with ordinary tokens stays the thought's text, and so does what
    follows it (vLLM's non-streamed parse, on the text, ends the thought there; its stream does not)."""
    tok = families(fam)[0]
    ids = [t for part in (head, *closer, "\n\nAnswer") for t in tok.encode(part, add_special_tokens=False).ids]
    assert families(fam)[1].end not in ids
    for k in (None, 1, 3):
        p = span_parser(fam, case, False)
        reasoning = ""
        for i, (t, piece) in enumerate(zip(ids, pieces_of(fam, [], ids), strict=True)):
            p.feed(t, piece)
            if k and (i + 1) % k == 0:
                reasoning += p.take()[0]
        p.finish()
        reasoning += p.take()[0]
        want = head.removeprefix("<|channel>thought\n") + "".join(closer) + "\n\nAnswer"
        assert (p.reasoning(), p.content(), reasoning if k else want) == (want, None, want), k


def test_the_count_feeds_a_copy_that_sends_nothing(monkeypatch) -> None:
    """`count_whole` feeds a copy of the parser, once. The copy shares the families and the tools' schemas. It holds
    no call's pieces (an earlier kept call's `ArgStream` is not copied), and builds none for a call it is fed. The
    parser itself is unchanged."""
    tok = families(GEMMA4)[0]
    p = span_parser(GEMMA4, "b", True)
    head = tok.encode(f"<|channel>thought\nLet me think<channel|>{GEMMA_CALL}", add_special_tokens=False).ids
    for t, piece in zip(head, pieces_of(GEMMA4, [], head), strict=True):
        p.feed(t, piece)
    assert p.inner.calls_[0].stream is not None
    before = (p.tokens, len(p.inner.calls_), p.take())
    dups: list = []

    def deepcopy(x, memo=None):
        dups.append(copy.deepcopy(x, memo))
        return dups[-1]
    monkeypatch.setattr(reasoning_module, "copy", SimpleNamespace(deepcopy=deepcopy))
    built = [0]
    init = tool_args.ArgStream.__init__

    def counted(self, *a, **kw):
        built[0] += 1
        init(self, *a, **kw)
    monkeypatch.setattr(tool_args.ArgStream, "__init__", counted)
    rest = tok.encode(f"Bye{GEMMA_CALL}", add_special_tokens=False).ids
    p.count_whole(list(zip(rest, pieces_of(GEMMA4, head, rest), strict=True)))
    assert len(dups) == 1
    dup = dups[0]
    assert built[0] == 0 and len(dup.inner.calls_) == 2  # the copy saw the second call, and built it no pieces
    assert dup.inner.calls_[0].stream is None
    assert dup.fam is p.fam and dup.inner.fam is p.inner.fam
    assert dup.inner.schemas is p.inner.schemas
    assert (p.tokens, len(p.inner.calls_), p.take()) == (before[0], before[1], ("", "", []))


# -- whitespace around a call: vLLM's stream and non-stream, by the reference -------------------------

# Content, a call, whitespace, content: each output, then vLLM's (non-streamed content, streamed content) by case.
# The last two of each family hold a call region with no name. No call is found, so the message's content is not
# stripped; whitespace before the region is dropped, as the stream drops it
NAMELESS = {GEMMA4: [" <|tool_call>call:(x)<tool_call|>Bye", "\n<|tool_call>call:(x)<tool_call|>\nBye"],
            QWEN3: ["\n<tool_call>\n</tool_call>Bye", " <tool_call>\n</tool_call>\n Bye"]}
GEMMA_WS = [f"Hi{GEMMA_CALL}\nBye", f"Hi {GEMMA_CALL} \n Bye", f"{GEMMA_CALL}\nBye", f"Hi{GEMMA_CALL}\n",
            f"<|channel>thought\nhmm<channel|>Hi{GEMMA_CALL}\nBye",
            f"<|channel>thought\nhmm<channel|>{GEMMA_CALL}\n\nBye", *NAMELESS[GEMMA4]]
QWEN_WS = [f"Hi{QWEN_CALL}\nBye", f"hmm</think>\n\nHi{QWEN_CALL}\nBye", f"hmm</think>\n\n{QWEN_CALL}\nBye",
           f"hmm</think>Hi{QWEN_CALL}  Bye", *NAMELESS[QWEN3]]
WS_WANT = {
    (GEMMA4, "b"): [("HiBye", "HiBye"), ("Hi  Bye", "Hi  Bye"), ("Bye", "Bye"), ("Hi", "Hi"), ("Hi\nBye", "Hi\nBye"),
                    ("Bye", "Bye"), ("Bye", "Bye"), ("Bye", "Bye")],
    (GEMMA4, "c"): [("Bye", "Bye"), ("Bye", " Bye"), ("Bye", "Bye"), (None, None), ("Hi\nBye", "Hi\nBye"),
                    ("Bye", "Bye"), ("Bye", "Bye"), ("Bye", "Bye")],
    (GEMMA4, "a"): [("Hi\nBye", "Hi\nBye"), ("Hi  \n Bye", "Hi  \n Bye"), ("Bye", "Bye"), ("Hi", "Hi\n"),
                    ("Hi\nBye", "Hi\nBye"), ("Bye", "Bye"), ("Bye", "Bye"), ("Bye", "Bye")],
    (GEMMA4, "d"): [("Hi\nBye", "Hi\nBye"), ("Hi  \n Bye", "Hi  \n Bye"), ("Bye", "Bye"), ("Hi", "Hi\n"),
                    ("Hi\nBye", "Hi\nBye"), ("Bye", "Bye"), ("Bye", "Bye"), ("Bye", "Bye")],
    (QWEN3, "c"): [("Bye", "Bye"), ("Hi\nBye", "\n\nHi\nBye"), ("Bye", "Bye"), ("Hi  Bye", "Hi  Bye"), ("Bye", "Bye"),
                   (" Bye", " Bye")],
    (QWEN3, "a"): [("Hi\nBye", "Hi\nBye"), ("hmm\n\nHi\nBye", "hmm\n\nHi\nBye"),
                   ("hmm\n\n\nBye", "hmm\n\n\nBye"), ("hmmHi  Bye", "hmmHi  Bye"), ("Bye", "Bye"), (" Bye", " Bye")],
}
# The nameless regions under `required` (as under `auto`) and `none`. Under `none` no region is a call: whitespace
# before it stays, but in case c, where it is the thought's
NAMELESS_NONE = {
    (GEMMA4, "b"): [(" Bye", " Bye"), ("\n\nBye", "\n\nBye")], (GEMMA4, "c"): [("Bye", "Bye"), ("\nBye", "\nBye")],
    (GEMMA4, "a"): [(" Bye", " Bye"), ("\n\nBye", "\n\nBye")],
    (GEMMA4, "d"): [(" Bye", " Bye"), ("\n\nBye", "\n\nBye")],
    (QWEN3, "c"): [("Bye", "Bye"), ("\n Bye", "\n Bye")], (QWEN3, "a"): [("\nBye", "\nBye"), (" \n Bye", " \n Bye")],
}
CALL_WS = [(fam, case, "auto", text, *want) for (fam, case), wants in WS_WANT.items()
           for text, want in zip(GEMMA_WS if fam == GEMMA4 else QWEN_WS, wants, strict=True)]
CALL_WS += [(fam, case, "required", text, *want) for (fam, case), wants in WS_WANT.items()
            for text, want in zip(NAMELESS[fam], wants[-2:], strict=True)]
CALL_WS += [(fam, case, "none", text, *want) for (fam, case), wants in NAMELESS_NONE.items()
            for text, want in zip(NAMELESS[fam], wants, strict=True)]


@pytest.mark.parametrize("fam,case,choice,text,content,stream_content", CALL_WS,
                         ids=[f"{c[0]}-{c[1]}-{c[2]}-{i}" for i, c in enumerate(CALL_WS)])
def test_whitespace_around_a_call_is_vllms(fam, case, choice, text, content, stream_content) -> None:
    """The non-streamed content against vLLM's, and the stream at one and three tokens a step against its stream's
    (they differ by the strip rule: the stream drops a token of whitespace after a call, the message strips its
    ends). Recorded from the reference's parse of each output."""
    tok = families(fam)[0]
    ids = tok.encode(text, add_special_tokens=False).ids
    pieces = pieces_of(fam, [], ids)
    p = span_parser(fam, case, True, choice)
    for t, piece in zip(ids, pieces, strict=True):
        p.feed(t, piece)
    p.finish()
    assert p.content() == content
    for k in (1, 3):
        p = span_parser(fam, case, True, choice)
        got = ""
        for i, (t, piece) in enumerate(zip(ids, pieces, strict=True)):
            p.feed(t, piece)
            if (i + 1) % k == 0:
                got += p.take()[1]
        p.finish()
        got += p.take()[1]
        assert (got or None) == stream_content, k


# -- the lazy path's chunks -----------------------------------------------------------------------------

def request_state(fam: str, case: str, lazy: bool, logprobs: bool, token_ids: bool,
                  stops: list[str] | None = None) -> RequestState:
    """A streamed request's state as `App._chat_request` builds it for a reply of the family (no tools): its parser
    built at once (eager), or at a marker (`lazy`: the output starts as content)."""
    tok, rf, _ = families(fam)
    client = SimpleNamespace(metrics=SimpleNamespace(itl=SimpleNamespace(observe=lambda s: None), tool_parser=None),
                             on_finished=lambda st: None, abort=lambda *a: None)
    st = RequestState(client, 1, 3, Detokenizer(tok, [], False), OutputText(StopStrings(stops) if stops else None),
                      ChunkTemplate("c", 0, "m"), True, False, False, [], 10**6, 0, 0, 0)
    st.reasoning, st.case = True, case
    if lazy:
        st.lazy = (rf, case, False)
    else:
        st.tools = ToolFeed(ReasoningParser(rf, case, None, False))
    if logprobs:
        st.lp = RequestLogprobs(TokenText(tok), 1, True)
    if token_ids:
        st.out_ids, st.prompt_ids = [], []
    return st


def chunks_of(st: RequestState, ids: list[int], stop: int) -> list[dict]:
    """The stream's choices, one token a step, then the stop token with the finish."""
    for k, t in enumerate([*ids, stop]):
        last = k == len(ids)
        st.on_output([t], int(FinishReason.STOP if last else FinishReason.NONE), 1, 1, ([-0.5], [[t]], [[-0.5]]))
    return [json.loads(c[6:])["choices"] for c in st.pending if c.startswith(b"data: {")]


@pytest.mark.parametrize("fam", [GEMMA4, QWEN3])
@pytest.mark.parametrize("text,stops,first", [("\n\nHello there", None, "\n\nHello"), ("\n\n", None, "\n\n"),
                                              (" \nHello there", ["Hello world"], " \nHello there")])
def test_the_lazy_path_holds_leading_whitespace(fam, text, stops, first) -> None:
    """Thinking off without tools (case a): the parser starts only at a marker, and until then the
    stream holds whitespace before the content's first other character, as the eager parse and vLLM hold it, also
    while a stop string holds the text after it back; a reply of whitespace alone goes out with the finish. Each step's
    logprobs and ids still go out, in an empty delta."""
    tok = families(fam)[0]
    ids = tok.encode(text, add_special_tokens=False).ids
    stop = 106 if fam == GEMMA4 else 248046
    for logprobs in (False, True):
        for token_ids in (False, True):
            got = [chunks_of(request_state(fam, "a", lazy, logprobs, token_ids, stops), ids, stop)
                   for lazy in (True, False)]
            assert got[0] == got[1], (logprobs, token_ids)
            deltas = [c[0]["delta"] for c in got[0] if c and c[0]["delta"].get("content")]
            assert deltas[0]["content"] == first, deltas


def test_the_lazy_paths_held_whitespace_is_scanned_once() -> None:
    """A reply of whitespace alone, a token a step (a model looping on newlines): the held whitespace is scanned once,
    as it comes, so holding it costs no more than the plain path, which sends it (no parser: `reasoning.parser` none),
    over 64,000 steps (the least of 5 interleaved runs; both grow the output's text alike)."""
    tok = families(QWEN3)[0]
    blank = tok.encode("\n\n", add_special_tokens=False).ids[0]

    def cost(lazy: bool):
        def run() -> None:
            st = request_state(QWEN3, "a", True, False, False)
            if not lazy:
                st.reasoning, st.lazy = False, None
            for _ in range(64_000):
                st.on_output([blank], int(FinishReason.NONE), 1, 1)
            assert (len(st.pending) == 1) == lazy  # the role chunk alone (held), or the text sent too
        return run

    held, sent = least_each(cost(True), cost(False))
    assert held < 1.5 * sent + 0.02, (held, sent)

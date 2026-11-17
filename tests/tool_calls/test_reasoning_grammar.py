"""The grammars with thinking on (`grammar.spec.thought`, `GrammarSpecs.thinking`, `grammar.tools.call_grammar`'s
`thought`), on llguidance and the real tokenizers, both families, under guidance (forcing) and auto: each form passes
the check with the markers' ids; Gemma's case b starts with its opener or the schema, never the closer or a call;
mid-thought every token passes but the markers and the call openers, and the closer ends it; Qwen's blank line after
its closer is optional; a schema's documents are the thinking-off grammar's after a thought (a differential over hand
schemas, both whitespace forms of guidance and auto); nothing is forced at the start; the cache keeps one entry per
family, case and schema. Through `App.grammar` (a stub App): cases a and d give the grammar texts recorded with
thinking off, a non-object `x-guidance` is ignored under guidance thinking on and off, and an ignored
`response_format` warns once a request."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

from llm_mini_infer.api.app import App
from llm_mini_infer.api.protocol import ApiError, decode_chat
from llm_mini_infer.api.tools import tool_request
from llm_mini_infer.grammar.engine import Grammars, LLGuidance
from llm_mini_infer.grammar.spec import GrammarSpecs, Pending, check, thought
from llm_mini_infer.grammar.threads import CompileThreads
from llm_mini_infer.grammar.tools import call_grammar
from llm_mini_infer.output.reasoning import ReasoningFamily
from llm_mini_infer.output.tool_calls import Family

from ..support.fixtures import model_dir
from ..support.paths import DATA

MODELS = {"gemma4": ("gemma-4-31b-it", "gemma4", 262144, "<eos>"),
          "qwen3": ("qwen3.8-27b", "qwen3_coder", 248320, "<|im_end|>")}
CASES = {"gemma4": ("b", "c"), "qwen3": ("c",)}
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}, "required": ["a", "n"],
          "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_w", "parameters": {
    "type": "object", "properties": {"c": {"type": "string"}}, "required": ["c"]}}}]
_cache: dict = {}


def setup(fam: str, backend: str):
    key = (fam, backend)
    if key not in _cache:
        d, tname, vocab, eos = MODELS[fam]
        path = model_dir(d) / "tokenizer.json"
        if not path.exists():
            pytest.skip(f"{path} is absent: uv run tests/scripts/fetch_fixtures.py")
        tok = Tokenizer.from_file(str(path))
        _cache[key] = (tok, ReasoningFamily(fam, tok), Family(tname, tok),
                       GrammarSpecs(backend=backend, disable_any_whitespace=backend == "guidance"),
                       Grammars(LLGuidance(str(path), vocab, eos)))
    return _cache[key]


def ids(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("backend", ["guidance", "auto"])
@pytest.mark.parametrize("fam", ["gemma4", "qwen3"])
def test_the_schema_after_a_thought(fam, backend) -> None:
    tok, rf, _, specs, g = setup(fam, backend)
    doc = '{"a":"x","n":3}' if backend == "guidance" else '{"a": "x", "n": 3}'
    for case in CASES[fam]:
        prefix = thought(fam, case, rf.start, rf.end)
        text = specs.thinking(json.dumps(SCHEMA).encode(), prefix, f"{fam}:{case}".encode(), rf.grammar_token_ids)
        assert not check(text, rf.grammar_token_ids)
        assert text.startswith(specs.header()) and f"start: {prefix} %json" in text
        m = g.matcher(g.compile(text))
        assert list(g.forced_prefix(text)) == []  # the thought may start at once: nothing forced
        if case == "b":  # Gemma's optional thought: its opener or the schema, never its closer or a call
            assert m.validate_tokens([rf.start]) == 1 and m.validate_tokens(ids(tok, "{")) == 1
            assert m.validate_tokens([rf.end]) == 0 and m.validate_tokens([rf.tool_start]) == 0
            assert m.consume_tokens([rf.start, *ids(tok, "thought\n")])
        assert m.consume_tokens(ids(tok, "Let me think: {not json} <b>"))
        for t in (rf.tool_start, rf.start, *sorted(rf.dropped)[:4]):  # mid-thought: no marker, no call opener
            assert m.validate_tokens([t]) == 0, t
        assert m.validate_tokens(ids(tok, "{")) == 1  # a brace is thought text, not the schema yet
        assert m.consume_tokens([rf.end])
        assert m.validate_tokens([rf.end]) == 0
        if fam == "qwen3":
            blank = ids(tok, "\n\n")
            assert m.validate_tokens(blank) == len(blank) and m.validate_tokens(ids(tok, doc)) == len(ids(tok, doc))
            assert m.consume_tokens(blank)
        body = ids(tok, doc)
        assert m.validate_tokens(body) == len(body)
        assert m.validate_tokens(ids(tok, "hello")) == 0  # after the closer, the schema's JSON only
        assert m.consume_tokens(body) and (m.is_stopped() or m.is_accepting())


@pytest.mark.parametrize("backend", ["guidance", "auto"])
@pytest.mark.parametrize("fam", ["gemma4", "qwen3"])
def test_calls_after_a_thought(fam, backend) -> None:
    tok, rf, tf, _, g = setup(fam, backend)
    for case in CASES[fam]:
        for choice in ("required", "named"):
            text, _ = call_grammar(tf.name, TOOLS, choice, name="get_w" if choice == "named" else None,
                                   forcing=backend == "guidance", ids=tf.grammar_ids,
                                   thought=thought(fam, case, rf.start, rf.end))
            assert not check(text, tf.grammar_token_ids)
            m = g.matcher(g.compile(text))
            if case == "b":
                assert m.validate_tokens([rf.start]) == 1 and m.validate_tokens([rf.end]) == 0
                assert m.consume_tokens([rf.start, *ids(tok, "thought\n")])
            assert m.consume_tokens(ids(tok, "I will call get_w."))
            assert m.validate_tokens([rf.tool_start]) == 0  # no call inside the thought
            assert m.consume_tokens([rf.end])
            if fam == "qwen3":
                assert m.consume_tokens(ids(tok, "\n\n"))
            assert m.validate_tokens([rf.tool_start]) == 1  # then the call


# -- the server's path (`App.grammar`, on a stub App) ----------------------------------------------------------------

OFF = json.loads((DATA / "reasoning" / "thinking_off_grammars.json").read_text())
STUB_MODELS = {"gemma4": ("gemma-4-31b-it", "gemma4"), "qwen3": ("qwen3.8-27b", "qwen3_coder")}
_apps: dict = {}


def stub_app(fam: str, backend: str, compact: bool, threads: CompileThreads | None = None) -> App:
    """An App with what `App.grammar` reads: the grammars (fresh with `threads`), both families and the size cap."""
    key = (fam, backend, compact)
    if threads is None and key in _apps:
        return _apps[key]
    tok = setup(fam, "auto")[0]
    app = object.__new__(App)
    app.cfg = SimpleNamespace(structured_output=SimpleNamespace(max_schema_bytes=1 << 20))
    app.grammars = GrammarSpecs(backend=backend, disable_any_whitespace=compact, threads=threads)
    app.tool_family = Family(STUB_MODELS[fam][1], tok)
    app.reasoning_family = ReasoningFamily(fam, tok)
    if threads is None:
        _apps[key] = app
    return app


def request(body: dict) -> tuple:
    """A body decoded and its tools checked, as `App.chat` does."""
    req = decode_chat(json.dumps({"messages": [{"role": "user", "content": "hi"}], **body}).encode())
    return req, tool_request(req.tools, req.tool_choice, req.parallel_tool_calls)


def off_body(row: list) -> dict:
    _, _, _, rf, choice, parallel, _, _ = row
    body = {}
    if OFF["response_formats"][rf] is not None:
        body["response_format"] = OFF["response_formats"][rf]
    if OFF["tool_choices"][choice] is not None:
        body.update(tools=OFF["tools"], tool_choice=OFF["tool_choices"][choice], parallel_tool_calls=parallel)
    return body


@pytest.mark.parametrize("fam", ["gemma4", "qwen3"])
def test_the_thinking_off_grammars_are_pinned(fam) -> None:
    """Cases a and d (the thought closed in the prompt, or thinking off) take the grammar text recorded with thinking
    off (before any thought), through the server's `App.grammar`: `json_object`, hand schemas, every `tool_choice`
    with and without a `response_format`, `parallel_tool_calls` either way, under guidance (compact and flexible) and
    auto. A thought in front of one fails."""
    rows = [r for r in OFF["rows"] if r[0] == fam]
    assert len(rows) == 162
    for row in rows:
        _, backend, compact, _, _, _, digest, length = row
        app = stub_app(fam, backend, compact)
        req, tools = request(off_body(row))
        for case in ("a", "d") if fam == "gemma4" else ("a",):
            text = app.grammar(req, tools, case)
            assert (hashlib.sha256(text.encode()).hexdigest()[:16], len(text)) == (digest, length), (row, case)


XG = [[], "x", 5, True, 0, False, "", 1.5, None]  # every JSON kind of x-guidance but an object


@pytest.mark.parametrize("fam", ["gemma4", "qwen3"])
def test_a_non_object_x_guidance(fam) -> None:
    """Under guidance, an `x-guidance` that is not an object is ignored, as llguidance's JSON form ignores it. The
    grammar is the one without it, thinking on and off, for a `response_format` and for `tool_choice: auto` with one.
    Under auto it is a 400 either way (null is absent under both)."""
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    tools = {"tools": OFF["tools"], "tool_choice": "auto"}
    for backend, compact in (("guidance", True), ("guidance", False), ("auto", False)):
        app = stub_app(fam, backend, compact)
        for value in XG:
            for extra in ({}, tools):
                plain = request({"response_format": {"type": "json_schema", "json_schema": {
                    "name": "s", "schema": schema}}, **extra})
                body = {"response_format": {"type": "json_schema", "json_schema": {
                    "name": "s", "schema": {**schema, "x-guidance": value}}}, **extra}
                req, tr = request(body)
                for case in ("a", *CASES[fam]):
                    try:
                        got = app.grammar(req, tr, case)
                    except ApiError as exc:
                        got = exc.status, exc.message
                    if backend == "guidance" or value is None:
                        assert got == app.grammar(*plain, case), (backend, value, extra, case)
                    else:
                        assert got[0] == 400 and "'x-guidance' must be an object" in got[1], (value, extra, case)


@pytest.mark.parametrize("choice", ["required", "named"])
def test_an_ignored_response_format_warns_once(choice, caplog) -> None:
    """`required` or a named tool with a `response_format`: one warning a request, as `App.chat` builds its grammars
    on the compile threads (a miss raises `Pending`, and the lookup runs again once built), thinking off and on (the
    thinking-off grammar first, then the one with the thought)."""
    tool_choice = {"type": "function", "function": {"name": "get_w"}} if choice == "named" else "required"
    req, tools = request({"response_format": {"type": "json_object"}, "tools": OFF["tools"],
                          "tool_choice": tool_choice})

    async def chat(app: App, case: str) -> None:
        try:
            app.grammar(req, tools, warn=True)
        except Pending as p:
            await app.grammar_built(p, req, tools)
        if case in ("b", "c"):
            try:
                app.grammar(req, tools, case)
            except Pending as p:
                await app.grammar_built(p, req, tools, case)

    for case in ("a", "b"):
        threads = CompileThreads(1, "test")
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="llm_mini_infer.api.app"):
                asyncio.run(chat(stub_app("gemma4", "guidance", True, threads), case))
        finally:
            threads.close()
        warned = [r for r in caplog.records if "ignores its response_format" in r.getMessage()]
        assert len(warned) == 1, (case, [r.getMessage() for r in caplog.records])


def test_the_forms_and_their_cache() -> None:
    _, g, _, specs, _ = setup("gemma4", "guidance")
    _, q, _, _, _ = setup("qwen3", "guidance")
    assert thought("gemma4", "b", g.start, g.end) == '(<[100]> "thought\\n" /(.|\\n)*/ <[101]>)?'
    assert thought("gemma4", "c", g.start, g.end) == "/(.|\\n)*/ <[101]>"
    assert thought("qwen3", "c", q.start, q.end) == '/(.|\\n)*/ <[248069]> "\\n\\n"?'
    schema = json.dumps(SCHEMA).encode()
    one = specs.thinking(schema, thought("gemma4", "b", g.start, g.end), b"gemma4:b", g.grammar_token_ids)
    assert specs.thinking(schema, "unused: the cached entry", b"gemma4:b", g.grammar_token_ids) is one
    other = specs.thinking(schema, thought("gemma4", "c", g.start, g.end), b"gemma4:c", g.grammar_token_ids)
    assert other != one and specs.thinking(None, thought("gemma4", "c", g.start, g.end), b"gemma4:c",
                                           g.grammar_token_ids) != other


# A schema's own `x-guidance` options, which win over the server's whitespace: separators, a whitespace pattern,
# whitespace flexible or not. The documents are written with each, and one breaks the schema
XG_BASE = {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}, "required": ["a"],
           "additionalProperties": False}
XG_OWN = [{"key_separator": " : "}, {"item_separator": ", "}, {"whitespace_pattern": "[ ]{0,2}"},
          {"whitespace_flexible": True}, {"whitespace_flexible": False}]
XG_DOCS = ['{"a":[1,2]}', '{"a": [1, 2]}', '{"a" : [1,2]}', '{"a":[1, 2]}', '{"a" : [1, 2]}', '{ "a" : [ 1 , 2 ] }',
           '{"a":[1,\n2]}', '{"a":   [1,2]}', '{"b":[1]}']
# The hand-schema differential (widened): key order, whitespace, number forms, bounds, patterns,
# enums, alternatives, `json_object` (None), top-level scalars and arrays, a recursive `$ref`, `prefixItems`, `const`
# and a `format`; each document accepted after a thought exactly when the thinking-off grammar accepts it
DIFF_SCHEMAS = [
    ({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "required": ["a", "b"],
      "additionalProperties": False},
     ['{"a":1,"b":"x"}', '{"b":"x","a":1}', '{ "a": 1, "b": "x" }', '{"a":1}', '{"a":1,"b":"x","c":2}',
      '{"a":1.0,"b":"x"}', '{"a":-0,"b":"x"}', '{"a":01,"b":"x"}', '{"a":1,"b":"\\u00e9\\n"}', '{"a":1,\n"b":"x"}']),
    ({"type": "object", "properties": {"n": {"type": "number", "minimum": 0, "maximum": 10}}, "required": ["n"],
      "additionalProperties": False},
     ['{"n":0}', '{"n":10}', '{"n":10.5}', '{"n":-1}', '{"n":1e1}', '{"n":2.50}', '{"n":1E-2}', '{"n":.5}']),
    ({"type": "object", "properties": {"s": {"type": "string", "pattern": "^[a-z]{2,4}$"},
                                       "e": {"enum": ["red", "green", 3]}}, "required": ["s", "e"],
      "additionalProperties": False},
     ['{"s":"ab","e":"red"}', '{"s":"abcde","e":"red"}', '{"s":"Ab","e":"green"}', '{"s":"abc","e":3}',
      '{"s":"abc","e":"blue"}', '{"s":"abc","e":"3"}']),
    ({"type": "object", "properties": {"v": {"anyOf": [{"type": "array", "items": {"type": "boolean"}, "maxItems": 2},
                                                       {"type": "null"}]}}, "required": ["v"],
      "additionalProperties": False},
     ['{"v":[true,false]}', '{"v":[true,false,true]}', '{"v":null}', '{"v":[]}', '{"v":[1]}', '{"v":"x"}']),
    (None, ['{}', '{"a":1}', '{ "a" : [1, {"b": null}] }', '[1]', '"x"', '{"a":}']),
    ({"type": "string", "maxLength": 3}, ['"ab"', '"abcd"', '"a\\"b"', '1', '"\u00e9"', '" "']),
    ({"type": "number"}, ['1', '-1.5e3', '0.0', '01', '1.', '"1"', '-0']),
    ({"type": "integer", "minimum": -2, "maximum": 5}, ['5', '-2', '6', '1.0', '-0', '3', '-3']),
    ({"type": "boolean"}, ['true', 'false', 'True', '1']),
    ({"type": "null"}, ['null', 'nul', '0', '"null"']),
    ({"enum": ["x", 1, None, [1]]}, ['"x"', '1', 'null', '[1]', '[ 1 ]', '"y"', '2']),
    ({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
     ['[1]', '[1,2,3]', '[1, 2]', '[]', '[1,2,3,4]', '["a"]']),
    ({"$defs": {"n": {"type": "object", "properties": {"v": {"type": "integer"}, "next": {
        "anyOf": [{"$ref": "#/$defs/n"}, {"type": "null"}]}}, "required": ["v"], "additionalProperties": False}},
      "$ref": "#/$defs/n"},
     ['{"v":1}', '{"v":1,"next":{"v":2,"next":null}}', '{"next":null}', '{"v":1,"next":{"v":"x"}}',
      '{"v":1,"next":1}']),
    ({"anyOf": [{"type": "string", "minLength": 2}, {"type": "integer"}]}, ['"ab"', '"a"', '3', '3.5']),
    ({"oneOf": [{"type": "integer", "minimum": 5}, {"type": "string", "maxLength": 2}]}, ['7', '"ab"', '4', '"abc"']),
    ({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "boolean"}], "items": False},
     ['[1,true]', '[1]', '[true,1]', '[1,true,2]']),
    ({"const": {"k": [1, 2]}}, ['{"k":[1,2]}', '{"k":[2,1]}', '{"k":[1,2],"j":0}']),
    ({"type": "string", "format": "date"}, ['"1996-02-29"', '"1999-13-01"', '"x"']),
    *[({**XG_BASE, "x-guidance": own}, XG_DOCS) for own in XG_OWN],
]
WHITESPACE = [("guidance", True), ("guidance", False), ("auto", False)]  # (backend, disable_any_whitespace)


def accepted(g, compiled, prefix: list[int], doc: list[int]) -> bool:
    m = g.matcher(compiled)
    if prefix and not m.consume_tokens(prefix):
        return False
    return m.validate_tokens(doc) == len(doc) and m.consume_tokens(doc) and (m.is_accepting() or m.is_stopped())


@pytest.mark.parametrize("backend,compact", WHITESPACE, ids=["guidance-compact", "guidance-flexible", "auto"])
@pytest.mark.parametrize("fam", ["gemma4", "qwen3"])
def test_the_language_after_the_thought_is_the_thinking_off_grammars(fam, backend, compact) -> None:
    tok, rf, _, _, g = setup(fam, backend)
    specs = GrammarSpecs(backend=backend, disable_any_whitespace=compact)
    thought_ids = ids(tok, "Plan: {\"a\": 1} first.")
    for schema, docs in DIFF_SCHEMAS:
        raw = None if schema is None else json.dumps(schema).encode()
        off = g.compile(specs.json_object() if schema is None else specs.json_schema(raw))
        for case in CASES[fam]:
            on = g.compile(specs.thinking(raw, thought(fam, case, rf.start, rf.end), f"{fam}:{case}".encode(),
                                          rf.grammar_token_ids))
            prefixes = [[rf.start, *ids(tok, "thought\n"), *thought_ids, rf.end]] if case == "b" else [
                [*thought_ids, rf.end]]
            if case == "b":
                prefixes.append([])  # Gemma's thought is optional
            if fam == "qwen3":
                prefixes.append([*thought_ids, rf.end, *ids(tok, "\n\n")])
            wants = set()
            for doc in docs:
                d = ids(tok, doc)
                want = accepted(g, off, [], d)
                wants.add(want)
                for p in prefixes:
                    assert accepted(g, on, p, d) == want, (schema, doc, case, len(p))
            assert wants == {True, False}, (schema, wants)  # each schema's documents are taken and refused

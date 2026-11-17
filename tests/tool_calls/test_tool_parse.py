"""The tool-call parsers (`output.tool_calls`) on vectors, at every chunk size: the reference vLLM's parser tests ported
as data (tests/data/tool_calls/parser_vectors.json: each test's model outputs, its chunk lists joined, with the tools
its parser types by, and its argument strings, run through its own engine on the real tokenizers, edited where a
decision deviates) and the models' recorded outputs (tests/data/tool_calls/recorded/). Each vector's ids go to the
parser one token at a time, its deltas taken every 1, 2, 3, 5, 7, 11, 23 or 1,000 tokens: the non-streamed message
equals the expected parse, and the streamed deltas, accumulated, give the same calls whatever the step
(`test_upstream_vectors`'s invariant). Then a call's arguments in pieces (`tool_args.ArgStream`): each a prefix of the
final arguments, in time linear in the call's text (the parse's work counted, the rest timed), and the final
arguments' top-level rules that keep it so (a key written twice, a cut Qwen parameter, a value nested deeper than
`MAX_NESTING`)."""

from __future__ import annotations

import json
import math

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from llm_mini_infer.output import tool_args
from llm_mini_infer.output.stop import OutputText, StopStrings
from llm_mini_infer.output.tool_calls import GEMMA4, QWEN3, Family, ToolCallParser, ToolFeed, api_finish_reason

from ..support.fixtures import model_dir
from ..support.paths import DATA as TESTS_DATA
from ..support.timing import least_each

DATA = TESTS_DATA / "tool_calls"
VECTORS = json.loads((DATA / "parser_vectors.json").read_text())
MODELS = {GEMMA4: "gemma-4-31b-it", QWEN3: "qwen3.8-27b"}
STOPS = {GEMMA4: {1, 106, 50}, QWEN3: {248046, 248044}}
CHUNKS = (1, 2, 3, 5, 7, 11, 23, 1000)
_tok: dict[str, tuple[Tokenizer, Family]] = {}


def tokenizer(fam: str) -> tuple[Tokenizer, Family]:
    if fam not in _tok:
        path = model_dir(MODELS[fam]) / "tokenizer.json"
        if not path.exists():
            pytest.skip(f"{path} is absent: uv run tests/scripts/fetch_fixtures.py")
        tok = Tokenizer.from_file(str(path))
        _tok[fam] = (tok, Family(fam, tok))
    return _tok[fam]


def schemas(tools: list[dict]) -> dict:
    out: dict = {}
    for t in tools:
        out.setdefault(t["function"]["name"], t["function"].get("parameters"))
    return out


def parse(fam: str, ids: list[int], tools: list[dict], choice, per_step: int, stop: str | None = None,
          parallel: bool = True) -> tuple:
    """As RequestState feeds it: the detokenizer's pieces (special tokens kept), the stop strings' release, the stop
    token last with no text. Returns (content, calls, streamed content, streamed {index: [name, arguments]})."""
    tok, family = tokenizer(fam)
    p = ToolCallParser(family, schemas(tools), choice, parallel)
    feed = ToolFeed(p)
    text = OutputText(StopStrings([stop]) if stop else None)
    ds = DecodeStream(skip_special_tokens=False)
    end = ids[-1] if ids and ids[-1] in STOPS[fam] else None
    streamed, calls = "", {}

    def take() -> None:
        nonlocal streamed
        c, t = p.take()
        streamed += c
        for e in t:
            got = calls.setdefault(e["index"], ["", ""])
            got[0] += e["function"].get("name", "")
            got[1] += e["function"].get("arguments", "")

    hit = False
    for k, t in enumerate(ids):
        if end is not None and k == len(ids) - 1:
            feed.push(t, "", len(text.text))
            break
        piece = ds.step(tok, t) or ""
        feed.push(t, piece, len(text.text) + len(piece))
        if piece and text.append(piece) is not None:
            feed.cut(len(text.text))
            hit = True
            break
        if (k + 1) % per_step == 0:
            text.take(final=False)
            feed.release(text.sent)
            take()
    if not hit:
        feed.release(len(text.text))
    p.finish()
    take()
    kept = [[c.name.strip(), c.arguments] for c in p.kept_calls()]
    return p.content(), kept, streamed, [calls[i] for i in sorted(calls)], p


@pytest.mark.parametrize("vec", VECTORS["vectors"], ids=[f"{i}-{v['family']}-{v['choice']}"
                                                         for i, v in enumerate(VECTORS["vectors"])])
def test_upstream_vectors(vec) -> None:
    fam, tools = vec["family"], VECTORS["tools"][vec["tools"]]  # the tools its source test's parser types by
    choice = vec["choice"]
    ref = None
    for k in CHUNKS:
        content, calls, streamed, scalls, _ = parse(fam, vec["ids"], tools, choice, k, vec.get("stop"))
        assert (content, calls) == (vec["content"], vec["calls"]), (k, vec.get("edit"))
        if ref is None:
            ref = (streamed, scalls)
            assert scalls == calls  # the deltas give the non-streamed calls
        assert (streamed, scalls) == ref, k


@pytest.mark.parametrize("vec", VECTORS["args"], ids=[str(i) for i in range(len(VECTORS["args"]))])
def test_gemma_argument_vectors(vec) -> None:
    fn = tool_args._parse_gemma4_args if vec["function"] == "_parse_gemma4_args" else tool_args._parse_gemma4_array
    assert fn(vec["input"], partial=vec["partial"]) == vec["output"]


INF, NAN = float("inf"), float("nan")
D = {"P": {"type": "integer"}, "Obj": {"type": "object", "properties": {"count": {"type": "integer"}}},
     "Node": {"type": "object", "properties": {"n": {"type": "integer"},
                                               "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}},
     "A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"},
     "Alt": {"anyOf": [{"$ref": "#/$defs/Alt"}, {"type": "number"}]}, "C": {"$ref": "#/$defs/D"},
     "D": {"type": "object"}}
# (the value's schema, the value as the parse gives it, the typed value): the reference's `_fix_arg_types`, `$ref` aside
TYPING = [
    ({"type": "integer"}, "42", 42), ({"type": "integer"}, "4.0", 4.0), ({"type": "integer"}, "x", "x"),
    ({"type": "integer"}, "123456789012345678901234567890", 123456789012345678901234567890),
    ({"type": "number"}, "123456789012345678901234567890", 123456789012345678901234567890),
    ({"type": "number"}, "2.5", 2.5), ({"type": "number"}, "2.0", 2), ({"type": "number"}, "1e400", "1e400"),
    ({"type": "number"}, "NaN", "NaN"), ({"type": "number"}, "Infinity", "Infinity"), ({"type": "number"}, "-0", 0),
    ({"type": "float"}, "3.14", 3.14), ({"type": "int64"}, "7", 7), ({"type": "bool"}, "TRUE", True),
    ({"type": "boolean"}, "True", True), ({"type": "boolean"}, "False", False), ({"type": "boolean"}, "1", True),
    ({"type": "boolean"}, "0", False), ({"type": "boolean"}, "yes", "yes"),
    ({"type": "null"}, "null", None), ({"type": "null"}, "None", "None"),
    ({"type": ["string", "integer"]}, "5", 5), ({"type": ["integer", "null"]}, "null", None),
    ({"type": ["number", "boolean"]}, "true", True), ({"type": "string"}, "5", "5"),
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "7", 7),
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "null", None),
    ({"anyOf": [{"type": "object"}, {"type": "null"}]}, '{"key": "value", "count": 42}', {"key": "value", "count": 42}),
    ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "5", 5), ({"enum": [1, 2]}, "2", 2),
    ({"enum": ["a", None]}, "null", None), ({}, "5", "5"), ({"description": "untyped"}, '{"a": 1}', '{"a": 1}'),
    ({"type": "object"}, '{"a": NaN}', '{"a": NaN}'), ({"type": "array"}, '["1", 2]', ["1", 2]),
    ({"type": "array", "items": {"type": "integer"}}, ["1", "x", 2], [1, "x", 2]),
    ({"type": "array", "items": {"type": "array", "items": {"type": "boolean"}}}, [["true"], ["0"]], [[True], [False]]),
    ({"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}},
     {"n": "3", "s": "3", "o": "3"}, {"n": 3, "s": "3", "o": "3"}),
    ({"type": "integer"}, 5, 5), ({"type": "string"}, 5, "5"), ({"type": "number"}, True, True),
    # a `$ref` into the tool's `$defs` (or `definitions`) is followed; a cycle of refs alone is not
    ({"$ref": "#/$defs/P"}, "5", 5), ({"$ref": "#/definitions/P"}, "5", 5),
    ({"type": "array", "items": {"$ref": "#/$defs/Obj"}}, [{"count": "2"}, {"count": "x"}],
     [{"count": 2}, {"count": "x"}]),
    ({"$ref": "#/$defs/Node"}, {"n": "1", "kids": [{"n": "2", "kids": [{"n": "3"}]}]},
     {"n": 1, "kids": [{"n": 2, "kids": [{"n": 3}]}]}),
    ({"anyOf": [{"$ref": "#/$defs/P"}, {"type": "null"}]}, "null", None), ({"$ref": "#/$defs/A"}, "5", "5"),
    ({"$ref": "#/$defs/Alt"}, "2.5", 2.5), ({"$ref": "#/$defs/missing"}, "5", "5"),
    # a chain that re-enters a definition an enclosing schema followed (no cycle): followed, its type over its own
    ({"type": ["string", "integer"], "allOf": [{"type": "null", "allOf": [
        {"$ref": "#/$defs/D", "allOf": [{}, {"$ref": "#/$defs/C", "type": "number"}]}]}]}, "1e3", 1000.0),
]


def same(a, b) -> bool:
    """Equal, with NaN equal to itself and the types kept apart (1 is not True)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize(("schema", "value", "want"), TYPING, ids=[str(i) for i in range(len(TYPING))])
def test_the_typing(schema, value, want) -> None:
    """`fix_types` on one argument, as each family's parse gives it (a Qwen string, or a Gemma value), with the tool's
    `$defs` and `definitions` beside it; `schema_types` and `coerce` underneath."""
    params = {"type": "object", "properties": {"x": schema}, "$defs": D, "definitions": {"P": D["P"]}}
    got = json.loads(tool_args.fix_types(json.dumps({"x": value}), params))["x"]
    assert same(got, want), (got, want)


def test_schema_types_and_coerce() -> None:
    assert sorted(tool_args.schema_types({"type": ["integer", "null"], "anyOf": [{"enum": [True, 1.5, "a"]}]})) == \
        ["boolean", "integer", "null", "number", "string"]
    assert tool_args.schema_types({"$ref": "#/$defs/P"}) == ["string"]  # no definitions given
    assert tool_args.schema_types({"$ref": "#/$defs/P"}, D) == ["integer"]
    assert tool_args.schema_types({"$ref": "#/$defs/A"}, D) == ["string"]  # a cycle of refs alone
    assert tool_args.coerce("5", ["string", "integer"]) == 5 and tool_args.coerce("5", "string") == "5"
    assert tool_args.coerce("1.5", ["integer", "number"]) == 1.5 and tool_args.coerce("x", ["integer"]) == "x"
    assert same(tool_args.coerce("Infinity", "number"), "Infinity") and tool_args.coerce("[1]", "object") == [1]


def test_text_the_detokenizer_attaches_to_a_marker_is_kept() -> None:
    """Gemma's byte-fallback tokens of an incomplete UTF-8 sequence before a marker: the detokenizer gives their
    replacement characters with the marker's text ('��<|tool_call>', '��<|"|>'); they stay, in the content and in the
    string before its closing quote, streamed and not, at every chunk size."""
    tok, _ = tokenizer(GEMMA4)
    b1, b2 = tok.token_to_id("<0xE2>"), tok.token_to_id("<0x82>")

    def enc(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False).ids

    ids = [*enc("Hi"), b1, b2, 48, *enc("call:f{x:"), 52, b1, b2, 52, *enc("}"), 49, 50]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": "string"}}}}}]
    for k in CHUNKS:
        content, calls, streamed, scalls, _ = parse(GEMMA4, ids, tools, "auto", k)
        assert (content, streamed) == ("Hi��", "Hi��"), k
        assert calls == scalls == [["f", '{"x": "��"}']], k


QWEN_SCAN_PIECES = ["<parameter=", "< parameter =\n", "</parameter>", "< / parameter >", ">", "<", "a", " ", "\n", "=",
                    "x>", "</function>", "<param", "= >", "=>"]


@pytest.mark.parametrize("piece", QWEN_SCAN_PIECES)
def test_the_qwen_parameter_scan_is_linear_in_the_text(piece) -> None:
    """The Qwen parameter scan (`tool_args.qwen_arguments`) takes time linear in the text, whatever its markup: a run of
    one piece of the markup about 64 KB long, and one 4 times longer, the second within 8 times the first (a quadratic
    scan takes 16). The texts are built before the timing and small enough to stay in the cache; each measure repeats
    the call until the shorter run takes some milliseconds."""
    n = 65_536 // len(piece)
    small_text, big_text = piece * n, piece * (4 * n)

    def timed(text: str, reps: int):
        return lambda: [tool_args.qwen_arguments(text) for _ in range(reps)]

    reps = 1
    while least_each(timed(small_text, reps), runs=1)[0] < 0.005 and reps < 4096:
        reps *= 2
    small, big = least_each(timed(small_text, reps), timed(big_text, reps))
    assert big < 8 * small + 0.002, (piece, reps, small, big)


@pytest.mark.parametrize("case", ["gemma-object", "gemma-array", "qwen-array"])
def test_a_degenerate_nesting_finishes_the_call(case) -> None:
    """A call nested far deeper than `MAX_NESTING`, streamed and not, after a typed member and before another: 3,000
    Gemma objects or arrays keep the value as written, a string, and the pieces stop at the deep member (the call's end
    sends the rest); a Qwen parameter typed as an array 100,000 levels deep stays a string, held as any value that is
    not a string type until the next member begins; the members around it typed. 100 levels convert and type."""
    fam = GEMMA4 if case.startswith("gemma") else QWEN3
    tok, _ = tokenizer(fam)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "x": {"type": "object" if case == "gemma-object" else "array"},
        "z": {"type": "integer"}}}}}]

    def call(depth: int) -> tuple[list[int], str]:
        if fam == GEMMA4:
            value = "{a:" * depth + "1" + "}" * depth if case == "gemma-object" else "[" * depth + "1" + "]" * depth
            text = f"<|tool_call>call:f{{n:5,x:{value},z:6}}<tool_call|>"
            return [*tok.encode(text, add_special_tokens=False).ids, 50], value
        value = "[" * depth + "]" * depth
        p = "<parameter={}>\n{}\n</parameter>\n".format
        text = f"<tool_call>\n<function=f>\n{p('n', '5')}{p('x', value)}{p('z', '6')}</function>\n</tool_call>"
        return [*tok.encode(text, add_special_tokens=False).ids, 248046], value

    ids, value = call(3000 if fam == GEMMA4 else 100_000)
    want = json.dumps({"n": 5, "x": value, "z": 6})
    for k in (1, 7, 1000):
        content, calls, _, scalls, p = parse(fam, ids, tools, "auto", k)
        assert content is None and calls == scalls == [["f", want]], k
        assert p.calls_[0].stream.broken == (fam == GEMMA4) and p.calls_[0].stream.sent.startswith('{"n": 5, "x": '), k
    ids, _ = call(100)
    _, calls, _, _, _ = parse(fam, ids, tools, "auto", 1000)
    got = json.loads(calls[0][1])
    value = got["x"]
    for _ in range(99):
        value = value["a"] if case == "gemma-object" else value[0]
    assert value == ({"a": 1} if case == "gemma-object" else ["1"] if fam == GEMMA4 else [])  # an open dict's 1
    assert (got["n"], got["z"]) == (5, 6)


CAP = 256  # tool_args.MAX_NESTING


@pytest.mark.parametrize("depth", [CAP - 1, CAP, CAP + 1])
@pytest.mark.parametrize("kind", ["object", "array"])
@pytest.mark.parametrize("fam", [GEMMA4, QWEN3])
def test_the_nesting_cap(fam, kind, depth) -> None:
    """A member's value converts and types to `MAX_NESTING` levels, its own bracket level 1; one level deeper stays as
    written, a string, the same in the stream and the final arguments, at every step; the members around it are
    converted and typed."""
    tok, _ = tokenizer(fam)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "x": {"type": kind}, "z": {"type": "integer"}}}}}]
    if fam == GEMMA4:
        value = "{a:" * depth + "1" + "}" * depth if kind == "object" else "[" * depth + "1" + "]" * depth
        text = f"<|tool_call>call:f{{n:5,x:{value},z:6}}<tool_call|>"
        ids = [*tok.encode(text, add_special_tokens=False).ids, 50]
    else:
        value = '{"a":' * depth + "1" + "}" * depth if kind == "object" else "[" * depth + "1" + "]" * depth
        p = "<parameter={}>\n{}\n</parameter>\n".format
        text = f"<tool_call>\n<function=f>\n{p('n', '5')}{p('x', value)}{p('z', '6')}</function>\n</tool_call>"
        ids = [*tok.encode(text, add_special_tokens=False).ids, 248046]
    want: object = value
    if depth <= CAP:  # converted; an open object's bare 1 an integer, an untyped array's Gemma 1 a string
        want = "1" if fam == GEMMA4 and kind == "array" else 1
        for _ in range(depth):
            want = {"a": want} if kind == "object" else [want]
    for k in (1, 7, 1000):
        _, calls, _, scalls, _ = parse(fam, ids, tools, "auto", k)
        assert calls == scalls == [["f", json.dumps({"n": 5, "x": want, "z": 6})]], k
    if depth > CAP:
        assert tool_args.MAX_NESTING == CAP


@pytest.mark.parametrize("depth", [CAP - 1, CAP, CAP + 1])
@pytest.mark.parametrize("kinds,cut", [("oa", False), ("ao", False), ("oa", True), ("ao", True), ("a", True),
                                       ("o", True)], ids=["obj-arr", "arr-obj", "obj-arr-cut", "arr-obj-cut",
                                                          "arr-cut", "obj-cut"])
def test_the_nesting_cap_counts_every_level(kinds, cut, depth) -> None:
    """Gemma: objects and arrays in turn, starting either way, closed or cut (no closers, then the call's closer).
    Also cut chains of arrays or objects alone. `MAX_NESTING` levels convert, a cut chain with its innermost value
    open (`{}` or `[]`). One level more stays as written, a string. The same in the stream and the final arguments."""
    tok, _ = tokenizer(GEMMA4)
    opens = ["{a:" if kinds[i % len(kinds)] == "o" else "[" for i in range(depth)]
    value = "".join(opens) + "1"
    if not cut:
        value += "".join("}" if o == "{a:" else "]" for o in reversed(opens))
    obj = opens[0] == "{a:"
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "x": {"type": "object" if obj else "array"}, "z": {"type": "integer"}}}}}]
    text = f"<|tool_call>call:f{{n:5,x:{value}" + ("<tool_call|>" if cut else ",z:6}<tool_call|>")
    ids = [*tok.encode(text, add_special_tokens=False).ids, 50]
    want: object = value
    if depth <= CAP:  # an open object's bare 1 an integer, an untyped array's a string; a cut value's innermost open
        levels = opens[:-1] if cut else opens
        want = ({} if opens[-1] == "{a:" else []) if cut else (1 if obj else "1")
        for o in reversed(levels):
            want = {"a": want} if o == "{a:" else [want]
    args = {"n": 5, "x": want} if cut else {"n": 5, "x": want, "z": 6}
    for k in (1, 7, 1000):
        _, calls, _, scalls, _ = parse(GEMMA4, ids, tools, "auto", k)
        assert calls == scalls == [["f", json.dumps(args)]], k


def test_the_nesting_cap_counts_levels_not_brackets() -> None:
    """The cap reads the levels open at once, outside strings: a value of `MAX_NESTING` levels beside another bracket
    converts; brackets inside a string (after an escaped quote too, or in one the text leaves open) are no level; a
    value deeper than the cap stays a string whatever its type."""
    chain = "[" * (CAP - 1) + "1" + "]" * (CAP - 1)
    assert not tool_args._too_deep("[" + chain + ",[2]]") and tool_args._too_deep("[[" + chain + "]]")
    assert not tool_args._too_deep("[" + '{"a":1},' * 300 + "1]")  # each closed object's level closes
    assert tool_args.coerce("[" + chain + ",[2]]", "array")[1] == [2]
    for text in ('["' + "[" * 300 + '"]', '["\\"' + "[" * 300 + '"]', '["\\\\' + "[" * 300, "[" + '"' + "[" * 300):
        assert not tool_args._too_deep(text), text
    deep = "[" * (CAP + 1) + "1" + "]" * (CAP + 1)
    for kind in ("integer", "number", "string", "array", "object", ["null", "array"]):
        assert tool_args.coerce(deep, kind) == deep, kind
    assert tool_args.coerce("[" * CAP + "1" + "]" * CAP, "integer") != "[" * CAP + "1" + "]" * CAP  # decoded


@pytest.mark.parametrize("gemma", [True, False], ids=["gemma", "qwen"])
def test_whitespace_held_in_an_open_value_comes_whole(gemma) -> None:
    """Whitespace an open string holds back (Gemma's before a possible `<|"|>` or closer, Qwen's after a `<` that may
    begin the value's end) is sent once text follows it, whitespace-only pieces included: the pieces and the call's end
    are the final arguments, each piece their prefix, at every step size."""
    schema = {"type": "object", "properties": {"k": {"type": "string"}, "n": {"type": "integer"}}}
    if gemma:
        bodies = ['k:<|"|>x' + " " * 9 + 'y<|"|>,n:1}', 'k:<|"|>x<|' + " \n " * 3 + 'y<|"|>}',
                  'k:<|"|>x}' + "  " * 4 + 'z<|"|>}', 'k:<|"|>x' + " " * 7 + '<|"|>}']
    else:
        bodies = ["<parameter=k>\nx<" + " " * 9 + "y\n</parameter>\n<parameter=n>\n1\n</parameter>\n",
                  "<parameter=k>\nx< /" + "  " * 4 + "z\n</parameter>\n",
                  "<parameter=k>\nx<" + " " * 5 + "/parameter>\n", "<parameter=k>\nx</ " + "\n" * 3 + "parameter>\n"]
    for body in bodies:
        final = final_args(gemma, schema, body)
        for k in (1, 2, 3, 5):
            pieces, s = stream_args(gemma, schema, chunked(body, k))
            sent = ""
            for piece in pieces:
                sent += piece
                assert final.startswith(sent), (body, k, sent, final)
            assert sent + (s.flush(final) or "") == final, (body, k)


def test_qwen_parameters_side_by_side_all_stream() -> None:
    """Qwen parameters with nothing between them (`</parameter><parameter=`, which Qwen3.8's tokenizer can split as
    `><`): a piece that ends one parameter and a bare `<` leaves the next one to the next piece, and a parameter named
    by whitespace alone (`""` once it closes) is held until it closes: the pieces and the call's end are the final
    arguments, each piece their prefix, at every step size and through the tokens."""
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in "abc"}}
    tok, _ = tokenizer(QWEN3)
    for body in ("<parameter=a>hello</parameter><parameter=b>x</parameter><parameter=c>world</parameter>",
                 "<parameter=a>1</parameter><parameter=\n>x</parameter><parameter=b>2</parameter>"):
        final = final_args(False, schema, body)
        texts = [chunked(body, k) for k in range(1, 8)]
        texts += [[body[:j], *chunked(body[j:], k)] for k in (2, 3, 5) for j in range(1, k)]
        for t in texts:
            pieces, s = stream_args(False, schema, [x for x in t if x])
            sent = ""
            for piece in pieces:
                sent += piece
                assert final.startswith(sent), (body, t, sent)
            assert sent + (s.flush(final) or "") == final, (body, t)
        text = f"<tool_call>\n<function=f>\n{body}\n</function>\n</tool_call>"
        ids = [*tok.encode(text, add_special_tokens=False).ids, 248046]
        tools = [{"type": "function", "function": {"name": "f", "parameters": schema}}]
        for k in (1, 2, 3):
            _, calls, _, scalls, _ = parse(QWEN3, ids, tools, "auto", k)
            assert calls == scalls == [["f", final]], k


def converted(*bodies: str) -> tuple[list[str | None], tuple[float, ...]]:
    """Gemma's converter on each body: its JSON (None when it gives way), and its time (`least_each`)."""
    out: list = [None] * len(bodies)

    def once(i: int, body: str):
        def run() -> None:
            try:
                out[i] = tool_args.gemma4_arguments(body)
            except RecursionError:
                out[i] = None
        return run
    times = least_each(*(once(i, b) for i, b in enumerate(bodies)))
    return out, times


@pytest.mark.parametrize("opener", ["{a:", "["])
def test_a_deep_gemma_argument_gives_way_at_once(opener) -> None:
    """Every level of Gemma's converter takes its closer from one scan of the body: a body nested deeper than the
    cap (a model repeating `{a:` or `[`, 40,000 levels) gives way in milliseconds (4 times the depth, about 4 times
    the time), and the call keeps that value as written; a body nested 200 levels with a long tail converts in time
    linear in its length; 100 levels convert."""
    (out, out_big), (small, big) = converted("x:" + opener * 10_000, "x:" + opener * 40_000)
    assert out is None and out_big is None and big < 8 * small + 0.05 and big < 0.5, (small, big)
    closer = "}" if opener == "{a:" else "]"

    outs, (small, big) = converted(*("x:" + opener * 200 + "y" * n + closer * 200 for n in (20_000, 80_000)))
    for out in outs:
        assert out is not None and json.loads(out)["x"], out
    assert big < 8 * small + 0.05 and big < 1, (small, big)
    (out,), _ = converted("x:" + opener * 100 + "1" + closer * 100)
    value = json.loads(out)["x"]
    for _ in range(99):
        value = value["a"] if opener == "{a:" else value[0]
    assert value == ({"a": "1"} if opener == "{a:" else ["1"])
    tok, _ = tokenizer(GEMMA4)
    body = "x:" + opener * 40_000
    ids = [*tok.encode(f"<|tool_call>call:f{{{body}<tool_call|>", add_special_tokens=False).ids, 50]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": "object"}}}}}]
    for k in (7, 1000):
        _, calls, _, scalls, _ = parse(GEMMA4, ids, tools, "auto", k)
        assert calls == scalls == [["f", json.dumps({"x": opener * 40_000})]], k


def test_brackets_the_converter_does_not_enter_convert_as_before() -> None:
    """Brackets inside a bare value or a key are no nesting: such bodies convert as gemma4.py's converter converts
    them, however many there are."""
    assert json.loads(tool_args.gemma4_arguments("a:x" + "{" * 2000)) == {"a": "x" + "{" * 2000}
    assert json.loads(tool_args.gemma4_arguments("{" * 2000 + ":1")) == {"{" * 2000: "1"}
    assert json.loads(tool_args.gemma4_arguments("a:[" + "x{," * 1100 + "]")) == {"a": ["x{"] * 1100}


def test_the_typing_walks_a_ref_diamond_once() -> None:
    """Definitions that are each an anyOf of three refs to the next (1.2 KB at 12 levels): each definition's types are
    computed once per call, so every depth up to 20 types at once, 20 levels about twice the time of 10."""
    def schema(n: int) -> dict:
        defs = {f"D{i}": {"anyOf": [{"$ref": f"#/$defs/D{i + 1}"} for _ in range(3)]} for i in range(n)}
        defs[f"D{n}"] = {"type": "integer"}
        params = {"type": "object", "properties": {"p": {"$ref": "#/$defs/D0"}, "q": {"$ref": "#/$defs/D0"}},
                  "$defs": defs}
        assert json.loads(tool_args.fix_types('{"p": "5", "q": "x"}', params)) == {"p": 5, "q": "x"}
        return params

    def typing(params: dict):
        return lambda: tool_args.fix_types('{"p": "5", "q": "x"}', params)

    schemas, times = {}, {}
    for n in range(10, 21, 2):  # in steps, so that a walk per path fails at its first slow depth
        schemas[n] = schema(n)
        times[n], = least_each(typing(schemas[n]), runs=3)
        assert times[n] < 0.5, times
    t10, t20 = least_each(typing(schemas[10]), typing(schemas[20]))  # the ratio, the runs interleaved
    assert t20 < 4 * t10 + 0.05, (t10, t20)


def test_a_cycle_of_refs_types_in_linear_time() -> None:
    """A cycle of definitions that are each a `$ref` to the next, under every item of an array: the chain is followed
    once a call, so 4 times the cycle and the items take about 4 times the time."""
    def case(m: int):
        defs = {f"A{i}": {"$ref": f"#/$defs/A{(i + 1) % m}"} for i in range(m)}
        params = {"type": "object", "properties": {"x": {"type": "array", "items": {"$ref": "#/$defs/A0"}}},
                  "$defs": defs}
        args = json.dumps({"x": ["5"] * m})
        assert json.loads(tool_args.fix_types(args, params)) == {"x": ["5"] * m}  # a cycle of refs alone: strings
        return lambda: tool_args.fix_types(args, params)

    small, big = least_each(case(500), case(2000))
    assert big < 8 * small + 0.05 and big < 1, (small, big)


@pytest.mark.parametrize("fam", [GEMMA4, QWEN3])
def test_a_root_ref_is_typed(fam) -> None:
    """`parameters` that is a `$ref` (pydantic's recursive model) types its definition's properties, as the grammar
    takes them: Qwen's `"1"` for an integer is 1, Gemma's bare `1` too; a string property stays a string."""
    tok, _ = tokenizer(fam)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {
        "$ref": "#/$defs/P", "$defs": {"P": {"type": "object", "properties": {
            "count": {"type": "integer"}, "name": {"type": "string"}, "next": {"$ref": "#/$defs/P"}}}}}}}]
    if fam == GEMMA4:
        Q = '<|"|>'
        text = f"<|tool_call>call:f{{count:1,name:{Q}2{Q},next:{{count:3}}}}<tool_call|>"
        ids = [*tok.encode(text, add_special_tokens=False).ids, 50]
    else:
        p = "<parameter={}>\n{}\n</parameter>\n".format
        params = p("count", "1") + p("name", "2") + p("next", '{"count": 3}')
        text = f"<tool_call>\n<function=f>\n{params}</function>\n</tool_call>"
        ids = [*tok.encode(text, add_special_tokens=False).ids, 248046]
    for k in (1, 1000):
        _, calls, _, scalls, _ = parse(fam, ids, tools, "auto", k)
        assert calls == scalls and json.loads(calls[0][1]) == {"count": 1, "name": "2", "next": {"count": 3}}, k


def test_gemma_open_values_take_their_literal_types() -> None:
    """Inside an object schema without `properties` (an open dict), Gemma's bare values take their literal types,
    nested open values too: integers, numbers, `true`, `false`, `null`. A value between `<|"|>` stays a string, and so
    does other bare text. Declared properties are typed by their schema, and Qwen's open values (raw text)
    stay strings."""
    tok, _ = tokenizer(GEMMA4)
    Q = '<|"|>'
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "o": {"type": "object"}, "n": {"type": "string"}, "rows": {"type": "array", "items": {"type": "object"}},
        "u": {"description": "untyped"}, "closed": {"type": "object", "properties": {"k": {"type": "string"}}}}}}},
        {"type": "function", "function": {"name": "free", "parameters": {"type": "object"}}}]
    o = (f"{{a:1,b:-2.5,c:true,d:false,e:null,f:{Q}7{Q},g:{{h:1e3,i:[1,{Q}x{Q},{{j:null}},[true]]}},k:abc,l:007,"
         f"m:True,big:123456789012345678901234567890,inf:1e400}}")
    text = (f"<|tool_call>call:f{{closed:{{k:5}},n:5,o:{o},rows:[{{z:2}}],u:7}}<tool_call|>"
            f"<|tool_call>call:free{{a:1,b:{{c:false}},s:{Q}1{Q}}}<tool_call|>")
    ids = [*tok.encode(text, add_special_tokens=False).ids, 50]
    want_o = {"a": 1, "b": -2.5, "c": True, "d": False, "e": None, "f": "7",
              "g": {"h": 1000.0, "i": [1, "x", {"j": None}, [True]]}, "k": "abc", "l": 7, "m": "True",
              "big": 123456789012345678901234567890, "inf": "1e400"}
    for k in CHUNKS:
        _, calls, _, scalls, _ = parse(GEMMA4, ids, tools, "auto", k)
        assert calls == scalls, k
        got = [json.loads(c[1]) for c in calls]
        assert same(got[0], {"closed": {"k": "5"}, "n": "5", "o": want_o, "rows": [{"z": 2}], "u": "7"}), got[0]
        assert same(got[1], {"a": 1, "b": {"c": False}, "s": "1"}), got[1]
    qtok, _ = tokenizer(QWEN3)
    p = "<parameter={}>\n{}\n</parameter>\n".format
    qtext = f"<tool_call>\n<function=free>\n{p('a', '1')}{p('b', 'true')}</function>\n</tool_call>"
    _, calls, _, _, _ = parse(QWEN3, [*qtok.encode(qtext, add_special_tokens=False).ids, 248046], tools, "auto", 1)
    assert json.loads(calls[0][1]) == {"a": "1", "b": "true"}


def test_a_typing_failure_keeps_the_converted_arguments(monkeypatch) -> None:
    """The guard behind the cap: with `MAX_NESTING` raised past the interpreter's limits, a recursive `$ref` typed
    deeper than the typing recurses (600 Gemma levels) keeps that member as the converter gave it, untyped, the member
    before it typed, streamed and not; 100 levels are typed."""
    monkeypatch.setattr(tool_args, "MAX_NESTING", 100_000, raising=False)
    tok, _ = tokenizer(GEMMA4)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"k": {"type": "integer"}, "n": {"$ref": "#/$defs/N"}}, "$defs": {"N": {
            "type": "object", "properties": {"v": {"type": "integer"}, "n": {"$ref": "#/$defs/N"}}}}}}}]

    def call(depth: int) -> list[int]:
        text = "<|tool_call>call:f{k:7,n:" + "{v:1,n:" * depth + "{v:1}" + "}" * depth + "}<tool_call|>"
        return [*tok.encode(text, add_special_tokens=False).ids, 50]

    def untyped(depth: int) -> dict:
        v: dict = {"v": "1"}
        for _ in range(depth):
            v = {"v": "1", "n": v}
        return {"k": 7, "n": v}

    for k in (1, 7, 1000):
        _, calls, _, scalls, _ = parse(GEMMA4, call(600), tools, "auto", k)
        assert calls == scalls and json.loads(calls[0][1]) == untyped(600), k
    _, calls, _, _, _ = parse(GEMMA4, call(100), tools, "auto", 1000)
    got = json.loads(calls[0][1])["n"]
    for _ in range(100):
        assert got["v"] == 1
        got = got["n"]
    assert got == {"v": 1}


def test_the_assembly_rules() -> None:
    """The reference vLLM's message-assembly, stream-processor and content-none tests, as a table: which choices keep
    the calls, the content under a forced choice, the finish reason, the first call without parallel_tool_calls."""
    fam = QWEN3
    tok, _ = tokenizer(fam)
    call = "<tool_call>\n<function=calculate_area>\n<parameter=shape>\nx\n</parameter>\n</function>\n</tool_call>"
    tools = VECTORS["tools"][VECTORS["assembly_tools"]]
    for row in VECTORS["assembly"]:
        choice = row["tool_choice"]
        text = (row.get("content") or "") + ((call + "\n" + call) if row["calls"] else "")
        ids = [*tok.encode(text, add_special_tokens=False).ids, 248046]
        content, calls, _, _, p = parse(fam, ids, tools, choice, 1, parallel=row.get("parallel_tool_calls", True))
        assert bool(calls) == (row["keeps"] and row["calls"]), row
        if "content_out" in row:
            assert content == row["content_out"], row
        if row.get("first_only"):
            assert len(calls) == 1
        if "finish" in row:
            assert api_finish_reason(row.get("engine", "stop"), p) == row["finish"], row


def recorded() -> list[tuple[str, dict]]:
    out = []
    for path in sorted((DATA / "recorded").glob("*.json")) if (DATA / "recorded").exists() else ():
        data = json.loads(path.read_text())
        out += [(path.stem, r) for r in data["responses"]]
    return out


@pytest.mark.parametrize(("source", "rec"), recorded(), ids=[f"{s}-{r['case']}" for s, r in recorded()])
def test_recorded_outputs(source, rec) -> None:
    """The models' own outputs (the GPU recordings, cut down): our parse is the reference vLLM's, edited where this
    server deviates, at every chunk size."""
    ref = None
    for k in CHUNKS:
        content, calls, streamed, scalls, _ = parse(rec["family"], rec["ids"], rec["tools"], rec["choice"], k,
                                                    parallel=rec.get("parallel", True))
        assert (content, calls) == (rec["content"], rec["calls"]), k
        if ref is None:
            ref = (streamed, scalls)
        assert (streamed, scalls) == ref, k


# -- a call's arguments in pieces (`tool_args.ArgStream`), each a prefix of the final arguments -----------------------

def chunked(text: str, k: int) -> list[str]:
    return [text[j:j + k] for j in range(0, len(text), k)]


def stream_args(gemma: bool, schema, text_pieces: list[str]) -> tuple[list[str], tool_args.ArgStream]:
    s = tool_args.ArgStream(gemma, schema)
    out = []
    for i, t in enumerate(text_pieces):
        got = s.append(t) if i == 0 else s.piece(t)
        if got:
            out.append(got)
    return out, s


def test_the_pieces_are_the_final_arguments_prefix() -> None:
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                               "tags": {"type": "array", "items": {"type": "string"}}}}
    body = 'city:<|"|>New York<|"|>,days:12,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}'
    final = final_args(True, schema, body)
    for k in (1, 2, 3, 5, 8):
        pieces, s = stream_args(True, schema, chunked(body, k))
        rest = s.flush(final)
        assert "".join(pieces) + (rest or "") == final == '{"city": "New York", "days": 12, "tags": ["a", "b"]}'
        assert pieces[0].startswith('{"city": "')  # the string streams
        assert not any(x.endswith('"days": 1') for x in pieces)  # the number waits for the member after it


def final_args(gemma: bool, schema, body: str) -> str:
    """The call's final arguments, as `ToolCallParser._end_call` gives them."""
    if gemma:
        parsed = tool_args.gemma4_parse(body, per_member=True)
        return tool_args.fix_types(json.dumps(parsed, ensure_ascii=False), schema,
                                   tool_args.gemma4_parse(body, per_member=True))
    return tool_args.fix_types(tool_args.qwen_arguments(body), schema)


def streamed_args(gemma: bool, schema, texts: list[str]) -> tuple[str, tool_args.ArgStream]:
    """The pieces of the call's texts, then the call's end: their concatenation."""
    pieces, s = stream_args(gemma, schema, texts)
    return "".join(pieces) + (s.flush(final_args(gemma, schema, "".join(texts))) or ""), s


def qwen_texts(params: list[tuple[str, str]]) -> list[str]:
    """Qwen parameters as its lexer gives them to the call (its markup whole)."""
    return [t for k, v in params for t in ("<parameter=", k, ">", "\n", v, "\n", "</parameter>", "\n")]


@pytest.mark.parametrize("gemma", [True, False], ids=["gemma", "qwen"])
def test_a_key_written_twice_keeps_its_first_value(gemma) -> None:
    """At the top level the first of two equal keys wins, since its value may be sent before the second begins (ours:
    vLLM's converters keep the last at the first's place, and its stream stops there): the pieces stop at the
    second, and the call's end sends the rest."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    body = 'a:<|"|>x<|"|>,b:1,a:<|"|>y<|"|>,b:2}'
    for texts in ([body[i:i + k] for i in range(0, len(body), k)] for k in (1, 2, 3, 7)) if gemma else [
            qwen_texts([("a", "x"), ("b", "1"), ("a", "y"), ("b", "2")])]:
        got, s = streamed_args(gemma, schema, texts)
        assert got == final_args(gemma, schema, "".join(texts)) == '{"a": "x", "b": 1}', texts
        assert s.broken, texts


@pytest.mark.parametrize("kind", ["object", "array"])
def test_a_nesting_deeper_than_the_converter_stops_the_pieces(kind) -> None:
    """3,000 levels: the pieces stop at the member's key; the final arguments keep its value as written, a string, and
    convert the members around it."""
    value = "{a:" * 3000 + "1" + "}" * 3000 if kind == "object" else "[" * 3000 + "1" + "]" * 3000
    body = f"n:5,x:{value},z:6}}"
    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "x": {"type": kind}, "z": {"type": "integer"}}}
    for k in (7, 1000):
        got, s = streamed_args(True, schema, chunked(body, k))
        assert s.broken
        assert got == final_args(True, schema, body) == json.dumps({"n": 5, "x": value, "z": 6}), k


def test_a_qwen_call_cut_inside_a_parameter_keeps_it() -> None:
    """The open parameter's key and string stream (vLLM's pieces), and the final arguments keep it, as Gemma's keep
    a cut value (ours: vLLM's non-streamed parse drops it, so its stream's arguments are not its final's), without
    a cut `</parameter>` (held back in the stream: vLLM's lexer holds it until the output ends)."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}
    texts = ["\n", "<parameter=", "city>", "\nPa", "ris\n", "</parameter>", "\n", "<parameter=", "days>", "\n1"]
    pieces, s = stream_args(False, schema, texts)
    assert "".join(pieces) == '{"city": "Paris", "days": '
    assert s.flush(final_args(False, schema, "".join(texts))) == "1}"
    assert final_args(False, schema, "<parameter=city>\nDal") == '{"city": "Dal"}'
    for cut in ("<", "</", "</para", "< /parameter", "<parameter"):  # a cut end leaves the value, streamed and not
        texts = ["<parameter=", "city", ">", "\n", "Dallas", "\n", cut]
        got, _ = streamed_args(False, schema, texts)
        assert got == final_args(False, schema, "".join(texts)) == '{"city": "Dallas"}', cut
    texts = ["<parameter=", "city", ">", "\n", "a", "<", "b", "\n", "</parameter>"]  # a "<" in a value is sent
    assert streamed_args(False, schema, texts)[0] == final_args(False, schema, "".join(texts)) == '{"city": "a<b"}'


# A call's argument text that begins no member, each piece with a structural character (`ArgStream`'s parse shapes),
# by its length in characters: Gemma's prose with commas, bare values with colons or `<`; Qwen's text then `<`s, and
# openers with no `>`
WORK_SHAPES = {
    "prose": (True, lambda n: ("Some prose, with commas, " * n)[:n]),
    "colons": (True, lambda n: "a:" + "x:" * (n // 2)),
    "lt": (True, lambda n: "a:" + "x<" * (n // 2)),
    "qwen-lt": (False, lambda n: "x" * (n // 2) + "<" * (n // 2)),
    "qwen-openers": (False, lambda n: "<parameter=" * (n // 11)),
}
# Text the stream sends members of, timed: many Gemma members, an array, a long string, a gap after a member, many
# Qwen parameters, a Qwen key written over and over (the pieces stop at its second), and whitespace held in an open
# string. Each with its smaller size in characters: large enough that 16 times its time would pass the bound's floor.
# A body is fed 3 characters at a time, or as the pieces it lists. Gemma's string opens by its own tokens there, so
# the whitespace after it reaches the open string's extender
TIMED_SHAPES = {
    "members": (True, 24_000, lambda n: ",".join(f"k{i}:{i}" for i in range(n // 8)) + "}"),
    "array": (True, 200_000, lambda n: "xs:[" + ",".join(str(i) for i in range(n // 6)) + "]}"),
    "string": (True, 24_000, lambda n: 's:<|"|>' + ("ab, c: {d} [e] " * n)[:n] + '<|"|>}'),
    "gap": (True, 24_000, lambda n: "a:1" + "," * n + "}"),
    "qwen-params": (False, 24_000, lambda n: "".join(f"<parameter=k{i}>\n{i}\n</parameter>\n" for i in range(n // 30))),
    "qwen-repeated": (False, 200_000, lambda n: "<parameter=a>\n1\n</parameter>\n" * (n // 28)),
    "string-spaces": (True, 200_000, lambda n: ["s:", '<|"|>', "x", *chunks3(" " * n), '<|"|>', "}"]),  # held
    "qwen-lt-spaces": (False, 24_000, lambda n: "<parameter=a>\nx<" + " " * n),  # a `<` that may begin the end
}
SMALL = 24_000  # characters: 8,000 chunks of 3, and 4 times that; both above MAX_GAP


def chunks3(text: str) -> list[str]:
    return [text[i:i + 3] for i in range(0, len(text), 3)]


@pytest.mark.parametrize("shape", list(WORK_SHAPES))
def test_the_pieces_parse_work_is_linear(shape, monkeypatch) -> None:
    """The text the converters parse while the pieces stream, summed, at 4 times the text: about the same (a call that
    begins no member in `MAX_GAP` characters stops its pieces, and the call's end sends the rest), never 16 times, as
    a parse of the whole text at each structural character would be. The text is fed 3 characters at a time."""
    gemma, body = WORK_SHAPES[shape]
    work = [0]
    for name in ("_parse_gemma4_args", "qwen_parameters"):
        real = getattr(tool_args, name)

        def counted(text, *a, real=real, **kw):
            work[0] += len(text)
            return real(text, *a, **kw)
        monkeypatch.setattr(tool_args, name, counted)

    def parsed(n: int) -> int:
        work[0] = 0
        stream_args(gemma, None, chunks3(body(n)))
        return work[0]

    small, large = parsed(SMALL), parsed(4 * SMALL)
    assert len(chunks3(body(4 * SMALL))) >= 8_000 and len(body(SMALL)) > tool_args.ArgStream.MAX_GAP
    assert large <= 5 * small, (small, large)


def test_whitespace_after_a_closed_object_parses_nothing(monkeypatch) -> None:
    """Gemma: a member's object, closed by a `}` that may be the call's closer, then whitespace 3 characters at a time.
    The object is parsed once, as it closes; the whitespace begins nothing. So the converter's input, summed, grows as
    the text. At 4 times the object and the whitespace it is about 4 times, not 16."""
    work = [0]
    real = tool_args._parse_gemma4_args

    def counted(text, *a, **kw):
        work[0] += len(text)
        return real(text, *a, **kw)
    monkeypatch.setattr(tool_args, "_parse_gemma4_args", counted)

    def parsed(n: int) -> int:
        work[0] = 0
        stream_args(True, None, chunks3("x:{" + "a:1," * (n // 8) + "z:1}" + " " * (n // 2)))
        return work[0]

    small, large = parsed(6_000), parsed(24_000)
    assert large <= 5 * small, (small, large)


def test_a_member_after_a_closed_object_streams() -> None:
    """After a closed object and whitespace, the next member begins and its string streams before the call ends."""
    body = 'x:{a:1}' + " " * 10 + ',y:<|"|>abc<|"|>}'
    for k in (1, 3):
        pieces, s = stream_args(True, None, chunked(body, k))
        want = final_args(True, None, body)
        assert "".join(pieces) == '{"x": {"a": "1"}, "y": "abc' and want == '{"x": {"a": "1"}, "y": "abc"}', k
        assert s.flush(want) == '"}', k


# Qwen text within `MAX_GAP` that begins no parameter, or one with no name. Its `<`s open nothing, after a parameter
# too; or they come in the value of a parameter named "" or " "
SCAN_SHAPES = {
    "lt": lambda n: "<" * n,
    "lt-p": lambda n: "<p" * (n // 2),
    "lt-gt": lambda n: "<>" * (n // 2),
    "lt-slash": lambda n: "</" * (n // 2),
    "member-lt": lambda n: "<parameter=a>1</parameter>" + "<" * n,
    "no-name-lt": lambda n: "<parameter=>" + "<" * n,
    "blank-name-lt": lambda n: "<parameter= >" + "<" * n,
    "no-name-lt-gt": lambda n: "<parameter=>" + "<>" * (n // 2),
}


@pytest.mark.parametrize("shape", list(SCAN_SHAPES))
def test_the_qwen_scan_resumes_where_it_stopped(shape, monkeypatch) -> None:
    """Qwen's scan runs at each piece with a `<` or `>`, one character at a time. It resumes where no earlier `<` can
    still open a parameter or end a value. So the openers it tries, summed, grow as the text. At 4 times the text they
    are about 4 times, not 16 as a scan of the whole window. Both sizes are within `MAX_GAP`."""
    real, tried = tool_args._OPEN, [0]

    class Counted:
        def match(self, *a):
            tried[0] += 1
            return real.match(*a)
    monkeypatch.setattr(tool_args, "_OPEN", Counted())

    def scanned(n: int) -> int:
        tried[0] = 0
        _, s = stream_args(False, None, list(SCAN_SHAPES[shape](n)))
        assert not s.broken
        return tried[0]

    small, large = scanned(1_000), scanned(4_000)
    assert len(SCAN_SHAPES[shape](4_000)) - 26 <= tool_args.ArgStream.MAX_GAP
    assert large <= 5 * small, (small, large)


def test_a_spaced_qwen_opener_opens() -> None:
    """A `<`, whitespace (up to 4,000 characters, within `MAX_GAP`), then `parameter=`: an opener (qwen3.py's
    `<\\s*parameter\\s*=`), however far its `<` is behind. The scan resumes at that `<` while the text after it may
    still grow into an opener. In short pieces and in long ones, each piece is a prefix of the final arguments, and
    they add up to them."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    for gap, k in ((" " * 50, 3), ("\n\t " * 17, 3), (" " * 4_000, 1)):
        body = f"x<{gap}parameter=a>\nhello\n</parameter>\n<parameter=b>\nx\n</parameter>\n"
        want = final_args(False, schema, body)
        assert want == '{"a": "hello", "b": "x"}'
        cut = body.index("eter=")
        for texts in (chunked(body, k), ["x", body[1:cut], body[cut:]]):
            pieces, s = stream_args(False, schema, texts)
            for i in range(len(pieces)):
                assert want.startswith("".join(pieces[:i + 1])), (gap, texts, pieces)
            assert "".join(pieces) == '{"a": "hello", "b": "x' and s.flush(want) == '"}', (gap, texts)


def test_a_tail_of_the_whole_word_and_whitespace_may_still_open() -> None:
    """A tail `<parameter ` (the whole word, then whitespace) may still grow into an opener: `<parameter =a>`. So the
    scan resumes at its `<`, and the value streams as it grows, not only at the call's end. A part of the word, or
    the word and another character, then whitespace, can't grow into one."""
    s = tool_args.ArgStream(False, None)
    pieces = [s.append("\n")] + [s.piece(t) for t in ["<parameter ", "=a>", "v", "</parameter>"]]
    assert pieces == [None, None, '{"a": "', "v", None] and s.flush('{"a": "v"}') == '"}', pieces
    may = tool_args._may_open
    assert all(may(t, 1) for t in ["x<parameter ", "x< parameter \n", "x<parameter =", "x<para", "x<"])
    assert not any(may(t, 1) for t in ["x<para ", "x<parameterx ", "x<parameter x", "x<param eter", "x<p a"])


def test_a_qwen_parameter_with_no_name_goes_out_as_it_closes() -> None:
    """A parameter named "" or whitespace alone is held while it is open. qwen3.py names it by its last whitespace
    character, then "" once it closes. The `<`s in its value change nothing; the piece that completes its end, cut or
    not, sends it."""
    for name in ("", " ", "\n"):
        for end in (["</para", "meter>"], ["</parameter>"], ["<", "/parameter", ">"]):
            texts = ["\n", f"<parameter={name}>", "x<<", "<>", *end]
            pieces, s = stream_args(False, None, texts)
            want = final_args(False, None, "".join(texts))
            assert want == '{"": "x<<<>"}' and pieces == ['{"": "x<<<>'] and s.flush(want) == '"}', (name, end)


@pytest.mark.parametrize("shape", list(TIMED_SHAPES))
def test_the_pieces_cost_is_linear(shape) -> None:
    """A call's argument text in pieces of 3 characters (or its shape's own): each member converted, typed and sent
    once, the parse from the last member on (an open string by its new text alone, an open array scanned by its new
    text alone), the text not kept once the pieces stop: 4 times the text costs about 4 times the time, not 16 (the
    least of 5 interleaved runs)."""
    gemma, n, body = TIMED_SHAPES[shape]

    def run(n: int):
        chunks = body(n)
        if isinstance(chunks, str):
            chunks = chunks3(chunks)
        return lambda: stream_args(gemma, None, chunks)

    small, large = least_each(run(n), run(4 * n))
    assert large < 6 * small + 0.05, (small, large)


@pytest.mark.parametrize("shape", ["members", "qwen-params"])
def test_the_window_starts_at_the_last_member(shape) -> None:
    """The text a parse reads (`raw`) starts at the last member, so no piece copies the call's whole text. Fed 3
    characters at a time, it never holds more than `MAX_GAP` characters, however many members came before."""
    gemma, n, body = TIMED_SHAPES[shape]
    texts = chunks3(body(n))
    assert len(body(n)) > 6 * tool_args.ArgStream.MAX_GAP
    s = tool_args.ArgStream(gemma, None)
    for i, t in enumerate(texts):
        s.append(t) if i == 0 else s.piece(t)
        assert len(s.raw) <= tool_args.ArgStream.MAX_GAP, (i, len(s.raw))
    assert not s.broken

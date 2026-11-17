"""The keywords a `response_format` takes (`grammar.bounds`): each name of JSON Schema drafts 3 to
2020-12 and of OpenAPI gets the verdict llguidance gives it.

- **The enumeration** (`NAMES`): the keywords llguidance acts on and the subset leaves out are refused (`REFUSED`);
  every other name, and any unknown key, is an annotation, walked past at every level, `$defs` included.
- **llguidance's verdicts** (in a child, its compiled grammar read from its debug log): an annotation changes neither
  llguidance's verdict nor its grammar beside a type or alone, under any draft's `$schema`; each refused keyword
  changes one of them.
- **Beside `anyOf`, `oneOf`, `$ref` and `allOf`**, llguidance merges any key outside its own annotations, so an
  unknown key there stays a 400 (the merging rule); `x-` keys, `discriminator` and `deprecated` pass, as before.
- **draft-04's `id`** stands at the root only in a document that names draft-04, where llguidance resolves the
  `$ref`s under a nested one against it; elsewhere it is an annotation.
- **Tools** keep their closed list.
"""

from __future__ import annotations

import json
import re

import pytest

from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.grammar.spec import json_schema_grammar
from llm_mini_infer.grammar.tools import ToolSchemaError

from ..support.grammar_cases import S, bounds, child, compiled_on_threads, grammar, obj

DRAFT3 = ["$schema", "id", "$ref", "type", "properties", "patternProperties", "additionalProperties", "items",
          "additionalItems", "required", "dependencies", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
          "minItems", "maxItems", "uniqueItems", "pattern", "minLength", "maxLength", "enum", "default", "title",
          "description", "format", "divisibleBy", "disallow", "extends", "links", "fragmentResolution", "readonly",
          "contentEncoding", "pathStart", "mediaType", "alternate", "root"]
DRAFT4 = ["multipleOf", "maxProperties", "minProperties", "allOf", "anyOf", "oneOf", "not", "definitions", "media",
          "base"]
DRAFT6 = ["$id", "const", "contains", "propertyNames", "examples"]
DRAFT7 = ["$comment", "if", "then", "else", "readOnly", "writeOnly", "contentMediaType"]
DRAFT2019 = ["$anchor", "$defs", "$recursiveRef", "$recursiveAnchor", "$vocabulary", "dependentRequired",
             "dependentSchemas", "unevaluatedItems", "unevaluatedProperties", "maxContains", "minContains",
             "deprecated", "contentSchema"]
DRAFT2020 = ["prefixItems", "$dynamicRef", "$dynamicAnchor"]
OPENAPI = ["discriminator", "xml", "externalDocs", "example", "nullable"]
UNKNOWN = ["host", "self", "x-foo", "propertyOrder", "javaType", "errorMessage", "Type", "$foo"]
NAMES = list(dict.fromkeys(DRAFT3 + DRAFT4 + DRAFT6 + DRAFT7 + DRAFT2019 + DRAFT2020 + OPENAPI + UNKNOWN))

DRAFTS = [None, "http://json-schema.org/draft-04/schema#", "http://json-schema.org/draft-07/schema#",
          "https://json-schema.org/draft/2019-09/schema", "https://json-schema.org/draft/2020-12/schema"]


def verdict(k: str) -> str:
    """refused, an annotation, or a keyword of the subset (with its own rules: `$schema` names a draft, `$id` and `id`
    stand at the root)."""
    b = bounds()
    if k in b.REFUSED:
        return "refused"
    if k in b.KEYWORDS and (k not in b.ANNOTATIONS or k in ("$schema", "$id", "id")):
        return "subset"
    return "annotation"


def test_the_enumeration_s_verdicts() -> None:
    """The list as built: llguidance acts on these and the subset leaves them out."""
    assert {k for k in NAMES if verdict(k) == "refused"} == {
        "not", "if", "then", "else", "dependencies", "dependentRequired", "dependentSchemas", "contains", "minContains",
        "maxContains", "propertyNames", "patternProperties", "additionalItems", "unevaluatedItems",
        "unevaluatedProperties", "$dynamicRef", "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor"}


@pytest.mark.parametrize("backend", ["auto", "guidance"])
@pytest.mark.parametrize("k", NAMES)
def test_every_name_gets_its_verdict(k, backend) -> None:
    """A refused keyword is a 400 that names it and its pointer, nested two objects deep and inside a definition. An
    annotation builds there, with a schema, a string, a number or a list as its value, and llguidance compiles it."""
    v = verdict(k)
    if v == "subset":
        return
    for value in ({"type": "string"}, "x", 1, [{"type": "string"}], {"a": ["b"]}):
        nested = obj({"a": obj({"b": {"type": "string", k: value}})})
        defined = {"$defs": {"D": {"type": "integer", k: value}}, **obj({"d": {"$ref": "#/$defs/D"}})}
        if v == "refused":
            for schema, at in ((nested, "#/properties/a/properties/b"), (defined, "#/$defs/D")):
                with pytest.raises(GrammarError, match=re.escape(f"the keyword {k!r} at {at} is not supported")):
                    json_schema_grammar(schema, backend, backend == "guidance")
            continue
        for schema in (nested, defined):
            assert compiled_on_threads(json_schema_grammar(schema, backend, backend == "guidance")) == "", (k, value)


LOG = r'''
import json, os, sys, tempfile
import llguidance
from llm_mini_infer.grammar import spec

TOK = spec._check_tokenizer()


def compiled(schema):
    """llguidance's verdict and its grammar: its debug log, timings dropped."""
    g = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    with tempfile.TemporaryFile() as f:
        old = os.dup(2)
        os.dup2(f.fileno(), 2)
        try:
            m = llguidance.LLMatcher(TOK, g, log_level=2)
            err = m.get_error() if m.is_error() else ""
        finally:
            os.dup2(old, 2)
            os.close(old)
        f.seek(0)
        log = f.read().decode("utf-8", "replace")
    return err, [ln for ln in log.splitlines() if "µs" not in ln and " ms" not in ln and "build grammar" not in ln]


S = {"type": "string"}
BASES = (S, {"type": "integer"}, {"type": "array", "items": S}, {"type": "object", "properties": {"a": S}}, {})
out, seen = {}, {}
for k in json.loads(sys.argv[1]):
    out[k] = 0
    for d in json.loads(sys.argv[2]):
        for base in BASES:
            b = {**base, **({"$schema": d} if d else {})}
            key = json.dumps(b)
            if key not in seen:
                seen[key] = compiled(b)
            for v in ({"type": "string"}, "x", [S]):
                out[k] += compiled({**b, k: v}) != seen[key]
print(json.dumps(out))
'''


def test_the_list_is_llguidance_s() -> None:
    """In a child, for each name outside the subset's: llguidance's verdict and compiled grammar with it and without,
    beside a string, an integer, an array and an object type and alone, under no `$schema` and four drafts', with a
    schema, a string and a list as its value. An annotation changes nothing; each refused keyword changes something."""
    names = [k for k in NAMES if verdict(k) != "subset"]
    got = child(LOG.replace("sys.argv[1]", repr(json.dumps(names))).replace("sys.argv[2]", repr(json.dumps(DRAFTS))),
                120)
    assert got.returncode == 0, got.stderr[-2000:]
    changed = json.loads(got.stdout)
    assert {k for k, n in changed.items() if n} == {k for k in names if verdict(k) == "refused"}, changed


def test_a_content_schema_holding_an_id_is_refused() -> None:
    """llguidance's resolver reads the schemas inside `contentSchema` as resources: one whose `$id` is the root's
    stands for the root, and the grammar becomes its: an integer in place of an object, with no error
    (guidance-ai/llguidance#442). The subset walks past it, so one that holds an `$id` or an `id` is refused; without,
    it is an annotation."""
    swap = {"contentSchema": {"$id": "json-schema:///", "type": "integer"}, **obj({"a": S})}
    got = child(LOG.split("S = {")[0] + f"""
a, b = compiled({json.dumps(swap)!r} and json.loads({json.dumps(swap)!r})), compiled({json.dumps(obj({"a": S}))!r} and
json.loads({json.dumps(obj({"a": S}))!r}))
print(a != b)
""", 60)
    assert got.returncode == 0 and got.stdout.strip() == "True", got.stderr[-1000:]
    for schema in (swap, {"contentSchema": {"type": "object", "properties": {"x": {"id": "a.json"}}}, **obj({"a": S})}):
        with pytest.raises(GrammarError, match="the 'contentSchema' at # is not supported"):
            json_schema_grammar(schema, "guidance", True)
    plain = {"contentMediaType": "application/json", "contentSchema": obj({"foo": S}), **obj({"a": S})}
    assert compiled_on_threads(json_schema_grammar(plain, "guidance", True)) == ""


@pytest.mark.parametrize("backend", ["auto", "guidance"])
def test_beside_a_merging_keyword_only_annotations_pass(backend) -> None:
    """llguidance merges a key outside its annotations into a `$ref` (it inlines the definition): an unknown key there
    is the merging rule's 400, as before. `x-` keys, `discriminator`, `deprecated` and llguidance's own annotations
    (`contentMediaType`, `contentEncoding`, `id`) pass."""
    defs = {"A": obj({"x": S})}
    for k in ("propertyOrder", "javaType", "example"):
        with pytest.raises(GrammarError, match=re.escape(f"{k!r} beside '$ref' at #/properties/r is not supported")):
            json_schema_grammar({**obj({"r": {"$ref": "#/$defs/A", k: 1}}), "$defs": defs}, backend,
                                backend == "guidance")
    for k in ("x-order", "deprecated", "contentMediaType", "contentEncoding", "id"):
        text = json_schema_grammar({**obj({"r": {"$ref": "#/$defs/A", k: "x"}}), "$defs": defs}, backend,
                                   backend == "guidance")
        assert compiled_on_threads(text) == ""


@pytest.mark.parametrize("backend", ["auto", "guidance"])
def test_a_nested_id_under_draft_04_is_refused(backend) -> None:
    """Under draft-04 (a `$schema` naming it, at the root or in a definition) llguidance resolves the `$ref`s under a
    nested `id` against it (guidance-ai/llguidance#444): a 400, as for a nested `$id`. The root's `id`, and a nested
    one under any other draft, are annotations."""
    d4 = "http://json-schema.org/draft-04/schema#"
    inner = obj({"b": {"$ref": "#/definitions/B"}})

    def doc(where: str | None, root_id: bool = False) -> dict:
        a = {**inner, "id": "http://example.com/a.json", **({"$schema": d4} if where == "A" else {})}
        return {**({"$schema": d4} if where == "root" else {}), **({"id": "http://example.com/r.json"} if root_id
                                                                    else {}),
                "definitions": {"A": a, "B": S}, **obj({"a": {"$ref": "#/definitions/A"}})}

    for where in ("root", "A"):
        with pytest.raises(GrammarError,
                           match=re.escape("the 'id' at #/definitions/A is not supported under draft-04")):
            json_schema_grammar(doc(where), backend, backend == "guidance")
    assert compiled_on_threads(json_schema_grammar(doc(None), backend, backend == "guidance")) == ""
    anchor = {"$schema": d4, "definitions": {"A": {"id": "#a", "type": "integer"}},
              **obj({"x": {"$ref": "#/definitions/A"}})}  # a fragment: an anchor, not an id
    assert compiled_on_threads(json_schema_grammar(anchor, backend, backend == "guidance")) == ""
    root = {"$schema": d4, "id": "http://example.com/r.json", "definitions": {"B": S},
            **obj({"a": {"$ref": "#/definitions/B"}})}
    assert compiled_on_threads(json_schema_grammar(root, backend, backend == "guidance")) == ""


@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_a_tool_keeps_its_closed_list(fam) -> None:
    """A tool's schema takes the tools' list alone (tool-calling's rule): an unknown key is a 400 there."""
    for k in ("propertyOrder", "host", "example"):
        with pytest.raises(ToolSchemaError, match=re.escape(f"the keyword {k!r} at #/properties/a is not supported")):
            grammar(fam, obj({"a": {"type": "string", k: 1}}))

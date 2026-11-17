"""The tool-call grammars' checks and cases (tests/tool_calls/test_tool_grammar.py)."""

from __future__ import annotations

import re

from llm_mini_infer.api.tools import template_tool
from llm_mini_infer.grammar import spec


def check(text: str) -> str:
    """The frontend's check of a call grammar, with the token ids its rules name (`GrammarSpecs.tool_calls`)."""
    return spec.check(text, sorted({int(t) for t in re.findall(r"<\[(\d+)\]>", text)}))


def tool(name: str, parameters: dict | None) -> dict:
    return template_tool({"type": "function", "function": {"name": name, "parameters": parameters}})


def chained_defs(chain: bool) -> dict:
    """Definitions of about 400 bytes. Without `chain`: 100 of them, and 100 array parameters, each into its own one.
    With: 14, each pointing to the next (the longest chain the nesting cap takes), and 100 array parameters, each with
    its own `maxItems`, into the first, so that each parameter reaches all 14."""
    n = 14 if chain else 100
    defs = {}
    for i in range(n):
        d = {"type": "object", "properties": {f"f{j}": {"type": "string", "description": "y" * 20} for j in range(8)}}
        if chain and i < n - 1:
            d["properties"]["next"] = {"$ref": f"#/$defs/D{i + 1}"}
        defs[f"D{i}"] = d
    if chain:
        props = {f"a{i}": {"type": "array", "maxItems": i + 1, "items": {"$ref": "#/$defs/D0"}} for i in range(100)}
    else:
        props = {f"a{i}": {"type": "array", "items": {"$ref": f"#/$defs/D{i}"}} for i in range(100)}
    return tool("amp", {"type": "object", "properties": props, "$defs": defs})


# Schema values of a wrong type, each where a keyword takes a schema, a type or a list of keys
MALFORMED = [{"type": [{"a": 1}]}, {"type": [["string"]]}, {"type": 5}, {"type": ["string", None]},
             {"type": "object", "properties": {"a": {}}, "required": 5},
             {"type": "object", "properties": {"a": {}}, "required": [["a"]]},
             {"type": "object", "properties": {"a": {}}, "required": "a"},
             {"type": "object", "properties": {"a": 5}}, {"type": "object", "properties": [{"a": {}}]},
             {"type": "array", "items": 1}, {"type": "object", "properties": {"a": {}}, "additionalProperties": 0},
             {"anyOf": [{"type": "string"}, 3]}]

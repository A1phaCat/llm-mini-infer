"""Diverse JSON schemas with a prompt each, for the validity bar: what llguidance compiles from the keywords clients
use (objects, arrays and their bounds, enums, numbers and their ranges, patterns, formats, optional keys,
`additionalProperties`, `anyOf`/`oneOf`, `$ref` with recursion, `const`, nullable types, tuples,
dictionaries, top-level arrays and strings). The prompts do not show the schemas: the grammar alone shapes the
output. `jsonschema` checks each response that ends by stop (formats included, where it can check them)."""

from __future__ import annotations

from typing import Any


def obj(props: dict[str, Any], required: list[str] | None = None, extra: bool = False, **kw: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": extra, **kw}


S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
N = {"type": "number"}
B = {"type": "boolean"}

CORPUS: list[tuple[str, dict[str, Any], str]] = [
    ("person", obj({"name": S, "age": I, "employed": B}),
     "Describe a fictional person: their name, age and whether they work."),
    ("sentiment", obj({"label": {"enum": ["positive", "negative", "neutral", "mixed"]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1}}),
     "Classify the sentiment of: 'The food was great but the service was painfully slow.'"),
    ("address", obj({"street": S, "city": S, "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
                     "country": {"type": "string", "minLength": 2, "maxLength": 2}}),
     "Give the postal address of a small bakery in a US town."),
    ("order", obj({"order_id": {"type": "string", "pattern": "^ORD-[0-9]{6}$"},
                   "items": {"type": "array", "minItems": 1, "maxItems": 5,
                             "items": obj({"sku": S, "qty": {"type": "integer", "minimum": 1, "maximum": 99},
                                           "price": {"type": "number", "exclusiveMinimum": 0}})},
                   "total": N}),
     "Write the order a customer placed for three office supplies."),
    ("event", obj({"title": S, "date": {"type": "string", "format": "date"},
                   "start": {"type": "string", "format": "time"}, "attendees": {"type": "array", "items": S}}),
     "Plan a team meeting next week: title, date, start time and who comes."),
    ("contact", obj({"name": S, "email": {"type": "string", "format": "email"},
                     "website": {"type": "string", "format": "uri"}}, required=["name", "email"]),
     "Make up the contact card of a freelance designer."),
    ("optional_keys", obj({"title": S, "subtitle": S, "year": I, "isbn": S}, required=["title"]),
     "Describe a classic novel."),
    ("open_object", {"type": "object", "properties": {"kind": {"enum": ["note", "task"]}}, "required": ["kind"]},
     "Write a short to-do note with any fields you like."),
    ("one_of", {"oneOf": [obj({"circle": obj({"radius": N})}),
                          obj({"rect": obj({"w": N, "h": N})})]},
     "Describe one geometric shape with its dimensions."),
    ("any_of_nullable", obj({"value": {"anyOf": [I, {"type": "null"}]},
                             "unit": {"type": ["string", "null"]}}),
     "Report a sensor reading that may be missing."),
    ("tree", {"$defs": {"node": obj({"name": S, "children": {"type": "array", "maxItems": 3,
                                                               "items": {"$ref": "#/$defs/node"}}})},
              "$ref": "#/$defs/node"},
     "Show a small company org chart: a CEO and two levels below."),
    ("const_version", obj({"version": {"const": 2}, "status": {"enum": ["ok", "degraded", "down"]}}),
     "Report the health of a web service."),
    ("bounded_strings", obj({"code": {"type": "string", "minLength": 3, "maxLength": 3},
                             "summary": {"type": "string", "maxLength": 80}}),
     "Summarise the weather in one short line, with an airport code."),
    ("integers", obj({"count": {"type": "integer", "minimum": 0, "maximum": 1000},
                      "even": {"type": "integer", "multipleOf": 2},
                      "negative": {"type": "integer", "maximum": -1}}),
     "Give three numbers: a count of apples, an even number and a negative one."),
    ("numbers", obj({"lat": {"type": "number", "minimum": -90, "maximum": 90},
                     "lon": {"type": "number", "minimum": -180, "maximum": 180},
                     "elevation_m": N}),
     "Where is Mount Fuji?"),
    ("tuple", obj({"point": {"type": "array", "prefixItems": [N, N, N], "items": False, "minItems": 3}}),
     "Give a point in 3-D space."),
    ("dictionary", obj({"scores": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0,
                                                                                "maximum": 10}}}),
     "Score four movies you know out of ten."),
    ("deep", obj({"a": obj({"b": obj({"c": obj({"d": S, "e": {"type": "array", "items": I, "maxItems": 3}})})})}),
     "Fill in a deeply nested configuration."),
    ("enum_numbers", obj({"stars": {"enum": [1, 2, 3, 4, 5]}, "recommend": B}),
     "Rate a hotel stay."),
    ("long_enum", obj({"country": {"enum": [
        "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "Chile", "China", "Colombia", "Croatia",
        "Czechia", "Denmark", "Egypt", "Finland", "France", "Germany", "Greece", "Hungary", "India", "Indonesia",
        "Ireland", "Israel", "Italy", "Japan", "Kenya", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nigeria",
        "Norway", "Peru", "Poland", "Portugal", "Romania", "Singapore", "South Africa", "South Korea", "Spain",
        "Sweden", "Switzerland", "Thailand", "Turkey", "Ukraine", "United Kingdom", "United States", "Vietnam"]}}),
     "Which country is Kyoto in?"),
    ("object_any", {"type": "object"}, "Return any JSON object describing a cat."),
    ("array_top", {"type": "array", "items": obj({"q": S, "a": S}), "minItems": 2, "maxItems": 4},
     "Write a few quiz questions with answers about the solar system."),
    ("string_top", {"type": "string", "enum": ["yes", "no", "unsure"]}, "Is Pluto a planet?"),
    ("pattern_ids", obj({"ids": {"type": "array", "minItems": 2, "maxItems": 6,
                                 "items": {"type": "string", "pattern": "^[A-Z]{2}[0-9]{4}$"}}}),
     "List a few flight numbers."),
    ("date_time", obj({"created": {"type": "string", "format": "date-time"},
                       "uuid": {"type": "string", "format": "uuid"}}),
     "Create a record with a timestamp and an identifier."),
    ("min_properties", {"type": "object", "minProperties": 2, "maxProperties": 4,
                        "additionalProperties": {"type": "string"}},
     "Translate the word 'hello' into a few languages."),
    ("invoice", {"$defs": {"line": obj({"desc": S, "amount": {"type": "number", "minimum": 0}})},
                 **obj({"invoice_no": S, "issued": {"type": "string", "format": "date"},
                        "lines": {"type": "array", "items": {"$ref": "#/$defs/line"}, "minItems": 1, "maxItems": 6},
                        "currency": {"enum": ["EUR", "USD", "GBP"]}, "paid": B,
                        "notes": {"anyOf": [S, {"type": "null"}]}})},
     "Draft an invoice for a two-day consulting job."),
    ("pydantic_like", {"$defs": {"Tag": {"enum": ["red", "green", "blue"], "title": "Tag", "type": "string"},
                                 "Item": {"additionalProperties": False, "description": "An item.",
                                          "properties": {"name": {"title": "Name", "type": "string"},
                                                         "tags": {"items": {"$ref": "#/$defs/Tag"}, "title": "Tags",
                                                                  "type": "array"}},
                                          "required": ["name", "tags"], "title": "Item", "type": "object"}},
                       "additionalProperties": False, "description": "A list of items.",
                       "properties": {"items": {"items": {"$ref": "#/$defs/Item"}, "maxItems": 5, "title": "Items",
                                                "type": "array"}},
                       "required": ["items"], "title": "Items", "type": "object"},
     "List some colored things you might find in a kitchen."),
    ("escapes", obj({"quote": S, "path": {"type": "string", "pattern": "^/[a-z/]+$"}}),
     'Give a famous quote that contains "double quotes" and a Unix path.'),
    ("bool_matrix", obj({"grid": {"type": "array", "minItems": 3, "maxItems": 3,
                                  "items": {"type": "array", "items": B, "minItems": 3, "maxItems": 3}}}),
     "Draw a tic-tac-toe board as booleans."),
    ("unicode", obj({"word": S, "language": S, "script": {"enum": ["latin", "cyrillic", "greek", "cjk", "arabic"]}}),
     "Give the word for 'peace' in Japanese, with its script."),
]


def check() -> list[str]:
    """Every corpus schema compiles with our grammar (frontend check). Returns the failures."""
    from llm_mini_infer.grammar.spec import check as llg_check
    from llm_mini_infer.grammar.spec import json_schema_grammar

    return [name for name, schema, _ in CORPUS if llg_check(json_schema_grammar(schema))]


if __name__ == "__main__":
    bad = check()
    print(f"{len(CORPUS)} schemas; {len(bad)} rejected by llguidance: {bad}")

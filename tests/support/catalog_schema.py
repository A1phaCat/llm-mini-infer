"""A schema with nested `$defs`, nullable enum lists and strict objects, as pydantic writes them, for the grammar, API
and bench tests: a made-up catalogue of the items in an archive's box. Its features shape a guidance grammar:

- the top level is `{total: integer, entries: array of Entry}`, with `maxItems: 5` on entries;
- Entry sits under `$defs` and is referenced with `$ref`;
- both objects forbid other keys (`additionalProperties: false`) and require every key (no defaults);
- Entry has an integer `id`, a `category` from a fixed set, a `year` bounded to 1800..2030, a boolean `signed`, and
  five attributes, each `anyOf: [array of enum strings, null]`, three of them with `minItems: 1`;
- the descriptions are the classes' docstrings.

`prompt(i)` is the bench kit's request to catalogue a box, with this schema (bench/structured/validity.py, bench/load/
run.py's catalog workload).

    uv run python -m tests.support.catalog_schema  # rewrites tests/data/catalog.schema.json
                                                   # (tests/repo/test_catalog_schema.py)
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .paths import DATA

NAME = "catalog"
PATH = DATA / "catalog.schema.json"

Category = Literal["book", "map", "letter", "photograph", "print", "record"]
Language = Literal["english", "french", "german", "spanish", "italian", "latin"]
Subject = Literal["history", "science", "travel", "music", "art", "law", "trade"]
Format = Literal["bound", "loose", "folded", "rolled", "framed"]
Condition = Literal["fine", "good", "fair", "poor"]
Provenance = Literal["donated", "purchased", "transferred", "found"]


class Entry(BaseModel):
    """One item in the box, with the attributes that describe it. An attribute is null when it cannot be told."""

    model_config = ConfigDict(extra="forbid")

    id: int
    category: Category
    year: int = Field(ge=1800, le=2030)
    signed: bool
    languages: list[Language] | None = Field(min_length=1)
    subjects: list[Subject] | None = Field(min_length=1)
    formats: list[Format] | None
    condition: list[Condition] | None = Field(min_length=1)
    provenance: list[Provenance] | None


class Catalog(BaseModel):
    """The items in an archive's box: how many there are, and up to five of them described, in the box's order."""

    model_config = ConfigDict(extra="forbid")

    total: int
    entries: list[Entry] = Field(max_length=5)


def schema() -> dict:
    return Catalog.model_json_schema()


def response_format(strict: bool = True) -> dict:
    """The request's `response_format` for this schema."""
    return {"type": "json_schema", "json_schema": {"name": NAME, "strict": strict, "schema": schema()}}


PLACES = ("a city archive", "a university library", "a museum's store", "a family's attic", "a law firm's basement",
          "a ship owner's office", "a parish hall", "an auction house")
BOXES = ("first", "second", "third", "last")


def prompt(i: int, numbered: bool = False) -> str:
    """The i-th request to catalogue a box: eight places in turn, a box of four in each; `numbered` names the box by
    `i`, so that no two prompts are the same (a load's requests, which a prefix cache would otherwise serve whole)."""
    box = f"box {i}" if numbered else f"the {BOXES[i % len(BOXES)]} box"
    return f"Picture {box} on a shelf of {PLACES[i % len(PLACES)]}. Catalogue the items in it, in the box's order."


# A compact document of the schema (guidance with disable_any_whitespace writes no spaces)
DOC = (b'{"total":2,"entries":[{"id":0,"category":"map","year":1887,"signed":false,"languages":["french"],'
       b'"subjects":["travel","trade"],"formats":["folded"],"condition":["fair"],"provenance":null},{"id":1,'
       b'"category":"letter","year":1921,"signed":true,"languages":["english"],"subjects":null,"formats":[],'
       b'"condition":["good"],"provenance":["donated"]}]}')


if __name__ == "__main__":
    PATH.write_text(json.dumps(schema(), indent=1) + "\n")
    print(f"{PATH.name}: {len(json.dumps(schema()))} bytes as a client sends it (json.dumps)")

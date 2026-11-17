"""The synthetic schema (tests/support/catalog_schema.py): its JSON file is the model's schema, its sample document is
valid under both the model and the schema, and its response format wraps the schema."""

from __future__ import annotations

import json

import jsonschema

from ..support.catalog_schema import DOC, NAME, PATH, Catalog, response_format, schema


def test_the_file_is_the_models_schema() -> None:
    assert json.loads(PATH.read_text()) == schema() == response_format()["json_schema"]["schema"]


def test_the_document_is_valid() -> None:
    jsonschema.Draft202012Validator(schema()).validate(json.loads(DOC))
    catalog = Catalog.model_validate_json(DOC)
    assert catalog.total == len(catalog.entries) == 2
    first, second = catalog.entries
    assert first.category == "map" and first.year == 1887 and not first.signed and first.languages == ["french"]
    assert first.subjects == ["travel", "trade"] and first.provenance is None
    assert second.category == "letter" and second.signed and second.subjects is None and second.formats == []


def test_the_response_format_wraps_the_schema() -> None:
    assert response_format() == {"type": "json_schema", "json_schema": {"name": NAME, "strict": True,
                                                                       "schema": schema()}}
    assert response_format(False)["json_schema"]["strict"] is False

"""The reference vLLM release is named in one place of the docs: the first paragraph of docs/vllm-compatibility.md,
which names `REFERENCE` (tests/support/vllm_ref.py). The other guides never name it, so a new reference is a one-place
edit. README.md and docs/benchmarks.md name the vLLM release that the benchmark measured: a measurement, not the
reference."""

from __future__ import annotations

import pytest

from ..support.paths import ROOT
from ..support.vllm_ref import REFERENCE


def test_the_compatibility_page_names_the_reference_release() -> None:
    text = (ROOT / "docs" / "vllm-compatibility.md").read_text()
    first = next(line for line in text.splitlines() if line.strip() and not line.startswith("#"))
    assert f"vLLM {REFERENCE}" in first, first
    assert text.count(f"vLLM {REFERENCE}") == 1


@pytest.mark.parametrize("doc", ["architecture.md", "operations.md", "development.md"])
def test_the_guides_do_not_name_the_reference_release(doc: str) -> None:
    assert f"vLLM {REFERENCE}" not in (ROOT / "docs" / doc).read_text()

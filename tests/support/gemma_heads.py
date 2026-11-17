"""The Gemma 4 heads tests' pair and grammars on the CPU (tests/models/test_gemma4_heads.py)."""

from __future__ import annotations

import pytest

from .fake_grammar import spec

VOCAB = 512


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    from .tiny_gemma import make_tiny_assistant, make_tiny_gemma

    d = tmp_path_factory.mktemp("g4heads")
    target = make_tiny_gemma(d / "target", window=8, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def json_like(seed: int, lo: int, hi: int) -> str:
    """A JSON grammar's shape on a small vocabulary: about 8 of 128 tokens per state, the EOS from `lo` on."""
    return spec(seed, lo, hi, one_in=16, universe=128)

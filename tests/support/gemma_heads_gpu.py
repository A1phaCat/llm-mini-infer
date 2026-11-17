"""The heads tests' shared prompts and grammars on the GPU (tests/models/test_gemma4_heads_gpu.py,
tests/kernels/test_pdl_gpu.py)."""

from __future__ import annotations

from .fake_grammar import spec
from .mtp_common import seq


def forced_json(i: int) -> str:
    """`qwen_gpu.json_like` with forced runs: its first two tokens (the forced prefix) and its last before the EOS."""
    return spec(60 + i, 6 + i % 5, 30, one_in=16, universe=128, lead=2, tail=1)


VOCAB = 512
SHARED = seq(72, 99)  # a long shared system prompt, cached, longer than the window
MANY = [SHARED + seq(3 + (i * 7) % 20, 10 + i) for i in range(24)]


def json_like(i: int) -> str:
    """About 8 of 128 tokens a state (`fake_grammar`'s sparse mode): a JSON grammar's few tokens and small universe."""
    return spec(60 + i, 6 + i % 5, 30, one_in=16, universe=128)

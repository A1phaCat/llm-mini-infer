"""The model fixtures' helpers (tests/.fixtures/, uv run tests/scripts/fetch_fixtures.py): the models, their corpora, a
skip when one is absent, and the kernel lookups a test stubs. Plain functions, not pytest fixtures; torch and the
kernels load inside the two that need them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .paths import FIXTURES, MODELS_DIR

MODELS = ["qwen3.8-27b", "qwen3-next-80b", "gemma-4-26b-a4b", "nemotron-3-nano-4b", "step-3.5-flash"]
QWEN = MODELS_DIR / "qwen3.8-27b"


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


def fixtures_path(name: str) -> Path:
    return FIXTURES / "corpus" / f"{name}.jsonl"


def load_cases(name: str) -> list[dict]:
    path = fixtures_path(name)
    if not path.exists() or not (model_dir(name) / "tokenizer.json").exists():
        pytest.skip(f"the {name} corpus is absent: uv run tests/scripts/fetch_fixtures.py")
    return [json.loads(line) for line in path.open()]


def require_qwen() -> Path:
    if not (QWEN / "tokenizer.json").exists() or not (QWEN / "config.json").exists():
        pytest.skip("tests/.fixtures/models/qwen3.8-27b is absent: uv run tests/scripts/fetch_fixtures.py")
    return QWEN


def fresh_kernel_lookup(monkeypatch) -> None:
    """The kernel libraries' lookup (llm_mini_infer.kernels.build) from scratch for one test: its memo reset (each
    library, whether it was tried, its state) and LLM_MINI_INFER_KERNELS unset."""
    from llm_mini_infer.kernels import build

    monkeypatch.delenv("LLM_MINI_INFER_KERNELS", raising=False)
    for name, value in (("_lib", None), ("_tried", False), ("stats", {}), ("_cutlass", None),
                        ("_cutlass_tried", False), ("stats_cutlass", {})):
        monkeypatch.setattr(build, name, value)


def simulated_cuda(monkeypatch) -> None:
    """torch reports one CUDA device on any host: a CUDA runner's constructor runs on the CPU up to its first device
    call past the checks (with memory_gb set, which skips the free-memory check). The switch the constructor sets
    (`ops.enabled`, from fused_ops) is restored after the test."""
    import torch

    from llm_mini_infer.kernels import ops

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(ops, "_enabled", ops.enabled())

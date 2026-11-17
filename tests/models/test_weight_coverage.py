"""The weight loaders check that every parameter's rows are written exactly once: the row ranges tile [0, rows), so a
shard loaded twice beside a missing one of the same size fails, where a count of rows would pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_mini_infer.models.common import Coverage
from llm_mini_infer.models.gemma4.assistant import build_assistant
from llm_mini_infer.models.gemma4.model import build_model as build_gemma
from llm_mini_infer.models.qwen3_5.model import build_model as build_qwen

from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma
from ..support.tiny_qwen import make_tiny_model


def params(**shapes: tuple[int, ...]) -> dict[str, torch.nn.Parameter]:
    return {k: torch.nn.Parameter(torch.zeros(s)) for k, s in shapes.items()}


def test_coverage_tiles_in_any_order() -> None:
    c = Coverage()
    c.add("w", 4, 4)
    c.add("w", 0, 4)
    c.add("w", 8, 2)
    assert c.incomplete(params(w=(10, 3))) == []


def test_coverage_names_a_gap_an_overlap_a_missing_and_an_overrun() -> None:
    c = Coverage()
    c.add("gap", 0, 2)
    c.add("gap", 3, 1)
    c.add("dup", 0, 4)
    c.add("dup", 0, 4)
    c.add("part", 0, 2)
    c.add("long", 0, 6)
    ps = params(gap=(4,), dup=(4,), part=(4,), none=(4,), long=(4,), ok=(1,))
    c.add("ok", 0, 1)
    assert c.incomplete(ps) == ["gap", "dup", "part", "none", "long"]


def test_coverage_equal_sizes_cannot_hide_an_overlap() -> None:
    c = Coverage()
    c.add("w", 0, 4)
    c.add("w", 0, 4)  # the second half twice is not the second half
    assert c.incomplete(params(w=(8,))) == ["w"]


def test_coverage_of_a_zero_dimensional_parameter() -> None:
    c = Coverage()
    c.add("s", 0, 1)
    assert c.incomplete({"s": torch.nn.Parameter(torch.zeros(()))}) == []
    c.add("s", 0, 1)
    assert c.incomplete({"s": torch.nn.Parameter(torch.zeros(()))}) == ["s"]


def drop_and_repeat(model_dir: Path, drop: str, repeat: str) -> None:
    """The checkpoint without tensor `drop` and with `repeat` also in a second file: the same count of rows when the
    two have the same size."""
    from safetensors.torch import load_file, save_file

    f = model_dir / "model.safetensors"
    sd = load_file(f)
    keys = list(sd)
    drop_key = next(k for k in keys if k.endswith(drop))
    repeat_key = next(k for k in keys if k.endswith(repeat))
    assert sd[drop_key].shape == sd[repeat_key].shape
    save_file({k: v for k, v in sd.items() if k != drop_key}, f, metadata={"format": "pt"})
    save_file({repeat_key: sd[repeat_key]}, model_dir / "model-2.safetensors", metadata={"format": "pt"})


def test_gemma_text_loader_refuses_a_repeated_shard_for_a_missing_one(tmp_path) -> None:
    make_tiny_gemma(tmp_path)
    build_gemma(tmp_path, torch.device("cpu"), torch.float32)
    drop_and_repeat(tmp_path, "layers.0.mlp.up_proj.weight", "layers.0.mlp.gate_proj.weight")
    with pytest.raises(ValueError, match=r"weights missing or partial in the checkpoint: .*layers.0.mlp.gate_up_proj"):
        build_gemma(tmp_path, torch.device("cpu"), torch.float32)


def test_qwen_loader_refuses_a_repeated_shard_for_a_missing_one(tmp_path) -> None:
    make_tiny_model(tmp_path)
    build_qwen(tmp_path, torch.device("cpu"), torch.float32)
    drop_and_repeat(tmp_path, "layers.0.mlp.up_proj.weight", "layers.0.mlp.gate_proj.weight")
    with pytest.raises(ValueError, match=r"weights missing or partial in the checkpoint: .*layers.0.mlp.gate_up_proj"):
        build_qwen(tmp_path, torch.device("cpu"), torch.float32)


def test_assistant_loader_refuses_a_repeated_half_for_a_missing_one(tmp_path) -> None:
    from llm_mini_infer.models.gemma4.model import Gemma4Config

    target = make_tiny_gemma(tmp_path / "target")
    asst = make_tiny_assistant(tmp_path / "assistant", target)
    cfg = Gemma4Config.from_dir(target)
    build_assistant(asst, cfg, torch.device("cpu"), torch.float32)
    drop_and_repeat(asst, "layers.0.mlp.up_proj.weight", "layers.0.mlp.gate_proj.weight")
    with pytest.raises(ValueError, match=r"assistant weights missing in the checkpoint: .*layers.0.mlp.gate_up_proj"):
        build_assistant(asst, cfg, torch.device("cpu"), torch.float32)


def test_gemma_vision_loader_refuses_a_repeated_shard_for_a_missing_one(tmp_path) -> None:
    from llm_mini_infer.models.gemma4.vision import load_gemma_vision

    from ..support.tiny_gemma import make_tiny_gemma_mm

    make_tiny_gemma_mm(tmp_path)
    load_gemma_vision(tmp_path, torch.device("cpu"), torch.float32)
    drop_and_repeat(tmp_path, "vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
                    "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight")
    with pytest.raises(ValueError, match=r"vision weights missing or partial in the checkpoint: .*layers.0.gate_up"):
        load_gemma_vision(tmp_path, torch.device("cpu"), torch.float32)

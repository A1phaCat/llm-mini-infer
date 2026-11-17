"""The RoPE tables' digests, run in a child process too (tests/models/test_rope_tables.py)."""

from __future__ import annotations

import hashlib

import torch


def words(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(w & 0xFFFFFFFF for w in t.contiguous().view(torch.int32).tolist())


def qwen_table() -> torch.Tensor:
    """Qwen3.8's vision table (no tracked config: its vision_config's numbers)."""
    from llm_mini_infer.models.qwen3_5.vision import VisionConfig, vision_buffers

    cfg = VisionConfig(depth=0, hidden=1152, heads=16, intermediate=4304, patch=16, temporal=2, merge=2,
                       out_hidden=5120, num_pos=2304, in_channels=3, act="gelu_pytorch_tanh", rope_theta=10000.0,
                       mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255)
    return vision_buffers(cfg)["inv_freq"]


def tables(model_dir: str) -> dict:
    """The capability ATen ran, both towers' words, and the sha256 of Gemma 4's text tables, for the checkpoint config
    in `model_dir`."""
    from llm_mini_infer.models.gemma4.model import FULL, SLIDING, Gemma4Config, inv_freq
    from llm_mini_infer.models.gemma4.vision import GemmaVisionConfig, vision_buffers

    text = Gemma4Config.from_dir(model_dir)
    sha = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()  # noqa: E731
    return {"ran": torch.backends.cpu.get_cpu_capability(),
            "gemma": words(vision_buffers(GemmaVisionConfig.from_dir(model_dir))["inv_freq"]),
            "qwen": words(qwen_table()), "sliding": sha(inv_freq(text, SLIDING)), "full": sha(inv_freq(text, FULL))}

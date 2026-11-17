"""The vision tower against transformers' (tests/vision/test_vision_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

VISION = {"depth": 2, "hidden_size": 64, "num_heads": 4, "intermediate_size": 96, "patch_size": 16,
          "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 48, "num_position_embeddings": 64,
          "in_channels": 3, "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []}


def make_tower(path: Path, seed: int = 0) -> torch.nn.Module:
    """A random HF vision tower saved as `model.visual.*` with a config.json and preprocessor_config.json."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.manual_seed(seed)
    cfg = Qwen3_5VisionConfig(**{k: v for k, v in VISION.items() if k != "deepstack_visual_indexes"})
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    path.mkdir(parents=True, exist_ok=True)
    save_file({f"model.visual.{k}": v.contiguous() for k, v in model.state_dict().items()},
              path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "vision_config": VISION,
                                                  "text_config": {"model_type": "qwen3_5_text"}}))
    (path / "preprocessor_config.json").write_text(json.dumps({"patch_size": 16, "temporal_patch_size": 2,
                                                               "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
                                                               "image_std": [0.5, 0.5, 0.5]}))
    return model


def hf_patches(px: torch.Tensor) -> torch.Tensor:
    """The reference for `patches`: HF's fused rescale and normalisation (mean and std times 255), then its patchify."""
    x = px.unsqueeze(0).float()
    mean = torch.tensor([0.5] * 3) * 255.0
    std = torch.tensor([0.5] * 3) * 255.0
    x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    p, m, t = 16, 2, 2
    b, c, hh, ww = x.shape
    gh, gw = hh // p, ww // p
    x = x.reshape(b, c, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(b, gh * gw, c * t * p * p)
    return x[0]


def images(seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (3, 16 * h, 16 * w), generator=g, dtype=torch.uint8) for h, w in ((4, 6), (8, 2))]

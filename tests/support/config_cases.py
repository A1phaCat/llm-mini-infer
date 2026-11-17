"""Model configurations the configuration tests write (tests/api/test_config.py)."""

from __future__ import annotations

import json
from pathlib import Path


def _model(tmp: Path, vocab: int, mtp: int) -> Path:
    tc = {"model_type": "qwen3_5_text", "vocab_size": vocab, "mtp_num_hidden_layers": mtp,
          "max_position_embeddings": 4096, "eos_token_id": vocab - 1}
    (tmp / "config.json").write_text(json.dumps({"text_config": tc}))
    return tmp


def _gemma(tmp: Path, positions: int = 4096) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    tc = {"model_type": "gemma4_text", "vocab_size": 262_144, "max_position_embeddings": positions, "eos_token_id": 1}
    (tmp / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": tc}))
    return tmp


def _assistant(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant", "backbone_hidden_size": 5376,
                                                 "text_config": {"model_type": "gemma4_text"}}))
    return tmp

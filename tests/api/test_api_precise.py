"""The precise prefix end to end: the real server against the strict fake runner taking PRECISE rows as a block-FP8
model's runner does (`dev.runner.fake.precise_rows`), with the Qwen3.8 tokenizer and chat template. A refine's piece
(`dev.cache.refine_chunk_tokens`) goes from the configuration to the scheduler, and its counters reach /metrics. The
derived piece for Gemma 4 with its assistant too (the assistant's largest k, whatever k is set)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from ..support.api_cache_cases import KW, SYSTEM
from ..support.api_util import Server, env_has_model
from ..support.fixtures import require_qwen
from ..support.paths import MODELS_DIR

GEMMA = MODELS_DIR / "gemma-4-31b-it"


def start(tmp: Path, cache: dict) -> Server:
    return Server(require_qwen(), tmp,
                  extra={"runner": {"kind": "fake"},
                         "dev": {"runner": {"fake": {"step_ms": 1.0, "strict": True, "precise_rows": True}},
                                 "cache": cache}}).start()


def shared_prefix(srv: Server, n: int = 5) -> list[int]:
    """`n` requests on the shared system prompt, one at a time: their cached tokens."""
    hits = []
    for i in range(n):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Item {i}: a blue chair"}]
        status, r = srv.post("/v1/chat/completions", {"messages": msgs, "chat_template_kwargs": KW, "max_tokens": 4,
                                                      "seed": 300 + i, "ignore_eos": True})
        assert status == 200, r
        hits.append(r["usage"]["prompt_tokens_details"]["cached_tokens"])
    return hits


def test_fixed_pieces_next_to_a_decoding_request(tmp_path) -> None:
    """Fixed pieces of 256 tokens while a long request decodes: the third request on the shared system prompt refines
    the prefix the first one's checkpoint holds, one PRECISE step per piece (steps the decoding request shares), and
    the later ones resume from the precise copy."""
    srv = start(tmp_path, {"refine_chunk_tokens": 256})
    try:
        cfg = json.loads(srv.get("/server_info")[1])["config"]["dev"]["cache"]
        assert cfg["refine_chunk_tokens"] == 256 and not {"refine_chunking", "refine_steps"} & set(cfg)
        done: list[int] = []
        body = {"messages": [{"role": "user", "content": "Describe a long walk."}], "chat_template_kwargs": KW,
                "max_tokens": 1500, "seed": 7, "ignore_eos": True}
        long = threading.Thread(target=lambda: done.append(srv.post("/v1/chat/completions", body)[0]))
        long.start()
        deadline = time.monotonic() + 10
        while srv.metric("vllm:num_requests_running") < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        hits = shared_prefix(srv)
        assert long.is_alive(), "the long request finished before the refine"
        long.join(60)
        assert done == [200]
        r = hits[1]
        assert hits == [0, r, 0, r, r] and r > 256, hits
        pieces = -(-r // 256)
        assert srv.metric("llm_mini_infer:prefix_cache_refines_total") == 1
        assert srv.metric("llm_mini_infer:prefix_cache_precise_steps_total") == pieces
        assert b"llm_mini_infer:prefix_cache_exclusive_steps_total" not in srv.get("/metrics")[1]
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_the_derived_piece(tmp_path) -> None:
    """The piece derived (64 sequences, no MTP: 2,048 - 64 on the 64-token grid): the engine reports it with its
    reason, and a refine of the system prompt takes one PRECISE step."""
    srv = start(tmp_path, {})
    try:
        info = json.loads(srv.get("/server_info")[1])
        assert info["config"]["dev"]["cache"]["refine_chunk_tokens"] == 1984
        assert info["reasons"]["dev.cache.refine_chunk_tokens"] == (
            "1,984: min(2,048, max_num_batched_tokens 2,048 - 64 rows x 1) on the 64-token grid: a piece fits next to "
            "a full batch of decode rows")
        hits = shared_prefix(srv)
        assert hits == [0, hits[1], 0, hits[1], hits[1]] and 0 < hits[1] < 1984, hits
        assert srv.metric("llm_mini_infer:prefix_cache_refines_total") == 1
        assert srv.metric("llm_mini_infer:prefix_cache_precise_steps_total") == 1
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_the_derived_piece_with_an_assistant(tmp_path) -> None:
    """Gemma 4 with its assistant at k = 3: the piece leaves room for decode rows of the assistant's largest k, 7, so it
    is 1,536 (2,048 - 64 rows x 8 on the 16-token window grid) at this k as at llm-mini-infer's default 5 or any other;
    derived from k = 3 itself, it would be 1,792."""
    if not env_has_model(GEMMA):
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it (configs and tokenizer) is absent: "
                    "uv run tests/scripts/fetch_fixtures.py")
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "config.json").write_text(json.dumps({"model_type": "gemma4_assistant",
                                                                    "text_config": {"model_type": "gemma4_text"}}))
    srv = Server(GEMMA, tmp_path, mtp=3,
                 extra={"runner": {"kind": "fake"},
                        "dev": {"runner": {"fake": {"step_ms": 1.0, "strict": True, "precise_rows": True}}},
                        "speculative": {"model": str(tmp_path / "assistant")}, "images": {"enabled": False}}
                 ).start(120)
    try:
        info = json.loads(srv.get("/server_info")[1])
        assert info["config"]["speculative"]["num_speculative_tokens"] == 3 and info["config"]["mtp_k_max"] == 7
        assert info["config"]["dev"]["cache"]["refine_chunk_tokens"] == 1536
        assert info["reasons"]["dev.cache.refine_chunk_tokens"] == (
            "1,536: min(2,048, max_num_batched_tokens 2,048 - 64 rows x 8 (1 + the assistant's largest k, 7)) on the "
            "16-token grid: a piece fits next to a full batch of decode rows at any k, so k never changes the precise "
            "copy")
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]

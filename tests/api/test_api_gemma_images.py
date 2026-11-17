"""Gemma 4's images through the real server (launcher, image pool, engine, frontend) against the fake runner, with
google/gemma-4-31B-it's own config, processor config, tokenizer and chat template (no weights).

`/tokenize` of a chat with images must give exactly the ids of transformers' `Gemma4Processor` (its chat template,
then `<|image>`, the soft tokens and `<image|>` per image, the count from the image's size), at the checkpoint's
budget and at images.max_soft_tokens 70; a chat completion runs through the scheduler's bidirectional spans and the
fake's checks of them (every span whole in one row). Image token ids in config.json that are not the tokenizer's fail
the start.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil

import numpy as np
import pytest
from PIL import Image

from ..support.api_util import Server, env_has_model
from ..support.paths import DATA, MODELS_DIR

GEMMA = MODELS_DIR / "gemma-4-31b-it"
SOFT = 258880  # the soft token, one per image token (`<|image|>`)
# serial: a Gemma server with two pool workers is 2.5 GB
pytestmark = [pytest.mark.skipif(not env_has_model(GEMMA) or not (GEMMA / "processor_config.json").exists(),
                                 reason="tests/.fixtures/models/gemma-4-31b-it (configs and tokenizer) is absent: "
                                        "uv run tests/scripts/fetch_fixtures.py"),
              pytest.mark.serial]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = Server(GEMMA, tmp_path_factory.mktemp("srv"),
                 extra={"images": {"workers": 2, "max_per_request": 3}}).start(120)
    yield srv
    srv.stop()


@pytest.fixture
def server70(tmp_path):
    """images.max_soft_tokens 70: the frontend, the engine and the pool load the family at 70, or it would not start.
    Its one test's own, and that test is the file's first: the two servers together are 5 GB of CI's 7.8."""
    srv = Server(GEMMA, tmp_path, extra={"images": {"workers": 2, "max_per_request": 3, "max_soft_tokens": 70}})
    try:
        yield srv.start(120)
    finally:
        srv.stop()


def jpeg(w: int, h: int, seed: int = 0) -> bytes:
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def hf_ids(images: list[bytes], text: str, system: str, max_soft_tokens: int | None = None) -> list[int]:
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(GEMMA, **({"max_soft_tokens": max_soft_tokens} if max_soft_tokens else {}))
    parts = [{"type": "image", "image": Image.open(io.BytesIO(d))} for d in images]
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [*parts, {"type": "text", "text": text}]}]
    out = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="np")
    return out["input_ids"][0].tolist()


def chat(images: list[bytes], text: str, system: str) -> list[dict]:
    parts = [{"type": "image_url", "image_url": {"url": uri(d)}} for d in images]
    return [{"role": "system", "content": system}, {"role": "user", "content": [*parts, {"type": "text",
                                                                                         "text": text}]}]


def test_tokenize_and_complete_at_70_soft_tokens(server70) -> None:
    """At 70, a 400 x 225 and a 400 x 711 image take 66 soft tokens each (264
    at 280): /tokenize gives the ids of HF's processor at 70, a chat completion counts the same prompt, and
    /server_info reports the setting."""
    jpegs = [jpeg(400, 225, 1), jpeg(400, 711, 2)]
    want = hf_ids(jpegs, "What is it?", "You classify products.", max_soft_tokens=70)
    status, body = server70.post("/tokenize", {"messages": chat(jpegs, "What is it?", "You classify products.")})
    assert status == 200, body
    assert body["tokens"] == want and want.count(SOFT) == 132
    assert len(hf_ids(jpegs, "What is it?", "You classify products.")) - len(want) == 2 * (264 - 66)
    status, body = server70.post("/v1/chat/completions", {"model": "qwen-test", "max_tokens": 4, "temperature": 0,
                                                          "ignore_eos": True,
                                                          "messages": chat(jpegs[:1], "What is it?", "Be brief.")})
    assert status == 200, body
    assert body["usage"]["prompt_tokens"] == len(hf_ids(jpegs[:1], "What is it?", "Be brief.", max_soft_tokens=70))
    status, raw = server70.get("/server_info")
    cfg = json.loads(raw)["config"]
    assert status == 200 and cfg["images"]["max_soft_tokens"] == 70 and cfg["image_tokens_max"] == 70
    # server-wide only: a per-request `mm_processor_kwargs` is an unknown field
    status, body = server70.post("/v1/chat/completions", {"model": "qwen-test", "max_tokens": 1,
                                                          "mm_processor_kwargs": {"max_soft_tokens": 280},
                                                          "messages": chat(jpegs[:1], "What is it?", "Be brief.")})
    assert status == 400 and "mm_processor_kwargs" in body["error"]["message"], body


@pytest.mark.parametrize("sizes", [[(400, 225)], [(225, 400), (1000, 333)], [(64, 48), (3000, 40)]])
def test_tokenize_matches_the_processor(server, sizes) -> None:
    images = [jpeg(w, h, i) for i, (w, h) in enumerate(sizes)]
    status, body = server.post("/tokenize", {"messages": chat(images, "What is it?", "You classify products.")})
    assert status == 200, body
    assert body["tokens"] == hf_ids(images, "What is it?", "You classify products.")


def test_a_chat_with_two_images_completes(server) -> None:
    """The spans are bidirectional: the scheduler keeps each whole in one row and the fake runner checks it; the
    request finishes by length with every token."""
    images = [jpeg(400, 225, 5), jpeg(300, 300, 6)]
    encoded = step_images(server)
    status, body = server.post("/v1/chat/completions", {"model": "qwen-test", "max_tokens": 12, "temperature": 0,
                                                        "ignore_eos": True,
                                                        "messages": chat(images, "Compare them.", "Be brief.")})
    assert status == 200, body
    assert body["usage"]["completion_tokens"] == 12 and body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == len(hf_ids(images, "Compare them.", "Be brief."))
    # the per-step histogram of encoded images (llm_mini_infer:step_encoder_images): both images, each encoded once
    assert step_images(server) - encoded == 2


def step_images(server) -> float:
    """Images encoded so far by the steps' count: llm_mini_infer:step_encoder_images' sum (absent before any step)."""
    _, text = server.get("/metrics")
    m = re.search(r'^llm_mini_infer:step_encoder_images_sum\{[^}]*\} (\S+)$', text.decode(), re.M)
    return float(m.group(1)) if m else 0.0


def pool_jobs(server) -> int:
    """The jobs the image pool answered this frontend: its pool stage's count in /metrics."""
    _, text = server.get("/metrics")
    m = re.search(r'^llm_mini_infer:image_stage_seconds_count\{[^}]*stage="pool"[^}]*\} (\S+)$', text.decode(), re.M)
    return int(float(m.group(1))) if m else 0


def test_a_poison_header_is_a_400_before_the_pool(server) -> None:
    """A poison image: a JPEG header declaring 65,535 x 65,535 pixels (12.9 GB of RGB in 17 bytes) is a 400 at
    the frontend's probe, for a chat and /tokenize, worded like vLLM's limit error with our setting's name. The
    pool never gets the job (its count stays), every worker lives, and the next request is served."""
    from ..support.vision_pool import POISON

    before = pool_jobs(server)
    pids = json.loads(server.get("/debug/images")[1])["pool_pids"]
    messages = chat([POISON], "What is it?", "Be brief.")
    status, body = server.post("/v1/chat/completions", {"model": "qwen-test", "max_tokens": 2, "messages": messages})
    assert status == 400 and body["error"]["type"] == "BadRequestError", body
    assert "exceed the maximum of 178956970 pixels (images.max_source_pixels)" in body["error"]["message"], body
    assert pool_jobs(server) == before
    status, body = server.post("/tokenize", {"messages": messages})
    assert status == 400, body
    assert pids
    for pid in pids:
        os.kill(pid, 0)  # alive
    messages = chat([jpeg(400, 225)], "What is it?", "Be brief.")
    status, body = server.post("/v1/chat/completions", {"model": "qwen-test", "max_tokens": 2, "messages": messages})
    assert status == 200, body


def test_image_token_ids_that_are_not_the_tokenizers_fail_the_start(tmp_path) -> None:
    """The frontend checks config.json's image token ids against the tokenizer before it serves (vision.gemma): the
    checkpoint's tracked configs and the fixture's tokenizer, with image_token_id shifted by one, stop the start; the
    frontend's error names the token and both ids, and the launcher's the frontend that exited."""
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "generation_config.json", "processor_config.json", "tokenizer_config.json"):
        shutil.copy(DATA / f"gemma-4-31b-it.{name}", model / name)
    for name in ("tokenizer.json", "chat_template.jinja"):
        shutil.copy(GEMMA / name, model / name)
    hf = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps({**hf, "image_token_id": hf["image_token_id"] + 1}))
    srv = Server(model, tmp_path, extra={"images": {"workers": 1}})
    try:
        with pytest.raises(RuntimeError, match="server exited"):
            srv.start(120)
    finally:
        code = srv.stop()
    log = srv.log_path.read_text()
    assert code == 1 and "frontend 0 exited during startup" in log, log[-3000:]
    assert "image_token_id is 258881 in config.json, but the tokenizer's <|image|> is 258880" in log, log[-3000:]

"""Images through the real server (launcher, image pool, engine, frontend) against the fake runner.

The fake samples from the prompt's radix key, in which each image span holds its image key (blake3 of the bytes and the
resize target, as the frontend computes it), and from the request's M-RoPE offset: the streamed text is predicted from
the expected prompt, image and offset, so a wrong placeholder count, a wrong key or a wrong offset changes it.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import threading
import time

import blake3
import numpy as np
import pytest
from PIL import Image

from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.runner.fake import fake_generate
from llm_mini_infer.vision.family import Span, span_key
from llm_mini_infer.vision.probe import probe
from llm_mini_infer.vision.qwen import QwenVision

from ..support.api_util import Predictor, Server, content_of, env_has_model
from ..support.fixtures import QWEN

# serial: servers with an image pool (2 GB with two workers), up to three alive in the URL test
pytestmark = [pytest.mark.skipif(not env_has_model(QWEN), reason="tests/.fixtures/models/qwen3.8-27b is absent: "
                                                                 "uv run tests/scripts/fetch_fixtures.py"),
              pytest.mark.serial]
PIXELS = {"min_pixels": 65536, "max_pixels": 327680}
KW = {"enable_thinking": False}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = Server(QWEN, tmp_path_factory.mktemp("srv"),  # thinking off: the answers are content
                 extra={"images": {**PIXELS, "workers": 2, "max_per_request": 3},
                        "model": {"chat_template_kwargs": KW}}).start(120)
    yield srv
    srv.stop()


def jpeg(w: int, h: int, color: tuple[int, int, int] = (200, 30, 40), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.full((h, w, 3), color, np.uint8)
    arr[rng.integers(0, h, 50), rng.integers(0, w, 50)] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def uri(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def chat(images: list[bytes], text: str = "What is it?", system: str = "You classify products.") -> list[dict]:
    parts = [{"type": "image_url", "image_url": {"url": uri(d)}} for d in images]
    return [{"role": "system", "content": system},
            {"role": "user", "content": [*parts, {"type": "text", "text": text}]}]


class ImagePredictor(Predictor):
    """The prompt with each image's span expanded, its radix key (span keys) and M-RoPE offset."""

    def __init__(self):
        super().__init__(QWEN)
        hf = json.loads((QWEN / "config.json").read_text())
        self.family = QwenVision.from_model_dir(QWEN, hf, **PIXELS)

    def expand(self, messages: list[dict], images: list[bytes], **kw) -> tuple[list[int], list[int], int]:
        conv = to_template_messages(messages, self.r.content_format, self.r.supports_developer_role,
                                    self.family.placeholder)
        ids = self.r.encode(self.r.render(conv, **{**KW, **kw})).tolist()
        out, key, spans = [], [], []
        it = iter(images)
        for t in ids:
            if t != self.family.pad_id:
                out.append(t)
                key.append(t)
                continue
            data = next(it)
            size = self.family.size(probe(data))
            h = blake3.blake3(data)
            h.update(struct.pack("<II", size.height, size.width))
            spans.append(Span(len(out), size))
            out += [t] * size.tokens
            key += [span_key(h.digest())] * size.tokens
        _, off = self.family.positions(len(out), spans)
        return out, key, off


@pytest.fixture(scope="module")
def pred():
    return ImagePredictor()


def test_stream_with_an_image_matches_the_prediction(server, pred) -> None:
    img = jpeg(400, 225)
    msgs = chat([img])
    ids, key, off = pred.expand(msgs, [img])
    assert ids.count(pred.family.pad_id) == 84 and off != 0  # 84 tokens for 400x225 under 65,536-327,680 pixels
    events = server.stream({"messages": msgs, "max_tokens": 12, "temperature": 0, "seed": 5, "ignore_eos": True,
                            "stream_options": {"include_usage": True}})
    toks = fake_generate(key, 5, 12, pred.vocab, pred.eos_ids, rope_off=off)
    assert content_of(events) == pred.text(ids, toks)[0]
    usage = [e for e in events if isinstance(e, dict) and e.get("usage")][-1]["usage"]
    assert usage["prompt_tokens"] == len(ids)


def test_tokenize_expands_the_placeholders(server, pred) -> None:
    imgs = [jpeg(400, 225, seed=1), jpeg(400, 711, seed=2)]
    msgs = chat(imgs)
    status, body = server.post("/tokenize", {"messages": msgs})
    ids, _, _ = pred.expand(msgs, imgs)
    assert status == 200 and body["tokens"] == ids and ids.count(pred.family.pad_id) == 84 + 264


def test_tokenize_refuses_token_strs_with_images_before_fetching_them(server) -> None:
    """'return_token_strs' with an image is a 400 before the image is fetched: an unreachable URL gives that 400, not
    a fetch error."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe it."},
                                         {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/a.png"}}]}]
    status, body = server.post("/tokenize", {"messages": msgs, "return_token_strs": True})
    assert status == 400 and body["error"]["param"] == "return_token_strs", body


def test_two_images_and_the_same_image_again_hit_the_cache(server, pred) -> None:
    """Two images in one message: predicted exactly. The same request again resumes from the cache past the images;
    the same text with another image does not hit past the first image's start."""
    imgs = [jpeg(320, 240, (10, 200, 10), seed=3), jpeg(240, 320, (10, 10, 200), seed=4)]
    msgs = chat(imgs, text="Compare them.")
    ids, key, off = pred.expand(msgs, imgs)
    body = {"messages": msgs, "max_tokens": 6, "temperature": 0, "seed": 9, "ignore_eos": True}
    first = server.post("/v1/chat/completions", body)
    second = server.post("/v1/chat/completions", body)
    want = pred.text(ids, fake_generate(key, 9, 6, pred.vocab, pred.eos_ids, rope_off=off))[0]
    for status, out in (first, second):
        assert status == 200 and out["choices"][0]["message"]["content"] == want
    start = ids.index(pred.family.pad_id)
    assert second[1]["usage"]["prompt_tokens_details"]["cached_tokens"] > start + 84
    other = chat([jpeg(320, 240, (10, 200, 11), seed=5), imgs[1]], text="Compare them.")
    status, out = server.post("/v1/chat/completions", {**body, "messages": other})
    assert status == 200 and out["usage"]["prompt_tokens_details"]["cached_tokens"] <= start


def test_colliding_span_keys_never_share_the_caches(tmp_path, pred) -> None:
    """dev.images.span_key_bits 0, a test hook: every image's span key is the same value (a forced collision of the 62
    bits). A second image after the first, in the same text, hits no further than its span's start (the radix tree
    checks the whole digest) and is encoded again (the encoder cache keys on the digest too); then each image hits
    inside its span or past it (the grid's checkpoint), on its own path."""
    srv = Server(QWEN, tmp_path, extra={"images": {**PIXELS, "workers": 1},
                                        "dev": {"images": {"span_key_bits": 0}}}).start(120)
    try:
        a, b = jpeg(320, 240, (10, 200, 10), seed=7), jpeg(320, 240, (10, 200, 10), seed=8)
        ids, _, _ = pred.expand(chat([a], text="Name it."), [a])
        start = ids.index(pred.family.pad_id)

        def cached(img: bytes) -> int:
            status, out = srv.post("/v1/chat/completions", {"messages": chat([img], text="Name it."), "max_tokens": 6,
                                                            "temperature": 0, "seed": 9, "ignore_eos": True})
            assert status == 200, out
            return out["usage"]["prompt_tokens_details"]["cached_tokens"]

        assert cached(a) == 0 and cached(b) <= start
        assert cached(a) > start and cached(b) > start
        # each image encoded once; the later hits inside the spans read each image's own embeddings
        assert srv.metric("llm_mini_infer:encoder_cache_misses_total") == 2
        assert srv.metric("llm_mini_infer:encoder_cache_hits_total") == 2
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


@pytest.mark.parametrize(("messages", "match"), [
    (chat([b"not an image"]), "unsupported image format"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}],
     "base64"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}]}], "data URIs"),
    ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/y.png"}}]}],
     "images.allow_urls"),
    (chat([jpeg(2010, 10)]), "aspect ratio"),
    (chat([jpeg(64, 64)] * 4), "At most 3 image(s) may be provided in one prompt"),
    ([{"role": "system", "content": [{"type": "image_url", "image_url": {"url": uri(jpeg(64, 64))}}]},
      {"role": "user", "content": "hi"}], "System message cannot contain images"),
    ([{"role": "user", "content": "a placeholder <|vision_start|><|image_pad|><|vision_end|> in text"}],
     "image placeholders"),
    ([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]}],
     "not supported"),
], ids=["format", "base64", "scheme", "url-off", "aspect", "too-many", "system", "pad-in-text", "audio"])
def test_image_errors_are_400s(server, messages, match) -> None:
    status, body = server.post("/v1/chat/completions", {"messages": messages, "max_tokens": 2})
    assert status == 400, body
    assert match in body["error"]["message"], body


def test_pixels_are_released_after_finishes_and_disconnects(server) -> None:
    """Every request's pixels go back to the frontend's arena partition once the engine releases the request: after
    a finish, after a client disconnect mid-stream (an abort with steps in flight)."""
    img = jpeg(400, 225, seed=7)
    for _ in range(3):
        server.post("/v1/chat/completions", {"messages": chat([img]), "max_tokens": 3})
    c = server.conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": chat([jpeg(400, 225, seed=8)]),
                                                                "max_tokens": 4000, "stream": True,
                                                                "ignore_eos": True}),
              headers={"content-type": "application/json"})
    r = c.getresponse()
    r.readline()
    c.close()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = json.loads(server.get("/debug/images")[1])
        if state["arena_used"] == 0 and state["owned"] == 0:
            break
        time.sleep(0.05)
    assert state["arena_used"] == 0 and state["owned"] == 0, state
    assert server.metric("llm_mini_infer:images_total") >= 4


class LocalImageSite:
    """A local HTTP server on 127.0.0.1 (its own thread and loop) serving one image at /img and counting the hits."""

    def __init__(self, data: bytes):
        from aiohttp import web

        self.hits = 0

        async def img(_):
            self.hits += 1
            return web.Response(body=data, content_type="image/jpeg")

        self.app = web.Application()
        self.app.add_routes([web.get("/img", img)])
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        import asyncio

        from aiohttp import web

        async def main() -> None:
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]
            self.stop = asyncio.Event()
            self.loop = asyncio.get_running_loop()
            self.ready.set()
            await self.stop.wait()
            await runner.cleanup()

        asyncio.run(main())

    def __enter__(self) -> LocalImageSite:
        self.thread.start()
        assert self.ready.wait(10)
        return self

    def __exit__(self, *exc) -> None:
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/img"


def url_chat(url: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": "What is it?"}]}]


def test_an_image_url_on_the_servers_own_network_is_refused_unless_its_network_is_allowed(tmp_path) -> None:
    """With images.allow_urls on, a URL to a local server is a 400 ('image: ...') and the server sees no request; with
    images.allowed_networks naming loopback, the same request is a 200."""
    with LocalImageSite(jpeg(320, 240)) as site:
        for networks, want in (([], 400), (["127.0.0.0/8"], 200)):
            (tmp_path / str(want)).mkdir()
            srv = Server(QWEN, tmp_path / str(want), extra={"images": {**PIXELS, "workers": 1, "allow_urls": True,
                                                                       "allowed_networks": networks},
                                                          "model": {"chat_template_kwargs": KW}}).start(120)
            try:
                status, body = srv.post("/v1/chat/completions", {"messages": url_chat(site.url), "max_tokens": 2})
                assert status == want, body
                if want == 400:
                    assert body["error"]["message"].startswith("image: ") and "127.0.0.1" not in str(body), body
                    assert site.hits == 0
                else:
                    assert site.hits == 1
            finally:
                assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_the_encoder_cache_is_keyed_by_cache_salt(server) -> None:
    """The same image sent with cache_salt a, b, a is encoded twice (a and b do not share) and read once (the second
    a)."""
    img = jpeg(320, 240, (90, 20, 160), seed=41)
    misses, hits = (server.metric(f"llm_mini_infer:encoder_cache_{k}_total") for k in ("misses", "hits"))
    for salt in ("a", "b", "a"):
        status, out = server.post("/v1/chat/completions", {"messages": chat([img], text="Salted."), "max_tokens": 2,
                                                           "cache_salt": salt})
        assert status == 200, out
    assert server.metric("llm_mini_infer:encoder_cache_misses_total") - misses == 2
    assert server.metric("llm_mini_infer:encoder_cache_hits_total") - hits == 1

"""vLLM compatibility: what a client written for vLLM expects of the engine, through the real server (launcher, engine,
frontend) on the fake runner, with the image's serving settings (deploy/config.yaml: its seed, guidance with compact
JSON) and settings small enough to test the limits at (a 4,096-token context, Gemma 4's smallest image budget, 70 soft
tokens, one image per request, a keep-alive of 1 s), serving Gemma 4 31B: google/gemma-4-31B-it's configuration,
tokenizer and chat template (no weights; the FP8 repo's tokenizer and generation configuration are the same files, its
chat template too but for a final newline Jinja drops, and its config.json adds a quantization_config, which nothing on
the fake runner uses), served under `RedHatAI/gemma-4-31B-it-FP8-block`, MTP from an assistant (a config is all the fake
reads of one), and the API key from `VLLM_API_KEY`.

- the API key on every path but `/health`, `/ready`, `/live` and `/metrics`; OPTIONS reaches no handler; a 401
  that never quotes the key;
- the engine marker: `system_fingerprint` and `x-llm-engine` on every response, streamed or not;
- a structured request with one image as a client sends it with the OpenAI SDK (`openai.OpenAI`, no retries): valid
  JSON in `content` that the catalog schema's pydantic model parses, nothing in `reasoning`;
- vLLM's context limits (a 400 on the text's tokens plus `max_tokens`, a clamp once images are expanded) and its
  one-image limit;
- keep-alive: an idle connection closes after `server.keep_alive_s`; during a drain every response says
  `Connection: close`;
- a GPU shared with another process: refuse to start when the share is not free.

The others are in docs/vllm-compatibility.md.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import signal
import socket
import time

import jsonschema
import numpy as np
import pytest
import yaml
from PIL import Image

from llm_mini_infer.api.app import UNCOUNTED
from llm_mini_infer.api.auth import ApiKeys, client_headers

from ..support.api_util import Server, env_has_model
from ..support.catalog_schema import NAME as SCHEMA_NAME
from ..support.catalog_schema import Catalog, schema
from ..support.paths import MODELS_DIR, ROOT

GEMMA = MODELS_DIR / "gemma-4-31b-it"
# serial: Gemma servers with an image pool (2.2 GB each), two alive in the seeded test
pytestmark = [pytest.mark.skipif(not env_has_model(GEMMA) or not (GEMMA / "processor_config.json").exists(),
                                 reason="tests/.fixtures/models/gemma-4-31b-it (configs and tokenizer) is absent: "
                                        "uv run tests/scripts/fetch_fixtures.py"),
              pytest.mark.serial]

KEY = "sk-test-" + "k" * 24  # made up; checked never to reach a log line, an error or /server_info
NAME = "RedHatAI/gemma-4-31B-it-FP8-block"  # the FP8 checkpoint's repo id (the default served name)

SCHEMA = schema()
IMAGE_FILE = yaml.safe_load((ROOT / "deploy" / "config.yaml").read_text())
# The image's serving settings; its model path, runner and port are the test server's own
SERVING = {k: IMAGE_FILE[k] for k in ("scheduler", "structured_output")}
CONFIG = {**SERVING, "model": {"served_model_name": NAME, "max_model_len": 4096},
          "images": {"max_soft_tokens": 70, "max_per_request": 1},
          "server": {"keep_alive_s": 1.0}, "observability": {"log_level": "debug"}}


def test_the_image_file_holds_what_these_tests_take_and_the_tests_own_settings() -> None:
    """A setting the image's file adds is either one of its serving settings, taken here, or the server's own."""
    assert set(IMAGE_FILE) == {"model", "runner", *SERVING, "server"}
    assert set(IMAGE_FILE["model"]) == {"path"} and set(IMAGE_FILE["server"]) == {"port"}
    assert SERVING == {"scheduler": {"seed": 0}, "structured_output": {"backend": "guidance",
                                                                       "disable_any_whitespace": True}}


@pytest.fixture(scope="module")
def assistant(tmp_path_factory) -> dict:
    """speculative.model: an assistant's config.json, which is all of it the fake runner reads (the servers pin k
    at 3)."""
    d = tmp_path_factory.mktemp("assistant")
    (d / "config.json").write_text(json.dumps({"architectures": ["Gemma4AssistantForCausalLM"],
                                               "model_type": "gemma4_assistant"}))
    return {"speculative": {"model": str(d)}}


@pytest.fixture(scope="module")
def server(tmp_path_factory, assistant):
    srv = Server(GEMMA, tmp_path_factory.mktemp("compat"), step_ms=0.2,
                 extra={**CONFIG, **assistant}, env={"VLLM_API_KEY": KEY}, mtp=3).start(120)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]
    assert KEY not in srv.log_path.read_text()  # never logged, at DEBUG either


@pytest.fixture
def seeded_server(tmp_path, assistant):
    """A server of its own whose scheduler draws the same seeds every run (`scheduler.seed`): the structured requests
    carry no seed, so each gets one drawn by the scheduler, and the fake runner's answers depend on it (temperature 0
    included). The module's server has served other tests' requests by then (how many depends on which tests run), so
    its draws, and with them how many answers end before max_tokens, would vary (on Qwen, 1 run in 8 fails).
    Seed 5: at least four of the six end at a stop, and one runs into max_tokens, so both branches of the test
    run, every time."""
    srv = Server(GEMMA, tmp_path, step_ms=0.2,
                 extra={**CONFIG, **assistant, "scheduler": {"seed": 5}}, env={"VLLM_API_KEY": KEY},
                 mtp=3).start(120)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]
    assert KEY not in srv.log_path.read_text()


def jpeg(w: int, h: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), np.uint8)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def call(srv: Server, method: str, path: str, body: dict | None = None, auth: str | None = f"Bearer {KEY}",
         headers: dict | None = None) -> tuple[int, dict, bytes]:
    c = srv.conn()
    try:
        h = {"content-type": "application/json", **(headers or {})}
        if auth is not None:
            h["authorization"] = auth
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


def chat(**kw) -> dict:
    return {"model": NAME, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, **kw}


# -- the API key ---------------------------------------------------------------------------------------------------

GUARDED = (("POST", "/v1/chat/completions", chat()), ("GET", "/v1/models", None),
           ("POST", "/tokenize", {"prompt": "hi"}), ("POST", "/detokenize", {"tokens": [1]}),
           ("GET", "/server_info", None), ("GET", "/debug/engine", None), ("GET", "/nope", None),
           ("GET", "/health/", None), ("GET", "/metrics/x", None))
WRONG_KEYS = (None, "Bearer wrong", f"Basic {KEY}", f"Bearer {KEY}x")


def test_the_api_key_guards_every_path_but_the_probes(server) -> None:
    before = server.metric("llm_mini_infer:unauthorized_requests_total")
    codes_before = server.metric("llm_mini_infer:http_responses_total", 0.0, status="401")
    instrumentator_before = server.metric("http_requests_total", 0.0, status="4xx")
    refused = [(path, call(server, method, path, body, auth=a)) for method, path, body in GUARDED
               for a in WRONG_KEYS]
    for path, (status, headers, body) in refused:
        assert status == 401, (path, body)
        err = json.loads(body)["error"]
        assert err["type"] == "AuthenticationError" and err["code"] == 401 and "API key" in err["message"]
        assert headers["www-authenticate"] == "Bearer" and headers["x-llm-engine"].startswith("llm-mini-infer/")
        assert KEY.encode() not in body
    assert server.metric("llm_mini_infer:unauthorized_requests_total") == before + len(refused)
    # the exact codes count them, but for the paths no HTTP counter counts (/server_info, /health/, /metrics/x);
    # vLLM's http_requests_total does not (its API-key middleware sits outside it)
    counted = [path for path, _ in refused if not UNCOUNTED.search(path)]
    assert len(counted) == len(refused) - 3 * len(WRONG_KEYS)
    assert server.metric("llm_mini_infer:http_responses_total", status="401") == codes_before + len(counted)
    assert server.metric("http_requests_total", 0.0, status="4xx") == instrumentator_before
    for auth in (f"Bearer {KEY}", f"bearer {KEY}", f"BEARER {KEY}"):  # the scheme is case-insensitive (RFC 9110)
        assert call(server, "POST", "/v1/chat/completions", chat(), auth=auth)[0] == 200
    for method, path, body in GUARDED:  # with the key, the paths the server serves answer; the rest, a 404
        missing = path in ("/nope", "/health/", "/metrics/x")
        assert call(server, method, path, body)[0] == (404 if missing else 200), path
    info = call(server, "GET", "/server_info")[2]
    assert json.loads(info)["auth"] == "api key" and KEY.encode() not in info


def test_the_probes_and_the_scrape_need_no_key_with_or_without_a_query(server) -> None:
    for path in ("/health", "/ready", "/live", "/metrics", "/health?ready=1", "/ready?x=1", "/live?x=1",
                 "/metrics?x=1"):
        assert call(server, "GET", path, auth=None)[0] == 200, path


def test_options_without_a_key_reaches_no_handler(server) -> None:
    stages = call(server, "GET", "/debug/stages")
    assert stages[0] == 200
    for path, allow in (("/server_info", "GET"), ("/v1/models", "GET"), ("/v1/chat/completions", "POST"),
                        ("/tokenize", "POST"), ("/debug/reset", "GET")):
        status, headers, body = call(server, "OPTIONS", path, auth=None)
        assert status == 405 and headers["allow"] == allow, (path, status)
        assert b"data" not in body and b"model_len" not in body and NAME.encode() not in body, path
        assert KEY.encode() not in body
    for path in ("/nope", "/debug/nope"):
        assert call(server, "OPTIONS", path, auth=None)[0] == 404, path
    assert call(server, "OPTIONS", "/health", auth=None)[0] == 200
    after = call(server, "GET", "/debug/stages")
    assert after[0] == 200 and json.loads(after[2]) == json.loads(stages[2])  # /debug/reset did not run


def test_either_variable_holds_the_key_and_only_digests_are_kept() -> None:
    keys = ApiKeys({"LLM_MINI_INFER_API_KEY": "a-key", "VLLM_API_KEY": "b-key"})
    assert keys.sources == ["LLM_MINI_INFER_API_KEY", "VLLM_API_KEY"]
    for token, ok in ((b"a-key", True), (b"b-key", True), (b"c-key", False), (b"", False)):
        assert keys.allows([(b"authorization", b"Bearer " + token)]) is ok
    assert not keys.allows([]) and not ApiKeys({}).enabled and not ApiKeys({"VLLM_API_KEY": ""}).enabled
    assert "a-key" not in repr(vars(keys)) and "b-key" not in repr(vars(keys))
    for method, path in (("POST", "/v1/chat/completions"), ("GET", "/server_info"), ("GET", "/tokenize"),
                         ("POST", "/v2/x"), ("GET", "/health/"), ("GET", "/healthz"), ("GET", "/metrics/x")):
        assert keys.guards(method, path), path
    for path in ("/health", "/ready", "/live", "/metrics"):
        assert not keys.guards("GET", path) and not keys.guards("POST", path)
    assert not keys.guards("OPTIONS", "/v1/models") and not keys.guards("OPTIONS", "/server_info")
    assert not ApiKeys({}).guards("POST", "/v1/chat/completions") and not ApiKeys({}).guards("GET", "/server_info")


def test_a_client_of_the_server_sends_the_first_key_set() -> None:
    both = {"LLM_MINI_INFER_API_KEY": "a-key", "VLLM_API_KEY": "b-key"}
    assert client_headers(both) == {"authorization": "Bearer a-key"}
    assert client_headers({"VLLM_API_KEY": "b-key"}) == {"authorization": "Bearer b-key"}
    assert client_headers({"LLM_MINI_INFER_API_KEY": "", "VLLM_API_KEY": "b-key"}) == {"authorization": "Bearer b-key"}
    assert client_headers({}) == {} and client_headers({"VLLM_API_KEY": ""}) == {}


# -- the engine marker ---------------------------------------------------------------------------------------------

def test_every_response_names_the_engine(server) -> None:
    status, headers, body = call(server, "POST", "/v1/chat/completions", chat())
    out = json.loads(body)
    assert status == 200 and out["system_fingerprint"].startswith("llm-mini-infer-")
    assert headers["x-llm-engine"] == "llm-mini-infer/" + out["system_fingerprint"].removeprefix("llm-mini-infer-")
    assert call(server, "GET", "/health")[1]["x-llm-engine"] == headers["x-llm-engine"]
    c = server.conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(chat(stream=True, max_tokens=8,
                                                                   stream_options={"include_usage": True})),
              headers={"content-type": "application/json", "authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("x-llm-engine") == headers["x-llm-engine"]
    events = [json.loads(line[6:]) for line in r.read().split(b"\n") if line.startswith(b"data: {")]
    c.close()
    assert len(events) >= 3 and all(e["system_fingerprint"] == out["system_fingerprint"] for e in events)
    assert events[-1]["usage"]["completion_tokens"] == 8  # the usage chunk carries it too


# -- a structured request with one image ---------------------------------------------------------------------------

def image_request(model: str, prompt: str, image: bytes, max_tokens: int = 512) -> dict:
    """The OpenAI SDK's keyword arguments for a structured request with one image, with thinking off
    (`chat_template_kwargs` through `extra_body`)."""
    url = f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "text", "text": "Describe this photo."},
                                         {"type": "image_url", "image_url": {"url": url}}]},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": SCHEMA}},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


PROMPT = ("You catalogue the items in a photo. Count them as total, then list up to five in entries, left to right, "
          "each with its id and its attributes. Answer with the JSON object only.")


def test_a_structured_image_request_from_the_openai_sdk_gets_valid_json(seeded_server) -> None:
    from openai import OpenAI

    server = seeded_server
    # no retries: a retry would hide a failed request; the timeout only bounds a hung test
    client = OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key=KEY, timeout=60.0, max_retries=0)
    validator = jsonschema.Draft202012Validator(SCHEMA)
    stops = lengths = 0
    for i in range(6):
        rid = f"req-{i}"
        max_tokens = 4 if i == 5 else 512  # the last one too short for any document: a length finish
        raw = client.chat.completions.with_raw_response.create(
            **image_request(NAME, PROMPT, jpeg(640 - 40 * i, 360 + 24 * i, seed=i), max_tokens),
            extra_headers={"X-Request-Id": rid})
        completion = raw.parse()
        assert raw.headers["x-request-id"] == rid and raw.headers["x-llm-engine"].startswith("llm-mini-infer/")
        assert completion.id == f"chatcmpl-{rid}" and completion.model == NAME
        assert completion.system_fingerprint.startswith("llm-mini-infer-")  # what the OpenAI client exposes
        choice = completion.choices[0]
        assert getattr(choice.message, "reasoning", None) is None  # thinking off: the whole answer is content
        if choice.finish_reason == "stop":
            stops += 1
            doc = json.loads(choice.message.content)  # what a client reads: choices[0].message.content
            validator.validate(doc)
            Catalog.model_validate_json(choice.message.content)  # as a client's pydantic model parses it
        else:
            assert choice.finish_reason == "length" and completion.usage.completion_tokens == max_tokens
            lengths += 1
    assert stops >= 4 and lengths >= 1  # both ends checked
    from openai import AuthenticationError

    with pytest.raises(AuthenticationError):  # the OpenAI client on a wrong key: an AuthenticationError, no retry
        OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key="wrong", max_retries=0).chat.completions.create(
            **image_request(NAME, PROMPT, jpeg(64, 64)))


# -- vLLM's limits -------------------------------------------------------------------------------------------------

def tokens(srv: Server, messages: list) -> int:
    status, _, body = call(srv, "POST", "/tokenize", {"messages": messages, "add_generation_prompt": True})
    assert status == 200, body
    return json.loads(body)["count"]


def text_of(srv: Server, n: int, image: bytes | None = None) -> list:
    """Messages whose rendered prompt has exactly `n` tokens (with an image: its placeholder counted once, as vLLM
    checks the text before it expands the images)."""
    def msgs(words: int) -> list:
        parts = [{"type": "text", "text": "Describe it." + " word" * words}]
        if image is not None:
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                                             base64.b64encode(image).decode()}})
        return [{"role": "user", "content": parts}]

    base = tokens(srv, msgs(0)) if image is None else _text_tokens(srv, msgs(0))
    words = n - base
    got = tokens(srv, msgs(words)) if image is None else _text_tokens(srv, msgs(words))
    assert got == n, (got, n)
    return msgs(words)


def _text_tokens(srv: Server, messages: list) -> int:
    """The prompt's tokens with the image as one placeholder: the expanded count minus the span and the tokens around
    it, plus one."""
    from llm_mini_infer.vision import family as vf

    fam = vf.load(GEMMA, max_soft_tokens=70)
    url = messages[0]["content"][1]["image_url"]["url"]
    from llm_mini_infer.vision.probe import probe

    span = fam.size(probe(base64.b64decode(url.split(",", 1)[1]))).tokens
    return tokens(srv, messages) - span - len(fam.span_prefix) - len(fam.span_suffix) + 1  # <|image> .. <image|>


def test_the_context_limit_matches_the_clients_expectation(server) -> None:
    limit = 4096
    ok = text_of(server, limit - 512)
    status, _, body = call(server, "POST", "/v1/chat/completions", chat(messages=ok, max_tokens=512))
    assert status == 200, body
    # max_tokens is a cap: a prompt that fits alone is served, its output clamped to the room it leaves
    over = text_of(server, limit - 511)
    status, _, body = call(server, "POST", "/v1/chat/completions",
                           chat(messages=over, max_tokens=512, ignore_eos=True, temperature=0))
    out = json.loads(body)
    assert status == 200 and out["usage"]["completion_tokens"] == 511, body
    assert out["choices"][0]["finish_reason"] == "length"
    # a prompt past the limit alone: vLLM tokenizes at most one token more than the limit, and says "at least"
    for n in (limit + 1, limit + 100):
        status, _, body = call(server, "POST", "/v1/chat/completions", chat(messages=text_of(server, n), max_tokens=1))
        err = json.loads(body)["error"]
        assert status == 400 and err["param"] == "input_tokens"
        assert err["message"] == ("This model's maximum context length is 4096 tokens. However, your prompt contains "
                                  "at least 4097 input tokens, which exceeds this limit even without any output "
                                  "tokens. Please reduce the length of the input prompt. (parameter=input_tokens, "
                                  "value=4097)")
    # past vLLM's bound on the characters, checked before it tokenizes: the vocabulary's longest token (31) a token
    status, _, body = call(server, "POST", "/v1/chat/completions",
                           chat(messages=[{"role": "user", "content": "a" * 130_000}], max_tokens=512))
    err = json.loads(body)["error"]
    n = int(err["message"].split("prompt contains ")[1].split(" ")[0])
    assert status == 400 and err["param"] == "input_text" and n > 130_000
    assert err["message"] == (f"This model's maximum context length is 4096 tokens. However, your prompt contains {n} "
                              "characters (more than 126976 characters, which is the upper bound for 4096 input "
                              "tokens). Please reduce the length of the input prompt. "
                              f"(parameter=input_text, value={n})")
    status, _, body = call(server, "POST", "/v1/chat/completions", chat(max_completion_tokens=limit + 1))
    err = json.loads(body)["error"]
    assert status == 400 and err["param"] == "max_completion_tokens"
    assert err["message"] == ("max_completion_tokens=4097 cannot be greater than max_model_len=4096. Please request "
                              "fewer output tokens. (parameter=max_completion_tokens, value=4097)")
    # a prompt that fills the context, with max_tokens or without: it passes the prompt's check, and vLLM's
    # get_max_tokens leaves 0, which SamplingParams refuses
    for asked in (None, 512):
        full = chat(messages=text_of(server, limit), max_tokens=asked)
        status, _, body = call(server, "POST", "/v1/chat/completions", full)
        err = json.loads(body)["error"]
        assert status == 400 and (err["message"], err["param"]) == (
            "max_tokens must be at least 1, got 0. (parameter=max_tokens, value=0)", "max_tokens")
    # an image: vLLM checks the text with one placeholder token, then clamps max_tokens to the room the expanded
    # prompt leaves
    img = jpeg(640, 480, seed=11)
    fits = text_of(server, limit - 512, image=img)
    status, _, body = call(server, "POST", "/v1/chat/completions",
                           chat(messages=fits, max_tokens=512, ignore_eos=True, temperature=0))
    out = json.loads(body)
    assert status == 200, body
    prompt = out["usage"]["prompt_tokens"]
    assert prompt > limit - 512 and out["usage"]["completion_tokens"] == limit - prompt  # clamped, not refused
    assert out["choices"][0]["finish_reason"] == "length"


def test_the_per_prompt_image_limit_matches_the_clients_expectation(server) -> None:
    two = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg(64, 64)).decode()}}
        for _ in range(2)]}]
    status, _, body = call(server, "POST", "/v1/chat/completions", chat(messages=two))
    err = json.loads(body)["error"]
    assert status == 400 and (err["message"], err["param"]) == (
        "At most 1 image(s) may be provided in one prompt. (parameter=image)", "image")


# -- keep-alive ----------------------------------------------------------------------------------------------------

def get_on(sock: socket.socket, path: str = "/health") -> bytes:
    """One request on a raw keep-alive connection; the response's head and body."""
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: t\r\n\r\n".encode())
    data = b""
    while b"\r\n\r\n" not in data:
        data += sock.recv(65536)
    head, _, rest = data.partition(b"\r\n\r\n")
    length = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                      if line.lower().startswith(b"content-length")))
    while len(rest) < length:
        rest += sock.recv(65536)
    return head


def test_idle_connections_close_after_keep_alive_s(server) -> None:
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    try:
        assert b" 200 " in get_on(s) and b" 200 " in get_on(s)  # kept alive between requests
        time.sleep(0.5)
        assert b" 200 " in get_on(s)  # within keep_alive_s
        t0 = time.monotonic()
        assert s.recv(1) == b""  # the server closes it once idle for keep_alive_s (1 s)
        assert 0.8 < time.monotonic() - t0 < 2.5
    finally:
        s.close()


def test_a_drain_closes_kept_alive_connections_after_their_next_response(tmp_path) -> None:
    srv = Server(GEMMA, tmp_path, step_ms=0.2,
                 extra={**CONFIG, "images": {"enabled": False},  # no image is sent: no pool
                        "server": {"keep_alive_s": 20.0, "drain_grace_s": 3.0}}).start(120)
    try:
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        head = get_on(s)
        assert b" 200 " in head and b"connection: close" not in head.lower()
        srv.proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        head = get_on(s)  # the same kept-alive connection, during the grace
        assert b" 503 " in head and b"connection: close" in head.lower()
        assert s.recv(1) == b""  # closed right after that response
        s.close()
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)  # a new one: still served in the grace
        c.request("POST", "/v1/chat/completions", body=json.dumps(chat()), headers={"content-type": "application/json"})
        r = c.getresponse()
        assert r.status == 200 and r.getheader("connection") == "close"
        r.read()
        c.close()
        assert srv.proc.wait(30) == 0  # the drain ends by itself (a second SIGTERM would fail what is in flight)
    finally:
        srv.stop(timeout=30)


# -- a shared GPU --------------------------------------------------------------------------------------------------

def test_a_gpu_without_our_share_free_refuses_to_start() -> None:
    from llm_mini_infer.runner.cuda.runner import check_free_memory

    gib = 2**30
    check_free_memory(int(90 * gib), 96 * gib, 0.9, "cuda:0")  # 86.4 GiB wanted, 90 free
    with pytest.raises(RuntimeError) as exc:
        check_free_memory(int(80 * gib), 96 * gib, 0.9, "cuda:0")  # another process holds 16 GiB
    msg = str(exc.value)
    assert "Free memory on device cuda:0 (80.00/96.00 GiB)" in msg and "86.40 GiB" in msg
    assert "runner.cuda.memory_utilization 0.9" in msg and "16.00 GiB are in use" in msg


# -- tool calls through the OpenAI SDK -----------------------------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather.",
                                                 "parameters": {"type": "object", "properties": {"city": {
                                                     "type": "string"}}, "required": ["city"]}}}
LOOP_USER = [{"role": "user", "content": "Weather in Paris?"}]
LOOP_CALL = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
LOOP_ANSWER = "It is 18 degrees and cloudy in Paris."
LOOP_SEEDS = (8101, 8102)


def loop_scripts() -> list:
    """The fake's two answers of a tool loop on Gemma 4: a call, then (after the tool's result) the answer."""
    from llm_mini_infer.api.tools import template_tool
    from llm_mini_infer.render.messages import to_template_messages
    from llm_mini_infer.render.renderer import Renderer

    r = Renderer(GEMMA)
    tools = [template_tool(WEATHER_TOOL)]

    def prompt_len(msgs: list) -> int:
        return len(r.encode(r.render(to_template_messages(msgs, r.content_format, r.supports_developer_role),
                                     tools=tools)))

    turn2 = [*LOOP_USER, {"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
             {"role": "tool", "tool_call_id": "x", "content": '{"temp_c": 18, "sky": "cloudy"}'}]
    enc = r.tokenizer.encode
    return [[LOOP_SEEDS[0], prompt_len(LOOP_USER), [*enc(LOOP_CALL, add_special_tokens=False).ids, 50]],
            [LOOP_SEEDS[1], prompt_len(turn2), [*enc(LOOP_ANSWER, add_special_tokens=False).ids, 106]]]


@pytest.fixture(scope="module")
def tool_server(tmp_path_factory, assistant):
    extra = {**CONFIG, **assistant, "images": {"enabled": False},  # no image is sent: no pool
             "dev": {"runner": {"fake": {"scripts": loop_scripts()}}}}
    srv = Server(GEMMA, tmp_path_factory.mktemp("tools"), step_ms=0.2, extra=extra, env={"VLLM_API_KEY": KEY},
                 mtp=3).start(120)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


@pytest.mark.parametrize("stream", [False, True], ids=["non-streamed", "streamed"])
def test_a_tool_loop_through_the_openai_sdk(tool_server, stream) -> None:
    """The SDK's own objects round trip: a call (the deltas accumulated by the SDK when streamed), the assistant message
    echoed back as the SDK dumps it (null refusal, annotations, audio, function_call; streamed, its parsed fields and
    the call's index), the tool's result, and the answer."""
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{tool_server.port}/v1", api_key=KEY, timeout=60.0, max_retries=0)

    def ask(messages: list, seed: int):
        kw = {"model": NAME, "messages": messages, "tools": [WEATHER_TOOL], "seed": seed, "max_tokens": 64}
        if not stream:
            return client.chat.completions.create(**kw)
        with client.chat.completions.stream(**kw) as s:
            return s.get_final_completion()

    first = ask(LOOP_USER, LOOP_SEEDS[0])
    choice = first.choices[0]
    # streamed, the role chunk's "" is the content the SDK accumulates
    assert choice.finish_reason == "tool_calls" and choice.message.content == ("" if stream else None)
    [call_] = choice.message.tool_calls
    assert call_.type == "function" and call_.function.name == "get_weather"
    assert json.loads(call_.function.arguments) == {"city": "Paris"} and call_.id.startswith("chatcmpl-tool-")
    echoed = choice.message.model_dump()  # the SDK's message, every field of it (streamed: parsed, a call's index)
    result = {"role": "tool", "tool_call_id": call_.id, "content": '{"temp_c": 18, "sky": "cloudy"}'}
    second = ask([*LOOP_USER, echoed, result], LOOP_SEEDS[1])
    assert second.choices[0].finish_reason == "stop" and second.choices[0].message.content == LOOP_ANSWER
    assert not second.choices[0].message.tool_calls

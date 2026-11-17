# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The oracle is adapted from vllm/entrypoints/openai/chat_completion/protocol.py, chat_completion/serving.py and
# vllm/v1/engine/logprobs.py of vLLM v0.33.0 (https://github.com/vllm-project/vllm), and changed for llm-mini-infer.
"""Chat completions' logprobs at the frontend (`output.logprobs`, `api.protocol`): the token strings vLLM writes
(transformers' per-id decode, with a byte-fallback correction), the JSON its two encoders write (the stream's
`model_dump_json`, the body's Python `json`), vLLM's request checks and messages, and the server end to end on
the fake runner. The oracle re-declares vLLM's pydantic models, response builder and byte-fallback correction."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from llm_mini_infer.api.protocol import ApiError, LogprobsLimit, decode_chat, logprobs_param, validate_chat
from llm_mini_infer.output.logprobs import RequestLogprobs, TokenText, content
from llm_mini_infer.runner.fake import fake_logprobs

from ..support.api_util import Predictor, Server
from ..support.fixtures import QWEN, require_qwen
from ..support.paths import MODELS_DIR

GEMMA = MODELS_DIR / "gemma-4-31b-it"
VOCAB = 248_320


def tokenizer_dirs() -> list:
    return [pytest.param(d, id=d.name) for d in (QWEN, GEMMA)]


def need(d) -> None:
    if not (d / "tokenizer.json").exists():
        pytest.skip(f"{d} is absent: uv run tests/scripts/fetch_fixtures.py")


# -- vLLM's models and builder (chat_completion/protocol.py:82-97, serving.py:1214-1305) -------------------------------


class LogProb(BaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class LogProbsContent(LogProb):
    top_logprobs: list[LogProb] = Field(default_factory=list)


class LogProbs(BaseModel):
    content: list[LogProbsContent] | None = None


def vllm_logprobs(tokens: list[int], token_lp: list[float], top_ids: list[list[int]], top: list[list[float]], k: int,
                  text: list[str], top_text: list[list[str]]) -> LogProbs:
    """vLLM's `_create_chat_logprobs` over its engine's rows ([sampled, top-k] per position, a dict deduplicated in
    insertion order, the first `k` items)."""
    out = []
    for t, v, ids, vals, s, ts in zip(tokens, token_lp, top_ids, top, text, top_text):
        d: dict = {t: (v, s)}
        for i, x, st in zip(ids[:k], vals[:k], ts[:k]):
            d[i] = (x, st)
        tops = [LogProb(token=st, logprob=max(x, -9999.0), bytes=list(st.encode("utf-8", errors="replace")))
                for i, (x, st) in list(d.items())[:k]]
        out.append(LogProbsContent(token=s, logprob=max(d[t][0], -9999.0), bytes=list(s.encode("utf-8")),
                                   top_logprobs=tops))
    return LogProbs(content=out)


def vllm_corrected(tok, token: int, context: list[int]) -> str:
    """vLLM's `LogprobsProcessor._correct_decoded_token` on a transformers tokenizer (`v1/engine/logprobs.py`)."""
    for num_ctx in range(1, min(len(context), 4) + 1):
        ctx = context[-num_ctx:]
        full = tok.decode([*ctx, token])
        if full.endswith("�"):
            continue
        clean_end = len(ctx)
        for j in range(len(ctx) - 1, -1, -1):
            if tok.decode([ctx[j]]).endswith("�"):
                clean_end = j
            else:
                break
        prefix = tok.decode(ctx[:clean_end]) if clean_end > 0 else ""
        if full.startswith(prefix):
            return full[len(prefix):]
        common = 0
        for a, b in zip(prefix, full):
            if a != b:
                break
            common += 1
        return full[common:]
    return ""


@pytest.mark.parametrize("d", tokenizer_dirs())
def test_each_ids_string_is_transformers_decode(d) -> None:
    """An id's string is transformers' `decode([id])` (special tokens kept); neither checkpoint has a Metaspace
    pre-tokenizer."""
    need(d)
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(d)
    tt = TokenText(Tokenizer.from_file(str(d / "tokenizer.json")))
    assert tt.marker is None
    n = len(hf)
    bad = [i for i in range(n) if tt.alone(i) != (hf.decode([i]) or "")]
    assert not bad, bad[:10]


@pytest.mark.parametrize("d", tokenizer_dirs())
def test_a_split_characters_pieces_are_corrected_as_vllm(d) -> None:
    """Characters byte-fallback splits, as output tokens: vLLM's correction (the last 1 to 4 sampled tokens as context)
    gives the open pieces "" and the closing one the character."""
    need(d)
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(d)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tt = TokenText(tok)
    pieces = [i for i in range(len(hf)) if tt.alone(i).endswith("�")]
    assert pieces  # both vocabularies have tokens that are pieces of a character
    rng = np.random.default_rng(0)
    text = "naïve € 你好 😀 ok"
    ids = tok.encode(text, add_special_tokens=False).ids
    seqs = [ids, [*rng.choice(pieces, 6).tolist(), *ids[:3]], [*ids[:2], *rng.choice(pieces, 3).tolist()]]
    for s in seqs:
        lp = RequestLogprobs(tt, 0, True)
        got = [json.loads(e)["token"] for e in lp.entries(s, [0.0] * len(s), [[]] * len(s), [[]] * len(s))]
        want, ctx = [], []
        for t in s:
            w = hf.decode([t]) or ""
            want.append(vllm_corrected(hf, t, ctx) if w.endswith("�") else w)
            ctx = [*ctx, t][-4:]
        assert got == want, (s, got, want)


VALUES = [0.0, -0.0, -1e-05, -1.2e-05, -9.9e-06, -1.1920928955078125e-07, -0.0001234, -2.5, -12.375, -9998.5,
          -1e4, -math.inf]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_the_entries_are_vllms_json(stream, k) -> None:
    """Our entries byte for byte against vLLM's models: in the stream `model_dump_json(exclude_unset=True)`
    (pydantic-core's floats), in the body Python's `json` (`repr`'s: `1e-05`, `-9999.0` for −inf); the sampled token
    first, out of rank order or ahead of a dropped k-th; special and non-ASCII tokens."""
    need(QWEN)
    tok = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    tt = TokenText(tok)
    rng = np.random.default_rng(k)
    n = len(VALUES)
    tokens = rng.integers(0, 248_000, n).tolist()
    tokens[0] = 248_044  # a special token
    top_ids = [rng.integers(0, 248_000, 5).tolist() for _ in range(n)]
    top_ids[1][1] = tokens[1]  # the sampled token second in rank
    top_ids[2][0] = tokens[2]
    top = [sorted(rng.choice(VALUES, 5).tolist(), reverse=True) for _ in range(n)]
    for i in range(n):  # a sampled token among the top ones has its own value there (one computation of both)
        top[i] = [VALUES[i] if t == tokens[i] else v for t, v in zip(top_ids[i], top[i])]
    lp = RequestLogprobs(tt, k, stream)
    got = content(lp.entries(tokens, VALUES, top_ids, top))
    text = [tt.alone(t) for t in tokens]
    top_text = [[tt.alone(i) for i in ids] for ids in top_ids]
    model = vllm_logprobs(tokens, VALUES, top_ids, top, k, text, top_text)
    want = model.model_dump_json(exclude_unset=True) if stream else \
        json.dumps(model.model_dump(), ensure_ascii=False, separators=(",", ":"))
    assert got.decode() == want


# -- the request -------------------------------------------------------------------------------------------------------


def checked(body: dict, vocab: int = VOCAB) -> int:
    req = decode_chat(json.dumps({"messages": [{"role": "user", "content": "hi"}], **body}).encode())
    validate_chat(req, vocab)
    return logprobs_param(req, vocab)


@pytest.mark.parametrize("body,k", [({}, -1), ({"logprobs": False}, -1), ({"logprobs": None}, -1),
                                    ({"logprobs": True}, 0), ({"logprobs": True, "top_logprobs": 0}, 0),
                                    ({"logprobs": True, "top_logprobs": None}, 0),  # vLLM's: null is absent
                                    ({"logprobs": True, "top_logprobs": 5}, 5),
                                    ({"logprobs": True, "top_logprobs": 20}, 20),
                                    ({"logprobs": False, "top_logprobs": 0}, -1), ({"top_logprobs": None}, -1)])
def test_the_requests_top_logprobs_are_vllms(body, k) -> None:
    assert checked(body) == k


@pytest.mark.parametrize("body,param,message", [
    ({"top_logprobs": 3}, "top_logprobs",
     "when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs)"),
    ({"logprobs": False, "top_logprobs": -1}, "top_logprobs",
     "when using `top_logprobs`, `logprobs` must be set to true. (parameter=top_logprobs)"),
    ({"logprobs": True, "top_logprobs": -2}, "top_logprobs",
     "`top_logprobs` must be a positive value or -1. (parameter=top_logprobs, value=-2)"),
    ({"logprobs": True, "top_logprobs": 21}, "logprobs",
     "Requested sample logprobs of 21, which is greater than max allowed: 20 (parameter=logprobs, value=21)"),
    ({"logprobs": True, "top_logprobs": -1}, "logprobs",
     f"Requested sample logprobs of {VOCAB}, which is greater than max allowed: 20 (parameter=logprobs, "
     f"value={VOCAB})")])
def test_refusals_are_vllms(body, param, message) -> None:
    with pytest.raises(ApiError) as e:
        checked(body)
    assert (e.value.status, e.value.param, e.value.message) == (400, param, message)
    assert isinstance(e.value, LogprobsLimit) == (param == "logprobs")


# -- end to end on the fake --------------------------------------------------------------------------------------------


MSG = [{"role": "user", "content": "Hello!"}]
KW = {"enable_thinking": False}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    model = require_qwen()
    srv = Server(model, tmp_path_factory.mktemp("lp"), mtp=3, extra={"server": {"logprobs": True}})
    srv.start()
    srv.pred = Predictor(model)
    srv.prompt_ids = srv.pred.prompt(MSG, **KW)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def chat(**kw) -> dict:
    return {"messages": MSG, "chat_template_kwargs": KW, "max_tokens": 9, "seed": 77, **kw}


def predicted(srv, k: int, n: int = 9) -> tuple[list[int], list]:
    """The fake's tokens and their values for the request `chat()` sends."""
    from llm_mini_infer.runner.fake import context_hash, fake_token

    toks = srv.pred.tokens(srv.prompt_ids, 77, n)
    h = context_hash(srv.prompt_ids)
    vals = []
    for i, t in enumerate(toks):
        pos = len(srv.prompt_ids) + i
        assert t == fake_token(77, pos, VOCAB, srv.pred.eos_ids, h)
        vals.append(fake_logprobs(77, pos, h, 0, t, None, VOCAB))
        h = context_hash((t,), h)
    return toks, vals


@pytest.mark.parametrize("k", [0, 4])
def test_the_body_and_the_stream_carry_every_tokens_entry(server, k) -> None:
    toks, vals = predicted(server, k)
    tt = TokenText(Tokenizer.from_file(str(QWEN / "tokenizer.json")))
    for stream in (False, True):
        lp = RequestLogprobs(tt, k, stream)
        want = [json.loads(e) for e in lp.entries(toks, [v for v, _, _ in vals], [i for _, i, _ in vals],
                                                  [t for _, _, t in vals])]
        if stream:
            ev = server.stream(chat(stream=True, logprobs=True, top_logprobs=k))
            chunks = [e for e in ev if isinstance(e, dict)]
            assert chunks[0]["choices"][0]["logprobs"] is None  # the role chunk
            got = [e for c in chunks[1:] for e in c["choices"][0]["logprobs"]["content"]]
            assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        else:
            st, r = server.post("/v1/chat/completions", chat(logprobs=True, top_logprobs=k))
            assert st == 200
            got = r["choices"][0]["logprobs"]["content"]
        assert got == want
        assert all(len(e["top_logprobs"]) == k for e in got)


def test_without_logprobs_the_wire_is_unchanged(server) -> None:
    st, r = server.post("/v1/chat/completions", chat())
    assert st == 200 and r["choices"][0]["logprobs"] is None
    ev = server.stream(chat(stream=True))
    assert all(c["choices"][0]["logprobs"] is None for c in ev if isinstance(c, dict) and c["choices"])
    st, r = server.post("/v1/chat/completions", chat(top_logprobs=None))  # null without logprobs: none
    assert st == 200 and r["choices"][0]["logprobs"] is None


def test_a_null_top_logprobs_is_absent(server) -> None:
    """vLLM's `TopLogprobsParam`: with `logprobs` true, `top_logprobs` null is 0, the sampled tokens' logprobs with
    empty top lists, as when it is absent, streamed or not."""
    st, r = server.post("/v1/chat/completions", chat(logprobs=True, top_logprobs=None))
    st0, r0 = server.post("/v1/chat/completions", chat(logprobs=True))
    got = r["choices"][0]["logprobs"]["content"]
    assert st == st0 == 200 and got and got == r0["choices"][0]["logprobs"]["content"]
    assert all(e["top_logprobs"] == [] for e in got)
    ev = server.stream(chat(stream=True, logprobs=True, top_logprobs=None))
    streamed = [e for c in ev if isinstance(c, dict) and c["choices"] and c["choices"][0]["logprobs"]
                for e in c["choices"][0]["logprobs"]["content"]]
    assert streamed == got


def test_a_streamed_request_over_the_limit_gets_one_error_event(server) -> None:
    """One error event, then [DONE]: the event's JSON compact, as vLLM's `model_dump_json` writes it."""
    ev = server.stream(chat(stream=True, logprobs=True, top_logprobs=21))
    msg = "Requested sample logprobs of 21, which is greater than max allowed: 20 (parameter=logprobs, value=21)"
    assert ev == [{"error": {"message": msg, "type": "BadRequestError", "param": "logprobs", "code": 400}}, "[DONE]"]
    c = server.conn()
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(chat(stream=True, logprobs=True, top_logprobs=21)),
                  headers={"content-type": "application/json"})
        raw = c.getresponse().read()
    finally:
        c.close()
    assert raw == (b'data: {"error":{"message":"' + msg.encode() + b'","type":"BadRequestError","param":"logprobs",'
                   b'"code":400}}\n\ndata: [DONE]\n\n')
    st, r = server.post("/v1/chat/completions", chat(logprobs=True, top_logprobs=21))
    assert st == 400 and r["error"]["message"] == msg


def test_logprobs_are_off_by_default_a_400(tmp_path) -> None:
    model = require_qwen()
    srv = Server(model, tmp_path)
    srv.start()
    try:
        st, r = srv.post("/v1/chat/completions", chat(logprobs=True))
        assert st == 400 and r["error"]["param"] == "logprobs"
        assert srv.post("/v1/chat/completions", chat())[0] == 200
    finally:
        assert srv.stop() == 0

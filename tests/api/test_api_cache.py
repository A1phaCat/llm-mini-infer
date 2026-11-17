"""The prefix cache end to end: the real server (launcher, engine and frontend processes) against the strict fake
runner, with the Qwen3.8 tokenizer and chat template. Outputs are predicted from the true context (the fake's tokens
depend on it), so a hit that reused the wrong prefix would change them."""

from __future__ import annotations

import pytest

from llm_mini_infer.render.messages import to_template_messages

from ..support.api_cache_cases import KW, SYSTEM
from ..support.api_util import Predictor, Server, content_of
from ..support.fixtures import require_qwen


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    model = require_qwen()
    srv = Server(model, tmp_path_factory.mktemp("srv"), extra={"dev": {"cache": {"decode_checkpoint_interval": 64}}})
    srv.start()
    srv.pred = Predictor(model)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def ask(srv: Server, messages: list[dict], seed: int, max_tokens: int = 6, **extra) -> tuple[dict, list[int]]:
    """POST a chat; check its text against the fake's prediction; return the response and the prompt IDs."""
    plain = [{k: v for k, v in m.items() if k != "cache_control"} for m in messages]
    prompt = srv.pred.prompt([{**m, "content": m["content"] if isinstance(m["content"], str) else
                               [{k: v for k, v in p.items() if k != "cache_control"} for p in m["content"]]}
                              for m in plain], **KW)
    status, r = srv.post("/v1/chat/completions", {"messages": messages, "chat_template_kwargs": KW,
                                                  "max_tokens": max_tokens, "seed": seed, "ignore_eos": True,
                                                  **extra})
    assert status == 200, r
    want = srv.pred.text(prompt, srv.pred.tokens(prompt, seed, max_tokens))[0]
    assert r["choices"][0]["message"]["content"] == want
    assert r["usage"]["prompt_tokens"] == len(prompt)
    return r, prompt


def cached(r: dict) -> int:
    return r["usage"]["prompt_tokens_details"]["cached_tokens"]


def test_shared_system_prompt_is_reused(server) -> None:
    """A long shared system prompt and short unique items, text only, one request at a time: every request after the
    first hits."""
    hits = []
    for i in range(6):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Item {i}: a blue chair"}]
        r, prompt = ask(server, msgs, seed=100 + i)
        hits.append(cached(r))
    assert hits[0] == 0 and all(h > 0.9 * len(prompt) - 64 for h in hits[1:]), (hits, len(prompt))


def hint_offset(server: Server, system: str) -> int:
    r = server.pred.r
    conv = to_template_messages([{"role": "system", "content": system}, {"role": "user", "content": "Item A"}],
                                r.content_format, r.supports_developer_role)
    text = r.render(conv, add_generation_prompt=True, **KW)
    return r.hint_offsets(conv, [0], text, r.encode(text), **KW)[0]


def test_cache_control_gives_an_exact_hit(server) -> None:
    """A hint on the system message: the next request with another user turn hits exactly where the system message
    ends, off the 64-token grid (the system prompt is chosen so that no grid point lies on the shared part after
    it), on a fresh namespace so nothing else is cached there."""
    system = next(s for s in (SYSTEM + " Answer briefly." + " Be precise." * k for k in range(64))
                  if 8 <= hint_offset(server, s) % 64 <= 40)
    end = hint_offset(server, system)
    first = [{"role": "system", "content": system, "cache_control": {"type": "ephemeral"}},
             {"role": "user", "content": "Item A"}]
    r1, _ = ask(server, first, seed=7, cache_salt="hints")
    second = [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]},
              {"role": "user", "content": "Item B, which is longer"}]
    r2, _ = ask(server, second, seed=8, cache_salt="hints")
    assert cached(r1) == 0 and cached(r2) == end, (cached(r2), end)


def test_cache_control_under_batch_invariance(tmp_path) -> None:
    """The fake reporting batch invariance (`dev.runner.fake.batch_invariant`: the scheduler cuts a GDN model's prompts
    on the 64-token grid from where their computation starts): a hint off the grid stays an exact cut, and the next
    request hits there."""
    model = require_qwen()
    srv = Server(model, tmp_path, extra={"dev": {"runner": {"fake": {"batch_invariant": True}}}}).start()
    try:
        srv.pred = Predictor(model)
        test_cache_control_gives_an_exact_hit(srv)
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_multi_turn_reuses_the_first_turn(server) -> None:
    turn1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Describe a red lamp.",
                                                      "cache_control": {"type": "ephemeral"}}]
    r1, p1 = ask(server, turn1, seed=21, max_tokens=12, cache_salt="chat")
    reply = r1["choices"][0]["message"]["content"]
    turn2 = [*turn1, {"role": "assistant", "content": reply}, {"role": "user", "content": "And a green one?"}]
    r2, p2 = ask(server, turn2, seed=22, cache_salt="chat")
    # the hint ends turn 1's user message; the rest of turn 1's prompt is the generation prompt, rendered the same
    assert cached(r2) >= len(p1) - 16 and p2[:cached(r2)] == p1[:cached(r2)]


def test_salt_isolates(server) -> None:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Item 99: a steel spoon"}]
    ask(server, msgs, seed=31)  # the unsalted namespace has the system prompt already
    r, _ = ask(server, msgs, seed=32, cache_salt="tenant-x")
    assert cached(r) == 0
    r, _ = ask(server, msgs, seed=33, cache_salt="tenant-x")
    assert cached(r) > 0


def test_streaming_usage_reports_cached_tokens(server) -> None:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Item 5: a wooden box"}]
    events = server.stream({"messages": msgs, "chat_template_kwargs": KW, "max_tokens": 3, "seed": 41,
                            "stream_options": {"include_usage": True}})
    assert events[-1] == "[DONE]" and content_of(events)
    assert events[-2]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_cache_metrics(server) -> None:
    queries = server.metric("vllm:prefix_cache_queries_total")
    hits = server.metric("vllm:prefix_cache_hits_total")
    assert 0 < hits < queries
    assert server.metric("llm_mini_infer:checkpoint_hits_total") > 0
    assert server.metric("llm_mini_infer:checkpoint_misses_total") > 0
    assert server.metric("llm_mini_infer:checkpoints_total") > 0
    assert server.metric("llm_mini_infer:prefix_cache_pages") > 0
    _, text = server.get("/metrics")
    assert b'llm_mini_infer:checkpoints_total{model_name="qwen-test",engine="0",source="hint"}' in text
    assert b'enable_prefix_caching="True"' in text
    for cause in ("writing", "checkpoint", "page", "precise"):  # the cache's lost tokens, every cause exported
        series = f'llm_mini_infer:prefix_cache_lost_tokens_total{{model_name="qwen-test",engine="0",cause="{cause}"}}'
        assert series.encode() in text


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "user", "content": "x", "cache_control": {"type": "persistent"}}]},
    {"messages": [{"role": "user", "content": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                                               {"type": "text", "text": "b"}]}]},
])
def test_cache_control_validation(server, body) -> None:
    status, r = server.post("/v1/chat/completions", {**body, "max_tokens": 2})
    assert status == 400, r

"""End to end: `llm-mini-infer serve --runner cuda` on CPU (the torch reference kernels) with a tiny random-weight
Qwen3.5 that uses the real Qwen tokenizer. Greedy outputs are checked against transformers' own greedy generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from ..support.api_util import Predictor, Server, content_of
from ..support.fixtures import require_qwen
from ..support.tiny_qwen import make_tiny_model

KW = {"enable_thinking": False}
PROMPTS = ["Hello!", "Name three colours.", "What is 2 + 2?", "Write a haiku about the sea.",
           "Translate 'cat' into French.", "Count to five.", "Why is the sky blue?", "Say something nice."]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    qwen = require_qwen()
    model = make_tiny_model(tmp_path_factory.mktemp("tiny-qwen"), vocab=248320, tie=True, seed=5, scale=0.2,
                            tokenizer_from=qwen)
    srv = Server(model, tmp_path_factory.mktemp("srv"), extra={
        "runner": {"kind": "cuda"}, "dev": {"runner": {"cuda": {"device": "cpu", "memory_gb": 0.05, "check": True}}},
        "scheduler": {"max_num_seqs": 4, "max_num_batched_tokens": 64}})
    srv.start(timeout=120)
    srv.pred = Predictor(model, vocab=248320)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def hf_greedy(model_dir, prompt: list[int], n: int) -> list[int]:
    from transformers import Qwen3_5ForCausalLM

    m = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    ids = torch.tensor([prompt])
    with torch.no_grad():  # an explicit mask: generate() would otherwise treat pad_token_id tokens as padding
        out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                         eos_token_id=None, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def chat(text: str, **kw) -> dict:
    return {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": KW, **kw}


def test_greedy_matches_transformers(tiny) -> None:
    for text in PROMPTS[:2]:
        prompt = tiny.pred.prompt([{"role": "user", "content": text}], **KW)
        want = tiny.pred.text(prompt, hf_greedy(tiny.model_dir, prompt, 12))[0]
        status, r = tiny.post("/v1/chat/completions", chat(text, max_tokens=12, temperature=0, ignore_eos=True))
        assert status == 200, r
        assert r["choices"][0]["message"]["content"] == want
        assert r["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": 12,
                              "total_tokens": len(prompt) + 12, "prompt_tokens_details": {"cached_tokens": 0},
                              "completion_tokens_details": {"reasoning_tokens": 0}}  # thinking off
        events = tiny.stream(chat(text, max_tokens=12, temperature=0, ignore_eos=True))
        assert content_of(events) == want and events[-1] == "[DONE]"


def test_seeded_sampling_is_reproducible_and_concurrent_requests_match_solo(tiny) -> None:
    body = chat(PROMPTS[3], max_tokens=10, temperature=0.8, top_p=0.9, seed=1234, ignore_eos=True)
    first = tiny.post("/v1/chat/completions", body)[1]["choices"][0]["message"]["content"]
    assert tiny.post("/v1/chat/completions", body)[1]["choices"][0]["message"]["content"] == first

    def one(i: int) -> str:
        return content_of(tiny.stream(chat(PROMPTS[i], max_tokens=8, temperature=0.7, seed=99 + i,
                                           ignore_eos=True)))

    solo = [one(i) for i in range(len(PROMPTS))]
    with ThreadPoolExecutor(len(PROMPTS)) as ex:  # 8 at once over 4 slots: queueing, chunked prefill, batching
        assert list(ex.map(one, range(len(PROMPTS)))) == solo

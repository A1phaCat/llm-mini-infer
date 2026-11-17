#!/usr/bin/env python3
"""The chat-template corpus: seeded chat-template cases rendered and tokenized by Hugging Face transformers
(tests/scripts/fetch_fixtures.py writes one per model into tests/.fixtures/corpus/).

  uv run python tests/scripts/template_corpus.py --model-dir tests/.fixtures/models/qwen3.8-27b --n 500 \\
      --out qwen3.8-27b.jsonl

Each JSONL line: {"id", "context": {messages, tools?, add_generation_prompt, ...kwargs},
                  "expected": {"text", "ids"} | {"error"}}
The context holds HF-format messages (tool-call arguments are mappings, not JSON strings): converting
OpenAI requests into this shape is the frontend's job and is tested separately.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, TokenizersBackend

TEXTS = [
    "Hello!", "What's the weather like in Paris today?", "Classify this item.",
    "Explain the difference between TCP and UDP in two sentences.",
    "  leading and trailing spaces  ", "\n\nnewlines around\n\n", "tabs\tand\tmore\ttabs",
    "Quotes \"double\" and 'single' and `back`", "Backslash \\ and \\n literal", "Emoji 🚀🔥👍🏽 and ZWJ 👩‍💻",
    "中文测试：今天天气怎么样？", "日本語のテキストです。", "한국어 문장입니다.", "العربية من اليمين إلى اليسار",
    "Ünïcödé àccents and combining é", "Math: ∑_{i=1}^{n} x_i² ≤ ∞", "HTML <b>bold</b> & <i>it</i> &amp;",
    "Code:\n```python\ndef f(x):\n    return x * 2\n```", '{"json": "in text", "n": 1}',
    "Special tokens <|im_end|> and <|im_start|>user inside text", "<think>fake think</think>",
    "<tool_call>\n<function=x>\n</function>\n</tool_call>", "<tool_response>looks like a tool response</tool_response>",
    "Numbers 0 -1 3.14159 1e10 007", "", " ", " non-breaking space", "line1\r\nline2\rline3",
    "A" * 300, ("The quick brown fox jumps over the lazy dog. " * 40).strip(),
]
REASONING = ["", "Let me think step by step.\nFirst, consider the input.", "  padded reasoning  ",
             "推理过程：先分析，再回答。", "<think> nested? </think>"]
ARG_VALUES = ["Paris", "multi\nline\nvalue", "unicode ✓ 中文", 42, -7, 3.5, 1.0, 1e-05, 1e16, 0.1, True, False,
              None, [1, 2, 3], ["a", "b"], {"nested": {"k": [1, {"x": None}]}}, "", "quote \" and \\ backslash",
              "<html>&</html>", "\u0001control\u001fchars"]
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search the web <fast> & cheap — returns JSON.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "filters": {"type": "object", "properties": {"lang": {"type": "string"}, "safe": {"type": "boolean"}}}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "classify_item", "description": "Assign one label to an item. 日本語の説明。",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "enum": ["a", "b", "c"]}, "confidence": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["label"]}}},
    {"type": "function", "function": {"name": "no_params", "description": "",
                                      "parameters": {"type": "object", "properties": {}}}},
]
VISION = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
          {"type": "image", "image": "file:///tmp/x.jpg"}, {"type": "video", "video": "file:///tmp/v.mp4"}]


def text(rng: random.Random) -> str:
    return rng.choice(TEXTS)


def content(rng: random.Random, allow_vision: bool) -> Any:
    r = rng.random()
    if r < 0.6 or (not allow_vision and r < 0.8):
        return text(rng)
    if r < 0.8 and not allow_vision:
        return [{"type": "text", "text": text(rng)} for _ in range(rng.randint(1, 3))]
    parts: list[dict[str, Any]] = []
    for _ in range(rng.randint(1, 4)):
        if allow_vision and rng.random() < 0.5:
            parts.append(dict(rng.choice(VISION)))
        else:
            parts.append({"type": "text", "text": text(rng)})
    return parts


def tool_call(rng: random.Random) -> dict[str, Any]:
    fn = rng.choice(TOOLS)["function"]["name"]
    keys = rng.sample(["city", "query", "limit", "label", "data", "x"], rng.randint(0, 3))
    args = {k: rng.choice(ARG_VALUES) for k in keys}
    call: dict[str, Any] = {"name": fn, "arguments": args}
    return {"type": "function", "id": f"call_{rng.randrange(10**6)}", "function": call} if rng.random() < 0.8 else call


def case(rng: random.Random) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if rng.random() < 0.6:
        sys = text(rng) if rng.random() < 0.8 else [{"type": "text", "text": text(rng)}]
        messages.append({"role": "system", "content": sys})
    for turn in range(rng.randint(1, 4)):
        messages.append({"role": "user", "content": content(rng, allow_vision=True)})
        last_turn = turn == 0 and rng.random() < 0.3
        if last_turn:
            break
        if rng.random() < 0.85:
            msg: dict[str, Any] = {"role": "assistant", "content": text(rng) if rng.random() < 0.85 else None}
            if rng.random() < 0.4:
                msg["reasoning_content"] = rng.choice(REASONING)
            if rng.random() < 0.35:
                msg["tool_calls"] = [tool_call(rng) for _ in range(rng.randint(1, 3))]
                messages.append(msg)
                for _ in range(rng.randint(1, 3)):
                    messages.append({"role": "tool", "content": text(rng)})
                if rng.random() < 0.5:
                    messages.append({"role": "assistant", "content": text(rng)})
            else:
                messages.append(msg)
    if messages[-1]["role"] == "assistant" and rng.random() < 0.7:
        messages.append({"role": "user", "content": content(rng, allow_vision=True)})

    ctx: dict[str, Any] = {"messages": messages, "add_generation_prompt": rng.random() < 0.85}
    if rng.random() < 0.4:
        ctx["tools"] = rng.sample(TOOLS, rng.randint(1, len(TOOLS)))
    for key, values in (("enable_thinking", [True, False]), ("reasoning_effort", ["xhigh", "medium", "low"]),
                        ("preserve_thinking", [True, False]), ("add_vision_id", [True, False])):
        if rng.random() < 0.4:
            ctx[key] = rng.choice(values)
    # A few deliberately invalid conversations: the template must raise in both implementations.
    r = rng.random()
    if r < 0.02:
        messages.append({"role": "system", "content": "late system message"})
    elif r < 0.03:
        ctx["reasoning_effort"] = "extreme"
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    # Reference token ids come from tokenizer.json as shipped (the generic fast backend). transformers'
    # class-based loading can rebuild the pipeline from a wrong hub tokenizer_class (Step-3.5 in 5.18:
    # Llama/Metaspace instead of the shipped byte-level BPE).
    generic = TokenizersBackend.from_pretrained(str(args.model_dir))
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors = auto_differs = 0
    with args.out.open("w") as f:
        for i in range(args.n):
            ctx = case(rng)
            kwargs = {k: v for k, v in ctx.items() if k not in ("messages", "tools", "add_generation_prompt")}
            try:
                text_out = tok.apply_chat_template(ctx["messages"], tools=ctx.get("tools"), tokenize=False,
                                                   add_generation_prompt=ctx["add_generation_prompt"], **kwargs)
                ids = generic.encode(text_out, add_special_tokens=False)
                expected: dict[str, Any] = {"text": text_out, "ids": ids}
                auto_ids = tok.encode(text_out, add_special_tokens=False)
                if auto_ids != ids:
                    expected["ids_auto_differs"] = True
                    auto_differs += 1
            except Exception as exc:  # template raise_exception and friends
                expected = {"error": f"{type(exc).__name__}: {exc}"}
                errors += 1
            f.write(json.dumps({"id": i, "context": ctx, "expected": expected}, ensure_ascii=False) + "\n")
    import tokenizers
    import transformers
    print(f"wrote {args.n} cases ({errors} expected errors, {auto_differs} where AutoTokenizer != tokenizer.json) to "
          f"{args.out} [transformers {transformers.__version__}, tokenizers {tokenizers.__version__}]")


if __name__ == "__main__":
    main()

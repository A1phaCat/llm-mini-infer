# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/entrypoints/chat_utils.py and vllm/renderers/hf.py of vLLM v0.30.0
# (https://github.com/vllm-project/vllm), and changed for llm-mini-infer.
"""OpenAI chat messages → the conversation a chat template expects, following vLLM's rules.

From vLLM's `entrypoints/chat_utils.py` (`_parse_chat_message_content`, `_postprocess_messages`; later releases move
them to `renderers/chat_utils.py`):
- content becomes text parts (a refusal part's text too); "openai"-format templates get
  `[{"type": "text", "text": ...}]` (a plain string becomes one part), "string"-format templates get the texts joined
  with "\n";
- tool messages get a string, unless an "openai"-format template gets one with an image: its parts stay as they are;
- assistant `reasoning` is passed as both `reasoning` and `reasoning_content`; tool-call `arguments` JSON
  strings become objects, and an empty `tool_calls` list is dropped;
- `developer` becomes `system` when the template has no developer role, and then the system messages become one at
  the start;
- `image_url` parts become `{"type": "image"}` parts in place for "openai"-format templates; for "string"-format
  ones, a placeholder for each image the text does not already hold is prepended ("\n"-joined). The API rejects the
  other media parts (`api.protocol`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .content_format import ContentFormat

log = logging.getLogger(__name__)


def _parse_arguments(args: Any) -> dict[str, Any]:
    if not args:
        return {}
    if isinstance(args, dict):
        return args
    parsed: Any = args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            log.warning("a tool call's arguments are not valid JSON; using {}")  # its name and arguments: content
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _parts(content: Any) -> list[dict[str, Any]]:
    """Text and image parts in order, as the template sees them."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content:
        kind = part.get("type")
        if kind in ("text", "refusal"):
            out.append({"type": "text", "text": part[kind]})
        elif kind == "image_url":
            out.append({"type": "image"})
    return out


def to_template_messages(messages: list[dict[str, Any]], content_format: ContentFormat,
                         supports_developer: bool, image_placeholder: str = "",
                         where: list[int] | None = None) -> list[dict[str, Any]]:
    """`where`, when given, receives each message's index in the result: a cache hint names a message of the request,
    and a merged system message (`_one_system`) moves the others."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        parts = _parts(msg.get("content"))
        texts = [p["text"] for p in parts if p["type"] == "text"]
        n_images = len(parts) - len(texts)
        # for an "openai"-format template, a tool message becomes text unless one of its parts is not text
        if content_format == "string" or (role == "tool" and not n_images):
            text = "\n".join(texts)
            if n_images and image_placeholder:
                missing = n_images - text.count(image_placeholder)
                if missing > 0:
                    text = "\n".join([image_placeholder] * missing + [text])
            content: Any = text
        else:
            content = parts
        conv: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                calls = []
                for call in tool_calls:
                    call = dict(call)
                    fn = dict(call.get("function") or {})
                    fn["arguments"] = _parse_arguments(fn.get("arguments"))
                    call["function"] = fn
                    calls.append(call)
                conv["tool_calls"] = calls
            reasoning = msg.get("reasoning", msg.get("reasoning_content"))
            if reasoning is not None:
                conv["reasoning"] = conv["reasoning_content"] = reasoning
        elif role == "tool" and msg.get("tool_call_id") is not None:
            conv["tool_call_id"] = msg["tool_call_id"]
        if isinstance(msg.get("name"), str):
            conv["name"] = msg["name"]
        if role == "developer" and not supports_developer:
            conv["role"] = "system"
        out.append(conv)
    if where is not None:
        where[:] = range(len(out))
    if not supports_developer and any(m["role"] == "developer" for m in messages):
        return _one_system(out, where)
    return out


def _one_system(conv: list[dict[str, Any]], where: list[int] | None = None) -> list[dict[str, Any]]:
    """After developer messages became system ones: when a system message is not first or there are several, one
    system message at the start with their texts (a blank line between), then the others in order. Some templates need
    the system message first. `where`: each message's index after it."""
    texts: list[str] = []
    rest: list[dict[str, Any]] = []
    merge = False
    moved = []
    for i, m in enumerate(conv):
        if m["role"] != "system":
            moved.append(1 + len(rest))
            rest.append(m)
            continue
        moved.append(0)
        merge = merge or i > 0 or bool(texts)
        c = m["content"]
        text = c if isinstance(c, str) else "\n".join(p["text"] for p in c if p["type"] == "text")
        if text:
            texts.append(text)
    if not merge:
        return conv
    if where is not None:
        where[:] = moved
    return [{"role": "system", "content": "\n\n".join(texts)}, *rest]

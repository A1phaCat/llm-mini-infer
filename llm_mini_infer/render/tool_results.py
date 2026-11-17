"""Tool results neutralized (architecture.md, "Tool calls"): inside a tool message's own text, a control token spelled
as text becomes the ordinary tokens of that text, never the control token. Everywhere else (system, user and
assistant text) a spelled control token still becomes the token, as transformers' tokenization makes it.

- **The span** of each tool message's text is located from the render itself, never by searching for the text (which
  a user's text can hold too): the conversation is rendered once more with each non-whitespace character of the tool
  messages' text parts swapped for one character the prompt does not hold (`swap`), so a template's whitespace
  handling (Qwen's `trim`) acts the same, and the two renders are compared (`spans`). What the template or the server
  puts in or beside the text (Gemma's `<|"|>` and `<|tool_response>`, the image parts' placeholders) is identical in
  both, so it stays outside. When the renders differ anywhere else (a template whose output depends on the text),
  nothing is neutralized, with a warning.
- **The tokens** (`Neutral.encode`): the prompt is tokenized with offsets; each added token of the vocabulary whose
  characters lie in a span is replaced by the tokens of its spelling under the same tokenizer without its added tokens
  (built with the `Neutral`, at the server's start). Encoding with special tokens split does not do it: Qwen's
  `<tool_call>`, `<tool_response>` and `<think>` are added tokens that are not special. Only the prompt's added tokens
  are placed, each by bisection in the sorted spans, so the cost is the tokenization's. A request without a tool
  message keeps the segment-cached path (`Renderer.encode`), and a tool result that spells no added token gets the same
  ids as there.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
from typing import Any

import numpy as np
from tokenizers import Tokenizer

log = logging.getLogger(__name__)
_NON_SPACE = re.compile(r"\S")
# The private use areas: the BMP's, then the two supplementary planes' (131,068 more code points)
_PRIVATE_USE = (range(0xE000, 0xF900), range(0xF0000, 0xFFFFE), range(0x100000, 0x10FFFE))


class NoSentinel(ValueError):
    """The prompt holds every character that could mark the tool results (the API's 400)."""


def has_tool_text(messages: list[dict[str, Any]]) -> bool:
    """Whether a message of the request (builtins) is a tool message with text."""
    for m in messages:
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str) and c:
                return True
            if isinstance(c, list) and any(p.get("type") in ("text", "refusal") and p.get(p["type"]) for p in c):
                return True
    return False


def sentinel(text: str) -> str:
    """A character the text does not hold: the private use areas' first such, else the first code point that is
    neither a surrogate nor whitespace; `NoSentinel` when the text holds all 1.1 million."""
    present = set(text)
    for area in _PRIVATE_USE:
        for cp in area:
            ch = chr(cp)
            if ch not in present:
                return ch
    for cp in range(0x21, 0x110000):
        ch = chr(cp)
        if not 0xD800 <= cp < 0xE000 and ch not in present and not ch.isspace():
            return ch
    raise NoSentinel("the prompt holds every character the server could mark its tool results with")


def swap(messages: list[dict[str, Any]], mark: str) -> list[dict[str, Any]]:
    """The request's messages (builtins) with each non-whitespace character of the tool messages' text parts (text and
    refusal parts, or a string content) replaced by `mark`."""
    out = []
    for m in messages:
        if m.get("role") != "tool":
            out.append(m)
            continue
        m = dict(m)
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = _NON_SPACE.sub(mark, c)
        elif isinstance(c, list):
            parts = []
            for p in c:
                kind = p.get("type")
                if kind in ("text", "refusal") and isinstance(p.get(kind), str):
                    p = {**p, kind: _NON_SPACE.sub(mark, p[kind])}
                parts.append(p)
            m["content"] = parts
        out.append(m)
    return out


def spans(text: str, swapped: str, mark: str) -> list[tuple[int, int]] | None:
    """The character ranges of the tool messages' text in `text`, from its render with the text swapped (`swap`): the
    runs of `mark` in `swapped`, whitespace between two of them included. None when the renders differ elsewhere."""
    if len(text) != len(swapped):
        return None
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        j = swapped.find(mark, i)
        if j < 0:
            if text[i:] != swapped[i:]:
                return None
            break
        if text[i:j] != swapped[i:j]:
            return None
        k = j
        while k < n and (swapped[k] == mark or (swapped[k].isspace() and swapped[k] == text[k])):
            k += 1
        while swapped[k - 1] != mark:  # trailing whitespace belongs to what follows
            k -= 1
        out.append((j, k))
        i = k
    return out


class Neutral:
    """Per tokenizer: the added tokens' ids, and the ordinary tokens of their spellings."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.added = frozenset(tokenizer.get_added_tokens_decoder())
        self._added_ids = np.fromiter(sorted(self.added), np.int64, len(self.added))
        self._memo: dict[str, tuple[list[int], list[str]]] = {}
        d = json.loads(tokenizer.to_str())  # the tokenizer without its added tokens (about 0.5 s: built here, once)
        d["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(d))

    def ordinary(self, spelling: str) -> tuple[list[int], list[str]]:
        got = self._memo.get(spelling)
        if got is None:
            enc = self.plain.encode(spelling, add_special_tokens=False)
            if self.added.intersection(enc.ids):  # a vocabulary whose merges rebuild the token: one per character
                ids, toks = [], []
                for ch in spelling:
                    e = self.plain.encode(ch, add_special_tokens=False)
                    ids += e.ids
                    toks += e.tokens
                got = (ids, toks)
            else:
                got = (list(enc.ids), list(enc.tokens))
            self._memo[spelling] = got
        return got

    def encode(self, text: str, where: list[tuple[int, int]], add_special_tokens: bool = False
               ) -> tuple[np.ndarray, list[str]]:
        """The ids (and token strings) of `text`, each added token inside a range of `where` replaced (module
        docstring)."""
        enc = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids, toks = enc.ids, enc.tokens
        arr = np.asarray(ids, np.int32)
        hits = np.flatnonzero(np.isin(arr, self._added_ids)).tolist() if where else []
        if not hits:
            return arr, list(toks)
        los, his = [lo for lo, _ in where], [hi for _, hi in where]  # sorted and disjoint (`spans`)
        offsets = enc.offsets
        out_ids: list[int] = []
        out_toks: list[str] = []
        done = 0
        for i in hits:
            a, b = offsets[i]
            j = bisect.bisect_right(his, a)  # the first span ending after the token's start
            if b > a and j < len(los) and los[j] < b:
                o_ids, o_toks = self.ordinary(text[a:b])
                out_ids += ids[done:i] + o_ids
                out_toks += toks[done:i] + o_toks
                done = i + 1
        if not done:
            return arr, list(toks)
        out_ids += ids[done:]
        out_toks += toks[done:]
        return np.asarray(out_ids, np.int32), out_toks


def locate(messages: list[dict[str, Any]], text: str, render: Any) -> list[tuple[int, int]]:
    """The spans of the tool messages' text in the prompt `text`; `render(messages)` renders request messages
    (builtins) as the prompt was rendered. [] when they cannot be located (logged)."""
    mark = sentinel(text)
    got = spans(text, render(swap(messages, mark)), mark)
    if got is None:
        log.warning("tool results not neutralized: the chat template renders their text in a way that depends on it")
        return []
    return got

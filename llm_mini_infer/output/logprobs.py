"""Chat completions' `logprobs` as vLLM writes them (docs/architecture.md, "Sampling"): per output token an entry
`{"token", "logprob", "bytes", "top_logprobs"}`, `top_logprobs` entries `{"token", "logprob", "bytes"}`.

- **token**: the id decoded alone, special tokens kept (with a Metaspace pre-tokenizer, the leading spaces the decode
  strips put back). One that ends in U+FFFD, a piece of a character that byte-fallback split, is decoded again after
  the request's last 1 to 4 output tokens, and keeps what it adds past them: "" while the character is still open.
  `bytes`: its UTF-8.
- **top_logprobs**: the sampled token first, then the top tokens without it, `top_logprobs` entries in all (the sampled
  token and the top-k, deduplicated in insertion order); none for `top_logprobs` 0.
- **logprob**: at least −9999.0 (a token the grammar refuses has −inf). Floats as vLLM writes them: the
  non-streamed body Python's `json` (`repr`: `1e-05`), the stream pydantic-core's, which msgspec's equal (`0.00001`).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import msgspec
from tokenizers import Tokenizer

FLOOR = -9999.0  # vLLM's clamp of a logprob (its -inf)
CONTEXT = 4  # the output tokens a byte-fallback piece is decoded after, at most (vLLM's)
_REPLACEMENT = "�"
_enc = msgspec.json.encode


def repr_float(v: float) -> bytes:
    """A float as Python's `json` writes it (the non-streamed body)."""
    return float.__repr__(v).encode()


def _metaspace(tokenizer: Tokenizer) -> str | None:
    """The Metaspace pre-tokenizer's replacement character, if the tokenizer has one."""
    pre = json.loads(tokenizer.to_str()).get("pre_tokenizer") or {}
    subs = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    for sub in subs:
        if sub.get("type") == "Metaspace":
            return sub.get("replacement", "▁")
    return None


class TokenText:
    """A frontend's token strings (module docstring): each id decoded alone once, then from a cache."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.marker = _metaspace(tokenizer)
        self._alone: dict[int, str] = {}

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def alone(self, t: int) -> str:
        s = self._alone.get(t)
        if s is None:
            s = self.decode([t]) or ""
            if self.marker:  # put back the leading spaces the decode strips
                raw = self.tok.id_to_token(t) or ""
                lead = len(raw) - len(raw.lstrip(self.marker))
                missing = lead - (len(s) - len(s.lstrip(" ")))
                s = " " * missing + s if lead and missing > 0 else s
            self._alone[t] = s
        return s

    def text(self, t: int, context: list[int]) -> str:
        """`t`'s string after the request's output tokens `context` (the last CONTEXT, oldest first)."""
        s = self.alone(t)
        return self._corrected(t, context) if s.endswith(_REPLACEMENT) else s

    def _corrected(self, t: int, context: list[int]) -> str:
        for n in range(1, min(len(context), CONTEXT) + 1):
            ctx = context[-n:]
            full = self.decode([*ctx, t])
            if full.endswith(_REPLACEMENT):
                continue
            clean = len(ctx)  # the context tokens before the ones that are pieces of this character
            while clean and self.decode([ctx[clean - 1]]).endswith(_REPLACEMENT):
                clean -= 1
            prefix = self.decode(ctx[:clean]) if clean else ""
            if full.startswith(prefix):
                return full[len(prefix):]
            common = 0
            for a, b in zip(prefix, full):
                if a != b:
                    break
                common += 1
            return full[common:]
        return ""


def _item(token: str, logprob: float, fmt: Callable[[float], bytes]) -> bytes:
    return (b'{"token":' + _enc(token) + b',"logprob":' + fmt(max(logprob, FLOOR)) + b',"bytes":'
            + _enc(list(token.encode("utf-8", errors="replace"))))


class RequestLogprobs:
    """One request's entries: `k` top entries each; `stream` picks the stream's float format."""

    __slots__ = ("context", "fmt", "k", "text")

    def __init__(self, text: TokenText, k: int, stream: bool):
        self.text, self.k = text, k
        self.fmt = _enc if stream else repr_float
        self.context: list[int] = []  # the last CONTEXT output tokens

    def entries(self, tokens: list[int], token_lp: list[float], top_ids: list[list[int]],
                top: list[list[float]]) -> list[bytes]:
        """The JSON entries of these output tokens (in order), each with its logprob and the top entries the engine
        sent (at least `k`, in descending order)."""
        out = []
        k, fmt, ctx = self.k, self.fmt, self.context
        for t, lp, ids, vals in zip(tokens, token_lp, top_ids, top):
            s = self.text.text(t, ctx)
            tops = []
            if k:
                tops.append(_item(s, lp, fmt) + b"}")
                for i, v in zip(ids[:k], vals[:k]):
                    if i != t and len(tops) < k:
                        tops.append(_item(self.text.text(i, ctx), v, fmt) + b"}")
            out.append(_item(s, lp, fmt) + b',"top_logprobs":[' + b",".join(tops) + b"]}")
            ctx.append(t)
            if len(ctx) > CONTEXT:
                del ctx[0]
        return out


def content(entries: list[bytes]) -> bytes:
    """The `logprobs` object of entries."""
    return b'{"content":[' + b",".join(entries) + b"]}"

"""Byte templates for streamed chat-completion chunks: a fixed prefix and suffix around the msgspec-encoded text
(about 0.1 µs per chunk). One SSE event per engine step.

As in the OpenAI API, a stream opens with a role-only chunk (`{"role": "assistant", "content": ""}`), written together
with the first step's text when that arrives; the chunks after it carry `content` (and, for a parsed reply, its
`reasoning` and `tool_calls`), and the step's tokens' `logprobs` for a request that asked (`output.logprobs`; null
otherwise). Every chunk carries the engine marker in
`system_fingerprint` (`llm_mini_infer.version`), as OpenAI's chunks carry theirs. With `return_token_ids` the role chunk
carries the prompt's `prompt_token_ids` and each chunk's choice its step's `token_ids`."""

from __future__ import annotations

import msgspec

_enc = msgspec.json.Encoder()
encode = _enc.encode

DONE = b"data: [DONE]\n\n"
_SUFFIX_OPEN = b',"logprobs":null,"finish_reason":null}]}\n\n'
_SUFFIX_LP = b',"finish_reason":null}]}\n\n'


class ChunkTemplate:
    """Per request: `event(text)` builds one `data: {...}` line for a content delta."""

    __slots__ = ("head", "prefix")

    def __init__(self, request_id: str, created: int, model: str, fingerprint: str | None = None):
        self.head = b'{"id":' + encode(request_id) + b',"object":"chat.completion.chunk","created":' + \
            str(created).encode() + b',"model":' + encode(model) + b',"system_fingerprint":' + encode(fingerprint)
        self.prefix = b"data: " + self.head + b',"choices":[{"index":0,"delta":'

    def role(self, prompt_ids: list[int] | None = None) -> bytes:
        """The opening role-only chunk; `prompt_ids`: return_token_ids' prompt ids."""
        if prompt_ids is None:
            return self.prefix + b'{"role":"assistant","content":""}' + _SUFFIX_OPEN
        return (self.prefix + b'{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],'
                b'"prompt_token_ids":' + encode(prompt_ids) + b"}\n\n")

    def event(self, text: str, logprobs: bytes | None = None, token_ids: list[int] | None = None) -> bytes:
        """`logprobs`: the step's tokens' `logprobs` object (`output.logprobs.content`); None: null. `token_ids`: the
        step's (return_token_ids)."""
        if token_ids is not None:
            return (self.prefix + b'{"content":' + encode(text) + b'},"logprobs":' + (logprobs or b"null")
                    + b',"finish_reason":null,"token_ids":' + encode(token_ids) + b"}]}\n\n")
        if logprobs is None:
            return self.prefix + b'{"content":' + encode(text) + b"}" + _SUFFIX_OPEN
        return self.prefix + b'{"content":' + encode(text) + b'},"logprobs":' + logprobs + _SUFFIX_LP

    def final(self, text: str, finish_reason: str, stop_reason: str | int | None,
              logprobs: bytes | None = None, token_ids: list[int] | None = None) -> bytes:
        tail = b"" if token_ids is None else b',"token_ids":' + encode(token_ids)
        return (self.prefix + b'{"content":' + encode(text) + b'},"logprobs":' + (logprobs or b"null")
                + b',"finish_reason":' + encode(finish_reason) + b',"stop_reason":' + encode(stop_reason) + tail
                + b"}]}\n\n")

    def delta(self, delta: bytes, logprobs: bytes | None = None, token_ids: list[int] | None = None) -> bytes:
        """A chunk of a request with tools: `delta` the delta object's JSON (`output.tool_calls.delta_json`)."""
        tail = b"" if token_ids is None else b',"token_ids":' + encode(token_ids)
        return (self.prefix + delta + b',"logprobs":' + (logprobs or b"null") + b',"finish_reason":null' + tail
                + b"}]}\n\n")

    def final_delta(self, delta: bytes, finish_reason: str, stop_reason: str | int | None,
                    logprobs: bytes | None = None, token_ids: list[int] | None = None) -> bytes:
        tail = b"" if token_ids is None else b',"token_ids":' + encode(token_ids)
        return (self.prefix + delta + b',"logprobs":' + (logprobs or b"null") + b',"finish_reason":'
                + encode(finish_reason) + b',"stop_reason":' + encode(stop_reason) + tail + b"}]}\n\n")

    def usage(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
              reasoning_tokens: int | None = None) -> bytes:
        """The final usage chunk; `reasoning_tokens`: with a reasoning parser, its count (vLLM's
        `completion_tokens_details`, which its stream leaves out without one)."""
        details = b"" if reasoning_tokens is None else \
            b',"completion_tokens_details":{"reasoning_tokens":' + str(reasoning_tokens).encode() + b"}"
        return (b"data: " + self.head + b',"choices":[],"usage":{"prompt_tokens":' + str(prompt_tokens).encode()
                + b',"total_tokens":' + str(prompt_tokens + completion_tokens).encode() + b',"completion_tokens":'
                + str(completion_tokens).encode() + b',"prompt_tokens_details":{"cached_tokens":'
                + str(cached_tokens).encode() + b"}" + details + b"}}\n\n")

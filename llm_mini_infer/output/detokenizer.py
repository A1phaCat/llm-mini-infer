# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/v1/engine/detokenizer.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm), and changed for
# llm-mini-infer.
"""Streaming detokenization with `tokenizers.decoders.DecodeStream` (about 1 µs per token).

The stream is primed with the end of the prompt: vLLM's `FastIncrementalDetokenizer` primes it with the whole prompt
(`v1/engine/detokenizer.py`), so that the first generated token decodes with the right context (leading spaces,
byte-fallback sequences). Not the whole prompt is passed here: `DecodeStream` decodes its whole priming prefix on the
first step (217 µs for a 1,400-token prompt, on the TTFT path) and keeps only a short window after that anyway. The
prime is the shortest suffix of `PRIME_MIN_TOKENS` to `PRIME_MAX_TOKENS` prompt tokens that decodes without U+FFFD, and
the whole prompt when none does. The emitted text is the whole prompt's (tests/api/test_output.py checks all five
models) except (i) after an open character followed by tokens that decode to nothing (special tokens skipped), where
the whole prompt's first piece repeats the open character and this one does not, and (ii) when the prompt's token ids
hold invalid UTF-8 before a clean tail, which no text or rendered chat prompt produces. A prompt that ends inside a
multi-byte character needs the fallback: every suffix ends in U+FFFD, and the first step emits prompt text, as the
whole prompt's does but not as a fixed window's. A rendered chat prompt ends with its template's tokens, on whole
characters.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

log = logging.getLogger(__name__)

# tokenizers' error when the stream's state breaks on non-monotonic UTF-8 (vLLM issue #17448).
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"
PRIME_MIN_TOKENS = 4
PRIME_MAX_TOKENS = 23


def prime_ids(tokenizer: Tokenizer, prompt_ids: Sequence[int], skip_special_tokens: bool) -> Sequence[int]:
    """The shortest suffix of 4 to 23 prompt tokens that decodes without U+FFFD, else the whole prompt."""
    if len(prompt_ids) <= PRIME_MIN_TOKENS:
        return prompt_ids
    for n in range(PRIME_MIN_TOKENS, min(len(prompt_ids), PRIME_MAX_TOKENS) + 1):
        suffix = prompt_ids[-n:]
        if "\ufffd" not in tokenizer.decode(list(suffix), skip_special_tokens=skip_special_tokens):
            return suffix
    return prompt_ids


class Detokenizer:
    __slots__ = ("skip_special_tokens", "stream", "tokenizer")

    def __init__(self, tokenizer: Tokenizer, prompt_ids: Sequence[int] | None = None, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        prime = None if prompt_ids is None else prime_ids(tokenizer, prompt_ids, skip_special_tokens)
        self.stream = DecodeStream(ids=prime, skip_special_tokens=skip_special_tokens)

    def step(self, token_id: int) -> str:
        """Text completed by this token ("" while a multi-byte sequence is still open)."""
        try:
            piece = self.stream.step(self.tokenizer, token_id)
        except (OverflowError, TypeError):
            log.exception("invalid token id while detokenizing")
            return ""
        except Exception as exc:
            if not str(exc).startswith(INVALID_PREFIX_ERR_MSG):
                raise
            # Recovery: restart the stream without context.
            log.warning("invalid prefix while detokenizing; resetting the decode stream")
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
            piece = self.stream.step(self.tokenizer, token_id)
        return piece or ""

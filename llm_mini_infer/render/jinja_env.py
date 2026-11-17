# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Team. All rights reserved.
# Adapted from src/transformers/utils/chat_template_utils.py of transformers 5.18.0
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""Hugging Face's chat-template Jinja environment, vendored so the serving path never imports transformers.

Vendored from transformers' `utils/chat_template_utils.py` (`_cached_compile_jinja_template`, Apache-2.0).
Rendering through the same environment is what makes our prompts byte-identical to `apply_chat_template` on every
case of the chat-template corpus (`tests/scripts/template_corpus.py`). Keep it in sync with transformers when the corpus
is regenerated:
  - `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with `loopcontrols`;
  - `tojson` = `json.dumps` with `ensure_ascii=False` (Jinja's own escapes HTML characters);
  - `raise_exception` raising `TemplateError`, and `strftime_now`;
  - `{% generation %}` blocks. HF uses them to compute assistant masks for training; serving only needs the
    passthrough, so the tracker is dropped.

Ours alone: a render counts its operations (attribute and item reads, calls) against a budget (`operations`), since a
template's cost grows with the conversation and the frontend renders on its event loop. Gemma 4's template is
quadratic in messages: 817 empty messages (a 4,096-token context) take 366k operations; the corpus's largest case
takes 841.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import lru_cache
from typing import Any

import jinja2
import jinja2.exceptions
import jinja2.ext
import jinja2.nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

# A request's template operations: about 1.7 s of rendering on a laptop core, the longest its rendering holds the
# event loop (a typical liveness probe restarts the container after about 30 s); the worst conversation of a
# 4,096-token context on Gemma 4 takes a third of it
MAX_OPERATIONS = 1 << 20
_left: ContextVar[list[int] | None] = ContextVar("llm_mini_infer_template_operations", default=None)  # [left, budget]


class OperationsExceeded(jinja2.exceptions.TemplateError):
    pass


@contextmanager
def operations(n: int) -> Iterator[None]:
    """Renders inside it run at most `n` operations in all; the next one raises `OperationsExceeded`."""
    token = _left.set([n, n])
    try:
        yield
    finally:
        _left.reset(token)


def metered() -> bool:
    """Inside `operations`."""
    return _left.get() is not None


def _spend() -> None:
    left = _left.get()
    if left is not None:
        left[0] -= 1
        if left[0] < 0:
            raise OperationsExceeded(f"rendering this conversation takes more than {left[1]:,} template operations, "
                                     "this server's bound on a request")


class _Metered(ImmutableSandboxedEnvironment):
    """The sandbox, each attribute and item read and each call counted against the budget in force (`operations`)."""

    def getattr(self, obj: Any, attribute: str) -> Any:
        _spend()
        return super().getattr(obj, attribute)

    def getitem(self, obj: Any, argument: Any) -> Any:
        _spend()
        return super().getitem(obj, argument)

    def call(__self, __context: Any, __obj: Any, *args: Any, **kwargs: Any) -> Any:  # jinja2's names
        _spend()
        return super().call(__context, __obj, *args, **kwargs)


class _GenerationPassthrough(jinja2.ext.Extension):
    """`{% generation %}...{% endgeneration %}` renders its body unchanged."""

    tags = {"generation"}  # noqa: RUF012 - jinja2's Extension API declares tags as a class attribute

    def parse(self, parser: Any) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_passthrough"), [], [], body).set_lineno(lineno)

    def _passthrough(self, caller: Any) -> str:
        return caller()


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x: Any, ensure_ascii: bool = False, indent: Any = None, separators: Any = None,
            sort_keys: bool = False) -> str:
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _strftime_now(format: str) -> str:
    return datetime.now().strftime(format)


def make_environment() -> ImmutableSandboxedEnvironment:
    env = _Metered(trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationPassthrough, jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env


@lru_cache(maxsize=16)
def compile_template(source: str) -> jinja2.Template:
    return make_environment().from_string(source)

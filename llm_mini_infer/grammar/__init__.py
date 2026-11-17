"""Structured output (docs/architecture.md, "Structured output"): grammars with llguidance.

- `spec`: in the frontends. `response_format` → the grammar text the engine compiles, checked (without a tokenizer)
  so that an invalid schema is a 400 that carries llguidance's error. Cached per schema.
- `engine`: in the engine process. The llguidance backend (its tokenizer built from `tokenizer.json`, without
  transformers), compiled grammars cached by their text and cloned per request, and the parallel mask fill into the
  runner's buffer.
"""

from __future__ import annotations


class GrammarError(ValueError):
    """A grammar that cannot be compiled (the frontend's 400, or the engine's rejection)."""


class NestingError(GrammarError):
    """A `response_format` schema nested past vLLM's JSON cap (`bounds.json_nesting`): vLLM's 400, which names no
    parameter."""

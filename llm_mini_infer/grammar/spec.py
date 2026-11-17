"""`response_format` → the grammar text the engine compiles (frontend side; architecture.md, "Structured output").

**Two languages**, by `structured_output.backend` (vLLM's `--structured-outputs-config` has the same `backend` and
`disable_any_whitespace` keys):
- **guidance** is llguidance's JSON-schema grammar, with or without `disable_any_whitespace`: llguidance's
  `grammar_from_json_schema` of the schema's stdlib `json.dumps`, with the defaults
  `{"whitespace_flexible": not disable_any_whitespace}` (`json_object`: the schema text `{"type": "object"}`).
  vLLM's guidance backend compiles the same text. That is llguidance's JSON form of a grammar (a
  `TopLevelGrammar`, not Lark), with its default canonical forcing: bytes the grammar fixes (`{"`, a key, the rest of
  an enum value) can only be their canonical tokens. The text does not depend on the schema's whitespace or string
  escapes (llguidance re-serializes it, keeping the key order); a schema's own `x-guidance` options win over the
  defaults, and one that is not an object is replaced by them. Compact (`disable_any_whitespace`): no whitespace
  anywhere. tests/structured/test_grammar_parity.py runs vLLM's own code at the release that `REV` names
  (tests/support/vllm_ref.py): the same text, tokenizer and masks, on the same llguidance. (That release pins an
  llguidance below 1.8, which differs in two places: it does not let the number after a `multipleOf` key start with a
  minus sign, and it takes an escaped slash and lone surrogates in the keys an object takes beside its named
  properties.)
- **auto** (the default) and **xgrammar** (the same here) emulate xgrammar's JSON language with
  `any_whitespace` (any whitespace between JSON tokens, none before the value or after it), which auto picks for these
  schemas. We compile it with llguidance as Lark: `%json` embeds the schema, its `x-guidance` options make whitespace
  flexible, and `%llguidance {"no_forcing": true}` lets the model choose how to tokenize bytes the grammar fixes, as
  xgrammar does. With canonical forcing the masks differ from xgrammar's wherever such bytes have non-canonical splits
  (a key's first letter as a token of its own, before the rest of the key), which xgrammar allows; with `no_forcing`
  they match xgrammar's bit for bit, and cost less. llguidance allows `-0` after a minus sign, which xgrammar 0.2.7 and
  0.2.8 do not. Not emulated: auto's fallback from xgrammar to guidance for schemas xgrammar cannot compile, and
  xgrammar's compact form (", " and ": " separators in vLLM; the same `-0` difference), which the
  configuration refuses.

**Checks.** A schema outside the subset and its caps (`grammar.bounds`) is refused before llguidance sees it. `check`
compiles the grammar over a small byte tokenizer (unknown or unsatisfiable keywords, bad `$ref`s and regexes are caught
here, and a grammar past llguidance's 65,524 symbols, so the client gets a 400 with llguidance's message). What needs
the model's tokenizer is checked by the engine's compile, which can still reject a request (an error finish) but almost
never does. Results are cached by the schema text, as the request carried it. A grammar with a thought is cached by
the family and the case too. Building one costs far more than a lookup, and a hit decodes nothing.

**Thinking.** Where the output may start with a thought (`output.reasoning`'s cases b and c), the grammar takes the
thought in front (`thought`). In case b it is Gemma's optional `<|channel>thought\n...<channel|>`. In case c it is the
rest of the thought the prompt opened, then its closer. Qwen's chat template follows the closer with a blank line, which
is optional here. Such a grammar is Lark under every backend, as llguidance's JSON form takes no prefix. Its first line
is the backend's (`GrammarSpecs.header`). The schema is a `%json` item with the server's whitespace
(`GrammarSpecs.json_item`), so the language after the thought is the thinking-off grammar's. Cases a and d take the
thinking-off grammar.

**Where.** With `threads` (the frontend's), a miss is built and checked on a compile thread with a large stack
(`grammar.threads`), off the event loop: the lookup raises `Pending`, which the request awaits (`resolve`, one build
per key however many requests wait for it) before it looks again. A hit returns at once, on the caller's thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import re
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import msgspec

from . import GrammarError, forced
from .bounds import BoundError, check_json_schema, json_nesting
from .threads import CompileThreads

BACKENDS = ("auto", "guidance", "xgrammar")
HEADER = '%llguidance {"no_forcing": true}\nstart: %json '  # auto/xgrammar: Lark, no forcing (module docstring)
WHITESPACE = {"whitespace_flexible": True}  # auto/xgrammar: any whitespace, as xgrammar's any_whitespace
OBJECT = '{"type": "object"}'  # json_object: the schema text guidance compiles
CACHE_SIZE = 256

BYTE_TOKENS, SPARE_IDS = 257, 16  # `check`'s tokenizer: 256 bytes and an end token, then ids for a grammar's tokens
MAX_SYMBOLS = (1 << 16) - 12  # llguidance's parser takes fewer symbols than u16::MAX - 10
_SYMBOLS = "panic: assertion failed: grammar.symbols.len() < u16::MAX"

_enc = msgspec.json.Encoder()
_AT = re.compile(r"^at \d+\(\d+\): ")
_TOKEN = re.compile(r"<\[\d+\]>")


def json_schema_grammar(schema: dict[str, Any], backend: str = "auto", disable_any_whitespace: bool = False) -> str:
    """The grammar text for a JSON schema (an object), in the backend's language. The schema's own `x-guidance`
    options win over ours, as with llguidance's `grammar_from_json_schema(defaults=...)`."""
    if not isinstance(schema, dict):
        raise GrammarError("the JSON schema must be an object")
    _bounded(schema)
    if _guidance(backend, disable_any_whitespace):
        return _guidance_text(json.dumps(schema), disable_any_whitespace)
    own = schema.get("x-guidance")
    if own is not None and not isinstance(own, dict):
        raise GrammarError("'x-guidance' must be an object")
    return HEADER + _enc.encode({**schema, "x-guidance": {**WHITESPACE, **(own or {})}}).decode()


def json_object_grammar(backend: str = "auto", disable_any_whitespace: bool = False) -> str:
    """`response_format: {"type": "json_object"}`: any JSON object."""
    if _guidance(backend, disable_any_whitespace):
        return _guidance_text(OBJECT, disable_any_whitespace)
    return json_schema_grammar({"type": "object"})


def _bounded(schema: dict) -> None:
    """A `response_format` schema's checks: vLLM's JSON nesting cap first (a `NestingError`), then the subset's."""
    json_nesting(schema)
    try:
        check_json_schema(schema)
        forced.check_json_schema(schema)
    except BoundError as exc:
        raise GrammarError(str(exc)) from exc
    except RecursionError:  # a schema nested past what the walks take (the subset refuses any it can reach)
        raise GrammarError("the JSON schema is nested too deeply") from None


def _guidance(backend: str, disable_any_whitespace: bool) -> bool:
    """Whether the backend is guidance; a ValueError for settings the configuration refuses."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown structured output backend {backend!r}")
    if disable_any_whitespace and backend != "guidance":
        raise ValueError("disable_any_whitespace needs the guidance backend")
    return backend == "guidance"


def _guidance_text(schema: str, disable_any_whitespace: bool) -> str:
    import llguidance

    return llguidance.LLMatcher.grammar_from_json_schema(
        schema, defaults={"whitespace_flexible": not disable_any_whitespace})


def thought(family: str, case: str, start: int, end: int) -> str:
    """A thinking-on output's thought, as Lark (module docstring, "Thinking"), by where the output starts
    (`output.reasoning.ReasoningFamily.case`). Case b: Gemma's optional `<|channel>thought\\n...<channel|>`. Case c:
    the rest of a thought the prompt opened, then its closer, and for Qwen an optional blank line (its chat
    template's)."""
    if case == "b":
        return f'(<[{start}]> "thought\\n" /(.|\\n)*/ <[{end}]>)?'
    if case == "c":
        return f'/(.|\\n)*/ <[{end}]>' + (' "\\n\\n"?' if family == "qwen3" else "")
    raise ValueError(f"no thought for case {case!r}")


def clean_error(err: str) -> str:
    """llguidance's message without the Lark position and source lines it adds to a `%json` grammar's errors; its
    assertion on the parser's size (a panic it catches) in words."""
    first = _AT.sub("", err.strip().split("\n", 1)[0])
    if first.startswith(_SYMBOLS):
        return f"the grammar needs more than {MAX_SYMBOLS:,} symbols, llguidance's limit"
    return first.removeprefix("failed to compile JSON schema: ") or err.strip()


def check(grammar: str, token_ids: Sequence[int] = ()) -> str:
    """'' if llguidance compiles the grammar, else its error. The compile builds a matcher over a small byte tokenizer
    (`_check_tokenizer`), at about the cost of `LLMatcher.validate_grammar`, which stops before the parser and so
    misses the parser's size limit the engine's compile meets. `token_ids`: the ids the grammar names (`<[48]>`),
    renamed to the small tokenizer's spare ids."""
    import llguidance

    if token_ids:
        spare = {f"<[{t}]>": f"<[{BYTE_TOKENS + i}]>" for i, t in enumerate(sorted(set(token_ids)))}
        if len(spare) > SPARE_IDS:
            raise ValueError(f"at most {SPARE_IDS} token ids, not {len(spare)}")
        grammar = _TOKEN.sub(lambda m: spare.get(m.group(0), m.group(0)), grammar)
    m = llguidance.LLMatcher(_check_tokenizer(), grammar, log_level=0)
    return clean_error(m.get_error()) if m.is_error() else ""


@functools.cache
def _check_tokenizer() -> Any:
    """`check`'s tokenizer: a token for each byte, an end token, and `SPARE_IDS` ids beyond them (built in ms)."""
    import llguidance
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<end>"] = BYTE_TOKENS - 1
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return llguidance.LLTokenizer(tok.to_str(), n_vocab=BYTE_TOKENS + SPARE_IDS, eos_token=BYTE_TOKENS - 1)


class Pending(Exception):
    """A grammar not in the cache yet: `GrammarSpecs.resolve` builds it (module docstring, "Where")."""

    def __init__(self, key: bytes, make: Callable[[], str | GrammarError]):
        super().__init__()
        self.key, self.make = key, make  # `make`: the grammar built and checked (`GrammarSpecs._make`)


class GrammarSpecs:
    """Per frontend: the grammar text of each `response_format` seen lately, checked once (LRU). The backend and its
    whitespace are the server's, the same for every request. `threads`: where misses are built (module docstring);
    without, a miss is built at once, on the caller's thread."""

    def __init__(self, size: int = CACHE_SIZE, backend: str = "auto", disable_any_whitespace: bool = False,
                 threads: CompileThreads | None = None):
        _guidance(backend, disable_any_whitespace)  # refused here, not on a request
        self.size = size
        self.backend, self.disable_any_whitespace = backend, disable_any_whitespace
        self.threads = threads
        self._cache: OrderedDict[bytes, str | GrammarError] = OrderedDict()
        self._building: dict[bytes, asyncio.Future] = {}  # a key's build in flight, which its requests share
        self._done: dict[bytes, str | GrammarError] = {}  # what a request's builds gave it (`using`)
        self.hits = self.misses = 0

    def json_schema(self, schema: Any) -> str:
        """`schema`: its JSON text (`msgspec.Raw` or bytes: the frontend's, as received), decoded only on a miss, or
        the schema itself."""
        text = schema if isinstance(schema, bytes | msgspec.Raw) else _enc.encode(schema)
        return self._get(b"s" + text, lambda: json_schema_grammar(msgspec.json.decode(text), self.backend,
                                                                   self.disable_any_whitespace))

    def json_object(self) -> str:
        return self._get(b"o", lambda: json_object_grammar(self.backend, self.disable_any_whitespace))

    def header(self) -> str:
        """A Lark grammar's first line under the server's backend: canonical forcing under guidance, none under auto
        and xgrammar (module docstring)."""
        return "%llguidance {}\n" if self.backend == "guidance" else '%llguidance {"no_forcing": true}\n'

    def thinking(self, schema: Any, prefix: str, key: bytes, token_ids: Sequence[int]) -> str:
        """`response_format`'s grammar with the thought in front (`thought`, module docstring "Thinking"). It is Lark
        under every backend, as guidance's JSON form takes no prefix. Its header is the backend's, and the schema is as
        `json_item` gives it, so the language after the thought is the thinking-off grammar's. `schema` None:
        `json_object`. Cached by `key` (the family and the case) and the schema text; `token_ids`: the markers' ids
        (`check`)."""
        text = b"o" if schema is None else b"s" + (schema if isinstance(schema, bytes | msgspec.Raw)
                                                    else _enc.encode(schema))
        return self._get(b"r" + key + b"\0" + text,
                         lambda: f"{self.header()}start: {prefix} {self.json_item(schema)}\n", token_ids)

    def json_item(self, schema: Any) -> str:
        """`response_format`'s schema as a Lark `%json` item (the branch of a tool-call grammar), with the server's
        whitespace: compact under guidance's `disable_any_whitespace`, else any. A schema's own `x-guidance` options
        win. One that is not an object is ignored under guidance, as llguidance's JSON form ignores it
        (`json_schema_grammar`). Under auto and xgrammar it is refused."""
        if schema is None:
            schema = {"type": "object"}
        elif isinstance(schema, bytes | msgspec.Raw):
            schema = msgspec.json.decode(schema)
        if not isinstance(schema, dict):
            raise GrammarError("the JSON schema must be an object")
        own = schema.get("x-guidance")
        if own is not None and not isinstance(own, dict):
            if self.backend != "guidance":
                raise GrammarError("'x-guidance' must be an object")
            own = None
        _bounded(schema)
        flexible = not (self.backend == "guidance" and self.disable_any_whitespace)
        opts = {"whitespace_flexible": flexible, **(own or {})}
        return "%json " + _enc.encode({**schema, "x-guidance": opts}).decode()

    def tool_calls(self, key: bytes, build: Any, token_ids: Sequence[int] = ()) -> str:
        """A tool-call grammar (`grammar.tools`), checked and cached by `key`: the request's tools and choice, as it
        carried them (`App.grammar`). `token_ids`: the ids its rules name (`check`)."""
        return self._get(b"t" + key, build, token_ids)

    def _get(self, key: bytes, build: Any, token_ids: Sequence[int] = ()) -> str:
        got = self._done.get(key) if self._done else None  # filled only inside `using`
        if got is None:
            got = self._cache.get(key)
            if got is not None:
                self.hits += 1
                self._cache.move_to_end(key)
            else:
                self.misses += 1
                if self.threads is not None:
                    raise Pending(key, lambda: self._make(build, token_ids))
                got = self._store(key, self._make(build, token_ids))
        if isinstance(got, GrammarError):
            raise type(got)(*got.args)  # a fresh one, of its kind: a cached error's traceback does not grow
        return got

    @staticmethod
    def _make(build: Any, token_ids: Sequence[int] = ()) -> str | GrammarError:
        """The grammar text, checked, or the error a client gets for it (its message alone: no build's frames; a
        `NestingError` stays one)."""
        try:
            text = build()
            err = check(text, token_ids)
            return GrammarError(err) if err else text
        except ValueError as exc:  # a GrammarError, or llguidance's own argument errors
            return (type(exc) if isinstance(exc, GrammarError) else GrammarError)(str(exc))

    def _store(self, key: bytes, got: str | GrammarError) -> str | GrammarError:
        self._cache[key] = got
        self._cache.move_to_end(key)
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return got

    async def resolve(self, p: Pending) -> str | GrammarError:
        """`p`'s grammar, built on a compile thread (once a key: a request that misses the same key meanwhile waits
        for the same build), and cached. An exception the build raised (other than a client's error) is raised."""
        fut = self._building.get(p.key)
        if fut is None:
            fut = asyncio.wrap_future(self.threads.submit(p.make))
            self._building[p.key] = fut
            fut.add_done_callback(lambda f, key=p.key: self._settle(key, f))
        return await asyncio.shield(fut)

    def _settle(self, key: bytes, fut: asyncio.Future) -> None:
        self._building.pop(key, None)
        if not fut.cancelled() and fut.exception() is None:
            self._store(key, fut.result())

    @contextlib.contextmanager
    def using(self, done: dict[bytes, str | GrammarError]) -> Iterator[None]:
        """Lookups find what `done` holds (a request's own resolved builds), whatever the cache evicted meanwhile."""
        self._done = done
        try:
            yield
        finally:
            self._done = {}

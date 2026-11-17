"""Tool calls in a model's output (architecture.md, "Tool calls"): a parser per request with tools, fed the engine's
tokens in the frontend, for Gemma 4 (`<|tool_call>call:NAME{...}<tool_call|>`) and Qwen3.8 (`<tool_call>`,
`<function=NAME>`, then `<parameter=K>`, the value and `</parameter>` per argument, then `</function>` and
`</tool_call>`, each on its own line): the formats their chat templates write.

**Markers by token id.** A call starts only at the opener token (Gemma's `<|tool_call>`, Qwen's `<tool_call>`) and its
closer is the closer token; the same text spelled with ordinary tokens is text. Every other special token (the
tokenizer's added tokens marked special) is dropped, Gemma's `<|"|>` aside: content keeps it (as its spelling, when
special tokens are not skipped), and inside a call it delimits strings. Gemma's thought markers are dropped from
content too. The stop token that ends the output is fed (and dropped) like any other.

**Inside a call** the text is lexed by the format's own rules, as its chat template writes them:
- Gemma: text up to the first `:` (`call:` or a bare `:`) is the preamble, then the name up to `{` or `(`, then the
  arguments up to the closer; a call the output ends inside (a `<turn|>` or `max_tokens` cut) is closed there.
- Qwen: `<function=` starts the name, `>` ends it, `</function>` ends the call (in the name too), and a call needs no
  `</tool_call>` before the next `<tool_call>`; a `<function=` inside the arguments is argument text (the older
  parse's rule: vllm-compatibility.md#tool-calls). Special tokens' spellings and the markers' are lexed whole, so
  their `>` ends no name.
The arguments are converted when the call ends (`output.tool_args`), and typed by the tool's schema. A call whose
stripped name is empty is dropped; names are not checked against the tools.

**The content** is the text outside calls, before, between and after them. Not streamed: when calls were made (and
`tool_choice` is not "none"), it is stripped. Streamed: raw, except that a token of whitespace alone before the
content's first other character is dropped once a call has started (held, and sent with what follows, otherwise):
decided token by token, so the result does not depend on how the steps split the tokens. Gemma's thought markers are
dropped from content; Qwen's `<think>` and `</think>` are ordinary added tokens, and stay.

**Streamed deltas** (`take`): a call's first delta carries its id, type, index and name, once its first argument text
comes; its arguments follow in pieces (`output.tool_args.ArgStream`), each extending the last, so they add up to the
JSON the non-streamed response carries, whatever the step sizes. `index` is the call's ordinal, and a step's pieces of
one call are one entry. With `tool_choice` "none" or null no call is kept: the calls leave the content, and none is
returned; `parallel_tool_calls` false keeps the first call. `output.reasoning` puts a thought's parse in front.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import msgspec

from .tool_args import ArgStream, fallback_arguments, fix_types, gemma4_parse, qwen_arguments

_enc = msgspec.json.Encoder()
GEMMA4, QWEN3 = "gemma4", "qwen3_coder"
FAMILIES = (GEMMA4, QWEN3)
CONTENT, PREAMBLE, NAME, ARGS, BETWEEN = range(5)
_QWEN_TERMINALS = ("<function=", "</function>", "<parameter=", "</parameter>", ">")


def call_id() -> str:
    """A tool call's id: chatcmpl-tool- and 16 lowercase hex digits (vllm-compatibility.md#tool-calls)."""
    return "chatcmpl-tool-" + os.urandom(8).hex()


NEEDED = {GEMMA4: ("<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>", "<|tool_response>"),
          QWEN3: ("<tool_call>", "</tool_call>", "<think>", "</think>")}


def missing_tokens(name: str, vocab: Any) -> list[str]:
    """The tokens a format names by id that a vocabulary (its tokens with the added ones) lacks: [] when it fits
    (`config.derive` refuses a parser set that does not)."""
    return [t for t in NEEDED[name] if t not in vocab]


class Family:
    """A format's token ids and lexing tables over one tokenizer (built once per frontend)."""

    def __init__(self, name: str, tokenizer: Any):
        if name not in FAMILIES:
            raise ValueError(f"unknown tool-call parser {name!r}")
        self.name = name
        added = tokenizer.get_added_tokens_decoder()
        self.special = frozenset(i for i, t in added.items() if t.special)
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        missing = missing_tokens(name, vocab)
        if missing:  # refused by config.derive first
            raise ValueError(f"tool_calls.parser {name} cannot be used with this tokenizer: it has no "
                             f"{', '.join(missing)}")
        if name == GEMMA4:
            self.opener, self.closer = vocab["<|tool_call>"], vocab["<tool_call|>"]
            self.quote = vocab['<|"|>']
            # in a call's name or arguments, these are their spelling; in content, dropped
            self.spelled = frozenset(vocab[t] for t in ("<|channel>", "<channel|>"))
        else:
            self.opener, self.closer = vocab["<tool_call>"], vocab["</tool_call>"]
            self.quote = -1
            self.spelled = frozenset(vocab[t] for t in ("<think>", "</think>"))
        self.spelling = {i: tokenizer.id_to_token(i) for i in (*self.special, *self.spelled, self.opener, self.closer)}
        # what the call grammar names by id (`grammar.tools`): Gemma's turn end after calls, Qwen's thought markers
        # (its triggered free text), and a thought in front of the calls (`grammar.spec.thought`): every family's
        # thought markers
        self.grammar_ids = {"opener": self.opener, "closer": self.closer, "quote": self.quote,
                            "end": vocab.get("<|tool_response>", -1),
                            "think": [vocab[t] for t in ("<think>", "</think>") if t in vocab]}
        thought = [vocab[t] for t in ("<|channel>", "<channel|>", "<think>", "</think>") if t in vocab]
        named = (self.opener, self.closer, self.quote, self.grammar_ids["end"], *thought)
        self.grammar_token_ids = sorted({i for i in named if i >= 0})  # the frontend's check renames them
        if name == QWEN3:  # the lexer's literals: the terminals, and every marker's and special token's spelling
            lits = {*_QWEN_TERMINALS, *(s for s in self.spelling.values() if s)}
            self.literals = sorted(lits, key=len, reverse=True)
            self.prefixes = frozenset(lit[:k] for lit in self.literals for k in range(1, len(lit)))


def template_format(template: str) -> str | None:
    """The tool-call format a chat template renders: Gemma 4's when it writes `<|tool_call>`, Qwen3's XML when it
    writes `<function=` and `<parameter=`; None otherwise (`config.ToolCallConfig.parser`'s default)."""
    if "<|tool_call>" in template:
        return GEMMA4
    if "<function=" in template and "<parameter=" in template:
        return QWEN3
    return None


@dataclass(slots=True)
class Call:
    index: int  # the slot's (every call region the output opened)
    name: str = ""
    args: list[str] = field(default_factory=list)
    ended: bool = False
    ordinal: int = -1  # among the kept calls (-1: not kept, or not known yet)
    id: str = ""
    arguments: str = ""  # the JSON, once ended
    named: bool = False  # its name sent (with its first argument text, or with its end)
    stream: ArgStream | None = None  # its arguments in pieces, while it streams (a kept call's)


class ToolCallParser:
    """One request's parser (module docstring). Feed it the output's tokens in order (`feed`), then `finish`; `take`
    returns what a stream sends since the last call. `content()` and `kept_calls()` are the non-streamed message's."""

    __slots__ = ("_buf", "_held_ws", "_out_content", "_out_tools", "_seen_text", "calls_", "choice", "fam", "keep",
                 "parallel", "parts", "schemas", "skip_special", "started", "state")

    def __init__(self, fam: Family, schemas: dict[str, Any], choice: str | None, parallel: bool = True,
                 skip_special: bool = False):
        self.fam = fam
        self.schemas = schemas
        self.choice = choice
        self.keep = choice in ("auto", "required", "named")
        self.parallel = parallel
        self.skip_special = skip_special
        self.state = CONTENT
        self.parts: list[str] = []  # the content
        self.calls_: list[Call] = []
        self._buf = ""  # Qwen: call-region text the lexer holds back
        self.started = False  # a call region opened
        self._seen_text = False  # the content's first non-whitespace character came
        self._held_ws = ""  # the content's leading whitespace, held (streamed and not)
        self._out_content: list[str] = []
        self._out_tools: list[dict] = []

    # -- input --------------------------------------------------------------------------------------------------

    def feed(self, tid: int, piece: str) -> None:
        """One output token and its text (the detokenizer's). A marker's or special token's text may start with text
        the detokenizer held back (an incomplete UTF-8 sequence, completed as replacement characters): that goes to
        the content or the call first, then the token is handled."""
        fam = self.fam
        if tid == fam.opener or tid == fam.closer or tid in fam.special or tid in fam.spelled:
            own = fam.spelling.get(tid) or ""
            own = own if own and piece.endswith(own) else ""
            if len(piece) > len(own):
                self.feed_text(piece[: len(piece) - len(own)])
                piece = own
        if tid == fam.opener or tid == fam.closer:
            self._flush_lexer()
            self._marker(tid)
        elif tid in fam.special or tid in fam.spelled:
            if self.state == CONTENT:
                if (tid == fam.quote and not self.skip_special) or (fam.name == QWEN3 and tid in fam.spelled):
                    self._content(piece)
                return  # other special tokens and Gemma's thought markers: dropped from content
            if tid in fam.spelled or tid == fam.quote:
                self._call_text(fam.spelling[tid], whole=True)
            # every other special token: dropped
        elif piece:
            if self.state == CONTENT:
                self._content(piece)
            else:
                self._call_text(piece)

    def feed_text(self, text: str) -> None:
        """Text with no marker in it (the kept part of a token a stop string cut)."""
        if text:
            if self.state == CONTENT:
                self._content(text)
            else:
                self._call_text(text)

    def count_only(self, tid: int) -> None:
        """A token a stop string cut (`ToolFeed.cut`): nothing to count without a reasoning parser."""

    def new_part(self) -> None:
        """The reasoning parser's first part ended (`output.reasoning.ReasoningParser`): the content that follows is a
        new one, as vLLM's tool engine starts it (whitespace before its first other character held)."""
        self._seen_text = False

    def finish(self) -> None:
        self._flush_lexer()
        if self.state != CONTENT and self.calls_:
            self._end_call()
        self.state = CONTENT
        if self._held_ws:
            self.parts.append(self._held_ws)
            self._out_content.append(self._held_ws)
        self._held_ws = ""

    # -- the state machine ----------------------------------------------------------------------------------------

    def _content(self, text: str) -> None:
        """Content text as it comes: mostly one token's, at times a part of one or several tokens' worth. Text of
        whitespace alone before the content's first other character is held from the stream and the message alike,
        and goes out with that character or at `finish`. A call's opener drops what is held, and after it, text of
        whitespace alone is dropped while the content has no other character yet. `new_part` starts a new content.
        With `tool_choice` "none" nothing is dropped."""
        if not self._seen_text and not text.strip() and self.started and self.choice != "none":
            return
        if self._seen_text:
            self.parts.append(text)
            self._out_content.append(text)
            return
        if not text.strip():
            self._held_ws += text
            return
        self._seen_text = True
        text, self._held_ws = self._held_ws + text, ""
        self.parts.append(text)
        self._out_content.append(text)

    def _start_call(self) -> None:
        self.started = True
        if self.choice != "none":
            self._held_ws = ""  # whitespace before a call: dropped
        self.calls_.append(Call(len(self.calls_)))

    def _marker(self, tid: int) -> None:
        fam, st = self.fam, self.state
        opener = tid == fam.opener
        if st == CONTENT:
            if opener:
                self._start_call()
                self.state = PREAMBLE
            elif fam.name == QWEN3:  # Qwen's closer outside a call: text; Gemma's: absorbed
                self._content(fam.spelling[tid])
            return
        if fam.name == GEMMA4:
            if not opener:  # the closer ends the call, wherever it is in it
                self._end_call()
                self.state = CONTENT
            elif st in (NAME, ARGS):
                self._call_text(fam.spelling[tid], whole=True)
            return
        # Qwen
        if opener:
            if st == NAME:
                self._end_call()
                self._start_call()
                self.state = PREAMBLE
            elif st == BETWEEN:
                self._start_call()
                self.state = PREAMBLE
            elif st == ARGS:
                self._call_text(fam.spelling[tid], whole=True)
            return
        if st in (PREAMBLE, NAME):
            self._end_call()
            self.state = CONTENT
        elif st == BETWEEN:
            self.state = CONTENT
        else:  # in the arguments: text
            self._call_text(fam.spelling[tid], whole=True)

    def _call_text(self, text: str, whole: bool = False) -> None:
        """Text inside a call region; `whole`: a marker's or special token's spelling, lexed as one unit."""
        if self.fam.name == GEMMA4:
            self._gemma_text(text)
        elif whole:
            self._flush_lexer()
            self._qwen_unit(text)
        else:
            self._buf += text
            self._qwen_lex(final=False)

    def _gemma_text(self, text: str) -> None:
        while text:
            st = self.state
            if st == PREAMBLE:
                i = text.find(":")
                if i < 0:
                    return  # dropped
                text, self.state = text[i + 1:], NAME
            elif st == NAME:
                i = min((j for j in (text.find("{"), text.find("(")) if j >= 0), default=-1)
                if i < 0:
                    self.calls_[-1].name += text
                    return
                self.calls_[-1].name += text[:i]
                text, self.state = text[i + 1:], ARGS
                self._name_done()
            else:  # ARGS
                self._args_text(text)
                return

    # Qwen's lexer: literals matched whole, longest first; a tail that may still grow into one is held back
    def _qwen_lex(self, final: bool) -> None:
        buf, fam = self._buf, self.fam
        i, n = 0, len(buf)
        out: list[tuple[bool, str]] = []  # (is a literal, text)
        while i < n:
            j = min((k for k in (buf.find("<", i), buf.find(">", i)) if k >= 0), default=-1)
            if j < 0:
                out.append((False, buf[i:]))
                i = n
                break
            if j > i:
                out.append((False, buf[i:j]))
            if buf[j] == ">":
                out.append((True, ">"))
                i = j + 1
                continue
            lit = next((lit for lit in fam.literals if buf.startswith(lit, j)), None)
            if lit is not None:
                out.append((True, lit))
                i = j + len(lit)
                continue
            if not final and buf[j:] in fam.prefixes:
                i = j
                break
            out.append((False, "<"))
            i = j + 1
        self._buf = buf[i:] if i < n else ""
        for is_lit, text in out:
            if is_lit:
                self._qwen_unit(text)
            else:
                self._qwen_plain(text)

    def _flush_lexer(self) -> None:
        if self._buf:
            self._qwen_lex(final=True)

    def _qwen_plain(self, text: str) -> None:
        st = self.state
        if st == NAME:
            self.calls_[-1].name += text
        elif st == ARGS:
            self._args_text(text)
        elif st == CONTENT:  # what follows an ended call in the same text
            self._content(text)
        # PREAMBLE, BETWEEN: dropped

    def _qwen_unit(self, lit: str) -> None:
        st = self.state
        if lit == "<function=" and st in (PREAMBLE, BETWEEN, NAME):
            if st == NAME:
                self._end_call()
                self._start_call()
            elif st == BETWEEN:
                self._start_call()
            self.state = NAME
        elif lit == ">" and st == NAME:
            self.state = ARGS
            self._name_done()
        elif lit == "</function>" and st in (NAME, ARGS):
            self._end_call()
            self.state = BETWEEN
        else:  # no transition: the state's text
            self._qwen_plain(lit)

    # -- calls --------------------------------------------------------------------------------------------------

    def _name_done(self) -> None:
        call = self.calls_[-1]
        name = call.name.strip()
        if self.keep and name and call.ordinal < 0:
            call.ordinal = sum(c.ordinal >= 0 for c in self.calls_)
            call.id = call_id()
            if self.parallel or call.ordinal == 0:
                call.stream = ArgStream(self.fam.name == GEMMA4, self.schemas.get(name))

    def _args_text(self, text: str) -> None:
        """A call's argument text: its name goes out with the first (vLLM's timing), then each text sends its piece
        of the arguments (`output.tool_args.ArgStream`)."""
        call = self.calls_[-1]
        call.args.append(text)
        s = call.stream
        if s is None or not text:
            return
        if not call.named:
            call.named = True
            s.append(text)
            self._out_tools.append({"id": call.id, "type": "function", "index": call.ordinal,
                                    "function": {"name": call.name.strip()}})
            return
        piece = s.piece(text)
        if piece:
            self._out_tools.append({"index": call.ordinal, "function": {"arguments": piece}})

    def _end_call(self) -> None:
        call = self.calls_[-1]
        if call.ended:
            return
        call.ended = True
        name = call.name.strip()
        if not name:
            return
        raw = "".join(call.args)
        call.args = [raw]
        schema = self.schemas.get(name)
        parsed = None
        try:
            if not raw.strip():
                arguments = "{}"
            elif self.fam.name == GEMMA4:
                parsed = gemma4_parse(raw, per_member=True)
                arguments = json.dumps(parsed, ensure_ascii=False)
            else:
                arguments = qwen_arguments(raw)
        except Exception:  # not expected (a member nested too deep is kept as written): the converter-failed path
            arguments, parsed = fallback_arguments(raw), None
        try:
            call.arguments = fix_types(arguments, schema, parsed)
        except Exception:  # not expected (each member is typed alone): the arguments untyped
            call.arguments = arguments
        if not self.keep:
            return
        if call.ordinal < 0:  # its name ended with the call: one delta with both
            call.ordinal = sum(c.ordinal >= 0 for c in self.calls_)
            call.id = call_id()
            if self.parallel or call.ordinal == 0:
                call.named = True
                self._out_tools.append({"id": call.id, "type": "function", "index": call.ordinal,
                                        "function": {"name": name, "arguments": call.arguments}})
            return
        if not (self.parallel or call.ordinal == 0):
            return
        if not call.named:  # no argument text came: its name and its arguments together
            call.named = True
            self._out_tools.append({"id": call.id, "type": "function", "index": call.ordinal,
                                    "function": {"name": name, "arguments": call.arguments}})
            return
        rest = call.stream.flush(call.arguments)
        if rest:
            self._out_tools.append({"index": call.ordinal, "function": {"arguments": rest}})

    # -- output -------------------------------------------------------------------------------------------------

    def take(self) -> tuple[str, list[dict]]:
        """The content and the tool-call delta entries since the last call (one entry per call index)."""
        text = "".join(self._out_content)
        self._out_content.clear()
        tools = self._out_tools
        if len(tools) > 1:
            merged: dict[int, dict] = {}
            for t in tools:
                m = merged.get(t["index"])
                if m is None:
                    merged[t["index"]] = {**t, "function": dict(t["function"])}
                elif "arguments" in t["function"]:
                    m["function"]["arguments"] = m["function"].get("arguments", "") + t["function"]["arguments"]
            tools = list(merged.values())
        self._out_tools = []
        return text, tools

    def in_call(self) -> bool:
        """Inside a call region (a call, or between calls), not in the content: the reasoning parser's test for
        whether its markers can apply (`output.reasoning.ReasoningParser.feed`)."""
        return self.state != CONTENT

    def kept_calls(self) -> list[Call]:
        """The calls the response carries (after `finish`): the named ones, in order, the first only without
        `parallel_tool_calls`; none for `tool_choice` "none" or null."""
        if not self.keep:
            return []
        calls = [c for c in self.calls_ if c.ordinal >= 0]
        return calls if self.parallel else calls[:1]

    def found(self) -> bool:
        """Whether the output holds a call with a name (the metric's outcome; content is stripped then)."""
        return any(c.ended and c.name.strip() for c in self.calls_)

    def content(self) -> str | None:
        """The non-streamed message's content (after `finish`): the text outside calls, stripped when calls were
        made (and `tool_choice` is not "none"), and empty when it is whitespace and a call region opened (or the
        choice forces a call); null when empty, "" for `required` and a named tool."""
        text = "".join(self.parts)
        forced = self.choice in ("required", "named")
        if self.choice != "none" and self.found():
            text = text.strip()
        elif not text.strip() and (forced or (self.choice != "none" and self.started)):
            text = ""
        return text if forced else text or None


class ToolFeed:
    """A request's parser behind its stop strings (`output.stop`): a token reaches the parser once the stop-string
    check has released all of its text, so the parse sees exactly the output a stop string leaves. The parser is the
    tool parser, or the reasoning parser in front of it (`output.reasoning.ReasoningParser`)."""

    __slots__ = ("parser", "queue", "reasoning")

    def __init__(self, parser: Any):
        self.parser = parser
        self.reasoning = not isinstance(parser, ToolCallParser)  # a ReasoningParser: `take` gives reasoning too
        self.queue: deque[tuple[int, str, int]] = deque()  # (id, its text, the output text's length after it)

    def push(self, tid: int, piece: str, end: int) -> None:
        self.queue.append((tid, piece, end))

    def release(self, upto: int) -> None:
        """Feed the tokens whose text ends at or before `upto` (the stop-string check's sent length)."""
        q, p = self.queue, self.parser
        while q and q[0][2] <= upto:
            tid, piece, _ = q.popleft()
            p.feed(tid, piece)

    def cut(self, upto: int) -> None:
        """A stop string cut the output text at `upto`: the tokens before it, then the kept part of the one it cut
        (as text); the rest is dropped from the parse, and counted as the engine's output: as the parse counts the
        accepted tokens fed whole (`count_whole`) where a cut token is a marker or a special token, or the first token
        is cut, else one by one (`count_only`)."""
        q, p = self.queue, self.parser
        whole = None
        if self.reasoning and (p.first or any(end > upto and tid in p.fam.watch for tid, _, end in q)):
            whole = p.count_whole([(tid, piece) for tid, piece, _ in q])
        while q:
            tid, piece, end = q.popleft()
            start = end - len(piece)
            if end <= upto:
                p.feed(tid, piece)
                continue
            if start < upto:
                p.feed_text(piece[: upto - start])
            p.count_only(tid)
        if whole is not None:
            p.tokens = whole

    def take(self) -> tuple[str, str, list[dict]]:
        """The reasoning, the content and the tool-call delta entries since the last call."""
        if self.reasoning:
            return self.parser.take()
        content, calls = self.parser.take()
        return "", content, calls


def delta_json(content: str, calls: list[dict], reasoning: str | None = "") -> bytes:
    """A streamed delta object, in vLLM's key order: `content`, `reasoning` and `tool_calls` when set (`{}` when none
    is). `reasoning` None: null on a delta with content or calls (`include_reasoning` false, as vLLM sends it)."""
    d: dict[str, Any] = {}
    if content:
        d["content"] = content
    if reasoning:
        d["reasoning"] = reasoning
    elif reasoning is None and (content or calls):
        d["reasoning"] = None
    if calls:
        d["tool_calls"] = calls
    return _enc.encode(d)


def api_finish_reason(engine: str, parser: ToolCallParser) -> str:
    """The response's finish reason (the OpenAI API's, by the rule of vllm-compatibility.md#tool-calls): `tool_calls`
    when calls are returned, the engine stopped by itself and the choice is not a named tool; else the engine's (a
    `max_tokens` cut stays `length`)."""
    if engine == "stop" and parser.choice != "named" and parser.kept_calls():
        return "tool_calls"
    return engine

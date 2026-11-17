# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/renderers/params.py and vllm/entrypoints/common/cli_args.py of vLLM v0.33.0
# (https://github.com/vllm-project/vllm), and changed for llm-mini-infer.
"""Reasoning in a model's output (architecture.md, "Reasoning"): a parser per request of a family with a reasoning
parser (`reasoning.parser`), fed the engine's tokens in the frontend in front of the tool-call parser: Gemma 4's
`<|channel>thought\\n...<channel|>` and Qwen3.8's `<think>...</think>`, as the reference vLLM's
`--reasoning-parser gemma4` and `qwen3` parse them (vllm-compatibility.md#reasoning).

**Where the output starts** (`ReasoningFamily.case`) is read from the prompt's ids, as vLLM's `is_reasoning_end`
and Gemma's `adjust_initial_state_from_prompt` read them, scanning back from the prompt's end:
- a: the thought ended in the prompt (its closer, or a tool call's opener, came last), or Qwen with thinking off;
- b: Gemma with thinking on, a turn begun with no marker since: content, which a thought may open;
- c: the prompt opened the thought (Qwen with thinking on, Gemma's `<|channel>` last): its rest, then its closer;
- d: Gemma with thinking off, a turn begun with no marker since (after a tool result): content.
The thinking flag is `enable_thinking` in the template's effective kwargs (`api.app.App.template_kwargs`), with
vLLM's defaults: Gemma off, Qwen on. The grammars read the same cases (`grammar.spec`).

**The thought** (`ReasoningParser`):
- markers count by token id; a marker spelled with ordinary tokens is text;
- the closer ends it, and so does a tool call's opener (Gemma's `<|tool_call>`, Qwen's `<tool_call>`), which opens a
  call in the content that follows; a second opener inside it is swallowed;
- Gemma: its `thought\\n` label is dropped from each thought's start (a thought that is only a prefix of it is no
  reasoning), and its trailing whitespace too; the first output token opens a thought when its text is `thought` or
  starts with `thought\\n` (vLLM's `_preprocess_feed`). Qwen keeps its text whole;
- special tokens that are not markers are dropped, Gemma's `<|"|>` aside (kept as its spelling).
**After it**, the content and the tool calls are the tool parser's (`output.tool_calls`), or this parser's own content
without tools: special tokens dropped (`<|"|>` kept as its spelling unless special tokens are skipped, as the tool
parse keeps it), whitespace before the content's first other character held while streaming (sent at the end).
Wherever the content is outside a call, Gemma's `<|channel>` opens another thought and its `<channel|>` is absorbed;
Qwen's `</think>` is absorbed and its `<think>` stays text (vLLM's transitions).

**The count** (`tokens`): the thoughts' tokens, Gemma's label included, the markers, a tool call's opener and dropped
special tokens excluded: vLLM's `reasoning_tokens`. Every token the engine accepted counts as the parse counts the
accepted ids fed whole, those a stop string cut too (`count_only`, `count_whole`), so it is the same streamed or not, at
every step size, and with the stop string kept or not. Where the output starts with a thought or may (cases b and c),
only the thoughts until the first one ends count, as vLLM's stream counts them.

The parse is the same at every step size, and streamed and not; where vLLM's stream and non-stream differ, it
follows the stream (vllm-compatibility.md#reasoning lists the cases).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

GEMMA4, QWEN3 = "gemma4", "qwen3"
PARSERS = (GEMMA4, QWEN3)
# vLLM's ReasoningEffort (renderers/params.py)
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# Each family's default `reasoning.effort_map`: Qwen3.8's template takes low, medium and xhigh (and
# answers any other value with a 400); Gemma 4's reads no effort, so every value but none turns thinking on
EFFORT_MAPS: dict[str, dict[str, str]] = {
    GEMMA4: {},
    QWEN3: {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"},
}
NEEDED = {GEMMA4: ("<|channel>", "<channel|>"), QWEN3: ("<think>", "</think>")}
LABEL = "thought\n"  # Gemma's channel label
_LABEL_TOKEN = "thought"
CONTENT, REASONING = 0, 1  # this parser's own states; the tool parser's own `in_call` tells its region


def missing_tokens(name: str, vocab: Any) -> list[str]:
    """The markers a vocabulary lacks: [] when the parser fits it (`config.derive` refuses one that does not)."""
    return [t for t in NEEDED[name] if t not in vocab]


def template_format(template: str) -> str | None:
    """The reasoning markers a chat template renders: Gemma 4's when it writes `<|channel>`, Qwen3's when it writes
    `<think>` and `</think>`; None otherwise (`config.ReasoningConfig.parser`'s default)."""
    if "<|channel>" in template:
        return GEMMA4
    if "<think>" in template and "</think>" in template:
        return QWEN3
    return None


def check_effort_map(value: Any, where: str = "reasoning.effort_map") -> None:
    """vLLM's check of `--reasoning-effort-map` (cli_args.py:512-528), with the setting's name."""
    if value is None:
        return
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        raise ValueError(f"Error: {where} must be a JSON object mapping reasoning efforts to strings.")
    unknown = sorted(set(value) - set(EFFORTS))
    if unknown:
        raise ValueError(f"Error: {where} has unknown efforts {unknown}; keys must be one of {list(EFFORTS)}.")


def effort_kwargs(effort: str | None, effort_map: dict[str, str], request_kwargs: dict | None) -> dict[str, Any]:
    """The template kwargs a request's `reasoning_effort` sets (vLLM's `build_chat_params`): the effort through the
    map (one it lacks passes as it is) as `reasoning_effort`, and `enable_thinking` its value != "none" unless the
    request's own `chat_template_kwargs` sets it. They win over the request's kwargs; a value of "auto" sets nothing,
    as `merge_kwargs` drops it. {} without an effort."""
    if effort is None:
        return {}
    mapped = effort_map.get(effort, effort)
    out: dict[str, Any] = {}
    if mapped != "auto":
        out["reasoning_effort"] = mapped
    if "enable_thinking" not in (request_kwargs or {}):
        out["enable_thinking"] = mapped != "none"
    return out


def thinking_flag(name: str, kwargs: dict) -> bool:
    """The parsers' thinking flag: `enable_thinking` in the effective kwargs, vLLM's default when absent (Gemma
    off, Qwen on), read for its truth."""
    return bool(kwargs.get("enable_thinking", name == QWEN3))


class ReasoningFamily:
    """A reasoning parser's token ids over one tokenizer (built once per frontend)."""

    def __init__(self, name: str, tokenizer: Any):
        if name not in PARSERS:
            raise ValueError(f"unknown reasoning parser {name!r}")
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        missing = missing_tokens(name, vocab)
        if missing:  # refused by config.derive first
            raise ValueError(f"reasoning.parser {name} cannot be used with this tokenizer: it has no "
                             f"{', '.join(missing)}")
        self.name = name
        added = tokenizer.get_added_tokens_decoder()
        self.special = frozenset(i for i, t in added.items() if t.special)
        opener, closer = NEEDED[name]
        self.start, self.end = vocab[opener], vocab[closer]
        if name == GEMMA4:
            tool, turns = "<|tool_call>", ("<|turn>", "<|tool_response>")
            self.quote = vocab.get('<|"|>', -1)
        else:
            tool, turns = "<tool_call>", ("<|im_start|>", "<|im_end|>")
            self.quote = -1
        self.tool_start = vocab.get(tool, -1)  # a tool call's opener: the thought's implicit end
        self.turns = frozenset(vocab[t] for t in turns if t in vocab)
        self.markers = frozenset(i for i in (self.start, self.end, self.tool_start) if i >= 0)
        # dropped in a thought: every special token but the markers and Gemma's quote
        self.dropped = self.special - self.markers - {self.quote}
        self.spelling = {i: tokenizer.id_to_token(i) or "" for i in (*self.special, *self.markers)}
        # the ids a request whose output starts as content hands to a parser (`api.engine_client`): until one comes,
        # the output is its text as it is
        self.watch = self.special | self.markers
        self.grammar_token_ids = sorted({self.start, self.end})  # what a grammar's thought names (`grammar.spec`)
        self.label_id = vocab.get(_LABEL_TOKEN, -1) if name == GEMMA4 else -1

    def case(self, prompt: Sequence[int], thinking: bool) -> str:
        """Where the output starts (module docstring): "a", "b", "c" or "d", from the prompt's ids and the thinking
        flag (`thinking_flag`)."""
        gemma = self.name == GEMMA4
        for t in reversed(prompt):
            if t == self.end or t == self.tool_start:
                return "a"
            if t == self.start:
                return "c" if gemma or thinking else "a"  # Qwen with thinking off: its opener opens nothing
            if t in self.turns:
                break
        if gemma:
            return "b" if thinking else "d"
        return "c" if thinking else "a"


class ReasoningParser:
    """One request's parse of the thought and what follows it (module docstring). Feed it the output's tokens in order
    (`feed`), then `finish`; `take` returns what a stream sends since the last call: the reasoning, the content and the
    tool-call delta entries. `reasoning()` and `content()` are the non-streamed message's. `inner`: the request's tool
    parser (`output.tool_calls.ToolCallParser`), None without tools; `skip_special`: the request's
    `skip_special_tokens` (Gemma's `<|"|>` in content, without tools).

    **vLLM's two engines.** Its stream parses with a reasoning engine until the thought's first end (cases b and c),
    then with its tool engine, which takes every later thought (and every thought in cases a and d). Each engine drops
    Gemma's label once, at its first thought, holding the text while it may still be the label, across thoughts (the
    held text of a thought that was only a prefix of it comes back in front of its next one). The count is the reasoning
    engine's (cases b and c); with the thought ended in the prompt (a and d), every thought this parse finds counts.
    (vLLM re-parses the whole output there, its first-feed opener read on the whole text.)"""

    __slots__ = ("_acc", "_counting", "_cout", "_ended", "_eng", "_held", "_latched", "_rout", "_seen", "_ws", "cparts",
                 "fam", "first", "inner", "rparts", "skip_special", "state", "tokens")

    def __init__(self, fam: ReasoningFamily, case: str, inner: Any = None, skip_special: bool = False):
        self.fam = fam
        self.inner = inner
        self.skip_special = skip_special
        gemma = fam.name == GEMMA4
        self.state = REASONING if case == "c" else CONTENT
        self.first = gemma and case != "c"  # the first token may open a thought by its text
        self.tokens = 0  # the reasoning tokens (module docstring, "The count")
        self._counting = True
        self._eng = 0 if case in ("b", "c") else 1  # vLLM's engine: its reasoning engine (0), its tool engine (1)
        self._acc = ["", ""]  # each engine's text held for Gemma's label
        self._latched = [not gemma, not gemma]  # each engine's label decided
        self._ws = ""  # Gemma: the thought's trailing whitespace, held
        self._ended = False  # the tool engine ended a thought or met a call's opener: vLLM's `_reasoning_ended`
        self.rparts: list[str] = []  # the reasoning sent so far
        self._rout: list[str] = []  # the reasoning not taken yet
        # the content, without tools: its text, what is not taken yet, the leading whitespace held, and whether its
        # first non-whitespace character came
        self.cparts: list[str] = []
        self._cout: list[str] = []
        self._held = ""
        self._seen = False

    # -- input --------------------------------------------------------------------------------------------------

    def feed(self, tid: int, piece: str) -> None:
        """One output token and its text (the detokenizer's). A marker's or special token's text may start with text
        the detokenizer held back (an incomplete UTF-8 sequence, completed as replacement characters): that is fed
        first, as text."""
        fam = self.fam
        if tid in fam.watch:
            own = fam.spelling.get(tid, "")
            if own and len(piece) > len(own) and piece.endswith(own):
                self._feed_text(piece[: len(piece) - len(own)])
                piece = own
        if self.state == REASONING:
            if tid == fam.end:
                self._end_thought()
            elif tid == fam.start or tid in fam.dropped:  # swallowed, or dropped
                if self._ended:  # vLLM's tool engine: a delta with nothing drops the held whitespace
                    self._ws = ""
            elif tid == fam.tool_start:  # the implicit end: the opener starts the content (a call, with tools)
                self._end_thought()
                self._content_token(tid, piece)
            else:
                self.tokens += self._counting
                self._reason(piece)
            return
        inner = self.inner
        if inner is None or not inner.in_call():  # outside a call: the markers' transitions
            if self.first:  # vLLM's `_preprocess_feed`: an opener in front of the label, or of a first closer
                self.first = False
                if tid not in fam.watch and (piece == _LABEL_TOKEN or piece.startswith(LABEL)):
                    self.state = REASONING
                    self.tokens += self._counting
                    self._reason(piece)
                    return
                if tid == fam.end:  # an empty thought: it ends the reasoning engine's part
                    self._end_thought()
                    return
            if tid == fam.tool_start:  # a call's opener ends the reasoning engine's part too (an implicit end)
                if self._eng == 0:
                    self._switch()
                else:
                    self._ended = True
            if tid == fam.end:
                return  # absorbed
            if tid == fam.start and fam.name == GEMMA4:
                self.state = REASONING
                self._ws = ""
                return
        self.first = False
        self._content_token(tid, piece)

    def feed_text(self, text: str) -> None:
        """Text with no marker in it (the kept part of a token a stop string cut, or of one the detokenizer held)."""
        self._feed_text(text)

    def _feed_text(self, text: str) -> None:
        if not text:
            return
        if self.state == REASONING:
            self._reason(text)
        elif self.inner is not None:
            self.inner.feed_text(text)
        else:
            self._content(text)

    def preload(self, text: str) -> None:
        """Content already sent before this parser was built (`api.engine_client.RequestState.start_parser`): it is
        the content's start, and is not sent again."""
        self.cparts.append(text)
        self._seen = self._seen or bool(text.strip())

    def count_only(self, tid: int) -> None:
        """A token the engine accepted that the parse never sees (a stop string cut it): counted as the thought's when
        the thought is open (and it is not a marker or a dropped special token). Exact while no cut token can move the
        parse's state: none is a marker or a special token, and the first token is not among them (`count_whole`)."""
        fam = self.fam
        if self.state == REASONING and tid not in fam.markers and tid not in fam.dropped:
            self.tokens += self._counting

    def count_whole(self, toks: Iterable[tuple[int, str]]) -> int:
        """The count once these tokens (ids and their texts) are fed whole: a copy of this parser fed them, for the
        tokens a stop string cut (`output.tool_calls.ToolFeed.cut`), whose markers move the count as they move the
        parse. The copy shares the families and the tools' schemas, and copies the tool parser's states but not its
        calls' pieces, which the count never reads."""
        memo: dict[int, Any] = {id(self.fam): self.fam}
        inner = self.inner
        if inner is not None:
            memo[id(inner.fam)] = inner.fam
            memo[id(inner.schemas)] = inner.schemas
            for call in inner.calls_:
                memo[id(call.stream)] = None
        dup = copy.deepcopy(self, memo)
        if dup.inner is not None:
            dup.inner.keep = False  # no call is sent: no pieces
        for tid, piece in toks:
            dup.feed(tid, piece)
        return dup.tokens

    def finish(self) -> None:
        if self.state == REASONING:
            self._end_thought()
        if self.inner is not None:
            self.inner.finish()
        elif self._held:
            self._cout.append(self._held)
            self._held = ""

    # -- the thought ----------------------------------------------------------------------------------------------

    def _end_thought(self) -> None:
        """The thought ended: Gemma's trailing whitespace is dropped; after the reasoning engine's first thought, the
        tool engine takes over (class docstring), the count stopped."""
        self.state = CONTENT
        self._ws = ""
        if self._eng == 0:
            self._switch()
        else:
            self._ended = True

    def _switch(self) -> None:
        """The reasoning engine's part ended: the tool engine parses the rest, its own content not begun (whitespace
        before its first other character is held, and dropped when calls follow), the count stopped."""
        self._eng = 1
        self._counting = False
        if self.inner is not None:
            self.inner.new_part()
        else:
            self._seen = False

    def _reason(self, text: str) -> None:
        if self.fam.name == GEMMA4:
            e = self._eng
            if not self._latched[e]:  # the engine's label undecided: its text held while it may still be the label
                acc = self._acc[e] + text
                if acc.startswith(LABEL):
                    text = text[len(LABEL) - (len(acc) - len(text)):]
                elif LABEL.startswith(acc):
                    self._acc[e] = acc
                    return
                else:
                    text = acc
                self._latched[e] = True
                self._acc[e] = ""
                if not text:
                    return
            text = self._ws + text  # trailing whitespace held until more of the thought comes
            kept = text.rstrip()
            self._ws = text[len(kept):]
            text = kept
            if not text:
                return
        self.rparts.append(text)
        self._rout.append(text)

    # -- the content without tools ----------------------------------------------------------------------------------

    def _content_token(self, tid: int, piece: str) -> None:
        if self.inner is not None:
            self.inner.feed(tid, piece)
            return
        fam = self.fam
        if tid in fam.special:
            if tid == fam.quote and not self.skip_special:
                self._content(piece)
            return  # every other special token: dropped
        if piece:
            self._content(piece)

    def _content(self, text: str) -> None:
        self.cparts.append(text)
        if self._seen:
            self._cout.append(text)
        elif not text.strip():
            self._held += text
        else:
            self._seen = True
            self._cout.append(self._held + text)
            self._held = ""

    # -- output ---------------------------------------------------------------------------------------------------

    def take(self) -> tuple[str, str, list[dict]]:
        """The reasoning, the content and the tool-call delta entries since the last call."""
        reasoning = "".join(self._rout)
        self._rout.clear()
        if self.inner is not None:
            content, tools = self.inner.take()
            return reasoning, content, tools
        content = "".join(self._cout)
        self._cout.clear()
        return reasoning, content, []

    def reasoning(self) -> str | None:
        """The non-streamed message's reasoning (after `finish`): null when there is none."""
        return "".join(self.rparts) or None

    def content(self) -> str | None:
        """The non-streamed message's content (after `finish`): the tool parser's, else the text, null when empty."""
        if self.inner is not None:
            return self.inner.content()
        return "".join(self.cparts) or None

    def kept_calls(self) -> list:
        return self.inner.kept_calls() if self.inner is not None else []

    def found(self) -> bool:
        return self.inner is not None and self.inner.found()

    @property
    def choice(self) -> str | None:
        return self.inner.choice if self.inner is not None else None

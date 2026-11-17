"""Detokenizer parity (the concatenated stream equals a one-shot decode) and stop-string hold-back."""

from __future__ import annotations

import random

import pytest
from tokenizers.decoders import DecodeStream

from llm_mini_infer.output.detokenizer import Detokenizer
from llm_mini_infer.output.stop import OutputText, StopStrings
from llm_mini_infer.render.renderer import Renderer

from ..support.fixtures import MODELS, load_cases, model_dir


@pytest.mark.parametrize("name", MODELS)
def test_stream_equals_one_shot_decode(name: str) -> None:
    cases = [c for c in load_cases(name) if "ids" in c["expected"]]
    tok = Renderer(model_dir(name)).tokenizer
    rng = random.Random(7)
    bad = []
    for case in cases:
        ids = case["expected"]["ids"]
        for skip in (False, True):
            # Unprimed: the whole sequence as if generated from an empty prompt.
            d = Detokenizer(tok, None, skip_special_tokens=skip)
            if "".join(d.step(t) for t in ids) != tok.decode(ids, skip_special_tokens=skip):
                bad.append((case["id"], skip, "unprimed"))
            # Primed (with a suffix of a prompt that decodes cleanly): the stream equals the decode of
            # the continuation given the whole prompt.
            cut = rng.randrange(1, len(ids))
            prompt, gen = ids[:cut], ids[cut:]
            head = tok.decode(prompt, skip_special_tokens=skip)
            full = tok.decode(ids, skip_special_tokens=skip)
            if head.endswith("�") or not full.startswith(head):
                continue
            d = Detokenizer(tok, prompt, skip_special_tokens=skip)
            if "".join(d.step(t) for t in gen) != full[len(head):]:
                bad.append((case["id"], skip, "primed"))
    assert not bad, bad[:5]


WINDOW = 32  # a fixed-window prime of the prompt's last tokens, which this server does not use: the cuts it starts
# inside a character are kept as regression cases

# Characters the tokenizers split into byte-fallback or multi-byte pieces: CJK, emoji, rare planes, combining marks.
MULTIBYTE_TEXT = ("The quick brown fox 狐狸跳过了懒狗 jumps 🦊🐕‍🦺 over 𠮷野家 the lazy dog; naïve café ǅ ᚠᚢᚦ ☃ 🧑🏽‍🚀 "
                  "ひらがなカタカナ 한국어 العربية שלום 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ")
GEN_LEN = 12


def reference_pieces(tok, prompt: list[int], gen: list[int], skip: bool) -> list[str]:
    """The reference: a DecodeStream primed with the whole prompt, stepped over the generated ids."""
    stream = DecodeStream(ids=prompt, skip_special_tokens=skip)
    return [stream.step(tok, t) or "" for t in gen]


@pytest.mark.parametrize("name", MODELS)
def test_priming_across_broken_characters(name: str) -> None:
    """The prime is a few prompt tokens, the reference primes with the whole prompt. Over every cut of a multi-byte text
    (CJK, emoji, rare planes) with the generated ids stepped one at a time, so multi-byte characters start right at the
    first step, the pieces equal the reference's step by step (not only in total: a piece emitted a step early or late
    would show) when
    - (a) a fixed window of the last 32 tokens, kept as a regression case, starts inside a character (its first token
      decodes alone to U+FFFD);
    - (b) the prompt's decoded head ends in U+FFFD (an open character, the case the primed check above skips; every
      suffix then ends in the open character, so the whole prompt is the prime).
    The cuts are searched for in each tokenizer and the test fails when a tokenizer offers none of a kind."""
    load_cases(name)  # skips when the fixture is absent
    tok = Renderer(model_dir(name)).tokenizer
    ids = tok.encode(MULTIBYTE_TEXT * 3, add_special_tokens=False).ids
    mid_window, open_head, bad = 0, 0, []
    for skip in (False, True):
        for cut in range(WINDOW + 1, len(ids) - GEN_LEN):
            prompt, gen = ids[:cut], ids[cut:cut + GEN_LEN]
            is_open = tok.decode(prompt, skip_special_tokens=skip).endswith("�")
            starts_mid = "�" in tok.decode(prompt[-WINDOW:][:1], skip_special_tokens=skip)
            if not (starts_mid or is_open):
                continue
            open_head += is_open
            mid_window += starts_mid and not is_open
            d = Detokenizer(tok, prompt, skip_special_tokens=skip)
            got = [d.step(t) for t in gen]
            want = reference_pieces(tok, prompt, gen, skip)
            if got != want:
                bad.append((cut, skip, is_open, got, want))
    assert mid_window, f"{name}: no cut starts a {WINDOW}-token window inside a multi-byte character"
    assert open_head, f"{name}: no cut leaves the prompt's decoded head ending in U+FFFD"
    assert not bad, (len(bad), bad[:2])


def run_stream(pieces: list[str], stops: list[str], include: bool = False) -> tuple[list[str], str | None]:
    out = OutputText(StopStrings(stops, include))
    sent = []
    for p in pieces:
        hit = out.append(p)
        sent.append(out.take(final=hit is not None))
        if hit is not None:
            return sent, hit
    sent.append(out.take(final=True))
    return sent, None


def test_stop_string_split_across_tokens_is_never_sent() -> None:
    sent, hit = run_stream(["Hello wo", "rld, E", "N", "D here"], ["END"])
    assert hit == "END"
    assert "".join(sent) == "Hello world, "
    assert sent[1] == "rld, "  # "E" held back: it could start "END"


def test_holdback_only_what_could_match() -> None:
    sent, hit = run_stream(["abc", "xyz"], ["</stop>"])
    assert hit is None and sent == ["abc", "xyz", ""]
    sent, hit = run_stream(["abc<", "/st", "x"], ["</stop>"])
    assert hit is None and sent == ["abc", "", "</stx", ""]


def test_include_stop_str_and_earliest_match() -> None:
    sent, hit = run_stream(["one STOP two END"], ["END", "STOP"], include=True)
    assert hit == "STOP" and "".join(sent) == "one STOP"


def test_stop_spanning_many_pieces_and_final_flush() -> None:
    sent, hit = run_stream(list("xx<|eot|>yy"), ["<|eot|>"])
    assert hit == "<|eot|>" and "".join(sent) == "xx"
    sent, hit = run_stream(["tail <|eo"], ["<|eot|>"])
    assert hit is None and "".join(sent) == "tail <|eo"  # flushed when the request finishes otherwise

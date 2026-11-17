"""tests/runner/test_batch_invariance_gpu.py's harness on the CPU: the teacher-forcing grammar (tests/support/forced.py)
allows the reference's next token alone (then the EOS), along drafts as llguidance fills them (a mask after each draft,
to the first the grammar rejects), and forces the forced prefix's tokens; a forced run's captures map each row to its
position and each logit column to its token id, and hold its images' encoder rows; and the golden
(tests/support/golden.py) digests, keys, records and compares.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_mini_infer.grammar.engine import words

from ..support import golden
from ..support.forced import Bits, Forced, ForcedMatcher, ForcedRun
from ..support.gemma_images import png
from ..support.mtp_common import EOS as TINY_EOS
from ..support.tiny_gemma import MM_IMAGE, make_tiny_assistant, make_tiny_gemma_mm_fp8

VOCAB, EOS = 100, 99
REF = [5, 7, 7, 31, 64]


def allowed(buf: np.ndarray, row: int) -> list[int]:
    return np.flatnonzero(np.unpackbits(buf[row].view(np.uint8), bitorder="little")[:VOCAB]).tolist()


def test_the_matcher_follows_the_reference_then_the_eos() -> None:
    m = ForcedMatcher(REF, EOS, forced=2)
    assert m.compute_ff_tokens() == [5] and m.consume_tokens([5]) and m.compute_ff_tokens() == [7]
    assert m.validate_tokens([7, 7, 30]) == 2 and m.n == 1  # validate does not move
    assert m.try_consume_tokens([7, 7, 30]) == 2 and m.n == 3 and not m.is_error()
    assert m.compute_ff_tokens() == []  # past the forced prefix
    m.rollback(1)
    assert m.n == 2 and m.consume_tokens([7, 31, 64]) and m.is_stopped()
    assert m.validate_tokens([EOS]) == 1 and m.consume_tokens([EOS])
    bad = ForcedMatcher(REF, EOS, forced=0)
    assert not bad.consume_tokens([6]) and bad.is_error()
    with pytest.raises(ValueError):
        bad.rollback(1)


def test_the_masks_along_drafts_stop_at_the_first_rejected() -> None:
    g = Forced(VOCAB, EOS, {0: REF}, forced=0)
    m = g.instance(g.compile("forced:0"))
    m.consume_tokens([5])
    buf = np.full((8, words(VOCAB)), -1, np.int32)
    g.fill([(m, 0)], [(m, 1, [7, 31, 64])], buf)  # the draft 31 is rejected (the reference has 7 there)
    assert allowed(buf, 0) == [7]
    assert allowed(buf, 1) == [7] and allowed(buf, 2) == [7]  # after the draft 7: the reference's next, 7
    assert (buf[3] == -1).all()  # past the rejected draft: never written (verification stops there)
    end = g.instance(REF)
    end.consume_tokens(REF[:4])
    g.fill([], [(end, 4, [64, EOS])], buf)
    assert allowed(buf, 4) == [64] and allowed(buf, 5) == [EOS] and allowed(buf, 6) == [EOS]
    assert g.only(end, 64) and not g.only(end, 7)
    assert m.n == 1 and end.n == 4  # filling moves no matcher


# -- the captures, on the CPU runner -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mm_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("repro_mm")
    target = make_tiny_gemma_mm_fp8(d / "target", seed=6)
    return target, make_tiny_assistant(d / "assistant", target, seed=7)


def text(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(3, 490, n).tolist()  # below the image tokens (500-503)


PROMPTS = {0: [text(20, 21), png(48, 36, 1), text(12, 22)], 1: [text(9, 23), png(40, 40, 2), text(5, 24)],
           2: [text(60, 25)], 3: [text(14, 26), png(30, 60, 3), text(8, 27)]}
ANSWERS = {r: text(30 + 7 * r, 30 + r) for r in PROMPTS}
SWITCHES = dict(sparse_head=True, draft_universe=True, always_split=True, head_fixed_m=True, encode_per_image=True)


def forced_bits(pair, beside: bool, arena: str) -> tuple[Bits, list[int]]:
    f = ForcedRun(pair, PROMPTS, ANSWERS, device="cpu", arena=arena, max_num_seqs=8, budget=256, num_pages=2048,
                  **SWITCHES)
    if beside:
        f.add(2)
        f.steps(3)
    f.add(0)
    if beside:
        f.add(1)
        f.steps(4)
        f.add(3)
    return f.bits(0), f.prompts[0][0] + ANSWERS[0] + [TINY_EOS]


def test_a_forced_runs_captures_map_rows_and_columns(mm_pair, tmp_path) -> None:
    """The captures' plumbing (`Logits`, `Capture`, `ForcedRun`) on the CPU runner, which claims nothing about bits (the
    torch references in fp32), in tests/runner/test_batch_invariance_gpu.py's image arm: a tiny multimodal FP8 model
    with its assistant at k = 3, every step split, the sparse head at one M, the tower one image a call; request 0 (an
    image) alone, then with requests 1-3 arriving around it (0's image and 1's in one ENCODE). Every kept position of
    request 0 has its state and its logits; each logit is its column's token's head row applied to that position's
    state, and the forced token is among the columns (rows map to positions, columns to token ids); beside, the same
    positions are kept and the same tokens sit in other columns. Its image's encoder rows sit at the image's first
    position, one a soft token, written (the pool starts zeroed), and alike in both runs."""
    from safetensors.torch import load_file

    from llm_mini_infer.models.gemma4.model import Gemma4Config

    alone, seq = forced_bits(mm_pair, False, str(tmp_path / "alone"))
    beside, _ = forced_bits(mm_pair, True, str(tmp_path / "beside"))
    assert alone.encodes == [1] and 2 in beside.encodes and set(beside.calls) == {1}, (alone.encodes, beside.encodes)
    assert alone.heads.sparse > 0 and beside.heads.sparse > 0
    assert set(alone.hidden) == set(beside.hidden) == set(alone.logits) == set(beside.logits)
    assert len(alone.hidden) >= len(ANSWERS[0])
    head = load_file(mm_pair[0] / "model.safetensors")["model.language_model.embed_tokens.weight"].double().numpy()
    cap = Gemma4Config.from_dir(mm_pair[0]).final_logit_softcapping
    moved = 0
    for p in sorted(alone.hidden):
        for bits in (alone, beside):
            ids, logits = bits.logits[p]
            assert seq[p + 1] in ids, p  # the forced token
            want = np.tanh(head[ids] @ bits.hidden[p].view(np.float32).astype(np.float64) / cap) * cap
            np.testing.assert_allclose(logits.view(np.float32), want, rtol=1e-4, atol=1e-5)
        _, xa, xb = np.intersect1d(alone.logits[p][0], beside.logits[p][0], assume_unique=True, return_indices=True)
        moved += int((xa != xb).sum())
    assert moved > 0  # beside, the unions hold more tokens: the same token sits in another column
    first, width = seq.index(MM_IMAGE), Gemma4Config.from_dir(mm_pair[0]).hidden_size
    assert set(alone.images) == set(beside.images) == {first}
    rows = alone.images[first].view(np.float32)
    assert rows.shape == (seq.count(MM_IMAGE), width) and np.isfinite(rows).all() and (rows != 0).any(axis=1).all()
    np.testing.assert_allclose(beside.images[first].view(np.float32), rows, rtol=1e-5, atol=1e-6)


def test_a_qwen_forced_runs_captures_map_rows_and_columns(tmp_path) -> None:
    """The same plumbing over the Qwen runner (`QwenForcedRun`, CPU): a tiny multimodal Qwen with its MTP layer at
    k = 3 under batch invariance (every step split, the sparse head at one M, the tower one image a call), request 0
    (an image) alone and beside the others. Under MTP the runner keeps every token's final hidden state, so the
    capture reads each sample token's own: each logit is its column's token's head row (`lm_head`, untied) applied to
    that position's state, the forced token among the columns."""
    from safetensors.torch import load_file

    from ..support.forced import QwenForcedRun
    from ..support.tiny_qwen import VL_TOKENS, make_tiny_vl_model

    model = tmp_path / "vl"
    make_tiny_vl_model(model, seed=3, mtp=True)
    pad = VL_TOKENS["image_token_id"]
    head = load_file(model / "model.safetensors")["lm_head.weight"].double().numpy()
    runs = []
    for beside in (False, True):
        f = QwenForcedRun(model, PROMPTS, ANSWERS, device="cpu", arena=str(tmp_path / f"arena{beside}"),
                          max_num_seqs=8, budget=256, num_pages=2048, sparse_head=True, draft_universe=True,
                          always_split=True, batch_invariant=True)
        if beside:
            f.add(2)
            f.steps(3)
        f.add(0)
        if beside:
            f.add(1)
            f.steps(4)
            f.add(3)
        runs.append((f.bits(0), f.prompts[0][0] + ANSWERS[0] + [TINY_EOS]))
    (alone, seq), (beside, _) = runs
    assert alone.heads.sparse > 0 and set(alone.hidden) == set(beside.hidden) == set(alone.logits)
    assert len(alone.hidden) >= len(ANSWERS[0]) and alone.encodes == [1] and set(beside.calls) == {1}
    for p in sorted(alone.hidden):
        for bits in (alone, beside):
            ids, logits = bits.logits[p]
            assert seq[p + 1] in ids, p
            want = head[ids] @ bits.hidden[p].view(np.float32).astype(np.float64)
            np.testing.assert_allclose(logits.view(np.float32), want, rtol=1e-4, atol=1e-5)
    first = seq.index(pad)
    assert set(alone.images) == {first} and alone.images[first].shape[0] == seq.count(pad)


# -- the golden --------------------------------------------------------------------------------------------------------


def fake_bits(seed: int) -> tuple[dict, dict, dict]:
    """A capture's shape: positions -> states' bits (uint16), -> (token ids, logits' bits), and an image's first
    position -> its encoder rows' bits (uint16)."""
    rng = np.random.default_rng(seed)
    hidden = {p: rng.integers(0, 2**16, 8, dtype=np.uint16) for p in range(5)}
    logits = {p: (np.sort(rng.choice(VOCAB, 3, replace=False)).astype(np.int32),
                  rng.integers(0, 2**32, 3, dtype=np.uint32)) for p in range(5)}
    images = {p: rng.integers(0, 2**16, (4, 8), dtype=np.uint16) for p in (1, 7)}
    return hidden, logits, images


def test_the_digests_follow_the_bits_alone(tmp_path) -> None:
    """`golden.bits_digest`: the same bits in another insertion order give the same digests; a flipped bit moves its
    part's digest only. `golden.weights_digest`: the tensors decide it, not the files' metadata, the configuration or
    the directory."""
    import torch
    from safetensors.torch import save_file

    hidden, logits, images = fake_bits(1)
    d = golden.bits_digest(hidden, logits, images)
    assert set(d) == {"hidden", "logits", "images"}
    assert golden.bits_digest(*(dict(reversed(c.items())) for c in (hidden, logits, images))) == d

    def moved(*capture: dict) -> set[str]:
        return {part for part, v in golden.bits_digest(*capture).items() if v != d[part]}

    assert moved({**hidden, 3: hidden[3] ^ np.uint16(1)}, logits, images) == {"hidden"}
    ids, bits = logits[2]
    assert moved(hidden, {**logits, 2: (ids, bits ^ np.uint32(1 << 31))}, images) == {"logits"}
    one = images[7].copy()
    one[3, 5] ^= np.uint16(1 << 15)
    assert moved(hidden, logits, {**images, 7: one}) == {"images"}
    assert moved(hidden, logits, {}) == {"images"}  # a text run's: the digest of nothing

    tensors = {"b.weight": torch.arange(6, dtype=torch.bfloat16).view(2, 3), "a.scale": torch.ones(2)}
    for name, meta in (("one", {"format": "pt"}), ("two", {"format": "pt", "note": "other"})):
        (tmp_path / name).mkdir()
        save_file(tensors, tmp_path / name / "model.safetensors", metadata=meta)
    (tmp_path / "two" / "config.json").write_text("{}")
    assert golden.weights_digest([tmp_path / "one"]) == golden.weights_digest([tmp_path / "two"])
    save_file({**tensors, "a.scale": torch.tensor([1.0, 2.0])}, tmp_path / "two" / "model.safetensors")
    assert golden.weights_digest([tmp_path / "one"]) != golden.weights_digest([tmp_path / "two"])


def test_the_golden_is_recorded_on_request_and_compared_otherwise(tmp_path, refresh_golden) -> None:
    """`golden.check` on a file in `tmp_path`: a missing file or entry fails, naming the entry, `--refresh-golden` and
    the test to rerun; a refresh writes the entry (the stack as metadata, the other entries kept); a run with the same
    digests passes, one whose logits moved fails naming them and both stacks, and one that lacks a part of the entry
    fails too. The key holds the GPU's SM count: two GPUs of one architecture have entries of their own."""
    assert isinstance(refresh_golden, bool)  # the option is registered (tests/conftest.py)
    path = tmp_path / "golden" / "g.json"
    entry = golden.key("image", "sm_120-188sm", "ab" * 32)
    assert entry == "image/sm_120-188sm/abababababababab"
    d = golden.bits_digest(*fake_bits(1))
    stack = {"torch": "2.0", "cuda": "13.0"}
    with pytest.raises(pytest.fail.Exception,
                       match=r"no golden for image/sm_120-188sm/abababababababab .*SM count.*--refresh-golden"):
        golden.check(path, entry, d, refresh=False, stack=stack, test="tests/test_x.py")
    golden.check(path, "text/sm_120-188sm/0123456789abcdef", d, refresh=True, stack=stack, test="tests/test_x.py")
    with pytest.raises(pytest.fail.Exception, match=r"pytest tests/test_x\.py --refresh-golden"):
        golden.check(path, entry, d, refresh=False, stack=stack, test="tests/test_x.py")
    golden.check(path, entry, d, refresh=True, stack=stack, test="tests/test_x.py")
    saved = json.loads(path.read_text())
    assert set(saved) == {entry, "text/sm_120-188sm/0123456789abcdef"}
    assert saved[entry] == {**d, "versions": stack}
    golden.check(path, entry, d, refresh=False, stack={"torch": "2.1"}, test="tests/test_x.py")  # the stack is metadata
    with pytest.raises(pytest.fail.Exception, match=r"no golden for image/sm_120-170sm/"):
        golden.check(path, golden.key("image", "sm_120-170sm", "ab" * 32), d, refresh=False, stack=stack,
                     test="tests/test_x.py")
    other = {**d, "logits": golden.bits_digest(*fake_bits(2))["logits"]}
    with pytest.raises(pytest.fail.Exception, match=r"the run's logits differ .*torch 2\.0.*torch 2\.1"):
        golden.check(path, entry, other, refresh=False, stack={"torch": "2.1"}, test="tests/test_x.py")
    other = {**golden.bits_digest(*fake_bits(2)), "logits": d["logits"]}
    with pytest.raises(pytest.fail.Exception, match=r"the run's hidden states and images' encoder rows differ"):
        golden.check(path, entry, other, refresh=False, stack=stack, test="tests/test_x.py")
    with pytest.raises(pytest.fail.Exception, match=r"the run's images' encoder rows differ"):
        golden.check(path, entry, {k: v for k, v in d.items() if k != "images"}, refresh=False, stack=stack,
                     test="tests/test_x.py")


def test_a_fetched_golden_merges_into_the_committed_one(tmp_path) -> None:
    """`golden.merge`: a golden file fetched from a GPU replaces the committed entries of its keys and keeps the
    others (another GPU's), in the file's own format."""
    d1, d2 = golden.bits_digest(*fake_bits(1)), golden.bits_digest(*fake_bits(2))
    dest, src = tmp_path / "committed" / "g.json", tmp_path / "fetched" / "g.json"
    golden.check(dest, "text/sm_120-170sm/0123456789abcdef", d1, refresh=True, stack={}, test="t")
    golden.check(dest, "text/sm_120-188sm/0123456789abcdef", d1, refresh=True, stack={}, test="t")
    golden.check(src, "text/sm_120-188sm/0123456789abcdef", d2, refresh=True, stack={"torch": "2.1"}, test="t")
    golden.check(src, "image/sm_120-188sm/0123456789abcdef", d2, refresh=True, stack={}, test="t")
    assert golden.merge(src, dest) == ["image/sm_120-188sm/0123456789abcdef", "text/sm_120-188sm/0123456789abcdef"]
    saved = json.loads(dest.read_text())
    assert saved == {"text/sm_120-170sm/0123456789abcdef": {**d1, "versions": {}},
                     "text/sm_120-188sm/0123456789abcdef": {**d2, "versions": {"torch": "2.1"}},
                     "image/sm_120-188sm/0123456789abcdef": {**d2, "versions": {}}}
    assert dest.read_text() == json.dumps(saved, indent=1, sort_keys=True) + "\n"

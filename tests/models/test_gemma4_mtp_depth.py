"""Gemma 4's MTP at k = 2, 5 and 7 (`test_gemma4_mtp` runs k = 3) in GemmaRunner on CPU (the torch reference
backends, fp32): `test_gemma4_mtp`'s properties at verify windows of 3, 6 and 8 tokens, through the scheduler in window
mode and the engine loop (its `Run`, the same tiny pair). At k = 7 a verify row is as long as the tiny model's sliding
window (8) and the assistant chains 7 steps at one position.

MTP on gives MTP off's outputs, greedy (the assistant's own drafts, an oracle's all right or a random number right)
and seeded; the acceptance counters add up (every verify row counts its k drafts, the accepted ones a prefix of them,
so the count per position does not grow with the position); the drafts are transformers' assistant's chain at the
committed positions, each sampled with the noise of the position it fills; a stop token and the length limits end an
accepted run inside its window; prefix hits, window checkpoints and structured output give MTP off's outputs, image
prompts transformers'.

The scheduler's own window-mode MTP (`test_window_mtp`, at k = 3, its fuzzing drawing k up to 5) runs at these depths
too, through the validating fake runner: verify rows sliding their windows, and an arena under pressure."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import reserve_positions

from ..support.gemma_mtp import PROMPTS, Run, W, counters
from ..support.harness import Spec
from ..support.mtp_common import EOS, draft_positions_spy, seq
from ..support.sched_cases import shared
from ..support.tiny_gemma import hf_assistant, hf_model, make_tiny_assistant, make_tiny_gemma
from ..support.window_mtp import bound
from ..support.window_mtp import harness as window_harness

DEPTHS = [2, 5, 7]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """`test_gemma4_mtp`'s pair: the same seeds, so k = 3 there and the depths here run the same models."""
    d = tmp_path_factory.mktemp("g4mtpk")
    target = make_tiny_gemma(d / "target", window=W, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


@pytest.mark.parametrize("k", DEPTHS)
def test_greedy_outputs_equal_without_mtp(pair, k) -> None:
    """MTP on (the assistant's own drafts; the oracle's, all right and a random number right) gives MTP off's greedy
    outputs, prompts longer than the window included; with the oracle's drafts all right whole windows are accepted,
    the k-th draft too."""
    base = Run(pair, 0)
    for i, p in enumerate(PROMPTS):
        base.add(i, p, max_tokens=18)
    base.run()
    ref = base.out
    for oracle, rng in ((None, None), (ref, None), (ref, np.random.default_rng(k))):
        r = Run(pair, k)
        for i, p in enumerate(PROMPTS):
            r.add(i, p, max_tokens=18)
        r.run(oracle, rng)
        assert r.out == ref, (k, oracle is not None, rng is not None)
        per = counters(r.sched, k)
        if oracle is not None and rng is None:
            assert r.sched.spec_accepted > r.sched.spec_draft_tokens // 2  # the oracle's drafts were accepted
            assert per[k - 1] > 0, per


@pytest.mark.parametrize("k", DEPTHS)
@pytest.mark.parametrize("params", [(0.8, 1.0, 0), (1.0, 0.9, 20)])
def test_seeded_sampling_outputs_equal_without_mtp(pair, params, k) -> None:
    """The coupling's promise at every depth: a seeded request samples the same tokens with MTP on and off."""
    temp, top_p, top_k = params
    runs = []
    for kk in (0, k):
        r = Run(pair, kk)
        for i, p in enumerate(PROMPTS):
            r.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
        runs.append(r.run())
    oracle = Run(pair, k)
    for i, p in enumerate(PROMPTS):
        oracle.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
    oracle.run(runs[0].out, np.random.default_rng(1))
    assert runs[1].out == runs[0].out and oracle.out == runs[0].out
    counters(runs[1].sched, k)
    counters(oracle.sched, k)


@pytest.mark.parametrize("k", DEPTHS)
def test_drafts_take_the_noise_of_the_positions_they_fill(pair, k) -> None:
    r = Run(pair, k)
    seen = draft_positions_spy(r.r, k)
    for i, p in enumerate(PROMPTS[:2]):
        r.add(i, p, max_tokens=8, temperature=1.0, seed=50 + i)
    r.run()
    assert len(seen) >= 2 * k
    for keyed, fill in seen:
        assert keyed == fill, (keyed, fill)


@pytest.mark.parametrize("k", DEPTHS)
def test_drafts_match_transformers_assistant(pair, k) -> None:
    """After every step, a running request's k drafts are transformers' assistant's greedy chain from the committed
    sequence: the target's hidden at the last committed position p and the token after it, every step at position p
    over the target's KV of [0, p] (sliding layers: the last W positions)."""
    target, assistant = pair
    ht, ha = hf_model(target), hf_assistant(assistant)
    r = Run(pair, k, caching=False)
    prompt = seq(13, 7)
    r.add(0, prompt, max_tokens=14)
    states: list[tuple[int, int, list[int]]] = []

    def record() -> None:
        if len(r.sched.slot_req) < 2 or r.sched.slot_req[1] is None:
            return
        tb = r.r.tables
        at = int(tb.seq_len[1])
        if at <= len(prompt) - 1 or int(tb.pending[1]) < 0:
            return
        states.append((at, int(tb.pending[1]), tb.drafts[1].tolist()))

    r.after_step = record
    r.run()
    full = prompt + r.out[0]
    checked = 0
    for at, pending, drafts in states:
        if at + 1 > len(full) or full[at] != pending:
            continue  # the slot moved on before this turn's outputs were read
        assert len(drafts) == k
        p = at - 1
        with torch.no_grad():
            out = ht.model(torch.tensor([full[:at]]), return_shared_kv_states=True)
            kv = out.shared_kv_states
            lo = max(0, p - W + 1)
            shared = {"full_attention": (kv["full_attention"][0][:, :, :p + 1], kv["full_attention"][1][:, :, :p + 1]),
                      "sliding_attention": (kv["sliding_attention"][0][:, :, lo:p + 1],
                                            kv["sliding_attention"][1][:, :, lo:p + 1])}
            h, tok, want = out.last_hidden_state[:, p:p + 1], torch.tensor([[pending]]), []
            for _ in range(k):
                res = ha(inputs_embeds=torch.cat([ht.model.embed_tokens(tok), h], -1),
                         position_ids=torch.tensor([[p]]), shared_kv_states=shared)
                tok = res.logits[:, -1].argmax(-1, keepdim=True)
                h = res.last_hidden_state
                want.append(int(tok))
        assert drafts == want, (at, drafts, want)
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize(("k", "m"), [(k, m) for k in DEPTHS for m in (0, k - 1)])
def test_stop_token_ends_the_accepted_run(pair, k, m) -> None:
    """A stop token among the oracle's drafts (all right) ends the accepted run: the request finishes on it, as with
    MTP off, and the drafts past it are never accepted. It is output k + 2 + m, the second verify row's first or last
    draft (the first row's window accepted whole), which the target samples at window index m (0, or k − 1: the
    deepest the commit limit checks): the m drafts before it are accepted, the row ends on it, the right drafts from
    it on are refused."""
    base = Run(pair, 0)
    base.add(0, PROMPTS[1], max_tokens=20)
    base.run()
    stop = base.out[0][k + 2 + m]
    assert base.out[0].index(stop) == k + 2 + m
    want = Run(pair, 0)
    want.add(0, PROMPTS[1], max_tokens=20, stop=[stop])
    want.run()
    r = Run(pair, k)
    r.add(0, PROMPTS[1], max_tokens=20, stop=[stop])
    r.run({0: base.out[0]})
    assert r.out == want.out and r.out[0][-1] == stop and r.finish[0] == FinishReason.STOP
    counters(r.sched, k)
    # two verify rows, k and m drafts accepted: the device committed nothing past the stop (the host's cut hides it)
    assert (r.sched.spec_drafts, r.sched.spec_accepted) == (2, k + m)


@pytest.mark.parametrize("k", DEPTHS)
def test_length_limits_end_the_accepted_run(pair, k) -> None:
    """The length limits fall inside verify windows and end each request where MTP off does: max_tokens 11 with the
    oracle's drafts right past it (never accepted), after a whole window accepted; and max_model_len (256) 6 to 19
    positions after the prompts, the windows' positions past it computed but never committed."""
    base = Run(pair, 0)
    base.add(0, PROMPTS[1], max_tokens=30)
    base.run()
    want = Run(pair, 0)
    want.add(0, PROMPTS[1], max_tokens=11)
    want.run()
    assert want.out[0] == base.out[0][:11]
    r = Run(pair, k)
    r.add(0, PROMPTS[1], max_tokens=11)
    r.run({0: base.out[0]})
    assert r.out == want.out and r.finish[0] == FinishReason.LENGTH
    assert counters(r.sched, k)[k - 1] > 0
    assert r.sched.spec_accepted == 10 - r.sched.spec_drafts  # the device committed 11 tokens: 1, then 1 + a a row
    prompts = [seq(243, 11), seq(250, 12), seq(237, 13)]
    base = Run(pair, 0, max_num_seqs=3)
    for i, p in enumerate(prompts):
        base.add(i, p, max_tokens=40)
    base.run()
    assert [len(p) + len(base.out[i]) for i, p in enumerate(prompts)] == [256] * 3
    for oracle, rng in ((None, None), (base.out, None), (base.out, np.random.default_rng(k))):
        r = Run(pair, k, max_num_seqs=3)
        for i, p in enumerate(prompts):
            r.add(i, p, max_tokens=40)
        r.run(oracle, rng)
        assert r.out == base.out and r.finish == base.finish, (k, oracle is not None, rng is not None)
        counters(r.sched, k)
        # what the device committed: each request's first token, then 1 + a a row (nothing past max_model_len)
        assert r.sched.spec_accepted == sum(len(o) for o in r.out.values()) - len(prompts) - r.sched.spec_drafts


@pytest.mark.parametrize("k", DEPTHS)
def test_prefix_hits_and_window_checkpoints_with_mtp(pair, k) -> None:
    """Prompts sharing a prefix longer than the window, decode window checkpoints every 8 positions and a follow-up
    turn hitting its own conversation, in an arena of two requests' startup bound at the depth: MTP on gives MTP off's
    outputs and hits; every page comes back."""
    base_p = seq(30, 9)
    prompts = [base_p + seq(5, 20 + i) for i in range(3)] + [seq(7, 30)]
    outs = []
    for kk in (0, k):
        reserve = reserve_positions(2, kk)
        one = window_pages_for_one_request(W, 64, 4, reserve) + pages_for_one_request(256, 16, reserve, True)
        r = Run(pair, kk, num_pages=2 * one + 1, max_num_seqs=3)
        for i, p in enumerate(prompts):
            r.add(i, p, max_tokens=13)
        r.run({} if kk else None)
        r.add(10, prompts[0] + r.out[0] + seq(3, 40), max_tokens=7)
        r.run()
        assert r.sched.prefix_hits > 0 and r.cached.get(10, 0) > len(prompts[0])
        r.sched.cache.clear()
        r.sched.check_accounting()
        assert r.sched.kv.num_free == r.sched.kv.num_pages - 1
        outs.append(r.out)
        if kk:
            counters(r.sched, kk)
    assert outs[1] == outs[0]


@pytest.mark.parametrize("k", DEPTHS)
@pytest.mark.parametrize("params", [(0.0, 1.0, 0), (0.8, 0.9, 0)])
def test_masked_outputs_equal_with_and_without_mtp(pair, params, k) -> None:
    """Structured output with MTP: a verify row samples its 1 + k positions under the masks along its drafts (8 at
    k = 7), constrained and unconstrained requests share steps; outputs equal MTP off's, every constrained token
    allowed where it was sampled, the grammar ending one request."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..support.fake_grammar import FakeGrammar, spec

    temp, top_p, _ = params
    reqs = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 4, 9)), dict(rid=1, prompt=seq(33, 1)),
            dict(rid=2, prompt=seq(9, 2), grammar=spec(2, 30, 40), max_tokens=12),
            dict(rid=3, prompt=seq(50, 3), grammar=spec(3, 0, 6))]
    outs = []
    for kk in (0, k):
        grammar = FakeGrammar(512, (EOS,))
        r = Run(pair, kk, grammars=Grammars(grammar), caching=False)
        for q in reqs:
            r.add(q["rid"], q["prompt"], max_tokens=q.get("max_tokens", 14), temperature=temp, top_p=top_p,
                  ignore_eos=False, grammar=q.get("grammar", ""))
        r.run()
        assert r.sched.grammars.errors == 0
        for q in reqs:
            if q.get("grammar"):
                m = grammar.instance(grammar.compile(q["grammar"]))
                for t in r.out[q["rid"]]:
                    assert m.allowed()[t] and (m.is_stopped() or m.consume_tokens([t])), (kk, q["rid"])
        outs.append((r.out, r.finish))
        if kk:
            counters(r.sched, kk, masked=True)
    assert outs[1] == outs[0]
    assert outs[0][1][0] == FinishReason.STOP and outs[0][0][0][-1] == EOS


@pytest.fixture(scope="module")
def mm(tmp_path_factory):
    """`test_gemma4_images`' multimodal model (window 8: its image spans are longer than the window) and an
    assistant for it."""
    from ..support.tiny_gemma import make_tiny_gemma_mm

    d = tmp_path_factory.mktemp("g4imgk")
    model = make_tiny_gemma_mm(d / "model", window=W, depth=2)
    return model, make_tiny_assistant(d / "assistant", model, seed=9)


@pytest.mark.parametrize("k", DEPTHS)
def test_images_with_mtp_match_transformers(mm, tmp_path, k) -> None:
    """Images with MTP: prompts with one and two images in one batch (chunks cut around the spans, drafts right after a
    span), then the first prompt again (a hit past its image). The greedy outputs equal transformers', with the
    assistant's drafts and with an oracle's (all right: windows and pages grow by accepted runs of up to 1 + k)."""
    from ..support.gemma_images import Engine, greedy_mm, png, text
    from ..support.tiny_gemma import hf_mm_model

    model, asst = mm
    want = full = None
    for oracle in (False, True):
        eng = Engine(model, tmp_path, budget=96, mtp_k=k, assistant=asst)
        p0 = eng.prompt([text(30, 3), png(48, 36, 4), text(5, 5)])
        p1 = eng.prompt([text(30, 3), png(40, 40, 6), text(4, 7), png(32, 48, 8), text(3, 9)])
        if want is None:
            hf = hf_mm_model(model)
            want = {0: greedy_mm(hf, p0[0], p0[2], 14), 1: greedy_mm(hf, p1[0], p1[2], 12)}
            want[2] = want[0]
            full = {0: p0[0] + want[0], 1: p1[0] + want[1], 2: p0[0] + want[0]}

        @torch.inference_mode()
        def plant(_turn, eng=eng, full=full):  # every running slot's drafts become its next k reference tokens
            tb = eng.runner.tables
            for slot, req in enumerate(eng.sched.slot_req):
                if req is None or slot == 0:
                    continue
                ref, at = full[req.rid], int(tb.seq_len[slot])
                for j in range(k):
                    tb.drafts[slot, j] = ref[at + 1 + j] if at + 1 + j < len(ref) else 7

        hook = plant if oracle else None
        eng.add(0, p0[0], p0[1], 14)
        eng.add(1, p1[0], p1[1], 12)
        eng.run(on_turn=hook)
        eng.add(2, p0[0], p0[1], 14)
        eng.run(on_turn=hook)
        span = p0[1][0].offset + p0[1][0].tokens
        assert eng.cached[2] >= span, (k, eng.cached[2], span)
        assert eng.outs == want, (k, oracle)
        per = counters(eng.sched, k)
        if oracle:
            assert eng.sched.spec_accepted > eng.sched.spec_draft_tokens // 2 and per[k - 1] > 0, per


# -- the scheduler's window mode at the depths, through the validating fake runner ----------------------------------


@pytest.mark.parametrize("k", DEPTHS)
@pytest.mark.parametrize("accept_p", [0.0, 0.6, 1.0])
def test_verify_rows_slide_their_windows(k, accept_p) -> None:
    """A 200-token prompt and 60 decode tokens with a window of 16 (pages of 4): whatever is accepted, every verify
    row of 1 + k tokens finds its window from its true first query on (the fake checks the device's positions), and
    its outputs are the predicted ones."""
    spec = Spec(rid=0, prompt_len=200, max_tokens=60, seed=5, ignore_eos=True)
    h = window_harness([spec], k=k, accept_p=accept_p, prefix_caching=False)
    h.run(check_every_turn=True)
    h.check()
    n = h.runner.counts
    assert n["accepted"] == 0 if accept_p == 0.0 else n["accepted"] > 0
    if accept_p == 1.0:
        assert n["accepted"] >= 0.8 * n["drafts"]  # whole windows, but the last


@pytest.mark.parametrize("k", DEPTHS)
@pytest.mark.parametrize("seed", range(4))
def test_arena_pressure(k, seed) -> None:
    """Eight requests sharing prefixes, arriving over 30 turns, three running at once, in an arena of one request's
    startup bound at the depth and up to half as much again: cached pages are evicted (and requests preempted with
    their verify rows in flight, at the larger depths); the fake validates every plan and predicts every output."""
    rng = np.random.default_rng(seed)
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(int(rng.integers(20, 90)), int(rng.integers(3, 30)), i % 3),
                  max_tokens=int(rng.integers(4, 40)), seed=5000 + i, arrive=int(rng.integers(0, 30)),
                  ignore_eos=bool(rng.integers(0, 2))) for i in range(8)]
    need = bound(8, 4, 20, 32, 256, k) + 1
    h = window_harness(specs, k=k, window=8, max_tokens=32, max_num_seqs=3, max_model_len=256,
                       num_pages=need + int(rng.integers(0, need)) // 2,
                       accept_p=float(rng.choice([0.0, 0.5, 0.95, 1.0])))
    h.run(check_every_turn=True)
    evicted = h.sched.cache.evicted_pages  # before `check` clears the cache
    h.check()
    assert evicted > 0

"""MTP with Gemma 4's assistant in GemmaRunner on CPU (the torch reference backends, fp32; architecture.md,
"Speculative decoding"): verification with the sampler's own noise, the assistant's k drafts, rollback by truncation,
through the real scheduler in window mode (the KV arena, window checkpoints, prefix hits) and the engine loop.

The tiny assistant is random, so its drafts are mostly rejected; an **oracle** that overwrites the drafts between steps
with the true continuation (all or a random number of them correct) drives every accepted count. MTP on and off give
the same outputs, greedy and seeded alike; and since outputs cannot show a drafter bug (a wrong draft is only
rejected), the drafts themselves are compared with transformers' assistant at the committed positions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.messages import FinishReason

from ..support.gemma_mtp import PROMPTS, Run, W
from ..support.mtp_common import EOS, draft_positions_spy, seq
from ..support.tiny_gemma import hf_assistant, hf_model, make_tiny_assistant, make_tiny_gemma


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("g4mtp")
    target = make_tiny_gemma(d / "target", window=W, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


@pytest.mark.parametrize("k", [1, 3])
def test_greedy_outputs_equal_without_mtp(pair, k) -> None:
    """MTP on (the assistant's own drafts; the oracle's, all right and a random number right) gives MTP off's greedy
    outputs, prompts longer than the window included: rollback is a truncation the next window overwrites."""
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
        if oracle is not None and rng is None:
            assert r.sched.spec_accepted > r.sched.spec_draft_tokens // 2  # the oracle's drafts were accepted


@pytest.mark.parametrize("params", [(0.8, 1.0, 0), (1.0, 0.9, 20)])
def test_seeded_sampling_outputs_equal_without_mtp(pair, params) -> None:
    """The coupling's promise: a seeded request samples the same tokens with MTP on and off."""
    temp, top_p, top_k = params
    runs = []
    for k in (0, 3):
        r = Run(pair, k)
        for i, p in enumerate(PROMPTS):
            r.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
        runs.append(r.run())
    oracle = Run(pair, 3)
    for i, p in enumerate(PROMPTS):
        oracle.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
    oracle.run(runs[0].out, np.random.default_rng(1))
    assert runs[1].out == runs[0].out and oracle.out == runs[0].out


def test_drafts_take_the_noise_of_the_positions_they_fill(pair) -> None:
    r = Run(pair, 3)
    seen = draft_positions_spy(r.r, 3)
    for i, p in enumerate(PROMPTS[:2]):
        r.add(i, p, max_tokens=8, temperature=1.0, seed=50 + i)
    r.run()
    assert len(seen) >= 6
    for keyed, fill in seen:
        assert keyed == fill, (keyed, fill)


@pytest.mark.parametrize("nxt", [False, True], ids=["p", "p+1"])
def test_drafts_match_transformers_assistant(pair, nxt) -> None:
    """After every step, a running request's k drafts are transformers' assistant's greedy chain from the committed
    sequence: the target's hidden at the last committed position p and the token after it, every step at position p
    (with dev.speculative.draft_query_next, p + 1) over the target's KV of [0, p] (sliding layers: the last W
    positions)."""
    target, assistant = pair
    ht, ha = hf_model(target), hf_assistant(assistant)
    k = 3
    r = Run(pair, k, caching=False, draft_query_next=nxt)
    prompt = seq(13, 7)
    r.add(0, prompt, max_tokens=14)
    states: list[tuple[list[int], list[int]]] = []

    def record() -> None:
        req = r.sched.slot_req[1] if len(r.sched.slot_req) > 1 else None
        if req is None:
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
        p = at - 1
        ids = torch.tensor([full[:at]])
        with torch.no_grad():
            out = ht.model(ids, return_shared_kv_states=True)
            kv = out.shared_kv_states
            lo = max(0, p - W + 1)
            shared = {"full_attention": (kv["full_attention"][0][:, :, :p + 1], kv["full_attention"][1][:, :, :p + 1]),
                      "sliding_attention": (kv["sliding_attention"][0][:, :, lo:p + 1],
                                            kv["sliding_attention"][1][:, :, lo:p + 1])}
            h, tok, want = out.last_hidden_state[:, p:p + 1], torch.tensor([[pending]]), []
            for _ in range(k):
                res = ha(inputs_embeds=torch.cat([ht.model.embed_tokens(tok), h], -1),
                         position_ids=torch.tensor([[p + nxt]]), shared_kv_states=shared)
                tok = res.logits[:, -1].argmax(-1, keepdim=True)
                h = res.last_hidden_state
                want.append(int(tok))
        assert drafts == want, (at, drafts, want)
        checked += 1
    assert checked >= 3


def test_the_drafters_position_changes_drafts_never_outputs(pair) -> None:
    """dev.speculative.draft_query_next: the drafts differ from those at p (the queries rotate otherwise), the greedy
    outputs are MTP off's: the verification decides every token."""
    base = Run(pair, 0)
    for i, p in enumerate(PROMPTS):
        base.add(i, p, max_tokens=18)
    base.run()
    drafts = {}
    for nxt in (False, True):
        r = Run(pair, 3, draft_query_next=nxt)
        seen: list[list[int]] = []
        r.after_step = lambda r=r, seen=seen: seen.append(r.r.tables.drafts[1:5].flatten().tolist())
        for i, p in enumerate(PROMPTS):
            r.add(i, p, max_tokens=18)
        r.run()
        assert r.out == base.out, nxt
        drafts[nxt] = seen
    assert drafts[False] != drafts[True]


def test_stop_token_ends_the_accepted_run(pair) -> None:
    """A stop token among the drafts ends the accepted run (the commit limit): the request finishes on it, as with MTP
    off, and the oracle's drafts past it are never accepted."""
    base = Run(pair, 0)
    base.add(0, PROMPTS[1], max_tokens=20)
    base.run()
    stop = base.out[0][6]
    want = Run(pair, 0)
    want.add(0, PROMPTS[1], max_tokens=20, stop=[stop])
    want.run()
    r = Run(pair, 3)
    r.add(0, PROMPTS[1], max_tokens=20, stop=[stop])
    r.run({0: base.out[0]})
    assert r.out == want.out and r.out[0][-1] == stop and r.finish[0] == FinishReason.STOP


def test_prefix_hits_and_window_checkpoints_with_mtp(pair) -> None:
    """Prompts sharing a prefix longer than the window, decode window checkpoints every 8 positions, a follow-up turn
    hitting its own conversation, and an arena small enough to evict windows: MTP on gives MTP off's outputs and hits;
    every page comes back."""
    from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request

    base_p = seq(30, 9)
    prompts = [base_p + seq(5, 20 + i) for i in range(3)] + [seq(7, 30)]
    outs = []
    for k in (0, 3):
        one = window_pages_for_one_request(W, 64, 4, 2 * (1 + k) + k) + pages_for_one_request(256, 16, 10, True)
        r = Run(pair, k, num_pages=2 * one + 1, max_num_seqs=3)
        for i, p in enumerate(prompts):
            r.add(i, p, max_tokens=13)
        r.run({} if k else None)
        r.add(10, prompts[0] + r.out[0] + seq(3, 40), max_tokens=7)
        r.run()
        assert r.sched.prefix_hits > 0 and r.cached.get(10, 0) > len(prompts[0])
        r.sched.cache.clear()
        r.sched.check_accounting()
        assert r.sched.kv.num_free == r.sched.kv.num_pages - 1
        outs.append(r.out)
    assert outs[1] == outs[0]


@pytest.mark.parametrize("params", [(0.0, 1.0, 0), (0.8, 0.9, 0)])
def test_masked_outputs_equal_with_and_without_mtp(pair, params) -> None:
    """Structured output with MTP (the grammar's split step on Gemma 4): a verify row samples its 1 + k positions under
    the masks along its drafts, the drafts stay unmasked, and constrained and unconstrained requests share steps;
    outputs equal MTP off's, every constrained token allowed where it was sampled, the grammar ending one request."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..support.fake_grammar import FakeGrammar, spec

    temp, top_p, _ = params
    reqs = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 4, 9)), dict(rid=1, prompt=seq(33, 1)),
            dict(rid=2, prompt=seq(9, 2), grammar=spec(2, 30, 40), max_tokens=12),
            dict(rid=3, prompt=seq(50, 3), grammar=spec(3, 0, 6))]
    outs = []
    for k in (0, 3):
        grammar = FakeGrammar(512, (EOS,))
        r = Run(pair, k, grammars=Grammars(grammar), caching=False)
        for q in reqs:
            r.add(q["rid"], q["prompt"], max_tokens=q.get("max_tokens", 14), temperature=temp, top_p=top_p,
                  ignore_eos=False, grammar=q.get("grammar", ""))
        r.run()
        assert r.sched.grammars.errors == 0
        for q in reqs:
            if q.get("grammar"):
                m = grammar.instance(grammar.compile(q["grammar"]))
                for t in r.out[q["rid"]]:
                    assert m.allowed()[t] and (m.is_stopped() or m.consume_tokens([t])), (k, q["rid"])
        outs.append((r.out, r.finish))
    assert outs[1] == outs[0]
    assert outs[0][1][0] == FinishReason.STOP and outs[0][0][0][-1] == EOS

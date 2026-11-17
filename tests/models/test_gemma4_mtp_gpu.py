"""MTP with Gemma 4's assistant on CUDA: GemmaRunner in BF16 with our kernels (FlashInfer's window and attn512 over
verify rows of 1 + k queries, the assistant's Q-only layers on the target's KV, the draft head) and CUDA graphs (a
decode graph holds the verification, the assistant's k steps and the draft head), through the scheduler in window
mode. A tiny model with the real head dims (sliding 256, full 512), a window of 40, and a tiny assistant.

BF16 rounds differently with the row shapes (a verify row of 4 queries against a decode row of one; split counts), so
two runs agree token for token up to a near tie: where they part, transformers' fp32 logits must put the top two
within twice the BF16 runs' error (`NEAR`). CUDA only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..support.gemma_gpu import NEAR, W, agree, gpu_run, reference, run_prompts
from ..support.gemma_gpu import pair as pair  # the fixture
from ..support.gemma_mtp import Run
from ..support.mtp_common import seq
from ..support.tiny_gemma import hf_assistant, hf_model

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def test_mtp_graphs_give_the_eager_tokens(pair) -> None:
    """k = 3 in CUDA graphs and eagerly: the same tokens up to a near tie, with the assistant's own drafts and with an
    oracle's (a random number right, so accepted runs of every length go through the graphs)."""
    eager = run_prompts(gpu_run(pair, 3, graphs=False))
    assert "decode" not in eager.modes
    ref = reference(pair[0], eager.out)
    n = sum(len(o) for o in eager.out.values())
    graphs = run_prompts(gpu_run(pair, 3))
    assert graphs.modes.count("decode") > 20
    assert agree(graphs.out, eager.out, ref) > 0.8 * n
    oracle = run_prompts(gpu_run(pair, 3), eager.out, np.random.default_rng(3))
    assert agree(oracle.out, eager.out, ref) > 0.8 * n
    assert oracle.sched.spec_accepted > 0


def test_mtp_gives_mtp_offs_tokens(pair) -> None:
    """MTP on (k = 1 and 3, graphs) gives MTP off's greedy tokens up to a near tie; an oracle's drafts are accepted."""
    off = run_prompts(gpu_run(pair, 0))
    ref = reference(pair[0], off.out)
    n = sum(len(o) for o in off.out.values())
    for k in (1, 3):
        on = run_prompts(gpu_run(pair, k))
        assert agree(on.out, off.out, ref) > 0.8 * n, k
        oracle = run_prompts(gpu_run(pair, k), off.out)
        assert agree(oracle.out, off.out, ref) > 0.8 * n, k
        assert oracle.sched.spec_accepted > oracle.sched.spec_draft_tokens // 2, k


def test_a_long_prompt_alone_prefills_in_breakable_graphs(pair) -> None:
    """A prompt longer than the step budget, alone: its first chunks are steps without sample rows, whose breakable
    graphs replay the drafter for padding rows only (an empty drafter meta, its attention breaks skipped); MTP on
    gives MTP off's tokens up to a near tie."""
    prompts = {0: seq(150, 9)}
    outs = []
    for k in (0, 3):
        r = gpu_run(pair, k, budget=64)
        r.add(0, prompts[0], max_tokens=24)
        r.run()
        r.r.shutdown()
        assert r.modes[:2] == ["break", "break"], r.modes
        outs.append(r.out)
    ref = reference(pair[0], outs[0], prompts)
    assert agree(outs[1], outs[0], ref, prompts) > 12


def test_drafts_match_transformers_assistant(pair) -> None:
    """After every step (decode graphs), a running request's drafts are transformers' assistant's greedy chain in fp32
    from the committed sequence (the target's hidden at the last committed position p and the token after it, every
    step at p over the target's KV of [0, p]), up to the first near tie of the chain's logits."""
    target, assistant = pair
    ht, ha = hf_model(target), hf_assistant(assistant)
    k = 3
    r = gpu_run(pair, k, caching=False)
    prompt = seq(50, 7)
    r.add(0, prompt, max_tokens=30)
    states: list[tuple[int, int, list[int]]] = []

    def record() -> None:
        if len(r.sched.slot_req) < 2 or r.sched.slot_req[1] is None:
            return
        tb = r.r.tables
        at, pending = int(tb.seq_len[1]), int(tb.pending[1])
        if at > len(prompt) - 1 and pending >= 0:
            states.append((at, pending, tb.drafts[1].tolist()))

    r.after_step = record
    r.run()
    r.r.shutdown()
    assert r.modes.count("decode") > 5
    full = prompt + r.out[0]
    checked = 0
    for at, pending, drafts in states:
        if at + 1 > len(full) or full[at] != pending:
            continue  # the slot moved on before this turn's outputs were read
        p = at - 1
        with torch.no_grad():
            out = ht.model(torch.tensor([full[:at]]), return_shared_kv_states=True)
            kv = out.shared_kv_states
            lo = max(0, p - W + 1)
            shared = {"full_attention": (kv["full_attention"][0][:, :, :p + 1], kv["full_attention"][1][:, :, :p + 1]),
                      "sliding_attention": (kv["sliding_attention"][0][:, :, lo:p + 1],
                                            kv["sliding_attention"][1][:, :, lo:p + 1])}
            h, tok = out.last_hidden_state[:, p:p + 1], torch.tensor([[pending]])
            for j in range(k):
                res = ha(inputs_embeds=torch.cat([ht.model.embed_tokens(tok), h], -1),
                         position_ids=torch.tensor([[p]]), shared_kv_states=shared)
                top = res.logits[0, -1].topk(2)
                if (top.values[0] - top.values[1]).item() < NEAR:
                    break  # a near tie: BF16 may pick either, and the chain after it differs
                assert drafts[j] == int(top.indices[0]), (at, j, drafts, int(top.indices[0]))
                checked += 1
                tok, h = top.indices[:1].view(1, 1), res.last_hidden_state
    assert checked >= 8, checked


SHARED = seq(72, 99)  # a long shared system prompt, longer than the window (cached after the first)
MANY = [SHARED + seq(3 + (i * 7) % 20, 10 + i) for i in range(24)]  # each with its own tail


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("masked", [False, True], ids=["plain", "masked"])
def test_many_rows_on_a_shared_prefix_in_graphs_give_the_eager_tokens(pair, masked, k) -> None:
    """24 requests on one cached prefix at k = 3 and 5 (the assistant's default), so CUDA graphs, prefix hits and MTP
    meet under load (a fault can need all three together): decode steps of 24 rows down to one (decode buckets 32, 24,
    16, 8, 4, 2 and 1, padded; at k = 5 a step's 256 tokens still hold 32 verify rows of 6) and mixed steps (breakable
    graphs) give the eager tokens up to a near tie. Masked: every row under a grammar that allows every token, as a
    structured request's rows are under its schema's, so every step splits into its forward and sampling graphs
    (decode_f/decode_s, break_f/break_s)."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..scripts.profile_verify import AllowAll

    def go(graphs: bool) -> Run:
        r = gpu_run(pair, k, graphs=graphs, max_num_seqs=32, budget=256, num_pages=2048,
                    grammars=Grammars(AllowAll(512)) if masked else None)
        r.sizes, inner = [], r.r._mode

        def sized(lay, split=False):
            m = inner(lay, split)
            r.sizes.append(m)
            return m

        r.r._mode = sized
        r.add(0, MANY[0], max_tokens=40, grammar="any" if masked else "")  # the first computes the prefix
        r.run()
        for i, p in enumerate(MANY[1:], 1):
            r.add(i, p, max_tokens=40, grammar="any" if masked else "")
        r.run()
        r.r.shutdown()
        assert sum(1 for c in r.cached.values() if c >= len(SHARED) - 16) >= 20, r.cached  # the others hit
        return r

    eager = go(False)
    graphs = go(True)
    assert "decode" not in eager.modes and graphs.modes.count("decode") > 10
    buckets = {b for kind, b in graphs.sizes if kind == "decode"}
    assert {16, 24} <= buckets and "break" in graphs.modes, sorted(buckets)
    ref = reference(pair[0], eager.out, MANY)
    n = sum(len(o) for o in eager.out.values())
    assert agree(graphs.out, eager.out, ref, MANY) > 0.8 * n

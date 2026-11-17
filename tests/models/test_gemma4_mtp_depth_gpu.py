"""Gemma 4's MTP on CUDA at k = 2, 5 and 7 (`test_gemma4_mtp_gpu` runs k = 3): `test_gemma4_mtp_gpu`'s properties
at verify windows of 3, 6 and 8 tokens. GemmaRunner in BF16 with our kernels (FlashInfer's window and attn512 over
verify rows of 1 + k queries, the assistant's Q-only layers on the target's KV, the draft head) and CUDA graphs (a
decode graph holds the verification, the assistant's k steps and the draft head), through the scheduler in window
mode, on the same tiny pair (the real head dims, a window of 40).

Two BF16 runs agree token for token up to a near tie of transformers' fp32 logits (`gemma_gpu.agree`); the
acceptance counters add up (`gemma_mtp.counters`). CUDA only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..support.gemma_gpu import NEAR, W, agree, gpu_run, reference, run_prompts, to_bf16
from ..support.gemma_mtp import counters
from ..support.mtp_common import seq
from ..support.tiny_gemma import hf_assistant, hf_model, make_tiny_assistant, make_tiny_gemma

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """`gemma_gpu`'s pair: the same seeds, BF16 weights and q and k norms."""
    d = tmp_path_factory.mktemp("g4mtpkgpu")
    target = make_tiny_gemma(d / "target", window=W, gpu=True, seed=3)
    to_bf16(target, ("q_norm.weight", "k_norm.weight"))
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=4)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant


@pytest.fixture(scope="module")
def off(pair):
    """MTP off's greedy tokens (graphs), transformers' fp32 logits along them, and how many there are."""
    r = run_prompts(gpu_run(pair, 0))
    return r.out, reference(pair[0], r.out), sum(len(o) for o in r.out.values())


@pytest.mark.parametrize("k", [5, 7])
def test_mtp_graphs_give_the_eager_tokens(pair, k) -> None:
    """k = 5 and 7 in CUDA graphs and eagerly: the same tokens up to a near tie, with the assistant's own drafts and
    with an oracle's (a random number right, so accepted runs of every length go through the graphs)."""
    eager = run_prompts(gpu_run(pair, k, graphs=False))
    assert "decode" not in eager.modes
    ref = reference(pair[0], eager.out)
    n = sum(len(o) for o in eager.out.values())
    graphs = run_prompts(gpu_run(pair, k))
    assert graphs.modes.count("decode") > 20
    assert agree(graphs.out, eager.out, ref) > 0.8 * n
    oracle = run_prompts(gpu_run(pair, k), eager.out, np.random.default_rng(k))
    assert agree(oracle.out, eager.out, ref) > 0.8 * n
    assert oracle.sched.spec_accepted > 0
    for r in (eager, graphs, oracle):
        counters(r.sched, k)


@pytest.mark.parametrize("k", [2, 5, 7])
def test_mtp_gives_mtp_offs_tokens(pair, off, k) -> None:
    """MTP on (graphs) gives MTP off's greedy tokens up to a near tie; an oracle's drafts (all right) are accepted,
    whole windows of them: the k-th draft too."""
    out, ref, n = off
    on = run_prompts(gpu_run(pair, k))
    assert agree(on.out, out, ref) > 0.8 * n
    counters(on.sched, k)
    oracle = run_prompts(gpu_run(pair, k), out)
    assert agree(oracle.out, out, ref) > 0.8 * n
    assert oracle.sched.spec_accepted > oracle.sched.spec_draft_tokens // 2
    assert counters(oracle.sched, k)[k - 1] > 0


@pytest.mark.parametrize("k", [7])
def test_a_long_prompt_alone_prefills_in_breakable_graphs(pair, k) -> None:
    """A prompt longer than the step budget, alone: its first chunks are steps without sample rows, whose breakable
    graphs replay the drafter's k steps for padding rows only; MTP on gives MTP off's tokens up to a near tie."""
    prompts = {0: seq(150, 9)}
    outs = []
    for kk in (0, k):
        r = gpu_run(pair, kk, budget=64)
        r.add(0, prompts[0], max_tokens=24)
        r.run()
        r.r.shutdown()
        assert r.modes[:2] == ["break", "break"], r.modes
        outs.append(r.out)
    ref = reference(pair[0], outs[0], prompts)
    assert agree(outs[1], outs[0], ref, prompts) > 12


@pytest.mark.parametrize("k", [5, 7])
def test_drafts_match_transformers_assistant(pair, k) -> None:
    """After every step (decode graphs), each of a running request's k drafts is transformers' assistant's greedy pick
    in fp32 up to a near tie, given the chain before it: from the committed sequence (the target's hidden at the last
    committed position p and the token after it, every step at p over the target's KV of [0, p]), each next step from
    the previous step's hidden and our previous draft. Teacher forced on our drafts: the random assistant's top two
    logits lie within `NEAR` at about half its steps, so a chain of transformers' own picks stops at its first near tie
    (`test_gemma4_mtp_gpu`'s check, two or three steps in) and never reaches the deep steps. Few tokens lie within
    `NEAR` of the top (one to three of the 512, on CPU), and most drafts must be the top itself."""
    target, assistant = pair
    ht, ha = hf_model(target), hf_assistant(assistant)
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
    checked = exact = 0
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
            h, tok = out.last_hidden_state[:, p:p + 1], torch.tensor([[pending]])
            for j in range(k):
                res = ha(inputs_embeds=torch.cat([ht.model.embed_tokens(tok), h], -1),
                         position_ids=torch.tensor([[p]]), shared_kv_states=shared)
                lg = res.logits[0, -1]
                below = (lg.max() - lg[drafts[j]]).item()
                assert below < NEAR, (at, j, drafts, int(lg.argmax()), below)
                checked += 1
                exact += drafts[j] == int(lg.argmax())
                tok, h = torch.tensor([[drafts[j]]]), res.last_hidden_state
    assert checked >= 8 * k and exact > checked // 2, (checked, exact)

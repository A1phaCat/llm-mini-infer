"""Programmatic dependent launch in a full serving configuration on CUDA, in both runners: Gemma 4's FP8 target with
its assistant at k = 5 (through the scheduler in window mode), and the FP8 Qwen with its FP8 MTP layer at k = 3 under
the switches structured output turns on (always-split graphs, the forced prefix, the EOS-free finish); the
grammar-aware heads under JSON-like grammars, 24 requests on one cached prefix (decode graphs of up to 32 rows with the
drafter's chain, mixed steps in breakable graphs, every step split into its forward and sampling graphs), greedy and
seeded. PDL changes when kernels start, never what they compute: the tokens, the drafts accepted and the steps are the
serialized launches'. For the Qwen runner also its eager logits and its graphs' greedy tokens, FP8 and BF16 (Gemma 4's
are test_gemma4_fp8_gpu.py's). CUDA, the kernel library and the CUTLASS library only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.kernels import ops

from ..support import qwen_gpu
from ..support.fake_grammar import FakeGrammar
from ..support.gemma_gpu import fp8_pair as fp8_pair  # the fixture
from ..support.gemma_gpu import gpu_run
from ..support.gemma_heads_gpu import MANY, VOCAB, forced_json, json_like
from ..support.mtp_common import EOS, seq
from ..support.qwen_gpu import tiny as tiny  # the fixture: the tiny Qwen with its MTP layer, kernel-sized, in BF16
from ..support.qwen_mtp import Run
from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cuda_runner, make_tiny_fp8_model, pools

pytestmark = [pytest.mark.gpu,
              pytest.mark.cutlass(reason="needs a CUDA GPU, the kernel library and the CUTLASS library")]


def run(pair, pdl: bool):
    r = gpu_run(pair, 5, max_num_seqs=32, budget=256, num_pages=2048, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))),
                sparse_head=True, draft_universe=True, pdl=pdl)
    assert ops.pdl() == pdl and r.r.stats["pdl"] == pdl
    for i, p in enumerate(MANY):  # the first computes the prefix; odd requests sample (seeded)
        r.add(i, p, max_tokens=40, grammar=json_like(i), ignore_eos=False, temperature=0.8 if i % 2 else 0.0)
        if i == 0:
            r.run()
    r.run()
    assert r.sched.grammars.errors == 0
    r.r.shutdown()
    return r


def test_pdl_with_mtp_and_the_heads_is_bit_for_bit(fp8_pair) -> None:
    off = run(fp8_pair, False)
    on = run(fp8_pair, True)
    ops.set_pdl(False)
    assert "decode" in on.modes and "break" in on.modes
    assert on.r.head_counters().sparse > 0 and on.sched.spec_accepted > 0
    assert on.modes == off.modes
    assert on.out == off.out
    s_on, s_off = on.sched, off.sched
    assert (s_on.spec_draft_tokens, s_on.spec_accepted) == (s_off.spec_draft_tokens, s_off.spec_accepted)
    assert s_on.spec_accepted_per_pos.tolist() == s_off.spec_accepted_per_pos.tolist()


# -- the Qwen runner ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_fp8(tmp_path_factory) -> Path:
    """The tiny Qwen in block-FP8 with its block-FP8 MTP layer, kernel-sized (the real FP8 checkpoints' shape)."""
    return make_tiny_fp8_model(tmp_path_factory.mktemp("qwenpdlgpu"), seed=5, mtp=True, gpu=True)[0]


def qwen_run(model, pdl: bool) -> Run:
    """`qwen_gpu.MANY` on the Qwen runner as structured output configures it: the first request computes the shared
    prefix, the others hit its checkpoint (decode steps of up to 23 rows with the MTP chain, mixed steps); odd ones
    sample (seeded)."""
    r = qwen_gpu.gpu_run(model, 3, max_num_seqs=32, budget=256, num_pages=2048, caching=True, ckpt_slots=8,
                         grammars=Grammars(FakeGrammar(qwen_gpu.VOCAB, (qwen_gpu.EOS,))), forced_prefix=True,
                         eos_free_finish=True, sparse_head=True, draft_universe=True, always_split=True, pdl=pdl)
    assert ops.pdl() == pdl and r.r.stats["pdl"] == pdl
    for i, p in enumerate(qwen_gpu.MANY):
        r.add(i, p, max_tokens=40, grammar=forced_json(i), ignore_eos=False, temperature=0.8 if i % 2 else 0.0)
        if i == 0:
            r.run()
    r.run()
    assert r.sched.grammars.errors == 0
    assert all(r.cached[i] == 64 for i in range(1, len(qwen_gpu.MANY))), r.cached
    r.r.shutdown()
    return r


def test_qwen_pdl_with_mtp_and_the_heads_is_bit_for_bit(qwen_fp8) -> None:
    """The Qwen runner's kernels wait and trigger between the FP8 GEMMs, FlashInfer's attention and GDN prefill, the GDN
    decode and MTP kernels, in the decode graphs with the MTP chain, the breakable graphs and the sampling graphs."""
    try:
        off = qwen_run(qwen_fp8, False)
        on = qwen_run(qwen_fp8, True)
    finally:
        ops.set_pdl(False)
    assert "decode" in on.modes and "break" in on.modes
    c = on.r.head_counters()
    assert c.sparse > 0 and c.universe_steps > 0 and on.sched.spec_accepted > 0 and on.sched.forced_requests > 0, c
    assert on.modes == off.modes and on.sizes == off.sizes
    assert on.out == off.out and on.finish == off.finish
    assert c == off.r.head_counters()
    s_on, s_off = on.sched, off.sched
    assert (s_on.spec_draft_tokens, s_on.spec_accepted) == (s_off.spec_draft_tokens, s_off.spec_accepted)
    assert s_on.spec_accepted_per_pos.tolist() == s_off.spec_accepted_per_pos.tolist()
    assert (s_on.forced_tokens, s_on.eos_free.tolist()) == (s_off.forced_tokens, s_off.eos_free.tolist())


def qwen_logits(model, pdl: bool) -> tuple[torch.Tensor, list[int]]:
    """Eagerly: prefill chunks of 50, 63 and 37 tokens (rows starting mid-page and past the GDN prefill's chunks),
    then 10 decode steps; every position's logits and the greedy tokens."""
    ids = seq(150, 2)
    pc = pools(max_num_seqs=2, budget=64, num_pages=200)
    drv = Driver(cuda_runner(model, pc, graphs=False, pdl=pdl), pc)
    assert ops.pdl() == pdl and drv.r.stats["pdl"] == pdl
    steps = [Step(rows=[Row(1, ids[:50], 0, False)], admit=[(1, 1, GREEDY)]),
             Step(rows=[Row(1, ids[50:113], 50, False)]), Step(rows=[Row(1, ids[113:], 113)])]
    res = drv.run(steps + [Step(rows=[Row(1, None, 150 + i)]) for i in range(10)])
    drv.r.shutdown()
    gen = [int(r.tokens[0, 0]) for r in res if len(r.n_out) and r.n_out[0]]
    return torch.stack([drv.logits[(1, p)] for p in range(160)]), gen


def qwen_graph_tokens(model, pdl: bool) -> np.ndarray:
    """In graphs: two prompts in a breakable graph, then 20 decode graphs of both; each step's greedy tokens."""
    a, b = seq(90, 2), seq(37, 3)
    pc = pools(max_num_seqs=4, budget=128, num_pages=200)
    drv = Driver(cuda_runner(model, pc, graphs=True, pdl=pdl), pc, capture=False)
    assert ops.pdl() == pdl
    modes, mode = [], drv.r._mode

    def spy(lay, split=False):
        m = mode(lay, split)
        modes.append(m[0])
        return m

    drv.r._mode = spy
    steps = [Step(rows=[Row(1, a, 0), Row(2, b, 0)], admit=[(1, 1, GREEDY), (2, 2, GREEDY)])]
    steps += [Step(rows=[Row(1, None, 90 + i), Row(2, None, 37 + i)]) for i in range(20)]
    toks = np.stack([r.tokens[:, 0] for r in drv.run(steps)])
    drv.r.shutdown()
    assert modes == ["break"] + ["decode"] * 20, modes
    return toks


@pytest.mark.parametrize("model", ["qwen_fp8", "tiny"])
def test_qwen_pdl_is_bit_for_bit(model, request) -> None:
    """test_gemma4_fp8_gpu.py's test_pdl_is_bit_for_bit for the Qwen runner, FP8 and BF16 (BF16: our norms, SiLU and
    sigmoid gate between cuBLAS GEMMs): eager logits with PDL on are the logits with it off bit for bit, and so are the
    graphs' greedy tokens; every path leaves the library's flag off for the next runner."""
    path = request.getfixturevalue(model)
    try:
        (off, gen_off), (on, gen_on) = qwen_logits(path, False), qwen_logits(path, True)
        toks_off, toks_on = qwen_graph_tokens(path, False), qwen_graph_tokens(path, True)
    finally:
        ops.set_pdl(False)
    assert torch.equal(off, on) and gen_off == gen_on
    assert np.array_equal(toks_off, toks_on)

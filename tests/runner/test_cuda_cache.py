"""The CUDA runner's prefix-cache paths on CPU (torch reference kernels, fp32): checkpoints, resumes, STATE_COPY,
KV_COPY and the conv-state gather, each compared with a cold run; then the scheduler's cache through the real
engine loop, compared with the cache disabled."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.runner.base import PoolConfig
from llm_mini_infer.runner.fake import PlanError
from llm_mini_infer.sched.plan import CKPT_END_OF_STEP, SAMPLE, PlanBuilder
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cpu_runner, make_tiny_model, pools

TOL = 2e-5
EOS = 511
PARAMS = (0.9, 0.95, 40, 0.02, 4321)  # temperature, top_p, top_k, min_p, seed


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("m"), seed=3)


@pytest.fixture(scope="module")
def runner(model):
    return cpu_runner(model, pools(max_num_seqs=4, budget=256, state_slots=12))


def seq(n: int, salt: int = 0) -> list[int]:
    return [(i * 37 + 11 + salt * 101) % 500 for i in range(n)]


def configure(runner, **kw):
    pc = pools(**{"max_num_seqs": 4, "budget": 256, "state_slots": 12, **kw})
    runner.configure(pc)
    return pc


def close(a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> bool:
    return (a - b).abs().max().item() < tol


def gdn_layers(runner) -> list:
    return [layer.linear_attn for layer in runner.model.layers if not layer.is_attn]


def test_resume_from_a_mid_chunk_checkpoint_equals_cold(runner) -> None:
    """One 150-token prefill takes checkpoints at 64 and 128 (FlashInfer-style, every 64 tokens of the row); a second
    slot shares the first 32 pages, restores the 128 checkpoint (STATE_COPY) and prefills only [128, 150): its
    logits and sampled tokens match the cold run."""
    s = seq(150)
    pc = configure(runner)
    d = Driver(runner, pc)
    cold = d.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, PARAMS)], ckpt=[(0, 64, 9), (0, 128, 10)])] +
                 [Step(rows=[Row(1, None, 150 + i)]) for i in range(5)])
    shared = d.pages[1][:32]
    warm = d.run([Step(rows=[Row(2, s[128:], 128)], admit=[(2, 2, PARAMS)], seq_len={2: 128},
                       set_pages={2: shared}, state_copy=[(10, 2)])] +
                 [Step(rows=[Row(2, None, 150 + i)]) for i in range(5)])
    assert [int(r.tokens[0, 0]) for r in warm] == [int(r.tokens[0, 0]) for r in cold]
    assert all(close(d.logits[(1, p)], d.logits[(2, p)]) for p in range(128, 155))


def test_checkpoints_equal_prefills_stopped_there(runner) -> None:
    """The mid-chunk checkpoints at 64 and 128 equal the final state of prefills that stop there (end-of-chunk
    checkpoints of separate runs)."""
    s = seq(150, 1)
    pc = configure(runner)
    d = Driver(runner, pc, capture=False)
    d.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, PARAMS)], ckpt=[(0, 64, 9), (0, 128, 10)]),
           Step(rows=[Row(2, s[:64], 0)], admit=[(2, 2, PARAMS)], ckpt=[(0, 64, 11)]),
           Step(rows=[Row(3, s[:128], 0)], admit=[(3, 3, PARAMS)], ckpt=[(0, 128, 8)])])
    p = runner.pools
    for mid, stopped in ((9, 11), (10, 8)):
        assert close(p.rec[mid], p.rec[stopped], 1e-5) and close(p.conv[mid], p.conv[stopped], 1e-5)


def test_conv_gather_is_exact(runner) -> None:
    """The conv state of a mid-chunk checkpoint at 128 is exactly the 3 conv inputs at positions 125-127 of that
    forward pass (captured from the GDN input projection)."""
    s = seq(150, 2)
    pc = configure(runner)
    d = Driver(runner, pc, capture=False)
    seen: dict[int, torch.Tensor] = {}
    hooks = [g.in_proj_qkvz.register_forward_hook(lambda mod, i, o, k=g.index: seen.setdefault(k, o.detach()))
             for g in gdn_layers(runner)]
    try:
        d.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, PARAMS)], ckpt=[(0, 128, 9)])])
    finally:
        for h in hooks:
            h.remove()
    conv_dim = runner.cfg.conv_dim
    for g in gdn_layers(runner):
        want = seen[g.index][125:128, :conv_dim].T
        assert torch.equal(runner.pools.conv[9, g.index], want), f"layer {g.index}"


def test_state_copy_kv_copy_and_end_of_step_checkpoints_are_exact(runner) -> None:
    """STATE_COPY and KV_COPY copy bit for bit; an end-of-step checkpoint equals the live state after the step."""
    s = seq(70, 3)
    pc = configure(runner)
    d = Driver(runner, pc, capture=False)
    d.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, PARAMS)], ckpt=[(0, 70, 9)]),
           Step(rows=[Row(1, None, 70)], ckpt=[(0, CKPT_END_OF_STEP, 10)], state_copy=[(9, 11)])])
    p = runner.pools
    assert torch.equal(p.rec[11], p.rec[9]) and torch.equal(p.conv[11], p.conv[9])
    assert torch.equal(p.rec[10], p.rec[1]) and torch.equal(p.conv[10], p.conv[1])
    src, dst = d.pages[1][16], d.alloc_pages(1)[0]  # positions 64-67: not written by the step that copies them
    d.run([Step(rows=[Row(1, None, 71)], kv_copy=[(src, dst)])])
    assert torch.equal(p.kv[:, dst], p.kv[:, src])


def test_off_page_hit_with_kv_copy_equals_cold(runner) -> None:
    """A hit at 70 (inside page 17 of 4-token pages): the partial page is copied (KV_COPY), the end-of-chunk checkpoint
    restored (STATE_COPY), and [70, 90) prefilled: the logits match a cold prefill of the 90 tokens."""
    s = seq(90, 4)
    pc = configure(runner)
    d = Driver(runner, pc)
    d.run([Step(rows=[Row(1, s[:70], 0)], admit=[(1, 1, PARAMS)], ckpt=[(0, 70, 9)]),
           Step(rows=[Row(3, s, 0)], admit=[(3, 3, PARAMS)])])
    q = d.alloc_pages(1)[0]
    d.run([Step(rows=[Row(2, s[70:], 70)], admit=[(2, 2, PARAMS)], seq_len={2: 70},
                set_pages={2: [*d.pages[1][:17], q]}, kv_copy=[(d.pages[1][17], q)], state_copy=[(9, 2)])])
    assert all(close(d.logits[(3, p)], d.logits[(2, p)]) for p in range(70, 90))


def test_checkpoint_violations_raise(runner) -> None:
    pc = configure(runner)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len)
    for ckpt, match in (((0, 50, 9), "offsets"), ((0, 64, 1), "live state"), ((1, 64, 9), "out of range")):
        configure(runner)
        b.begin()
        b.add_admit(1, 1, 0, *GREEDY)
        b.add_pages(1, 0, list(range(1, 40)))
        b.add_explicit_row(1, np.arange(150, dtype=np.int32), 0, SAMPLE | 2)
        b.add_ckpt(*ckpt)
        with pytest.raises(PlanError, match=match):
            runner.launch_forward(b.finish(1))


# -- the scheduler's cache through the engine loop -------------------------------------------------------------


def run_engine(runner, reqs: list[dict], *, caching: bool, max_num_seqs: int = 3, budget: int = 256,
               num_pages: int = 300, ckpt_slots: int = 8) -> tuple[dict[int, list[int]], Scheduler]:
    nss = max_num_seqs + 1 + ckpt_slots
    pc = PoolConfig(page_size=4, num_pages=num_pages, num_state_slots=nss, max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=budget, max_model_len=512, eos_token_ids=(EOS,))
    runner.logits_hook = None
    runner.configure(pc)
    sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=512, page_size=4,
                      num_pages=num_pages, num_state_slots=nss, eos_token_ids=(EOS,), prefix_caching=caching,
                      decode_checkpoint_interval=64)
    core = EngineCore(sched, runner)
    out: dict[int, list[int]] = {}
    pending = sorted(reqs, key=lambda r: r.get("arrive", 0))
    for turn in range(5000):
        while pending and pending[0].get("arrive", 0) <= turn:
            r = pending.pop(0)
            out[r["rid"]] = []
            core.handle(AddRequest(rid=r["rid"], frontend=0, prompt=np.asarray(r["prompt"], np.int32),
                                   max_tokens=r["max_tokens"], temperature=0.8, top_p=0.95, top_k=40, min_p=0.0,
                                   seed=r["seed"], stop_token_ids=[], ignore_eos=True))
        report = core.step()
        if report is not None:
            for fo in report.outputs.values():
                off = 0
                for rid, n in zip(fo.rids.tolist(), fo.n_new.tolist()):
                    out[rid].extend(fo.tokens[off:off + n].tolist())
                    off += n
        if not pending and not core.busy:
            return out, sched
    raise AssertionError("engine not idle")


def test_engine_cache_hits_match_the_cache_disabled(runner) -> None:
    """Shared 140-token prefixes (siblings arriving together, then later ones), a follow-up turn and a long decode:
    with the prefix cache the tokens are those of the cache-disabled run, and most of the prompt is reused."""
    base = seq(140, 5)
    reqs = [{"rid": i, "prompt": base + seq(9, 10 + i), "max_tokens": 6, "seed": 500 + i, "arrive": 0 if i < 3 else 4}
            for i in range(6)]
    reqs.append({"rid": 6, "prompt": seq(40, 7), "max_tokens": 100, "seed": 600, "arrive": 1})
    ref, _ = run_engine(runner, reqs, caching=False)
    reqs.append({"rid": 7, "prompt": seq(40, 7) + ref[6][:70] + [3, 4, 5], "max_tokens": 5, "seed": 700,
                 "arrive": 200})
    ref, _ = run_engine(runner, reqs, caching=False)
    got, sched = run_engine(runner, reqs, caching=True)
    assert got == ref
    assert sched.prefix_hits >= 4 * 128 + 64 and sched.cache_waits >= 2 and sched.ckpts_taken["decode"] >= 1
    sched.check_accounting()

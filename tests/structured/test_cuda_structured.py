"""Grammar masks in the CUDA runner on CPU (torch reference kernels, fp32): the split step (the forward in
`launch_forward`, sampling under the engine's masks in `launch_sample`), through the real scheduler and engine loop.

A fake grammar (`fake_grammar`) over the tiny model's vocabulary constrains some requests. The promises under test:
every constrained token is one its grammar allowed at that point (the engine's matcher would reject it otherwise),
and a request's tokens are the target's masked sampled sequence whatever the drafts, so MTP on and off give the same
output, greedy and seeded alike, with constrained and unconstrained requests in the same steps; and so do the
scheduler's grammar switches: the EOS-free finish, whose donations (the live GDN state at a closing token, folded under
MTP) a next turn resumes from exactly as from any checkpoint, and the forced prefix, whose tokens a final chunk's row
computes past a checkpoint at the prompt's end inside it (the GDN kernel's every-64 state), the drafter's KV at the
prompt's last position from the first of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, FinishReason
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.base import MASK_EMPTY
from llm_mini_infer.sched.plan import MASKED, SAMPLE, STATE_ZERO, PlanBuilder
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.fake_grammar import FakeGrammar, spec
from ..support.logprob_cases import seq
from ..support.tiny_qwen import cpu_runner, make_tiny_model, pools

EOS = 511
VOCAB = 512


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("so"), seed=3, mtp=True)


def run(model, k: int, reqs: list[dict], max_num_seqs: int = 4, budget: int = 96, *, caching: bool = False,
        turns: Any = None, stats: dict | None = None, watch: Any = None,
        **switches: bool) -> dict[int, tuple[list[int], int]]:
    """`reqs` through the scheduler and engine loop until idle: {rid: (tokens, finish reason)}. `caching`: the prefix
    cache, with checkpoint slots (and the default decode checkpoints, every 256 positions); `turns`: the next turns,
    made from these results, run after them (each may resume from what the first ones left); `switches`: the
    scheduler's grammar switches (`forced_prefix`, `eos_free_finish`); `stats`: gets the scheduler (`sched`) and each
    request's cached prompt tokens (`cached`); `watch(sched, runner, outputs)`: called after every engine turn."""
    pc = pools(max_num_seqs=max_num_seqs, budget=budget, mtp_k=k, num_pages=512,
               state_slots=max_num_seqs + 1 + (16 if caching else 0))
    r = cpu_runner(model, pc)
    grammar = FakeGrammar(VOCAB, (EOS,))
    sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=pc.max_model_len,
                      page_size=pc.page_size, num_pages=pc.num_pages, num_state_slots=pc.num_state_slots,
                      eos_token_ids=(EOS,), mtp_k=k, prefix_caching=caching, grammars=Grammars(grammar), **switches)
    core = EngineCore(sched, r)
    out: dict[int, list[int]] = {}
    fin: dict[int, int] = {}
    cached: dict[int, int] = {}

    def until_idle(reqs: list[dict]) -> None:
        for q in reqs:
            out[q["rid"]] = []
            core.handle(AddRequest(rid=q["rid"], frontend=0, prompt=np.asarray(q["prompt"], np.int32),
                                   max_tokens=q.get("max_tokens", 14), temperature=q.get("temperature", 0.0),
                                   top_p=q.get("top_p", 1.0), top_k=q.get("top_k", 0), min_p=0.0,
                                   seed=q.get("seed", 100 + q["rid"]), stop_token_ids=[],
                                   ignore_eos=q.get("ignore_eos", False), grammar=q.get("grammar", "")))
        n = 0
        while core.busy and n < 2000:
            rep = core.step()
            n += 1
            if watch is not None:
                watch(sched, r, out)
            if rep is None:
                continue
            for o in rep.outputs.values():
                cached.update(zip(o.first_rids.tolist(), o.first_cached.tolist()))
                off = 0
                for rid, nn, f in zip(o.rids.tolist(), o.n_new.tolist(), o.finish.tolist()):
                    out[rid].extend(o.tokens[off:off + nn].tolist())
                    off += nn
                    if f:
                        fin[rid] = f
        assert not core.busy and sched.grammars.errors == 0
        sched.check_accounting()

    until_idle(reqs)
    if turns is not None:
        more = turns({rid: (out[rid], fin.get(rid, 0)) for rid in out})
        until_idle(more)
        reqs = [*reqs, *more]
    if stats is not None:
        stats.update(sched=sched, cached=cached)
    for q in reqs:  # every constrained token was allowed where it was sampled
        if q.get("grammar"):
            m = grammar.instance(grammar.compile(q["grammar"]))
            for t in out[q["rid"]]:
                assert m.allowed()[t], (q["rid"], out[q["rid"]])
                assert m.is_stopped() or m.consume_tokens([t])  # a stopped grammar's EOS ends it, unconsumed
    return {rid: (out[rid], fin.get(rid, 0)) for rid in out}


REQS = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 4, 9)),
        dict(rid=1, prompt=seq(33, 1)),
        dict(rid=2, prompt=seq(9, 2), grammar=spec(2, 30, 40), max_tokens=12),
        dict(rid=3, prompt=seq(50, 3), grammar=spec(3, 0, 6))]


def test_greedy_masked_outputs_equal_with_and_without_mtp(model) -> None:
    base = run(model, 0, REQS)
    assert base[0][1] == FinishReason.STOP and base[0][0][-1] == EOS  # ended by its grammar (EOS forced at 9)
    assert len(base[2][0]) == 12
    for k in (1, 3):
        assert run(model, k, REQS) == base, k


@pytest.mark.parametrize("params", [(0.8, 1.0, 0), (1.0, 0.9, 0), (0.7, 1.0, 20)])
def test_seeded_masked_sampling_equals_without_mtp(model, params) -> None:
    t, p, top_k = params
    reqs = [{**q, "temperature": t, "top_p": p, "top_k": top_k} for q in REQS]
    base = run(model, 0, reqs)
    assert run(model, 3, reqs) == base


FIRST = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 9, 9, lead=2, tail=1)),  # its closing token forced
         dict(rid=1, prompt=seq(33, 1), max_tokens=40, ignore_eos=True),  # decoding past the others
         dict(rid=2, prompt=seq(70, 2), grammar=spec(2, 6, 6, lead=1, tail=1), max_tokens=20),
         dict(rid=3, prompt=seq(9, 3), grammar=spec(3, 4, 12))]


def next_turns(res: dict[int, tuple[list[int], int]]) -> list[dict]:
    """The constrained requests' next turns: each answer as a chat template holds it (with its EOS, or without it),
    then a message; one of them constrained again."""
    return [dict(rid=10, prompt=FIRST[0]["prompt"] + res[0][0] + seq(5, 7), grammar=spec(4, 5, 5, lead=1, tail=1)),
            dict(rid=11, prompt=FIRST[2]["prompt"] + res[2][0][:-1] + seq(4, 8)),
            dict(rid=12, prompt=FIRST[3]["prompt"] + res[3][0] + seq(3, 9), max_tokens=10)]


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_next_turns_resume_exactly_under_the_grammar_switches(model, k, temperature) -> None:
    """Grammar-ended first turns next to an unconstrained request, then their next turns, with the prefix cache, greedy
    and seeded: with the forced prefix and the EOS-free finish each on and off, the first turns give the tokens of
    k = 0 without the cache, and every next turn the tokens it gives cold (alone, without the cache). The next turns
    resume past their first turn's prompt, from its finish's donation (no decode checkpoint is due, and a prefill
    checkpoint lies inside the prompt): with the EOS-free finish, requests finished EOS-free and donated their live
    state at their closing token (folded one record into the dropped row under MTP), which a turn holding the answer
    with or without its EOS resumes from; without it, the finish computed the EOS, and only the turns holding it
    resume past the prompt. The forced prefix changes neither."""
    first = [{**q, "temperature": temperature} for q in FIRST]
    base = run(model, 0, first)
    assert all(base[i][1] == FinishReason.STOP and base[i][0][-1] == EOS for i in (0, 2, 3)), base
    cold = run(model, 0, next_turns(base))
    for fp, ef in ((False, False), (True, False), (False, True), (True, True)):
        stats: dict = {}
        got = run(model, k, first, caching=True, turns=next_turns, stats=stats, forced_prefix=fp, eos_free_finish=ef)
        assert {rid: got[rid] for rid in base} == base and {rid: got[rid] for rid in cold} == cold, (fp, ef)
        s, cached = stats["sched"], stats["cached"]
        assert s.ckpts_taken["finish"] and (s.eos_free[1] > 0) == ef and (s.forced_tokens > 0) == fp
        assert cached[10] > len(FIRST[0]["prompt"]) and cached[12] > len(FIRST[3]["prompt"]), cached
        assert (cached[11] > len(FIRST[2]["prompt"])) == ef, cached  # its answer without the EOS


def at_the_prompts_end(n: int) -> tuple[dict, dict]:
    """A constrained request (its prompt of n tokens: two chunks, [0, 64) and the rest, under the budget of 96) and
    one whose prompt extends it, arriving with it, which waits for a branch checkpoint at n that the first one's final
    row takes (`Scheduler._branch_wait`): that row's record lies at its prompt's end."""
    prompt = seq(n, 5)
    return (dict(rid=10, prompt=prompt, grammar=spec(8, 12, 12, lead=2, tail=1), max_tokens=20),
            dict(rid=11, prompt=prompt + seq(3, 9), max_tokens=10))


@pytest.mark.parametrize("n", [120, 128])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_the_forced_prefix_and_a_checkpoint_at_the_prompts_end(model, k, n) -> None:
    """`at_the_prompts_end` with the prefix cache: the branch checkpoint lies at the writer's prompt's end, 64 tokens
    into its final row (n = 128: on the kernel's every-64 grid, so its forced tokens follow it and the record is inside
    the row) or 56 (n = 120: off it, so the writer samples them). With the forced prefix on and off, both requests'
    tokens are those they give cold (alone, without the cache, k = 0), the waiting one resumed at the prompt's end."""
    a, b = at_the_prompts_end(n)
    cold = run(model, 0, [a]) | run(model, 0, [b])
    for fp in (False, True):
        stats: dict = {}
        got = run(model, k, [a, b], caching=True, stats=stats, forced_prefix=fp)
        assert got == cold, fp
        s, cached = stats["sched"], stats["cached"]
        assert (cached[10], cached[11]) == (0, n) and s.ckpts_taken["branch"] == 1
        kept = fp and n % 64 == 0
        assert (s.forced_tokens, s.forced_declined) == ((2, 0) if kept else (0, int(fp)))


def drafter_kv(rid: int, n: int, outputs: int, kv: dict) -> Any:
    """A `run` watch: the drafter's KV over positions [0, n) of request `rid` (`kv[rid]`), read once it has `outputs`
    output tokens (positions it committed past are final: the drafter's KV at i depends on the token at i + 1)."""
    def watch(sched: Scheduler, r: Any, out: dict[int, list[int]]) -> None:
        if rid in kv or len(out.get(rid, ())) < outputs:
            return
        slot = next(s for s in range(1, len(sched.slot_req)) if sched.slot_req[s] is not None and
                    sched.slot_req[s].rid == rid)
        layer, ps = r.model.mtp.layers[0].self_attn.index, r.pools.page_size
        pos = torch.arange(n)
        kv[rid] = r.pools.kv[layer][r.tables.page_table[slot, pos // ps].long(), :, pos % ps].clone()

    return watch


@pytest.mark.parametrize("k", [1, 3])
def test_the_drafters_kv_around_the_forced_prefix(model, k) -> None:
    """The MTP drafter's KV (`at_the_prompts_end`, n = 128): the writer's, over its prompt and its first forced token,
    is the same with the forced prefix on, its final row computing it at the prompt's last position from that token,
    and off, from the sample there (the token its one-token mask allows), then from a verify row; and the waiting
    request, resuming from the checkpoint inside that row (holding the hidden vector at the prompt's last position),
    has the drafter KV over its prompt it has cold."""
    a, b = at_the_prompts_end(128)
    na, nb = len(a["prompt"]) + 1, len(b["prompt"]) - 1
    cold: dict = {}
    run(model, k, [b], watch=drafter_kv(11, nb, 1, cold))
    seen = []
    for fp in (False, True):
        kv: dict = {}
        watches = (drafter_kv(10, na, 4, kv), drafter_kv(11, nb, 1, kv))
        run(model, k, [a, b], caching=True, forced_prefix=fp, watch=lambda *x, w=watches: [f(*x) for f in w])
        torch.testing.assert_close(kv[11], cold[11], atol=1e-5, rtol=1e-5)
        seen.append(kv[10])
    torch.testing.assert_close(seen[1], seen[0], atol=1e-5, rtol=1e-5)


def test_an_empty_mask_flags_the_row(model) -> None:
    """Directly through the runner: a MASKED final chunk whose mask allows nothing samples token 0 with MASK_EMPTY;
    an all-ones mask samples as no mask does."""
    pc = pools(max_num_seqs=2, budget=64, mtp_k=0, num_pages=64)
    r = cpu_runner(model, pc)
    toks = {}
    for name, fill in (("none", None), ("all", -1), ("empty", 0)):
        r.reset()
        b = PlanBuilder(2, 64, pc.page_size, pc.max_model_len, 0)
        b.begin()
        b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, seed=5)
        b.add_pages(1, 0, list(range(1, 9)))
        b.add_explicit_row(1, np.asarray(seq(12), np.int32), 0, STATE_ZERO | SAMPLE | (MASKED if fill is not None
                                                                                        else 0))
        h = r.launch_forward(b.finish(1))
        n = 0
        if fill is not None:
            r.mask_buffer(h)[:1] = fill
            n = 1
        r.launch_sample(h, n)
        res = r.wait(h)
        toks[name] = (int(res.tokens[0, 0]), None if res.status is None else int(res.status[0]))
    assert toks["all"] == toks["none"] and toks["none"][1] is None
    assert toks["empty"] == (-1, MASK_EMPTY) or toks["empty"][1] == MASK_EMPTY


@torch.inference_mode()
def test_masked_logits_reference() -> None:
    """The reference sampler: a masked row's token is the masked argmax / Gumbel-max; the non-finite check reads the
    logits before the mask."""
    from llm_mini_infer.runner.cuda.sampler import plan_sampling, row_keys, sample

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 64, generator=g)
    logits[2, 5] = float("nan")
    allowed = torch.zeros(3, 64, dtype=torch.bool)
    allowed[0, [3, 9, 40]] = True
    allowed[1, :] = True
    bits = np.packbits(allowed.numpy(), axis=1, bitorder="little").view(np.int32)
    mask = torch.from_numpy(bits.copy())
    rows = torch.tensor([0, 1, 0, 2])
    t = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    plan = plan_sampling(t, np.ones(4, np.float32), np.zeros(4, np.int64), np.zeros(4, np.float32), 64)
    toks, st = sample(logits, plan, torch.zeros(4), torch.ones(4), torch.zeros(4, dtype=torch.int64), torch.zeros(4),
                      row_keys(torch.arange(4), torch.arange(4)), torch.arange(64), mask=mask, mask_row=rows)
    assert int(toks[0]) == [3, 9, 40][int(logits[0, [3, 9, 40]].argmax())]
    assert int(toks[1]) == int(logits[1].argmax())  # an all-ones mask
    assert int(st[2]) == 1 and int(toks[2]) == 0  # NaN wins over the mask
    assert int(st[3]) == MASK_EMPTY and int(toks[3]) == 0

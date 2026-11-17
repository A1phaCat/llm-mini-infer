"""Always-split graphs in the Qwen runner on CPU (`CudaRunner`, `dev.runner.cuda.always_split`; architecture.md, "CUDA
graphs"): with grammar masks on, every step with rows runs as a forward and a sampling part, masked or not, a prompt's
chunk without sample rows included, and the outputs are the fused path's: tokens, finishes and accepted drafts, greedy
and seeded, with prefix-cache hits and checkpoints, and with images; on CPU no step reports a forward still running.
The torch reference kernels in fp32 (`test_cuda_mtp`'s tiny model and harness, the real scheduler and the engine
loop); Gemma 4's twin is `test_always_split.py`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar, spec
from ..support.mtp_common import EOS, seq
from ..support.qwen_images import Engine, Img, prompt_of, text
from ..support.qwen_images import vl as vl  # the fixture: a tiny multimodal Qwen with its MTP layer
from ..support.qwen_mtp import Run
from ..support.qwen_mtp import model as model  # the fixture: a tiny Qwen with its MTP layer

VOCAB = 512
BUDGET = 32  # a 50-token prompt in two chunks: the first a step without sample rows


def requests(constrained: bool) -> list[dict]:
    """Four requests, the first's prompt in two chunks; with `constrained`, the first two under a JSON-like grammar
    (about 8 of 128 tokens per state, the EOS from 4 tokens on, forced at 10)."""
    out = []
    for i, n in enumerate((50, 11, 23, 4)):
        q = {"rid": i, "prompt": seq(n, i + 1), "max_tokens": 12}
        if constrained and i < 2:
            q["grammar"] = spec(i, 4, 10, one_in=16, universe=128)
        out.append(q)
    return out


def go(model, k: int, reqs: list[dict], *, split: bool, temperature: float = 0.0,
       oracle: dict[int, list[int]] | None = None, rng: np.random.Generator | None = None, budget: int = BUDGET,
       **kw) -> Run:
    """The requests through the engine loop (`Run.run`'s `oracle`); `steps`: per launched step, (split, rows, sample
    rows); `states`: per step once it sampled (a fused one: in its forward), every state slot, live and checkpoint
    (the GDN states, conv windows and under MTP the drafter's hidden vectors); `results`: per waited step, its tokens,
    counts and drafts."""
    r = Run(model, k, budget=budget, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), always_split=split, **kw)
    r.steps, r.states, r.results = [], [], []
    launch, sample, wait = r.r.launch_forward, r.r.launch_sample, r.r.wait

    def spy(plan, *a, **kw):
        h = launch(plan, *a, **kw)
        r.steps.append((h.split is not None, h.n_sched, h.n_sample))
        return h

    def sampled(h, *a, **kw):
        sample(h, *a, **kw)
        p = r.r.pools
        r.states.append([None if x is None else x.clone() for x in (p.rec, p.conv, p.hid)])

    def waited(h):
        res = wait(h)
        r.results.append((res.tokens.copy(), res.n_out.copy(), None if res.drafts is None else res.drafts.copy()))
        return res

    r.r.launch_forward, r.r.launch_sample, r.r.wait = spy, sampled, waited
    for q in reqs:
        r.add(q["rid"], q["prompt"], max_tokens=q["max_tokens"], temperature=temperature, ignore_eos=False,
              grammar=q.get("grammar", ""))
    r.run(oracle, rng)
    assert r.sched.grammars.errors == 0
    return r


def same(fused: Run, split: Run) -> None:
    """The same plans, outputs, every step's drafts (the drafter's, accepted or not), drafts accepted and every step's
    state slots once it sampled (a checkpoint taken too early can hold a state no output shows); every split run's step
    with rows split."""
    assert split.out == fused.out and split.finish == fused.finish
    assert (split.sched.spec_drafts, split.sched.spec_accepted) == (fused.sched.spec_drafts, fused.sched.spec_accepted)
    assert [s[1:] for s in split.steps] == [s[1:] for s in fused.steps]
    assert len(split.results) == len(fused.results)
    for a, b in zip(split.results, fused.results):
        assert all(np.array_equal(x, y) if x is not None else y is None for x, y in zip(a, b))
    assert len(split.states) == len(fused.states)
    for i, (a, b) in enumerate(zip(split.states, fused.states)):
        for name, x, y in zip(("rec", "conv", "hid"), a, b):
            assert (x is None and y is None) or torch.equal(x, y), (i, name)
    assert all(s for s, n, _ in split.steps if n)


@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("constrained", [False, True])
def test_always_split_gives_the_fused_paths_tokens(model, k, constrained) -> None:
    reqs = requests(constrained)
    for temperature in (0.0, 0.8):
        fused = go(model, k, reqs, split=False, temperature=temperature)
        split = go(model, k, reqs, split=True, temperature=temperature)
        same(fused, split)
        assert any(n and not ns for _, n, ns in split.steps)  # a prompt's first chunk: no sample rows, split too
        assert any(s for s, _, _ in fused.steps) == constrained  # fused, only the steps with masks split
        if k:  # the oracle's drafts (`test_cuda_mtp`), a random number of them right at each step: accepted alike.
            # Unconstrained requests only: a grammar's masks follow the drafts the scheduler was given, not the oracle's
            out = fused.out
            ref = {q["rid"]: out[q["rid"]] for q in reqs if not q.get("grammar")}
            fused = go(model, k, reqs, split=False, temperature=temperature, oracle=ref, rng=np.random.default_rng(k))
            split = go(model, k, reqs, split=True, temperature=temperature, oracle=ref, rng=np.random.default_rng(k))
            same(fused, split)
            assert split.out == out and split.sched.spec_accepted > 0


@pytest.mark.parametrize("budget", [32, 40, 96])
def test_always_split_with_cache_hits_and_checkpoints(model, budget) -> None:
    """Prefix-cache hits under MTP (FOLD on donations, the drafter's KV recomputed at a hit) with prompts in chunks,
    constrained and not, from end-of-chunk checkpoints (budgets 32 and 40: a split step's sampling part copies them;
    without those copies outputs and drafts change, and the state slots show copies made before it) and mid-chunk ones
    (96): the split run's outputs, drafts, state slots, hits and checkpoints are the fused run's
    (`test_cuda_mtp.test_cache_hits_resume_exactly_with_the_drafter`'s turns). Decode checkpoints with drafts
    accepted, which these turns do not take: below."""
    base = seq(130, 11)
    first = [base + seq(5, 1), base + seq(9, 2)]
    grammar = spec(5, 4, 12, one_in=16, universe=128)
    runs = {split: go(model, 3, [dict(rid=i, prompt=p, max_tokens=20, grammar=grammar if i else "")
                                 for i, p in enumerate(first)],
                      split=split, caching=True, ckpt_slots=8, interval=64, budget=budget)
            for split in (False, True)}
    out = runs[False].out
    assert runs[True].out == out
    turns = [first[0] + out[0] + seq(4, 5), first[1] + out[1][:13] + seq(6, 6), base + seq(3, 9)]
    for r in runs.values():
        for i, p in enumerate(turns):
            r.add(10 + i, p, max_tokens=12, ignore_eos=False, grammar=grammar if i == 2 else "")
        r.run()
    fused, split = runs[False], runs[True]
    same(fused, split)
    assert split.cached == fused.cached and split.sched.ckpts_taken == fused.sched.ckpts_taken
    assert split.cached[10] >= len(first[0]) and min(split.cached[10 + i] for i in range(3)) > 100
    assert split.sched.ckpts_taken["prefill"] and split.sched.ckpts_taken["finish"]


def test_always_split_with_decode_checkpoints_and_accepted_drafts(model) -> None:
    """Decode rows' checkpoints (every 16 tokens) under MTP with drafts accepted (the oracle's, a random number right
    at each step): a split step's sampling part commits each row's accepted drafts (`fold_n`), and only then are its
    rows' live states copied and folded into checkpoints (`launch_sample`: `_after_body`). Folded before it, a
    checkpoint holds the previous step's commit, which no output of these turns shows: every step's state slots are
    the fused run's, and the later turns' outputs, hits and checkpoints."""
    base = seq(40, 11)
    first = [base + seq(5, 1), base + seq(9, 2)]
    reqs = [dict(rid=i, prompt=p, max_tokens=40) for i, p in enumerate(first)]
    kw = dict(caching=True, ckpt_slots=8, interval=16, budget=64)
    ref = go(model, 3, reqs, split=False, **kw).out
    runs = {split: go(model, 3, reqs, split=split, oracle=ref, rng=np.random.default_rng(1), **kw)
            for split in (False, True)}
    fused, split = runs[False], runs[True]
    same(fused, split)
    assert split.out == ref and split.sched.spec_accepted > 0 and split.sched.ckpts_taken["decode"]
    turns = [first[0] + ref[0][:30] + seq(4, 5), first[1] + ref[1][:33] + seq(6, 6)]
    for r in runs.values():
        for i, p in enumerate(turns):
            r.add(10 + i, p, max_tokens=12, ignore_eos=False)
        r.run()
    same(fused, split)
    assert split.cached == fused.cached and split.sched.ckpts_taken == fused.sched.ckpts_taken
    assert all(split.cached[10 + i] > len(p) for i, p in enumerate(first))  # inside the first turns' outputs


@pytest.mark.parametrize("k", [0, 2])
def test_always_split_with_images(vl, tmp_path, k) -> None:
    """An image between text, prefilled in 8-token chunks that split its span, then greedy decoding (eager: the logits
    hook): split, every prompt position's logits and the tokens are the fused run's, bit for bit."""
    path, _ = vl
    ids, spans = prompt_of([text(9), Img(4, 6, 1), text(7, 1)])
    runs = {}
    for split in (False, True):
        (tmp_path / str(split)).mkdir()
        e = Engine(path, tmp_path / str(split), budget=8, k=k, always_split=split)
        e.splits = []
        launch = e.runner.launch_forward

        def spy(plan, *a, e=e, launch=launch, **kw):
            h = launch(plan, *a, **kw)
            e.splits.append(h.split is not None)
            return h

        e.runner.launch_forward = spy
        e.add(0, ids, spans, max_tokens=8)
        e.run()
        runs[split] = e
    fused, split = runs[False], runs[True]
    assert all(split.splits) and not any(fused.splits) and len(split.splits) == len(fused.splits) > 4
    assert split.out == fused.out and len(split.out[0]) == 8
    assert split.logits.keys() == fused.logits.keys()
    assert all(torch.equal(split.logits[key], fused.logits[key]) for key in fused.logits)
    assert split.sched.encoder_images == 1


def test_a_cpu_step_never_reports_its_forward_running(model) -> None:
    """The Qwen runner reports no forward's end (`engine.plan_at_boundary` stays Gemma 4's, config.py), and on CPU no
    step times a sampling part."""
    r = go(model, 3, requests(False), split=True)
    assert r.r.stats["always_split"] is True
    assert not r.r.capabilities().forward_end
    h = r.r._staging[0].handle
    assert r.r.forward_pending(h) is False and h.result.gpu_sample_ns == 0
    assert r.core.gpu_sample_ns == 0


def test_always_split_needs_the_grammar_masks(model) -> None:
    with pytest.raises(ValueError, match="need the grammar masks"):
        Run(model, 0, always_split=True, grammar=False)

"""Always-split graphs in GemmaRunner on CPU (`dev.runner.cuda.always_split`; architecture.md, "CUDA graphs"): with
grammar masks on, every step with rows runs as a forward and a sampling part, masked or not, and the outputs are the
fused path's; a step without masks samples on the dense head and moves none of the grammar heads' counters; on CPU no
step reports a forward still running (`forward_pending`). The torch reference kernels in fp32 (`test_gemma4_mtp`'s
tiny pair, the scheduler in window mode and the engine loop); the graphs' twin is `test_always_split_gpu.py`."""

from __future__ import annotations

import pytest

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar
from ..support.gemma_heads import VOCAB, json_like
from ..support.gemma_heads import pair as pair  # the fixture: a tiny target and its assistant
from ..support.gemma_mtp import Run
from ..support.mtp_common import EOS, seq


def requests(constrained: bool) -> list[dict]:
    """Four requests, two past the window; with `constrained`, the first two under a JSON-like grammar."""
    out = []
    for i, n in enumerate((11, 23, 4, 30)):
        q = {"rid": i, "prompt": seq(n, i + 1), "max_tokens": 12}
        if constrained and i < 2:
            q["grammar"] = json_like(i, 4, 10)
        out.append(q)
    return out


def go(pair, k: int, reqs: list[dict], *, split: bool, temperature: float = 0.0) -> Run:
    grammar = FakeGrammar(VOCAB, (EOS,))
    r = Run(pair, k, grammars=Grammars(grammar), caching=False, sparse_head=True, draft_universe=k > 0,
            always_split=split)
    r.splits = []
    launch = r.r.launch_forward

    def spy(plan, *a, **kw):
        h = launch(plan, *a, **kw)
        r.splits.append(h.split is not None)
        return h

    r.r.launch_forward = spy
    for q in reqs:
        r.add(q["rid"], q["prompt"], max_tokens=q["max_tokens"], temperature=temperature, ignore_eos=False,
              grammar=q.get("grammar", ""))
    r.run()
    assert r.sched.grammars.errors == 0
    return r


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("constrained", [False, True])
def test_always_split_gives_the_fused_paths_tokens(pair, k, constrained) -> None:
    for temperature in (0.0, 0.8):
        fused = go(pair, k, requests(constrained), split=False, temperature=temperature)
        split = go(pair, k, requests(constrained), split=True, temperature=temperature)
        assert split.out == fused.out and split.finish == fused.finish, (k, constrained, temperature)
        assert split.sched.spec_accepted == fused.sched.spec_accepted
        assert all(split.splits) and len(split.splits) == len(fused.splits)
        # fused, only the steps with masks split; the heads' counters count those alone either way
        assert any(fused.splits) == constrained
        assert split.r.head_counters() == fused.r.head_counters()


def test_a_cpu_step_never_reports_its_forward_running(pair) -> None:
    r = go(pair, 3, requests(False), split=True)
    assert not r.r.capabilities().forward_end  # no device: nothing to plan around
    assert r.r.forward_pending(r.r._staging[0].handle) is False


def test_always_split_needs_the_grammar_masks(pair) -> None:
    with pytest.raises(ValueError, match="need the grammar masks"):
        Run(pair, 0, always_split=True, grammar=False)

"""Logprobs in both CUDA runners on CPU (torch reference kernels, fp32), through the real scheduler and engine loop:
a request's tokens are the same with logprobs and without, on the dense head, the sparse head and the draft universe,
under MTP; and each token's values are their definition (`runner.cuda.logprobs.reference`) on the logits the runner
computed at its position (its logits hook), its grammar's mask applied: the right sample's row, a forced or appended EOS
token's one-token mask; under MTP, a delivered token's are its own window position's (the hook labels each token with
the step's own position), and those without MTP, up to the rows' rounding."""

from __future__ import annotations

import pytest
import torch

from llm_mini_infer.runner.cuda.logprobs import reference

from ..support.fake_grammar import FakeGrammar
from ..support.logprob_cases import EOS, REQS, VOCAB, run
from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma
from ..support.tiny_qwen import make_tiny_model


@pytest.fixture(scope="module")
def qwen(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("lpq"), seed=3, mtp=True)


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    d = tmp_path_factory.mktemp("lpg")
    target = make_tiny_gemma(d / "target", window=8, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def check_values(reqs: list[dict], out: dict, logits: dict) -> int:
    """Each value against its definition on the logits at the position before its token, the mask the grammar had
    there (`reference`); the top ids up to near ties (the hook's GEMM has other rows than the sampler's). Returns how
    many tokens were checked."""
    grammar = FakeGrammar(VOCAB, (EOS,))
    checked = 0
    for q in reqs:
        toks, _, vals = out[q["rid"]]
        k = q.get("logprobs", -1)
        if k < 0:
            assert not vals
            continue
        assert len(vals) == len(toks), q["rid"]
        m = grammar.instance(grammar.compile(q["grammar"])) if q.get("grammar") else None
        for i, (t, (v, ids, top)) in enumerate(zip(toks, vals)):
            allowed = torch.from_numpy(m.allowed().copy()) if m is not None else None
            x = logits[(q["rid"], len(q["prompt"]) + i - 1)]
            want, _, want_top = reference(x, t, allowed)
            assert abs(v - want) < 2e-5, (q["rid"], i, v, want)
            assert len(ids) == len(top) >= k  # the message's widest request's (`Logprobs`)
            ids, top = ids[:k], top[:k]
            full = torch.log_softmax(x if allowed is None else x.masked_fill(~allowed, float("-inf")), -1)
            for j, (a, b) in enumerate(zip(ids, top)):
                assert abs(b - want_top[j]) < 2e-5 or b == want_top[j], (q["rid"], i, j, b, want_top[j])
                assert abs(float(full[a]) - b) < 2e-5 or float(full[a]) == b, (q["rid"], i, j)
            if m is not None:
                if t == EOS:
                    m = None
                else:
                    assert m.consume_tokens([t])
            checked += 1
    return checked


def same_values(a: dict, b: dict) -> None:
    """Two runs' values of the same tokens: equal up to rounding, the top ids up to near ties."""
    for rid in a:
        assert a[rid][0] == b[rid][0]
        for (v1, i1, t1), (v2, i2, t2) in zip(a[rid][2], b[rid][2]):
            assert abs(v1 - v2) < 2e-5 and all(x == y or abs(x - y) < 2e-5 for x, y in zip(t1, t2)), rid
            assert all(x == y or abs(p - q) < 2e-5 for x, y, p, q in zip(i1, i2, t1, t2)), rid


def invariant(kind: str) -> dict:
    """Batch invariance's switches of the runner (`runner.cuda.make_cuda_runner`'s, those a CPU runner takes); a hybrid
    model's budget then leaves a 64-token grid and a forced prefix beside the decode rows."""
    return dict(batch_invariant=True, **({"head_fixed_m": True} if kind == "gemma" else {"budget": 128}))


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
@pytest.mark.parametrize("heads", ["dense", "sparse", "universe", "invariant"])
def test_tokens_unchanged_and_values_from_the_sampled_logits(qwen, gemma, kind, heads) -> None:
    kw = dict(always_split=True, sparse_head=heads != "dense", **(invariant(kind) if heads == "invariant" else {}))
    model = qwen if kind == "qwen" else gemma
    seen: dict = {}
    base, logits = run(kind, model, 0, REQS, True, seen, **kw)
    assert check_values(REQS, base, logits) >= 40
    assert (seen["heads"].sparse > 0) == (heads != "dense")
    for k in (0, 3):
        kk = {**kw, "draft_universe": heads in ("universe", "invariant") and k > 0}
        plain, _ = run(kind, model, k, REQS, False, **kk)
        got, lg = run(kind, model, k, REQS, True, **kk) if k else (base, None)
        assert {r: v[:2] for r, v in got.items()} == {r: v[:2] for r, v in plain.items()}, k
        if k:
            assert check_values(REQS, got, lg) >= 40
            same_values(base, got)

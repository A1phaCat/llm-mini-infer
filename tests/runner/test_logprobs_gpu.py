"""Logprobs on CUDA (`runner.cuda.logprobs`): the pass selects the CPU's tokens exactly (its keys are unique) with
values up to the sums' order, on the dense and the sparse head, under masks and many ties, with no sync; and both
runners on CUDA graphs (always split, the sparse head, MTP), under the suite's sync check at error: a request's tokens
are the same with logprobs and without, and its values transformers' fp32 ones up to a BF16 run's error."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.runner.cuda.logprobs import CHUNK, LogprobsPass

from ..support.gemma_gpu import W
from ..support.gemma_gpu import cuda_runner as gemma_cuda_runner
from ..support.gemma_gpu import pair as pair  # the fixture: the tiny Gemma pair in BF16
from ..support.logprob_cases import REQS, run
from ..support.mask_words import words_of
from ..support.qwen_gpu import tiny as qwen_tiny  # noqa: F401 - the fixture: the tiny Qwen in BF16
from ..support.tiny_gemma import hf_logits as gemma_hf_logits
from ..support.tiny_qwen import QWEN38_ATTN, make_tiny_model
from ..support.tiny_qwen import cuda_runner as qwen_cuda_runner
from ..support.tiny_qwen import hf_logits as qwen_hf_logits

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
DEV = torch.device("cuda")
V = 262_144
NEAR = 0.3  # twice a BF16 run's largest logit error against transformers in fp32 (qwen_gpu, gemma_gpu)
# Batch invariance's runner switches (`runner.cuda.make_cuda_runner`'s; test_batch_invariance_gpu's, without images)
INVARIANT = {"gemma": dict(batch_invariant=True, head_fixed_m=True, window_split_keys=16, attn512_split_keys=0),
             "qwen": dict(batch_invariant=True, attn_split_keys=64)}  # splits of 64 keys: the contexts span several


def both(logits: torch.Tensor, tokens: torch.Tensor, ids: torch.Tensor | None, words: torch.Tensor | None,
         mrow: torch.Tensor | None, cap: int = 0) -> tuple:
    """The pass on CUDA (under torch's sync debug mode at error) and on CPU, on the same inputs."""
    out = []
    for dev in (DEV, torch.device("cpu")):
        n = len(logits)
        lp = LogprobsPass(n, V, cap, dev)
        (lp.sparse if ids is not None else lp.dense)[:n] = logits.to(dev)
        idx = torch.tensor(list(range(n)) + [0] * (-n % CHUNK), dtype=torch.int64, device=dev)
        mask = (words.to(dev), mrow.to(dev)) if words is not None else None
        t, cols = tokens.to(dev), ids.to(dev) if ids is not None else None
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
        try:
            lp.run(0, idx, n, t, cols, mask)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        out.append((lp.tok[0][:n].cpu(), lp.ids[0][:n].cpu(), lp.top[0][:n].cpu()))
    return out


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    fin = torch.isfinite(a)
    return torch.equal(fin, torch.isfinite(b)) and torch.equal(a[~fin], b[~fin]) and \
        bool(((a[fin] - b[fin]).abs() <= 1e-5 + 1e-6 * b[fin].abs()).all())


@pytest.mark.parametrize("rows", ["bf16", "ternary"])
def test_the_pass_on_cuda_is_the_cpus(rows) -> None:
    """Gemma 4's vocabulary, BF16-rounded or ternary logits (tens of thousands tied at the cut), rows unmasked and
    under masks of most of the vocabulary, of 25 tokens, of 3 (fillers) and of one."""
    rng = np.random.default_rng(1)
    n = 3 * CHUNK - 2
    if rows == "bf16":
        x = (torch.randn((n, V), generator=torch.Generator().manual_seed(1)) * 4).bfloat16().float()
    else:
        x = torch.from_numpy(rng.integers(-1, 2, (n, V)).astype(np.float32))
    allowed = np.zeros((4, V), bool)
    allowed[0] = rng.random(V) < 0.7
    allowed[1, rng.choice(V, 25, replace=False)] = True
    allowed[2, [7, 1000, 200_000]] = True
    allowed[3, 99] = True
    mrow = torch.tensor([i % 5 - 1 for i in range(n)], dtype=torch.int64)
    tokens = torch.tensor([int(np.flatnonzero(allowed[m])[0]) if m >= 0 else i for i, m in enumerate(mrow.tolist())])
    (ga, gi, gt), (ca, ci, ct) = both(x, tokens, None, words_of(allowed), mrow)
    assert torch.equal(gi, ci)
    assert close(ga, ca) and close(gt, ct)


def test_the_sparse_head_on_cuda_is_the_cpus() -> None:
    rng = np.random.default_rng(2)
    n, cap = CHUNK + 3, 1024
    allowed = np.zeros((n, V), bool)
    for i in range(n):
        allowed[i, rng.choice(V, int(rng.integers(1, 60)), replace=False)] = True
    allowed[:, [3, 40]] |= rng.random((n, 2)) < 0.3  # ids below 64 in the union, allowed or not
    union = np.flatnonzero(allowed.any(0))
    ids = torch.full((cap,), -1, dtype=torch.int32)
    ids[:len(union)] = torch.from_numpy(union.astype(np.int32))
    sparse = (torch.randn((n, cap)) * 3).bfloat16().float()
    tokens = torch.tensor([int(np.flatnonzero(a)[-1]) for a in allowed])
    (ga, gi, gt), (ca, ci, ct) = both(sparse, tokens, ids, words_of(allowed), torch.arange(n), cap)
    assert torch.equal(gi, ci) and close(ga, ca) and close(gt, ct)


def test_a_samples_values_on_cuda_do_not_depend_on_its_chunk() -> None:
    x = (torch.randn((2 * CHUNK, V), generator=torch.Generator().manual_seed(3)) * 4).bfloat16().float().to(DEV)
    tokens = torch.arange(2 * CHUNK, device=DEV) * 7
    lp = LogprobsPass(2 * CHUNK, V, 0, DEV)
    lp.dense.copy_(x)
    lp.run(0, torch.arange(2 * CHUNK, device=DEV), 2 * CHUNK, tokens, None, None)
    for i in (0, CHUNK + 3):
        lp.run(1, torch.tensor([i] * CHUNK, device=DEV), 1, tokens, None, None)
        assert torch.equal(lp.tok[1][0], lp.tok[0][i]) and torch.equal(lp.top[1][0], lp.top[0][i]) and \
            torch.equal(lp.ids[1][0], lp.ids[0][i])


# -- the runners on CUDA graphs ----------------------------------------------------------------------------------------


def check_against_transformers(kind: str, model, out: dict) -> int:
    """Each token's logprob against transformers' fp32 ones over its request's prompt and tokens (teacher forced), its
    grammar's mask applied, within a BF16 run's error; the top token's too. Returns the tokens checked."""
    from ..support.fake_grammar import FakeGrammar

    fake = FakeGrammar(512, (511,))
    n = 0
    for q in REQS:
        if q.get("logprobs", -1) < 0:
            continue
        toks, _, vals = out[q["rid"]]
        full = q["prompt"] + toks
        ref = (qwen_hf_logits(model, full) if kind == "qwen" else gemma_hf_logits(model[0], full)).float()
        m = fake.instance(fake.compile(q["grammar"])) if q.get("grammar") else None
        for i, (t, (v, ids, _)) in enumerate(zip(toks, vals)):
            x = ref[len(q["prompt"]) + i - 1]
            if m is not None:
                x = x.masked_fill(~torch.from_numpy(m.allowed().copy()), float("-inf"))
                if t != 511:
                    assert m.consume_tokens([t])
                else:
                    m = None
            lp = torch.log_softmax(x, -1)
            assert abs(v - float(lp[t])) < NEAR, (q["rid"], i, v, float(lp[t]))
            if ids:
                assert float(lp[ids[0]]) > float(lp.max()) - NEAR, (q["rid"], i)
            n += 1
    return n


@pytest.fixture(scope="module")
def qwen_inv(tmp_path_factory):
    """The tiny Qwen in BF16 with Qwen3.8's attention (head dim 256, 6 query heads over 1 KV head), which batch
    invariance's attention kernel takes (`kernels.attn_window.FullAttention`)."""
    return make_tiny_model(tmp_path_factory.mktemp("lpqinv"), seed=4, mtp=True, gpu=True, bf16=True, attn=QWEN38_ATTN)


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("heads", ["dense", "sparse", "invariant"])
def test_tokens_unchanged_with_logprobs_on_cuda_graphs(qwen_tiny, pair, request, kind, k, heads) -> None:  # noqa: F811
    kw = dict(always_split=True, sparse_head=heads != "dense", draft_universe=heads != "dense" and k > 0, hook=False,
              **(INVARIANT[kind] if heads == "invariant" else {}))
    if kind == "qwen":
        model, make = qwen_tiny, lambda m, pc, **rk: qwen_cuda_runner(m, pc, True, **rk)
        if heads == "invariant":  # the 64-token grid beside the decode rows; the kernel's pages of 16
            model = request.getfixturevalue("qwen_inv")
            kw.update(budget=128, wps=16)
    else:
        model, make = pair, lambda t, pc, **rk: gemma_cuda_runner(t, pc, True, **rk)
        kw.update(window=W, wps=16)
    plain, _ = run(kind, model, k, REQS, False, make=make, **kw)
    got, _ = run(kind, model, k, REQS, True, make=make, **kw)
    assert {r: v[:2] for r, v in got.items()} == {r: v[:2] for r, v in plain.items()}
    assert check_against_transformers(kind, model, got) >= 40

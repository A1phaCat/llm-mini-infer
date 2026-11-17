"""The grammar-aware heads on CUDA (`runner.cuda.heads`): GemmaRunner in BF16 with our kernels and CUDA graphs,
each split step's sampling graph in the variant its heads name (dense and whole `_s`, sparse and whole `_q`,
sparse and universe `_p`), through the scheduler in window mode on `gemma_gpu`'s tiny pair (the real head
dims). Plus the numerics the sparse head rests on, at Gemma 4 31B's head width: the gathered head's logits are the dense
head's columns, bit for bit at up to `models.gemma4.model.SMALL_M` rows (our small-M kernel computes each row alike),
within one BF16 ulp from cuBLAS above (it picks its kernel by N).

Two BF16 runs agree token for token up to a near tie of transformers' fp32 logits (`gemma_gpu.agree`).
CUDA only."""

from __future__ import annotations

import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar
from ..support.gemma_gpu import NEAR, gpu_run, reference
from ..support.gemma_gpu import pair as pair  # the fixture: test_gemma4_mtp_gpu's tiny pair in BF16
from ..support.gemma_heads_gpu import MANY, VOCAB, json_like
from ..support.mtp_common import EOS

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def agree(got: dict[int, list[int]], want: dict[int, list[int]], ref: dict[int, torch.Tensor],
          prompts: list[list[int]]) -> int:
    """`gemma_gpu.agree` under the grammars: where `got` parts from `want`, the reference's two best tokens
    among those the request's grammar allowed there are a near tie; how many tokens agree before the partings."""
    fake = FakeGrammar(VOCAB, (EOS,))
    same = 0
    for rid, w in want.items():
        g, start = got[rid], len(prompts[rid])
        i = next((j for j, (x, y) in enumerate(zip(g, w)) if x != y), None)
        if i is None:
            assert len(g) == len(w), rid
            same += len(w)
            continue
        m = fake.instance(fake.compile(json_like(rid)))
        assert m.consume_tokens(w[:i]), rid
        logits = ref[rid][start + i - 1].clone()
        logits[~torch.as_tensor(m.allowed())] = float("-inf")
        top = logits.topk(2).values
        assert (top[0] - top[1]).item() < NEAR, (rid, i, g[i], w[i], (top[0] - top[1]).item())
        same += i
    return same


def run_many(pair, k: int, graphs: bool, *, sparse: bool, universe: bool, n: int = 24):
    r = gpu_run(pair, k, graphs=graphs, max_num_seqs=32, budget=256, num_pages=2048,
                grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), sparse_head=sparse, draft_universe=universe)
    r.add(0, MANY[0], max_tokens=40, grammar=json_like(0), ignore_eos=False)  # the first computes the prefix
    r.run()
    for i, p in enumerate(MANY[1:n], 1):
        r.add(i, p, max_tokens=40, grammar=json_like(i), ignore_eos=False)
    r.run()
    assert r.sched.grammars.errors == 0
    r.r.shutdown()
    return r


@pytest.mark.parametrize("k", [3, 5])
def test_the_heads_in_graphs_give_the_dense_heads_tokens(pair, k) -> None:
    """24 constrained requests on one cached prefix: decode steps of 24 rows down to one and mixed steps, in graphs.
    The heads' graphs (`decode_p`, `break_p`) carry the split steps, and give the dense run's tokens up to a near tie;
    the variants exist for every bucket that has a split pair."""
    dense = run_many(pair, k, True, sparse=False, universe=False)
    heads = run_many(pair, k, True, sparse=True, universe=True)
    g = heads.r.stats["graphs"]
    assert g["sampling_variants"] == ["s", "q", "p"]
    for kind in ("decode", "break"):
        for b in g[kind]:
            if (f"{kind}_f", b) in heads.r._graphs:
                assert all((f"{kind}_{v}", b) in heads.r._graphs for v in "sqp"), (kind, b)
    c = heads.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained == 0 and c.dense_over_cap == 0 and c.universe_steps > 0, c
    assert "decode" in heads.modes and "break" in heads.modes
    prompts = MANY[:24]
    ref = reference(pair[0], dense.out, prompts)
    n = sum(len(o) for o in dense.out.values())
    assert agree(heads.out, dense.out, ref, prompts) > 0.8 * n


def test_the_heads_graphs_give_their_eager_tokens(pair) -> None:
    """With the heads on, the graphs give the eager run's tokens (the same heads, both ways) up to a near tie."""
    eager = run_many(pair, 3, False, sparse=True, universe=True, n=12)
    graphs = run_many(pair, 3, True, sparse=True, universe=True, n=12)
    assert "decode" not in eager.modes and graphs.modes.count("decode") > 5
    assert eager.r.head_counters().sparse > 0 and graphs.r.head_counters().sparse > 0
    prompts = MANY[:12]
    ref = reference(pair[0], eager.out, prompts)
    n = sum(len(o) for o in eager.out.values())
    assert agree(graphs.out, eager.out, ref, prompts) > 0.8 * n


def _key(x: torch.Tensor) -> torch.Tensor:
    """A monotone integer key of BF16 values: neighbours differ by one (an ulp), across zero too."""
    b = x.contiguous().view(torch.int16).int()
    return torch.where(b < 0, -32768 - b, b)


@pytest.mark.parametrize("m", [1, 2, 6, 24, 48, 96, 384])
def test_the_gathered_head_is_at_least_as_exact_as_the_dense_one(m) -> None:
    """At Gemma 4 31B's head width (5,376; 65,536 rows here, 300 of them a union padded to 1,024) the sparse head's
    logits (the rows gathered, the GEMM with fp32 partial sums: `fp32_reduction`, the BF16 softcap of 30) are the dense
    head's columns bit for bit at up to SMALL_M rows (our small-M kernel), and above it at least as close to the
    fp32-accumulated result as the dense head's (cuBLAS picks its kernel by N; with BF16 partial sums, torch's default,
    a narrow GEMM put 40% of them off, by up to their sign): 99.9% or more exact, and the argmax over a mask's tokens
    the dense head's but where the dense head itself is off the fp32 result."""
    from llm_mini_infer.models.gemma4.model import SMALL_M, linear, softcap
    from llm_mini_infer.runner.cuda.draft_head import fp32_reduction

    g = torch.Generator(device="cuda").manual_seed(m)
    w = (torch.randn((65_536, 5_376), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    h = torch.randn((m, 5_376), generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randperm(65_536, generator=g, device="cuda")[:300].sort().values
    idx = torch.cat([ids, torch.zeros(1_024 - 300, dtype=ids.dtype, device="cuda")])  # padding gathers row 0
    with fp32_reduction():
        sub = softcap(linear(h, w.index_select(0, idx)), 30.0)[:, :300]
    full = softcap(linear(h, w), 30.0)[:, ids]
    exact = softcap((h.float() @ w.index_select(0, ids).float().t()).to(torch.bfloat16), 30.0)
    if m <= SMALL_M:
        assert torch.equal(sub, full)
        return
    ds, df = (_key(sub) - _key(exact)).abs(), (_key(full) - _key(exact)).abs()
    s_same, f_same = float((ds == 0).float().mean()), float((df == 0).float().mean())
    assert s_same >= max(0.999, f_same - 1e-4) and int(ds.max()) <= max(16, int(df.max())), (s_same, f_same)
    for t in range(200):  # masks of 2-36 tokens: the argmax parts only where the dense head is off the fp32 result
        sel = torch.randperm(300, generator=g, device="cuda")[:2 + t % 35]
        a, b, e = (x[:, sel].float().argmax(1) for x in (sub, full, exact))
        assert bool(((a == b) | (b != e)).all()), t

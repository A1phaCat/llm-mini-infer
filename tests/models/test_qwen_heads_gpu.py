"""The grammar-aware heads in the Qwen runner on CUDA (`CudaRunner`; `runner.cuda.heads`): BF16 with our kernels and
CUDA graphs, each split step's sampling graph in the variant its heads name (dense and whole `_s`, sparse and whole
`_q`, sparse and universe `_p`), through the scheduler and the engine loop on `qwen_gpu`'s tiny Qwen with its MTP layer
(the dims our kernels take). Plus the numerics the sparse head rests on, at Qwen3.8's head (248,320 rows of 5,120):
the gathered head's logits are at least as close to the fp32-accumulated result as the dense head's, at every row
count (cuBLAS picks its kernel by N, so no M gives the dense head's bits); and the bound on our BF16 logits' error that
the near ties rest on (`qwen_gpu.NEAR`).

Two BF16 runs agree token for token up to a near tie of transformers' fp32 logits (`qwen_gpu.agree`). Gemma 4's twin is
test_gemma4_heads_gpu.py. CUDA only."""

from __future__ import annotations

import pytest
import torch

from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar
from ..support.gemma_heads_gpu import json_like
from ..support.mtp_common import EOS, seq
from ..support.qwen_gpu import MANY, NEAR, VOCAB, agree, gpu_run, reference
from ..support.qwen_gpu import tiny as tiny  # the fixture: the tiny Qwen with its MTP layer, kernel-sized, in BF16
from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cuda_runner, hf_logits, pools

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def run_many(model, k: int, graphs: bool, *, sparse: bool, universe: bool, n: int = 24):
    """`n` constrained requests on one cached prefix (`qwen_gpu.MANY`): the first computes it, the others hit its
    checkpoint (decode steps of up to 23 rows down to one, mixed steps). The runner stays up (its graphs) until the
    test shuts it down."""
    r = gpu_run(model, k, graphs=graphs, max_num_seqs=32, budget=256, num_pages=2048, caching=True, ckpt_slots=8,
                grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), sparse_head=sparse, draft_universe=universe)
    r.add(0, MANY[0], max_tokens=40, grammar=json_like(0), ignore_eos=False)  # the first computes the prefix
    r.run()
    for i, p in enumerate(MANY[1:n], 1):
        r.add(i, p, max_tokens=40, grammar=json_like(i), ignore_eos=False)
    r.run()
    assert r.sched.grammars.errors == 0
    assert all(r.cached[i] == 64 for i in range(1, n)), r.cached  # the others hit the checkpoint at 64
    return r


@pytest.mark.parametrize("k", [1, 3])
def test_the_heads_in_graphs_give_the_dense_heads_tokens(tiny, k) -> None:
    """24 constrained requests on one cached prefix: decode steps of 23 rows down to one and mixed steps, in graphs.
    The heads' graphs (`decode_p`, `break_p`) carry the split steps, and give the dense run's tokens up to a near tie;
    every bucket has its split pair and a sampling graph per variant."""
    dense = run_many(tiny, k, True, sparse=False, universe=False)
    heads = run_many(tiny, k, True, sparse=True, universe=True)
    g = heads.r.stats["graphs"]
    assert g["sampling_variants"] == ["s", "q", "p"] and dense.r.stats["graphs"]["sampling_variants"] == ["s"]
    for kind in ("decode", "break"):
        assert g[kind], kind
        for b in g[kind]:
            assert all((f"{kind}_{v}", b) in heads.r._graphs for v in "fsqp"), (kind, b)
    heads.r.shutdown()
    dense.r.shutdown()
    c = heads.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained == 0 and c.dense_over_cap == 0 and c.universe_steps > 0, c
    assert "decode" in heads.modes and "break" in heads.modes and "eager" not in heads.modes, heads.modes
    grammars = {i: json_like(i) for i in dense.out}
    ref = reference(tiny, dense.out, dense.prompts)
    n = sum(len(o) for o in dense.out.values())
    assert agree(heads.out, dense.out, ref, dense.prompts, grammars) > 0.8 * n


def test_the_heads_graphs_give_their_eager_tokens(tiny) -> None:
    """With the heads on, the graphs give the eager run's tokens (the same heads, both ways) up to a near tie."""
    eager = run_many(tiny, 3, False, sparse=True, universe=True, n=12)
    graphs = run_many(tiny, 3, True, sparse=True, universe=True, n=12)
    eager.r.shutdown()
    graphs.r.shutdown()
    assert "decode" not in eager.modes and graphs.modes.count("decode") > 5
    assert eager.r.head_counters().sparse > 0 and graphs.r.head_counters().sparse > 0
    grammars = {i: json_like(i) for i in eager.out}
    ref = reference(tiny, eager.out, eager.prompts)
    n = sum(len(o) for o in eager.out.values())
    assert agree(graphs.out, eager.out, ref, eager.prompts, grammars) > 0.8 * n


def test_eager_logits_match_transformers(tiny) -> None:
    """The bound the near ties rest on (`qwen_gpu.NEAR`): our kernels in BF16 (FlashInfer's attention and GDN prefill,
    the GDN decode kernel, the fused ops), eagerly over prefill chunks of 50, 63 and 37 tokens (rows starting mid-page
    and past the GDN prefill's chunks) and 10 decode steps, against transformers in fp32 on the same weights: within
    NEAR / 2, the argmax equal wherever transformers' two best are NEAR apart."""
    ids = seq(150, 1)
    pc = pools(max_num_seqs=2, budget=64, num_pages=200)
    drv = Driver(cuda_runner(tiny, pc, graphs=False), pc)
    steps = [Step(rows=[Row(1, ids[:50], 0, False)], admit=[(1, 1, GREEDY)]),
             Step(rows=[Row(1, ids[50:113], 50, False)]), Step(rows=[Row(1, ids[113:], 113, True)])]
    res = drv.run(steps + [Step(rows=[Row(1, None, 150 + i, True)]) for i in range(10)])
    drv.r.shutdown()
    gen = [int(r.tokens[0, 0]) for r in res if len(r.n_out) and r.n_out[0]]
    n = max(p for (_, p) in drv.logits) + 1
    ours = torch.stack([drv.logits[(1, p)] for p in range(n)])
    ref = hf_logits(tiny, ids + gen)[:n]
    err = (ours - ref).abs()
    assert n == 160 and err.max().item() < NEAR / 2, (n, err.max().item(), err.mean().item())
    top2 = ref.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > NEAR
    assert torch.equal(ours.argmax(-1)[clear], ref.argmax(-1)[clear])


def _key(x: torch.Tensor) -> torch.Tensor:
    """A monotone integer key of BF16 values: neighbours differ by one (an ulp), across zero too."""
    b = x.to(torch.bfloat16).contiguous().view(torch.int16).int()
    return torch.where(b < 0, -32768 - b, b)


@pytest.mark.parametrize("m", [1, 2, 6, 24, 48, 96, 384])
def test_the_gathered_head_is_at_least_as_exact_as_the_dense_one(m) -> None:
    """At Qwen3.8's head (248,320 rows of 5,120; 300 of them a union padded to 1,024 with −1) the sparse head's
    logits (`GrammarSampling.sparse_logits` as the Qwen runner builds it: the rows gathered, `F.linear` with fp32
    partial sums, `.float()`) are at least as close to the fp32-accumulated result as the dense head's
    (`Qwen35Model.logits`: `F.linear` over the whole head, torch's default BF16 partial sums) at every row count, with
    no kernel that gives the dense head's bits (Gemma 4's small-M one does: test_gemma4_heads_gpu): 99.9% or more
    exact, and the argmax over a mask's tokens the dense head's but where the dense head itself is off the fp32
    result."""
    import torch.nn.functional as F

    from llm_mini_infer.runner.cuda.heads import GrammarSampling

    g = torch.Generator(device="cuda").manual_seed(m)
    w = torch.randn((248_320, 5_120), generator=g, device="cuda", dtype=torch.bfloat16).mul_(0.02)
    h = torch.randn((m, 5_120), generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randperm(248_320, generator=g, device="cuda")[:300].sort().values
    gs = GrammarSampling(grammar=True, sparse=True, universe=False, rows=1_024, vocab=248_320, head=w)
    gs.allocate(m)
    gs.sp_src.fill_(-1)
    gs.sp_src[:300] = ids.to(torch.int32)  # padding gathers row 0
    sub = gs.sparse_logits(h)[:, :300]
    full = F.linear(h, w).float()[:, ids]
    exact = (h.float() @ w.index_select(0, ids).float().t()).to(torch.bfloat16)
    assert sub.dtype == full.dtype == torch.float32
    ds, df = (_key(sub) - _key(exact)).abs(), (_key(full) - _key(exact)).abs()
    s_same, f_same = float((ds == 0).float().mean()), float((df == 0).float().mean())
    assert s_same >= max(0.999, f_same - 1e-4) and int(ds.max()) <= max(16, int(df.max())), (s_same, f_same)
    for t in range(200):  # masks of 2-36 tokens: the argmax parts only where the dense head is off the fp32 result
        sel = torch.randperm(300, generator=g, device="cuda")[:2 + t % 35]
        a, b, e = (x[:, sel].float().argmax(1) for x in (sub, full, exact))
        assert bool(((a == b) | (b != e)).all()), t

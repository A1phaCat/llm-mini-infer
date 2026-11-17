"""The kernels' tests' references, guarded pools and cases (tests/kernels/test_kernels.py,
tests/kernels/test_gemma_kernels.py, tests/kernels/test_qwen_kernels_gpu.py, tests/scripts/numerics_kernels.py)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from llm_mini_infer.kernels import ops

DEV = "cuda"
BF = torch.bfloat16


def ref(fn, *args, **kw):
    """Run an op with the torch reference."""
    ops.enabled(False)
    try:
        return fn(*args, **kw)
    finally:
        ops.enabled(True)


def ulp_close(a: torch.Tensor, b: torch.Tensor, max_ulps: int = 1, min_equal: float = 0.99) -> None:
    """BF16 tensors equal up to `max_ulps` units in the last place, and mostly identical."""
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    d = (ia - ib).abs()
    assert int(d.max()) <= max_ulps, f"max {int(d.max())} ulps"
    assert float((d == 0).float().mean()) >= min_equal, float((d == 0).float().mean())


GUARD = 0xA5  # the byte a guarded pool's backing is filled with


class GuardedPool:
    """A KV layer's view of `shape` ([pages, ...], BF16) inside a backing filled with 0xA5 bytes: at a nonzero storage
    offset, each page followed by `pad` guard elements (so its stride is not its size) and the last by `pad` more; the
    pages hold random values. A kernel writing into `view(copy of backing)` must leave every element but its slots'
    as it was (`check`), so a write to a wrong slot, page or plane, or past the view, shows."""

    def __init__(self, shape: tuple[int, ...], seed: int, lead: int = 64, pad: int = 64) -> None:
        page = math.prod(shape[1:])
        self.shape, self.lead = shape, lead
        self.stride = (page + pad, *torch.empty(shape[1:], device="meta").stride())
        n = lead + shape[0] * (page + pad) + pad
        self.backing = torch.full((2 * n,), GUARD, dtype=torch.uint8, device=DEV).view(BF)
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.view(self.backing).copy_(torch.randn(shape, device=DEV, generator=g))

    def view(self, backing: torch.Tensor) -> torch.Tensor:
        return backing.as_strided(self.shape, self.stride, self.lead)

    def check(self, got: torch.Tensor, want: torch.Tensor, page: torch.Tensor, off: torch.Tensor) -> None:
        """`got` (the kernel's backing) and `want` (the reference's) after writing the slots (page, off) per token, K
        and V planes ([pages, 2, page_size, ...]) or the one ([pages, page_size, ...]): everything else unchanged bit
        for bit, the guards, the pages never written and the rest of the written ones; the slots within a BF16 ulp."""
        mask = torch.zeros(self.backing.shape, dtype=torch.bool, device=DEV)
        if len(self.shape) == 5:
            self.view(mask)[page, :, off] = True
        else:
            self.view(mask)[page, off] = True
        before = self.backing.view(torch.int16)[~mask]
        assert torch.equal(got.view(torch.int16)[~mask], before), "the kernel wrote outside its slots"
        assert torch.equal(want.view(torch.int16)[~mask], before), "the reference wrote outside its slots"
        ulp_close(got[mask], want[mask], min_equal=0.98)


def kv_slots(ps: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The KV slots (page, offset) of a step's tokens in a 10-page pool of `ps`-slot pages whose page 0 is the null
    page: three rows, each through its own page table over shuffled pages (with ps = 16, 35 tokens: a chunk at
    positions 5 .. 27, across two pages; one at 0 .. 8, on one; a decode token at 40, on the third page of its table;
    the positions scale with ps), so the slots are neither contiguous nor in page order and five pages are never
    written; then two padding tokens, which the CUDA graphs' padding rows write to the start of the null page (slot 0's
    page table is zero), from one input."""
    g = torch.Generator().manual_seed(seed)
    order = (torch.randperm(9, generator=g) + 1).tolist()
    rows = ((order[:2], range(ps // 3, 2 * ps - ps // 4)), (order[2:3], range(ps // 2 + 1)),
            (order[3:6], [2 * ps + ps // 2]))
    page, off = [], []
    for table, positions in rows:
        page += [table[p // ps] for p in positions]
        off += [p % ps for p in positions]
    return torch.tensor([*page, 0, 0], device=DEV), torch.tensor([*off, 0, 0], device=DEV)


def _rope_tables(d: int, rot_pairs: int, n: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (1e6 ** (torch.arange(0, 2 * rot_pairs, 2, device=DEV).float() / d))
    inv = torch.cat([inv, torch.zeros(d // 2 - rot_pairs, device=DEV)])
    freqs = torch.outer(torch.arange(float(n), device=DEV), inv)
    return freqs.cos().to(BF), freqs.sin().to(BF)


class _Pools:
    def __init__(self, fkv, ps):
        self.fkv, self.page_size = fkv, ps


class _Layer:
    def __init__(self, kw):
        self.index = 0
        self.k_norm = type("N", (), {"weight": kw})()


class _Rot:
    def __init__(self, cos, sin):
        self.cos, self.sin = {"full_attention": cos}, {"full_attention": sin}


def _full_case(rows: list[tuple[int, int]], h: int, hkv: int, ps: int, seed: int):
    """rows: (n_query, kv_len). V planes on shuffled pages, queries, the k_norm weight and the RoPE tables."""
    torch.manual_seed(seed)
    d = 512
    npg = [-(-kv // ps) for _, kv in rows]
    pages = torch.randperm(sum(npg), device=DEV) + 1
    fkv = (torch.randn(sum(npg) + 1, ps, hkv, d, device=DEV) * 0.5).to(BF)
    table = torch.zeros(len(rows) + 1, max(npg) + 2, dtype=torch.int32, device=DEV)
    o = 0
    for r, n in enumerate(npg):
        table[r + 1, :n] = pages[o:o + n].int()
        o += n
    t = sum(nq for nq, _ in rows)
    q = (torch.randn(t, h, d, device=DEV) * 0.06).to(BF)
    kw = (torch.randn(d, device=DEV) * 0.3 + 1).to(BF)
    cos, sin = _rope_tables(d, 64, 8192)
    return fkv, table, q, kw, cos, sin


def _window_setup(lens: list[int], hkv: int, d: int, ps: int, seed: int):
    """Window pages for slots 1.. (each `lens[i]` positions, shuffled pages, the table by absolute page) and the
    layer's pool [pages, 2, ps, Hkv, D]."""
    torch.manual_seed(seed)
    npg = [-(-n // ps) for n in lens]
    pool = (torch.randn(sum(npg) + 1, 2, ps, hkv, d, device=DEV) * 0.5).to(BF)
    table = torch.zeros(len(lens) + 1, max(npg) + 2, dtype=torch.int32, device=DEV)
    perm = torch.randperm(sum(npg), device=DEV) + 1
    o = 0
    for i, n in enumerate(npg):
        table[i + 1, :n] = perm[o:o + n].int()
        o += n
    return pool, table


def _window_ref(pool, table, slot: int, q: torch.Tensor, p0: int, window: int, ps: int,
                span: tuple[int, int] | None = None) -> torch.Tensor:
    """fp32 sliding-window attention of `q` [n, H, D] at positions p0.. of `slot`: keys [p − window + 1, p], or up to
    the span's end for a query in the image span [a, e)."""
    n, h, d = q.shape
    kv_len = p0 + n
    pages = table[slot, :-(-kv_len // ps)].long()
    k = pool[pages, 0].reshape(-1, pool.shape[3], d)[:kv_len].float()
    v = pool[pages, 1].reshape(-1, pool.shape[3], d)[:kv_len].float()
    g = h // k.shape[1]
    out = torch.empty(n, h, d, device=DEV)
    for i in range(n):
        p = p0 + i
        lo = max(0, p - window + 1)
        hi = span[1] if span and span[0] <= p < span[1] else p + 1
        kk, vv = k[lo:hi].repeat_interleave(g, dim=1), v[lo:hi].repeat_interleave(g, dim=1)
        s = torch.einsum("hd,khd->hk", q[i].float(), kk)
        out[i] = torch.einsum("hk,khd->hd", torch.softmax(s, dim=-1), vv)
    return out


def _window_run(a, pool, table, rows, q, *, bucket: int = 0, decode: int | None = None, true=None, span_end=None):
    """Our kernel over `rows` [(slot, first token, n_query, planned KV length)] (decode rows first: `decode` of them, by
    default all), with the true KV lengths `true` written on the device (default: the planned ones); `span_end`: each
    token's image-span end (0: none)."""
    from types import SimpleNamespace

    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731
    rr = np.asarray(rows, np.int32)
    kv = torch.as_tensor(rr[:, 3] if true is None else np.asarray(true), dtype=torch.int64, device=DEV)
    meta = SimpleNamespace(wpage_table=table, fix_len=True, n_rows=len(rr), kv_len_dev=kv, bidi_plan=None)
    if bucket:
        a.plan_graph(meta, bucket=bucket, rows=rr, stage=stage)
    else:
        a.plan(meta, parity=0, rows=rr, stage=stage, fixed_rows=len(rr) if decode is None else decode,
               span_end=None if span_end is None else np.asarray(span_end, np.int32))
    out = torch.full_like(q, float("nan"))
    a.run(meta, type("L", (), {"index": 0})(), q, out)
    torch.cuda.synchronize()
    return out


def _stage(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, np.int32)).pin_memory()


def _run(a, table, rows, q, *, decode: int, bucket: int = 0, true=None) -> torch.Tensor:
    """Our kernel over `rows` [(slot, first token, n_query, planned KV length)] (the first `decode` of them decode
    rows), their true KV lengths `true` written on the device (default: the planned ones)."""
    rr = np.asarray(rows, np.int32)
    kv = torch.as_tensor(rr[:, 3] if true is None else np.asarray(true), dtype=torch.int64, device=DEV)
    meta = SimpleNamespace(page_table=table, fix_len=True, n_rows=len(rr), kv_len_dev=kv)
    if bucket:
        a.plan_graph(meta, bucket=bucket, rows=rr, stage=_stage, real=decode)
    else:
        a.plan(meta, parity=0, rows=rr, stage=_stage, fixed_rows=decode)
    out = torch.full_like(q, float("nan"))
    a.run(meta, 0, q, out)
    torch.cuda.synchronize()
    return out


def _gdn_seq(n: int, seed: int) -> dict:
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(n, 16, 128, generator=g, device=DEV)
    k = torch.randn(n, 16, 128, generator=g, device=DEV)
    return dict(q=(q / q.norm(dim=-1, keepdim=True)).to(BF), k=(k / k.norm(dim=-1, keepdim=True)).to(BF),
                v=torch.randn(n, 48, 128, generator=g, device=DEV).to(BF),
                a=torch.exp(-torch.rand(n, 48, generator=g, device=DEV) * 0.3),
                b=torch.sigmoid(torch.randn(n, 48, generator=g, device=DEV)),
                s=torch.randn(48, 128, 128, generator=g, device=DEV) * 0.1)


def _gdn_step(rows: list[dict], ckpts: dict[int, list[int]] | None = None):
    """A step of `rows` packed for the GDN prefill, `ckpts` {row: offsets} its mid-chunk checkpoints: (the packed
    inputs, its meta, the checkpoints (row, offset) and their workspace indices)."""
    from llm_mini_infer.kernels.flashinfer import gdn_segments
    from llm_mini_infer.models.qwen3_5.backend import CkptMeta

    lens = np.asarray([r["q"].shape[0] for r in rows], np.int64)
    cu = np.concatenate([[0], np.cumsum(lens)])
    cat = {x: torch.cat([r[x] for r in rows]).contiguous() for x in ("q", "k", "v", "a", "b")}
    meta = SimpleNamespace(pf_cu=torch.as_tensor(cu, device=DEV), pf_cu_host=cu, ckpt=None)
    want = sorted((r, o) for r, offs in (ckpts or {}).items() for o in offs)
    src = np.zeros(0, np.int64)
    if want:
        counts = lens // 64
        starts = np.concatenate([[0], np.cumsum(counts)])
        src = np.asarray([starts[r] + o // 64 - 1 for r, o in want], np.int64)
        ws = torch.full((int(starts[-1]), 48, 128, 128), float("nan"), device=DEV)
        seg = gdn_segments(cu, np.asarray([r for r, _ in want]), np.asarray([o for _, o in want]), src,
                           lambda a: torch.as_tensor(np.asarray(a, np.int64), device=DEV))
        meta.ckpt = CkptMeta(every=64, cu_starts_host=starts, workspace=ws,
                             src=None, dst=None, conv=None, segments=seg)
    return cat, meta, want, src


def _gdn_call(rows: list[dict], ckpts: dict[int, list[int]] | None = None):
    """`flashinfer_gdn_prefill` over `rows` packed in one step, from a state pool of their own (row i in slot i),
    `ckpts` {row: offsets} its mid-chunk checkpoints; returns per row (output, final state, its checkpoints' states by
    offset)."""
    from llm_mini_infer.kernels.flashinfer import flashinfer_gdn_prefill

    cat, meta, want, src = _gdn_step(rows, ckpts)
    pool = torch.stack([r["s"] for r in rows]).contiguous()
    ids = torch.arange(len(rows), device=DEV)
    o = flashinfer_gdn_prefill(cat["q"], cat["k"], cat["v"], cat["a"], cat["b"], pool, ids, meta)
    torch.cuda.synchronize()
    cu, out = meta.pf_cu_host, []
    for i in range(len(rows)):
        cks = {o_: meta.ckpt.workspace[int(src[j])].clone() for j, (r, o_) in enumerate(want) if r == i}
        out.append((o[cu[i]:cu[i + 1]].clone(), pool[i].clone(), cks))
    return out


def _packed_reference(q, k, v, a, b, pool: torch.Tensor, slots: torch.Tensor, meta) -> torch.Tensor:
    """FlashInfer's per-sequence call on the rows' states gathered into a packed tensor, the final states scattered
    back to their slots (cut rows: each segment's call from the gathered states, as without `state_indices`)."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    kw = dict(scale=None, output_final_state=True, use_qk_l2norm_in_kernel=False, use_cp=False)
    state = pool[slots]
    if meta.ckpt is None:
        o, state = chunk_gated_delta_rule(q=q, k=k, v=v, g=a, beta=b, initial_state=state, cu_seqlens=meta.pf_cu,
                                          **kw)
    else:
        o = torch.empty_like(v)
        for tok, cu, rows, ck_seq, ck_src in meta.ckpt.segments:
            oj, fj = chunk_gated_delta_rule(q=q.index_select(0, tok), k=k.index_select(0, tok),
                                            v=v.index_select(0, tok), g=a.index_select(0, tok),
                                            beta=b.index_select(0, tok), initial_state=state.index_select(0, rows),
                                            cu_seqlens=cu, **kw)
            o.index_copy_(0, tok, oj)
            state.index_copy_(0, rows, fj)
            if ck_seq is not None:
                meta.ckpt.workspace.index_copy_(0, ck_src, fj.index_select(0, ck_seq))
    pool[slots] = state
    return o

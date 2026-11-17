"""The Qwen3.5/3.8 hybrid model's per-step batch metadata, the forward context the model calls, and its attention and
GDN backends (docs/architecture.md, "Models"; Gemma 4's are `models.gemma4.backend`).

**Token order.** Rows [0, n_dec) have one query token each and take the GDN decode kernel; the prefill rows follow as
one contiguous slice (`StepMeta.n_dec`, `pf_cu`). Under MTP the first `MtpMeta.n` rows are verify windows of 1 + k
tokens. Which rows decode is the runner's choice (`CudaRunner._layout`), which maps results back to plan order.

**Pools** (`Pools`), allocated by the runner from `PoolConfig`:
- KV: `[n_attn_layers, num_pages, 2, page_size, num_kv_heads, head_dim]` in the model dtype (FlashInfer's NHD layout
  per layer; under MTP the MTP layer's last). Page 0 is the null page: CUDA-graph padding rows write their K and V
  there.
- conv state: `[num_state_slots, n_gdn_layers, conv_dim, conv_kernel - 1]` in the model dtype: the last
  `conv_kernel - 1` conv inputs of each GDN layer, oldest first.
- recurrent state: `[num_state_slots, n_gdn_layers, HV, V, K]` fp32 (FlashInfer's `[N, H_v, V, K]` per layer).
- hidden (MTP): `[num_state_slots, hidden]`: the drafter's input hidden vector at the slot's position − 1.
Each state pool is slot-major, so a checkpoint or STATE_COPY of a slot is one copy per pool (the runner does those,
and zeroes the live slot of a STATE_ZERO row before the forward, so no kernel masks zero rows). Mid-chunk checkpoints
are taken inside the forward (`CkptMeta`).

**Metadata** (`StepMeta`): token-level tensors are computed on the device; row-level fields are what the eager breaks
read (`ForwardContext`).

**Backends:**
- attention: `TorchAttention` (the reference: SDPA over the gathered pages, one row at a time; any device),
  `kernels.flashinfer.FlashInferAttention` (FlashInfer's paged prefill wrapper, also in CUDA-graph mode) and, under
  batch invariance, `kernels.attn_window.FullAttention`. The K and V writes happen before, in
  `ForwardContext.qk_rope_store` (fused with the QK norm and RoPE).
- GDN: `GDN` runs the causal conv, the decode rows through the fused decode kernel (`ops.gdn_decode`) or under MTP the
  verify windows through `ops.gdn_decode_mtp`, the prefill rows through `torch_gdn_prefill` (token-by-token fp32
  recurrence) or FlashInfer's chunked kernel (`kernels.flashinfer`), then the gated output norm.

The CUDA backends read no device memory from the host: every size comes from host-side `StepMeta` fields (the torch
reference reads the true KV lengths under MTP).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from ...kernels import fp8, ops
from ...kernels.fp8 import QuantAct
from ..common import weak_view

if TYPE_CHECKING:
    from .model import Attention, GatedDeltaNet, Rotary


@dataclass
class Pools:
    kv: torch.Tensor
    conv: torch.Tensor
    rec: torch.Tensor
    page_size: int
    hid: torch.Tensor | None = None  # MTP only (module docstring)


@dataclass
class MtpMeta:
    """MTP verify rows [0, n) of a step (GDN replay; docs/architecture.md, "Speculative decoding"): their request slots
    and the window records (`kernels.ops.ConvMtp`, `GdnMtp`), as whole tensors over the GDN layers."""

    n: int
    width: int  # 1 + k tokens per row
    req: torch.Tensor  # int64 [n]
    req_i32: torch.Tensor  # int32 [n]
    fold_n: torch.Tensor  # int32 [req slots]: records each slot's next fold applies
    rec_conv: torch.Tensor  # [req slots, layers, conv_dim, width]
    rec_k: torch.Tensor  # [req slots, layers, HV, width, K]
    rec_v: torch.Tensor  # [req slots, layers, HV, width, V]
    rec_ab: torch.Tensor  # [req slots, layers, HV, width, 2]
    fold_n64: torch.Tensor | None = None  # int64 copy of fold_n for the conv


@dataclass
class CkptMeta:
    """Mid-chunk checkpoints of one step (prefill rows only): a workspace with one fp32 state per 64 tokens of each
    prefill row, counted from the row's start (`cu_starts_host`). The reference prefill fills every slot, FlashInfer's
    path only the requested ones (`kernels.flashinfer.gdn_segments`); `src`/`dst` copy them to their checkpoint slots,
    and the conv kernel writes the conv state at each one (docs/architecture.md, "Checkpoints")."""

    every: int  # the workspace's spacing in tokens: 64, the GDN chunk
    cu_starts_host: np.ndarray  # [n_prefill_rows + 1]: workspace offset of each prefill row's every-64 slots
    workspace: torch.Tensor  # [>= the step's every-64 slots, HV, V, K] fp32, reused by every layer
    src: torch.Tensor  # int64 [n]: workspace index of each requested checkpoint
    dst: torch.Tensor  # int64 [n]: its checkpoint state slot
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # for the conv: (ptr [R + 1], offset, dst slot) by row
    # FlashInfer's prefill: the calls that take these checkpoints by cutting their rows
    # (`kernels.flashinfer.gdn_segments`); None on the torch reference
    segments: Any = None


@dataclass
class StepMeta:
    """One step's batch as the backends need it. `*_host` fields are numpy (host), the others device tensors."""

    num_tokens: int
    positions: torch.Tensor  # [T] int64
    kv_page: torch.Tensor  # [T] int64: the page holding each token's K and V
    kv_off: torch.Tensor  # [T] int64: offset in that page
    n_rows: int
    n_dec: int  # rows (and tokens) [0, n_dec) have one query token each
    row_state: torch.Tensor  # [n_rows] int64: live state slot of each row
    row_cu: torch.Tensor  # [n_rows + 1] int64: token offset of each row
    pf_cu: torch.Tensor | None  # [prefill rows + 1] int64 cu_seqlens of the rows after the decode or verify rows
    pf_cu_host: np.ndarray
    # Host copies for the torch reference attention.
    row_start_host: np.ndarray  # [n_rows] first token of each row
    row_nq_host: np.ndarray  # [n_rows]
    row_kv_len_host: np.ndarray  # [n_rows] KV length after this step (seq_len + n_query)
    row_slot_host: np.ndarray  # [n_rows]
    page_table: torch.Tensor | None = None  # [max_num_seqs + 1, max_pages] int32 (the runner's device table)
    row_state_i32: torch.Tensor | None = None  # int32 copy of row_state (the verify kernel's slot indices)
    ckpt: CkptMeta | None = None  # mid-chunk checkpoints, only on steps that take some
    attn_plan: Any = None  # set by the attention backend's plan()
    # The decode kernel's int32 inputs for rows [0, n_dec): (state slots [n_dec, 1], cu_seqlens 0..n_dec, accepted 1)
    decode_i32: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    mtp: MtpMeta | None = None  # MTP verify rows (tokens [0, mtp.n × mtp.width)); then n_dec is 0
    # MTP: the rows' true KV lengths on the device (int64 [n_rows]) when the host knows only bounds. The torch reference
    # reads them; with `fix_len` the CUDA backends rewrite their plans' lengths from them before each run (the drafter's
    # chain in a breakable graph); elsewhere the runner fixes each plan once.
    kv_len_dev: torch.Tensor | None = None
    fix_len: bool = False
    chain: Any = None  # MTP: the drafter's chain's metadata (a StepMeta), planned with this step
    # RoPE positions when they differ from `positions` (M-RoPE with images): [3, T] or [T]; None: positions
    rope: torch.Tensor | None = None


class ForwardContext:
    """What the model's layers call during one forward pass.

    With `capture` set (a `runner.cuda.graphs.BreakableCapture`), attention and the GDN mixer are eager breaks
    (`runner.cuda.graphs`, module docstring). Each reads `self.meta` when it runs, which the runner points at the
    current step's row metadata before each replay. A replay may pad the step to the graph's token count: the breaks see
    only the first `meta.num_tokens` tokens, and the padding rows of their outputs keep whatever they held."""

    __slots__ = ("_attn", "_gdn", "capture", "meta", "pools", "rotary")

    def __init__(self, meta: StepMeta, attn: Any, gdn: GDN, rotary: Rotary, pools: Pools, capture: Any = None):
        self.meta = meta
        self._attn = attn
        self._gdn = gdn
        self.rotary = rotary
        self.pools = pools
        self.capture = capture

    def qk_rope_store(self, layer: Attention, qkv: torch.Tensor) -> torch.Tensor:
        """q [T, H, D] after the QK norm and RoPE; K and V go to this layer's KV pages."""
        m, r = self.meta, self.rotary
        rope = m.positions if m.rope is None else m.rope
        return ops.qk_norm_rope(qkv, heads=layer.h, kv_heads=layer.hkv, head_dim=layer.d, gated=layer.gated,
                                q_w=layer.q_norm.weight, k_w=layer.k_norm.weight, eps=layer.q_norm.eps,
                                positions=rope, cos_cache=r.cos, sin_cache=r.sin, rotary_dim=r.rotary_dim,
                                kv_layer=self.pools.kv[layer.index], kv_page=m.kv_page, kv_off=m.kv_off,
                                mrope=r.mrope if rope.dim() == 2 else None)

    def attention(self, index: int, q: torch.Tensor) -> torch.Tensor:
        """q [T, H, D] -> attention output [T, H * D]."""
        out = torch.empty_like(q)
        if self.capture is None:
            self._attn.run(self.meta, index, q, out)
        else:
            qw, ow = weak_view(q), weak_view(out)

            def run() -> None:  # the step's real tokens; padding tokens are not in the plan
                n = self.meta.num_tokens
                if n:  # (the drafter's chain in a step without sample rows has none, and no plan)
                    self._attn.run(self.meta, index, qw[:n], ow[:n])

            self.capture.eager(run)
        return out.view(q.shape[0], -1)

    def gdn(self, layer: GatedDeltaNet, mixed_qkv: torch.Tensor, z: torch.Tensor, a: torch.Tensor,
            b: torch.Tensor, quant: bool = False) -> torch.Tensor | QuantAct:
        """The GDN mixer: conv, recurrence and the gated output norm -> [T, HV, V]; with `quant`, its quantization for
        a block-FP8 out_proj instead (one group per value head)."""
        t = mixed_qkv.shape[0]
        out = qa = None
        if quant:
            qa = fp8.empty_act(t, layer.value_dim, mixed_qkv.device)
        else:
            out = torch.empty((t, layer.hv, layer.dv), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        if self.capture is None:
            self._gdn.forward(self.meta, layer, mixed_qkv, z, a, b, out, qa)
        else:
            # A strong reference in the closure, which the graph keeps, would pin the output in the graph pool (one per
            # layer and bucket).
            args = [weak_view(x) for x in (mixed_qkv, z, a, b)]
            res = [weak_view(x) for x in ((out,) if qa is None else (qa.q, qa.s))]

            def run() -> None:
                n = self.meta.num_tokens
                if len(res) == 1:
                    self._gdn.forward(self.meta, layer, *(x[:n] for x in args), res[0][:n])
                else:
                    self._gdn.forward(self.meta, layer, *(x[:n] for x in args), None,
                                      QuantAct(res[0][:n], res[1][:, :n]))

            self.capture.eager(run)
        return out if qa is None else qa


# -- attention -----------------------------------------------------------------------------------------------------


class TorchAttention:
    """Reference paged attention: per row, gather its pages and run causal SDPA (queries are the last n_query
    positions of the row's KV)."""

    name = "torch"

    def __init__(self, pools: Pools, head_dim: int):
        self.pools = pools
        self.scale = head_dim ** -0.5

    def plan(self, meta: StepMeta, **_: Any) -> None:
        meta.attn_plan = None

    def run(self, meta: StepMeta, index: int, q: torch.Tensor, out: torch.Tensor) -> None:
        layer = self.pools.kv[index]
        kc, vc = layer[:, 0], layer[:, 1]  # [num_pages, page_size, Hkv, D]
        ps = self.pools.page_size
        hkv, d = kc.shape[2], kc.shape[3]
        lengths = meta.kv_len_dev.tolist() if meta.kv_len_dev is not None else meta.row_kv_len_host
        for r in range(meta.n_rows):
            s, n, length = int(meta.row_start_host[r]), int(meta.row_nq_host[r]), int(lengths[r])
            pages = meta.page_table[int(meta.row_slot_host[r]), :-(-length // ps)].long()
            keys = kc[pages].reshape(-1, hkv, d)[:length]
            vals = vc[pages].reshape(-1, hkv, d)[:length]
            qpos = torch.arange(length - n, length, device=q.device)
            mask = torch.arange(length, device=q.device)[None, :] <= qpos[:, None]
            o = F.scaled_dot_product_attention(q[s:s + n].transpose(0, 1)[None], keys.transpose(0, 1)[None],
                                               vals.transpose(0, 1)[None], attn_mask=mask, scale=self.scale,
                                               enable_gqa=True)
            out[s:s + n] = o[0].transpose(0, 1)


# -- GDN -----------------------------------------------------------------------------------------------------------


PrefillKernel = Callable[..., torch.Tensor]


def torch_gdn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                      pool: torch.Tensor, slots: torch.Tensor, meta: StepMeta) -> torch.Tensor:
    """Reference chunked-prefill replacement: the recurrence token by token, in fp32.

    q, k [Tp, HK, K] (L2-normalised, model dtype), v [Tp, HV, V], the decay alpha = exp(g) and beta [Tp, HV] fp32;
    pool [S, HV, V, K] fp32, the layer's recurrent states, and slots [n] (int64), each prefill row's. Each row starts
    from its slot's state and writes its final state there. Returns o [Tp, HV, V] in the model dtype."""
    rep = v.shape[1] // q.shape[1]
    scale = q.shape[-1] ** -0.5
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    final = pool[slots]
    cu = meta.pf_cu_host
    ck = meta.ckpt
    for r in range(len(cu) - 1):
        h = final[r:r + 1]  # a view: updated in place
        for t in range(int(cu[r]), int(cu[r + 1])):
            qt, kt = q[t:t + 1].float(), k[t:t + 1].float()
            if rep > 1:
                qt, kt = qt.repeat_interleave(rep, 1), kt.repeat_interleave(rep, 1)
            out[t] = ops.recurrent_step(h, qt * scale, kt, v[t:t + 1].float(), alpha[t:t + 1], beta[t:t + 1])[0]
            done = t - int(cu[r]) + 1
            if ck is not None and done % ck.every == 0:
                ck.workspace[int(ck.cu_starts_host[r]) + done // ck.every - 1] = h[0]
    pool[slots] = final
    return out.to(v.dtype)


class GDN:
    """The Gated DeltaNet mixer of a whole step, one layer per `forward` (module docstring, "Backends"). Reads and
    writes the conv and recurrent state of each row's live state slot (zeroed beforehand for STATE_ZERO rows)."""

    def __init__(self, pools: Pools, prefill: PrefillKernel):
        self.pools = pools
        self.prefill = prefill

    def forward(self, meta: StepMeta, layer: GatedDeltaNet, mixed_qkv: torch.Tensor, z: torch.Tensor,
                a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None, quant: QuantAct | None = None) -> None:
        """Writes the normed output into `out` [T, HV, V], or its FP8 quantization into `quant` (q [T, HV * V],
        s [HV, >= T]): the decode kernel quantizes in its epilogue, prefill rows through `ops.gated_rmsnorm_quant`."""
        t, nd, i = meta.num_tokens, meta.n_dec, layer.index
        p = self.pools
        ck = meta.ckpt
        mt = meta.mtp
        cm = ops.ConvMtp(mt.n, mt.width, mt.req, mt.fold_n64, mt.rec_conv[:, i]) if mt is not None and mt.n else None
        x = ops.causal_conv(mixed_qkv, layer.conv_weight, p.conv[:, i], meta.row_cu, meta.row_state,
                            ck.conv if ck is not None else None, nd, cm)
        if cm is not None:  # MTP verify windows: replay the previous commit, then the window without writing it
            nd = mt.n * mt.width
            ops.gdn_decode_mtp(x[:nd], a[:nd], b[:nd], a_log=layer.A_log, dt_bias=layer.dt_bias, state=p.rec[:, i],
                               slots=meta.row_state_i32[:mt.n] if meta.row_state_i32 is not None else
                               meta.row_state[:mt.n], z=z[:nd], norm_w=layer.norm.weight, eps=layer.norm.eps,
                               hk=layer.hk, dk=layer.dk,
                               mtp=ops.GdnMtp(mt.width, mt.req_i32, mt.fold_n, mt.rec_k[:, i], mt.rec_v[:, i],
                                              mt.rec_ab[:, i]),
                               out=out[:nd] if out is not None else None,
                               quant=QuantAct(quant.q[:nd], quant.s[:, :nd]) if quant is not None else None)
        elif nd:
            o = ops.gdn_decode(x[:nd], a[:nd], b[:nd], a_log=layer.A_log, dt_bias=layer.dt_bias, state=p.rec[:, i],
                               slots=meta.row_state[:nd], z=z[:nd], norm_w=layer.norm.weight, eps=layer.norm.eps,
                               hk=layer.hk, dk=layer.dk, out=out[:nd] if out is not None else None,
                               quant=QuantAct(quant.q[:nd], quant.s[:, :nd]) if quant is not None else None,
                               **(dict(zip(("slots_i32", "cu_i32", "accepted_i32"), meta.decode_i32))
                                  if meta.decode_i32 is not None else {}))
            if out is not None and o.data_ptr() != out.data_ptr():
                out[:nd] = o
        if nd < t:
            o = self.prefill_rows(meta, layer, x[nd:], a[nd:], b[nd:])
            if quant is not None:
                ops.gated_rmsnorm_quant(o, z[nd:], layer.norm.weight, layer.norm.eps,
                                        QuantAct(quant.q[nd:], quant.s[:, nd:]))
            else:
                ops.gated_rmsnorm(o, z[nd:], layer.norm.weight, layer.norm.eps, out=out[nd:])

    def prefill_rows(self, meta: StepMeta, layer: GatedDeltaNet, x: torch.Tensor, a: torch.Tensor,
                     b: torch.Tensor) -> torch.Tensor:
        """The prefill rows: `ops.gdn_prep`, then the prefill kernel from each row's state slot, which takes the final
        states; the checkpoints go to their slots after it. Returns the recurrence output [Tp, HV, V] before the
        norm."""
        nd, i = meta.n_dec + (meta.mtp.n if meta.mtp is not None else 0), layer.index  # the first prefill row
        qn, kn, v, alpha, beta = ops.gdn_prep(x, a, b, a_log=layer.A_log, dt_bias=layer.dt_bias, hk=layer.hk,
                                              hv=layer.hv, dk=layer.dk)
        o = self.prefill(qn, kn, v, alpha, beta, self.pools.rec[:, i], meta.row_state[nd:], meta)
        if meta.ckpt is not None:
            self.pools.rec[meta.ckpt.dst, i] = meta.ckpt.workspace[meta.ckpt.src]
        return o

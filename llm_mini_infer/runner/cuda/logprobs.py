"""Logprobs (docs/architecture.md, "Sampling"; `runner.base.LogprobRows`): vLLM's default `raw_logprobs` of the
samples a step's LOGPROBS rows draw, from the logits the sampler read, computed after the step's sampling for those
samples alone; one implementation for both CUDA runners and their CPU path.

**The logits.** With logprobs on, the target head's logits land in a static fp32 buffer (`LogprobsPass.dense`; the
sparse head's [samples, rows] in its first bytes, `sparse`) instead of a temporary: the copy that widens them to fp32
writes there (`models.*.logits`, `heads.GrammarSampling.sparse_logits`), so every step runs the same kernels as without.
A drafter's logits never land there.

**The values**, over a sample's logits x with its grammar mask applied (a refused token at −inf):
`lp(t) = (x_t − m) − log Σ exp(x − m)` in fp32, m = max x. The top `MAX_LOGPROBS`: the highest x, ties to the
lowest id (`torch.topk` leaves ties unordered), selected through an int64 key per column (x's bits in sortable form,
then the id reversed), unique, so the CPU and CUDA pick the same tokens. A sample whose mask allows fewer tokens
gets the lowest refused ids, ascending, at −inf. On the sparse head x covers the union's columns (every token the
sample's mask allows: the rest of the vocabulary adds exp(−inf) = 0), plus ids 0..63 as refused candidates: a mask
that allows fewer than MAX_LOGPROBS tokens refuses at least 44 of them, so the fillers are the dense head's.

**The pass** runs eagerly on the compute stream after the step's sampling (graph or eager), before the outputs' copy,
over its LOGPROBS samples in chunks of `CHUNK`: reductions of one shape, so a sample's values do not depend on the
samples beside it. Its indices are staged with the step's inputs and its sizes are host ints: nothing syncs. Its results
reach the host on the copy stream with the tokens (`copy_out`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...sched.plan import LOGPROBS
from ..base import MAX_LOGPROBS, LogprobRows, PlanError

CHUNK = 8  # samples per pass: at Gemma 4's vocabulary about 80 MB of temporaries
EXTRA = 64  # the sparse head's refused candidates: ids 0..63
_REV = 0x7FFFFFFF  # the key's low half: _REV − id (the lowest id the largest)
_MIN64 = torch.iinfo(torch.int64).min


def sortable(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> int32 in the same order (−inf lowest; −0.0 just below +0.0, which no logit takes)."""
    b = x.view(torch.int32)
    return b ^ ((b >> 31) & _REV)


def top_keys(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """int64 keys [n, C] for the columns of `x` [n, C] fp32 whose token ids are `ids` [C] int64 (>= 0): by x
    descending, then id ascending; distinct for distinct ids."""
    return (sortable(x).long() << 32) | (_REV - ids)[None, :]


def unpack(words: torch.Tensor, shifts: torch.Tensor, vocab: int) -> torch.Tensor:
    """int32 mask words [n, W] -> bool [n, vocab] (token t: bit t % 32 of word t // 32; little-endian bytes)."""
    b = words.view(torch.uint8)
    return ((b[:, :, None] >> shifts) & 1).view(len(words), -1)[:, :vocab].bool()


class LogprobsPass:
    """A runner's logprobs (module docstring): the logits buffer, and per staging parity the outputs and their
    page-locked copies (a step's copy may still run while the next step computes; the one after that waits for it).
    `smax`: the most samples a step has; `rows`: the sparse head's columns (0: none)."""

    def __init__(self, smax: int, vocab: int, rows: int, device: torch.device):
        f32, i32, cuda = torch.float32, torch.int32, device.type == "cuda"
        self.vocab, self.cuda = vocab, cuda
        self.dense = torch.zeros((smax, vocab), dtype=f32, device=device)
        self.sparse = self.dense.view(-1)[:smax * rows].view(smax, rows) if rows else None
        cap = smax + CHUNK  # the last chunk's padding samples write past the step's
        self.tok = [torch.zeros(cap, dtype=f32, device=device) for _ in range(2)]
        self.ids = [torch.zeros((cap, MAX_LOGPROBS), dtype=i32, device=device) for _ in range(2)]
        self.top = [torch.zeros((cap, MAX_LOGPROBS), dtype=f32, device=device) for _ in range(2)]
        self.h_tok = [torch.zeros(smax, dtype=f32, pin_memory=cuda) for _ in range(2)]
        self.h_ids = [torch.zeros((smax, MAX_LOGPROBS), dtype=i32, pin_memory=cuda) for _ in range(2)]
        self.h_top = [torch.zeros((smax, MAX_LOGPROBS), dtype=f32, pin_memory=cuda) for _ in range(2)]
        self.shifts = torch.arange(8, dtype=torch.uint8, device=device)
        self.extra = torch.arange(EXTRA, dtype=torch.int64, device=device)
        self.vocab_ids = torch.arange(vocab, dtype=torch.int64, device=device)

    def layout(self, plan: Any, sample: np.ndarray, admitted_lp: np.ndarray, order: np.ndarray, c_row: np.ndarray,
               r_s0: np.ndarray, n_stok: int, width: int, st: Any) -> tuple[np.ndarray, int, int] | None:
        """A step's LOGPROBS samples (None: none), checked against the plan (on SAMPLE rows of slots admitted with
        ADMIT_LOGPROBS: `admitted_lp` per plan row): (their rows in plan order, the staged offset of their samples'
        indices in whole chunks, the samples' count). Each row's `width` samples: its first (`r_s0`, in the step's row
        `order`; `c_row`: the rows that sample) and the ones after it, within the step's `n_stok`."""
        lpf = (plan.sched_flags & LOGPROBS) != 0
        if not lpf.any():
            return None
        if (lpf & ~sample).any() or not admitted_lp[lpf].all():
            raise PlanError(f"step {plan.step_id}: LOGPROBS on a row without SAMPLE, or of a slot admitted without "
                            "ADMIT_LOGPROBS")
        sel = np.flatnonzero(lpf[order[c_row]])
        idx = np.minimum(r_s0[c_row[sel]][:, None] + np.arange(width)[None, :], n_stok - 1).reshape(-1)
        n = len(idx)
        return order[c_row[sel]].astype(np.int32), st.put(np.concatenate([idx, np.zeros(-n % CHUNK, np.int64)])), n

    def step(self, lay_lp: tuple[np.ndarray, int, int], parity: int, st: Any, tokens: torch.Tensor,
             ids: torch.Tensor | None, mask: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        """`run` for a step's samples (`layout`'s), their indices from its staging `st` (on the device by now)."""
        _, off, n = lay_lp
        self.run(parity, st.i(off, -(-n // CHUNK) * CHUNK), n, tokens, ids, mask)

    @torch.inference_mode()
    def run(self, parity: int, idx: torch.Tensor, n: int, tokens: torch.Tensor, ids: torch.Tensor | None,
            mask: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        """The values of `n` samples into `parity`'s outputs. `idx` int64 [n in whole chunks] (`padded`): each
        sample's row in the logits buffer and in `tokens` (the sampled tokens). `ids`: the sparse head's columns' token
        ids (ascending, then −1; None: the dense head). `mask`: (the step's mask buffer, each sample's row in it or −1),
        None for a step without masks."""
        sparse = ids is not None
        logits = self.sparse if sparse else self.dense
        o_tok, o_ids, o_top = self.tok[parity], self.ids[parity], self.top[parity]
        neg = float("-inf")
        if sparse:
            cid = ids.long()
            real = cid.clamp(min=0)
            in_union = (cid[None, :] == self.extra[:, None]).any(-1)  # an id 0..63 the union holds is its column
            cand = torch.cat([real, self.extra])
            excluded = torch.cat([cid < 0, in_union])
            word, bit = (real >> 5)[None, :], (real & 31)[None, :]
        for c0 in range(0, n, CHUNK):
            r = idx[c0:c0 + CHUNK]
            x = logits.index_select(0, r)
            t = tokens.index_select(0, r)
            if mask is not None:
                mrow = mask[1].index_select(0, r)
                words = mask[0].index_select(0, mrow.clamp(min=0))
                if sparse:
                    allowed = ((words.gather(1, word.expand(len(r), -1)) >> bit) & 1).bool()
                else:
                    allowed = unpack(words, self.shifts, self.vocab)
                x = x.masked_fill(~(allowed | (mrow < 0)[:, None]), neg)
            if sparse:
                x = x.masked_fill((cid < 0)[None, :], neg)
            m = x.amax(-1, keepdim=True)
            lse = torch.log(torch.exp(x - m).sum(-1, keepdim=True))
            col = (cid[None, :] == t[:, None]).int().argmax(-1) if sparse else t
            o_tok[c0:c0 + CHUNK] = ((x.gather(1, col[:, None]) - m) - lse)[:, 0]
            if sparse:
                xc = torch.cat([x, x.new_full((len(r), EXTRA), neg)], 1)
                key = top_keys(xc, cand).masked_fill_(excluded[None, :], _MIN64)
                pos = torch.topk(key, MAX_LOGPROBS, dim=-1).indices
                o_ids[c0:c0 + CHUNK] = cand[pos].int()
            else:
                xc = x
                pos = torch.topk(top_keys(x, self.vocab_ids), MAX_LOGPROBS, dim=-1).indices
                o_ids[c0:c0 + CHUNK] = pos.int()
            o_top[c0:c0 + CHUNK] = (xc.gather(1, pos) - m) - lse

    def copy_out(self, parity: int, n: int) -> None:
        """The values of the step's `n` samples to `parity`'s page-locked buffers (on CUDA, inside the caller's copy
        stream context)."""
        nb = self.cuda
        self.h_tok[parity][:n].copy_(self.tok[parity][:n], non_blocking=nb)
        self.h_ids[parity][:n].copy_(self.ids[parity][:n], non_blocking=nb)
        self.h_top[parity][:n].copy_(self.top[parity][:n], non_blocking=nb)

    def result(self, parity: int, rows: np.ndarray, width: int) -> LogprobRows:
        """`LogprobRows` of `rows` (plan rows, `width` samples each) from `parity`'s buffers, once their copy ran."""
        m, n = len(rows), len(rows) * width
        return LogprobRows(rows=rows, token=self.h_tok[parity][:n].numpy().reshape(m, width).copy(),
                           top_ids=self.h_ids[parity][:n].numpy().reshape(m, width, MAX_LOGPROBS).copy(),
                           top=self.h_top[parity][:n].numpy().reshape(m, width, MAX_LOGPROBS).copy())


def reference(x: torch.Tensor, token: int, allowed: torch.Tensor | None = None) -> tuple[float, list[int], list[float]]:
    """One sample's values by the definition (module docstring; tests): `x` [V] fp32 logits, `allowed` bool [V] or
    None. A full stable sort instead of the keys."""
    if allowed is not None:
        x = x.masked_fill(~allowed, float("-inf"))
    m = x.max()
    lse = torch.log(torch.exp(x - m).sum())
    lp = (x - m) - lse
    order = torch.sort(x, descending=True, stable=True).indices[:MAX_LOGPROBS]
    return float(lp[token]), order.tolist(), lp[order].tolist()

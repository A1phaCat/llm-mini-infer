"""The grammar-aware heads (docs/architecture.md, "Grammar-aware heads"): on the host, which heads a split step samples
with, from its grammar masks, and the draft universe (`GrammarHeads`); and `GrammarSampling`, a CUDA runner's grammar
masks and heads from the staging buffers to the sparse head's logits, one implementation for both CUDA runners (its
head's GEMM and finishing are parameters: Gemma 4's runner passes its own, the defaults are Qwen's).

**The sparse head.** A split step (its sampling runs in `launch_sample`, after the engine filled the masks: a step with
MASKED rows, and every step under `always_split`) whose every sample token has a mask is *eligible*: the union of its
mask rows holds every token any of its rows may sample, so the target's head need score only those rows, the sampler
testing each row's mask through the token ids (`kernels.ops.sample`, `padded`). Within `rows` tokens the step is
sparse, and the union's ascending ids go to the device with the masks, padded with −1 to `rows` (a static shape for
the graphs); past them, or with a mask that allows every token, it takes the dense head. The host computes the union,
its popcount and its ids while the step's forward runs (`GrammarHeads.choose`).

**The universe.** The OR of the unions of the eligible steps that fit in `rows`. On those steps, whether or not the
sparse head is on, the drafter (Gemma 4's assistant, Qwen's MTP layer) drafts over it (`draft_head.SubsetHead`) instead
of its own head: once it holds the next step's masks, a draft outside it could never equal the masked sample. A drafter
whose head is FR-Spec's ranked rows (Qwen's `draft_vocab`) is replaced by the whole universe, not its intersection with
those rows, so a schema's tokens that the ranking lacks are drafted at once. Past `rows` tokens the universe turns off
for good, logged once.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..base import UNION_BUCKETS, HeadCounters
from .draft_head import SubsetHead, fp32_reduction

log = logging.getLogger(__name__)

Heads = tuple[bool, bool]  # (the sparse target head, the universe draft head)
NO_HEADS: Heads = (False, False)
# The sampling graphs' suffixes per heads (`graphs.graph_names`)
VARIANT: dict[Heads, str] = {NO_HEADS: "s", (True, False): "q", (True, True): "p", (False, True): "u"}
VARIANT_HEADS: dict[str, Heads] = {v: h for h, v in VARIANT.items()}


def union_words(masks: np.ndarray, tmp: np.ndarray) -> np.ndarray:
    """The OR of mask rows [n, W] (any one integer dtype, n >= 1) into one row [W]: a tree of whole-row ORs into `tmp`
    ([>= n // 2, W], the same dtype). A view of `masks` or `tmp`: copy it to keep it."""
    n = len(masks)
    if n == 1:
        return masks[0]
    h = n // 2
    np.bitwise_or(masks[:h], masks[h:2 * h], out=tmp[:h])
    if n % 2:
        np.bitwise_or(tmp[0], masks[n - 1], out=tmp[0])
    while h > 1:
        g = h // 2
        np.bitwise_or(tmp[:g], tmp[g:2 * g], out=tmp[:g])
        if h % 2:
            np.bitwise_or(tmp[0], tmp[h - 1], out=tmp[0])
        h = g
    return tmp[0]


def ids_of(words: np.ndarray) -> np.ndarray:
    """The set bits of unsigned mask words (token t is bit t % w of word t // w, w the word's bits; a little-endian
    host) as ascending int32 token ids."""
    w = words.dtype.itemsize * 8
    nz = np.flatnonzero(words)
    bits = np.flatnonzero(np.unpackbits(words[nz].view(np.uint8), bitorder="little"))
    return (nz[bits // w] * w + bits % w).astype(np.int32)


class GrammarHeads:
    """A runner's grammar-aware heads on the host (module docstring): `choose` per split step, the universe, the
    counters (`counters`: `runner.base.HeadCounters`). `words`: int32 words per mask row; `vocab`: the model's."""

    def __init__(self, *, sparse: bool, universe: bool, rows: int, vocab: int, words: int):
        self.sparse, self.universe, self.rows, self.vocab = sparse, universe, rows, vocab
        self.dtype = np.uint64 if words % 2 == 0 else np.uint32  # the OR's word (two int32 mask words when they pair)
        self.width = words // 2 if self.dtype == np.uint64 else words
        self._tmp = np.zeros((0, self.width), self.dtype)
        self._uni = np.zeros(self.width, self.dtype)
        self.uni_n = 0
        self.uni_off = False  # the universe passed `rows`: off for good
        self.pending: np.ndarray | None = None  # the universe's ids, to install before the next step drafting over it
        self.n_sparse = self.n_off = self.n_unconstrained = self.n_over_cap = 0
        self.union_counts = np.zeros(len(UNION_BUCKETS) + 1, np.int64)
        self.union_sum = 0
        self.uni_steps = self.uni_rebuilds = 0

    @property
    def on(self) -> bool:
        return self.sparse or self.universe

    def choose(self, masks: np.ndarray, constrained: bool) -> tuple[Heads, np.ndarray | None]:
        """The heads of a split step whose mask rows are `masks` (int32 [n, words], every row the engine filled for it)
        and whose every real sample token has one (`constrained`), and the union's ascending ids when it samples on the
        sparse head. Counts the step; grows the universe (its ids in `pending` when it grew)."""
        if not self.on:
            self.n_off += 1
            return NO_HEADS, None
        if not constrained:
            self.n_unconstrained += 1
            return NO_HEADS, None
        if len(self._tmp) < len(masks) // 2:
            self._tmp = np.zeros((len(masks) // 2, self.width), self.dtype)
        u = union_words(masks.view(self.dtype), self._tmp)
        pop = int(np.bitwise_count(u).sum())
        self.union_counts[bisect_left(UNION_BUCKETS, pop)] += 1
        self.union_sum += pop
        if pop >= self.vocab:  # a mask of every token (an unconstrained row's −1 words), even within the rows
            self.n_unconstrained += 1
            return NO_HEADS, None
        if pop > self.rows:
            self.n_over_cap += 1
            return NO_HEADS, None
        uni = False
        if self.universe and not self.uni_off:
            if (u & ~self._uni).any():  # it grows: the step's own union is in before its drafting
                np.bitwise_or(self._uni, u, out=self._uni)
                n = int(np.bitwise_count(self._uni).sum())
                if n > self.rows:
                    self.uni_off, self.uni_n, self.pending = True, 0, None
                    log.warning("the grammar's universe passed dev.runner.cuda.sparse_head_rows (%d tokens): the "
                                "drafter scores its own head from now on (the whole vocabulary, or its ranked "
                                "subset: dev.speculative.draft_vocab)", self.rows)
                else:
                    self.uni_n, self.pending = n, ids_of(self._uni)
                    self.uni_rebuilds += 1
            uni = not self.uni_off
        self.uni_steps += uni
        if not self.sparse:
            self.n_off += 1
            return (False, uni), None
        self.n_sparse += 1
        return (True, uni), ids_of(u)

    def counters(self) -> HeadCounters:
        return HeadCounters(sparse=self.n_sparse, dense_off=self.n_off, dense_unconstrained=self.n_unconstrained,
                            dense_over_cap=self.n_over_cap, union_counts=tuple(int(c) for c in self.union_counts),
                            union_sum=self.union_sum, universe=self.uni_n, universe_steps=self.uni_steps,
                            universe_rebuilds=self.uni_rebuilds)


class GrammarSampling:
    """A CUDA runner's grammar masks and grammar-aware heads (as `fp8.Fp8Runtime` and `images.ImageEncoder`): the host
    side (`heads`: `GrammarHeads`, None without `grammar`), the universe's draft head (`uni_head`: a `SubsetHead` over
    `draft_weight`, the drafter's head weight; None without the universe), the sampling graphs' variants (`variants`,
    in capture order), and per `allocate` the buffers: the masks' page-locked buffer per staging parity, which the
    engine fills between `launch_forward` and `launch_sample` (the sparse head's ids are staged beside them), and the
    device buffers a split step's sampling reads (`mask_src`, `sp_src`: the step's host buffers on CPU). Per split step
    the runner calls `choose`, then `upload` and `install_universe`, then runs its sampling, where `sparse_logits`
    scores the union. `head`: the target head's weight [vocab, hidden]; `gemm` and `finish`: the GEMM its logits take
    and what follows it before the widening to fp32, the model's own (`sparse_logits`). `fixed_m`: the sparse head's
    GEMM at one M for every step (`allocate`'s mask rows)."""

    def __init__(self, *, grammar: bool, sparse: bool, universe: bool, rows: int, vocab: int, head: torch.Tensor,
                 draft_weight: torch.Tensor | None = None,
                 gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.linear,
                 finish: Callable[[torch.Tensor], torch.Tensor] = lambda y: y, fixed_m: bool = False):
        if rows < 1:
            raise ValueError(f"dev.runner.cuda.sparse_head_rows {rows}: at least 1")
        self.grammar, self.head, self.gemm, self.finish, self.fixed_m = grammar, head, gemm, finish, fixed_m
        self.cuda = head.device.type == "cuda"
        self.rows = min(rows, vocab)  # more rows than the vocabulary: no union is ever over
        self.sparse = sparse and grammar
        self.universe = universe and grammar and draft_weight is not None
        self.words = (vocab + 31) // 32  # int32 words per mask row
        self.uni_head = SubsetHead(draft_weight, self.rows) if self.universe else None
        self.heads = GrammarHeads(sparse=self.sparse, universe=self.universe, rows=self.rows, vocab=vocab,
                                  words=self.words) if grammar else None
        self.variants = ["s"] + (["q"] if self.sparse else []) + (["p"] if self.sparse and self.universe else []) + \
            (["u"] if self.universe and not self.sparse else [])
        self.release()  # no buffers before `allocate`

    def allocate(self, mask_rows: int) -> None:
        """The buffers of `mask_rows` mask rows (the most sample tokens a step has: max_num_seqs × (1 + k))."""
        cuda, dev = self.cuda, self.head.device
        if self.grammar:
            self.mask_host = [torch.zeros((mask_rows, self.words), dtype=torch.int32, pin_memory=cuda)
                              for _ in range(2)]
            self.mask_np = [t.numpy() for t in self.mask_host]
            self.mask_dev = torch.full((mask_rows, self.words), -1, dtype=torch.int32, device=dev) if cuda else None
            self.mask_src = self.mask_dev
            self.mask_event = torch.cuda.Event() if cuda else None
        if self.sparse:  # the sparse head's ids per step (staged like the masks) and its gathered rows
            hidden, dt = self.head.shape[1], self.head.dtype
            self.sp_host = [torch.full((self.rows,), -1, dtype=torch.int32, pin_memory=cuda) for _ in range(2)]
            self.sp_np = [t.numpy() for t in self.sp_host]
            self.sp_dev = torch.full((self.rows,), -1, dtype=torch.int32, device=dev) if cuda else None
            self.sp_src = self.sp_dev if cuda else self.sp_host[0]
            self.sp_w = torch.zeros((self.rows, hidden), dtype=dt, device=dev)
            self.sp_x = torch.zeros((mask_rows, hidden), dtype=dt, device=dev) if self.fixed_m else None

    def release(self) -> None:
        self.mask_host = self.mask_np = self.mask_dev = self.mask_src = self.mask_event = None
        self.sp_host = self.sp_np = self.sp_dev = self.sp_src = self.sp_w = self.sp_x = None

    def mask_buffer(self, parity: int) -> np.ndarray:
        """The masks' host buffer of a staging parity (`Runner.mask_buffer`)."""
        return self.mask_np[parity]

    def choose(self, parity: int, n_masks: int, constrained: bool, *, internal: bool = False,
               internal_heads: Heads | None = None) -> Heads:
        """A split step's heads, from the `n_masks` masks the engine just filled into `parity`'s buffer (`constrained`:
        every real sample token has one), the union's ids staged with them for the sparse head. An internal step
        (profile, warm-up, capture) takes `internal_heads` (None: NO_HEADS), the sparse head's ids those of the first
        tokens, and moves no counter. A step without masks (always split) samples on the dense head, as its fused graph
        would, and moves no counter either."""
        heads, ids = NO_HEADS, None
        if internal:
            heads = internal_heads or NO_HEADS
            ids = np.arange(min(16, self.rows), dtype=np.int32) if heads[0] else None
        elif self.heads is not None and n_masks:
            heads, ids = self.heads.choose(self.mask_np[parity][:n_masks], constrained)
        if ids is not None:
            a = self.sp_np[parity]
            a[:len(ids)] = ids
            a[len(ids):] = -1
        return heads

    def upload(self, parity: int, n_masks: int, heads: Heads, copy_stream: Any) -> None:
        """The step's masks and the sparse head's ids to the device buffers on `copy_stream` (the forward may still
        run), which the current stream waits for; nothing to upload without masks or ids (always split). On CPU the
        sampling reads `parity`'s host buffers."""
        if self.cuda:
            if n_masks or heads[0]:
                with torch.cuda.stream(copy_stream):
                    self.mask_dev[:n_masks].copy_(self.mask_host[parity][:n_masks], non_blocking=True)
                    if heads[0]:
                        self.sp_dev.copy_(self.sp_host[parity], non_blocking=True)
                    self.mask_event.record(copy_stream)
                torch.cuda.current_stream().wait_event(self.mask_event)
        else:
            self.mask_src = self.mask_host[parity]
            if heads[0]:
                self.sp_src = self.sp_host[parity]

    def install_universe(self, heads: Heads, internal: bool) -> None:
        """The universe's head rebuilt once it grew (`GrammarHeads.pending`), on the current stream before the sampling
        that drafts over it; internal steps never touch it."""
        if heads[1] and not internal and self.heads.pending is not None:
            self.uni_head.install(self.heads.pending)
            self.heads.pending = None

    def sparse_logits(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """[n, rows] fp32: the head's rows of this step's union (`sp_src`: ascending ids, then −1, whose row 0 is
        gathered, finite and never drawn), their GEMM (`gemm`, fp32 partial sums only: `draft_head.fp32_reduction`),
        `finish` and the widening, as the model's logits compute every row's; graph-safe. With `fixed_m` the GEMM runs
        at one M, the most sample tokens a step has (`sp_x`, the rest zeros): cuBLAS picks its kernel by M, so a row's
        logits would otherwise depend on the step's rows. `out` ([>= n, rows] fp32, the logprobs' buffer:
        `runner.cuda.logprobs`): the widening writes its first n rows, returned."""
        torch.index_select(self.head, 0, self.sp_src.clamp(min=0).long(), out=self.sp_w)
        n, x = hidden.shape[0], hidden
        if self.sp_x is not None:
            x = self.sp_x
            x[:n].copy_(hidden)
            x[n:].zero_()
        with fp32_reduction():
            y = self.finish(self.gemm(x, self.sp_w)[:n])
        return y.float() if out is None else out[:n].copy_(y)

    def counters(self) -> HeadCounters:
        """`Runner.head_counters`: host ints, no device sync (zeros without grammar masks)."""
        return self.heads.counters() if self.heads is not None else HeadCounters()

    def describe(self) -> str:
        """The startup line (with grammar masks)."""
        mib = self.rows * self.head.shape[1] * self.head.element_size() / 2**20
        return (f"grammar heads: the sparse head {'on' if self.sparse else 'off'}, the draft universe "
                f"{'on' if self.universe else 'off'}, at most {self.rows} tokens ({mib:.1f} MiB of the target's head "
                "rows per sparse step)")

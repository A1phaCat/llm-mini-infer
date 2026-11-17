"""The CUDA runners' shared grammar sampling and graph helpers, on CPU: the graphs' names in capture order and their
parsing (`runner.cuda.graphs`), `stats["graphs"]`, a split step's heads, staging and sparse head
(`runner.cuda.heads.GrammarSampling`: the union's columns of the dense logits, for Gemma 4's softcapped head and a
plain one), the sampler's call (`runner.cuda.sampler.draw`, `host_plan`), the split decision and a step's timing
arithmetic (`runner.cuda.runner`). The runners' own tests (tests/models/test_gemma4_heads.py,
tests/structured/test_always_split.py) run them end to end."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from llm_mini_infer.models.gemma4.model import linear, softcap
from llm_mini_infer.runner.cuda.graphs import (
    BreakableCapture,
    Capture,
    graph_name,
    graph_names,
    graph_summary,
    parse_graph_name,
)
from llm_mini_infer.runner.cuda.heads import NO_HEADS, VARIANT, VARIANT_HEADS, GrammarSampling
from llm_mini_infer.runner.cuda.runner import forward_running, plan_bracket, split_step, step_timing
from llm_mini_infer.runner.cuda.sampler import draw, host_plan, row_keys, sample

V, H = 300, 16  # a vocabulary of 300 tokens: 10 mask words per row


def masks_of(rows: list[list[int]], words: int = (V + 31) // 32) -> np.ndarray:
    m = np.zeros((len(rows), words), np.uint32)
    for i, toks in enumerate(rows):
        for t in toks:
            m[i, t // 32] |= np.uint32(1 << (t % 32))
    return m.view(np.int32)


@pytest.mark.parametrize("variants", [["s"], ["s", "q"], ["s", "q", "p"], ["s", "u"]])
def test_the_graphs_names_in_capture_order_and_back(variants) -> None:
    """Per bucket kind the fused graph, then the forward and a sampling graph per variant (with grammar masks); always
    split, the fused one is not captured. Every name parses into the (kind, part, heads) that names it again."""
    for kind in ("decode", "break"):
        split = (f"{kind}_f", *(f"{kind}_{v}" for v in variants))
        assert graph_names(kind, grammar=False, always_split=False, variants=variants) == (kind,)
        assert graph_names(kind, grammar=True, always_split=False, variants=variants) == (kind, *split)
        assert graph_names(kind, grammar=True, always_split=True, variants=variants) == split
        for name in (kind, *split):
            k, part, heads = parse_graph_name(name)
            assert k == kind and graph_name(k, part, heads) == name
        assert parse_graph_name(kind) == (kind, "all", NO_HEADS)
        assert parse_graph_name(f"{kind}_f") == (kind, "fwd", NO_HEADS)
        for v in variants:
            assert parse_graph_name(f"{kind}_{v}") == (kind, "smp", VARIANT_HEADS[v])
            assert graph_name(kind, "smp", VARIANT_HEADS[v]) == f"{kind}_{v}"
    assert graph_name("decode", "smp", (True, True)) == "decode_p" and VARIANT[NO_HEADS] == "s"


def test_the_graphs_summary() -> None:
    """`stats["graphs"]` (both runners' keys, in this order): the buckets with graphs, fused or split, the forward
    graphs, the fused ones, every graph, the sampling variants and the tower's buckets."""
    variants = ["s", "q", "p"]
    for always in (False, True):
        keys = {(name, b): None for b in (64, 32) for name in graph_names("break", grammar=True, always_split=always,
                                                                          variants=variants)}
        keys.update({(name, b): None for b in (4, 2, 1) for name in graph_names("decode", grammar=True,
                                                                                 always_split=always,
                                                                                 variants=variants)})
        s = graph_summary(keys, variants=variants, vision=[256])
        assert list(s) == ["decode", "break", "split", "fused", "captured", "sampling_variants", "vision"]
        assert s == {"decode": [1, 2, 4], "break": [32, 64], "split": 5, "fused": 0 if always else 5,
                     "captured": 5 * (4 if always else 5), "sampling_variants": variants, "vision": [256]}
    keys = {(k, b): None for k, b in (("break", 16), ("decode", 1), ("decode", 2))}
    assert graph_summary(keys, variants=[], vision=[]) == {"decode": [1, 2], "break": [16], "split": 0, "fused": 3,
                                                            "captured": 3, "sampling_variants": [], "vision": []}


class StubGraph:
    """torch.cuda.CUDAGraph's capture and replay calls, recorded in `calls` in order (`fail_begin`: the
    second capture_begin raises; `fail_end`: capture_end raises)."""

    calls: ClassVar[list[str]] = []
    made = 0
    fail_end = False
    fail_begin = False

    def __init__(self) -> None:
        StubGraph.made += 1
        self.n = StubGraph.made

    def capture_begin(self, pool: object = None) -> None:
        StubGraph.calls.append(f"begin {self.n}")
        if StubGraph.fail_begin and self.n > 1:
            raise RuntimeError("capture_begin failed")

    def capture_end(self) -> None:
        StubGraph.calls.append(f"end {self.n}")
        if StubGraph.fail_end:
            raise RuntimeError("capture_end failed too")

    def replay(self) -> None:
        StubGraph.calls.append(f"replay {self.n}")


@pytest.fixture
def stub_graphs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "CUDAGraph", StubGraph, raising=False)
    monkeypatch.setattr(StubGraph, "calls", [])
    monkeypatch.setattr(StubGraph, "made", 0)
    monkeypatch.setattr(StubGraph, "fail_end", False)
    monkeypatch.setattr(StubGraph, "fail_begin", False)
    return StubGraph.calls


def test_a_breakable_capture_replays_its_segments_in_order(stub_graphs) -> None:
    """Graph segments and the eager ops between them replay in capture order, once the `with` has ended; an eager
    break outside a capture, or a replay before its end, is refused."""
    calls = stub_graphs
    cap = BreakableCapture(None)
    with pytest.raises(RuntimeError, match="ended when NEW"):
        cap.eager(lambda: None)
    with cap:
        with pytest.raises(RuntimeError, match="RECORDING, not CAPTURED"):
            cap.replay()
        cap.eager(lambda: calls.append("op"))
    assert cap.state is Capture.CAPTURED and (cap.graphs, cap.breaks) == (2, 1)
    assert calls == ["begin 1", "end 1", "op", "begin 2", "end 2"]
    calls.clear()
    cap.replay()
    assert calls == ["replay 1", "op", "replay 2"]
    with pytest.raises(RuntimeError, match="entered when CAPTURED"):
        cap.__enter__()


def test_a_failed_capture_ends_once_and_never_replays(stub_graphs) -> None:
    """An eager op that raises leaves its segment ended once (`capture_end` is not called again on the way out); an
    error inside a segment ends it, and is the error raised even when ending fails too. Either capture is FAILED."""
    calls = stub_graphs

    def boom() -> None:
        raise ValueError("the op failed")

    cap = BreakableCapture(None)
    with pytest.raises(ValueError, match="the op failed"), cap:
        cap.eager(boom)
    assert calls == ["begin 1", "end 1"] and cap.state is Capture.FAILED
    with pytest.raises(RuntimeError, match="FAILED, not CAPTURED"):
        cap.replay()
    calls.clear()
    StubGraph.fail_end = True
    cap = BreakableCapture(None)
    with pytest.raises(ValueError, match="inside the segment"), cap:
        raise ValueError("inside the segment")
    assert calls == ["begin 2", "end 2"] and cap.state is Capture.FAILED and cap.segments == []


def test_a_capture_end_that_raises_on_a_clean_exit_is_the_error_and_fails_the_capture(stub_graphs) -> None:
    """The segment is FAILED before `capture_end` runs, so a `capture_end` that raises at the end of the `with` is the
    error raised, called once, and leaves the capture FAILED."""
    calls = stub_graphs
    StubGraph.fail_end = True
    cap = BreakableCapture(None)
    with pytest.raises(RuntimeError, match="capture_end failed too"), cap:
        pass
    assert calls == ["begin 1", "end 1"] and cap.state is Capture.FAILED and cap.segments == []


def test_a_capture_begin_that_raises_after_an_eager_op_fails_without_ending(stub_graphs) -> None:
    """The next segment's `capture_begin` raising after the op ran leaves the capture in BREAK with no segment open:
    leaving the `with` marks it FAILED and calls no `capture_end`."""
    calls = stub_graphs
    StubGraph.fail_begin = True
    cap = BreakableCapture(None)
    with pytest.raises(RuntimeError, match="capture_begin failed"), cap:
        cap.eager(lambda: calls.append("op"))
    assert calls == ["begin 1", "end 1", "op", "begin 2"] and cap.state is Capture.FAILED


def grammar_sampling(head: torch.Tensor, rows: int = 64, **kw) -> GrammarSampling:
    gs = GrammarSampling(grammar=True, sparse=True, universe=False, rows=rows, vocab=V, head=head, **kw)
    gs.allocate(8)
    return gs


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("fixed_m", [False, True])
def test_the_sparse_head_scores_the_dense_heads_columns(gemma, fixed_m) -> None:
    """A constrained split step: the union of its masks' ids is staged (ascending, then −1 to the rows) and the sparse
    head's logits are the dense logits' columns of those ids, through the model's GEMM and finishing (Gemma 4's
    `linear` and BF16-order softcap, or a plain head); padding columns gather row 0, finite. At one M too."""
    g = torch.Generator().manual_seed(0)
    head = torch.randn((V, H), generator=g)
    hidden = torch.randn((5, H), generator=g) * 4
    gemm, finish = (linear, lambda y: softcap(y, 3.0).float()) if gemma else (F.linear, lambda y: y.float())
    gs = grammar_sampling(head, gemm=gemm, finish=finish, fixed_m=fixed_m)
    rows = [[3, 290, 41], [41, 7], [299, 0, 128, 3], [64], [100, 101, 102]]
    gs.mask_np[1][:5] = masks_of(rows)
    heads = gs.choose(1, 5, True)
    assert heads == (True, False)
    gs.upload(1, 5, heads, None)  # CPU: the sampling reads parity 1's host buffers
    ids = sorted({t for r in rows for t in r})
    assert gs.sp_src.tolist() == ids + [-1] * (64 - len(ids)) and gs.mask_src is gs.mask_host[1]
    got = gs.sparse_logits(hidden)
    want = finish(gemm(hidden, head))
    assert got.shape == (5, 64) and got.dtype == torch.float32
    torch.testing.assert_close(got[:, :len(ids)], want[:, ids], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[:, len(ids):], want[:, :1].expand(5, 64 - len(ids)), rtol=1e-5, atol=1e-5)
    c = gs.counters()
    assert (c.sparse, c.dense_off, c.dense_unconstrained, c.dense_over_cap, c.union_sum) == (1, 0, 0, 0, len(ids))


def test_the_heads_each_split_step_takes() -> None:
    """The dense head for an unconstrained step, a union over the rows or a step without masks (always split: no
    counter moves); internal steps take their caller's heads, the first tokens' ids, and move no counter."""
    gs = grammar_sampling(torch.zeros((V, H)), rows=4)
    assert gs.choose(0, 0, True) == NO_HEADS  # no masks
    gs.mask_np[0][:2] = masks_of([[1, 2], [3]])
    assert gs.choose(0, 2, False) == NO_HEADS  # a sample token without a mask
    gs.mask_np[0][:2] = masks_of([[1, 2, 5], [3, 9]])
    assert gs.choose(0, 2, True) == NO_HEADS  # 5 tokens: over the 4 rows
    c = gs.counters()
    assert (c.sparse, c.dense_unconstrained, c.dense_over_cap) == (0, 1, 1)
    assert gs.choose(1, 0, False, internal=True, internal_heads=(True, False)) == (True, False)
    assert gs.sp_np[1].tolist() == [0, 1, 2, 3]
    assert gs.choose(1, 3, True, internal=True) == NO_HEADS
    assert gs.counters() == c
    off = GrammarSampling(grammar=False, sparse=True, universe=True, rows=2048, vocab=V, head=torch.zeros((V, H)),
                          draft_weight=torch.zeros((V, H)))
    assert (off.heads, off.uni_head, off.sparse, off.universe, off.rows, off.variants) == (None, None, False, False, V,
                                                                                         ["s"])
    with pytest.raises(ValueError, match="at least 1"):
        GrammarSampling(grammar=True, sparse=True, universe=False, rows=0, vocab=V, head=torch.zeros((V, H)))


def test_the_variants_and_the_startup_line() -> None:
    """The sampling graphs' variants per switch (dense and whole first), and the startup line."""
    head = torch.empty((262144, 5376), dtype=torch.bfloat16, device="meta")
    drafts = torch.empty((262144, 1024), dtype=torch.bfloat16, device="meta")

    def gs(sparse: bool, universe: bool, draft=drafts) -> GrammarSampling:
        return GrammarSampling(grammar=True, sparse=sparse, universe=universe, rows=1024, vocab=262144, head=head,
                               draft_weight=draft)

    assert gs(False, False).variants == ["s"] and gs(True, False).variants == ["s", "q"]
    assert gs(True, True).variants == ["s", "q", "p"] and gs(False, True).variants == ["s", "u"]
    assert gs(True, True, draft=None).variants == ["s", "q"]  # no drafts: no universe
    assert gs(True, False).describe() == ("grammar heads: the sparse head on, the draft universe off, at most 1024 "
                                          "tokens (10.5 MiB of the target's head rows per sparse step)")


def test_draw_on_cpu_is_the_reference_sampler() -> None:
    """`draw` on CPU: the torch reference over `host_plan`'s groups, keyed at qpos + 1, writing the tokens, their
    statuses and (when given) each slot's pending token; with ids, the columns' tokens."""
    n, slots = 4, torch.tensor([1, 2, 3, 4])
    mirror = SimpleNamespace(temperature=np.array([0, 0, 0.8, 1.0, 1.0], np.float32),
                             top_p=np.array([1, 1, 0.9, 1, 1], np.float32), top_k=np.array([0, 0, 0, 5, 0]),
                             min_p=np.zeros(5, np.float32))
    tables = SimpleNamespace(**{f: torch.as_tensor(getattr(mirror, f)) for f in ("temperature", "top_p", "top_k",
                                                                                 "min_p")},
                             seed=torch.arange(5, dtype=torch.int64) + 11, pending=torch.full((5,), -1))
    plan = host_plan(mirror, slots.numpy(), V)
    assert [g.kind for g in plan.groups] == ["topk", "sort", "plain"] and all(g.idx is not None for g in plan.groups)
    logits = torch.randn((n, V), generator=torch.Generator().manual_seed(1))
    qpos = torch.tensor([5, 9, 2, 30])
    out, status = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)
    draw(logits, None, False, cuda=False, tables=tables, slots=slots, qpos=qpos, out=out, status=status,
         pending=tables.pending, plan=plan, mask=None, vocab_ids=torch.arange(V))
    want, st = sample(logits, plan, tables.temperature[slots], tables.top_p[slots], tables.top_k[slots],
                      tables.min_p[slots], row_keys(tables.seed[slots], qpos + 1), torch.arange(V))
    assert out.tolist() == want.tolist() and status.tolist() == st.tolist() == [0] * n
    assert tables.pending[1:].tolist() == want.tolist()
    ids = torch.arange(V, dtype=torch.int32).flip(0)  # columns in another order: the same tokens, as ids
    draw(logits.flip(1), ids, False, cuda=False, tables=tables, slots=slots, qpos=qpos, out=out, status=status,
         pending=None, plan=plan, mask=None, vocab_ids=torch.arange(V))
    assert out.tolist() == want.tolist()


class _Ev:
    def __init__(self, t: float, done: bool = True):
        self.t, self.done = t, done

    def elapsed_time(self, end: _Ev) -> float:
        return end.t - self.t

    def query(self) -> bool:
        return self.done


def test_the_split_decision_and_a_steps_timing() -> None:
    """A step splits with masks, or always split with rows; its forward runs until its forward-end event; its timing
    (ms events, ns fields): busy less the gap between its parts, idle before since the previous step's end, a late
    launch's planning moved from busy to idle before, a split step's sampling; on CPU nothing idles."""
    assert split_step(2, 3, False) and split_step(0, 3, True)
    assert not split_step(0, 3, False) and not split_step(0, 0, True)
    h = SimpleNamespace(fwd_end=None)
    assert not forward_running(h)
    h.fwd_end = _Ev(0, done=False)
    assert forward_running(h)
    h.fwd_end.done = True
    assert not forward_running(h)
    h = SimpleNamespace(events=[_Ev(10.0), _Ev(14.0), _Ev(15.5), _Ev(17.0)], prev_end=_Ev(9.0), plan_events=None,
                        enc_events=[_Ev(10.25), _Ev(11.0)], fwd_end=_Ev(14.0))
    assert step_timing(h, True) == dict(gpu_busy_ns=5_500_000, gpu_idle_before_ns=1_000_000,
                                        gpu_idle_inside_ns=1_500_000, encoder_busy_ns=750_000,
                                        gpu_sample_ns=1_500_000)
    h.plan_events, h.fwd_end, h.enc_events = (_Ev(11.0), _Ev(12.0)), None, None  # a late launch's plans: 1 ms
    assert step_timing(h, True) == dict(gpu_busy_ns=4_500_000, gpu_idle_before_ns=2_000_000,
                                        gpu_idle_inside_ns=1_500_000, encoder_busy_ns=0, gpu_sample_ns=0)
    h.plan_events = None
    assert step_timing(h, False) == dict(gpu_busy_ns=5_500_000, gpu_idle_before_ns=0, gpu_idle_inside_ns=0,
                                         encoder_busy_ns=0, gpu_sample_ns=0)
    h.plan_events = "kept"
    with plan_bracket(h, False):  # untimed (CPU, a capture): the handle's events stay
        pass
    assert h.plan_events == "kept"

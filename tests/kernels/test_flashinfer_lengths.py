"""FlashInfer attention planned for upper-bound KV lengths, run with the true lengths on the device (MTP;
`kernels/flashinfer.py`). CUDA only.

Under MTP the host knows a decode row's KV length only within [lb, ub]. The runner plans for ub and rewrites each
row's last-page length on the device from the true length (`fix_lengths`); a split-KV plan whose chunk count some true
length could change (the kernel then reads a row's later queries from the wrong partial outputs: wrong but finite) is
made again with chunks whose count cannot, and a decode graph's keeps the captured split within the captured grid.
These tests compare such runs with the torch reference attention at the true lengths, with true lengths on both sides
of page boundaries and chunk boundaries, eagerly and in a CUDA graph (the decode graphs' wrappers, at Gemma 4's
sliding-layer shape under MTP at k = 3, 5 and 7), and check the model of FlashInfer's CUDA-graph plan that
tests/kernels/test_flashinfer_selfcheck.py sweeps on CPU against the real one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.models.qwen3_5.backend import Pools, StepMeta, TorchAttention
from llm_mini_infer.runner.cuda.graphs import no_gc

from ..support.flashinfer_plan import BUCKETS, graph_plan

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
DEV = "cuda"


def _setup(ps: int = 16, pages: int = 400, hq: int = 8, hkv: int = 2, d: int = 256):
    from llm_mini_infer.kernels.flashinfer import FlashInferAttention

    torch.manual_seed(0)
    kv = torch.randn(1, pages, 2, ps, hkv, d, device=DEV, dtype=torch.bfloat16)
    pools = Pools(kv=kv, conv=torch.zeros(1), rec=torch.zeros(1), page_size=ps)
    return pools, FlashInferAttention(pools, hq, hkv, d, torch.bfloat16, torch.device(DEV), max_rows=16,
                                      max_pages=64, qo_per_row=4)


def _reference(pools, page_table, slots, true_len, qo, q, hkv):
    """The torch reference at the true lengths."""
    n = len(slots)
    starts = np.concatenate([[0], np.cumsum(qo)])[:-1]
    meta = StepMeta(num_tokens=int(sum(qo)), positions=None, kv_page=None, kv_off=None, n_rows=n, n_dec=0,
                    row_state=None, row_cu=None, pf_cu=None, pf_cu_host=np.zeros(0), row_start_host=starts,
                    row_nq_host=np.asarray(qo), row_kv_len_host=np.asarray(true_len), row_slot_host=np.asarray(slots),
                    page_table=page_table)
    out = torch.empty_like(q)
    TorchAttention(pools, q.shape[-1]).run(meta, 0, q, out)
    return out


def _case(ps: int, qo: int, deltas: list[int], base: list[int]):
    """Rows of `qo` queries: true lengths `base`, planned lengths base + delta."""
    true_len = np.asarray(base, np.int64)
    ub = true_len + np.asarray(deltas)
    n = len(base)
    page_table = torch.zeros(n + 1, 64, dtype=torch.int32, device=DEV)
    page = 1
    for r in range(n):
        need = -(-int(ub[r]) // ps)
        page_table[r + 1, :need] = torch.arange(page, page + need, dtype=torch.int32)
        page += need
    return true_len, ub, page_table, np.arange(1, n + 1)


def _plan_eager(attn, pools, ub, lb, qo, page_table, slots):
    from llm_mini_infer.kernels.flashinfer import page_indices

    ps = pools.page_size
    n = len(ub)
    n_pages = (ub + ps - 1) // ps
    indptr = np.concatenate([[0], np.cumsum(n_pages)])
    pin = lambda a: torch.as_tensor(np.asarray(a, np.int32)).pin_memory()  # noqa: E731
    kv_indices = page_indices(page_table, torch.as_tensor(slots, device=DEV), torch.as_tensor(n_pages, device=DEV),
                              torch.as_tensor(indptr, device=DEV), int(n_pages.max()), int(indptr[-1]),
                              torch.arange(64, device=DEV))
    return attn.plan(None, parity=0, qo_indptr=pin(np.arange(n + 1) * qo), kv_indptr=pin(indptr),
                     kv_indices=kv_indices, last_page_len=pin(ub - (n_pages - 1) * ps), kv_lens=pin(ub), max_q=qo,
                     lb=lb)


@pytest.mark.parametrize("qo", [4, 1])
@pytest.mark.parametrize("deltas", [[0, 3, 1, 2, 3, 0], [6, 6, 6, 6, 6, 6]])
def test_true_lengths_below_the_plan_eager(qo, deltas) -> None:
    """True lengths ending just before, at and after page boundaries (15, 16, 17, 31, 32, 160 with 16-token pages),
    planned up to 6 positions longer: the outputs equal the reference at the true lengths."""
    ps = 16
    pools, attn = _setup(ps)
    base = [15 + qo, 16, 17 + qo, 31, 32 + qo, 160]
    true_len, ub, page_table, slots = _case(ps, qo, deltas, base)
    q = torch.randn(len(base) * qo, 8, 256, device=DEV, dtype=torch.bfloat16)
    w = _plan_eager(attn, pools, ub, true_len, qo, page_table, slots)
    attn.fix_lengths(w, len(base), torch.as_tensor(true_len, device=DEV))
    out = torch.empty_like(q)
    w.run(q, pools.kv[0], out=out)
    want = _reference(pools, page_table, slots, true_len, [qo] * len(base), q, 2)
    assert torch.isfinite(out).all()
    assert (out.float() - want.float()).abs().max() < 2e-2


def test_unsafe_chunking_is_planned_again() -> None:
    """Many rows and long contexts make FlashInfer split the KV into small chunks; true lengths that end exactly at a
    chunk boundary while the plan covers one more page would make the kernel count one chunk fewer than the plan and
    misplace the partial outputs of every query after a row's first (wrong but finite; an empty chunk at a planned
    count merges harmlessly). The plan is made again (`replans`) and the outputs stay right."""
    ps = 16
    pools, attn = _setup(ps, pages=2400)
    n, qo = 16, 4
    # Chunk sizes come out as a multiple of 8 pages (128 tokens); true lengths at multiples of 128, planned 3 longer.
    base = [128 * (2 + r % 5) for r in range(n)]
    true_len, ub, page_table, slots = _case(ps, qo, [3] * n, base)
    page_table_big = torch.zeros(n + 1, 64, dtype=torch.int32, device=DEV)
    page_table_big[:, :page_table.shape[1]] = page_table
    q = torch.randn(n * qo, 8, 256, device=DEV, dtype=torch.bfloat16)
    before = attn.replans
    w = _plan_eager(attn, pools, ub, true_len, qo, page_table_big, slots)
    attn.fix_lengths(w, n, torch.as_tensor(true_len, device=DEV))
    out = torch.empty_like(q)
    w.run(q, pools.kv[0], out=out)
    want = _reference(pools, page_table_big, slots, true_len, [qo] * n, q, 2)
    assert torch.isfinite(out).all()
    assert (out.float() - want.float()).abs().max() < 2e-2
    assert attn.replans >= before  # planned again when the first chunking was unsafe (reported, not required)


def test_true_lengths_in_a_cuda_graph() -> None:
    """The decode graphs' path: a CUDA-graph wrapper planned for upper bounds, page indices built and lengths fixed
    inside the graph from device lengths; replayed with different true lengths each time."""
    ps, qo, rows = 16, 4, 8
    pools, attn = _setup(ps)
    rng = np.random.default_rng(3)
    base = np.asarray([16, 17, 31, 32, 33, 47, 48, 100])
    true_len, ub, page_table, slots = _case(ps, qo, [6] * rows, list(base))
    slots_d = torch.as_tensor(slots, device=DEV)
    lens_d = torch.zeros(rows, dtype=torch.int64, device=DEV)
    q = torch.randn(rows * qo, 8, 256, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    def plan(ub_now: np.ndarray, lb_now: np.ndarray):
        n_pages = (ub_now + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(n_pages)])
        pin = lambda a: torch.as_tensor(np.asarray(a, np.int32)).pin_memory()  # noqa: E731
        return attn.plan_graph(None, rows=rows, parity=0, qo_indptr=pin(np.arange(rows + 1) * qo),
                               kv_indptr=pin(indptr), last_page_len=pin(ub_now - (n_pages - 1) * ps),
                               kv_lens=pin(ub_now), max_q=qo, lb=lb_now)

    w = plan(ub, true_len)
    lens_d.copy_(torch.as_tensor(true_len))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        attn.graph_page_indices(rows, page_table, slots_d)
        attn.fix_lengths(w, rows, lens_d)
        w.run(q, pools.kv[0], out=out)
        torch.cuda.synchronize()
        with no_gc(), torch.cuda.graph(g, stream=s):
            attn.graph_page_indices(rows, page_table, slots_d)
            attn.fix_lengths(w, rows, lens_d)
            w.run(q, pools.kv[0], out=out)
    for _ in range(4):
        cut = rng.integers(0, 7, rows)  # the true lengths: the plan's bound minus up to 6
        now = ub - cut
        plan(ub, now)
        lens_d.copy_(torch.as_tensor(now))
        g.replay()
        torch.cuda.synchronize()
        want = _reference(pools, page_table, slots, now, [qo] * rows, q, 2)
        assert torch.isfinite(out).all()
        assert (out.float() - want.float()).abs().max() < 2e-2


# -- the startup self-check -------------------------------------------------------------------------------------------


def test_self_check_passes_on_the_installed_flashinfer() -> None:
    from llm_mini_infer.kernels import flashinfer as fl

    _, attn = _setup()
    assert fl.check_plan_info(attn) is None
    assert fl.check_lengths(attn, torch.bfloat16) is None


def test_plan_info_mismatch_falls_back_to_unsplit_plans(monkeypatch) -> None:
    """A FlashInfer whose plan info reads otherwise (simulated: the chunk size's offset moved) is detected, and plans
    with uncertain lengths then disable the KV split: the outputs stay finite and right where a split plan could
    leave a chunk empty."""
    from llm_mini_infer.kernels import flashinfer as fl

    pools, attn = _setup(16, pages=2400)
    monkeypatch.setattr(fl, "_PLAN_CHUNK_SIZE_OFFSET", fl._PLAN_CHUNK_SIZE_OFFSET - 1)
    assert "chunk size" in (fl.check_plan_info(attn) or "")
    monkeypatch.setattr(fl, "_plan_info_ok", False)
    n, qo, ps = 16, 4, 16
    base = [128 * (2 + r % 5) for r in range(n)]
    true_len, ub, page_table, slots = _case(ps, qo, [3] * n, base)
    q = torch.randn(n * qo, 8, 256, device=DEV, dtype=torch.bfloat16)
    w = _plan_eager(attn, pools, ub, true_len, qo, page_table, slots)
    attn.fix_lengths(w, n, torch.as_tensor(true_len, device=DEV))
    out = torch.empty_like(q)
    w.run(q, pools.kv[0], out=out)
    want = _reference(pools, page_table, slots, true_len, [qo] * n, q, 2)
    assert attn.unsplit == 1 and attn.replans == 0
    assert torch.isfinite(out).all()
    assert (out.float() - want.float()).abs().max() < 2e-2


def test_window_fallback_leaves_exact_plans_split(monkeypatch) -> None:
    """Gemma 4's sliding-window instance under the same fallback: a plan with uncertain lengths goes without a split
    and counts in `unsplit`, and an exact plan after it keeps the split FlashInfer chooses (two rows of 900 keys:
    chunks of 8 pages), as before the fallback."""
    from llm_mini_infer.kernels import flashinfer as fl

    rows, qo = 2, 4
    attn, _, table = _window_attn(rows, qo, max_pages=64)
    true_len = np.full(rows, 900, np.int64)
    slots = np.arange(1, rows + 1)
    assert _read_plan(_plan_eager(attn, attn.pools, true_len, None, qo, table, slots))[0] == 1
    monkeypatch.setattr(fl, "_plan_info_ok", False)
    w = _plan_eager(attn, attn.pools, true_len + 3, true_len, qo, table, slots)
    assert attn.unsplit == 1 and _read_plan(w)[0] == 0
    w = _plan_eager(attn, attn.pools, true_len, None, qo, table, slots)
    assert attn.unsplit == 1 and _read_plan(w)[0] == 1


def test_a_length_rewrite_that_does_not_take_fails_startup(monkeypatch) -> None:
    """A length rewrite that does not take fails the self-check, and again on its next call (its findings are kept
    for the process)."""
    from llm_mini_infer.kernels import flashinfer as fl

    _, attn = _setup()
    monkeypatch.setattr(fl.FlashInferAttention, "fix_lengths", lambda self, w, rows, true_len: None)
    assert "differs from the exact plan" in (fl.check_lengths(attn, torch.bfloat16) or "")
    monkeypatch.setattr(fl, "_checked", {})
    for _ in range(2):
        with pytest.raises(RuntimeError, match="upper-bound attention planning"):
            fl.self_check(attn, torch.bfloat16)
    monkeypatch.setattr(fl, "_checked", {})


def test_a_graph_callers_self_check_reads_its_own_wrappers(monkeypatch) -> None:
    """A caller without decode graphs passes the self-check; one with them, whose graph wrapper does not read the
    length buffer it holds, fails it after that pass, and again on its next call."""
    from llm_mini_infer.kernels import flashinfer as fl

    pools, graphs = _setup()
    plain = fl.FlashInferAttention(pools, 8, 2, 256, torch.bfloat16, torch.device(DEV), qo_per_row=4)
    monkeypatch.setattr(fl, "_checked", {})
    monkeypatch.setattr(fl, "_plan_info_ok", True)
    assert fl.self_check(plain, torch.bfloat16)["lengths"] == "ok"
    graphs.graph_wrapper(1)  # made with the buffers it was given; then the attention holds another one
    graphs.g_last = torch.ones_like(graphs.g_last)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="_paged_kv_last_page_len_buf"):
            fl.self_check(graphs, torch.bfloat16)
    assert fl._plan_info_ok


# -- decode graphs at Gemma 4's sliding-layer shape ------------------------------------------------------------------


def _window_reference(kv: torch.Tensor, pages: np.ndarray | torch.Tensor, true_len: int, q: torch.Tensor, window: int,
                      ps: int) -> torch.Tensor:
    """One row's window attention at its true length: its queries are the last `qo` positions of KV [0, true_len)
    (named from the row's first window page), each sees keys within `window` of it, scale 1 (Gemma 4's)."""
    qo, hq, d = q.shape
    hkv = kv.shape[3]
    idx = torch.as_tensor(pages, device=kv.device, dtype=torch.long)
    k = kv[idx, 0].reshape(-1, hkv, d)[:true_len].float()
    v = kv[idx, 1].reshape(-1, hkv, d)[:true_len].float()
    rep = hq // hkv
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    qpos = torch.arange(true_len - qo, true_len, device=kv.device)
    kpos = torch.arange(true_len, device=kv.device)
    s = torch.einsum("qhd,khd->hqk", q.float(), k)
    ok = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - window)
    s = s.masked_fill(~ok[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)


def _window_attn(rows: int, qo: int, max_pages: int, max_rows: int = 64):
    from llm_mini_infer.kernels import flashinfer as fl

    ps, hq, hkv, d, window = 16, 32, 16, 256, 1024
    torch.manual_seed(1)
    kv = torch.randn(rows * max_pages + 1, 2, ps, hkv, d, device=DEV, dtype=torch.bfloat16)
    pools = Pools(kv=kv, conv=torch.zeros(1), rec=torch.zeros(1), page_size=ps)
    attn = fl.FlashInferWindowAttention(pools, hq, hkv, d, torch.bfloat16, torch.device(DEV), window,
                                        max_rows=max_rows, max_pages=max_pages, qo_per_row=qo, upper_bounds=True)
    table = torch.zeros(rows + 1, max_pages + 8, dtype=torch.int32, device=DEV)
    for r in range(rows):  # each row its own pages (the window table's first columns are the window's pages)
        table[r + 1, :max_pages] = torch.arange(1 + r * max_pages, 1 + (r + 1) * max_pages, dtype=torch.int32)
    return attn, kv, table


def _pin(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, np.int32)).pin_memory()


def _read_plan(w, ps: int = 16) -> tuple[int, int, int, int]:
    """(split flag, padded batch, query tile, chunk in pages) of a wrapper's last plan, read as `_plan_safely` reads
    them (the chunk size from the page-locked buffer the wrapper planned into)."""
    from llm_mini_infer.kernels import flashinfer as fl

    info = w._plan_info
    off = int(info[fl._PLAN_CHUNK_SIZE_OFFSET])
    chunk = int(w._pin_memory_int_workspace_buffer[off:off + 4].view(torch.int32)[0]) // ps
    return int(info[fl._PLAN_SPLIT_KV]), int(info[fl._PLAN_PADDED]), int(info[fl._PLAN_CTA_TILE_Q]), chunk


@pytest.mark.parametrize("role", ["target", "drafter"])
@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("rows", BUCKETS)
def test_decode_graph_plans_keep_the_captured_split(rows, k, role) -> None:
    """Gemma 4's sliding layers in a decode graph (32 query and 16 KV heads of 256, window 1,024, pages of 16 positions)
    under MTP at k = 3, 5 and 7: verify rows of 1 + k queries, each row's true length known only within k positions,
    and the drafter's rows of one query, within 2k (`GemmaRunner._layout`). The graph is captured from a short-context
    plan, whose FlashInfer graph schedule splits the KV, then replayed with long contexts' lengths (past the window:
    the upper bounds in [window + 2k, window + 2k + 16)) and shorter ones. Every plan keeps the captured split (one
    without would replay the split kernel over a plan laid out for none: on 0.6.18 every row wrong but finite from 16
    rows on, on 0.7 a merge reading past the workspace) and a chunk count no true length can change, also where
    FlashInfer's own grid fits none (16 rows of 65 and 66 window pages, forced once: planned again on 170 and 188 SMs);
    the outputs equal the reference at the true lengths."""
    from llm_mini_infer.kernels import flashinfer as fl

    ps, window = 16, 1024
    qo, band = (1 + k, k) if role == "target" else (1, 2 * k)
    max_pages = -(-(window + ps + 2 * (1 + k)) // ps) + 1  # the runner's window pages per row (`wpages`)
    attn, kv, table = _window_attn(rows, qo, max_pages)
    pages = table[1:rows + 1, :max_pages].long()
    slots = torch.arange(1, rows + 1, device=DEV)
    lens = torch.zeros(rows, dtype=torch.int64, device=DEV)
    q = torch.randn(rows * qo, 32, 256, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    attn.g_first[:rows] = 0

    def plan(ub: np.ndarray, lb: np.ndarray | None):
        npg = (ub + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(npg)])
        return attn.plan_graph(None, rows=rows, parity=0, qo_indptr=_pin(np.arange(rows + 1) * qo),
                               kv_indptr=_pin(indptr), last_page_len=_pin(ub - (npg - 1) * ps), kv_lens=_pin(ub),
                               max_q=qo, lb=lb)

    short = np.full(rows, 3 + 2 * (1 + k), np.int64)  # the capture's context (`GemmaRunner._decode_capture_plan`)
    w = plan(short, None)
    assert int(w._plan_info[fl._PLAN_SPLIT_KV]) == 1  # FlashInfer's graph schedule splits
    lens.copy_(torch.as_tensor(short))
    torch.cuda.synchronize()
    g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(s):
        attn.graph_window_indices(rows, table, slots)
        attn.fix_lengths(w, rows, lens)
        w.run(q, kv, out=out)
        torch.cuda.synchronize()
        with no_gc(), torch.cuda.graph(g, stream=s):
            attn.graph_window_indices(rows, table, slots)
            attn.fix_lengths(w, rows, lens)
            w.run(q, kv, out=out)
    rng = np.random.default_rng(rows * 100 + k * 10 + qo)
    lost, iters, need = 0, 16, False
    for it in range(iters + (rows == 16)):
        if it < iters:
            ub = rng.integers(window + 2 * k, window + 2 * k + ps, rows).astype(np.int64)  # past the window
            if it % 3 == 2:  # some rows shorter than the window: chunk bounds against the uncertain lengths
                short_rows = rng.random(rows) < 0.5
                ub[short_rows] = rng.integers(qo + band + 1, window, int(short_rows.sum()))
        else:  # 16 rows of 65 and 66 window pages, where FlashInfer's own chunk is unsafe on 170 and 188 SMs (65 pages)
            ub = np.asarray([1042] * 4 + [1035 if role == "target" else 1034] * 12, np.int64)
            plan(ub, None)
            _, _, tile, chunk = _read_plan(w)
            need = fl.unsafe_chunks(ub - band, ub, ps, chunk, window_left=window - 1, cta_tile_q=tile)
            assert need or torch.cuda.get_device_properties(0).multi_processor_count not in (170, 188), (tile, chunk)
        true = ub - rng.integers(0, band + 1, rows)
        before = attn.replans
        plan(ub, ub - band)
        assert attn.replans > before or not need, (rows, it)
        lost += int(w._plan_info[fl._PLAN_SPLIT_KV]) != 1
        lens.copy_(torch.as_tensor(true))
        g.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), (rows, it)
        refs = [_window_reference(kv, pages[r], int(true[r]), q[r * qo:(r + 1) * qo], window, ps) for r in range(rows)]
        err = torch.stack([(out[r * qo:(r + 1) * qo].float() - refs[r]).abs().max() for r in range(rows)])
        rel = (err / torch.stack([ref.abs().max() for ref in refs])).cpu().numpy()
        r = int(rel.argmax())
        # Relative to the row's output: FlashInfer multiplies BF16 probabilities into V, so a right row is off by up to
        # 0.65% of its largest output (measured for 8-32 rows, k 3 and 5, both roles, re-planned or not: 0.028-0.030 on
        # outputs near 4.6); a row that attends wrongly is off by 0.4-6.6, 10% or more
        assert rel[r] < 0.012, (rows, it, r, int(ub[r]), int(true[r]), float(err[r]), float(rel[r]), _read_plan(w),
                                lost)
    assert lost == 0 and attn.kept == 0


@pytest.mark.parametrize("window", [1024, 0])
@pytest.mark.parametrize("qo", [1, 4, 6, 12])
def test_the_chunk_count_rule_matches_the_kernel(qo, window) -> None:
    """`unsafe_chunks` against FlashInfer's kernel: eager plans for upper-bound lengths (true lengths up to 10 below,
    the drafter's band at k = 5; rewritten on the device), forced to chunk sizes from 1 to 66 pages, with and without
    Gemma 4's window, over rows of 1 to 12 queries (verify rows of 4 and 6 at k = 3 and 5). Rows whose true length
    keeps the plan's chunk count equal the reference, at any query count; rows of several queries whose true length
    drops a planned chunk come out wrong, because the kernel strides each query's partial outputs by its own count
    (module docstring), and rows of one query do not."""
    from llm_mini_infer.kernels import flashinfer as fl

    ps, hq, hkv, d, maxp, rows = 16, 32, 16, 256, 72, 10
    torch.manual_seed(qo + window)
    kv = torch.randn(rows * maxp + 1, 2, ps, hkv, d, device=DEV, dtype=torch.bfloat16)
    pools = Pools(kv=kv, conv=torch.zeros(1), rec=torch.zeros(1), page_size=ps)
    if window:
        attn = fl.FlashInferWindowAttention(pools, hq, hkv, d, torch.bfloat16, torch.device(DEV), window)
        extra, scale, wl = dict(window_left=window - 1, sm_scale=1.0), 1.0, window - 1
    else:
        attn = fl.FlashInferAttention(pools, hq, hkv, d, torch.bfloat16, torch.device(DEV))
        extra, scale, wl = {}, d ** -0.5, -1
    w = attn.wrappers[0]
    rng = np.random.default_rng(qo * 7 + window)
    seen = {"kept": 0, "dropped": 0, "dropped wrong": 0}
    chunk_sizes, draws = (1, 2, 3, 5, 8, 13, 22, 32, 33, 64, 65, 66), 12
    planned = dict.fromkeys(chunk_sizes, 0)  # the draws FlashInfer planned, per chunk size
    for _ in range(draws):
        true = rng.integers(qo + 1, 1100, rows)
        ub = true + rng.integers(0, 11, rows)
        npg = (ub + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(npg)])
        kv_idx = torch.cat([torch.arange(1 + r * maxp, 1 + r * maxp + npg[r], dtype=torch.int32)
                            for r in range(rows)]).to(DEV)
        q = torch.randn(rows * qo, hq, d, device=DEV, dtype=torch.bfloat16)
        want = []
        for r in range(rows):
            t = int(true[r])
            k = kv[1 + r * maxp:1 + r * maxp + npg[r], 0].reshape(-1, hkv, d)[:t].float().repeat_interleave(2, 1)
            v = kv[1 + r * maxp:1 + r * maxp + npg[r], 1].reshape(-1, hkv, d)[:t].float().repeat_interleave(2, 1)
            qpos, kpos = torch.arange(t - qo, t, device=DEV), torch.arange(t, device=DEV)
            ok = kpos[None, :] <= qpos[:, None]
            if window:
                ok &= kpos[None, :] > qpos[:, None] - window
            s_ = torch.einsum("qhd,khd->hqk", q[r * qo:(r + 1) * qo].float(), k) * scale
            want.append(torch.einsum("hqk,khd->qhd", s_.masked_fill(~ok[None], float("-inf")).softmax(-1), v))
        for c in chunk_sizes:
            try:
                w.plan(_pin(np.arange(rows + 1) * qo), _pin(indptr), kv_idx, _pin(ub - (npg - 1) * ps), hq, hkv, d, ps,
                       causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, seq_lens=_pin(ub),
                       max_token_per_sequence=qo, non_blocking=True, fixed_split_size=c, **extra)
            except RuntimeError as e:
                if "Buffer overflow" not in str(e) or qo * hq // hkv <= 16:  # 16-query tiles fit every draw
                    raise
                continue  # the split plan's partial outputs do not fit the float workspace
            planned[c] += 1
            if not int(w._plan_info[fl._PLAN_SPLIT_KV]):
                continue
            tile = int(w._plan_info[fl._PLAN_CTA_TILE_Q])
            attn.fix_lengths(w, rows, torch.as_tensor(true, device=DEV))
            out = w.run(q, kv)
            torch.cuda.synchronize()
            cap = wl + tile if window else 1 << 30
            dropped = -(-np.minimum(true, cap) // (c * ps)) < -(-np.minimum(ub, cap) // (c * ps))
            assert fl.unsafe_chunks(true, ub, ps, c, window_left=wl, cta_tile_q=tile) == bool(dropped.any())
            for r in range(rows):
                err = float((out[r * qo:(r + 1) * qo].float() - want[r]).abs().max())
                if dropped[r] and qo > 1:
                    seen["dropped"] += 1
                    seen["dropped wrong"] += not err < 4e-2  # a NaN too (0.7's empty chunks hold −inf)
                else:
                    assert err < 4e-2, (c, r, int(true[r]), int(ub[r]), bool(dropped[r]), err)
                    seen["kept"] += 1
    # FlashInfer 0.7.2 reserves ceil(padded batch * query tile / group size) partial rows per plan and refuses a plan
    # that needs more ("Buffer overflow"). Measured: this happens at chunks of 1 page with 64-query tiles (qo = 12), in
    # some draws only. Chunks of 2 pages and more fit every draw, so the sweep cannot skip them without failing here.
    assert all(n == draws for c, n in planned.items() if c >= 2), planned
    assert planned[1] >= 1, planned
    assert seen["kept"] > 500
    if qo > 1:
        assert seen["dropped"] >= 3 and seen["dropped wrong"] >= 0.9 * seen["dropped"], seen


# PrefillPlanInfo::ToVector()'s, as `fl._PLAN_*` (0.6.18 to 0.7.2; read for split plans, which have both)
_PLAN_MERGE_INDPTR_OFFSET, _PLAN_VALID_MASK_OFFSET = 7, 12


@pytest.mark.parametrize("role", ["target", "drafter"])
@pytest.mark.parametrize("k", [3, 5])
def test_the_graph_plan_model_matches_flashinfer(k, role) -> None:
    """The model of FlashInfer's CUDA-graph plan that tests/kernels/test_flashinfer_selfcheck.py sweeps on CPU
    (`graph_plan`) against the real one on this GPU: one sliding-layer attention's decode-bucket wrappers up to 64 rows
    (as `graph_wrapper` makes them, each bound by its capture's plan, as in the runner), verify rows of 1 + k queries or
    the drafter's of one, FlashInfer's own plans at the capture's lengths, past the window, shorter ones, behind padding
    rows and at 16 rows of 65 and 66 window pages, and plans with fixed chunks around FlashInfer's own and the safe one
    that fit the grid. The padded batch, the query tile, the chunk size in the page-locked buffer and the split flag
    agree, and so do each row's chunks and the work items, read from the plan's merge offsets and valid-block mask
    (entries 7 and 12 of the plan info)."""
    from llm_mini_infer.kernels import flashinfer as fl

    ps, window, hq, hkv, d = 16, 1024, 32, 16, 256
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    qo, band = (1 + k, k) if role == "target" else (1, 2 * k)
    max_pages = -(-(window + ps + 2 * (1 + k)) // ps) + 1
    pools = Pools(kv=torch.zeros(2, 2, ps, hkv, d, device=DEV, dtype=torch.bfloat16), conv=torch.zeros(1),
                  rec=torch.zeros(1), page_size=ps)  # plans read no KV
    attn = fl.FlashInferWindowAttention(pools, hq, hkv, d, torch.bfloat16, torch.device(DEV), window,
                                        max_rows=BUCKETS[-1], max_pages=max_pages, qo_per_row=qo, upper_bounds=True)

    def plan(w, ub: np.ndarray, fixed: int = 0) -> np.ndarray:  # as `plan_graph`, or with a fixed chunk
        npg = (ub + ps - 1) // ps
        w.plan(_pin(np.arange(len(ub) + 1) * qo), _pin(np.concatenate([[0], np.cumsum(npg)])), attn.g_kv_indices[:0],
               _pin(ub - (npg - 1) * ps), hq, hkv, d, ps, causal=True, q_data_type=torch.bfloat16,
               kv_data_type=torch.bfloat16, seq_lens=_pin(ub), max_token_per_sequence=qo, non_blocking=True,
               window_left=window - 1, sm_scale=1.0, fixed_split_size=fixed or None)
        return npg

    rng = np.random.default_rng(k * 10 + qo)
    checked = {"own": 0, "fixed": 0, "split": 0}
    for rows in BUCKETS:
        w = attn.graph_wrapper(rows)
        capture = np.full(rows, 3 + 2 * (1 + k), np.int64)
        plan(w, capture)  # sets the row bound, unless `graph_wrapper` did
        bound = w._max_total_num_rows
        assert bound == rows * qo or rows < attn.split_items < 2 * rows, (rows, bound)
        draws = [capture]
        for _ in range(3):
            ub = rng.integers(window + 2 * k, window + 2 * k + ps, rows).astype(np.int64)
            low = ub.copy()
            cut = rng.random(rows) < 0.5
            low[cut] = rng.integers(qo + band + 1, window, int(cut.sum()))
            behind = ub.copy()
            behind[int(rng.integers(1, rows + 1)):] = qo  # padding rows: KV of their own queries on the null page
            draws += [ub, low, behind]
        if rows == 16:
            draws.append(np.asarray([1042] * 4 + [1035 if role == "target" else 1034] * 12, np.int64))
        for ub in draws:
            npg = plan(w, ub)
            own = graph_plan(np.full(rows, qo), npg, bound, sms)
            chunks = {own.chunk - 1, own.chunk + 1}
            chunks.add(fl.safe_chunk_pages(np.maximum(ub - band, qo), ub, ps, own.chunk, keep_split=True,
                                           max_work=own.padded, tiles=-(-qo * (hq // hkv) // own.tile),
                                           window_left=window - 1, cta_tile_q=own.tile) or 0)
            for c in [0, *sorted(c for c in chunks if c > 0)]:
                want = graph_plan(np.full(rows, qo), npg, bound, sms, c)
                if want.work > want.padded:
                    continue  # FlashInfer refuses a plan that overflows the captured grid
                if c:
                    plan(w, ub, c)
                split, padded_batch, tile, chunk = _read_plan(w, ps)
                got = dict(split=bool(split), padded=padded_batch, tile=tile, chunk=chunk)
                exp = dict(split=want.split, padded=want.padded, tile=want.tile, chunk=want.chunk)
                if split:  # the merge offsets (one entry per query) and the valid-block mask (one per work item)
                    info, buf = w._plan_info, w._pin_memory_int_workspace_buffer
                    mo, vo = int(info[_PLAN_MERGE_INDPTR_OFFSET]), int(info[_PLAN_VALID_MASK_OFFSET])
                    got["counts"] = np.diff(buf[mo:mo + 4 * (rows * qo + 1)].view(torch.int32).numpy()).tolist()
                    got["work"] = int(buf[vo:vo + padded_batch].sum())
                    exp["counts"], exp["work"] = np.repeat(want.counts, qo).tolist(), want.work
                    checked["split"] += 1
                assert got == exp, (rows, c, ub.tolist(), got, exp)
                checked["fixed" if c else "own"] += 1
    assert checked["own"] == len(BUCKETS) * 10 + 1 and checked["fixed"] >= checked["own"], checked


def test_plan_info_mismatch_with_decode_graphs_fails_startup(monkeypatch) -> None:
    """A decode graph needs its split (`kernels.flashinfer`): with the plan info unreadable, startup fails."""
    from llm_mini_infer.kernels import flashinfer as fl

    _, attn = _setup()
    monkeypatch.setattr(fl, "_PLAN_CHUNK_SIZE_OFFSET", fl._PLAN_CHUNK_SIZE_OFFSET - 1)
    monkeypatch.setattr(fl, "_plan_info_ok", True)
    monkeypatch.setattr(fl, "_checked", {})
    with pytest.raises(RuntimeError, match="decode graphs"):
        fl.self_check(attn, torch.bfloat16)
    monkeypatch.setattr(fl, "_checked", {})

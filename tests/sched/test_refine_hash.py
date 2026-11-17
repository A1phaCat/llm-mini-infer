"""The refine measurement's helpers (tests/scripts/refine_harness.py) on the tiny block-FP8 Gemma 4 with its assistant
(k = 3), on the CPU's torch references: the scheduler as the harness builds it, the shared prefix's request, then its
reuses while load requests decode (`warm_up`), and the hash of the precise copy the prompts resume from (`copy_digest`).
The hash repeats across fresh engines and moves with any KV value of the copy (with none past it); the refine takes its
whole piece next to the load, which decodes through it."""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

from llm_mini_infer.cache.radix import KEY_DTYPE, precise_ns
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.runner.base import Capabilities
from llm_mini_infer.runner.mtp import MAX_GEMMA_MTP_K
from llm_mini_infer.sched.scheduler import derive_mixed_step_row_tokens

from ..scripts import refine_harness as tf
from ..support.tiny_gemma import gemma_cpu_runner, gemma_pools, make_tiny_assistant, make_tiny_gemma_fp8

K = 3  # the assistant's drafts
SEQS, BUDGET, WPS, LEN = 8, 512, 4, 1024  # max_num_seqs, max_num_batched_tokens, window pages (the grid), max_model_len
PRE = np.asarray([(i * 37 + 11) % 500 for i in range(296)], np.int32)  # the shared prefix
# its FP8 checkpoint (the end of its prefill, on the grid), which its second reuse refines: the precise copy ends there,
# inside a full page and inside one of its sub-pages (of 8 positions), so the pages holding it hold more
R = 292
LOAD, LOAD_TOKENS = 4, 24
# The step costs a GPU's memory profile measures (the CPU measures none): a 512-token prefill step in 4,480 ms, 8.75 ms
# a token, so the derived bound is 3,500 / 8.75 = 400 decode rows x prefill tokens: 100 tokens next to the load's 4 rows
COSTS = dict(decode_step_ms=10.0, decode_step_rows=1, prefill_step_ms=4480.0, prefill_step_tokens=512)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("refinehash")
    target, _ = make_tiny_gemma_fp8(d, seed=3)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def launched(runner) -> list[tuple[int, int]]:
    """(step, decode rows) of every plan the runner launches from now on."""
    seen: list[tuple[int, int]] = []
    orig = runner.launch_forward

    def launch(plan, _internal=False, _capture=None):
        if not _internal:
            seen.append((plan.step_id, int((plan.sched_tok_off < 0).sum())))
        return orig(plan, _internal=_internal, _capture=_capture)

    runner.launch_forward = launch
    return seen


def refine(pair, *, load: int) -> SimpleNamespace:
    """A fresh engine as the harness builds it (`load` given: the bound derived from `COSTS`; the piece's room for the
    assistant's largest k), the shared prefix computed once (the FP8 copy, hashed), then its two reuses while `load`
    requests decode (the second refines it), and the precise copy hashed."""
    target, assistant = pair
    pc = gemma_pools(max_num_seqs=SEQS, budget=BUDGET, page_size=WPS, num_pages=1024, max_model_len=LEN, mtp_k=K)
    runner = gemma_cpu_runner(target, pc, fp8_cold_prefill="reuse", mtp_k=K, assistant=assistant)
    caps = msgspec.structs.replace(runner.capabilities(), **COSTS)
    kw, info = tf.refine_settings(caps, budget=BUDGET, max_num_seqs=SEQS, load=load)
    sched = tf.make_scheduler(caps, pc, max_num_seqs=SEQS, budget=BUDGET, max_model_len=LEN, caching=True,
                              fp8_cold_prefill="reuse", refine=kw, mtp_k_max=MAX_GEMMA_MTP_K)
    core = EngineCore(sched, runner)
    _, rows = tf.watch_steps(runner, sched)
    plans = launched(runner)
    turns = tf.Turns(core)
    core.handle(tf.prefix_request(-1, PRE))
    turns.idle()
    fp8 = tf.copy_digest(sched, runner, PRE, None, len(PRE) - 1)
    reqs = tf.load_requests(load, LOAD_TOKENS, caps.vocab_size)
    tf.warm_up(core, turns, PRE, 2, reqs)
    precise = tf.copy_digest(sched, runner, PRE, precise_ns(None), len(PRE))
    return SimpleNamespace(runner=runner, sched=sched, core=core, turns=turns, info=info, rows=rows, plans=plans,
                           load=[r.rid for r in reqs], fp8=fp8, precise=precise)


def test_the_hash_repeats_across_fresh_engines(pair) -> None:
    """Two fresh engines (runner, scheduler, engine core) refining the shared prefix next to the same load build the
    same precise copy, bit for bit, in the same PRECISE rows, over the same FP8 copy."""
    a, b = (refine(pair, load=LOAD) for _ in range(2))
    assert a.precise[0] is not None and a.precise[1] == R and len(a.precise[2]) == 6
    assert a.fp8[0] is not None and a.fp8[1] == R and a.fp8[0] != a.precise[0]
    assert a.precise == b.precise and a.fp8 == b.fp8 and a.rows == b.rows


def test_a_copy_computed_alone(pair) -> None:
    """With nothing decoding, the refine is one PRECISE row alone in its step. Next to decode rows the CPU's references
    give that row's KV other bits from the third layer on (the GPU's kernels do not: one copy at every load), so only
    the FP8 copy, computed before the refine, is the same next to the load: the control."""
    alone, loaded = refine(pair, load=0), refine(pair, load=LOAD)
    assert [{k: v for k, v in p.items() if k != "step"} for p in alone.rows] == \
        [{"rows": [[0, R]], "plan_rows": 1, "decode_rows": 0, "tokens": R}]
    assert alone.fp8 == loaded.fp8


def test_the_refine_takes_its_whole_piece_next_to_the_load(pair) -> None:
    """Next to the load's 4 decode rows (verify rows of 1 + k tokens), the refine is stepped as on the server: it takes
    its whole piece (min(2,048, 512 - 8 rows x 8) = 448 on the grid: rows of the assistant's largest k, 7, as the
    server's) next to the rows, past the derived mixed-step bound (400 rows x tokens: 100 tokens next to 4 rows)."""
    r = refine(pair, load=LOAD)
    assert r.info["mixed_step_row_tokens"] == 400 and r.info["mixed_step_bound"]
    assert r.sched.piece == 448
    assert [p["rows"] for p in r.rows] == [[[0, R]]] and r.sched.precise_steps == 1
    for p in r.rows:
        assert p["decode_rows"] == LOAD and p["plan_rows"] == 1 + LOAD
        assert p["tokens"] == LOAD * (1 + K) + p["rows"][0][1]


def test_the_load_decodes_through_the_refine(pair) -> None:
    """`warm_up` turns the engine only until each warm request finishes, so the load still decodes when the refine runs
    (turned until idle, the load would finish first): the plans before and after its PRECISE steps carry the load's
    rows, and so do its PRECISE steps; none of the load finished. It is aborted after the warm loop, and the engine is
    idle."""
    r = refine(pair, load=LOAD)
    dec = dict(r.plans)
    assert dec[r.rows[0]["step"] - 1] == dec[r.rows[-1]["step"] + 1] == LOAD
    assert all(dec[p["step"]] == LOAD for p in r.rows)
    assert all(r.turns.out.get(rid) for rid in r.load) and not r.turns.done & set(r.load)
    assert not r.core.busy and r.sched.num_aborts == LOAD and not r.sched.reqs


def test_the_hash_moves_with_one_kv_value_of_the_copy(pair) -> None:
    """One KV value of the precise copy moves the hash, and that layer's digest alone: a full layer's at position 0 (its
    first page) or at r - 1 (the partial page holding r), a sliding layer's at its window's first position (the K
    plane) or at r - 1 (the V plane). One at r, in the sub-page holding r (where the refiner's own continuation
    follows), moves nothing."""
    r = refine(pair, load=LOAD)
    sched, a, cfg = r.sched, r.runner.arena, r.runner.cfg
    m = sched.cache.match(precise_ns(None), np.asarray(PRE, KEY_DTYPE), 1 << 62, len(PRE))
    full, tail = sched.cache.hit_pages(m)
    lo, wpages = sched.windows.get(m.node.ckpt)
    assert m.hit == R and R % a.sub and tail  # r inside a sub-page of the tail page
    base = r.precise

    def bumped(x: torch.Tensor) -> tuple:
        """The precise copy's digest with the one value `x` (a view into the arena) moved by 1."""
        with torch.inference_mode():
            keep = x.clone()
            x += 1.0
        try:
            return tf.copy_digest(sched, r.runner, PRE, precise_ns(None), len(PRE))
        finally:
            with torch.inference_mode():
                x.copy_(keep)

    def sub_page(pos: int) -> int:  # the tail page's sub-page holding `pos` (j * num_pages + page)
        return pos % a.page_size // a.sub * a.num_pages + tail

    cases = [(cfg.full_layers[0], a.fkv[0, full[0], 0, 0, :1]),
             (cfg.full_layers[1], a.fkv[1, sub_page(R - 1), (R - 1) % a.sub, 0, :1]),
             (cfg.sliding_layers[0], a.wkv[0, wpages[0], 0, 0, 0, :1]),
             (cfg.sliding_layers[1], a.wkv[1, wpages[(R - 1) // WPS - lo], 1, (R - 1) % WPS, 0, :1])]
    for layer, x in cases:
        got = bumped(x)
        assert got[0] != base[0] and got[1] == R
        assert [i for i, (u, v) in enumerate(zip(got[2], base[2])) if u != v] == [layer]
    assert bumped(a.fkv[0, sub_page(R), R % a.sub, 0, :1]) == base
    assert tf.copy_digest(sched, r.runner, PRE, precise_ns(None), len(PRE)) == base


def test_the_mixed_step_bound_comes_with_any_refine_option() -> None:
    """`refine_settings`: without a refine option, the derived piece and no bound; with either, even a load of 0, the
    bound derived as the engine derives it; a PRECISE token weighs the measured ratio either way. Without measured step
    costs (the CPU), no bound, for the engine's reason."""
    caps = Capabilities(backend="cuda", device="test", vocab_size=512, max_positions=4096, memory_total_bytes=1,
                        memory_free_bytes=1, kv_bytes_per_token=1, state_slot_bytes=0, precise_rows=True,
                        precise_step_ms=1.8 * 4480.0, **COSTS)
    kw, info = tf.refine_settings(caps, budget=BUDGET, max_num_seqs=SEQS)
    assert kw == dict(mixed_step_row_tokens=0, precise_cost=pytest.approx(1.8), refine_chunk_tokens=0)
    assert not info["refine_options"] and not info["mixed_step_bound"]
    assert info["mixed_step_row_tokens_reason"].startswith("off: no refine option given")
    derived = derive_mixed_step_row_tokens(decode_ms=10.0, decode_rows=1, prefill_ms=4480.0, prefill_tokens=512,
                                           max_num_batched_tokens=BUDGET, max_num_seqs=SEQS)
    assert derived[0] == 400
    for given in (dict(load=0), dict(chunk_tokens=0)):
        kw, info = tf.refine_settings(caps, budget=BUDGET, max_num_seqs=SEQS, **given)
        assert kw["mixed_step_row_tokens"] == 400 and info["refine_options"] and info["mixed_step_bound"]
        assert info["mixed_step_row_tokens_reason"] == derived[1]
    kw, _ = tf.refine_settings(caps, budget=BUDGET, max_num_seqs=SEQS, chunk_tokens=256, load=4)
    assert kw == dict(mixed_step_row_tokens=400, precise_cost=pytest.approx(1.8), refine_chunk_tokens=256)
    bare = msgspec.structs.replace(caps, decode_step_ms=0.0, prefill_step_ms=0.0, precise_step_ms=0.0)
    kw, info = tf.refine_settings(bare, budget=BUDGET, max_num_seqs=SEQS, load=4)
    assert kw["mixed_step_row_tokens"] == 0 and kw["precise_cost"] == 1.0 and not info["mixed_step_bound"]
    assert "measured no step costs" in info["mixed_step_row_tokens_reason"]


def test_the_load_and_the_logits_hooks_rows() -> None:
    """The load's requests are the same at every start, unique, from the vocabulary's middle half, ignore_eos with
    4,096 tokens out, their ids apart from the prompts'; the logits hook keeps a prompt's rows before its sequence's
    last position only, never the shared prefix's (negative ids, which would index the last prompts from the end) or
    the load's."""
    a, b = tf.load_requests(LOAD, LOAD_TOKENS, 512), tf.load_requests(LOAD, LOAD_TOKENS, 512)
    assert [r.prompt.tolist() for r in a] == [r.prompt.tolist() for r in b]
    assert len({tuple(r.prompt.tolist()) for r in a}) == LOAD and all(len(r.prompt) == LOAD_TOKENS for r in a)
    assert all(128 <= t < 384 for r in a for t in r.prompt.tolist())
    assert all(r.max_tokens == 4096 and r.ignore_eos and r.temperature == 0.0 and r.rid >= tf.LOAD_RID for r in a)
    rid = np.asarray([-1, -2, 0, 1, tf.LOAD_RID, 2, 0], np.int64)
    pos = np.asarray([5, 1, 5, 9, 5, 2, 8])
    assert tf.prompt_rows(rid, pos, [10, 10, 3]).tolist() == [2, 6]

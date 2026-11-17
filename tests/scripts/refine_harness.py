"""The engine core turned by hand over a runner, to measure how the precise prefix is refined: the scheduler as the
engine builds it for a runner (`make_scheduler`, with `refine_settings`' piece and mixed-step bound), the shared
prefix's requests (`prefix_request`) and their reuses next to decoding load (`load_requests`, `warm_up`), the steps'
rows (`watch_steps`), and a hash of the cached copy the prompts resume from (`copy_digest`).
tests/sched/test_refine_hash.py runs it on the tiny models on the CPU."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import torch

LOAD_RID = 1 << 30  # the load's request ids: past every prompt's, apart from the prefix's (negative)
LOAD_SEED = 4242  # the RNG of its token ids
LOAD_MAX_TOKENS = 4096


def refine_settings(caps, *, budget: int, max_num_seqs: int, chunk_tokens: int | None = None,
                    load: int | None = None) -> tuple[dict, dict]:
    """How a refine is stepped: the scheduler's settings, and what the info records of them. A refine is cut in fixed
    pieces as on the server, the piece derived from max_num_seqs and the drafter's largest k as the server derives it
    (`make_scheduler`'s `mtp_k_max`) unless `chunk_tokens` sets it. With `chunk_tokens` or `load` given (None: not
    given), the server's plans: the mixed-step bound derived as `engine/process.py` derives it
    (`derive_mixed_step_row_tokens` from the runner's measured step costs; 0 when it measured none, as on the CPU).
    With neither, no bound. A PRECISE token weighs the runner's measured ratio of a PRECISE step to an FP8 one in the
    bound, as on the server."""
    from llm_mini_infer.sched.scheduler import derive_mixed_step_row_tokens

    given = any(v is not None for v in (chunk_tokens, load))
    row_tokens, why = 0, "off: no refine option given (chunk_tokens or load)"
    if given:
        row_tokens, why = derive_mixed_step_row_tokens(
            decode_ms=caps.decode_step_ms, decode_rows=caps.decode_step_rows, prefill_ms=caps.prefill_step_ms,
            prefill_tokens=caps.prefill_step_tokens, max_num_batched_tokens=budget, max_num_seqs=max_num_seqs)
    cost = caps.precise_step_ms / caps.prefill_step_ms if caps.precise_step_ms and caps.prefill_step_ms else 1.0
    kw = dict(mixed_step_row_tokens=row_tokens, precise_cost=cost, refine_chunk_tokens=chunk_tokens or 0)
    info = dict(refine_options=given, refine_chunk_tokens=kw["refine_chunk_tokens"], mixed_step_bound=row_tokens > 0,
                mixed_step_row_tokens=row_tokens, mixed_step_row_tokens_reason=why, precise_cost=round(cost, 4))
    return kw, info


def make_scheduler(caps, pc, *, max_num_seqs: int, budget: int, max_model_len: int, caching: bool, positions=None,
                   fp8_cold_prefill: str = "fp8", refine: dict, mtp_k_max: int = 0):
    """The scheduler over the runner's pools (`pc`) as the engine builds it for this runner (window mode, the encoder,
    MTP, the precise prefix's policy where the runner computes PRECISE rows), with `refine` from `refine_settings` and
    the drafter's largest k (`mtp_k_max`: the derived piece's room, as the engine's `config.mtp_k_max`; 0: k)."""
    from llm_mini_infer.sched.scheduler import Scheduler

    return Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=max_model_len,
                     page_size=pc.page_size, num_pages=pc.num_pages, num_state_slots=pc.num_state_slots,
                     eos_token_ids=(), steps_in_flight=caps.max_steps_in_flight,
                     num_live_slots=min(pc.num_state_slots - 1, max_num_seqs) if not caps.window else None,
                     hybrid=caps.state_slot_bytes > 0, prefix_caching=caching, encoder_rows=caps.encoder_rows,
                     encoder_budget=caps.encoder_budget, positions=positions, window=caps.window,
                     window_page_size=pc.window_page_size, bidirectional_images=caps.bidirectional_images,
                     checkpoint_grid=pc.window_page_size or 64, mtp_k=caps.mtp_depth, draft_kv=caps.draft_kv,
                     precise=fp8_cold_prefill if caps.precise_rows else "fp8", mtp_k_max=mtp_k_max, **refine)


def prompt_rows(rid: np.ndarray, pos: np.ndarray, lengths: list[int]) -> np.ndarray:
    """The logits hook's rows (request ids `rid`, positions `pos`) that predict a token of a prompt's sequence (its
    length in `lengths`): rows of the prompts' requests (ids 0 .. len(lengths) - 1; the shared prefix's, negative ids
    that would index the last prompts from the end, and the load's are none) before their last position."""
    keep = np.flatnonzero((rid >= 0) & (rid < len(lengths)))
    return keep[pos[keep] + 1 < np.asarray(lengths, np.int64)[rid[keep]]]


def watch_steps(runner, sched) -> tuple[dict[int, list[int]], list[dict]]:
    """Wrap the runner's launch_forward: each step's SCHED rows' request ids (what the logits hook maps rows by), and
    every plan with PRECISE rows: its step, those rows' [pos0, n_query], its rows, its decode rows (device input) and
    its query tokens."""
    from llm_mini_infer.sched.plan import PRECISE

    step_rows: dict[int, list[int]] = {}
    precise: list[dict] = []
    orig = runner.launch_forward

    def record(plan, _internal=False, _capture=None):
        if not _internal:
            step_rows[plan.step_id] = [int(sched.fe_rid[s]) for s in plan.sched_slot.tolist()]
            rows = np.flatnonzero(plan.sched_flags & PRECISE).tolist()
            if rows:
                precise.append({"step": int(plan.step_id),
                                "rows": [[int(plan.sched_pos0[i]), int(plan.sched_n_query[i])] for i in rows],
                                "plan_rows": len(plan.sched_slot), "decode_rows": int((plan.sched_tok_off < 0).sum()),
                                "tokens": int(plan.n_query_tokens)})
        return orig(plan, _internal=_internal, _capture=_capture)

    runner.launch_forward = record
    return step_rows, precise


class Turns:
    """The engine's loop, turned by hand: each request's tokens so far (`out`) and the requests that finished
    (`done`)."""

    def __init__(self, core) -> None:
        self.core = core
        self.out: dict[int, list[int]] = {}
        self.done: set[int] = set()

    def until(self, cond: Callable[[], bool]) -> int:
        """Turn until `cond()` holds (checked before every turn); returns the turns."""
        turns = 0
        while not cond():
            if not self.core.busy:
                raise RuntimeError("the engine went idle first")
            rep = self.core.step()
            for fo in (rep.outputs.values() if rep is not None else ()):
                at = 0
                for rid, n, f in zip(fo.rids.tolist(), fo.n_new.tolist(), fo.finish.tolist()):
                    self.out.setdefault(int(rid), []).extend(int(t) for t in fo.tokens[at:at + n])
                    at += n
                    if f:
                        self.done.add(int(rid))
            turns += 1
            if turns > 1_000_000:
                raise RuntimeError("engine not idle")
        return turns

    def finish(self, rid: int) -> int:
        """Turn until request `rid` finished."""
        return self.until(lambda: rid in self.done)

    def idle(self) -> int:
        return self.until(lambda: not self.core.busy)


def prefix_request(rid: int, prompt: np.ndarray):
    """The shared prefix's request: rid -1 computes it, -2, -3, ... reuse it (`warm_up`); one greedy token each."""
    from llm_mini_infer.engine.messages import AddRequest

    return AddRequest(rid=rid, frontend=0, prompt=prompt, max_tokens=1, temperature=0.0, top_p=1.0, top_k=0,
                      min_p=0.0, seed=0, stop_token_ids=[], ignore_eos=True, images=[])


def load_requests(n: int, tokens: int, vocab: int, seed: int = LOAD_SEED) -> list:
    """The refine's load: `n` unique text requests of `tokens` token ids each, drawn from a fixed RNG in the middle
    half of the vocabulary (no special or image token), greedy, ignore_eos, `LOAD_MAX_TOKENS` out: they decode through
    the warm loop and are aborted after it."""
    from llm_mini_infer.engine.messages import AddRequest

    ids = np.random.default_rng(seed).integers(vocab // 4, 3 * vocab // 4, (n, tokens), dtype=np.int64)
    return [AddRequest(rid=LOAD_RID + j, frontend=0, prompt=ids[j].astype(np.int32), max_tokens=LOAD_MAX_TOKENS,
                       temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=j, stop_token_ids=[], ignore_eos=True,
                       images=[]) for j in range(n)]


def warm_up(core, turns: Turns, pre: np.ndarray, warm: int, load: list) -> int:
    """The shared prefix's `warm` reuses next to the load (`load_requests`; may be empty): the load is added and
    turned until each of its requests decodes (has produced a token); then each warm request runs, the engine turned
    only until that request finishes (until idle, the load would finish first and the refine run next to none of it);
    then the load is aborted and the engine turned until idle. Returns the turns."""
    from llm_mini_infer.engine.messages import Abort

    rids = [r.rid for r in load]
    for r in load:
        core.handle(r)
    n = turns.until(lambda: all(turns.out.get(rid) or rid in turns.done for rid in rids))
    for w in range(warm):
        core.handle(prefix_request(-2 - w, pre))
        n += turns.finish(-2 - w)
    if rids:
        core.handle(Abort(frontend=0, rids=rids))
    return n + turns.idle()


def copy_digest(sched, runner, key, ns, limit: int) -> tuple[str | None, int, list[str]]:
    """A cached copy's KV, hashed (Gemma 4's KV arena, window mode): the hit r of `key` in namespace `ns` (at most
    `limit` tokens, every entry visible), then sha256 over its KV at positions below r, layer by layer in the model's
    order: a full layer's sub-pages of the hit's tree pages (the full ones and the partial last one,
    `RadixCache.hit_pages`), a sliding layer's window pages of the hit's checkpoint (`WindowStore.get`, both planes).
    Positions from r on are left out: the pages holding r also hold what the copy's writer computed after it.
    Returns (the sha256, r, each layer's digest's first 16 hex digits), or (None, 0, []) without a hit. The lookup
    touches the path's LRU clocks, as an admission's does (the next admission on the prefix touches them again)."""
    from llm_mini_infer.cache.radix import KEY_DTYPE

    m = sched.cache.match(ns, np.asarray(key, KEY_DTYPE), 1 << 62, limit)
    r = m.hit
    if not r:
        return None, 0, []
    full, tail = sched.cache.hit_pages(m)
    pages = full + ([tail] if tail else [])
    lo, wpages = sched.windows.get(m.node.ckpt)
    a, cfg = runner.arena, runner.cfg
    fidx = {layer: i for i, layer in enumerate(cfg.full_layers)}
    sidx = {layer: i for i, layer in enumerate(cfg.sliding_layers)}
    if a.mem.is_cuda:
        torch.cuda.synchronize(a.mem.device)
    total, layers = hashlib.sha256(), []
    with torch.inference_mode():
        for layer in range(cfg.num_layers):
            h = hashlib.sha256()
            parts = []
            if layer in fidx:  # sub-page j of full page p: j * num_pages + p, `sub` positions from its start
                f = fidx[layer]
                for i, p in enumerate(pages):
                    for j in range(a.groups):
                        n = min(a.sub, r - i * a.page_size - j * a.sub)
                        if n > 0:
                            x = a.fkv[f, j * a.num_pages + p]
                            parts.append(x[:n] if cfg.k_eq_v else x[:, :n])
            else:  # window page lo + i: its K and V planes
                s = sidx[layer]
                for i, p in enumerate(wpages):
                    n = min(a.window_page_size, r - (lo + i) * a.window_page_size)
                    if n > 0:
                        parts.append(a.wkv[s, p, :, :n])
            for x in parts:
                b = x.cpu().contiguous().view(torch.uint8).numpy()
                h.update(b)
                total.update(b)
            layers.append(h.hexdigest()[:16])
    return total.hexdigest(), r, layers

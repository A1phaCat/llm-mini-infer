"""The window cache's and Gemma's image spans' workloads and their drivers (tests/sched/test_window_cache.py,
tests/sched/test_gemma_spans.py, tests/support/fuzzing.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import reserve_positions

from .fuzzing import event, nan_report, nan_write, st
from .harness import Harness, Spec, check_nan_invariants
from .sched_cases import VOCAB, check_plans, nan_events, output_of, stem
from .workloads import GRIDS, text


def harness(specs: list[Spec], *, window: int = 16, page_size: int = 4, full: int = 0, num_pages: int = 0,
            max_tokens: int = 256, max_model_len: int = 512, **kw) -> Harness:
    """Window mode with one arena: window pages of `page_size` positions, full pages of `full` (default 5 x page_size),
    `num_pages` pages of either kind (default: room for four requests of max_model_len)."""
    full = full or 5 * page_size
    if not num_pages:
        num_pages = 4 * arena_bound(window, page_size, full, max_tokens, max_model_len) + 16
    return Harness(specs=specs, vocab=VOCAB, window=window, window_page_size=page_size, page_size=full,
                   num_pages=num_pages, grid=kw.pop("grid", page_size), max_tokens=max_tokens,
                   max_model_len=max_model_len, **kw)


def arena_bound(window: int, page_size: int, full: int, max_tokens: int, max_model_len: int) -> int:
    """The arena pages one request may need at once: its full pages at max_model_len and its peak of window pages
    (the startup bound of `cache.pools.size_arena`)."""
    reserve = reserve_positions(2, 0)
    return pages_for_one_request(max_model_len, full, reserve, True) + \
        window_pages_for_one_request(window, max_tokens, page_size, reserve)


# Full page / window page: Gemma 4 31B's is 20 (320 / 16); 3 keeps the two kinds off each other's grid
FULL_RATIOS = [1, 2, 3, 5, 20]


@st.composite
def window_workloads(draw):
    """`workloads.cache_workloads` in window mode: windows from one key to wider than most prompts, pools
    from the startup bound up, shared prefixes, follow-up turns, hints, aborts, priorities, NaNs."""
    page_size = draw(st.sampled_from([1, 2, 4, 8]))
    window = draw(st.sampled_from([1, 2, 5, 8, 16, 33, 64]))
    max_model_len = 160
    longest = draw(st.sampled_from([100, 100, 150]))
    stems = [stem(i, draw(st.integers(4, longest - 30))) for i in range(draw(st.integers(1, 3)))]
    n = draw(st.integers(1, 10))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["stem", "stem", "turn", "random"]))
        if kind == "turn" and specs:
            prev = draw(st.sampled_from(specs))
            out = output_of(prev)
            prompt = list(prev.prompt) + out[:draw(st.integers(0, len(out)))]
        elif kind in ("stem", "turn"):
            s = draw(st.sampled_from(stems))
            prompt = s[:draw(st.integers(1, len(s)))]
        else:
            prompt = []
        prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=20))
        prompt = [t if t != 999 else 0 for t in prompt][:longest]
        eos = draw(st.one_of(st.none(), st.integers(0, 30)))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 40)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 30)), eos_after=eos,
                          ignore_eos=draw(st.booleans()) if eos is not None else False,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 60))), priority=draw(st.integers(0, 2)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=2)),
                          salt=draw(st.sampled_from([None, None, "s"])),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    max_tokens = draw(st.integers(16, 128))
    full = page_size * draw(st.sampled_from(FULL_RATIOS))
    min_pages = arena_bound(window, page_size, full, max_tokens, max_model_len) + 1
    return dict(specs=specs, page_size=full, window_page_size=page_size, report_unsampled=nan_report(draw),
                grid=page_size * draw(st.sampled_from([1, 1, 2])), max_model_len=max_model_len, window=window,
                max_num_seqs=draw(st.integers(1, 5)), max_tokens=max_tokens,
                num_pages=draw(st.integers(min_pages, min_pages * 4)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 13])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
                prefix_caching=draw(st.sampled_from([True, True, True, False])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])))


def run_window_workload(w: dict) -> Harness:
    h = Harness(vocab=VOCAB, **w)
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    for spec in h.specs.values():
        r = h.results[spec.rid]
        if spec.nonfinite_after is not None and r.finish not in (FinishReason.ABORT,):
            out = h.expected(spec)
            stops = [i for i, t in enumerate(out) if t == 999 and not spec.ignore_eos]
            if spec.nonfinite_after < min(stops[0] + 1 if stops else len(out), spec.max_tokens):
                assert r.finish == FinishReason.ERROR, "the NaN request was not failed"
    if s.prefix_hits:
        event("prefix hit")
    if h.runner.counts["kv_copy"]:
        event("copy-on-write")
    for src in ("decode", "finish", "hint", "branch"):
        if s.ckpts_taken[src]:
            event(f"{src} window")
    if s.cache_waits:
        event("sibling waited")
    if s.cache is not None and s.cache.evicted_ckpts:
        event("window evicted")
    if s.num_preemptions:
        event("preempted")
    if s.num_failed:
        event("failed (NaN)")
    nan_events(h)
    if h.runner.counts["kind_switch"]:
        event("a page reused as the other kind")
    if s.cache is not None and s.cache.evicted_nodes:
        event("full pages evicted")
    if s.cache is not None and s.cache.shared_tails:
        event("junction tails shared")
    return h


@st.composite
def window_pressure(draw):
    """Requests whose prompts share long prefixes and run past the window, in an arena at the startup bound (one
    request's full pages and peak of window pages) plus a few pages: windows of the cache, sibling waits and
    preemptions compete for the same pages. The hazards of window pages: a page released or reused while a query can
    still see it, a window that pins the pages a running request needs, a hit whose window is evicted before it is
    read."""
    page_size = draw(st.sampled_from([1, 2, 4]))
    window = draw(st.sampled_from([2, 4, 8, 12, 24]))
    max_model_len = 160
    base = stem(draw(st.integers(0, 50)), draw(st.integers(30, 120)))
    n = draw(st.integers(2, 4))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs = []
    for rid in range(n):
        share = draw(st.integers(1, len(base)))
        off = draw(st.integers(0, VOCAB - 2))
        tail = [(j * 11 + off) % (VOCAB - 1) for j in range(draw(st.integers(1, 150 - share)))]
        prompt = [t if t != 999 else 0 for t in base[:share] + tail]
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 60)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 10)), ignore_eos=True,
                          priority=draw(st.integers(0, 2)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    max_tokens = draw(st.sampled_from([16, 32, 64, 128]))
    full = page_size * draw(st.sampled_from(FULL_RATIOS))
    bound = arena_bound(window, page_size, full, max_tokens, max_model_len)
    return dict(specs=specs, page_size=full, window_page_size=page_size, grid=page_size,
                max_model_len=max_model_len, max_num_seqs=n, window=window, max_tokens=max_tokens,
                report_unsampled=nan_report(draw),
                num_pages=bound + 1 + draw(st.integers(0, 12)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 8, 16])),
                decode_checkpoint_interval=draw(st.sampled_from([4, 8, 256])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])))


def unique_seeds(draw, n: int) -> list[int]:
    """n distinct seeds from one draw (a list drawn with `unique=True` is often rejected at these sizes)."""
    base = draw(st.integers(1 << 32, 1 << 61))
    return [base + 7919 * i for i in range(n)]


@st.composite
def arena_pressure(draw):
    """Full and window pages from one arena near its startup bound: long prompts (mostly full pages past the window)
    next to short ones (mostly window pages), over shared prefixes the cache pins while running requests hold them,
    follow-up turns that resume from finish windows, hints, decode windows, aborts, priorities and preemption, full
    pages from as large as a window page to 20 times one. The hazards of one arena: a page freed as one kind and written
    as the other while the cache or a running request still reads it, a full page of a cached prefix reused as a window
    page before a hit reads it, evictions that free nothing (a window sharing a running request's pages) ending in a
    wait that never ends, and accounting that counts a page as both kinds."""
    page_size = draw(st.sampled_from([1, 2, 4]))
    full = page_size * draw(st.sampled_from(FULL_RATIOS))
    window = draw(st.sampled_from([2, 4, 8, 16, 24]))
    max_model_len = 160
    stems = [stem(i + 7, draw(st.integers(window, 120))) for i in range(draw(st.integers(1, 3)))]
    n = draw(st.integers(2, 7))
    seeds = unique_seeds(draw, n)
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["long", "long", "short", "turn"]))
        if kind == "turn" and specs:
            prev = draw(st.sampled_from(specs))
            out = output_of(prev)
            prompt = list(prev.prompt) + out[:draw(st.integers(0, len(out)))]
        elif kind == "long":
            s = draw(st.sampled_from(stems))
            prompt = s[:draw(st.integers(max(1, len(s) // 2), len(s)))]
        else:
            prompt = []
        tail = draw(st.integers(1, 30 if kind == "short" else 12))
        off = draw(st.integers(0, VOCAB - 2))
        prompt = [t if t != 999 else 0 for t in prompt + [(j * 17 + off) % (VOCAB - 1) for j in range(tail)]][:150]
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 50)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 20)), ignore_eos=True,
                          abort_at=draw(st.sampled_from([None] * 6 + [5, 15, 30])), priority=draw(st.integers(0, 2)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    max_tokens = draw(st.sampled_from([16, 32, 64]))
    bound = arena_bound(window, page_size, full, max_tokens, max_model_len)
    return dict(specs=specs, page_size=full, window_page_size=page_size, grid=page_size,
                max_model_len=max_model_len, max_num_seqs=draw(st.integers(2, 6)), window=window,
                report_unsampled=nan_report(draw),
                max_tokens=max_tokens, num_pages=bound + 1 + draw(st.integers(0, 2 * bound)),
                policy=draw(st.sampled_from(["fcfs", "priority", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 8])), decode_checkpoint_interval=draw(st.sampled_from([4, 16])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160])))


@st.composite
def arena_hits(draw):
    """Many requests over two or three hot prefixes, arriving over time in an arena that holds a few requests: the
    prefixes' full pages and windows stay pinned while siblings hit them, and the cache's other windows and full pages
    are evicted and their pages reused, either kind, between the plan that matches a hit and the one that reads it."""
    page_size = draw(st.sampled_from([1, 2, 4]))
    full = page_size * draw(st.sampled_from(FULL_RATIOS))
    window = draw(st.sampled_from([3, 8, 16]))
    max_model_len = 160
    stems = [stem(i + 31, draw(st.integers(10, 90))) for i in range(draw(st.integers(2, 3)))]
    n = draw(st.integers(4, 12))
    seeds = unique_seeds(draw, n)
    specs = []
    for rid in range(n):
        s = draw(st.sampled_from(stems))
        cut = draw(st.sampled_from([len(s), len(s), draw(st.integers(1, len(s)))]))
        off = draw(st.integers(0, VOCAB - 2))
        prompt = [t if t != 999 else 0 for t in s[:cut] + [(j * 5 + off) % (VOCAB - 1)
                                                         for j in range(draw(st.integers(1, 16)))]]
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 40)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 40)), ignore_eos=True,
                          priority=draw(st.integers(0, 1)), hints=[len(s)] if draw(st.booleans()) else [],
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    max_tokens = draw(st.sampled_from([16, 32, 64]))
    bound = arena_bound(window, page_size, full, max_tokens, max_model_len)
    return dict(specs=specs, page_size=full, window_page_size=page_size, grid=page_size,
                max_model_len=max_model_len, max_num_seqs=draw(st.integers(2, 5)), window=window,
                report_unsampled=nan_report(draw),
                max_tokens=max_tokens, num_pages=bound + 1 + draw(st.integers(0, bound)),
                policy=draw(st.sampled_from(["fcfs", "priority"])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 256])),
                row_tokens=draw(st.sampled_from([0, 0, 40, 160])))


@st.composite
def gemma_image_workloads(draw):
    """Gemma 4's shape in miniature: window mode (one arena) and bidirectional image spans. Requests with 0-2 images
    (ids shared: prefix and encoder-cache hits on the same image, misses on others at the same text), shared stems,
    hints (some inside spans), chunk budgets that cut around spans and mixed-step budgets smaller than a span, encoder
    budgets and caches near the image sizes, page pressure, priorities and aborts; NaN
    (`tests/sched/test_nan_safety.py`)."""
    page_size = draw(st.sampled_from([1, 2, 4]))
    window = draw(st.sampled_from([2, 5, 8, 16, 33]))
    max_model_len = 192
    stems = [text(i, draw(st.integers(2, 40))) for i in range(draw(st.integers(1, 3)))]
    ids = draw(st.lists(st.integers(1, 6), min_size=1, max_size=4))
    n = draw(st.integers(1, 8))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs = []
    for rid in range(n):
        prompt = list(draw(st.sampled_from(stems)))
        images = []
        for _ in range(draw(st.integers(0, 2))):
            iid = draw(st.sampled_from(ids))
            gh, gw = GRIDS[iid % len(GRIDS)]
            images.append((len(prompt), gh, gw, iid))
            prompt += [0] * (gh * gw // 4)
            prompt += draw(st.lists(st.integers(0, 979), min_size=draw(st.sampled_from([1, 2, 3])), max_size=5))
        prompt += draw(st.lists(st.integers(0, 979), min_size=1, max_size=10))
        prompt = prompt[:150]
        images = [im for im in images if im[0] + im[1] * im[2] // 4 <= len(prompt)]
        if images and images[-1][0] + images[-1][1] * images[-1][2] // 4 == len(prompt):
            prompt.append(5)
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 24)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 20)), images=images,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 40))), priority=draw(st.integers(0, 1)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=2)),
                          eos_after=draw(st.one_of(st.none(), st.integers(0, 20))),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    max_tokens = draw(st.sampled_from([16, 24, 32, 64, 96]))
    full = page_size * draw(st.sampled_from([1, 2, 5]))
    min_pages = arena_bound(window, page_size, full, max_tokens, max_model_len) + 1
    return dict(specs=specs, page_size=full, window_page_size=page_size, window=window,
                report_unsampled=nan_report(draw),
                grid=page_size * draw(st.sampled_from([1, 2, 4])), max_model_len=max_model_len,
                max_num_seqs=draw(st.integers(1, 4)), max_tokens=max_tokens,
                num_pages=draw(st.integers(min_pages, min_pages * 3)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 13])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 20, 40, 160])),
                encoder_budget=draw(st.sampled_from([24, 32, 64])), encoder_rows=draw(st.sampled_from([32, 48, 128])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
                prefix_caching=draw(st.sampled_from([True, True, True, False])))


def run_gemma_image_workload(w: dict) -> Harness:
    h = Harness(bidirectional=True, vocab=1000, **w)
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    nan_events(h)
    if s.prefix_hits:
        event("prefix hit")
    if s.encoder_hits:
        event("encoder cache hit")
    if s.num_preemptions:
        event("preempted")
    if any(len(p["spans"]) > 1 for p in h.plans):
        event("several spans in one step")
    if any(p["spans"] and any(not d for d in p["device"]) and any(p["device"]) for p in h.plans):
        event("a span next to decode rows")
    return h

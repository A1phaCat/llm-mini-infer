"""The scheduler, cache and image suites' workloads (Hypothesis strategies) and their drivers
(tests/sched/test_cache_sched.py, tests/sched/test_scheduler.py, tests/sched/test_image_sched.py,
tests/support/fuzzing.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import reserve_positions

from .fuzzing import event, nan_report, nan_write, st
from .harness import EOS, Harness, Spec, check_nan_invariants, prompt_of
from .sched_cases import VOCAB, check_plans, nan_events, output_of, stem


@st.composite
def cache_workloads(draw):
    page_size = draw(st.sampled_from([1, 2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
    max_model_len = 160
    # Long prompts (up to 150 of 160 positions) let one request and the duplicates its lock pins exceed the pool.
    longest = draw(st.sampled_from([100, 100, 150]))
    stems = [stem(i, draw(st.integers(4, longest - 30))) for i in range(draw(st.integers(1, 3)))]
    n = draw(st.integers(1, 10))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["stem", "stem", "turn", "random"]))
        if kind == "turn" and specs:  # a follow-up turn: an earlier prompt, its output, then new tokens
            prev = draw(st.sampled_from(specs))
            out = output_of(prev)
            keep = draw(st.integers(0, len(out)))
            prompt = list(prompt_of(prev, VOCAB)) + out[:keep]
        elif kind in ("stem", "turn"):
            s = draw(st.sampled_from(stems))
            prompt = s[:draw(st.integers(1, len(s)))]
        else:
            prompt = []
        prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=20))
        prompt = [t if t != EOS else 0 for t in prompt][:longest]
        hints = draw(st.lists(st.integers(1, len(prompt)), max_size=2))
        eos = draw(st.one_of(st.none(), st.integers(0, 30)))
        nan = draw(st.sampled_from([None] * 8 + [0, 3]))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 40)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 30)), eos_after=eos,
                          ignore_eos=draw(st.booleans()) if eos is not None else False,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 60))), priority=draw(st.integers(0, 2)),
                          hints=hints, salt=draw(st.sampled_from([None, None, "s"])), nonfinite_after=nan,
                          **nan_write(draw, len(prompt))))
    k = draw(st.sampled_from([0, 0, 1, 3]))
    min_pages = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True) + 1
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                report_unsampled=nan_report(draw),
                accept_p=draw(st.sampled_from([0.0, 0.5, 0.8, 1.0])),
                max_num_seqs=draw(st.integers(1, 5)), max_tokens=draw(st.integers(16, 128)),
                num_pages=draw(st.integers(min_pages, min_pages * 3)), checkpoint_slots=draw(st.integers(1, 8)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 13])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
                hybrid=draw(st.sampled_from([True, True, False])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])))


def run_cache_workload(w: dict) -> Harness:
    h = Harness(vocab=VOCAB, **w)
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    for spec in h.specs.values():
        r = h.results[spec.rid]
        if spec.nonfinite_after is not None and r.finish not in (FinishReason.ABORT,):
            out = h.expected(spec)  # the grammar's picks for a constrained request
            stops = [i for i, t in enumerate(out) if (t == EOS and not spec.ignore_eos) or t in spec.stop_ids]
            if spec.nonfinite_after < min(stops[0] + 1 if stops else len(out), spec.max_tokens):
                assert r.finish == FinishReason.ERROR, "the NaN request was not failed"
    if s.prefix_hits:
        event("prefix hit")
    if h.runner.counts["kv_copy"]:
        event("copy-on-write")
    if s.ckpts_taken["decode"]:
        event("decode checkpoint")
    if s.ckpts_taken["finish"]:
        event("finish checkpoint")
    if s.ckpts_taken["hint"]:
        event("hint checkpoint")
    if s.cache_waits:
        event("sibling waited")
    if s.cache is not None and (s.cache.evicted_pages or s.cache.evicted_ckpts):
        event("evicted")
    if s.cache is not None and s.cache.shared_tails:
        event("junction tails shared")
    if not s.hybrid:
        event("attention-only")
    if s.num_preemptions:
        event("preempted")
    if s.num_failed:
        event("failed (NaN)")
    nan_events(h)
    if h.runner.k:
        event(f"MTP k={h.runner.k}")
        if h.runner.counts["folds"]:
            event("MTP: FOLD on donation")
        if any(r.finish == FinishReason.STOP for r in h.results.values()):
            event("MTP: stop inside a run")
        if s.num_overruns:
            event("MTP: committed past an unknown stop token")
    return h


@st.composite
def pressure_pairs(draw):
    """Two or three requests whose prompts share long prefixes, in a pool at the startup bound: one may recompute a
    prefix another still holds and then run alone, so the duplicates its lock pins decide whether it fits (the
    hazard `Scheduler._readopt` removes; the workloads above reach it far less often), and so do the junction tails
    and, attention-only, the node its hit ends inside (`RadixCache.share_tails`, `RadixCache.end_at_hit`). Chunking,
    priorities, hints and the cache mode (GDN or attention-only) vary."""
    page_size = draw(st.sampled_from([1, 2, 4]))
    max_model_len = 160
    base = stem(draw(st.integers(0, 50)), draw(st.integers(30, 120)))
    n = draw(st.integers(2, 3))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs = []
    for rid in range(n):
        share = draw(st.integers(1, len(base)))
        off = draw(st.integers(0, VOCAB - 2))
        tail = [(j * 11 + off) % (VOCAB - 1) for j in range(draw(st.integers(1, 150 - share)))]
        prompt = [t if t != EOS else 0 for t in base[:share] + tail]
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 60)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 10)), ignore_eos=True,
                          priority=draw(st.integers(0, 2)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    k = draw(st.sampled_from([0, 0, 1, 3]))
    bound = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True)
    return dict(specs=specs, page_size=page_size, grid=8, max_model_len=max_model_len, max_num_seqs=n, mtp_k=k,
                report_unsampled=nan_report(draw),
                accept_p=draw(st.sampled_from([0.0, 0.8, 1.0])),
                max_tokens=draw(st.sampled_from([16, 32, 64, 128])), num_pages=bound + 1 + draw(st.integers(0, 6)),
                checkpoint_slots=draw(st.integers(1, 4)), policy=draw(st.sampled_from(["fcfs", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 8, 16])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 256])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])),
                hybrid=draw(st.sampled_from([True, True, False])))


@st.composite
def workloads(draw):
    n = draw(st.integers(1, 12))
    page_size = draw(st.sampled_from([1, 2, 4, 16]))
    max_model_len = 96
    # Unique seeds: the fake keys EOS injection by seed.
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs = []
    for rid in range(n):
        prompt = draw(st.integers(1, 60))
        max_tokens = draw(st.integers(1, 30))
        eos = draw(st.one_of(st.none(), st.integers(0, 35)))
        specs.append(Spec(rid=rid, prompt_len=prompt, max_tokens=max_tokens,
                          seed=seeds[rid], arrive=draw(st.integers(0, 25)),
                          eos_after=eos, ignore_eos=draw(st.booleans()) if eos is not None else False,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 60))), priority=draw(st.integers(0, 2)),
                          stop_ids=draw(st.lists(st.integers(0, 998), max_size=2))))
    # The pool always fits one maximal request (prompt + the reserved positions); often not many more, to force
    # preemption. MTP (k drafts) widens the decode rows and the reservation.
    k = draw(st.sampled_from([0, 0, 1, 3]))
    min_pages = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), False) + 1
    return dict(specs=specs, page_size=page_size, max_model_len=max_model_len,
                max_num_seqs=draw(st.integers(1, 6)), max_tokens=draw(st.integers(8 if not k else 4 * (1 + k), 80)),
                num_pages=draw(st.integers(min_pages, min_pages * 3)),
                policy=draw(st.sampled_from(["fcfs", "priority"])), chunk_cap=draw(st.sampled_from([0, 0, 7])),
                mtp_k=k, accept_p=draw(st.sampled_from([0.0, 0.5, 0.8, 1.0])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 24, 96, 400])))


def run_workload(w: dict) -> None:
    h = Harness(**w)
    h.run()
    check_plans(h)
    h.check()
    if h.sched.num_preemptions:
        event("preempted")
    if any(p["free"] and set(p["free"]) & set(p["admit"]) for p in h.plans):
        event("slot freed and readmitted in one plan")


def text(i: int, n: int) -> list[int]:
    return [(j * 31 + i * 17 + 5) % 980 for j in range(n)]


def image_prompt(pre: list[int], img_tokens: int, post: list[int]) -> list[int]:
    return [*pre, *([0] * img_tokens), *post]


GRIDS = [(2, 2), (4, 4), (4, 8), (8, 8), (8, 12)]  # patch grids: 1, 4, 8, 16 and 24 tokens


@st.composite
def image_workloads(draw):
    """Requests with 0-2 images (ids shared across requests: encoder- and prefix-cache hits on the same image, misses
    on others at the same text), shared text stems, chunk budgets small and large, encoder budgets and caches near
    the image sizes (cuts, evictions, re-encodes), MTP, page pressure, priorities and aborts; NaN
    (`tests/sched/test_nan_safety.py`)."""
    page_size = draw(st.sampled_from([2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
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
            gh, gw = GRIDS[iid % len(GRIDS)]  # an image's key comes with its size (both are hashed)
            images.append((len(prompt), gh, gw, iid))
            prompt += [0] * (gh * gw // 4)
            # text after each image, as templates emit (the end token; the next image's start token)
            prompt += draw(st.lists(st.integers(0, 979), min_size=draw(st.sampled_from([1, 2, 3])), max_size=5))
        prompt += draw(st.lists(st.integers(0, 979), min_size=1, max_size=10))
        prompt = prompt[:150]
        images = [im for im in images if im[0] + im[1] * im[2] // 4 <= len(prompt)]
        if images and images[-1][0] + images[-1][1] * images[-1][2] // 4 == len(prompt):
            prompt.append(5)  # a prompt ends with text
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 24)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 20)), images=images,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 40))), priority=draw(st.integers(0, 1)),
                          hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          eos_after=draw(st.one_of(st.none(), st.integers(0, 20))),
                          nonfinite_after=draw(st.sampled_from([None] * 8 + [0, 3])),
                          **nan_write(draw, len(prompt))))
    k = draw(st.sampled_from([0, 0, 1, 3]))
    min_pages = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True) + 1
    budget = draw(st.sampled_from([24, 32, 64]))  # the largest image: 24 tokens (2 blocks of 16 rows)
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                report_unsampled=nan_report(draw),
                accept_p=draw(st.sampled_from([0.0, 0.8])), max_num_seqs=draw(st.integers(1, 4)),
                max_tokens=draw(st.integers(16, 96)), num_pages=draw(st.integers(min_pages, min_pages * 3)),
                checkpoint_slots=draw(st.integers(1, 6)), policy=draw(st.sampled_from(["fcfs", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 13])), row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])),
                encoder_budget=budget, encoder_rows=draw(st.sampled_from([32, 48, 64, 128])))


def run_image_workload(w: dict) -> Harness:
    h = Harness(vocab=VOCAB, **w)
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    nan_events(h)
    if s.encoder_hits:
        event("encoder cache hit")
    if s.encoder.evicted:
        event("encoder cache eviction")
    if s.prefix_hits:
        event("prefix hit")
    if s.num_preemptions:
        event("preempted")
    if any(len(p["encode"]) > 1 for p in h.plans):
        event("several images in one step")
    if any(p["image_tokens"] and any(n < 1 for n in p["n_query"]) for p in h.plans):
        event("odd")
    if h.runner.k:
        event(f"MTP k={h.runner.k}")
    return h

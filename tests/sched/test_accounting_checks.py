"""The accounting checks' default mode against their reference: `Scheduler.check_accounting` and `RadixCache.check`
against the same calls with `full=True`, the implementation the default replaces.

The default counts the running requests' page tables at once, and walks the radix tree only when its `version` moved
since the last walk; what the walk reads outside the nodes (each checkpoint slot's kind, the counts, `tree_refs`, the
refcounts) it checks every time. Two properties make it fail exactly where the reference fails, with the same error:
- **The version covers the walk.** Every field the walk reads (the roots, and each node's place, children, key, pages,
  tail, lock, checkpoint and its demotion) changes only with a `version` bump: checked before every check call, over
  random workloads of the fuzz suites below.
- **Both modes fail alike.** The same workloads, with one fault injected at a random check call, run both modes at
  every call: they must pass or fail together, with the same exception and message. Two kinds of fault:
  - *State outside the nodes*, corrupted just before a check call (`DIRECT`), in half the workloads a call where the
    default skips the walk (there the modes run different code): a refcount, the free list, `tree_refs`, a leaked
    state slot, a node's checkpoint freed in its store, a page table (a page more, or a table that is no list), a page
    id out of the pool, a window page table (the suites with windows), an unverified mark, a counter, an encoder
    reference (the suites with images). The checks catch each at once: both modes must fail there.
  - *A radix method's wrong write*, inside the engine step (`METHOD`): `lock` taking a node past its parent, `insert`
    leaving a node a wrong page, `_split` dropping a tail, `attach` dropping the checkpoint. The method bumps the
    version as it should; its write is what is wrong. The engine may trip over it before a check does.
Without a fault every workload passes. The radix suite drives `RadixCache` alone: there the comparison is of
`RadixCache.check` itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_mini_infer.cache.pools import StatePool
from llm_mini_infer.cache.radix import Node, RadixCache
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.fuzzing import event, fuzz_settings
from ..support.mtp_workloads import mtp_workloads, run_mtp_workload
from ..support.nan_cases import nan_workloads, run_nan_workload
from ..support.precise_cases import precise_workloads, run_precise_workload
from ..support.radix_model import run_scenario, scenarios
from ..support.window_workloads import (
    arena_hits,
    gemma_image_workloads,
    run_gemma_image_workload,
    run_window_workload,
    window_pressure,
)
from ..support.workloads import cache_workloads, image_workloads, pressure_pairs, run_cache_workload, run_image_workload

# suite: (its workloads, its run function, the faults it can have besides `COMMON` and `METHOD`); the fuzz suites of
# tests/scripts/fuzz_campaign.py that cover the checks' branches: GDN and attention-only caches, the arena's windows,
# images (the encoder cache), MTP, the precise prefix, NaN rollback
SUITES: dict[str, tuple[Callable[[], st.SearchStrategy], Callable[[dict], Any], tuple[str, ...]]] = {
    "cache_sched": (cache_workloads, run_cache_workload, ()),
    "cache_pressure": (pressure_pairs, run_cache_workload, ()),
    "window_pressure": (window_pressure, run_window_workload, ("window_table",)),
    "arena_hits": (arena_hits, run_window_workload, ("window_table",)),
    "images": (image_workloads, run_image_workload, ("encoder",)),
    "gemma_images": (gemma_image_workloads, run_gemma_image_workload, ("window_table", "encoder")),
    "mtp": (mtp_workloads, run_mtp_workload, ()),
    "precise": (precise_workloads, run_precise_workload, ()),
    "nan": (nan_workloads, run_nan_workload, ("window_table",)),  # some in window mode
}
COMMON = ("refcount", "free_list", "tree_refs", "state_slot", "ckpt_slot", "page_table", "table_type", "page_id",
          "unverified", "counter")
DIRECT = (*COMMON, "window_table", "encoder")
RADIX_DIRECT = ("refcount", "tree_refs", "ckpt_slot", "counter")  # the faults a `RadixCache` alone can have
METHOD = ("lock", "page", "tail", "checkpoint")


class Fault:
    """One workload's fault: its kind, the check call from which it is injected (`at`), whether a `DIRECT` one waits
    for a call that skips the walk (`on_skip`: where the modes run different code) and its random choices (`rng`);
    then the call before which it was injected (a method's: the first call after its step), what each call found."""

    def __init__(self, kind: str, at: int, on_skip: bool, seed: int):
        self.kind, self.at, self.on_skip = kind, at, on_skip
        self.rng = np.random.default_rng(seed)
        self.fired_on_skip = False
        self.calls = 0
        self.fired_at: int | None = None
        self.failed: tuple[int, str] | None = None  # (call, error) where both modes failed
        self.mismatch: str | None = None
        self.unbumped: int | None = None  # a call before which the tree changed without a version bump
        self.walks_skipped = 0
        self.tree: tuple | None = None
        self.version = -1

    def __repr__(self) -> str:
        return f"Fault({self.kind!r}, at={self.at}, on_skip={self.on_skip})"

    def armed(self) -> bool:
        return self.fired_at is None and self.calls >= self.at


class Failed(Exception):
    """Both modes failed at a check call (or disagreed): the workload stops there."""


def outcome(fn: Callable[[], None]) -> str | None:
    try:
        fn()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def tree_state(cache: RadixCache) -> tuple:
    """Every field `RadixCache._walk` reads, node by node."""
    rows = []
    for ns, root in cache.roots.items():
        stack = [root]
        while stack:
            x = stack.pop()
            rows.append((ns, x.id, x.dead, x.parent.id if x.parent is not None else -1, x.ns, x.start, x.end,
                         len(x.key), int(x.key[0]) if len(x.key) else -1, id(x.spans), x.lock, tuple(x.pages), x.tail,
                         x.ckpt, x.demoted, tuple((k, c.id) for k, c in x.children.items())))
            stack.extend(x.children.values())
    return tuple(rows)


def nodes(cache: RadixCache) -> list[Node]:
    out = []
    for root in cache.roots.values():
        stack = list(root.children.values())
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(x.children.values())
    return out


def swap_ckpt_slot(cache: RadixCache, rng: np.random.Generator) -> bool:
    """Free a node's checkpoint in its store and take another in its place (every count and reference stays): the
    tree holds a slot, or a window handle, that is no checkpoint any more."""
    states = cache.states
    held = [x for x in nodes(cache) if x.ckpt]
    if not held or (isinstance(states, StatePool) and not states.num_free_ckpt):
        return False
    h = held[int(rng.integers(len(held)))].ckpt
    if isinstance(states, StatePool):
        states.alloc_ckpt()
    else:
        states.make(*states.get(h))
    states.release(h)
    return True


def inject(f: Fault, s: Scheduler) -> bool:
    """Apply a `DIRECT` fault to `s`; False when it has nothing to apply to yet."""
    rng, kv, cache, n = f.rng, s.kv, s.cache, s.kv.num_pages
    running = [r for r in s.slot_req if r is not None]
    match f.kind:
        case "refcount":
            p = int(rng.integers(1, n))
            kv.refcount[p] += 1 if not kv.refcount[p] or rng.random() < 0.5 else -1
        case "free_list":
            if not kv._free:
                return False
            kv._free.pop(int(rng.integers(len(kv._free))))
        case "tree_refs":
            if cache is None:
                return False
            cache.tree_refs[int(rng.integers(1, n))] += 1
        case "state_slot":
            if s.states.alloc_live() is None and s.states.alloc_ckpt() is None:
                return False
        case "ckpt_slot":
            return cache is not None and swap_ckpt_slot(cache, rng)
        case "page_table":
            if not running:
                return False
            running[int(rng.integers(len(running)))].pages.append(int(rng.integers(1, n)))
        case "table_type":  # one `np.asarray` refuses
            if not running:
                return False
            r = running[int(rng.integers(len(running)))]
            r.pages = set(r.pages) if rng.random() < 0.5 else None
        case "page_id":
            reqs = [r for r in running if r.pages]
            if not reqs:
                return False
            r = reqs[int(rng.integers(len(reqs)))]
            i = int(rng.integers(len(r.pages)))
            # -1: `np.add.at` counts it as page n - 1 (the reference catches it unless that page was replaced)
            r.pages[i] = int(rng.choice([n, -n - 1, 1 << 40] + ([-1] if r.pages[i] != n - 1 else [])))
        case "window_table":
            if not s.window or not running:
                return False
            r = running[int(rng.integers(len(running)))]
            if rng.random() < 0.5:
                r.wpages.append(0)
            else:
                s.num_wpages[r.slot] += 1
        case "unverified":
            used = np.flatnonzero(kv.refcount[1:] > 0) + 1
            if cache is None or not len(used):
                return False
            cache.unverified[int(rng.choice(used))] = 1 << 40
        case "counter":
            counts = ["num_nodes", "num_ckpts", "num_demoted"]
            obj, names = (cache, counts) if cache is not None and rng.random() < 0.5 else (s, ["_n_unv", "_n_unrep"])
            name = str(rng.choice(names))
            setattr(obj, name, getattr(obj, name) + 1)
        case "encoder":
            if s.encoder is None or not s.encoder.entries:
                return False
            entries = list(s.encoder.entries.values())
            entries[int(rng.integers(len(entries)))].refs += 1
    return True


def inject_radix(f: Fault, cache: RadixCache) -> bool:
    """Apply a `RADIX_DIRECT` fault to a `RadixCache` alone (the radix suite)."""
    rng, kv = f.rng, cache.kv
    match f.kind:
        case "refcount":  # fewer references than the tree holds
            pages = np.flatnonzero(cache.tree_refs)
            if not len(pages):
                return False
            p = int(rng.choice(pages))
            kv.refcount[p] = cache.tree_refs[p] - 1
        case "tree_refs":
            cache.tree_refs[int(rng.integers(1, kv.num_pages))] += 1
        case "ckpt_slot":
            return swap_ckpt_slot(cache, rng)
        case "counter":
            name = str(rng.choice(["num_nodes", "num_ckpts", "num_demoted"]))
            setattr(cache, name, getattr(cache, name) + 1)
    return True


def compared(f: Fault, cache_of: Callable[[Any], RadixCache | None], apply: Callable[[Fault, Any], bool],
             check: Callable[..., None]) -> Callable[..., None]:
    """`check` (`Scheduler.check_accounting` or `RadixCache.check`) as the fuzz runs call it: the version contract
    first, then a `DIRECT` fault when armed, then both modes on the same state."""

    def run(self: Any, full: bool = False) -> None:
        n = f.calls
        cache = cache_of(self)
        skip = cache is None or cache.version == f.version  # the default mode skips the walk (with a cache)
        if cache is not None and f.fired_at is None:
            state = tree_state(cache)
            if f.tree is not None and state != f.tree and skip:
                f.unbumped = n
                raise Failed("the tree changed without a version bump")
            f.walks_skipped += skip
            f.tree, f.version = state, cache.version
        if f.kind in DIRECT and f.armed() and (skip or not f.on_skip) and apply(f, self):
            f.fired_at, f.fired_on_skip = n, skip
        f.calls += 1
        fast, ref = outcome(lambda: check(self)), outcome(lambda: check(self, full=True))
        if fast != ref:
            f.mismatch = f"check call {n}: the default mode {fast!r}, the reference {ref!r}"
            raise Failed(f.mismatch)
        if ref is not None:
            f.failed = (n, ref)
            raise Failed(ref)

    return run


@contextmanager
def faulty(f: Fault, radix_alone: bool = False) -> Iterator[None]:
    """The checks compared (`compared`) and `METHOD` faults armed, for one workload."""
    saved: dict[tuple[type, str], Any] = {}

    def put(cls: type, name: str, fn: Any) -> None:
        saved[(cls, name)] = cls.__dict__[name]
        setattr(cls, name, fn)

    if radix_alone:
        put(RadixCache, "check", compared(f, lambda c: c, inject_radix, RadixCache.check))
    else:
        put(Scheduler, "check_accounting", compared(f, lambda s: s.cache, inject, Scheduler.check_accounting))
    lock, insert, split, attach = RadixCache.lock, RadixCache.insert, RadixCache._split, RadixCache.attach

    def fire() -> bool:
        if f.kind not in METHOD or not f.armed():
            return False
        f.fired_at = f.calls
        return True

    def lock_wrong(self: RadixCache, node: Node | None) -> None:
        lock(self, node)
        if f.kind == "lock" and node is not None and node.parent is not None and fire():
            node.lock = node.parent.lock + 1

    def insert_wrong(self: RadixCache, *a: Any, **kw: Any) -> Node:
        node = insert(self, *a, **kw)
        x = node
        while f.kind == "page" and x.parent is not None and not (x.pages or x.tail):
            x = x.parent
        if f.kind == "page" and x.parent is not None and fire():
            q = int(f.rng.integers(1, self.kv.num_pages))
            if x.pages:
                j = int(f.rng.integers(len(x.pages)))
                x.pages[j] = q if q != x.pages[j] else q % (self.kv.num_pages - 1) + 1
            else:
                x.tail = q if q != x.tail else q % (self.kv.num_pages - 1) + 1
        return node

    def split_wrong(self: RadixCache, n: Node, m: int) -> Node:
        p = split(self, n, m)
        if f.kind == "tail" and p.tail and fire():
            p.tail = 0
        return p

    def attach_wrong(self: RadixCache, node: Node, *a: Any, **kw: Any) -> None:
        attach(self, node, *a, **kw)
        if f.kind == "checkpoint" and fire():
            node.ckpt = 0

    for name, fn in (("lock", lock_wrong), ("insert", insert_wrong), ("_split", split_wrong),
                     ("attach", attach_wrong)):
        put(RadixCache, name, fn)
    try:
        yield
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)


def faults(kinds: tuple[str, ...]) -> st.SearchStrategy:
    """A workload's `Fault`: none, or one of `kinds` from a check call in [0, 16) or in [0, 64) (a workload makes a few
    calls or a few dozen). Drawn before the workload: what Hypothesis draws after a large example leans to zeros, here
    the same fault."""
    return st.builds(Fault, st.sampled_from(("none", *kinds)), st.integers(0, 15) | st.integers(0, 63), st.booleans(),
                     st.integers(0, (1 << 32) - 1))


def run_faulty(run: Callable[[], Any], f: Fault, radix_alone: bool = False) -> None:
    """Run a workload with its fault (module docstring), and check what both modes found."""
    error = None
    with faulty(f, radix_alone):
        try:
            run()
        except Failed:
            pass
        except Exception as exc:  # the engine, or the harness, tripped over a fault before a check did
            error = f"{type(exc).__name__}: {exc}"
    assert f.unbumped is None, f"check call {f.unbumped}: the radix tree changed without a version bump"
    assert f.mismatch is None, f.mismatch
    if f.fired_at is None:
        assert f.failed is None and error is None, f"no fault injected, yet {f.failed or error}"
        event("no fault" if f.kind == "none" else "a fault due after the last check call" if f.calls <= f.at else
              "a fault with nothing to apply to")
    elif f.kind in DIRECT:
        assert f.failed is not None and f.failed[0] == f.fired_at, \
            f"{f.kind} injected before check call {f.fired_at}: not caught there ({f.failed or error})"
        event(f"{f.kind}: caught at once{', the walk skipped' if f.fired_on_skip else ''}")
    elif f.failed is not None:
        event(f"{f.kind}: caught {'at once' if f.failed[0] == f.fired_at else 'later'}")
    else:
        event(f"{f.kind}: {'the engine tripped first' if error else 'not caught'}")
    if f.walks_skipped:
        event("a walk skipped")


@pytest.mark.parametrize("suite", sorted(SUITES))
@fuzz_settings(80)
@given(data=st.data())
def test_default_checks_fail_as_the_reference(suite: str, data: st.DataObject) -> None:
    workloads, run, kinds = SUITES[suite]
    f = data.draw(faults(COMMON + kinds + METHOD), label="fault")
    w = data.draw(workloads(), label="workload")
    run_faulty(lambda: run(w), f)


@fuzz_settings(200)
@given(fault=faults(RADIX_DIRECT + METHOD), s=scenarios())
def test_default_radix_check_fails_as_the_reference(fault: Fault, s: dict) -> None:
    run_faulty(lambda: run_scenario(s), fault, radix_alone=True)

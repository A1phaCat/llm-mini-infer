"""The radix prefix cache: one tree of token IDs over KV pages and state checkpoints (architecture.md, "The cache").

**Keys.** A request's key is its token IDs as int64, under a namespace: the `cache_salt` (None when absent). Each
namespace has its own root, so salted requests never share anything with other namespaces. Key values from 2^62
(`SPAN_BASE`) are image spans' keys, while the model sees the placeholder IDs.

**Image spans** (`Spans`; architecture.md, "The radix tree"). Every position of an image's span carries the same key
value, 62 bits of the image's content hash (`vision.family.span_key`), and a node keeps its writer's spans with the full
256-bit digest of each. A child that starts in a span is filed under the value with the span's digest, and a match or
an insert compares the digest of every span that starts inside a node it walks with its own: a different image whose 62
bits collide diverges before its span, as a different token does.

**Nodes.** A node covers the key positions `[start, end)` of every path through it. It holds:
- `pages`: the KV pages that *end* inside it, i.e. page indices `[start // ps, end // ps)`. A page's content depends
  only on the tokens before its end, so each full page belongs to exactly one node of a path;
- `tail`: when `end` is not page-aligned, the page of index `end // ps`, valid up to `end` only. A child that starts
  mid-page holds its own copy of that page index (the request that wrote it computed or copied those positions), so
  a page can be referenced twice by one path: as a parent's tail and as a child's full page;
- `ckpt`: at most one checkpoint, valid exactly at `end`.

The tree holds one page reference per `pages`/`tail` entry (`PagePool` refcounts are the single source of truth) and
owns its checkpoint slots. `tree_refs` counts the tree's share: a refcount equal to it marks a page only the tree
references, which `_adopt` and `share_tails` may swap.

**Window mode** (sliding-window models). The checkpoint store is a `pools.WindowStore`: a node's `ckpt` is a window, the
refcounted pages of the KV arena holding the sliding layers' KV of the positions [end − W + 1, end). A window is made
from a request's own pages, never from a slot budget. Everything else below holds unchanged.

**Match rule** (`match`). Attention-only models hit at the page-aligned length of the visible KV match. Hybrid and
sliding-window models hit at the deepest node on the path that has both KV and a checkpoint. A hit is at most `limit`
tokens (the caller passes its tokens − 1, so the last one is computed and sampled from), and never at a position
`forbid` names (inside a bidirectional image span: the hit backs off to the deepest checkpoint below it).

**Lost tokens** (`Match.lost`, `lost_tokens`). What the tree holds for a key within the limit (`Match.held`) past the
hit, by cause: `writing`, the KV or checkpoint that the plan under construction writes; `checkpoint`, a hybrid or
sliding-window model's KV past the deepest checkpoint the hit may use; `page`, an attention-only model's partial page
past its page-aligned hit.

**Visibility.** Every entry is stamped with the step that writes it (`ready`, `ckpt_ready`): a prefill chunk's when its
step is planned, a decode checkpoint's and a leaving request's later, with their earlier step. A plan with step id S
uses only entries stamped below S; entries stamped S are reported as `pending`, which lets a sibling wait one step.

**Locks.** A running request locks the path to its deepest node (`lock`, counted on the node and every ancestor); an
attention-only hit, which can end inside a node, splits it there first (`end_at_hit`). Locked nodes keep their KV;
their checkpoints are not pinned (a running request never reads a checkpoint after its STATE_COPY).

**Eviction** (architecture.md, "Eviction"). Pages: LRU over unlocked leaves; a leaf takes its checkpoint with it, and a
parent left without children becomes a candidate. Checkpoints: the demoted ones first, then the rest, least recently
used first in each tier, keeping their KV, except the ones the plan under construction writes (stamped S) or reads
(`pin_checkpoint`: a STATE_COPY's source, or in window mode an admission's hit window; released by
`unpin_checkpoints`), and on request the locked nodes' (`spare_locked`). Both use lazy heaps. Entries whose writing step
is still in flight are evicted like any other: a page or slot reallocated by plan N+1 is written by step N+1, after
step N on the same stream, and rollback skips dead nodes.

**Demoted checkpoints** (`demote`): ones their conversation has moved past, which the scheduler names when a request
leaves a deeper checkpoint on its path (`sched.scheduler`, `_supersede`), unless the tree forks below one before any
deeper checkpoint (`forks`). A demoted checkpoint is not dropped: it is evicted before every live one, and a hit makes
it live again (`match`). The heap's key is (tier, recency): a demotion lowers a node's key and pushes its new entry at
once, a use raises it and its entry is pushed again when popped, so every checkpoint has an entry at or below its key
(`check`).

**Pending writes and rollback.** Every insertion and checkpoint is recorded under its writing step and request (`aid`).
When a step's results arrive (`complete`), the writes of requests whose row reported NONFINITE are removed: KV nodes
with their whole subtree, checkpoints alone. Callers fail the requests whose hit used a removed write
(`pending_writers`). An insert never adopts a request's copy of the page holding its hit into a node that starts
before the hit (`_adopt`), so the node a hit ends in, which other requests branch from, never carries the step of a
request that only resumed from it.

**Unverified pages** (`unverified`, kept by `sched.scheduler`): a page that a request's prefill wrote before a SAMPLE
row of it could show it finite holds that admission's `aid` until then. An insert never swaps such a page in for one
that holds none, except under page pressure (`readopt`'s `force`). A request's pages lie on its key's path only, so
`remove_unverified` finds every node holding one there and removes it as rollback does.

**Junction tails** (`share_tails`). A node that ends inside a page holds that page as its tail, and each of its children
a page of that index too. A tail that no child shares is one page more on every path through the junction. An insert
swaps the request's page in for it (`_adopt`), but never its copy of the page holding its hit into a node that starts
before the hit, nor an unverified page for a verified one; so the junctions above a hit inside the page it copies keep
their tails, beyond the startup bound (`pools.pages_for_one_request`), which can hold a lone request past the pool.
Under page pressure
`share_tails` gives such a locked junction one of its children's pages instead (the same KV below the junction),
written by no step in flight, with no stamp or record.

**The precise prefix** (architecture.md, "The precise prefix"): what PRECISE rows computed from a precise start lives in
a namespace of its own, `precise_ns(ns)` beside the request's `ns`. Refining a cached prefix is an insertion there, and
every rule above holds for it unchanged; the two namespaces share the pools and the eviction heaps. An FP8 continuation
of a precise hit goes into the FP8 namespace, and its insert adopts the precise copy's pages wherever the tree alone
holds the FP8 copy's.
"""

from __future__ import annotations

import heapq
import itertools
import logging
from bisect import bisect_right
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import numpy as np

from ..engine.messages import CHECKPOINT_SOURCES
from .pools import PagePool, StatePool, WindowStore

SOURCE_INDEX = {s: i for i, s in enumerate(CHECKPOINT_SOURCES)}  # `RadixCache.attach`'s source -> Node.ckpt_src

log = logging.getLogger(__name__)

KEY_DTYPE = np.int64
SPAN_BASE = 1 << 62  # key values from here on are image spans' (`vision.family.KEY_BASE`), not tokens
KV, CKPT = 1, 2
PRECISE_NS = "bf16"


def precise_ns(ns: Hashable) -> tuple[str, Hashable]:
    """The namespace of the precise prefix (module docstring) beside the request namespace `ns` (its cache_salt)."""
    return PRECISE_NS, ns


_KEPT_ROOTS = (None, precise_ns(None))  # the namespaces without a cache salt: their roots are never dropped


def is_precise(ns: Hashable) -> bool:
    return isinstance(ns, tuple) and len(ns) == 2 and ns[0] == PRECISE_NS


class Spans:
    """A key's image spans (module docstring, Image spans): where each starts and ends, and its image's full content
    digest, in key order. Never changed; the nodes its request writes share it."""

    __slots__ = ("digests", "ends", "starts")

    def __init__(self, starts: list[int], ends: list[int], digests: list[bytes]):
        self.starts, self.ends, self.digests = starts, ends, digests

    def digest_at(self, pos: int) -> bytes | None:
        """The digest of the span holding key position `pos`; None outside every span."""
        i = bisect_right(self.starts, pos) - 1
        return self.digests[i] if i >= 0 and pos < self.ends[i] else None


def child_key(v: int, spans: Spans | None, pos: int) -> Hashable:
    """Where a child starting at key position `pos` with the key value `v` is filed in its parent's `children`: the
    value, and in an image span the span's digest with it (module docstring, Image spans)."""
    return v if v < SPAN_BASE else (v, spans.digest_at(pos) if spans is not None else None)


def verified(child: Node, pos: int, n: int, spans: Spans | None) -> int:
    """`n`, the length of a key's match with `child` from position `pos`, cut before the first image span starting
    inside it whose digest is not the one `child` holds there (a span starting at `pos` was checked by its filing)."""
    if spans is None:
        return n
    starts, mine = spans.starts, child.spans
    i = bisect_right(starts, pos)
    while i < len(starts) and starts[i] < pos + n:
        if mine is None or mine.digest_at(starts[i]) != spans.digests[i]:
            return starts[i] - pos
        i += 1
    return n


def _empty_root(root: Node) -> bool:
    """An idle root: no child, and no lock but its own (a root's lock is 1 plus its subtree's: `RadixCache._root`)."""
    return root.lock == 1 and not root.children


class Node:
    __slots__ = ("children", "ckpt", "ckpt_ready", "ckpt_src", "ckpt_used", "dead", "demoted", "end", "hits", "id",
                 "key", "last_used", "lock", "ns", "on_precise", "pages", "parent", "ready", "served", "spans", "start",
                 "tail", "writer")

    def __init__(self, nid: int, ns: Hashable, parent: Node | None, start: int, end: int, key: np.ndarray):
        self.id = nid
        self.ns = ns
        self.parent = parent
        self.start = start
        self.end = end
        self.key = key
        self.spans: Spans | None = None  # its writer's image spans (module docstring, Image spans)
        self.children: dict[Hashable, Node] = {}
        self.pages: list[int] = []
        self.tail = 0
        self.ckpt = 0
        self.ckpt_ready = -1
        self.ckpt_used = 0
        self.hits = 0  # resumes from its checkpoint since it was attached (the precise prefix's refine rule)
        self.ckpt_src = 0  # its checkpoint's source (a CHECKPOINT_SOURCES index)
        self.served = 0  # resumes from its checkpoint since it was attached, in either namespace (the counters)
        self.on_precise = False  # its checkpoint's writer stood on a precise copy (resumed from one, or refined)
        self.demoted = False  # its checkpoint is evicted before every live one (module docstring, Demoted checkpoints)
        self.ready = -1  # step that writes this node's KV
        self.writer = 0  # admission ID of the writing request
        self.lock = 0
        self.last_used = 0
        self.dead = False

    def __repr__(self) -> str:
        return (f"Node({self.id}, [{self.start}, {self.end}), pages {self.pages}, tail {self.tail}, ckpt {self.ckpt}, "
                f"lock {self.lock}, ready {self.ready})")

    @property
    def evictable(self) -> bool:
        """A leaf that eviction may take: unlocked, live, childless and not a root (`RadixCache.evict_pages`)."""
        return not self.lock and not self.dead and not self.children and self.parent is not None


@dataclass
class Match:
    kv_len: int = 0  # tokens of the key found in the tree, whatever their visibility
    hit: int = 0  # usable hit length (a checkpoint's offset, or page-aligned KV for attention-only models)
    node: Node | None = None  # hybrid: the node ending at `hit`; attention-only: the node holding position hit - 1
    pending: int = 0  # a deeper hit written by the plan being built: usable from the next plan
    path: list[Node] = field(default_factory=list)  # nodes walked, root excluded; the last may match partially
    held: int = 0  # kv_len within the limit: what the tree holds for the key
    # [hit, held) by cause (module docstring, Lost tokens): (cause, end) in order, each from the previous one's end
    lost: tuple[tuple[str, int], ...] = ()


def lost_tokens(m: Match, lo: int, hi: int, out: dict[str, int]) -> None:
    """Add to `out` the positions of [lo, hi) that `m` holds past its hit, by cause (`Match.lost`)."""
    start = m.hit
    for cause, end in m.lost:
        n = min(end, hi) - max(start, lo)
        if n > 0:
            out[cause] += n
        start = end


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    # A whole node matched (most of a walk) takes no ufunc; on long keys the two copies cost more than the ufunc
    if n <= 2048 and a.dtype == b.dtype and a[:n].tobytes() == b[:n].tobytes():
        return n
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


class RadixCache:
    def __init__(self, page_size: int, kv: PagePool, states: StatePool | WindowStore | None, *, hybrid: bool):
        self.ps = page_size
        self.kv = kv
        self.states = states
        self.hybrid = hybrid
        self.roots: dict[Hashable, Node] = {}
        self.tree_refs = np.zeros(kv.num_pages, np.int32)
        self.done_step = 0  # results of every step up to this one have been processed
        self._ids = itertools.count(1)
        self._clock = itertools.count(1)
        self._leaf_heap: list[tuple[int, int, Node]] = []
        self._ckpt_heap: list[tuple[int, int, int, Node]] = []  # (tier: 0 demoted, 1 live; recency; id; node)
        self._pinned: set[int] = set()  # checkpoints the plan under construction reads (`pin_checkpoint`)
        self._writes: dict[int, list[tuple[int, int, Node, int]]] = {}
        self.unverified = np.zeros(kv.num_pages, np.int64)  # per page: the aid whose prefill wrote it unverified, or 0
        self.num_nodes = 0
        self.num_ckpts = 0
        # Bumped by every change to what `check` walks: the roots, and each node's place, key, pages, lock and
        # checkpoint and its demotion. Only the methods below write these (the scheduler reads nodes, and counts `hits`)
        self.version = 0
        self._checked: tuple[int, np.ndarray, dict[int, Node], int, int] | None = None  # `check`'s last walk
        # Bumped whenever a node gives a page up (a swap: `_adopt`, `share_tails`) or a rollback removes nodes, locked
        # or not (`remove_subtree`). Nothing else takes a page off a locked path (eviction takes unlocked nodes, and a
        # split keeps each page where it was on the path), so a duplicate a request was found to hold stays one until
        # this changes (the scheduler's `Req.dup_swaps`; tests/sched/test_cache_sched.py checks every cache call for it)
        self.swaps = 0
        self.evicted_pages = 0  # pages freed by evicting tree nodes
        self.evicted_nodes = 0
        self.evicted_ckpts = 0  # checkpoints evicted (alone or with their node)
        self.num_demoted = 0  # checkpoints demoted now (module docstring, Demoted checkpoints)
        self.demotions = 0  # ... and ever
        # ... by source and by whether they served a resume ([source][0]: none, [1]: some); window mode: the window
        # pages those evictions freed (nodes, a NaN rollback or a running request's replaced decode checkpoint: none)
        self.evicted_by = [[0, 0] for _ in CHECKPOINT_SOURCES]
        self.evicted_window_pages = 0
        self.rolled_back = 0
        self.kept_dups = 0  # pages `_adopt` could not swap: another request still referenced the tree's, or it was
        # the request's own unverified page and the tree's was not (module docstring)
        self.shared_tails = 0  # junction tails freed by sharing a child's page (`share_tails`)
        self._clearing = False  # `clear` runs: its evictions log nothing (`_evicting`)

    # -- structure ---------------------------------------------------------------------------------------------

    def _root(self, ns: Hashable) -> Node:
        root = self.roots.get(ns)
        if root is None:
            root = self.roots[ns] = Node(0, ns, None, 0, 0, np.zeros(0, KEY_DTYPE))
            root.lock = 1  # never evicted
            self.version += 1
        return root

    def _drop_root(self, root: Node) -> None:
        """A cache salt's root once nothing is left below it (no child, no lock but its own) goes, as a salt per
        tenant would keep one each; the unsalted namespaces' roots stay. `_root` makes it again on the next insert."""
        if _empty_root(root) and root.ns not in _KEPT_ROOTS and self.roots.get(root.ns) is root:
            del self.roots[root.ns]
            self.version += 1

    def _touch(self, node: Node) -> None:
        node.last_used = next(self._clock)

    def touch_path(self, node: Node | None) -> None:
        """Mark a path as just used (a preempted request will be back for it)."""
        while node is not None and node.parent is not None:
            node.last_used = next(self._clock)
            node = node.parent

    @staticmethod
    def _filed(node: Node) -> Hashable:
        """`node`'s key in its parent's `children` (`child_key`)."""
        return child_key(int(node.key[0]), node.spans, node.start)

    def _new_node(self, parent: Node, start: int, end: int, key: np.ndarray, pages: list[int], step: int,
                  aid: int, spans: Spans | None = None) -> Node:
        ps = self.ps
        node = Node(next(self._ids), parent.ns, parent, start, end, np.array(key, KEY_DTYPE))
        node.spans = spans
        node.pages = list(pages[start // ps:end // ps])
        node.tail = pages[end // ps] if end % ps else 0
        held = node.pages + ([node.tail] if node.tail else [])
        if len(node.pages) != max(0, end // ps - start // ps) or (end % ps and not node.tail) or 0 in held:
            raise ValueError(f"pages {pages} do not cover positions [{start}, {end})")
        self.version += 1
        self.kv.incref(held)
        self.tree_refs[held] += 1
        node.ready, node.writer = step, aid
        parent.children[self._filed(node)] = node
        self.num_nodes += 1
        self._touch(node)
        self._push_leaf(node)  # a candidate: the pop re-validates (`Node.evictable`)
        self._record(step, aid, KV, node, 0)
        return node

    def _split(self, n: Node, m: int) -> Node:
        """Split `n` at key position m (n.start < m < n.end): a new parent P = [n.start, m) takes the full pages ending
        before m and references the page holding m as its tail; `n` keeps [m, end), its tail and its checkpoint.
        Returns P."""
        self.version += 1
        ps = self.ps
        k = m - n.start
        p = Node(next(self._ids), n.ns, n.parent, n.start, m, n.key[:k].copy())
        p.spans = n.spans
        n.key = n.key[k:].copy()
        cut = m // ps - n.start // ps
        p.pages, n.pages = n.pages[:cut], n.pages[cut:]
        if m % ps:
            p.tail = n.pages[0] if m // ps < n.end // ps else n.tail
            self.kv.incref([p.tail])
            self.tree_refs[p.tail] += 1
        n.start = m
        p.ready, p.writer, p.lock, p.last_used = n.ready, n.writer, n.lock, n.last_used
        n.parent.children[self._filed(p)] = p
        p.children[self._filed(n)] = n
        n.parent = p
        self.num_nodes += 1
        # Pending: a rollback of any write of n in flight must remove p too, not only of the one that stamped n last (an
        # adoption stamps a node with its step, while the pages it did not swap stay an earlier write's)
        for step, aid in self._pending_kv(n):
            self._record(step, aid, KV, p, 0)
        return p

    def _pending_kv(self, node: Node) -> list[tuple[int, int]]:
        """(step, aid) of the writes in flight recorded on `node`'s KV, each once, in the order they were recorded."""
        out = [(s, aid) for s, recs in self._writes.items() if s > self.done_step
               for aid, kind, x, _ in recs if x is node and kind == KV]
        return list(dict.fromkeys(out))

    def _page(self, node: Node, pos: int) -> int:
        """The page of `node` holding key position `pos` (inside it)."""
        i = pos // self.ps
        return node.pages[i - node.start // self.ps] if i < node.end // self.ps else node.tail

    def _detach(self, node: Node) -> int:
        """Remove one node (no children left) from the tree: drop its page references and its checkpoint. Returns
        the pages that became free."""
        self.version += 1
        parent = node.parent
        filed = self._filed(node)
        if parent.children.get(filed) is node:
            del parent.children[filed]
        held = node.pages + ([node.tail] if node.tail else [])
        self.tree_refs[held] -= 1
        freed = self.kv.release(held)
        if node.ckpt:
            self._drop_ckpt(node)
        node.dead = True
        node.pages, node.tail = [], 0
        self.num_nodes -= 1
        if not parent.children:
            if parent.parent is not None:
                self._push_leaf(parent)  # a candidate, locked or dying: the pop re-validates (`Node.evictable`)
            else:
                self._drop_root(parent)
        return freed

    def _drop_ckpt(self, node: Node) -> None:
        self.version += 1
        self.states.release(node.ckpt)
        node.ckpt = 0
        node.ckpt_ready = -1
        self.num_ckpts -= 1
        if node.demoted:
            node.demoted = False
            self.num_demoted -= 1

    # -- matching ----------------------------------------------------------------------------------------------

    def match(self, ns: Hashable, key: np.ndarray, step: int, limit: int,
              forbid: Callable[[int], bool] | None = None, spans: Spans | None = None) -> Match:
        """Walk the tree along `key` for a plan with step id `step` (see the module docstring). `forbid(pos)`: a
        checkpoint at `pos` may not be the hit (nor a pending one). `spans`: the key's image spans."""
        m = Match()
        root = self.roots.get(ns)
        if root is None or not len(key):
            return m
        node, pos = root, 0
        usable = True  # every node so far is visible to this plan
        kv_usable = 0
        best_ckpt: tuple[int, Node | None] = (0, None)
        n_key = len(key)
        while pos < n_key:
            v = int(key[pos])
            child = node.children.get(v if v < SPAN_BASE else child_key(v, spans, pos))
            if child is None:
                break
            n = verified(child, pos, common_prefix(child.key, key[pos:]), spans)
            m.path.append(child)
            child.last_used = next(self._clock)
            if child.ready >= step:
                usable = False
            if usable:
                kv_usable = pos + n
            if n < len(child.key):
                pos += n
                break
            pos += n
            node = child
            if child.ckpt and child.end <= limit and (forbid is None or not forbid(child.end)):
                if usable and child.ckpt_ready < step:
                    best_ckpt = (child.end, child)
                else:
                    m.pending = child.end
        m.kv_len = pos
        m.held = held = min(pos, limit)
        if self.hybrid:
            m.hit, m.node = best_ckpt
            if m.node is not None:
                m.node.ckpt_used = next(self._clock)
                if m.node.demoted:  # live again: its heap entry moves up when popped (`evict_checkpoints`)
                    self.promote(m.node)
            m.lost = (("writing", min(max(m.pending, m.hit), held)), ("checkpoint", held))
        else:
            ps = self.ps
            visible = min(kv_usable, limit)
            m.hit = visible // ps * ps
            all_kv = held // ps * ps
            m.pending = all_kv if all_kv > m.hit else 0
            m.node = self.node_covering(m.path, m.hit - 1) if m.hit else None
            m.lost = (("page", visible), ("writing", held))
        if m.pending <= m.hit:
            m.pending = 0
        return m

    def end_at_hit(self, m: Match) -> None:
        """Split the node holding an attention-only hit's last position (`m.node`) at the hit: `m.node` becomes the
        new node ending there, inserted into `m.path`, so a lock on it covers what the hit reads and nothing past it.
        Locking the whole node would pin pages the hit never reads, and a lone request (nothing else running) whose
        first chunk does not fit beside them would never run: a livelock, in pools at the startup bound."""
        old = m.node
        if old.end != m.hit:
            m.node = self._split(old, m.hit)
            m.path.insert(m.path.index(old), m.node)

    @staticmethod
    def node_covering(path: list[Node], pos: int) -> Node | None:
        for node in path:
            if node.start <= pos < node.end:
                return node
        return None

    def hit_pages(self, m: Match) -> tuple[list[int], int]:
        """The tree pages of a hit: full pages for positions [0, hit // ps × ps), and its partial last page when the
        hit is not page-aligned (the caller copies it: KV_COPY), else 0."""
        ps, length = self.ps, m.hit
        full: list[int] = []
        tail = 0
        for node in m.path:
            if node.start >= length:
                break
            lo, hi = node.start // ps, min(node.end, length) // ps
            if hi > lo:
                full.extend(node.pages[:hi - lo])
            if length % ps and node.start < length <= node.end:
                idx = length // ps
                tail = node.pages[idx - lo] if idx < node.end // ps else node.tail
        if len(full) != length // ps:
            raise RuntimeError(f"hit of {length} tokens found {len(full)} full pages")
        return full, tail

    # -- locks ---------------------------------------------------------------------------------------------------

    def lock(self, node: Node | None) -> None:
        self.version += 1
        while node is not None:
            node.lock += 1
            node = node.parent

    def unlock(self, node: Node | None) -> None:
        self.version += 1
        root = None
        while node is not None:
            node.lock -= 1
            if node.lock < 0:
                raise RuntimeError(f"{node} unlocked more often than locked")
            if node.lock == 0 and not node.dead and not node.children:
                self._push_leaf(node)
            root, node = node, node.parent
        if root is not None and not root.children:  # its subtree was removed while locked (rollback)
            self._drop_root(root)

    # -- insertion -----------------------------------------------------------------------------------------------

    def insert(self, ns: Hashable, key: np.ndarray, end: int, pages: list[int], step: int, aid: int,
               cow: int = 0, hit: int = 0, spans: Spans | None = None) -> Node:
        """Make key[:end] a path of the tree. New nodes take the request's pages (`pages[i]` is its page of index i)
        and are stamped with `step` and `aid`. Where the tree already holds positions the request also computed, the
        tree adopts the request's pages for every page only the tree references (`_adopt`), so a request never holds
        more memory through the cache than without it; except `cow`, the request's copy of the page holding its hit
        `hit` (KV_COPY), in a node that starts before the hit, and an unverified page of the request where the tree's
        is not (module docstring, Pending writes and rollback, Unverified pages). Returns the node of the path ending
        exactly at `end` (split there if needed), so a lock taken on it covers what the request computed and nothing
        more. `spans`: the key's image spans."""
        if end <= 0:
            raise ValueError("empty insert")
        node, pos = self._root(ns), 0
        while pos < end:
            v = int(key[pos])
            child = node.children.get(v if v < SPAN_BASE else child_key(v, spans, pos))
            if child is None:
                return self._new_node(node, pos, end, key[pos:end], pages, step, aid, spans)
            n = verified(child, pos, common_prefix(child.key, key[pos:end]), spans)
            self._touch(child)
            if n < len(child.key) and pos + n < end:
                node = self._split(child, pos + n)
                self._adopt(node, pages, end, step, aid, cow, hit)
                pos += n
                continue
            if pos + n >= end and child.end != end and end % self.ps and self.unverified[self._page(child, end)]:
                # The tree's page holding `end` holds unverified writes past the request's end: split first, so the
                # node ending at `end` takes the request's page (all of its positions there) as its tail
                child = self._split(child, end)
            self._adopt(child, pages, end, step, aid, cow, hit)
            if pos + n >= end:
                return child if child.end == end else self._split(child, end)
            node, pos = child, pos + n
        return node

    def _adopt(self, node: Node, pages: list[int], end: int, step: int, aid: int, cow: int = 0, hit: int = 0,
               force: bool = False) -> None:
        """Swap the node's pages that only the tree references (and that the request wrote completely: positions below
        `end`) for the request's own pages of the same indices, freeing the tree's duplicates; never its copy-on-write
        page `cow` in a node that starts before its hit (`insert`), nor, unless `force`, one of its unverified pages
        for a page that is not (module docstring). The node then holds what `step` writes, so it is stamped with it
        (unless a later step stamped it already) and recorded under it: a rollback of `step` removes it. A tree page
        another request still references, or one the request's unverified page may not replace, is kept and counted
        (`kept_dups`): the inserting request then holds a duplicate that its lock pins once the other request leaves,
        until an insert or `readopt` of its path adopts it."""
        ps, kv, refs, unv = self.ps, self.kv, self.tree_refs, self.unverified
        lo = node.start // ps
        if node.start >= hit:
            cow = 0  # the node starts at or after the hit: the copy holds the request's own positions there
        swap = []
        for j, p in enumerate(node.pages):
            mine = pages[lo + j] if lo + j < len(pages) else 0
            if (lo + j + 1) * ps <= end and mine and mine != p and mine != cow:
                if kv.refcount[p] == refs[p] and (force or not unv[mine] or unv[p]):
                    swap.append((j, p, mine))
                else:
                    self.kept_dups += 1
        t = node.tail
        tail_mine = pages[node.end // ps] if t and node.end // ps < len(pages) and node.end <= end else 0
        if tail_mine and tail_mine != t and tail_mine != cow:
            if kv.refcount[t] == refs[t] and (force or not unv[tail_mine] or unv[t]):
                swap.append((-1, t, tail_mine))
            else:
                self.kept_dups += 1
        if not swap:
            return
        self.version += 1
        self.swaps += 1
        old = [p for _, p, _ in swap]
        new = [m for _, _, m in swap]
        kv.incref(new)
        refs[new] += 1
        refs[old] -= 1
        kv.release(old)
        for j, _, m in swap:
            if j < 0:
                node.tail = m
            else:
                node.pages[j] = m
        if step > node.ready:
            node.ready = step
            node.writer = aid
        self._record(step, aid, KV, node, 0)

    def readopt(self, deep: Node, pages: list[int], end: int, step: int, aid: int, cow: int = 0, hit: int = 0,
                force: bool = False) -> bool:
        """`_adopt` a running request's pages for every node on the path to `deep`, positions below `end`, without
        splitting anything (`cow`, `hit`: as `insert`; `force`: its unverified pages too). Returns whether some
        duplicate remains (another request still references the tree's page, or an unverified page was kept out)."""
        before = self.kept_dups
        node = deep
        while node.parent is not None:
            if node.start < end:
                self._adopt(node, pages, end, step, aid, cow, hit, force)
            node = node.parent
        return self.kept_dups != before

    def tree_only(self) -> bool:
        """Whether some page is the tree's alone (no request's page table holds it): the only kind of page `_adopt`
        and `share_tails` give up for a duplicate. Without one, neither swaps nor frees anything."""
        refs = self.tree_refs
        return bool(((refs > 0) & (self.kv.refcount == refs)).any())

    def share_tails(self, force: bool = False) -> int:
        """Give every locked junction whose tail only the tree references, and which no child shares, the page of that
        index of its first child that qualifies (module docstring, Junction tails): the junction and the child settled
        (no write of either in flight), and, unless `force`, the child's page verified where the tail is. Children
        before parents, so the junctions inside one page all end on one page. Neither stamps nor records anything.
        Returns the pages freed."""
        ps, kv, refs, unv, done = self.ps, self.kv, self.tree_refs, self.unverified, self.done_step
        freed = 0
        for root in self.roots.values():
            order, stack = [], [root]
            while stack:  # the locked part of the tree, parents first (a lock counts on every ancestor too)
                node = stack.pop()
                order.append(node)
                stack.extend(c for c in node.children.values() if c.lock)
            for node in reversed(order):
                t = node.tail
                if not t or not node.children or node.ready > done or kv.refcount[t] != refs[t]:
                    continue
                i = node.end // ps
                held = [(c, c.pages[0] if c.end // ps > i else c.tail) for c in node.children.values()]
                if any(p == t for _, p in held):
                    continue  # a child shares it: no page more
                for c, p in held:
                    if c.ready <= done and (force or not unv[p] or unv[t]):
                        self.version += 1
                        self.swaps += 1
                        kv.incref([p])
                        refs[p] += 1
                        refs[t] -= 1
                        freed += kv.release([t])
                        node.tail = p
                        break
        self.shared_tails += freed
        return freed

    def node_ending_at(self, deep: Node, c: int) -> Node:
        """The node ending exactly at key position c on the path to `deep` (splitting one if needed)."""
        node = deep
        while node.start >= c:
            node = node.parent
        if node.parent is None or c > node.end:
            raise ValueError(f"no node of the path to {deep} holds position {c - 1}")
        return node if node.end == c else self._split(node, c)

    # -- checkpoints ---------------------------------------------------------------------------------------------

    def alloc_checkpoint(self, step: int) -> int:
        """A free checkpoint slot for the plan with step id `step` (evicting in `evict_checkpoints`' order, the demoted
        ones first, none that plan writes or reads, if needed), or 0. The slot store's only: window mode makes a window
        from a request's own pages, and an eviction here would lose a window for nothing."""
        if self.states is None:
            return 0
        if isinstance(self.states, WindowStore):
            raise RuntimeError("alloc_checkpoint in window mode: a window is made from a request's pages")
        s = self.states.alloc_ckpt()
        if s is None and self.evict_checkpoints(1, step):
            s = self.states.alloc_ckpt()
        return s or 0

    def pin_checkpoint(self, node: Node) -> None:
        """The plan under construction reads node's checkpoint: keep it until `unpin_checkpoints`."""
        self._pinned.add(node.ckpt)

    def unpin_checkpoints(self) -> None:
        """The plan under construction is built (or abandoned)."""
        self._pinned.clear()

    @staticmethod
    def _owns(node: Node, slot: int, step: int) -> bool:
        """`node` still holds the checkpoint `slot` that step `step` attached, as undoing that write checks: an evicted
        slot may be reused by a later step. A removed node fails it (its checkpoint went: slot 0)."""
        return node.ckpt == slot and node.ckpt_ready == step

    def drop_checkpoint(self, node: Node, slot: int, step: int) -> bool:
        """Free the checkpoint `slot` that `step` attached to `node`, if the node still holds it and nothing pins it."""
        if node.dead or not self._owns(node, slot, step) or slot in self._pinned:
            return False
        self._drop_ckpt(node)
        return True

    @staticmethod
    def forks(node: Node) -> bool:
        """Whether the tree forks at or below `node` before any deeper checkpoint: two paths share its checkpoint as
        the deepest above their divergence (a shared prefix, or a branch point a grid below the divergence)."""
        while len(node.children) == 1:
            node = next(iter(node.children.values()))
            if node.ckpt:
                return False
        return len(node.children) > 1

    def demote(self, node: Node) -> bool:
        """Evict node's checkpoint before every live one (module docstring, Demoted checkpoints). Returns whether it was
        live."""
        if not node.ckpt or node.dead or node.demoted:
            return False
        self.version += 1
        node.demoted = True
        self.num_demoted += 1
        self.demotions += 1
        self._push_ckpt(node)  # a lower key: its entry goes in now
        return True

    def promote(self, node: Node) -> None:
        """Make a demoted checkpoint live again (a hit, or a demotion undone): its key rises, so its entry moves when
        popped."""
        if node.demoted:
            self.version += 1
            node.demoted = False
            self.num_demoted -= 1

    def attach(self, node: Node, slot: int, step: int, aid: int, source: str, on_precise: bool = False) -> None:
        """Make `slot` (an allocated or donated checkpoint slot, or a window) the state at node.end, written by `step`.
        `source`: why it was taken (CHECKPOINT_SOURCES). `on_precise`: its writer stood on a precise copy (the precise
        prefix: its state does not carry an FP8 copy's draw of the prefix below it)."""
        if node.ckpt:
            raise RuntimeError(f"{node} already has a checkpoint")
        if not self.states.is_checkpoint(slot):
            raise RuntimeError(f"checkpoint {slot} is not an allocated checkpoint slot or window")
        self.version += 1
        node.ckpt = slot
        node.ckpt_ready = step
        node.ckpt_used = next(self._clock)
        node.hits = node.served = 0
        node.ckpt_src = SOURCE_INDEX[source]
        node.on_precise = on_precise
        node.demoted = False
        self.num_ckpts += 1
        self._push_ckpt(node)
        self._compact_ckpt_heap()
        self._record(step, aid, CKPT, node, slot)

    def _record(self, step: int, aid: int, kind: int, node: Node, slot: int) -> None:
        if step > self.done_step:  # writes of completed steps need no rollback record
            self._writes.setdefault(step, []).append((aid, kind, node, slot))

    # -- eviction ------------------------------------------------------------------------------------------------

    def _push_leaf(self, node: Node) -> None:
        heapq.heappush(self._leaf_heap, (node.last_used, node.id, node))

    @staticmethod
    def _ckpt_key(node: Node) -> tuple[int, int]:
        """A checkpoint's eviction key: demoted ones first, then least recently used (module docstring)."""
        return (0 if node.demoted else 1), node.ckpt_used

    def _push_ckpt(self, node: Node) -> None:
        heapq.heappush(self._ckpt_heap, (*self._ckpt_key(node), node.id, node))

    def _compact_ckpt_heap(self) -> None:
        """Drop the heap's entries for nodes without a checkpoint once they pile up (never inside a pop loop)."""
        if len(self._ckpt_heap) > 4 * self.num_ckpts + 1024:
            self._ckpt_heap = [e for e in self._ckpt_heap if e[3].ckpt and not e[3].dead]
            heapq.heapify(self._ckpt_heap)

    def evict_pages(self, n: int) -> int:
        """Evict least recently used unlocked leaves, in flight or not (module docstring), until they free at least n
        pages or nothing is left to evict. Returns the pages freed."""
        freed = 0
        heap = self._leaf_heap
        while freed < n and heap:
            stamp, _, node = heapq.heappop(heap)
            if not node.evictable:
                continue  # pushed again when it becomes an unlocked leaf
            if stamp != node.last_used:
                self._push_leaf(node)
                continue
            if node.ckpt:
                self._evicting(node)
            freed += self._detach(node)
            self.evicted_nodes += 1
        self.evicted_pages += freed
        if len(heap) > 4 * self.num_nodes + 1024:
            self._leaf_heap = [e for e in heap if not e[2].dead]
            heapq.heapify(self._leaf_heap)
        return freed

    def evict_checkpoints(self, n: int, step: int, *, spare_locked: bool = False,
                          enough: Callable[[], bool] | None = None) -> int:
        """Free up to n checkpoint slots, or with `enough` until it holds: the demoted checkpoints first, then the live
        ones, least recently used first in each tier, locked or not and in flight or not, except the ones the plan with
        step id `step` writes or reads (module docstring). `spare_locked` also spares locked nodes' checkpoints (window
        mode's first pass). The nodes keep their KV."""
        done = 0
        heap, later = self._ckpt_heap, []
        while done < n and heap and not (enough is not None and enough()):
            tier, stamp, _, node = heapq.heappop(heap)
            if node.dead or not node.ckpt:
                continue  # attach pushes it again
            key = self._ckpt_key(node)
            if key != (tier, stamp):
                if key > (tier, stamp):  # used or made live since: its entry moves up
                    self._push_ckpt(node)
                continue  # else demoted since: the entry its demotion pushed comes first
            if node.ckpt_ready >= step or node.ckpt in self._pinned or (spare_locked and node.lock):
                later.append(node)
                continue
            self._evicting(node)
            self._drop_ckpt(node)
            done += 1
        for node in later:
            self._push_ckpt(node)
        self._compact_ckpt_heap()
        return done

    def _evicting(self, node: Node) -> None:
        """Count the eviction of node's checkpoint (`evicted_ckpts`, `evicted_by`, `evicted_window_pages`), before it
        goes. A refined one that served resumes (a shared prefix's precise copy) is also logged."""
        self.evicted_ckpts += 1
        self.evicted_by[node.ckpt_src][node.served > 0] += 1
        if node.ckpt_src == SOURCE_INDEX["refine"] and node.served > 0 and not self._clearing:
            log.warning("evicting a refined checkpoint that served %d resumes (node %d, position %d): a later "
                        "request on its prefix may refine it again", node.served, node.id, node.end)
        if isinstance(self.states, WindowStore):
            self.evicted_window_pages += self.states.freed_by(node.ckpt)

    def clear(self) -> int:
        """Evict everything unlocked (tests). Returns the pages freed."""
        self.done_step = max(self.done_step, max(self._writes, default=0))
        for root in self.roots.values():
            stack = [root]
            while stack:
                node = stack.pop()
                if not node.children and node.parent is not None:
                    self._push_leaf(node)  # locked ones too: the pop re-validates (`Node.evictable`)
                stack.extend(node.children.values())
        self._clearing = True
        try:
            freed = self.evict_pages(1 << 62)
        finally:
            self._clearing = False
        for root in list(self.roots.values()):  # also one a rollback left empty
            self._drop_root(root)
        return freed

    # -- completion and rollback ---------------------------------------------------------------------------------

    def complete(self, step: int, bad: set[int] | frozenset[int] = frozenset()) -> int:
        """Step `step`'s results arrived. Writes by requests in `bad` (rows that reported NONFINITE) are removed: KV
        nodes with their subtree, checkpoints alone. Returns how many entries were rolled back."""
        self.done_step = max(self.done_step, step)
        writes = self._writes.pop(step, ())
        if not bad:
            return 0
        n = 0
        for aid, kind, node, slot in writes:
            if aid not in bad or node.dead:
                continue
            if kind == KV:
                n += self.remove_subtree(node)
            elif self._owns(node, slot, step):
                self._drop_ckpt(node)
                n += 1
        self.rolled_back += n
        return n

    def drop_writes(self, step: int, aids: set[int]) -> int:
        """Roll back the not-yet-completed writes of `aids` at `step` (requests failed before their step's results
        arrived)."""
        writes = self._writes.get(step)
        if not writes:
            return 0
        n = 0
        keep = []
        for w in writes:
            aid, kind, node, slot = w
            if aid not in aids:
                keep.append(w)
            elif not node.dead:
                if kind == KV:
                    n += self.remove_subtree(node)
                elif self._owns(node, slot, step):
                    self._drop_ckpt(node)
                    n += 1
        self._writes[step] = keep
        self.rolled_back += n
        return n

    def remove_subtree(self, node: Node) -> int:
        """Remove `node` and everything below it, locked or not (rollback). Returns the nodes removed."""
        self.swaps += 1
        order, stack = [], [node]
        while stack:
            x = stack.pop()
            order.append(x)
            stack.extend(x.children.values())
        for x in reversed(order):  # children first
            x.children.clear()
            self._detach(x)
        return len(order)

    def remove_unverified(self, ns: Hashable, key: np.ndarray, aids: list[int], lo: int = 0,
                          spans: Spans | None = None) -> int:
        """Remove what the tree holds on `key`'s path from the first position at or after `lo` held in a page one of
        `aids` wrote unverified (module docstring), locked or not, as rollback does: the node holding it is split at
        that page's start (or `lo`), and the subtree from there goes, the paths below it included (they read it).
        Positions before it, and what branches off before it, stay. Never while a plan is built. `spans`: the key's
        image spans. Returns the nodes removed."""
        ps, unv = self.ps, self.unverified
        node, at = self.roots.get(ns), 0
        while node is not None and at < len(key):
            v = int(key[at])
            child = node.children.get(v if v < SPAN_BASE else child_key(v, spans, at))
            if child is None:
                return 0
            n = verified(child, at, common_prefix(child.key, key[at:]), spans)
            first, stop = max(child.start, lo), at + n  # child's positions on the path, from lo
            if stop > first:
                base = child.start // ps
                held = np.asarray([*child.pages, child.tail] if child.tail else child.pages, np.int64)
                i0 = first // ps - base
                hits = np.flatnonzero(np.isin(unv[held[i0:(stop - 1) // ps - base + 1]], aids))
                if len(hits):
                    q = max(first, (base + i0 + int(hits[0])) * ps)
                    if child.start < q:
                        self._split(child, q)  # `child` keeps [q, end)
                    k = self.remove_subtree(child)
                    self.rolled_back += k
                    return k
            if n < len(child.key):
                return 0
            node, at = child, child.end
        return 0

    def pending_writers(self, path: list[Node], hit: int, node: Node | None) -> set[tuple[int, int]]:
        """(aid, step) of the not-yet-completed writes a hit depends on: the KV nodes of its path whose pages it reads
        (the full pages below it, and the page holding it, which it copies; not a tail that a deeper node's full page of
        the same index overrides), and its checkpoint. A KV node's are every write in flight recorded on it, not only
        its stamp's: two adoptions in one step leave the node the second's pages under the first's stamp (`_adopt`)."""
        ps = self.ps
        deps = set()
        kv = None  # node id -> (aid, step) of its KV writes in flight, built once if some node needs it
        for x in path:
            if x.start >= hit:
                break
            if x.ready > self.done_step and (min(x.end, hit) // ps > x.start // ps or (hit % ps and hit <= x.end)):
                if kv is None:
                    kv = {}
                    for s, recs in self._writes.items():
                        if s > self.done_step:
                            for aid, kind, n2, _ in recs:
                                if kind == KV:
                                    kv.setdefault(n2.id, set()).add((aid, s))
                deps.update(kv.get(x.id, ()))
                deps.add((x.writer, x.ready))
        if node is not None and node.ckpt and node.ckpt_ready > self.done_step:
            for aid, kind, n2, _ in self._writes.get(node.ckpt_ready, ()):
                if kind == CKPT and n2 is node:
                    deps.add((aid, node.ckpt_ready))
        return deps

    # -- accounting ----------------------------------------------------------------------------------------------

    @property
    def cached_pages(self) -> int:
        return int(np.count_nonzero(self.tree_refs))

    def check(self, full: bool = False) -> None:
        """Every structural invariant (tests). Raises AssertionError. The walk over the nodes (`_walk`) runs when the
        tree changed since the last check (`version`), and always with `full` (the reference). Otherwise the last walk's
        results stand, and what lies outside the nodes is checked again: each checkpoint slot's kind in its pool, the
        node, checkpoint and demoted counts, `tree_refs`, the pages' refcounts, and the checkpoint heap (every
        checkpoint has an entry at or below its key, module docstring). Both fail on the same assertion: the walk's
        other assertions read only what bumps `version` (`tests/sched/test_accounting_checks.py`)."""
        last = self._checked
        if full or last is None or last[0] != self.version:
            refs, ckpts, n_nodes, n_demoted = self._walk()
            if not full:
                self._checked = (self.version, refs, ckpts, n_nodes, n_demoted)
        else:
            _, refs, ckpts, n_nodes, n_demoted = last
            for slot, node in ckpts.items():  # in the walk's order
                assert self.states.is_checkpoint(slot), node
        assert n_nodes == self.num_nodes, (n_nodes, self.num_nodes)
        assert len(ckpts) == self.num_ckpts, (len(ckpts), self.num_ckpts)
        assert n_demoted == self.num_demoted, (n_demoted, self.num_demoted)
        assert np.array_equal(refs, self.tree_refs), "tree page references out of sync"
        assert (self.kv.refcount >= self.tree_refs).all(), "a tree page has fewer references than the tree holds"
        least: dict[int, tuple[int, int]] = {}  # node id -> its least heap entry
        for tier, stamp, nid, _ in self._ckpt_heap:
            if nid not in least or (tier, stamp) < least[nid]:
                least[nid] = (tier, stamp)
        for node in ckpts.values():  # the walk's checkpoints: a use since only raises a key
            assert least.get(node.id, (2, 0)) <= self._ckpt_key(node), f"{node}: no heap entry at or below its key"

    def _walk(self) -> tuple[np.ndarray, dict[int, Node], int, int]:
        """`check`'s walk: every node's assertions, then the tree's references per page, its checkpoint slots (with
        their nodes, in the walk's order), its node count and its demoted checkpoints."""
        ps = self.ps
        refs = np.zeros_like(self.tree_refs)
        ckpts = {}
        n_nodes = n_demoted = 0
        for ns, root in self.roots.items():
            assert root.lock >= 1 and root.start == root.end == 0
            stack = [root]
            while stack:
                node = stack.pop()
                child_locks = 0
                for first, c in node.children.items():
                    assert not c.dead, c
                    assert c.parent is node and c.start == node.end and c.ns == ns, c
                    assert len(c.key) == c.end - c.start > 0 and self._filed(c) == first, c
                    child_locks += c.lock
                    stack.append(c)
                assert node.lock >= child_locks, f"{node}: lock below its children's"
                if node.parent is None:
                    continue
                n_nodes += 1
                assert len(node.pages) == max(0, node.end // ps - node.start // ps), node
                assert bool(node.tail) == bool(node.end % ps), node
                for p in node.pages + ([node.tail] if node.tail else []):
                    assert 0 < p < len(refs), node
                    refs[p] += 1
                if node.ckpt:
                    assert node.ckpt not in ckpts, f"checkpoint slot {node.ckpt} owned twice"
                    ckpts[node.ckpt] = node
                    assert self.states.is_checkpoint(node.ckpt), node
                    n_demoted += node.demoted
                else:
                    assert not node.demoted, f"{node}: demoted without a checkpoint"
        return refs, ckpts, n_nodes, n_demoted

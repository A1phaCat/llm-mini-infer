"""The radix suite's model of the cache (`Model`, which tests/scripts/plandiff.py patches), its workloads and their
driver (tests/sched/test_radix.py, tests/support/fuzzing.py)."""

from __future__ import annotations

import os

import numpy as np

from llm_mini_infer.cache.pools import PagePool, StatePool
from llm_mini_infer.cache.radix import RadixCache

from .fuzzing import event, st
from .harness import FULL_CHECKS_ENV


class Model:
    def __init__(self, ps: int, num_pages: int, n_ckpt: int, hybrid: bool):
        self.ps = ps
        self.kv = PagePool(num_pages)
        self.states = StatePool(1 + 4 + n_ckpt, 4)
        self.cache = RadixCache(ps, self.kv, self.states, hybrid=hybrid)
        self.content = np.full((num_pages, ps), -1, np.int64)  # token at each position of each page
        self.ckpt_tokens: dict[int, tuple[int, ...]] = {}  # checkpoint slot -> prefix it represents
        self.running: list[dict] = []  # requests holding pages and a lock
        self.step = 0
        self.aid = 0
        self.full_checks = os.environ.get(FULL_CHECKS_ENV) == "1"  # the tree check's reference implementation

    # -- accounting ------------------------------------------------------------------------------------------------

    def check(self) -> None:
        self.cache.check(self.full_checks)
        held = np.zeros(self.kv.num_pages, np.int32)
        for r in self.running:
            for p in r["pages"]:
                held[p] += 1
            assert not r["lock"].dead, "a locked node was removed"
        expect = self.cache.tree_refs + held
        expect[0] = 1
        assert np.array_equal(self.kv.refcount, expect), "refcounts != tree references + page tables"
        assert (self.kv.refcount >= 0).all()
        assert self.kv.num_free == int((self.kv.refcount == 0).sum()), "free list out of sync"
        live_ckpts = {s for s in self.ckpt_tokens if self.states.used[s] and self.states.kind[s] == 2}
        assert len(live_ckpts) == self.cache.num_ckpts

    def alloc(self, n: int) -> list[int] | None:
        if self.kv.num_free < n and self.cache.evict_pages(n - self.kv.num_free):
            event("evicted under pressure")
        return self.kv.alloc(n)

    # -- one request ---------------------------------------------------------------------------------------------

    def request(self, ns, tokens: np.ndarray, end: int, ckpts: list[int]) -> bool:
        """Match, take pages (shared, copied, new), check what the hit hands out, insert [0, end), add checkpoints."""
        ps, c = self.ps, self.cache
        self.step += 1
        self.aid += 1
        m = c.match(ns, tokens, self.step, len(tokens) - 1)
        hit = m.hit
        assert m.held == min(m.kv_len, len(tokens) - 1) and m.held >= hit
        ends = [end for _, end in m.lost]
        assert ends == sorted(ends) and (ends[-1] == m.held and ends[0] >= hit if ends else not m.held), m.lost
        if hit:
            event("hit")
            assert m.node is not None and m.kv_len >= hit
            if c.hybrid:
                assert m.node.end == hit and m.node.ckpt, "a checkpoint is used only at its exact offset"
                assert self.ckpt_tokens[m.node.ckpt] == tuple(tokens[:hit].tolist()), "checkpoint of another prefix"
            else:
                assert hit % ps == 0, "attention-only hits are page-aligned"
            c.lock(m.node)
        full, tail = c.hit_pages(m)
        for i, p in enumerate(full):
            assert (self.content[p] == tokens[i * ps:(i + 1) * ps]).all(), "shared page of another prefix"
        if hit % ps:
            lo = hit // ps * ps
            assert tail and (self.content[tail, :hit - lo] == tokens[lo:hit]).all(), "partial page of another prefix"
            event("copy-on-write")
        need = (1 if hit % ps else 0) + max(0, -(-end // ps) - -(-hit // ps))
        own = self.alloc(need)
        if own is None:
            c.unlock(m.node)
            return False
        self.kv.incref(full)
        pages = list(full)
        if hit % ps:
            cp = own.pop(0)
            self.content[cp] = self.content[tail]  # KV_COPY
            pages.append(cp)
        pages += own
        for pos in range(hit, end):
            self.content[pages[pos // ps], pos % ps] = tokens[pos]
        deep = c.insert(ns, tokens, end, pages, self.step, self.aid)
        c.lock(deep)
        if hit:
            c.unlock(m.node)
        for pos in sorted(set(ckpts)):
            if not 0 < pos <= end:
                continue
            node = c.node_ending_at(deep, pos)
            if node.ckpt:
                continue
            slot = c.alloc_checkpoint(self.step)
            if not slot:
                continue
            c.attach(node, slot, self.step, self.aid, "prefill")
            self.ckpt_tokens[slot] = tuple(tokens[:pos].tolist())
        self.running.append({"pages": pages, "lock": deep, "aid": self.aid, "step": self.step})
        return True

    def finish(self, i: int) -> None:
        r = self.running.pop(i)
        self.cache.unlock(r["lock"])
        self.kv.release(r["pages"])

    def complete(self, bad: set[int]) -> None:
        for step in range(self.cache.done_step + 1, self.step + 1):
            self.cache.complete(step, bad)
        # Requests whose locked node was rolled back let it go: the scheduler fails those that used a removed write,
        # and the others run on, caching nothing more
        if bad:
            for i in reversed(range(len(self.running))):
                if self.running[i]["lock"].dead:
                    self.finish(i)


ALPHABET = st.integers(0, 3)


@st.composite
def scenarios(draw):
    ps = draw(st.sampled_from([1, 2, 4, 8]))
    hybrid = draw(st.booleans())
    n = draw(st.integers(1, 30))
    stem = draw(st.lists(ALPHABET, min_size=1, max_size=40))  # a common prefix many requests share
    reqs = []
    for _ in range(n):
        if draw(st.booleans()):
            toks = stem[:draw(st.integers(0, len(stem)))] + draw(st.lists(ALPHABET, min_size=1, max_size=20))
        else:
            toks = draw(st.lists(ALPHABET, min_size=2, max_size=50))
        if len(toks) < 2:
            toks = [*toks, 0]
        end = draw(st.integers(1, len(toks)))
        ckpts = draw(st.lists(st.integers(1, len(toks)), max_size=3))
        reqs.append(dict(tokens=np.asarray(toks, np.int64), end=end, ckpts=ckpts,
                         ns=draw(st.sampled_from([None, None, "salt"])),
                         action=draw(st.sampled_from(["keep", "finish", "finish", "complete", "evict", "bad"]))))
    return dict(ps=ps, hybrid=hybrid, reqs=reqs, num_pages=draw(st.integers(24, 120)),
                n_ckpt=draw(st.integers(1, 6)))


def run_scenario(s: dict) -> None:
    mdl = Model(s["ps"], s["num_pages"], s["n_ckpt"], s["hybrid"])
    for r in s["reqs"]:
        mdl.request(r["ns"], r["tokens"], r["end"], r["ckpts"])
        mdl.check()
        act = r["action"]
        if act == "finish" and mdl.running:
            mdl.finish(0)
        elif act == "complete":
            mdl.complete(set())
        elif act == "evict":
            mdl.complete(set())
            locked = [x["lock"] for x in mdl.running]
            mdl.cache.evict_pages(5)
            mdl.cache.evict_checkpoints(2, mdl.step + 1)
            assert not any(x.dead for x in locked), "a locked path was evicted"
        elif act == "bad" and mdl.running:
            mdl.complete({mdl.running[-1]["aid"]})
            event("rollback")
        mdl.check()
    while mdl.running:
        mdl.finish(0)
    mdl.complete(set())
    mdl.check()
    mdl.cache.clear()
    mdl.check()
    assert mdl.kv.num_free == mdl.kv.num_pages - 1, "pages leaked"
    assert mdl.cache.num_nodes == 0 and mdl.states.num_free_ckpt == mdl.states.n_ckpt

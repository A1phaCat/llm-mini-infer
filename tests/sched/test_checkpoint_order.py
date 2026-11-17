"""Which checkpoint an eviction takes on multi-turn traffic (architecture.md, "Eviction"): the checkpoints a
conversation has moved past are demoted when one of its requests leaves a deeper checkpoint of its own, and go before
every live one. The scheduler and the fake runner run over the tiny hybrid model's geometry, built as the multi-turn
measurement builds them (tests/scripts/multi_turn.py); the tree's side, the demoted tier, is tested in
tests/sched/test_radix.py."""

from __future__ import annotations

import numpy as np
import pytest

from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, AddRequest
from llm_mini_infer.sched.scheduler import RUNNING

from ..scripts.multi_turn import Engine, build

FINISH = CHECKPOINT_SOURCES.index("finish")
PROMPT, REPLY, TOOL = 40, 30, 100


@pytest.fixture(scope="module")
def mdir(tmp_path_factory):
    from ..support.tiny_qwen import make_tiny_model

    return make_tiny_model(tmp_path_factory.mktemp("multiturn") / "hybrid")


class Driver:
    """The scheduler and engine core with two rows, driven by hand: each request's output tokens and the cached tokens
    its first admission reported."""

    def __init__(self, mdir, slots: int, demote: bool = True, interval: int = 256):
        eng = Engine(max_num_seqs=2, slots=slots, pages=512, order="engine" if demote else "lru",
                     decode_interval=interval)
        self.core, self.sched, self.runner, self.pools = build(eng, mdir, 512)
        self.out: dict[int, list[int]] = {}
        self.cached: dict[int, int] = {}
        self.done: set[int] = set()

    def send(self, rid: int, prompt: list[int], hints: list[int] | None = None, seed: int | None = None,
             max_tokens: int = REPLY, priority: int = 0) -> None:
        """Send a request; `seed` (default: `rid`) sets its sampled tokens, so two requests with the same seed and
        prompt reply alike."""
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(prompt, np.int32), max_tokens=max_tokens,
                                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=rid if seed is None else seed,
                                    stop_token_ids=[], ignore_eos=True, cache_hints=hints or [], priority=priority))

    def step(self) -> None:
        rep = self.core.step()
        for fo in (rep.outputs.values() if rep is not None else ()):
            self.cached.update(zip(fo.first_rids.tolist(), fo.first_cached.tolist()))
            at = 0
            for rid, n, fin in zip(fo.rids.tolist(), fo.n_new.tolist(), fo.finish.tolist()):
                self.out.setdefault(rid, []).extend(fo.tokens[at:at + n].tolist())
                at += n
                if fin:
                    self.done.add(rid)

    def finish(self, *rids: int) -> None:
        for _ in range(10_000):
            if all(r in self.done for r in rids):
                self.sched.check_accounting(full=True)
                return
            self.step()
        raise AssertionError("the requests did not finish")

    def preempt(self, rid: int) -> None:
        """Preempt request `rid` in the next plan, as page pressure would, and order the queue by priority."""
        sched, plan = self.sched, self.sched._schedule
        sched.policy = "priority"

        def hook():
            req = sched.reqs.get((0, rid))
            if req is not None and req.status == RUNNING:
                sched._schedule = plan
                sched._preempt(req.slot)
            return plan()

        sched._schedule = hook

    def checkpoints(self) -> dict[int, bool]:
        """The positions that hold a checkpoint, each with whether it is demoted."""
        return {n.end: n.demoted for n in nodes(self.sched.cache) if n.ckpt}


def nodes(cache) -> list:
    out, stack = [], list(cache.roots.values())
    while stack:
        n = stack.pop()
        if n.parent is not None:
            out.append(n)
        stack.extend(n.children.values())
    return out


def history(rng: np.random.Generator, n: int) -> list[int]:
    return rng.integers(128, 384, n).tolist()


@pytest.mark.parametrize("demote", [True, False])
def test_a_waiting_conversation_keeps_its_finish_checkpoint(mdir, demote: bool) -> None:
    """Two conversations, A and B, each a first turn of 40 prompt tokens (under the 64-token grid: no end-of-prefill
    checkpoint) and a reply of 30, so each leaves a finish checkpoint at position 69. Then A waits while B runs a second
    turn: its prompt is its history and 100 more tokens, so it resumes from its finish checkpoint, takes an
    end-of-prefill checkpoint at 133 and leaves a finish one at 199. Three slots hold the first three; B's second
    finish needs a fourth, so exactly one checkpoint is evicted. B's end-of-prefill checkpoint demotes the one it
    resumed from, and its finish demotes the end-of-prefill one; the eviction takes the least recently used of those,
    its first finish checkpoint, not A's, and A's next turn resumes where its history ends. With
    `dev.cache.demote_superseded` off, the least recently used of all goes: A's, which never served a resume, and A's
    next turn recomputes its whole history."""
    e = Driver(mdir, slots=3, demote=demote)
    rng = np.random.default_rng(7)
    a1, b1 = history(rng, PROMPT), history(rng, PROMPT)
    e.send(0, a1)
    e.send(1, b1)
    e.finish(0, 1)
    end1 = PROMPT + REPLY - 1
    assert e.checkpoints() == {end1: False}  # both at 69, on two paths
    assert sum(1 for n in nodes(e.sched.cache) if n.ckpt) == 2
    b2 = b1 + e.out[1] + history(rng, TOOL)
    e.send(2, b2)
    e.finish(2)
    assert e.cached[2] == end1  # B resumed from its finish checkpoint
    cache = e.sched.cache
    end2 = len(b2) + REPLY - 1
    assert cache.evicted_ckpts == 1
    if demote:
        assert cache.demotions == 2
        assert cache.evicted_by[FINISH] == [0, 1]  # a finish checkpoint that had served a resume: B's first
        assert e.checkpoints() == {end1: False, 133: True, end2: False}  # A's, B's end-of-prefill (demoted), B's last
    else:
        assert cache.demotions == 0
        assert cache.evicted_by[FINISH] == [1, 0]  # a finish checkpoint that had served no resume: A's
        assert e.checkpoints() == {end1: False, 133: False, end2: False}  # B's three
    e.send(3, a1 + e.out[0] + history(rng, TOOL))
    e.finish(3)
    assert e.cached[3] == (end1 if demote else 0)


def test_a_shared_prefix_checkpoint_is_never_demoted(mdir) -> None:
    """Two conversations share a prefix of 128 tokens (two grids), each first prompt 40 tokens past it. The first
    takes its end-of-prefill checkpoint at 128, the prefix's end (the grid point of its prompt's end): its finish leaves
    it live, as a first turn's end-of-prefill checkpoint may be the prefix its siblings resume from. The second resumes
    from it there, and the node becomes a junction. From then on no finish demotes it, though each supersedes it (the
    second's first turn resumed from it): another request resumed from it, and the tree forks below it. Their second
    turns continue where their first stopped, and demote the finish checkpoint they resumed from and their own
    end-of-prefill one; the prefix's checkpoint stays live through them."""
    e = Driver(mdir, slots=8)
    rng = np.random.default_rng(11)
    pre = history(rng, 128)
    x1, y1 = pre + history(rng, PROMPT), pre + history(rng, PROMPT)
    e.send(0, x1)
    e.finish(0)
    assert e.checkpoints()[128] is False and e.sched.cache.demotions == 0
    e.send(1, y1)
    e.finish(1)
    assert e.cached[1] == 128
    prefix = next(n for n in nodes(e.sched.cache) if n.end == 128)
    assert prefix.ckpt and not prefix.demoted and prefix.served == 1 and len(prefix.children) == 2
    assert e.sched.cache.demotions == 0
    e.send(2, x1 + e.out[0] + history(rng, TOOL))
    e.send(3, y1 + e.out[1] + history(rng, TOOL))
    e.finish(2, 3)
    assert prefix.ckpt and not prefix.demoted
    assert e.sched.cache.demotions == 4  # each second turn: the finish it resumed from, its end-of-prefill one
    demoted = {end for end, d in e.checkpoints().items() if d}
    assert 128 not in demoted and len(demoted) == 2  # 197 and 261, on both paths


def test_sibling_requests_keep_the_first_ones_prefix_checkpoint(mdir) -> None:
    """Requests that share a long prefix and differ in a short tail, one after another: the first's end-of-prefill
    checkpoint (the grid point of its prompt's end, inside the shared prefix) is what its siblings resume from, though
    nothing has resumed from it when the first finishes. A first request continues nothing, so it demotes none of its
    own; under pressure from other groups' requests (three slots, a group every few requests), each group's second
    request still finds its prefix's checkpoint, as with every checkpoint least recently used first."""
    e = Driver(mdir, slots=3)
    rng = np.random.default_rng(17)
    groups = [history(rng, 300) for _ in range(4)]
    rid = 0
    for g in groups:
        for _ in range(2):
            e.send(rid, g + history(rng, 6))
            e.finish(rid)
            rid += 1
    assert [e.cached[2 * k + 1] for k in range(4)] == [256] * 4
    assert e.sched.cache.demotions == 0


def test_a_branch_point_is_never_demoted(mdir) -> None:
    """A prefix of 100 tokens, off the grid. The second conversation's KV match reaches past the deepest checkpoint on
    its path, so its prefill takes a branch checkpoint (at 64, the grid point below the prompts' divergence); the third
    resumes from it. That checkpoint is then superseded by the third's own (the one it resumed from, served by it
    alone, on a node with one child), but the tree forks below it, at the divergence, before any deeper checkpoint:
    it is the deepest checkpoint above the fork, and stays live."""
    e = Driver(mdir, slots=8)
    rng = np.random.default_rng(13)
    pre = history(rng, 100)
    for rid in range(3):
        e.send(rid, pre + history(rng, PROMPT))
        e.finish(rid)
    assert e.cached[2] == 64
    branch = next(n for n in nodes(e.sched.cache) if n.end == 64)
    assert CHECKPOINT_SOURCES[branch.ckpt_src] == "branch" and branch.served == 1 and len(branch.children) == 1
    assert branch.ckpt and not branch.demoted and e.sched.cache.forks(branch)


def test_a_checkpoint_two_requests_resumed_from_stays_live(mdir) -> None:
    """A conversation's first turn leaves a finish checkpoint at 69. Two requests then send its history and 50 more
    tokens (under the next grid point, so neither takes an end-of-prefill checkpoint) and both resume from it. The first
    to finish leaves a deeper checkpoint on a path that does not fork, but the one it resumed from served the second
    request too, which may still want it: it stays live, and so it does after the second finishes."""
    e = Driver(mdir, slots=8)
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    e.send(0, h1)
    e.finish(0)
    p = h1 + e.out[0] + history(rng, 50)
    e.send(1, p)
    e.send(2, p)
    e.finish(1, 2)
    assert e.cached[1] == e.cached[2] == PROMPT + REPLY - 1
    resumed = next(n for n in nodes(e.sched.cache) if n.end == PROMPT + REPLY - 1)
    assert resumed.served == 2 and not resumed.demoted and e.sched.cache.demotions == 0


def test_a_hint_checkpoint_a_conversation_resumed_from_stays_live(mdir) -> None:
    """A client hint at 30 in a prompt of 60. A retry, the same prompt and the same replies, resumes from the hint while
    the first request decodes, and finishes on the first one's finish checkpoint. Nothing forks below the hint, and the
    retry resumed from it alone, but a hint is the client's: it stays live, though it lies above the retry's finish."""
    e = Driver(mdir, slots=8)
    p = history(np.random.default_rng(7), 60)
    e.send(0, p, hints=[30], seed=5)
    e.send(1, p, hints=[30], seed=5)
    e.finish(0, 1)
    assert e.out[0] == e.out[1] and e.cached[1] == 30
    assert e.checkpoints() == {30: False, 89: False} and e.sched.cache.demotions == 0


def test_a_retry_that_ends_on_a_cached_checkpoint_supersedes(mdir) -> None:
    """A first turn, then two requests with the same second prompt (170 tokens) and the same replies. The first resumes
    from the finish checkpoint at 69 and takes an end-of-prefill one at 133, which the second resumes from; the first's
    finish at 199 demotes the one it resumed from. The second's finish finds that checkpoint already attached, and
    still demotes the end-of-prefill one it resumed from (which its hit had made live again)."""
    e = Driver(mdir, slots=8)
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    e.send(0, h1)
    e.finish(0)
    p = h1 + e.out[0] + history(rng, TOOL)
    e.send(1, p, seed=5)
    e.send(2, p, seed=5)
    e.finish(1, 2)
    assert (e.cached[1], e.cached[2]) == (69, 133)
    assert e.checkpoints() == {69: True, 133: True, 199: False}


def test_a_preemption_checkpoint_is_its_writers_to_demote(mdir) -> None:
    """A request is preempted 15 tokens into its reply, and a higher-priority one that sends its prompt and those
    tokens takes its row and resumes from the preemption checkpoint (the other row is busy), finishing before the
    preempted one is readmitted. The checkpoint stays live: it is the preempted request's to resume. Its readmission
    then resumes from it, and its finish demotes it when nobody else resumed from it (the second case)."""
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    ref = Driver(mdir, slots=8)
    ref.send(0, h1)
    ref.finish(0)
    e = Driver(mdir, slots=8)
    e.send(1, history(rng, PROMPT))
    e.send(0, h1)
    for _ in range(15):
        e.step()
    e.preempt(0)
    e.send(2, h1 + ref.out[0][:20] + history(rng, 10), max_tokens=3, priority=-1)
    e.finish(2)
    left = next(n for n in nodes(e.sched.cache) if n.ckpt_src == CHECKPOINT_SOURCES.index("preempt"))
    assert e.sched.num_preemptions == 1 and e.cached[2] == left.end and 0 not in e.done
    assert left.served == 1 and not left.demoted and e.sched.cache.demotions == 0
    alone = Driver(mdir, slots=8)
    alone.send(0, h1)
    for _ in range(15):
        alone.step()
    alone.preempt(0)
    alone.finish(0)
    assert alone.sched.num_preemptions == 1 and alone.checkpoints()[left.end] is True


def test_a_failed_request_makes_the_checkpoints_it_demoted_live_again(mdir) -> None:
    """A second turn resumes from the finish checkpoint at 69 and takes an end-of-prefill one at 133, which demotes the
    one it resumed from. It then fails on a non-finite logit while decoding: its writes are rolled back, and the
    checkpoint it demoted is its conversation's latest again, live."""
    e = Driver(mdir, slots=8)
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    e.send(0, h1)
    e.finish(0)
    p = h1 + e.out[0] + history(rng, TOOL)
    e.runner.nonfinite_at = {1: {len(p) + 3}}
    e.send(1, p)
    e.finish(1)
    assert e.sched.num_failed == 1 and e.sched.cache.demotions == 1
    assert e.checkpoints() == {69: False, 133: False}


def test_decode_checkpoints_demote_and_are_demoted(mdir) -> None:
    """With a decode checkpoint every 16 positions: a first turn's decode checkpoint is demoted by its finish
    checkpoint, which lies deeper. A second turn of 20 tokens past the history takes no end-of-prefill checkpoint, so
    its first decode checkpoint is what demotes the finish checkpoint it resumed from, while it still decodes."""
    e = Driver(mdir, slots=8, interval=16)
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    e.send(0, h1)
    e.finish(0)
    end1 = PROMPT + REPLY - 1
    assert e.checkpoints() == {64: True, end1: False}
    e.send(1, h1 + e.out[0] + history(rng, 20))
    taken = e.sched.ckpts_taken["decode"]
    while e.sched.ckpts_taken["decode"] == taken:
        e.step()
    assert e.checkpoints() == {64: True, end1: True, 96: False}
    e.finish(1)
    assert e.checkpoints()[96 + 16] is True and e.cached[1] == end1


def test_an_admission_through_a_cached_paths_last_node_supersedes_its_own(mdir) -> None:
    """Three first turns leave three finish checkpoints at 69 in two slots: the first conversation's is evicted, its KV
    stays cached. Its second turn resumes from nothing, but its key runs through the whole of the path's last node,
    where the first turn stopped, so it continues that turn: its finish demotes its end-of-prefill checkpoint at 128."""
    e = Driver(mdir, slots=2)
    rng = np.random.default_rng(7)
    h1 = history(rng, PROMPT)
    e.send(0, h1)
    e.finish(0)
    for rid in (1, 2):
        e.send(rid, history(rng, PROMPT))
        e.finish(rid)
    e.send(3, h1 + e.out[0] + history(rng, TOOL))
    e.finish(3)
    assert e.cached[3] == 0
    assert e.checkpoints() == {128: True, PROMPT + REPLY + TOOL + REPLY - 1: False}

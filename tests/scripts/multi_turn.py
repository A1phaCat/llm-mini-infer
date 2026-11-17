"""Multi-turn traffic over the prefix cache's checkpoints, measured: the engine core turned by hand over the fake runner
(`runner/fake.py`), its pools sized and its scheduler built as `engine/process.py` builds them, deterministic per seed.

    uv run python -m tests.scripts.multi_turn --model hybrid --slots 8 12 16 24 32 40
    uv run python -m tests.scripts.multi_turn --model window --pages 400 600 900 --sessions=8
    uv run python -m tests.scripts.multi_turn --model hybrid --slots 12 --order leaf-last --idle 600

**The workload** (`Workload`). `sessions` conversations share a system prefix of `prefix` tokens. A conversation's first
prompt is the prefix and a first message of `tool` tokens; each later prompt is the previous prompt, its reply and a
tool result of `tool` tokens: the whole history, as an agent loop sends it. A reply is `reply` tokens (`max_tokens`, EOS
ignored). After a reply, the conversation thinks: a time in engine steps drawn per turn from a lognormal of median
`think` and shape `sigma`, from the conversation's own RNG, so a few conversations wait far longer than the rest (a long
tool call). With `edits` > 0, each later turn is with that probability an edit instead: the previous reply and tool
result are dropped and another tool result follows the turn before (an edited context), so it can resume only at the
checkpoint the previous turn resumed from. With `retries` > 0, each later turn is with that probability a retry instead:
the previous turn's prompt again (its reply regenerated), which can resume only at that turn's end-of-prefill
checkpoint. Each later prompt continues the previous one token for token, the reply as generated, as a client does that
sends the reply back unchanged; a chat client whose template re-renders the reply (reasoning dropped, tool calls
re-serialized) diverges inside it, the finish checkpoint then serves nothing and the turn resumes at the previous
prompt's end-of-prefill checkpoint. `edits` and `retries` are per-turn probabilities among the later turns where they
apply (an edit needs two finished turns and is drawn only when the turn is not a retry), so the realized rates are
lower; the readout gives each kind's count. Tokens come from the conversation's RNG; the replies are the fake runner's.
The engine steps while a request is in it; when none is, time jumps to the next arrival.

**The models** (`--model`): `hybrid`, the tiny Qwen3.5's geometry (`tests/support/tiny_qwen.py`: GDN state slots, a
64-token grid), its checkpoints in slots sized by `dev.cache.checkpoint_slots` (`--slots`); `window`, the tiny Gemma
4's (`tests/support/tiny_gemma.py`, its window set by `--window`), its window checkpoints in the KV arena, sized by its
pages (`--pages`). The KV pool is otherwise large enough for every conversation (`--kv-pages` sets it), so only
checkpoint pressure evicts.

**What it reads out**, per configuration: the hit rate (Σ cached / Σ prompt tokens, over every turn and over the later
turns); the uncached tokens per turn (p50, p90, max) by turn index; the scheduler's lost tokens by cause; checkpoint
evictions by source and by whether they served a resume; checkpoints taken and resumed from by source; and the turns
that resumed below the position their conversation's history had reached (the previous turn's end, or for an edit the
end of the turn it follows), with how far below; retries are left out of that count (their end-of-prefill checkpoint
is not modelled) and read out with the edits as their own hit rates.

**The eviction order** (`--order`): `engine`, the scheduler's default (its own tree's order); `lru`, the scheduler
with `dev.cache.demote_superseded` off (every checkpoint least recently used first); `leaf-last`, a comparison that
replaces the tree's checkpoint eviction in this process only: the inner checkpoints (a deeper checkpoint below them, and
no branching: one child, at most one resume since attached) and the leaves unused for `--idle` steps of the workload's
clock go first, then the rest, least recently used first within each tier.

Each configuration's readout goes to a JSON file in `--out` named `<label>-<model>-<order>` (by default in the results
folder's `multi_turn`, `tests.support.paths.RESULTS`) and a table to stdout."""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODELS = ("hybrid", "window")
ORDERS = ("engine", "lru", "leaf-last")
EOS_FREE = (0,)  # stop ids: replies ignore EOS


@dataclass(frozen=True)
class Workload:
    sessions: int = 16
    turns: int = 8
    prefix: int = 512  # the shared system prefix
    tool: int = 192  # the first message, and each tool result
    reply: int = 64
    think: float = 150.0  # median think time, in engine steps
    sigma: float = 1.2  # the think times' lognormal shape
    edits: float = 0.0  # chance that a later turn edits instead of extending (module docstring)
    retries: float = 0.0  # chance that a later turn retries the previous one (module docstring)
    seed: int = 1


@dataclass(frozen=True)
class Engine:
    model: str = "hybrid"
    max_num_seqs: int = 8
    budget: int = 1024  # max_num_batched_tokens
    page_size: int = 16  # hybrid: the KV page; window: the window page (the grid)
    slots: int | None = None  # hybrid: dev.cache.checkpoint_slots (None: the engine's default)
    pages: int | None = None  # the KV pool (window: the arena); None: room for every conversation
    window: int = 128  # the window model's sliding window
    decode_interval: int = 256  # dev.cache.decode_checkpoint_interval
    order: str = "engine"
    idle: int = 600  # leaf-last: steps after which an unused leaf counts as inner


def lognormal_steps(rng: np.random.Generator, median: float, sigma: float) -> int:
    return max(1, round(median * math.exp(sigma * rng.standard_normal())))


@dataclass
class Turn:
    session: int
    turn: int
    prompt: int
    cached: int = -1
    ideal: int = 0  # the deepest position its conversation's history reached that the prompt still extends
    edit: bool = False
    retry: bool = False
    arrive: int = 0
    first: int = -1  # step of its first token
    done: int = -1


@dataclass
class Conversation:
    """One conversation's state: its history (token ids) and the computed ends of its turns, in history order."""

    index: int
    rng: np.random.Generator
    history: list[int] = field(default_factory=list)
    ends: list[tuple[int, int]] = field(default_factory=list)  # (history length after the turn's reply, its end)
    turn: int = 0
    rid: int = -1  # the request in flight
    prompt: int = 0  # the length of its last prompt


def model_dir(model: str, window: int, root: Path) -> Path:
    """The tiny model's directory: its config.json gives the fake runner its geometry (`runner.fake.model_geometry`)."""
    if model == "hybrid":
        from ..support.tiny_qwen import make_tiny_model

        return make_tiny_model(root / "hybrid")
    from ..support.tiny_gemma import make_tiny_gemma

    return make_tiny_gemma(root / f"window{window}", window=window)


def build(eng: Engine, mdir: Path, max_model_len: int):
    """The fake runner over the tiny model's geometry, its pools sized as `engine/process.py` sizes them (the
    checkpoint slots or the arena's pages from `eng`), and the scheduler and engine core built as the engine builds
    them. Returns (core, scheduler, runner, the pools' numbers)."""
    from llm_mini_infer.cache.pools import size_arena, size_pools
    from llm_mini_infer.config import FakeRunnerConfig
    from llm_mini_infer.engine.core import EngineCore
    from llm_mini_infer.runner.base import PoolConfig
    from llm_mini_infer.runner.fake import make_fake_runner
    from llm_mini_infer.sched.plan import reserve_positions
    from llm_mini_infer.sched.scheduler import Scheduler

    runner = make_fake_runner(mdir, FakeRunnerConfig(step_ms=0.0, memory_gb=64.0), 0)
    caps = runner.capabilities()
    reserve = reserve_positions(caps.max_steps_in_flight, caps.mtp_depth)
    grid, wps = caps.checkpoint_grid, 0
    if caps.window:
        grid = wps = eng.page_size
        sizes = size_arena(memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                           window_kv_bytes_per_token=caps.window_kv_bytes_per_token, window=caps.window,
                           window_page_size=wps, max_num_seqs=eng.max_num_seqs, max_model_len=max_model_len,
                           max_num_batched_tokens=eng.budget, reserve_tokens=reserve, num_pages=eng.pages)
        num_pages, page_size, num_states, live, slots = sizes.num_pages, sizes.page_size, eng.max_num_seqs + 1, \
            eng.max_num_seqs, 0
    else:
        sizes = size_pools(memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                           state_slot_bytes=caps.state_slot_bytes, page_size=eng.page_size,
                           max_num_seqs=eng.max_num_seqs, max_model_len=max_model_len, num_pages=eng.pages,
                           checkpoint_slots=eng.slots, reserve_tokens=reserve)
        num_pages, page_size, num_states, live, slots = sizes.num_pages, eng.page_size, sizes.num_state_slots, \
            sizes.num_live_slots, sizes.num_checkpoint_slots
    runner.configure(PoolConfig(page_size=page_size, num_pages=num_pages, num_state_slots=num_states,
                                max_num_seqs=eng.max_num_seqs, max_num_batched_tokens=eng.budget,
                                max_model_len=max_model_len, eos_token_ids=EOS_FREE, mtp_k=caps.mtp_depth,
                                window_page_size=wps))
    sched = Scheduler(max_num_seqs=eng.max_num_seqs, max_num_batched_tokens=eng.budget, max_model_len=max_model_len,
                      page_size=page_size, num_pages=num_pages, num_state_slots=num_states, eos_token_ids=EOS_FREE,
                      steps_in_flight=caps.max_steps_in_flight, mtp_k=caps.mtp_depth, seed=0, num_live_slots=live,
                      hybrid=caps.state_slot_bytes > 0, prefix_caching=True, checkpoint_grid=grid,
                      decode_checkpoint_interval=eng.decode_interval, demote_superseded=eng.order != "lru",
                      window=caps.window, window_page_size=wps,
                      draft_kv=caps.draft_kv, logprobs=caps.logprobs)
    pools = dict(num_pages=int(num_pages), page_size=int(page_size), checkpoint_slots=int(slots),
                 window_page_size=int(wps), grid=int(grid), vocab=int(caps.vocab_size))
    return EngineCore(sched, runner), sched, runner, pools


class LeafLast:
    """The `leaf-last` comparison (module docstring): replaces the tree's `evict_checkpoints` on one RadixCache. A
    checkpoint is inner when a node below its own holds a checkpoint and its node branches nowhere (one child, at most
    one resume since attached), or when it is a leaf unused for `idle` steps of `clock` (the workload's, which also
    runs while the engine is empty, as wall time does); inner ones go first, then the rest, least recently used first in
    each tier. The plan's pinned and in-flight checkpoints, and `spare_locked`, are kept as the tree keeps them. Each
    eviction scans the tree's checkpoints: for measurement only."""

    def __init__(self, sched, idle: int, clock):
        self.sched, self.cache, self.idle, self.clock = sched, sched.cache, idle, clock
        self.used: dict[int, int] = {}  # node id -> the time of its checkpoint's last use (attached, or a hit)
        cache = self.cache
        orig_match, orig_attach = cache.match, cache.attach

        def match(*a, **kw):
            m = orig_match(*a, **kw)
            if m.node is not None and m.node.ckpt:
                self.used[m.node.id] = clock()
            return m

        def attach(node, *a, **kw):
            orig_attach(node, *a, **kw)
            self.used[node.id] = clock()

        cache.match, cache.attach, cache.evict_checkpoints = match, attach, self.evict_checkpoints

    def _nodes(self) -> list:
        out, stack = [], list(self.cache.roots.values())
        while stack:
            n = stack.pop()
            if n.parent is not None and n.ckpt:
                out.append(n)
            stack.extend(n.children.values())
        return out

    @staticmethod
    def _deeper(node) -> bool:
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            if n.ckpt:
                return True
            stack.extend(n.children.values())
        return False

    def tier(self, node) -> int:
        """0: inner (evicted first), 1: leaf."""
        if not self._deeper(node):
            return 0 if self.clock() - self.used.get(node.id, 0) > self.idle else 1
        return 1 if len(node.children) > 1 or node.served > 1 else 0

    def evict_checkpoints(self, n: int, step: int, *, spare_locked: bool = False, enough=None) -> int:
        cache, done = self.cache, 0
        while done < n and not (enough is not None and enough()):
            best = None
            for node in self._nodes():
                if node.ckpt_ready >= step or node.ckpt in cache._pinned or (spare_locked and node.lock):
                    continue
                key = (self.tier(node), node.ckpt_used, node.id)
                if best is None or key < best[0]:
                    best = (key, node)
            if best is None:
                break
            cache._evicting(best[1])
            cache._drop_ckpt(best[1])
            done += 1
        return done


def run(wl: Workload, eng: Engine, mdir: Path) -> dict:
    """One configuration (module docstring)."""
    from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, AddRequest

    max_len = wl.prefix + wl.turns * (wl.tool + wl.reply) + wl.tool + 64
    core, sched, _, pools = build(eng, mdir, max_len)
    clock = [0]  # the workload's time in steps, `now` below
    if eng.order == "leaf-last":
        LeafLast(sched, eng.idle, lambda: clock[0])
    vocab = pools["vocab"]
    base = np.random.default_rng([wl.seed, 0])
    prefix = base.integers(vocab // 4, 3 * vocab // 4, wl.prefix).tolist()
    convs = [Conversation(i, np.random.default_rng([wl.seed, 1, i]), list(prefix)) for i in range(wl.sessions)]
    turns: dict[int, Turn] = {}
    arrivals: list[tuple[int, int]] = [(0, i) for i in range(wl.sessions)]  # (step, conversation)
    heapq.heapify(arrivals)
    out: dict[int, list[int]] = {}
    now, next_rid, t0 = 0, 0, time.perf_counter()

    def send(c: Conversation) -> None:
        nonlocal next_rid
        rng = c.rng
        retry = c.turn >= 1 and wl.retries > 0 and rng.random() < wl.retries
        edit = not retry and len(c.ends) >= 2 and wl.edits > 0 and rng.random() < wl.edits
        if retry:  # the previous prompt again: its reply is dropped
            c.history = c.history[:c.prompt]
            c.ends = c.ends[:-1]
        elif edit:  # drop the previous reply and tool result: back to the end of the turn before
            keep = c.ends[-2][0]
            c.history = c.history[:keep]
            c.ends = c.ends[:-1]
        if not retry:
            c.history += rng.integers(vocab // 4, 3 * vocab // 4, wl.tool).tolist()
        c.prompt = len(c.history)
        ideal = 0
        for length, end in c.ends:  # the deepest computed end on this history
            if length <= len(c.history):
                ideal = max(ideal, end)
        rid = next_rid
        next_rid += 1
        c.rid = rid
        turns[rid] = Turn(c.index, c.turn, len(c.history), ideal=0 if retry else ideal, edit=edit, retry=retry,
                          arrive=now)
        core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(c.history, np.int32), max_tokens=wl.reply,
                               temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=rid, stop_token_ids=[],
                               ignore_eos=True))

    by_rid = {}
    while arrivals or core.busy:
        while arrivals and arrivals[0][0] <= now:
            _, i = heapq.heappop(arrivals)
            send(convs[i])
            by_rid[convs[i].rid] = convs[i]
        if not core.busy:
            now = clock[0] = arrivals[0][0]
            continue
        rep = core.step()
        now += 1
        clock[0] = now
        for fo in (rep.outputs.values() if rep is not None else ()):
            for rid, cached in zip(fo.first_rids.tolist(), fo.first_cached.tolist()):
                turns[rid].cached = int(cached)
                turns[rid].first = now
            at = 0
            for rid, n, fin in zip(fo.rids.tolist(), fo.n_new.tolist(), fo.finish.tolist()):
                out.setdefault(rid, []).extend(int(x) for x in fo.tokens[at:at + n])
                at += n
                if not fin:
                    continue
                t = turns[rid]
                t.done = now
                c = by_rid.pop(rid)
                reply = out.pop(rid)
                end = len(c.history) + len(reply) - 1  # its last token is not computed
                c.history += reply
                c.ends.append((len(c.history), end))
                c.turn += 1
                if c.turn < wl.turns:
                    heapq.heappush(arrivals, (now + lognormal_steps(c.rng, wl.think, wl.sigma), c.index))
        if now > 50_000_000:
            raise RuntimeError("the workload did not finish")
    sched.check_accounting()
    if sched.cache is not None:
        sched.cache.check()
    return readout(wl, eng, pools, sched, turns, now, time.perf_counter() - t0, CHECKPOINT_SOURCES)


def pct(xs: list[int], q: float) -> int:
    return int(np.percentile(xs, q, method="lower")) if xs else 0


def readout(wl: Workload, eng: Engine, pools: dict, sched, turns: dict[int, Turn], steps: int, wall: float,
            sources: tuple[str, ...]) -> dict:
    ts = sorted(turns.values(), key=lambda t: (t.session, t.turn))
    if any(t.cached < 0 for t in ts):
        raise RuntimeError("a turn reported no first token")
    later = [t for t in ts if t.turn > 0]

    def rate(xs: list[Turn]) -> float:
        p = sum(t.prompt for t in xs)
        return round(sum(t.cached for t in xs) / p, 4) if p else 0.0

    by_turn = {}
    for k in range(wl.turns):
        u = [t.prompt - t.cached for t in ts if t.turn == k]
        by_turn[k] = dict(n=len(u), p50=pct(u, 50), p90=pct(u, 90), max=max(u, default=0))
    below = [t.ideal - t.cached for t in later if t.cached < t.ideal]  # retries left out: their ideal is 0

    def kind(xs: list[Turn]) -> dict:
        return dict(n=len(xs), hit_rate=rate(xs), cached=sum(t.cached for t in xs), prompt=sum(t.prompt for t in xs),
                    below=sum(t.cached < t.ideal for t in xs))
    cache = sched.cache
    ev = {s: dict(unserved=cache.evicted_by[i][0], served=cache.evicted_by[i][1]) for i, s in enumerate(sources)
          if any(cache.evicted_by[i])}
    uncached_later = [t.prompt - t.cached for t in later]
    return dict(
        workload=asdict(wl), engine=asdict(eng), pools=pools, steps=steps, wall_s=round(wall, 2),
        requests=len(ts), edits=kind([t for t in ts if t.edit]), retries=kind([t for t in ts if t.retry]),
        hit_rate=rate(ts), hit_rate_later=rate(later), cached=sum(t.cached for t in ts),
        prompt=sum(t.prompt for t in ts), cached_later=sum(t.cached for t in later),
        prompt_later=sum(t.prompt for t in later),
        uncached_later=dict(p50=pct(uncached_later, 50), p90=pct(uncached_later, 90), max=max(uncached_later,
                                                                                               default=0),
                            sum=sum(uncached_later)),
        uncached_by_turn=by_turn,
        below=dict(n=len(below), of=sum(not t.retry for t in later), p50=pct(below, 50), max=max(below, default=0),
                   sum=sum(below)),
        lost=dict(sched.lost_tokens),
        evicted=ev, evicted_total=cache.evicted_ckpts,
        taken={s: n for s, n in sched.ckpts_taken.items() if n},
        resumes={s: n for s, n in zip(sources, sched.ckpt_resumes) if n},
        skipped=sched.ckpts_skipped, waits=sched.cache_waits, preemptions=sched.num_preemptions,
        evicted_pages=cache.evicted_pages, evicted_window_pages=cache.evicted_window_pages,
    )


def budget(r: dict) -> str:
    return (f"{r['pools']['checkpoint_slots']} slots" if r["engine"]["model"] == "hybrid"
            else f"{r['pools']['num_pages']} pages")


def table(rows: list[dict]) -> str:
    """One line per budget, summed over its seeds (the hit rates: Σ cached / Σ prompt over all of them)."""
    head = ("| budget | seeds | hit rate | later turns | uncached p50 / p90 (median seed) | below: turns, tokens | "
            "lost to checkpoints | evicted finish (served) | evicted others (served) | edits: hit rate | "
            "retries: hit rate |")
    lines = [head, "|" + "---|" * 11]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(budget(r), []).append(r)
    for b, rs in groups.items():
        cached = sum(r["cached"] for r in rs)
        prompt = sum(r["prompt"] for r in rs)
        cached_l = sum(r["cached_later"] for r in rs)
        prompt_l = sum(r["prompt_later"] for r in rs)
        mid = sorted(rs, key=lambda r: r["hit_rate"])[(len(rs) - 1) // 2]  # the lower median
        fin = [r["evicted"].get("finish", {}) for r in rs]
        oth = [v for r in rs for k, v in r["evicted"].items() if k != "finish"]

        def kind(name: str, rs: list[dict] = rs) -> str:  # turns of that kind over the seeds, and their hit rate
            n = sum(r[name]["n"] for r in rs)
            if not n:
                return "-"
            hit = sum(r[name]["cached"] for r in rs) / sum(r[name]["prompt"] for r in rs)
            return f"{n}, {hit:.3f}"

        lines.append(
            f"| {b} | {len(rs)} | {cached / prompt:.3f} | {cached_l / prompt_l:.3f} | "
            f"{mid['uncached_later']['p50']} / {mid['uncached_later']['p90']} | "
            f"{sum(r['below']['n'] for r in rs)} of {sum(r['below']['of'] for r in rs)}, "
            f"{sum(r['below']['sum'] for r in rs):,} | {sum(r['lost'].get('checkpoint', 0) for r in rs):,} | "
            f"{sum(f.get('unserved', 0) + f.get('served', 0) for f in fin)} ({sum(f.get('served', 0) for f in fin)}) | "
            f"{sum(v['unserved'] + v['served'] for v in oth)} ({sum(v['served'] for v in oth)}) | {kind('edits')} | "
            f"{kind('retries')} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ..support.paths import RESULTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="hybrid")
    ap.add_argument("--slots", type=int, nargs="+", help="hybrid: dev.cache.checkpoint_slots values to sweep")
    ap.add_argument("--pages", type=int, nargs="+", help="window: the arena's pages to sweep")
    ap.add_argument("--kv-pages", type=int, help="hybrid: the KV pool's pages (default: room for every conversation)")
    ap.add_argument("--order", choices=ORDERS, default="engine")
    ap.add_argument("--idle", type=int, default=Engine.idle)
    ap.add_argument("--max-num-seqs", type=int, default=Engine.max_num_seqs)
    ap.add_argument("--budget", type=int, default=Engine.budget)
    ap.add_argument("--window", type=int, default=Engine.window)
    ap.add_argument("--seeds", type=int, nargs="+", default=[Workload.seed], help="one run per seed and budget")
    for f in ("sessions", "turns", "prefix", "tool", "reply"):
        ap.add_argument(f"--{f}", type=int, default=getattr(Workload, f))
    for f in ("think", "sigma", "edits", "retries"):
        ap.add_argument(f"--{f}", type=float, default=getattr(Workload, f))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path, default=RESULTS / "multi_turn")
    a = ap.parse_args(argv)
    wl = Workload(sessions=a.sessions, turns=a.turns, prefix=a.prefix, tool=a.tool, reply=a.reply, think=a.think,
                  sigma=a.sigma, edits=a.edits, retries=a.retries)
    if a.model == "hybrid":
        ctx = wl.sessions * (wl.prefix + wl.turns * (wl.tool + wl.reply) + wl.tool + 64)
        kv = a.kv_pages or 2 * ctx // Engine.page_size + 64
        engines = [Engine(model="hybrid", max_num_seqs=a.max_num_seqs, budget=a.budget, slots=s, pages=kv,
                          order=a.order, idle=a.idle) for s in (a.slots or [None])]
    else:
        engines = [Engine(model="window", max_num_seqs=a.max_num_seqs, budget=a.budget, pages=p, window=a.window,
                          order=a.order, idle=a.idle) for p in (a.pages or [None])]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        mdir = model_dir(a.model, a.window, Path(tmp))
        for eng in engines:
            for seed in a.seeds:
                r = run(dataclasses.replace(wl, seed=seed), eng, mdir)
                rows.append(r)
                print(f"{a.model} {eng.order} seed {seed}: {table([r]).splitlines()[-1]}  ({r['steps']} steps, "
                      f"{r['wall_s']} s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"{a.label}-{a.model}-{a.order}"
    dest = a.out / f"{name}.json"
    dest.write_text(json.dumps(rows, indent=1) + "\n")
    print(table(rows))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

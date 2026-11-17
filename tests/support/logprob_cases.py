"""The logprobs tests' requests and runs on the tiny models (tests/runner/test_cuda_logprobs.py,
tests/runner/test_logprobs_gpu.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.sched.scheduler import Scheduler

from .fake_grammar import FakeGrammar, spec
from .tiny_gemma import gemma_cpu_runner, gemma_pools
from .tiny_qwen import cpu_runner, pools

EOS = 511
VOCAB = 512


def seq(n: int, salt: int = 0) -> list[int]:
    return [(i * 37 + 11 + salt * 97) % 500 for i in range(n)]


REQS = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 4, 9, lead=2, tail=1), logprobs=5),
        dict(rid=1, prompt=seq(33, 1), logprobs=20, temperature=0.8, seed=5),
        dict(rid=2, prompt=seq(9, 2), grammar=spec(2, 30, 40, one_in=4), max_tokens=12, logprobs=0),
        dict(rid=3, prompt=seq(50, 3), grammar=spec(3, 0, 6)),
        dict(rid=4, prompt=seq(14, 4), grammar=spec(4, 2, 10, one_in=3), logprobs=3, temperature=1.0, top_p=0.9)]


def run(kind: str, model: Any, k: int, reqs: list[dict], logprobs: bool, counters: dict | None = None,
        make: Any = None, hook: bool = True, window: int = 8, wps: int = 4, budget: int = 64,
        **kw: Any) -> tuple[dict, dict]:
    """`reqs` through the scheduler and engine loop until idle, with each one's `logprobs` (or none: `logprobs`
    False): ({rid: (tokens, finish, values)}, {(rid, position): the logits there, the last computed; `hook`});
    `counters`: gets the runner's head counters (`heads`); `make(model, pools, **kw)`: the runner (default: on CPU)."""
    max_num_seqs = 4
    if kind == "qwen":
        pc = pools(max_num_seqs=max_num_seqs, budget=budget, page_size=wps, mtp_k=k, num_pages=512)
        r = (make or cpu_runner)(model, pc, logprobs=True, **kw)
        extra: dict = {}
    else:
        pc = gemma_pools(max_num_seqs=max_num_seqs, budget=budget, page_size=wps, mtp_k=k)
        r = (make or gemma_cpu_runner)(model[0], pc, logprobs=True, mtp_k=k, assistant=model[1] if k else None, **kw)
        extra = dict(window=window, window_page_size=pc.window_page_size, draft_kv=False, checkpoint_grid=wps)
    grammar = FakeGrammar(VOCAB, (EOS,))
    sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=pc.max_model_len,
                      page_size=pc.page_size, num_pages=pc.num_pages, num_state_slots=pc.num_state_slots,
                      eos_token_ids=(EOS,), mtp_k=k, grammars=Grammars(grammar), forced_prefix=True,
                      eos_free_finish=True, logprobs=True, batch_invariant=r.capabilities().batch_invariant, **extra)
    core = EngineCore(sched, r)
    logits: dict[tuple[int, int], torch.Tensor] = {}
    rids: list[int] = []
    orig = r.launch_forward

    def launch(plan: Any, *a: Any, **kw: Any) -> Any:
        rids[:] = [sched.slot_req[s].rid for s in plan.sched_slot.tolist()]
        return orig(plan, *a, **kw)

    def on_logits(chunk: Any) -> None:
        lg = chunk.logits.detach().to("cpu", torch.float32)
        for i, (row, pos) in enumerate(zip(chunk.rows.tolist(), chunk.positions.tolist())):
            logits[(rids[row], pos)] = lg[i].clone()

    r.launch_forward = launch
    if hook:
        r.logits_hook = on_logits
    out: dict[int, tuple[list[int], int, list]] = {q["rid"]: ([], 0, []) for q in reqs}
    asked = {q["rid"] for q in reqs if logprobs and q.get("logprobs", -1) >= 0}
    for q in reqs:
        core.handle(AddRequest(rid=q["rid"], frontend=0, prompt=np.asarray(q["prompt"], np.int32),
                               max_tokens=q.get("max_tokens", 14), temperature=q.get("temperature", 0.0),
                               top_p=q.get("top_p", 1.0), top_k=0, min_p=0.0, seed=q.get("seed", 100 + q["rid"]),
                               stop_token_ids=[], grammar=q.get("grammar", ""),
                               logprobs=q.get("logprobs", -1) if logprobs else -1))
    n = 0
    while core.busy and n < 2000:
        rep = core.step()
        n += 1
        if rep is None:
            continue
        for o in rep.outputs.values():
            off = 0
            for rid, nn, f in zip(o.rids.tolist(), o.n_new.tolist(), o.finish.tolist()):
                toks, fin, vals = out[rid]
                toks.extend(o.tokens[off:off + nn].tolist())
                if rid in asked and nn:  # the message's values are aligned with every request's tokens
                    assert o.logprobs is not None
                    vals.extend((float(o.logprobs[0][j]), o.logprobs[1][j].tolist(), o.logprobs[2][j].tolist())
                                for j in range(off, off + nn))
                off += nn
                out[rid] = (toks, f or fin, vals)
    assert not core.busy and sched.grammars.errors == 0
    if counters is not None:
        counters["heads"] = r.head_counters()
    r.shutdown()
    return out, logits

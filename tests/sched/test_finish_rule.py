"""One finish rule (`Scheduler._finish_at`): a held sample's tokens (`_deliver_held`: a preempted request, waiting, with
no slot) and an MTP row's (`_cut`: a running request's slot) finish at the same token for the same reason, over random
rows of tokens, output counts, `max_tokens`, stop ids, EOS ids and ignore_eos."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from llm_mini_infer.engine.messages import AddRequest, FinishReason
from llm_mini_infer.sched.scheduler import Req, Scheduler

EOS = (7, 9)
K = 3
SLOT, AID = 1, 5


def scheduler() -> Scheduler:
    return Scheduler(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=1 << 12, page_size=4, num_pages=64,
                     num_state_slots=4, eos_token_ids=EOS, mtp_k=K, prefix_caching=False)


def draw(rnd: random.Random) -> tuple[list[int], int, int, list[int], bool]:
    """A row: its tokens (EOS ids, stop-id candidates and others), the request's output count before it (past the
    ring's width at times, so the ring wraps), its `max_tokens`, its stop ids and ignore_eos."""
    toks = [rnd.choice((7, 9, 11, 12, 13, 20, 21, 22)) for _ in range(rnd.randint(1, 1 + K))]
    n_out = rnd.randint(0, 600)
    return toks, n_out, n_out + rnd.randint(1, 5), rnd.sample((7, 11, 12, 13), rnd.randint(0, 3)), rnd.random() < 0.3


def padded(toks: list[int]) -> np.ndarray:
    row = np.zeros((1, 1 + K), np.int32)
    row[0, :len(toks)] = toks
    return row


def by_cut(s: Scheduler, toks: list[int], n_out: int, max_out: int, stop_ids: list[int],
           ignore_eos: bool) -> tuple[int, int]:
    """A running request's row through `_cut`: (the tokens it delivers, its finish reason)."""
    s.n_out[SLOT], s.max_out[SLOT], s.ignore_eos[SLOT] = n_out, max_out, ignore_eos
    s.slot_req[SLOT] = SimpleNamespace(out_chunks=[])
    s._set_stop_ids(SLOT, np.asarray(stop_ids, np.int32))
    cut, _, finish = s._cut(np.array([SLOT]), padded(toks), np.array([len(toks)], np.int32))
    s.stop_ids[SLOT] = -1
    s._n_stop_reqs = 0
    s.slot_req[SLOT] = None
    return int(cut[0]), int(finish[0])


def by_held(s: Scheduler, toks: list[int], n_out: int, max_out: int, stop_ids: list[int],
            ignore_eos: bool) -> tuple[int, int]:
    """A preempted request's held sample through `_deliver_held`: (the tokens it delivers, its finish reason)."""
    msg = AddRequest(rid=0, frontend=0, prompt=np.arange(1, 5, dtype=np.int32), max_tokens=max_out, temperature=1.0,
                     top_p=1.0, top_k=0, min_p=0.0, seed=1, stop_token_ids=stop_ids, ignore_eos=ignore_eos)
    req = Req(msg, 1, 0, s.max_model_len)
    req.n_out, req.first_done, req.held = n_out, True, AID
    s._held[AID] = req
    s.num_waiting += 1  # it waits: a finish counts it out
    inf = SimpleNamespace(sample=np.array([True]), aids=np.array([AID]), device=np.array([False]))
    s._held_rows.clear()
    assert s._deliver_held(inf, padded(toks), np.array([len(toks)], np.int32), np.array([False]), None, None, 0) == 1
    _, _, new, finish, *_ = s._held_rows[0]
    return len(new), finish


def oracle(toks: list[int], n_out: int, max_out: int, stop_ids: list[int], ignore_eos: bool) -> tuple[int, int]:
    """The per-token loop `_deliver_held` ran before `_finish_at`: a token finishes STOP if it is an EOS (unless
    ignore_eos) or a stop id, else LENGTH when the output count reaches max_out; the first finish ends the row."""
    for i, t in enumerate(toks, 1):
        if (t in EOS and not ignore_eos) or t in stop_ids:
            return i, int(FinishReason.STOP)
        if n_out + i >= max_out:
            return i, int(FinishReason.LENGTH)
    return len(toks), 0


def test_a_held_sample_and_a_running_row_finish_alike() -> None:
    rnd = random.Random(20261109)
    s = scheduler()
    finishes = set()
    seen_out = exact = False
    for _ in range(3000):
        row = draw(rnd)
        got = by_cut(s, *row)
        assert by_held(s, *row) == got, row
        assert got == oracle(*row), row
        finishes.add(got[1])
        seen_out |= row[1] > 0
        exact |= got[1] == FinishReason.LENGTH and row[1] + got[0] == row[2] and got[0] == len(row[0])
    assert len(finishes) == 3  # none, STOP and LENGTH all drawn
    assert seen_out and exact  # rows with output, and a max_tokens reached exactly at the last token


def test_an_unknown_unverified_disposition_raises() -> None:
    s = scheduler()
    with pytest.raises(RuntimeError, match="99"):
        s._settle_unverified(SimpleNamespace(unv=0), 99)  # type: ignore[arg-type]
    s._settle_unverified(SimpleNamespace(unv=-1), 99)  # type: ignore[arg-type]  # nothing unverified

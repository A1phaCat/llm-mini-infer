"""`Scheduler.check_accounting`'s checks of the slots' phases, the flags' counts and the waiting heap: a scheduler
stopped between two turns, with one slot decoding, one still prefilling and requests waiting, then one field
corrupted. Each check raises in both modes, the heap's walk only with `full`."""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np
import pytest

from llm_mini_infer.sched.scheduler import WAITING, Scheduler

from ..support.harness import Harness, Spec


def stopped() -> Harness:
    """Two slots: request 0 decoding, request 1 prefilling (its chunks capped); requests 2 and 3 waiting."""
    h = Harness(specs=[], max_num_seqs=2, chunk_cap=8, max_tokens=24)
    h.specs = {s.rid: s for s in (Spec(rid=0, prompt_len=6, max_tokens=40, seed=1),
                                  Spec(rid=1, prompt_len=60, max_tokens=40, seed=2),
                                  Spec(rid=2, prompt_len=6, max_tokens=4, seed=3),
                                  Spec(rid=3, prompt_len=6, max_tokens=4, seed=4))}
    for spec in h.specs.values():
        h.add(spec)
    for _ in range(3):
        rep = h.core.step()
        if rep is not None:
            h.collect(rep)
    s = h.sched
    running = [r for r in s.slot_req if r is not None]
    assert {r.rid for r in running} == {0, 1} and s.num_waiting == 2
    assert s.decoding[s.reqs[(0, 0)].slot] and not s.decoding[s.reqs[(0, 1)].slot]
    for full in (False, True):
        s.check_accounting(full)
    return h


def free_slot(s: Scheduler) -> int:
    return int(np.flatnonzero(s.aid == 0)[0])


def corrupt_decoding_left_to_prefill(s: Scheduler) -> None:
    s.num_sched[s.reqs[(0, 0)].slot] -= 1


def corrupt_pending_on_a_free_slot(s: Scheduler) -> None:
    s.pending[free_slot(s)] = 1


def corrupt_inflight_while_prefilling(s: Scheduler) -> None:
    s.dev_inflight[s.reqs[(0, 1)].slot] = 1


def corrupt_a_free_slot_decoding(s: Scheduler) -> None:
    s.decoding[free_slot(s)] = True


def corrupt_a_free_slot_masked(s: Scheduler) -> None:
    s.masked[free_slot(s)] = True
    s.n_masked += 1


def corrupt_a_free_slot_with_logprobs(s: Scheduler) -> None:
    s.lp[free_slot(s)] = 2
    s.n_lp += 1


def corrupt_masked_count(s: Scheduler) -> None:
    s.n_masked += 1


def corrupt_logprobs_count(s: Scheduler) -> None:
    s.lp[s.reqs[(0, 0)].slot] = 2  # set without its count


BOTH: dict[str, tuple[Callable[[Scheduler], None], str]] = {
    "decoding left to prefill": (corrupt_decoding_left_to_prefill, "a decoding slot left to prefill"),
    "pending on a free slot": (corrupt_pending_on_a_free_slot, "rows in flight for a slot that is not decoding"),
    "device rows while prefilling": (corrupt_inflight_while_prefilling,
                                     "rows in flight for a slot that is not decoding"),
    "a free slot decoding": (corrupt_a_free_slot_decoding, "a free slot holds a request's state"),
    "a free slot masked": (corrupt_a_free_slot_masked, "a free slot holds a request's state"),
    "a free slot with logprobs": (corrupt_a_free_slot_with_logprobs, "a free slot holds a request's state"),
    "masked count": (corrupt_masked_count, "masked-slot count out of sync"),
    "logprobs count": (corrupt_logprobs_count, "logprobs-slot count out of sync"),
}


@pytest.mark.parametrize("full", [False, True], ids=["default", "full"])
@pytest.mark.parametrize("fault", sorted(BOTH))
def test_a_slot_check_fails_in_both_modes(fault: str, full: bool) -> None:
    h = stopped()
    corrupt, message = BOTH[fault]
    corrupt(h.sched)
    with pytest.raises(AssertionError, match=message):
        h.sched.check_accounting(full)


def corrupt_waiting_count(s: Scheduler) -> None:
    s.num_waiting += 1


def corrupt_a_request_queued_twice(s: Scheduler) -> None:
    entry = next(e for e in s._waiting if e[2].status == WAITING)
    heapq.heappush(s._waiting, entry)
    s.num_waiting += 1


def corrupt_a_waiting_request_lost(s: Scheduler) -> None:
    lost = next(e for e in s._waiting if e[2].status == WAITING)
    s._waiting.remove(lost)
    heapq.heapify(s._waiting)
    s.num_waiting -= 1


HEAP: dict[str, tuple[Callable[[Scheduler], None], str]] = {
    "waiting count": (corrupt_waiting_count, "waiting requests out of sync"),
    "queued twice": (corrupt_a_request_queued_twice, "a request is in the waiting heap twice"),
    "lost from the heap": (corrupt_a_waiting_request_lost, "waiting heap out of sync"),
}


@pytest.mark.parametrize("fault", sorted(HEAP))
def test_a_waiting_heap_check_fails_in_the_full_mode(fault: str) -> None:
    h = stopped()
    corrupt, message = HEAP[fault]
    corrupt(h.sched)
    h.sched.check_accounting(False)  # the default leaves the heap's walk out
    with pytest.raises(AssertionError, match=message):
        h.sched.check_accounting(True)

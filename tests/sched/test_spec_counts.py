"""The speculative-decoding counters against what the fake runner verified, and per-request acceptance.

`vllm:spec_decode_num_drafts` counts one request's verify step, `_num_accepted_tokens` its accepted drafts (as
in vLLM: a preempted request's step in flight counts too). The verify rows whose results a request
gets are those of running requests and the held rows of requests preempted while the row was in flight
(`Scheduler._deliver_held`); both count. Every output token after a request's first is then a verify row's, a forced
prefix's, the EOS an EOS-free finish appends or a readmission's first sample: `Harness.check_spec_counts` reconciles
them on every harness run, the fuzz suites' included. Under load the EOS-free finish appends its EOS to requests whose
row is in flight (one at a time the gate leaves it to that row), so drafts + accepted drafts + forced tokens fall short
of the tokens after the first by exactly `llm_mini_infer:grammar_eos_free_finishes_total`.

Per request (`Scheduler.verify_rows`, `verify_tokens`, folded into the request at preemption): its verify rows and the
tokens they delivered, reported with its finish; the frontend observes their ratio in
llm_mini_infer:request_tokens_per_verify_step and writes both on the request log line (tests/api/test_api.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from llm_mini_infer.sched.scheduler import WAITING

from ..support.fake_grammar import spec as grammar_spec
from ..support.forced_cases import Probe, make, run
from ..support.harness import Harness, Spec
from ..support.sched_cases import stem


def verified_rows(h: Harness) -> dict[str, Any]:
    """Recompute the counters from each step's results (`process` wrapped): a verify row counts when its request
    was the row's at `process` and no failure there took it (`_fail` wrapped), or its request was preempted while it
    was in flight and waits for it (a held row), and the row reported no NONFINITE or empty mask."""
    s = h.sched
    got: dict[str, Any] = {"drafts": 0, "accepted": 0, "per_pos": np.zeros(s.k, np.int64), "held": 0}
    process, fail = s.process, s._fail
    failed: set[int] = set()

    def _fail(req: Any, *a: Any, **kw: Any) -> Any:
        failed.add(int(s.aid[req.slot]))
        return fail(req, *a, **kw)

    def _process(inf: Any, tokens: np.ndarray, n_out: np.ndarray, *a: Any) -> Any:
        valid = s.aid[inf.slots] == inf.aids
        held = inf.sample & np.array([(r := s._held.get(x)) is not None and r.status == WAITING
                                      for x in inf.aids.tolist()], bool)
        failed.clear()
        out = process(inf, tokens, n_out, *a)
        status = a[1] if len(a) > 1 else None
        ok = status == 0 if status is not None else np.ones(len(n_out), bool)
        rows = inf.device & ok & ((valid & ~np.isin(inf.aids, list(failed))) | held)
        acc = n_out[rows].astype(np.int64) - 1
        got["drafts"] += int(rows.sum())
        got["accepted"] += int(acc.sum())
        got["per_pos"] += (acc[:, None] > np.arange(s.k)[None, :]).sum(axis=0)
        got["held"] += int((rows & held).sum())
        return out

    s.process, s._fail = _process, _fail
    return got


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_held_verify_rows_count_as_drafts(mode, k) -> None:
    """A long request is preempted at each of 12 turns by a higher-priority one that takes the only slot: whenever its
    decode row is in flight then, that row's tokens are delivered to it while it waits, and its verify step counts in
    the counters as every other does. Without the held rows, drafts + accepted drafts fall short of the tokens the
    request got."""
    held = 0
    for at in range(2, 14):
        specs = [Spec(rid=0, prompt_len=0, prompt=stem(30, 24), max_tokens=40, seed=0x5C0, priority=5, ignore_eos=True,
                      preempt_at=at),
                 Spec(rid=1, prompt_len=0, prompt=stem(31, 12), max_tokens=6, seed=0x5C1, priority=0, arrive=at)]
        h = make(mode, specs, k, max_num_seqs=1, policy="priority", accept_p=0.6)
        want = verified_rows(h)
        run(h)
        s = h.sched
        assert s.num_preemptions == 1
        assert (s.spec_drafts, s.spec_accepted) == (want["drafts"], want["accepted"]), (at, want)
        assert s.spec_accepted_per_pos.tolist() == want["per_pos"].tolist()
        held += want["held"]
    assert held, "no preemption caught a decode row in flight"


def load(n: int, every: int) -> list[Spec]:
    """Constrained requests (compact JSON's shape: forced runs at the start and at the end) arriving every `every`
    turns."""
    return [Spec(rid=i, prompt_len=0, prompt=stem(100 + i, 20 + (i * 7) % 30), max_tokens=60, seed=0x1000 + i,
                 grammar=grammar_spec(i, 8 + i % 7, 8 + i % 7, lead=2, tail=1), arrive=i * every) for i in range(n)]


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("mode", ["window16", "attention"])
def test_eos_free_finishes_explain_the_gap_under_load(mode, k) -> None:
    """One at a time, drafts + accepted drafts + forced tokens equal the tokens after each request's first. Under load
    they fall short by the EOS-free finishes, every one appended while the request's row was in flight (the gate holds
    only while nothing else stays in it): exactly those tokens, which no verify row delivered."""
    for specs, seqs, alone in ((load(12, 60), 4, True), (load(24, 1), 8, False)):
        h = make(mode, specs, k, max_num_seqs=seqs, accept_p=0.7, forced_prefix=True, eos_free_finish=True)
        h.probe = Probe(h)
        run(h)
        s = h.sched
        after_first = sum(n - 1 for n in h.sent.values() if n)
        three = s.spec_drafts + s.spec_accepted + s.forced_tokens
        assert s.num_preemptions == 0 and s.spec_trimmed == 0 and s.resampled == 0 and s.forced_tokens > 0
        if alone:
            assert s.eos_free.tolist() == [0, 0] and after_first == three
        else:
            assert s.eos_free[0] == 0 and s.eos_free[1] > 0 and after_first - three == s.eos_free[1], \
                (after_first, three, s.eos_free.tolist())


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_each_request_reports_its_verify_rows_and_tokens(mode, k) -> None:
    """Every draft accepted: each verify row delivers 1 + k tokens (max_tokens 1 + 6 (1 + k): the first token, then six
    full rows), so a request reports 6 rows of 1 + k tokens; none accepted: one token a row. Several in flight at once,
    and request 0 preempted mid-decode by a higher-priority arrival: its rows before and after the preemption add up
    (folded into the request while it waits), its tokens after the first being theirs but its readmission's first
    sample (`Scheduler.resampled`). The sums are those counters (`Harness.check_spec_counts`)."""
    n_out = 1 + 6 * (1 + k)
    for p, per_row in ((1.0, 1 + k), (0.0, 1)):
        specs = [Spec(rid=i, prompt_len=0, prompt=stem(60 + i, 16 + 3 * i), max_tokens=n_out, seed=0x7A0 + i,
                      ignore_eos=True, priority=5, arrive=i) for i in range(4)]
        specs[0].preempt_at = 6
        specs.append(Spec(rid=9, prompt_len=0, prompt=stem(69, 12), max_tokens=3, seed=0x7AF, priority=0, arrive=6))
        h = make(mode, specs, k, max_num_seqs=4, policy="priority", accept_p=p)
        run(h)
        s = h.sched
        assert s.num_preemptions == 1 and s.resampled == 1 and set(h.spec) == {0, 1, 2, 3, 9}
        for rid, (rows, toks) in h.spec.items():
            assert toks == len(h.results[rid].tokens) - 1 - (rid == 0), (p, rid, rows, toks)
            if rid in (1, 2, 3):
                assert (rows, toks) == ((6 if p else n_out - 1), rows * per_row), (p, rid, rows, toks)
        assert sum(r for r, _ in h.spec.values()) == s.spec_drafts

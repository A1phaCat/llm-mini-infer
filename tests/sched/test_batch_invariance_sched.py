"""The scheduler's side of batch invariance (`Scheduler.batch_invariant`; architecture.md, "Batch invariance"): on a
GDN model every prefill chunk is a whole number of grids from where its prompt's computation started (its hit, or the
last exact cut: a hint, a refine's end, an image the step cannot have), which the strict fake checks on every plan; on
every model a final chunk takes its forced prefix whole, as many tokens as the request alone decides. Outputs are the
sampled path's either way (the harness predicts them)."""

from __future__ import annotations

import pytest

from llm_mini_infer.sched.plan import reserve_positions
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.fake_grammar import spec
from ..support.forced_cases import MODES, Probe, forced_of, make, row_at, run
from ..support.harness import Harness, Spec
from ..support.sched_cases import VOCAB, stem


def cuts(h: Harness) -> dict[int, list[tuple[int, int]]]:
    """Each request's explicit rows that do not sample, (first position, end), by its slot's admissions in order."""
    out: dict[int, list[tuple[int, int]]] = {}
    for p in h.plans:
        for slot, p0, nq, dev, sm in zip(p["slots"], p["pos0"], p["n_query"], p["device"], p["sample"]):
            if not dev and not sm:
                out.setdefault(slot, []).append((p0, p0 + nq))
    return out


def anchored(h: Harness, grid: int, exact: set[int]) -> bool:
    """Whether every explicit row that does not sample ends a whole number of grids past where its prompt's computation
    started: its hit (the row's first position after an ADMIT), or an exact cut before it."""
    anchor: dict[int, int] = {}
    for p in h.plans:
        for slot, h0 in zip(p["admit"], p["seq_len"]):
            anchor[slot] = h0
        for slot, p0, nq, dev, sm in zip(p["slots"], p["pos0"], p["n_query"], p["device"], p["sample"]):
            if dev or sm:
                continue
            end = p0 + nq
            if (end - anchor[slot]) % grid:
                if end not in exact:
                    return False
                anchor[slot] = end
    return True


@pytest.mark.parametrize("caching", [True, False])
def test_every_cut_lies_on_the_grid_from_where_the_prompt_started(caching) -> None:
    """A long prompt prefilled under a small budget beside requests that decode (they take its budget each step), with
    a hint at 37: off, its budget cuts fall wherever the step's budget ends (or on the absolute grid while checkpoints
    lie ahead); on, every cut is a whole number of grids from position 0, and after the hint (an exact cut, off the
    grid) from 37; a later request that hits the hint's checkpoint cuts from 37 too. The strict fake checks every
    plan; outputs are the sampled path's."""
    p = stem(6, 150)

    def specs() -> list[Spec]:
        return [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=4, seed=91, hints=[37], arrive=3),
                Spec(rid=1, prompt_len=0, prompt=stem(7, 9), max_tokens=60, seed=92),
                Spec(rid=2, prompt_len=0, prompt=stem(8, 11), max_tokens=60, seed=93),
                Spec(rid=3, prompt_len=0, prompt=[*p[:120], 1, 2, 3], max_tokens=2, seed=94, arrive=40)]

    def harness(on: bool) -> Harness:
        h = Harness(specs=specs(), vocab=VOCAB, max_tokens=50, page_size=4, num_pages=300, checkpoint_slots=6,
                    grid=16, max_model_len=256, prefix_caching=caching, row_tokens=60, batch_invariant=on)
        h.run(check_every_turn=True)
        h.check()
        return h

    off, on = harness(False), harness(True)
    assert on.runner.aligned and on.sched.aligned and not off.sched.aligned
    assert not anchored(off, 16, {37}) and anchored(on, 16, {37})
    if caching:
        assert on.results[3].cached in (37, 117)  # the hint's checkpoint, or the first prompt's past it
    assert {r: x.tokens for r, x in on.results.items()} == {r: x.tokens for r, x in off.results.items()}


def test_a_chunk_short_of_a_grid_waits() -> None:
    """A step that leaves a prefill less than a grid gives it nothing: its next chunk comes with the next step whose
    budget holds a grid, not a few tokens now (which would move its cuts with the rows beside it). Two 90-token prompts
    arrive together beside a decode row under a budget of 20, without the prefix cache (no checkpoint target rounds a
    cut): off, chunks take what the step leaves (19), and the second prompt starts on the 5 tokens the first's final
    chunk leaves; on, chunks of 16, and the second prompt waits for the step after that final chunk."""
    def specs() -> list[Spec]:
        return [Spec(rid=0, prompt_len=0, prompt=stem(1, 5), max_tokens=40, seed=1),
                Spec(rid=1, prompt_len=0, prompt=stem(2, 90), max_tokens=2, seed=2, arrive=2),
                Spec(rid=2, prompt_len=0, prompt=stem(3, 90), max_tokens=2, seed=3, arrive=2)]

    def explicit(on: bool) -> list[list[tuple[int, bool]]]:
        """Each plan's explicit rows: (n_query, samples)."""
        h = Harness(specs=specs(), vocab=VOCAB, max_tokens=20, page_size=4, num_pages=200, checkpoint_slots=4, grid=16,
                    max_model_len=256, prefix_caching=False, batch_invariant=on)
        h.run(check_every_turn=True)
        h.check()
        assert anchored(h, 16, set()) == on
        return [[(nq, bool(sm)) for nq, dev, sm in zip(p["n_query"], p["device"], p["sample"]) if not dev]
                for p in h.plans]

    off, on = explicit(False), explicit(True)
    assert [(19, False)] in off and [(14, True), (5, False)] in off
    assert all(nq == 16 for rows in on for nq, sm in rows if not sm)
    i = on.index([(10, True)])  # the first prompt's final chunk leaves 9 tokens: the second prompt waits
    assert on[i + 1] == [(16, False)]


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("mode", MODES)
def test_a_final_chunk_takes_its_forced_prefix_whole(mode, k) -> None:
    """Under a chunk cap of 24, prompts of 48, 47 and 46 end on chunks of 24, 23 and 22: off, a final chunk takes the
    forced tokens its cap leaves room for (0, 1 and 2); on, each takes them whole (the request decides how many: its
    forced run, within max_tokens and the positions it reserved less its drafter's chain), and a final chunk without
    room for them is cut short of the prompt's end and takes them in a later step. Outputs are the sampled path's."""
    def specs() -> list[Spec]:
        return [Spec(rid=i, prompt_len=0, prompt=stem(20 + i, n), max_tokens=20, seed=0xB0D + i, arrive=6 * i,
                     grammar=spec(30 + i, 6, 6, lead=3, tail=1)) for i, n in enumerate((48, 47, 46))]

    got = {}
    for on in (False, True):
        h = make(mode, specs(), k, max_num_seqs=1, chunk_cap=24, forced_prefix=True, batch_invariant=on)
        h.probe = Probe(h)
        run(h)
        assert h.sched.forced_whole == on and not h.sched.forced_declined
        got[on] = ([h.probe.forced.get(i, (0,))[0] for i in range(3)], {r: x.tokens for r, x in h.results.items()})
    whole = min(len(forced_of(spec(30, 6, 6, lead=3, tail=1))), reserve_positions(2, k) - (
        k - 1 if k and mode in ("gdn", "attention") else 0))
    assert got[False][0] == [0, 1, 2] and got[True][0] == [whole] * 3 and got[True][1] == got[False][1]


@pytest.mark.parametrize("k", [0, 3])
def test_a_branch_point_at_a_forcing_rows_off_grid_end(k) -> None:
    """Qwen's shape (grid 8): a constrained request resumes at 24 and its final chunk is the rest of its prompt, 30; a
    request extending that prompt, admitted after it in the same plan, waits for a branch checkpoint off.
    Off, the writer's row takes it at its prompt's end, 6 into the row and off its grid, and so samples its forced
    tokens (`forced_declined`); on, a branch point there would not replay a computation from the grid, so no
    checkpoint goes there: the writer takes its forced tokens, and the waiter resumes at 24 and computes the rest."""
    def specs() -> list[Spec]:
        a = Spec(rid=0, prompt_len=0, prompt=stem(33, 30), max_tokens=20, seed=0x4B0, arrive=20,
                 grammar=spec(33, 8, 8, lead=2, tail=1))
        return [Spec(rid=2, prompt_len=0, prompt=[*a.prompt[:24], 1, 2, 3], max_tokens=4, seed=0x4B2), a,
                Spec(rid=1, prompt_len=0, prompt=[*a.prompt, 5, 6, 7], max_tokens=8, seed=0x4B1, arrive=20)]

    for on in (False, True):
        h = make("gdn", specs(), k, forced_prefix=True, batch_invariant=on)
        h.probe = Probe(h)
        run(h)
        assert h.sched.forced_declined == (not on)
        assert h.results[0].cached == 24 and h.results[1].cached == (24 if on else 30)
        if on:
            assert row_at(h, 24) == (6 + 2, []) and h.probe.forced == {0: (2, "_admit_one")}


def test_the_switch_needs_a_budget_a_prefill_progresses_in() -> None:
    """Refused at construction: a budget whose decode rows (max_num_seqs − 1 of 1 + k tokens) leave a prefill less than
    a grid and a final chunk's forced prefix, and a chunk cap below the grid; a sliding-window model (not aligned) needs
    only one token and the forced prefix."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..support.fake_grammar import FakeGrammar

    base = dict(max_model_len=256, page_size=4, num_pages=100, num_state_slots=10, eos_token_ids=(511,),
                checkpoint_grid=16, forced_prefix=True, grammars=Grammars(FakeGrammar(512, (511,))))
    Scheduler(max_num_seqs=8, max_num_batched_tokens=16 + 8 + 7 * 4, mtp_k=3, batch_invariant=True, **base)
    with pytest.raises(ValueError, match="leaves 23 tokens next to 7 decode rows of 4, under the 24"):
        Scheduler(max_num_seqs=8, max_num_batched_tokens=16 + 8 + 7 * 4 - 1, mtp_k=3, batch_invariant=True, **base)
    with pytest.raises(ValueError, match="long_prefill_token_threshold 15 is below the 16-token grid"):
        Scheduler(max_num_seqs=8, max_num_batched_tokens=256, mtp_k=3, batch_invariant=True,
                  long_prefill_token_threshold=15, **base)
    Scheduler(max_num_seqs=8, max_num_batched_tokens=256, mtp_k=3, batch_invariant=False,
              long_prefill_token_threshold=15, **base)


def test_the_strict_fake_refuses_a_cut_off_the_grid() -> None:
    """The fake's check itself: the scheduler's aligned cuts made plain (`aligned` off in the scheduler, on in the
    fake), a budget cut off the grid fails the plan."""
    from llm_mini_infer.runner.base import PlanError

    specs = [Spec(rid=0, prompt_len=0, prompt=stem(1, 60), max_tokens=2, seed=1)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=20, page_size=4, num_pages=100, checkpoint_slots=4, grid=16,
                max_model_len=128, prefix_caching=False)
    h.runner.aligned = True  # the fake checks; the scheduler cuts at 20 tokens a step
    with pytest.raises(PlanError, match="a prefill cut at 20, off the 16-token grid from 0"):
        h.run()

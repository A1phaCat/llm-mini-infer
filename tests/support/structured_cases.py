"""The structured-output scheduler suite's schema, workloads and driver (tests/structured/test_structured.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.sched.plan import reserve_positions

from .fake_grammar import spec
from .fuzzing import event, st
from .harness import EOS, Harness, Spec
from .sched_cases import VOCAB, check_plans, stem

SCHEMA = {
    "$defs": {"Entry": {"additionalProperties": False, "description": "One entry.", "properties": {
        "id": {"title": "Id", "type": "integer"},
        "subjects": {"anyOf": [{"items": {"enum": ["history", "science", "travel"], "type": "string"}, "minItems": 1,
                                "type": "array"}, {"type": "null"}], "title": "Subjects"},
        "formats": {"anyOf": [{"items": {"enum": ["bound", "loose", "folded"], "type": "string"}, "type": "array"},
                              {"type": "null"}], "title": "Formats"}},
        "required": ["id", "subjects", "formats"], "title": "Entry", "type": "object"}},
    "additionalProperties": False, "description": "A small catalogue.",
    "properties": {"total": {"title": "Total", "type": "integer"},
                   "entries": {"items": {"$ref": "#/$defs/Entry"}, "maxItems": 8, "title": "Entries",
                               "type": "array"}},
    "required": ["total", "entries"], "title": "Catalog", "type": "object"}


def run(specs, **kw) -> Harness:
    kw.setdefault("vocab", VOCAB)
    h = Harness(specs=specs, **kw)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    assert not h.sched.n_masked and not h.sched.masked.any()
    assert all(r.matcher is None for r in h.sched.slot_req if r is not None)
    return h


def masked_rows(h: Harness) -> int:
    return sum(sum(p["masked"]) for p in h.plans)


@st.composite
def structured_workloads(draw):
    """Constrained requests among unconstrained ones, in small pools: MTP verify rows masked along their drafts,
    final prefill chunks, preemption and readmission (matchers rebuilt), aborts, finishes by the grammar's EOS, by
    max_tokens and by stop ids, ignore_eos past the grammar's end, NaNs, empty masks (MASK_EMPTY: the request fails
    alone), follow-up turns that hit the cache. MTP at up to 7 drafts, 5 as often as 3 (5 is Gemma 4's assistant's
    default: a grammar masked along 1 + 5 tokens)."""
    k = draw(st.sampled_from([0, 1, 3, 3, 5, 5, 7]))
    page_size = draw(st.sampled_from([2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
    max_model_len = 160
    stems = [stem(i, draw(st.integers(8, 80))) for i in range(draw(st.integers(1, 3)))]
    n = draw(st.integers(1, 7))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs: list[Spec] = []
    for rid in range(n):
        s = draw(st.sampled_from(stems))
        prompt = s[:draw(st.integers(1, len(s)))] + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=10))
        prompt = [t if t != EOS else 0 for t in prompt][:110]
        grammar = ""
        if draw(st.booleans()) or draw(st.booleans()):
            lo = draw(st.integers(0, 25))
            grammar = spec(draw(st.integers(0, 1 << 30)), lo, lo + draw(st.integers(0, 20)))
        eos = draw(st.one_of(st.none(), st.integers(0, 20)))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 40)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 25)), eos_after=eos,
                          ignore_eos=draw(st.booleans()), abort_at=draw(st.one_of(st.none(), st.integers(0, 40))),
                          priority=draw(st.integers(0, 2)),
                          stop_ids=draw(st.lists(st.integers(0, VOCAB - 2), max_size=2)),
                          nonfinite_after=draw(st.sampled_from([None] * 12 + [0, 3])), grammar=grammar,
                          mask_empty_after=draw(st.sampled_from([None] * 8 + [0, 3])) if grammar else None))
    bound = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True)
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                accept_p=draw(st.sampled_from([0.0, 0.6, 0.9, 1.0])), max_num_seqs=draw(st.integers(1, 5)),
                max_tokens=draw(st.integers(4 * (1 + k), 96)), num_pages=bound + 1 + draw(st.integers(0, 50)),
                checkpoint_slots=draw(st.integers(1, 5)), policy=draw(st.sampled_from(["fcfs", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 9])), decode_checkpoint_interval=draw(st.sampled_from([8, 16])),
                row_tokens=draw(st.sampled_from([0, 0, 40, 160])))


def run_structured_workload(w: dict) -> None:
    from .workloads import run_cache_workload

    h = run_cache_workload(w)
    sc = h.sched
    assert not sc.n_masked and sc.grammars.errors == 0
    assert sc.spec_accepted <= sc.spec_draft_tokens <= sc.k * sc.spec_drafts  # drafts a grammar refuses: not counted
    if h.sched.num_preemptions and any(s.grammar for s in w["specs"]):
        event("structured: preempted")
    if masked_rows(h):
        event("structured: masked rows")
    if any(s.grammar and s.ignore_eos for s in w["specs"]):
        event("structured: ignore_eos with a grammar")
    if h.sched.end_waits:
        event("structured: rows skipped behind a row that could end its grammar")

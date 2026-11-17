"""Failing examples saved by fuzz campaigns (`tests/support/fuzzing.py`), kept in `tests/data/fuzz_regressions/` and
replayed."""

from __future__ import annotations

import pytest

from ..support.fuzzing import REGRESSIONS, load_example, run_example


@pytest.mark.parametrize("path", sorted(REGRESSIONS.glob("*.json")), ids=lambda p: p.stem)
def test_saved_fuzz_failure(path) -> None:
    suite, example = load_example(path)
    run_example(suite, example)


def test_a_saved_workload_naming_an_unsupported_refine_option(tmp_path) -> None:
    """Refines step in fixed pieces in mixed steps only: a saved `reuse` workload naming `bound` or `exclusive` is
    refused (its refines would replay otherwise); a `bf16` one replays as it ran (neither option acts outside a reuse
    refine), its piece unset when the room check would refuse it; the supported values load and are dropped."""
    import json

    base = json.loads((REGRESSIONS / "precise-nan-fp8-continuation.json").read_text())  # bf16, max_tokens 16, 2 seqs
    assert base["example"]["precise"] == "bf16"
    cases = [("bf16", {"refine_chunking": "fixed", "refine_steps": "mixed"}, None),
             ("bf16", {"refine_chunking": "bound", "refine_steps": "exclusive", "refine_chunk_tokens": 16}, 0),
             ("reuse", {"refine_steps": "exclusive"}, "refine_steps 'exclusive' is not supported"),
             ("reuse", {"refine_chunking": "bound"}, "refine_chunking 'bound' is not supported")]
    for i, (precise, keys, want) in enumerate(cases):
        data = json.loads(json.dumps(base))
        data["example"].update(keys, precise=precise)
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(data))
        if isinstance(want, str):
            with pytest.raises(ValueError, match=want):
                load_example(path)
            continue
        suite, example = load_example(path)
        assert "refine_chunking" not in example and "refine_steps" not in example
        assert example.get("refine_chunk_tokens", 0) == (want if want is not None else 0)
        run_example(suite, example)

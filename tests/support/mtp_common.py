"""The MTP tests' end token, prompts and draft spy, shared by the Qwen and Gemma runners' tests."""

from __future__ import annotations

EOS = 511


def seq(n: int, salt: int = 0) -> list[int]:
    return [(i * 53 + 7 + salt * 131) % 500 for i in range(n)]


def draft_positions_spy(runner, k: int) -> list[tuple[list[int], list[int]]]:
    """Wrap a CUDA runner's `_sample` (CPU) to record, for each draft call, the positions the sampler keys (`qpos` + 1)
    and the positions those drafts fill: draft j of a step (j = 0 .. k − 1) fills `seq_len + 1 + j` of its slot, where
    `seq_len` (read when it is drafted, after the step's advance) is the bonus's position. Returns the record."""
    seen: list[tuple[list[int], list[int]]] = []
    orig = runner._sample
    calls = {"n": 0}

    def spy(hidden, qpos, slots, out, status, pending, splan, draft=False, mask=None, **kw):
        if draft:
            j = calls["n"] % k
            calls["n"] += 1
            fill = (runner.tables.seq_len.index_select(0, slots) + 1 + j).tolist()
            seen.append(((qpos + 1).tolist(), fill))
        return orig(hidden, qpos, slots, out, status, pending, splan, draft=draft, mask=mask, **kw)

    runner._sample = spy
    return seen

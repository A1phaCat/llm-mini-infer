"""Token ids for the tiny models' prompts."""

from __future__ import annotations


def ids(n: int, seed: int = 0) -> list[int]:
    return [(i * 37 + 11 + seed * 101) % 500 for i in range(n)]

"""The one iterative Tarjan (`bounds.components`) against copies of the two loops it replaced."""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from llm_mini_infer.grammar import bounds


def _ref_components(root, edges):
    index, low = {root: 0}, {root: 0}
    on, path, out = {root}, [root], []
    work = [(root, iter(edges.get(root, ())))]
    while work:
        v, it = work[-1]
        for w in it:
            if w not in index:
                index[w] = low[w] = len(index)
                path.append(w)
                on.add(w)
                work.append((w, iter(edges.get(w, ()))))
                break
            if w in on:
                low[v] = min(low[v], index[w])
        else:
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = path.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def _ref_deepest(height, uses):
    index: dict[Any, int] = {None: 0}
    low: dict[Any, int] = {None: 0}
    comp: dict[Any, int] = {}
    best: dict[Any, int] = {}
    path: list[Any] = [None]
    on: set[Any] = {None}
    work = [(None, iter(sorted(uses.get(None, {}), key=str)))]
    n_comp = 0
    while work:
        v, it = work[-1]
        for w in it:
            if w not in index:
                index[w] = low[w] = len(index)
                path.append(w)
                on.add(w)
                work.append((w, iter(sorted(uses.get(w, {}), key=str))))
                break
            if w in on:
                low[v] = min(low[v], index[w])
        else:
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                members = []
                while True:
                    w = path.pop()
                    on.discard(w)
                    comp[w] = n_comp
                    members.append(w)
                    if w == v:
                        break
                _ref_component(members, n_comp, comp, height, uses, best)
                n_comp += 1
    return best[None]


def _ref_component(members, c, comp, height, uses, best):
    out = {}
    for m in members:
        named = uses.get(m, {})
        out[m] = max([height[m], *(lv + best[w] for w, lv in named.items() if comp[w] != c)])
    inner = {m: [(w, lv) for w, lv in uses.get(m, {}).items() if comp[w] == c] for m in members}
    if not any(inner.values()):
        best.update(out)
        return
    steps = 0
    for start in members:
        top, seen = out[start], {start}
        stack = [(start, 0, iter(inner[start]))]
        while stack and steps <= bounds.PATHS:
            m, acc, it = stack[-1]
            for w, lv in it:
                steps += 1
                if w not in seen:
                    seen.add(w)
                    top = max(top, acc + lv + out[w])
                    stack.append((w, acc + lv, iter(inner[w])))
                    break
            else:
                stack.pop()
                seen.discard(m)
        if steps > bounds.PATHS:
            bound = sum(max(lv for _, lv in inner[m]) for m in members if inner[m]) + max(out.values())
            best.update(dict.fromkeys(members, bound))
            return
        best[start] = top


def _graph(rng: random.Random, n: int, m: int) -> dict[int, list[int]]:
    edges: dict[int, list[int]] = {}
    for _ in range(m):
        edges.setdefault(rng.randrange(n), []).append(rng.randrange(n + 2))  # n, n + 1 have no entry
    return edges


@pytest.mark.parametrize("seed", range(300))
def test_components_equal_the_reference_on_random_graphs(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    edges = _graph(rng, n, rng.randint(0, 3 * n))
    root = rng.randrange(n + 2)
    assert list(bounds.components(root, lambda v: edges.get(v, ()))) == _ref_components(root, edges)


@pytest.mark.parametrize("back", [False, True])
def test_components_equal_the_reference_on_a_deep_chain(back: bool) -> None:
    edges = {i: [i + 1] for i in range(2000)}
    if back:
        edges[1999].append(0)
    assert list(bounds.components(0, lambda v: edges.get(v, ()))) == _ref_components(0, edges)


@pytest.mark.parametrize("seed", range(300))
def test_deepest_equals_the_reference_on_random_inputs(seed: int) -> None:
    rng = random.Random(seed)
    names: list[Any] = [None, *"abcdefgh"][: rng.randint(1, 9)]
    height = {k: rng.randint(0, 6) for k in names}
    uses: dict[Any, dict[Any, int]] = {}
    for k in names:
        if rng.random() < 0.8:
            uses[k] = {w: rng.randint(0, 5) for w in rng.sample(names, rng.randint(0, len(names)))}
    if rng.random() < 0.5:
        uses.setdefault(None, {})[names[-1]] = 1
    assert bounds.deepest(height, uses) == _ref_deepest(height, uses)


def test_deepest_equals_the_reference_on_a_deep_chain() -> None:
    names: list[Any] = [None, *range(1500)]
    height = {k: 1 for k in names}
    uses = {a: {b: 1} for a, b in itertools.pairwise(names)}
    uses[1499] = {7: 1}
    assert bounds.deepest(height, uses) == _ref_deepest(height, uses)

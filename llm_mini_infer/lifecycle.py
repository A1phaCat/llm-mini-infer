"""A long-lived object's phase, moved only along its machine's edges: the frontend's mode (`api.app.Mode`), an image
pool worker's (`launcher.Phase`) and a CUDA runner's image encoder's (`runner.cuda.images.Phase`). An edge outside
them is a bug: `Lifecycle.to` refuses it, with a RuntimeError."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import IntEnum


class Lifecycle:
    """`phase`, an IntEnum, and `edges`, each phase's legal successors; `name` names the machine in the error."""

    __slots__ = ("edges", "name", "phase")

    def __init__(self, name: str, phase: IntEnum, edges: Mapping[IntEnum, Collection[IntEnum]]) -> None:
        self.name, self.phase, self.edges = name, phase, edges

    def to(self, new: IntEnum) -> None:
        if new not in self.edges.get(self.phase, ()):
            raise RuntimeError(f"{self.name}: no edge from {self.phase.name} to {new.name}")
        self.phase = new

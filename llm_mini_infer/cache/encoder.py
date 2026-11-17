"""The encoder cache: image embeddings on the runner's device, keyed by the image's identity: its span key with its full
content digest (the same 256 bits the radix tree checks: `cache.radix`, Image spans), so two images whose 62-bit span
keys collide never share embeddings, and the request's `cache_salt`, so two salts never share them either: one image
under N salts holds N entries (docs/architecture.md, "The vision encoder and its cache").

The scheduler owns the bookkeeping, the runner the memory (`Capabilities.encoder_rows` rows of the model's hidden
size): as with KV pages, the scheduler decides and the runner only writes where a plan says.
- **Entries** hold one image's embeddings in blocks of `BLOCK` rows; `Entry.rows` lists its rows in token order. An
  entry is written by the ENCODE of the plan that allocates it, before that step's forward, so rows of the same plan
  may read it (stream order).
- **References:** a running request holds one for each of its spans that a chunk has reached and not yet finished (it
  gives them back when its chunk waits for pages behind a request that waits for rows and the plans would repeat:
  architecture.md, "The scheduler"). Unreferenced entries stay cached and are evicted least recently used when rows are
  needed. A plan may reallocate an evicted entry's rows at once: the steps still reading them run before its ENCODE on
  the stream.
- **Reservations:** the scheduler plans a chunk's images by reserving them (`reserve`), and the chunk then `commit`s,
  or, cut or abandoned, `rollback`s to the cache as it was.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np

BLOCK = 16


def _name(key: Hashable) -> str:
    """An entry's key as messages print it: its span key in hex."""
    k = key[0] if isinstance(key, tuple) else key
    return f"{k:#x}" if isinstance(k, int) else repr(k)


@dataclass(eq=False)
class Entry:
    key: Hashable  # the image's identity: (its span key, its full digest, the request's cache_salt or None)
    tokens: int
    blocks: list[int]
    refs: int = 0
    last_used: int = 0
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


class _Reservation:
    """The open reservation (`EncoderCache.reserve`): its undo log, oldest first, of (an entry it referenced, its
    recency before), (an entry it evicted, its blocks) or (an entry it allocated, None); and the keys it evicted."""

    __slots__ = ("gone", "undo")

    def __init__(self) -> None:
        self.undo: list[tuple[Entry, int | list[int] | None]] = []
        self.gone: set[Hashable] = set()


class EncoderCache:
    def __init__(self, rows: int):
        self.n_blocks = rows // BLOCK
        self.rows = self.n_blocks * BLOCK
        self._free = list(range(self.n_blocks - 1, -1, -1))
        self.entries: dict[Hashable, Entry] = {}
        self._clock = itertools.count(1)
        self.evicted = 0
        self._open: _Reservation | None = None  # from a chunk's first `reserve` to its `commit` or `rollback`

    @staticmethod
    def blocks_for(tokens: int) -> int:
        return -(-tokens // BLOCK)

    def get(self, key: Hashable) -> Entry | None:
        e = self.entries.get(key)
        if e is not None:
            e.last_used = next(self._clock)
        return e

    def alloc(self, key: Hashable, tokens: int) -> Entry | None:
        """A new entry for `key` (evicting unreferenced ones, least recently used first), or None if it cannot fit."""
        if key in self.entries:
            raise ValueError(f"encoder cache: {_name(key)} is cached already")
        need = self.blocks_for(tokens)
        if need > len(self._free):
            victims = sorted((e for e in self.entries.values() if not e.refs), key=lambda e: e.last_used)
            got = len(self._free)
            take = []
            for e in victims:
                if got >= need:
                    break
                take.append(e)
                got += len(e.blocks)
            if got < need:
                return None
            for e in take:
                if self._open is not None:
                    self._open.undo.append((e, e.blocks))
                    self._open.gone.add(e.key)
                self._drop(e)
                self.evicted += 1
        blocks = [self._free.pop() for _ in range(need)]
        e = Entry(key, tokens, blocks, last_used=next(self._clock))
        e.rows = (np.asarray(blocks, np.int64)[:, None] * BLOCK + np.arange(BLOCK)[None, :]).reshape(-1)[:tokens]
        self.entries[key] = e
        if self._open is not None:
            self._open.undo.append((e, None))
        return e

    def reserve(self, key: Hashable, tokens: int = 0) -> Entry | None:
        """Reference `key`'s entry for a chunk being planned (`commit` keeps it, `rollback` undoes it): the cached one,
        or with `tokens` a new one (`alloc`). None, with nothing reserved, when it is not cached and `tokens` is 0,
        this reservation evicted it, or it does not fit."""
        if self._open is None:
            self._open = _Reservation()
        e = self.entries.get(key)
        if e is not None:
            self._open.undo.append((e, e.last_used))
        elif not tokens or key in self._open.gone or (e := self.alloc(key, tokens)) is None:
            return None
        self.ref(e)
        return e

    def commit(self) -> None:
        """The open reservation's entries are the plan's: their references stay (with none open: nothing)."""
        self._open = None

    def rollback(self) -> None:
        """Undo the open reservation, newest change first: its new entries go and give their blocks back, the entries
        it evicted come back with theirs (the free list as it was, block for block), and the cached ones it referenced
        get back their references and recency. With none open (rolled back already, or nothing reserved): nothing."""
        for e, before in reversed(self._open.undo if self._open is not None else ()):
            if before is None:
                del self.entries[e.key]
                self._free.extend(reversed(e.blocks))
                e.blocks = []
            elif isinstance(before, list):
                if self._free[len(self._free) - len(before):] != before:
                    raise RuntimeError(f"encoder cache: the blocks of {_name(e.key)} are not where its eviction left "
                                       "them")
                del self._free[len(self._free) - len(before):]
                e.blocks = before
                self.entries[e.key] = e
                self.evicted -= 1
            else:
                e.refs -= 1
                e.last_used = before
        self.commit()

    def ref(self, e: Entry) -> None:
        e.refs += 1
        e.last_used = next(self._clock)

    def unref(self, e: Entry) -> None:
        e.refs -= 1
        if e.refs < 0:
            raise RuntimeError(f"encoder cache entry {_name(e.key)} unreferenced more often than referenced")

    def drop(self, key: Hashable) -> None:
        """Forget an unreferenced entry (a failed request's image)."""
        e = self.entries.get(key)
        if e is not None:
            if e.refs:
                raise RuntimeError(f"encoder cache entry {_name(key)} dropped while referenced")
            self._drop(e)

    def _drop(self, e: Entry) -> None:
        del self.entries[e.key]
        self._free.extend(e.blocks)
        e.blocks = []

    def check(self) -> None:
        """Every block is free or held by exactly one entry (tests)."""
        held = [b for e in self.entries.values() for b in e.blocks]
        assert len(held) == len(set(held)), "an encoder block is held twice"
        assert not set(held) & set(self._free), "a held encoder block is free"
        assert len(held) + len(self._free) == self.n_blocks, "encoder blocks leaked"
        assert all(e.refs >= 0 for e in self.entries.values())
        assert self._open is None, "an encoder reservation left open"

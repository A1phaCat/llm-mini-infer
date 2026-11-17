"""Fetching a request's images concurrently: one failure cancels its siblings."""

from __future__ import annotations

import asyncio

import pytest

from llm_mini_infer.vision.fetch import FetchError, fetch_all


class FakeFetcher:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.cancelled: list[str] = []

    async def fetch(self, url: str, budget=None) -> bytes:
        if url == "bad":
            await self.started.wait()
            raise FetchError("boom")
        if url.startswith("ok"):
            return url.encode()
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(url)
            raise
        return url.encode()


def test_failure_cancels_pending_siblings() -> None:
    async def main() -> None:
        f = FakeFetcher()
        before = asyncio.all_tasks()
        with pytest.raises(FetchError, match="boom"):
            await fetch_all(f, ["slow1", "bad", "slow2"])
        assert sorted(f.cancelled) == ["slow1", "slow2"]
        assert asyncio.all_tasks() == before

    asyncio.run(main())


def test_results_keep_url_order_and_caller_cancellation_cancels_all() -> None:
    async def main() -> None:
        f = FakeFetcher()
        assert await fetch_all(f, ["ok1", "ok2", "ok3"]) == [b"ok1", b"ok2", b"ok3"]
        task = asyncio.ensure_future(fetch_all(f, ["slow1", "slow2"]))
        await f.started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(f.cancelled) == ["slow1", "slow2"]

    asyncio.run(main())

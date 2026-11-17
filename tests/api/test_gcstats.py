"""The garbage-collector counters (`llm_mini_infer.gcstats`): a `gc.callbacks` timer per generation in the engine and
in each frontend, on /metrics as llm_mini_infer:gc_seconds_total and llm_mini_infer:gc_collections_total
{process,generation}, the engine's carried in `EngineStats`; and the rate-limited warning for long pauses, logged
outside the collector."""

from __future__ import annotations

import asyncio
import gc
import logging
import shutil

import numpy as np

from llm_mini_infer import gcstats
from llm_mini_infer.api.engine_client import EngineClient
from llm_mini_infer.config import Config
from llm_mini_infer.engine import ipc
from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import StepOutputs
from llm_mini_infer.engine.process import EngineLoop
from llm_mini_infer.gcstats import GcStats
from llm_mini_infer.metrics import Metrics

from ..support.harness import Harness


def test_each_generation_counts_its_collections_and_time() -> None:
    g = GcStats("engine").install()
    try:
        for gen in range(3):
            before = list(g.collections), list(g.ns)
            gc.collect(gen)
            assert g.collections[gen] >= before[0][gen] + 1 and g.ns[gen] > before[1][gen], (gen, g.collections)
    finally:
        g.uninstall()
    after = list(g.collections)
    gc.collect(2)
    assert g.collections == after and g._callback not in gc.callbacks  # uninstalled: no more counts
    assert g.install().install() and gc.callbacks.count(g._callback) == 1  # installed once
    g.uninstall()


def test_long_pauses_are_warned_once_a_minute_outside_the_collector(monkeypatch, caplog) -> None:
    """Every collection counts as long here: none logs from inside the collector, the next `maybe_warn` logs them
    all in one line, and another within WARN_EVERY_S logs nothing."""
    g = GcStats("frontend").install()
    g._slow_ns = 0
    clock = [10**12]
    monkeypatch.setattr(gcstats.time, "monotonic_ns", lambda: clock[0])
    g._since = clock[0]
    try:
        with caplog.at_level(logging.WARNING, logger="llm_mini_infer.gcstats"):
            gc.collect(0)
            gc.collect(1)
            assert not caplog.records and g.slow >= 2
            clock[0] += 5 * 10**9
            g.maybe_warn()
            assert len(caplog.records) == 1 and "frontend: " in caplog.text and "in the last 5 s" in caplog.text
            assert g.slow == 0
            gc.collect(0)
            clock[0] += 30 * 10**9
            g.maybe_warn()  # within the minute: held
            assert len(caplog.records) == 1 and g.slow >= 1
            clock[0] += 31 * 10**9
            g.maybe_warn()
            assert len(caplog.records) == 2 and "in the last 61 s" in caplog.records[1].getMessage()
    finally:
        g.uninstall()


def test_the_engines_counts_and_the_frontends_own_reach_metrics() -> None:
    """In process on the fake runner: the engine loop's counts cross the IPC encoding in its stats; the frontend
    exports them under process="engine", next to its own (process="frontend"), every generation a series, and they
    grow with collections."""
    h = Harness(specs=[])
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), EngineCore(h.sched, h.runner), path)
    loop.gc.install()
    metrics = Metrics("m", 4096)
    client = EngineClient(path, 0, metrics)
    client.gc.install()
    e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)

    def scrape() -> dict[tuple[str, str, str], float]:
        msg = StepOutputs(step=0, rids=e64, n_new=e32, tokens=e32, finish=np.zeros(0, np.uint8), first_rids=e64,
                          first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64, first_runner_ns=e64,
                          t_sent_ns=0, stats=loop._stats(None))
        client._on_stats(ipc.Decoder().decode(ipc.Encoder().encode(msg)).stats)
        metrics.set_gc("frontend", client.gc.ns, client.gc.collections)
        out = {}
        for line in metrics.render().decode().splitlines():
            if line.startswith("llm_mini_infer:gc_"):
                name, rest = line.split("{", 1)
                labels = dict(kv.split("=") for kv in rest.split("}")[0].split(","))
                out[name, labels["process"].strip('"'), labels["generation"].strip('"')] = float(line.rsplit(" ")[1])
        return out

    try:
        before = scrape()
        assert set(before) == {(f"llm_mini_infer:gc_{n}_total", p, str(g)) for n in ("seconds", "collections")
                               for p in ("engine", "frontend") for g in range(3)}
        for gen in range(3):
            gc.collect(gen)
        after = scrape()
        for p in ("engine", "frontend"):
            for g in range(3):
                assert after["llm_mini_infer:gc_collections_total", p, str(g)] > \
                    before["llm_mini_infer:gc_collections_total", p, str(g)], (p, g)
                assert after["llm_mini_infer:gc_seconds_total", p, str(g)] > \
                    before["llm_mini_infer:gc_seconds_total", p, str(g)], (p, g)
        assert "# TYPE llm_mini_infer:gc_seconds_total counter" in metrics.render().decode()
    finally:
        loop.gc.uninstall()
        asyncio.run(client.close())
        loop.close()
        shutil.rmtree(path, ignore_errors=True)

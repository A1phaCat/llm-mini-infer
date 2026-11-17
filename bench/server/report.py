#!/usr/bin/env python3
"""Aggregate bench/server/run.py results (several runs of each) into markdown tables.

    uv run python bench/server/report.py results/server/*.json

A summary table's cell is the median over runs, with the range [min-max] when there was more than one run; the stage
and phase tables give the median only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def cell(xs: list[float], fmt: str = "{:.2f}") -> str:
    if not xs:
        return "-"
    m = statistics.median(xs)
    if len(xs) == 1:
        return fmt.format(m)
    return f"{fmt.format(m)} [{fmt.format(min(xs))}-{fmt.format(max(xs))}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path, help="run.py's result files (JSON)")
    files = ap.parse_args().results
    ttft: dict[str, list[dict]] = defaultdict(list)
    decode: dict[int, list[dict]] = defaultdict(list)
    idle: dict[int, list[dict]] = defaultdict(list)
    sse: list[dict] = []
    hosts: dict[str, list[str]] = defaultdict(list)  # host -> its files
    for f in files:
        d = json.loads(f.read_text())
        hosts[f"{d['host']}, {d['cores']} cores, Python {d['python']}"].append(f.name)
        for t in d.get("ttft", []):
            mode = "warm" if d.get("warm") else ("as is, fake sleeps" if t.get("fake_wait") == "sleep" else "as is")
            ttft[mode].append(t)
        for x in d.get("decode", []):
            decode[x["batch"]].append(x)
        for x in d.get("idle", []):
            idle[x["conc"]].append(x)
        sse += d.get("sse", [])
    if len(hosts) == 1:
        print(f"Host: {next(iter(hosts))}\n")
    else:  # the tables below mix them
        print("Hosts (MIXED): " + "; ".join(f"{h} ({', '.join(fs)})" for h, fs in hosts.items()) + "\n")
        print(f"warning: the results come from {len(hosts)} hosts", file=sys.stderr)
    if ttft:
        print("### TTFT overhead, concurrency 1 (ms, stage timestamps minus runner time)\n")
        print("| CPU | runs | p50 | p99 | client-side TTFT minus runner, p50 | p99 |")
        print("|---|---|---|---|---|---|")
        for mode, rs in sorted(ttft.items()):
            print(f"| {mode} | {len(rs)} | {cell([r['overhead_ms']['p50'] for r in rs], '{:.3f}')} "
                  f"| {cell([r['overhead_ms']['p99'] for r in rs], '{:.3f}')} "
                  f"| {cell([r['client_ttft_minus_runner_ms']['p50'] for r in rs], '{:.3f}')} "
                  f"| {cell([r['client_ttft_minus_runner_ms']['p99'] for r in rs], '{:.3f}')} |")
        print("\nStage breakdown (ms, median over runs of each run's p50 / p99):\n")
        keys = list(next(iter(ttft.values()))[0]["stages_ms"])
        cols = sorted(ttft)
        print("| stage | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for k in keys:
            row = []
            for c in cols:
                rs = ttft[c]
                row.append(f"{statistics.median(r['stages_ms'][k]['p50'] for r in rs):.3f} / "
                           f"{statistics.median(r['stages_ms'][k]['p99'] for r in rs):.3f}")
            print(f"| {k} | " + " | ".join(row) + " |")
        prompt = next(iter(ttft.values()))[0]["prompt_tokens"]
        print(f"\nPrompt: {prompt} tokens.\n")
    if decode:
        print("### Engine CPU per decode step (µs, excluding the runner's calls and device time)\n")
        print("| batch | runs | p50 | p99 | runner launch calls, p50 | engine process cores |")
        print("|---|---|---|---|---|---|")
        for b, rs in sorted(decode.items()):
            print(f"| {b} | {len(rs)} | {cell([r['engine_cpu_us']['p50'] for r in rs], '{:.0f}')} "
                  f"| {cell([r['engine_cpu_us']['p99'] for r in rs], '{:.0f}')} "
                  f"| {cell([r['runner_calls_us']['p50'] for r in rs], '{:.0f}')} "
                  f"| {cell([r['cpu_cores']['engine'] for r in rs], '{:.2f}')} |")
        print("\nMean per step by loop phase (µs; `wait` is device time, the fake spins through it):\n")
        phases = list(next(iter(decode.values()))[0]["phases_us_per_step"])
        print("| batch | " + " | ".join(phases) + " |")
        print("|---|" + "---|" * len(phases))
        for b, rs in sorted(decode.items()):
            print(f"| {b} | " + " | ".join(f"{statistics.median(r['phases_us_per_step'][p] for r in rs):.1f}"
                                           for p in phases) + " |")
        print()
    if idle:
        print("### GPU idle caused by the CPU (classification-shaped: ~1,024 tokens in, 64 out)\n")
        print("| concurrency | runs | idle with work queued | device busy, % of wall | steps | requests |")
        print("|---|---|---|---|---|---|")
        for c, rs in sorted(idle.items()):
            print(f"| {c} | {len(rs)} | {cell([r['gpu_idle_with_work_pct'] for r in rs], '{:.2f}%')} "
                  f"| {cell([r['device_busy_pct_of_wall'] for r in rs], '{:.1f}')} "
                  f"| {cell([r['steps'] for r in rs], '{:.0f}')} | {cell([r['requests'] for r in rs], '{:.0f}')} |")
        print()
    if sse:
        print("### SSE throughput, one frontend (128 streams × 256 tokens, 0 ms fake steps)\n")
        print("| runs | delivered events/s | frontend CPU µs/event | frontend cores | engine cores |")
        print("|---|---|---|---|---|")
        print(f"| {len(sse)} | {cell([r['events_per_s'] for r in sse], '{:,.0f}')} "
              f"| {cell([r['frontend_cpu_us_per_event'] for r in sse], '{:.2f}')} "
              f"| {cell([r['frontend_cores'] for r in sse], '{:.2f}')} "
              f"| {cell([r['engine_cores'] for r in sse], '{:.2f}')} |")


if __name__ == "__main__":
    main()

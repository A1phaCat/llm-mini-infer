#!/usr/bin/env python3
"""Compare arms (servers, or one server's settings) and passes of the same loads: bench/load/run.py's reports, and
bench/structured/validity.py's, side by side.

    uv run python bench/load/compare.py A1=results/A1 A2=results/A2 B1=results/B1 B2=results/B2 --floor A1,A2

Each argument is an arm, `[NAME=]PATH`: a folder (every run.py and validity.py report in it, told apart by their
contents; other JSON files are skipped) or one report. The name defaults to the folder's name, or the file's stem. An
arm holds one report of each workload's load at most.

**Per workload and load,** one row per arm: the client's numbers, and the engine's from the server's counters over the
window (run.py's `engine`, `llm_mini_infer.hostcheck.engine_delta`). A dash: the server does not count it, its counters
were not read, or the window had none.

**Answers:** for each pair of arms, the share of the answers both hold (run.py's by workload index, validity.py's by
id) that are identical: the same digest of the content (validity: the same text), the same token count and finish
reason. Without batch invariance a server's answers under load move with the batches it happened to form, so two
passes of one arm (`--floor A1,A2`; several floors: the lowest counts) are the floor every other pair is judged
against: the paired gap of its agreement to the floor's, with a 95% interval (`tests/scripts/parity_compare.py`); a pair
whose interval reaches 0 is not significantly below the floor.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VALIDITY = "validity"


def read_arm(spec: str) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    """An arm's name and its sections: (workload, load) -> the load's row and answers; (VALIDITY, "") -> the validity
    report's `all` summary and its answers."""
    name, sep, path = spec.partition("=")
    if not sep:
        name, path = "", spec
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{spec}: no such file or folder")
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    name = name or (p.name if p.is_dir() else p.stem)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for f in files:
        try:
            rep = json.loads(f.read_text())
        except ValueError:
            continue
        if not isinstance(rep, dict):
            continue
        if isinstance(rep.get("loads"), list):  # run.py's
            wl = rep.get("workload") or {}
            workload = str(wl.get("name") or wl.get("mode") or "?")
            for load in rep["loads"]:
                key = (workload, str(load.get("load")))
                if key in out:
                    raise SystemExit(f"{name}: two reports of {workload}, {key[1]} ({f}): name each one NAME=FILE")
                # Only a session load's first turn compares: later turns ("index.turn") follow each engine's replies.
                answers = {k: tuple(v) for k, v in (load.get("answers") or {}).items() if "." not in k}
                out[key] = {"row": load, "answers": answers}
        elif isinstance(rep.get("results"), list) and isinstance(rep.get("summary"), dict):  # validity.py's
            if (VALIDITY, "") in out:
                raise SystemExit(f"{name}: two validity reports ({f}): name each one NAME=FILE")
            out[(VALIDITY, "")] = {"row": rep["summary"].get("all") or {},
                                   "answers": {r["id"]: (r.get("text"), r.get("tokens"), r.get("finish"))
                                               for r in rep["results"] if "error" not in r and "id" in r}}
    if not out:
        raise SystemExit(f"{spec}: no run.py or validity.py report")
    return name, out


# -- the tables --------------------------------------------------------------------------------------------------------

def fmt(v: Any, nd: int | None = None) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and nd is not None:
        return f"{v:.{nd}f}"
    return f"{v:g}" if isinstance(v, float) else str(v)


def pct(d: dict | None, k: str) -> str:
    return "-" if not d or not d.get("n") else fmt(d.get(k))


def render(head: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    rows = [head, *rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in rows]
    return "\n".join([lines[0], "  ".join("-" * w for w in widths), *lines[1:]])


def client_row(arm: str, r: dict[str, Any]) -> tuple[str, ...]:
    finish = " ".join(f"{k}:{v}" for k, v in sorted((r.get("finish") or {}).items()))
    tps = r.get("output_tokens_per_s")
    return (arm, f"{r.get('ok')}/{r.get('measured')}", finish or "-", fmt(r.get("served_rps")), fmt(tps),
            pct(r.get("e2e_ms"), "p50"), pct(r.get("e2e_ms"), "p99"), pct(r.get("ttft_ms"), "p50"),
            pct(r.get("ttft_ms"), "p99"), pct(r.get("tpot_ms"), "p50"), pct(r.get("tpot_ms"), "p99"),
            pct(r.get("reply_tokens"), "mean"), hit_rate(r.get("cache_hit_rate")))


def hit_rate(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def by_turn_line(arm: str, r: dict[str, Any]) -> str | None:
    turns = r.get("cache_by_turn")
    if not turns:
        return None
    return f"  {arm} by turn: " + ", ".join(f"{t['turn']} {hit_rate(t.get('hit_rate'))} ({t['requests']})"
                                            for t in turns)


CLIENT = ("arm", "ok", "finish", "req/s", "tok/s", "e2e p50", "e2e p99", "ttft p50", "ttft p99", "tpot p50",
          "tpot p99", "reply", "hit rate")


def engine_row(arm: str, e: dict[str, Any] | None) -> tuple[str, ...]:
    if not e:
        return (arm, *["-"] * (len(ENGINE) - 1))
    acc = e.get("accept") or {}
    union = e.get("union_tokens") or {}
    eos = e.get("eos_free_finishes")
    eos_cell = "-" if eos is None else f"{round(sum(eos.values()))}" + (f" ({round(eos['yes'])})" if eos.get("yes")
                                                                         else "")
    return (arm, fmt(acc.get("tokens_per_verify")), "/".join(f"{x:.2f}" for x in acc.get("per_pos") or []) or "-",
            fmt(e.get("sparse_share")), fmt(union.get("mean"), 1) if union else "-",
            f"<={union['p90']}" if union.get("p90") is not None else "-", fmt(e.get("draft_universe_tokens")),
            fmt(e.get("forced_tokens_per_request")), eos_cell, fmt(e.get("gpu_idle_pct")),
            f"{fmt(e.get('requests_failed'))}/{fmt(e.get('grammar_errors'))}")


ENGINE = ("arm", "tok/verify", "accept per pos", "sparse", "union", "union p90", "universe", "forced/req",
          "eos-free", "idle %", "failed/grammar")


def validity_row(arm: str, s: dict[str, Any]) -> tuple[str, ...]:
    return (arm, fmt(s.get("requests")), f"{fmt(s.get('valid'))}/{fmt(s.get('stop'))}", fmt(s.get("invalid")),
            fmt(s.get("duplicate_keys")), fmt(s.get("length")), fmt(s.get("errors")), fmt(s.get("mean_tokens"), 1))


VALIDITY_HEAD = ("arm", "requests", "valid/stop", "invalid", "repeated keys", "length", "errors", "mean tokens")


# -- the answers -------------------------------------------------------------------------------------------------------

def same(a: dict[str, tuple], b: dict[str, tuple], keys: list[str]) -> list[int]:
    return [int(a[k] == b[k]) for k in keys]


def agreement(arms: dict[str, dict[str, tuple]], floors: list[tuple[str, str]]) -> list[str]:
    """Each pair's agreement over the answers both hold; with floors, each other pair's paired gap to the lowest floor
    over the answers all of them hold."""
    names = list(arms)
    lines = []
    floor_pairs = [(a, b) for a, b in floors if a in arms and b in arms]
    for x, y in itertools.combinations(names, 2):
        keys = sorted(set(arms[x]) & set(arms[y]))
        if not keys:
            lines.append(f"  {x} vs {y}: no answers in common")
            continue
        s = same(arms[x], arms[y], keys)
        line = f"  {x} vs {y}: {sum(s)}/{len(keys)} identical ({100 * sum(s) / len(keys):.1f}%)"
        if floor_pairs and (x, y) not in floor_pairs and (y, x) not in floor_pairs:
            common = set(keys)
            for a, b in floor_pairs:
                common &= set(arms[a]) & set(arms[b])
            common_keys = sorted(common)
            if common_keys:
                import numpy as np

                from tests.scripts.parity_compare import bootstrap_min_gap

                ours = np.array(same(arms[x], arms[y], common_keys))
                own = {f"{a} vs {b}": np.array(same(arms[a], arms[b], common_keys)) for a, b in floor_pairs}
                g = bootstrap_min_gap(ours, own)
                lo, hi = g["ci95"]
                verdict = "not below the floor" if hi >= 0 else "BELOW the floor"
                line += (f"; against the floor ({g['lowest']}, {len(common_keys)} answers): gap {100 * g['gap']:+.1f} "
                         f"points [{100 * lo:+.1f}, {100 * hi:+.1f}]: {verdict}")
        lines.append(line)
    return lines


def floor_pair(text: str) -> tuple[str, str]:
    a, sep, b = text.partition(",")
    if not sep or not a or not b:
        raise argparse.ArgumentTypeError(f"{text!r}: two arms, A,B")
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="[NAME=]PATH", help="an arm: a folder of reports, or one report")
    ap.add_argument("--floor", type=floor_pair, action="append", default=[], metavar="A,B",
                    help="two passes of one arm: the agreement every other pair is judged against")
    args = ap.parse_args(argv)
    if args.floor:
        sys.path.insert(0, str(ROOT))  # tests/scripts/parity_compare.py
    arms: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for spec in args.arms:
        name, sections = read_arm(spec)
        if name in arms:
            raise SystemExit(f"two arms named {name}: name them NAME=PATH")
        arms[name] = sections
    for a, b in args.floor:
        if a not in arms or b not in arms:
            raise SystemExit(f"--floor {a},{b}: no arm named {a if a not in arms else b}")
    order = list(dict.fromkeys(k for sections in arms.values() for k in sections))
    blocks = []
    for key in order:
        held = {name: sections[key] for name, sections in arms.items() if key in sections}
        if key[0] == VALIDITY:
            parts = ["== validity.py", render(VALIDITY_HEAD, [validity_row(n, s["row"]) for n, s in held.items()])]
        else:
            parts = [f"== {key[0]}: {key[1]}",
                     render(CLIENT, [client_row(n, s["row"]) for n, s in held.items()]),
                     "(latencies in ms; reply: mean output tokens)",
                     *[line for n, s in held.items() if (line := by_turn_line(n, s["row"]))], "",
                     render(ENGINE, [engine_row(n, s["row"].get("engine")) for n, s in held.items()]),
                     "(union: the mean tokens of the steps' mask unions, and the bucket bound of their p90;\n"
                     " eos-free: finishes, those with a row in flight in brackets; idle %: GPU idle caused by the CPU)"]
        answers = {n: s["answers"] for n, s in held.items() if s["answers"]}
        if len(answers) > 1:
            parts += ["", "answers (identical: the same content, tokens and finish reason):",
                      *agreement(answers, args.floor)]
        blocks.append("\n".join(parts))
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())

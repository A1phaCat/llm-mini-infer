"""Where a fuzz workload's time goes, per suite (`tests/scripts/fuzz_campaign.py`'s suites, the campaign's settings).

    uv run python -m tests.scripts.fuzz_profile                        # every suite, 300 workloads each, seed 1
    uv run python -m tests.scripts.fuzz_profile --suites window cache_sched --examples 1000
    uv run python -m tests.scripts.fuzz_profile --no-profile --jobs 1  # milliseconds per workload, one run at a time

Each suite runs as its own pytest process under `HYPOTHESIS_PROFILE=long`, first plain (its test's duration: the
milliseconds per workload a campaign pays), then under cProfile, whose cumulative times split a workload into:

- **checks**: the strict fake's `_check_kv` and `_check_sched`, and `Scheduler.check_accounting` (the radix tree's
  and the encoder cache's `check` inside it, also shown alone);
- **sched**: the engine turn (`EngineCore.step`, `handle`) without the runner's calls: the scheduler, the cache, the
  grammar masks;
- **fake**: the fake runner's simulation (`launch_forward`, `launch_sample`, `wait`) without its checks;
- **harness**: the test harness around them (the plan recorder, adding and aborting requests, the final checks);
- **draw**: the rest of the test's time, Hypothesis drawing the workloads and the harness's setup.

cProfile slows Python calls more than NumPy's, so the shares are approximate; the plain run's milliseconds are the
measurement. Timings need an otherwise idle machine. Results land in `results/fuzz-profile/<time>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import pstats
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..support.paths import RESULTS, ROOT
from .fuzz_campaign import SUITES

_CALL = re.compile(r"([\d.]+)s call ")
# (file suffix, function name) of the profile's boundaries
_FUNCS = {
    "test": ("hypothesis/core.py", "wrapped_test"),
    "run": ("tests/support/harness.py", "run"),
    "final": ("tests/support/harness.py", "check"),
    "step": ("engine/core.py", "step"),
    "handle": ("engine/core.py", "handle"),
    "record": ("tests/support/harness.py", "record"),
    "forward": ("runner/fake.py", "launch_forward"),
    "sample": ("runner/fake.py", "launch_sample"),
    "wait": ("runner/fake.py", "wait"),
    "mask_buffer": ("runner/fake.py", "mask_buffer"),
    "forward_pending": ("runner/fake.py", "forward_pending"),
    "check_kv": ("runner/fake_checks.py", "_check_kv"),
    "check_sched": ("runner/fake_checks.py", "_check_sched"),
    "accounting": ("sched/scheduler.py", "check_accounting"),
    "radix_check": ("cache/radix.py", "check"),
    "encoder_check": ("cache/encoder.py", "check"),
}


def _cumulative(path: Path) -> dict[str, float]:
    stats = pstats.Stats(str(path)).stats  # type: ignore[attr-defined]
    out = dict.fromkeys(_FUNCS, 0.0)
    for (file, _, name), (_, _, _, cum, _) in stats.items():
        for key, (suffix, fn) in _FUNCS.items():
            if name == fn and file.replace(os.sep, "/").endswith(suffix):
                out[key] += cum
    return out


def split(c: dict[str, float]) -> dict[str, float]:
    """A profile's cumulative times as the module docstring's shares (seconds)."""
    runner = c["record"] + c["sample"] + c["wait"] + c["mask_buffer"] + c["forward_pending"]
    checks = c["check_kv"] + c["check_sched"] + c["accounting"]
    return {"checks": checks,
            "sched": c["step"] + c["handle"] - runner,
            "fake": c["forward"] + c["sample"] + c["wait"] - c["check_kv"] - c["check_sched"],
            "harness": c["run"] + c["final"] - c["step"] - c["handle"] - c["accounting"] + c["record"] - c["forward"],
            "draw": c["test"] - c["run"] - c["final"],
            "radix_check": c["radix_check"], "encoder_check": c["encoder_check"], "total": c["test"]}


def run_one(suite: str, seed: int, examples: int, out: Path, profile: bool) -> dict:
    env = {**os.environ, "HYPOTHESIS_PROFILE": "long", "FUZZ_EXAMPLES": str(examples), "FUZZ_SEED": str(seed),
           "FUZZ_FAILURES": str(out / "failures")}
    test = [SUITES[suite], "-q", "-p", "no:cacheprovider", f"--hypothesis-seed={seed}"]
    proc = subprocess.run([sys.executable, "-m", "pytest", *test, "--durations=1", "--durations-min=0"], cwd=ROOT,
                          env=env, capture_output=True, text=True)
    m = _CALL.search(proc.stdout)
    r: dict = {"suite": suite, "status": "passed" if proc.returncode == 0 else "FAILED",
               "ms_per_workload": round(float(m.group(1)) * 1000 / examples, 2) if m else None}
    if profile and proc.returncode == 0:
        prof = out / f"{suite}.prof"
        subprocess.run([sys.executable, "-m", "cProfile", "-o", str(prof), "-m", "pytest", *test], cwd=ROOT, env=env,
                       capture_output=True, text=True, check=True)
        r["seconds"] = {k: round(v, 3) for k, v in split(_cumulative(prof)).items()}
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", choices=sorted(SUITES), default=list(SUITES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--examples", type=int, default=300, help="workloads per suite")
    ap.add_argument("--jobs", type=int, default=0, help="parallel suites (default: half the CPU count)")
    ap.add_argument("--no-profile", dest="profile", action="store_false", help="the plain timing only")
    args = ap.parse_args()
    out = RESULTS / "fuzz-profile" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    jobs = args.jobs or max(1, (os.cpu_count() or 2) // 2)
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(lambda s: run_one(s, args.seed, args.examples, out, args.profile), args.suites))
    (out / "summary.json").write_text(json.dumps({"examples": args.examples, "seed": args.seed, "runs": results},
                                                 indent=1))
    keys = ("checks", "sched", "fake", "harness", "draw")
    print(f"{'suite':<18} {'ms/wl':>7}  " + "  ".join(f"{k:>7}" for k in keys) + "   radix  encoder")
    for r in results:
        line = f"{r['suite']:<18} {r['ms_per_workload'] or float('nan'):>7.2f}  "
        if "seconds" in r:
            s, total = r["seconds"], r["seconds"]["total"] or 1.0
            line += "  ".join(f"{s[k] / total:>7.0%}" for k in keys)
            line += f"  {s['radix_check'] / total:>6.0%}  {s['encoder_check'] / total:>7.0%}"
        print(line + ("" if r["status"] == "passed" else "  FAILED"))
    print(f"results in {out.relative_to(ROOT)}")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Derives summary.json and summary.md from the published reports in this folder.

Usage: python3 bench/reports/v0.8.0/summarize.py

It reads runs/, manifests/, control/ and bounds/ beside the script and writes summary.json (raw numbers) and
summary.md (the tables) next to them. Standard library only. A cell that did not run is null in the JSON and reads
"not run" or the manifest's status in the Markdown. Cross-checks that disagree with the published files are printed
to standard error.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("gemma", "qwen")
ENGINES = ("ours", "vllm", "sglang")
LOADS = ("scene-c1", "chat-c1", "scene-c16", "scene-c64", "chat-c16", "chat-c64")
HOST_ORDER = ("p3", "p2", "p1")
Q_HOSTS = ("q-vllm", "q-sglang")
C_ARMS = ("ours", "ours-nobi", "vllm", "vllm-bi", "vllm-bi-nodg", "sglang", "sglang-bi")
NON_DEFAULT = {"ours": "ours-nobi", "vllm": "vllm-bi", "sglang": "sglang-bi"}
I_OTHER = {"ours": "ours-nobi", "vllm": "vllm-bi-nodg", "sglang": "sglang-bi"}
BOUND_FROM = {"gemma": ("p1", ("ours", "vllm", "sglang")), "qwen": ("p2", ("ours", "vllm"))}
B_ARMS = ("vllm-k1", "vllm-k4")
Z = 1.959964
MODEL_TITLE = {"gemma": "Gemma", "qwen": "Qwen"}
WARNINGS: list[str] = []


def read(path: Path) -> dict | None:
    """The parsed JSON file, or None when it does not exist."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def host_of(model: str, family: str, unit: str, *names: str) -> str | None:
    """The host holding runs/<host>/<model>/<family>/<unit>/<each name> (the folder alone without names): p3, p2, p1."""
    for host in Q_HOSTS if family == "Q" else HOST_ORDER:
        path = HERE / "runs" / host / model / family / unit
        if all((path / name).exists() for name in names) and path.exists():
            return host
    return None


def run_file(model: str, family: str, unit: str, name: str, host: str | None = None) -> dict | None:
    """One report under runs/, from the given host or else the host that holds it."""
    host = host or host_of(model, family, unit, name)
    return None if host is None else read(HERE / "runs" / host / model / family / unit / name)


def first_load(report: dict | None) -> dict | None:
    """The first load entry of a run.py report."""
    if not report or not report.get("loads"):
        return None
    return report["loads"][0]


def num(d: dict | None, key: str) -> float | None:
    """A field of a dict that may be absent."""
    return None if d is None else d.get(key)


def load_cell(load: dict | None) -> dict | None:
    """The A1 metrics of one load."""
    if load is None:
        return None
    e2e, ttft, tpot = load.get("e2e_ms"), load.get("ttft_ms"), load.get("tpot_ms")
    return {
        "n": load.get("ok"),
        "failed": load.get("failed"),
        "dropped": load.get("dropped"),
        "e2e_p50": num(e2e, "p50"),
        "e2e_p90": num(e2e, "p90"),
        "e2e_p99": num(e2e, "p99"),
        "e2e_mean": num(e2e, "mean"),
        "ttft_p50": num(ttft, "p50"),
        "tpot_p50": num(tpot, "p50"),
        "tpot_p90": num(tpot, "p90"),
        "tpot_p99": num(tpot, "p99"),
        "tpot_mean": num(tpot, "mean"),
        "served_rps": load.get("served_rps"),
        "tok_s": load.get("output_tokens_per_s"),
        "reply_mean": num(load.get("reply_tokens"), "mean"),
        "tokens_per_verify": num(load.get("spec"), "tokens_per_verify"),
    }


def wilson(x: int, n: int) -> tuple[float, float]:
    """The 95% Wilson score interval for x of n, in percent."""
    p = x / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def manifests() -> dict[str, dict]:
    """Every manifest, keyed host/model/family/unit."""
    out = {}
    for path in sorted((HERE / "manifests").glob("*/*/*/*.json")):
        host, model, family = path.parts[-4:-1]
        data = json.loads(path.read_text(encoding="utf-8"))
        out[f"{host}/{model}/{family}/{path.stem}"] = data
    return out


MANIFESTS = manifests()


def rank(host: str) -> int:
    """The position of a host in the order of preference."""
    return HOST_ORDER.index(host) if host in HOST_ORDER else len(HOST_ORDER)


def find_manifest(model: str, family: str, unit: str, host: str | None = None) -> tuple[str | None, dict | None]:
    """The host and manifest of the given host, else of the host the unit's data comes from, else of any."""
    hosts = (host, host_of(model, family, unit), *sorted({k.split("/")[0] for k in MANIFESTS}, key=rank))
    for candidate in hosts:
        if candidate and (key := f"{candidate}/{model}/{family}/{unit}") in MANIFESTS:
            return candidate, MANIFESTS[key]
    return None, None


def manifest_of(model: str, family: str, unit: str, host: str | None = None) -> dict | None:
    """The manifest of model/family/unit (see find_manifest)."""
    return find_manifest(model, family, unit, host)[1]


def kernel_label(resolved: dict) -> str | None:
    """The FP8 kernel of a manifest: llm-mini-infer's list of shape rules is CUTLASS."""
    kernel = resolved.get("fp8_kernel")
    return "CUTLASS" if isinstance(kernel, list) else kernel


def missing_status(model: str, unit: str, family: str | None = None) -> str:
    """Why a unit has no data: its manifest's status on the preferred host that has one, or "not run"."""
    found = sorted(
        ((rank(host), data) for key, data in MANIFESTS.items()
         for host, m, fam, u in [key.split("/")] if m == model and u == unit and family in (None, fam)),
        key=lambda pair: pair[0],
    )
    for r, data in found:
        if r != found[0][0]:
            break
        if data.get("status") not in (None, "ok"):
            return str(data["status"])
    return "not run"


def a1_cells() -> dict:
    """Family A1: every engine's six loads, all from the first host that holds all six (else the one with c=1)."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for engine in ENGINES:
            files = [f"{load}.json" for load in LOADS]
            host = host_of(model, "A1", engine, *files) or host_of(model, "A1", engine, "scene-c1.json")
            out[model][engine] = {"host": host} | {
                load: load_cell(first_load(run_file(model, "A1", engine, f"{load}.json", host))) if host else None
                for load in LOADS
            }
    return out


def bound_cells() -> dict:
    """The bound, recomputed from the A1 one-in-flight reports it was frozen from and compared with the file."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        host, frozen_from = BOUND_FROM[model]
        for workload in ("scene", "chat"):
            engines = {}
            for engine in frozen_from:
                cell = load_cell(first_load(run_file(model, "A1", engine, f"{workload}-c1.json", host)))
                if cell and not cell["failed"] and not cell["dropped"] and cell["e2e_p50"] is not None:
                    engines[engine] = cell["e2e_p50"]
            file = read(HERE / "bounds" / f"bound-{model}-{workload}.json")
            median = statistics.median(engines.values()) if engines else None
            value = None if median is None else 2 * median
            in_file = num(file, "value")
            if file is not None and value is not None and abs(value - in_file) > 0.05:
                WARNINGS.append(f"bound {model}-{workload}: recomputed {value} differs from the file's {in_file}")
            out[model][workload] = {
                "host": host,
                "median_ms": median,
                "value_ms": value,
                "value_in_file_ms": in_file,
                "engines": engines,
            }
    return out


def probe_passes(load: dict, bound: float | None) -> bool:
    """The A2 pass rule applied to one probe's load."""
    offered = load.get("offered_rps") or 0
    p99 = (load.get("e2e_ms") or {}).get("p99")
    return (
        load.get("failed") == 0
        and load.get("dropped") == 0
        and (load.get("served_rps") or 0) >= 0.95 * offered
        and p99 is not None
        and bound is not None
        and p99 <= bound
    )


def a2_cells(bounds: dict) -> dict:
    """Family A2: the rate search, with every probe's verdict recomputed."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for engine in ENGINES:
            host = host_of(model, "A2", engine, "find-rate.json")
            report = run_file(model, "A2", engine, "find-rate.json", host)
            if report is None or "search" not in report:
                out[model][engine] = None
                continue
            search = report["search"]
            bound = bounds[model]["scene"]["value_in_file_ms"]
            # The loads are the probes in the order run (a rate's confirmation repeats its rate): pair them in order
            checked = mismatches = 0
            for probe, load in zip(search.get("probes", []), report.get("loads", []), strict=False):
                checked += 1
                if probe_passes(load, bound) != probe["passed"]:
                    mismatches += 1
            out[model][engine] = {
                "host": host,
                "best_rate": search.get("best_rate"),
                "confirmed": search.get("confirmed"),
                "first_failing_rate": search.get("first_failing_rate"),
                "low": search.get("low"),
                "high": search.get("high"),
                "probes_checked": checked,
                "probe_verdict_mismatches": mismatches,
            }
    return out


def compare_text(model: str, arm: str, host: str) -> tuple[int, int] | None:
    """The identical/N line of a published comparison text on the unit's host, for the cross-check."""
    path = HERE / "runs" / host / model / "C" / f"{arm}-compare.txt"
    if not path.exists():
        return None
    found = re.findall(r"p1 vs p2: (\d+)/(\d+) identical", path.read_text(encoding="utf-8"))
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def c_cells() -> dict:
    """Family C: the share of answers identical between two fresh-server passes, with the arm's kernel."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for arm in C_ARMS:
            if model == "qwen" and arm == "vllm-bi-nodg":
                continue
            host = host_of(model, "C", f"{arm}-p1", "stability.json")
            man_host, manifest = find_manifest(model, "C", f"{arm}-p1", host)
            resolved = (manifest or {}).get("resolved") or {}
            meta = {
                "host": host or man_host, "fp8_kernel": kernel_label(resolved),
                "batch_invariant_active": resolved.get("batch_invariant_active"),
            }
            answers = []
            for p in ("p1", "p2"):
                load = first_load(run_file(model, "C", f"{arm}-{p}", "stability.json", host)) if host else None
                answers.append(None if load is None else load.get("answers"))
            if answers[0] is None or answers[1] is None:
                statuses = [(manifest_of(model, "C", f"{arm}-{p}", man_host) or {}).get("status") for p in ("p1", "p2")]
                bad = [st for st in statuses if st not in (None, "ok")]
                status = bad[0] if bad else missing_status(model, arm)
                out[model][arm] = meta | {
                    "identical": None, "n": None, "pct": None, "ci_lo_pct": None, "ci_hi_pct": None, "status": status,
                    "distinct_p1": None, "distinct_p2": None, "top_p1": None, "top_p2": None,
                }
                continue
            both = sorted(set(answers[0]) & set(answers[1]), key=int)
            same = sum(1 for i in both if answers[0][i] == answers[1][i])
            n = len(both)
            lo, hi = wilson(same, n) if n else (None, None)
            counts = [Counter(answers[k][i][0] for i in both) for k in (0, 1)]
            out[model][arm] = meta | {
                "distinct_p1": len(counts[0]), "distinct_p2": len(counts[1]),
                "top_p1": max(counts[0].values(), default=0), "top_p2": max(counts[1].values(), default=0),
                "identical": same, "n": n, "pct": 100 * same / n if n else None, "ci_lo_pct": lo, "ci_hi_pct": hi,
                "status": "ok",
            }
            text = compare_text(model, arm, host)
            if text is not None and text != (same, n):
                WARNINGS.append(f"C {model}/{arm}: recount {same}/{n} differs from the comparison text {text}")
    return out


def c_vllm_kernels(c: dict) -> dict:
    """vLLM's batch invariance on Gemma by FP8 kernel: DeepGEMM on (vllm-bi) and off (vllm-bi-nodg)."""
    out: dict = {}
    for arm in ("vllm-bi", "vllm-bi-nodg"):
        cell = c["gemma"][arm]
        out[arm] = {
            "deep_gemm": "on (vLLM's default)" if arm == "vllm-bi" else "off (VLLM_USE_DEEP_GEMM=0)",
            **{k: cell[k] for k in (
                "host", "fp8_kernel", "batch_invariant_active", "identical", "n", "pct", "ci_lo_pct", "ci_hi_pct",
                "status", "distinct_p1", "distinct_p2", "top_p1", "top_p2")},
        }
    return out


COST_KEYS = ("tok_s", "tpot_p50", "served_rps", "reply_mean")


def cost_cell(model: str, workload: str, default: tuple[str, str], other: tuple[str, str], source: str) -> dict:
    """One engine's default load against its other-mode load, both from the host of the other-mode load."""
    (d_family, d_unit), (o_family, o_unit) = default, other
    name = f"{workload}-c64.json"
    o_host = host_of(model, o_family, o_unit, name)
    d_host = host_of(model, d_family, d_unit, name)
    cell: dict = {"host": o_host, "source": source, "other_arm": o_unit.removesuffix("-p1"), "status": "ok"}
    d = o = None
    if o_host is None:
        cell["host"] = find_manifest(model, o_family, o_unit)[0]
        cell["status"] = missing_status(model, o_unit, o_family)
    else:
        o = load_cell(first_load(run_file(model, o_family, o_unit, name, o_host)))
        d = load_cell(first_load(run_file(model, d_family, d_unit, name, o_host)))
        if d is None:
            cell["status"] = "cross-host" if d_host else missing_status(model, d_unit, d_family)
            cell["default_host"] = d_host
            cell["host"] = None if d_host else o_host
            cell["other_host"] = o_host
    for key in COST_KEYS:
        cell[f"default_{key}"] = num(d, key) if d and o else None
        cell[f"other_{key}"] = num(o, key) if d and o else None
    cell["ratio"] = o["tok_s"] / d["tok_s"] if d and o and d["tok_s"] else None
    return cell


def cost_cells() -> dict:
    """The other mode at c = 64 against the default mode, per engine: family I on Gemma, A1 against C pass 1 on Qwen."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for engine in ENGINES:
            out[model][engine] = {}
            for workload in ("scene", "chat"):
                if model == "gemma":
                    cell = cost_cell(model, workload, ("I", engine), ("I", I_OTHER[engine]), "I")
                else:
                    cell = cost_cell(
                        model, workload, ("A1", engine), ("C", f"{NON_DEFAULT[engine]}-p1"), "A1 vs C pass 1"
                    )
                out[model][engine][workload] = cell
    return out


def d_cells() -> dict:
    """Family D: cold and warm start-up."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for engine in ENGINES:
            host = (
                host_of(model, "D", engine, "startup.json") or host_of(model, "D", engine, "startup-cold.json")
                or host_of(model, "D", engine, "startup-warm.json")
            )
            startup = run_file(model, "D", engine, "startup.json", host)
            cold = warm = None
            if startup and "starts" in startup:
                for start in startup["starts"]:
                    if start.get("kind") == "cold":
                        cold = start
                    elif start.get("kind") == "warm":
                        warm = start
            else:
                cold = run_file(model, "D", engine, "startup-cold.json", host)
                warm = run_file(model, "D", engine, "startup-warm.json", host)
                manifest = manifest_of(model, "D", engine, host)
                if warm is None and manifest and manifest.get("startup"):
                    warm = manifest["startup"]
                    cold = warm.get("cold")
            if cold is None and warm is None:
                out[model][engine] = None
                continue
            out[model][engine] = {
                "host": host,
                "cold_first_answer_s": num(cold, "first_answer_s"),
                "warm_first_answer_s": num(warm, "first_answer_s"),
                "cold_ready_s": num(cold, "ready_s"),
                "warm_ready_s": num(warm, "ready_s"),
            }
    return out


def s_cells() -> dict:
    """Validity of the structured replies; a missing row says why when the manifests do."""
    keys = ("requests", "stop", "valid", "invalid", "duplicate_keys", "length", "errors")
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for engine in ENGINES:
            host = host_of(model, "D", engine, "validity.json")
            report = run_file(model, "D", engine, "validity.json", host)
            summary = ((report or {}).get("summary") or {}).get("all")
            status, cell_host = "ok", host
            if summary is None:
                summary = {}
                if model == "gemma" and engine == "sglang":
                    manifest = manifest_of(model, "D", engine, "p2") or {}
                    if (manifest.get("resolved") or {}).get("notes") or manifest.get("notes"):
                        status, cell_host = "struck (see README)", "p2"
                elif model == "qwen" and engine == "sglang":
                    manifest = manifest_of(model, "D", engine, "p3") or {}
                    if any("--skip-validity" in str(d) for d in manifest.get("deviations") or []):
                        status, cell_host = "not run (no structured-output row for SGLang)", "p3"
                if status == "ok":
                    out[model][engine] = None
                    continue
            out[model][engine] = {"host": cell_host, "status": status} | {k: summary.get(k) for k in keys}
    return out


def b_cells() -> dict:
    """vLLM's own draft counts."""
    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for arm in B_ARMS:
            host = host_of(model, "B", arm, "scene-c1.json") or host_of(model, "B", arm, "scene-c64.json")
            c1 = load_cell(first_load(run_file(model, "B", arm, "scene-c1.json", host)))
            c64 = load_cell(first_load(run_file(model, "B", arm, "scene-c64.json", host)))
            if c1 is None and c64 is None:
                out[model][arm] = None
                continue
            out[model][arm] = {
                "host": host,
                "c1_tpot_p50": num(c1, "tpot_p50"),
                "c1_e2e_p50": num(c1, "e2e_p50"),
                "c1_tokens_per_verify": num(c1, "tokens_per_verify"),
                "c64_served_rps": num(c64, "served_rps"),
                "c64_tpot_p50": num(c64, "tpot_p50"),
            }
    return out


def controls() -> dict:
    """The GPU controls."""
    out: dict = {}
    for path in sorted((HERE / "control").glob("*.json")):
        phases = json.loads(path.read_text(encoding="utf-8")).get("phases", {})
        out[path.stem] = {
            "bf16_tflops": num(phases.get("bf16"), "tflops"),
            "fp8_tflops": num(phases.get("fp8"), "tflops"),
            "copy_gbs": num(phases.get("copy"), "gbs"),
            "sm_mhz_median_bf16": num(phases.get("bf16"), "sm_mhz_median"),
            "power_w_median_bf16": num(phases.get("bf16"), "power_w_median"),
        }
    return out


def q_cells(a1: dict, ctl: dict) -> dict:
    """The vendor-image check against the A1 host, after the host ratio of the start controls."""
    out: dict = {}
    for engine in ("vllm", "sglang"):
        q_load = load_cell(first_load(run_file("gemma", "Q", engine, "scene-c1.json", host=f"q-{engine}")))
        p1 = a1["gemma"][engine]["scene-c1"]
        p1_ctl, q_ctl = ctl.get("p1-start"), ctl.get(f"q-{engine}-start")
        if q_load is None or p1 is None or not p1_ctl or not q_ctl:
            out[engine] = None
            continue
        ratio = statistics.mean(
            [p1_ctl["bf16_tflops"] / q_ctl["bf16_tflops"], p1_ctl["fp8_tflops"] / q_ctl["fp8_tflops"]]
        )
        deviation = p1["tpot_p50"] / (q_load["tpot_p50"] / ratio) - 1
        out[engine] = {
            "host_ratio": ratio,
            "q_tpot_p50": q_load["tpot_p50"],
            "p1_tpot_p50": p1["tpot_p50"],
            "deviation": deviation,
            "verdict": "pass" if abs(deviation) <= 0.05 else "fail",
        }
    return out


def manifest_summary() -> dict:
    """The manifests' status, deviations and resolved settings."""
    out: dict = {}
    for key, data in MANIFESTS.items():
        resolved = data.get("resolved") or {}
        kernel = kernel_label(resolved)
        out[key] = {
            "status": data.get("status"),
            "status_line": data.get("status_line"),
            "k": data.get("k"),
            "k_source": data.get("k_source"),
            "batch_invariant": data.get("batch_invariant"),
            "deviations": data.get("deviations"),
            "resolved": {
                "kv_tokens": resolved.get("kv_tokens"),
                "max_running_requests": resolved.get("max_running_requests"),
                "attention_backend": resolved.get("attention_backend"),
                "fp8_kernel": kernel,
                "chunk_tokens": resolved.get("chunk_tokens"),
                "memory_fraction": resolved.get("memory_fraction"),
                "prefix_cache": resolved.get("prefix_cache"),
                "batch_invariant_active": resolved.get("batch_invariant_active"),
            },
            "versions": data.get("versions"),
        }
    return out


def clock_summary() -> dict:
    """The GPU clock, power and throttle readings of each load."""
    out: dict = {}
    for key, data in MANIFESTS.items():
        for load in data.get("loads") or []:
            out[f"{key}/{load['name']}"] = {
                "sm_mhz_median": load.get("sm_mhz_median"),
                "power_w_median": load.get("power_w_median"),
                "throttle": load.get("throttle"),
            }
    return out


def grammar_cost() -> dict:
    """llm-mini-infer's grammar compile cost per model, from the D family's grammar-cost.json on host p2."""
    keys = (
        "model", "vocab", "backend", "compact", "llguidance", "k", "tokenizer_build_s", "compile_corpus_ms_p50_max",
        "compile_json_object_ms", "compile_catalog_cold_ms", "compile_catalog_again_ms", "clone_us_p50",
        "masks_p50_p99",
    )
    out: dict = {}
    for model in MODELS:
        report = run_file(model, "D", "ours", "grammar-cost.json", host="p2")
        out[model] = None if report is None else {k: report.get(k) for k in keys}
    return out


CLOCK_TOLERANCE = 0.03


def clock_check(ctl_raw: dict[str, dict]) -> dict:
    """Per host: the median of the loads' SM clock medians, the loads more than 3% from it, and the bf16 controls."""
    by_host: dict[str, list[tuple[str, str, dict]]] = {}
    for key, data in MANIFESTS.items():
        host, rest = key.split("/", 1)
        for load in data.get("loads") or []:
            by_host.setdefault(host, []).append((rest, load["name"], load))
    out: dict = {}
    for host in sorted(set(by_host) | {name.rsplit("-", 1)[0] for name in ctl_raw}):
        loads = by_host.get(host, [])
        clocks = [load["sm_mhz_median"] for _, _, load in loads if load.get("sm_mhz_median") is not None]
        median = statistics.median(clocks) if clocks else None
        outliers = []
        for unit, name, load in loads:
            mhz = load.get("sm_mhz_median")
            if median is None or mhz is None or abs(mhz - median) / median <= CLOCK_TOLERANCE:
                continue
            outliers.append({
                "unit": unit, "load": name, "sm_mhz_median": mhz, "diff_pct": 100 * (mhz - median) / median,
                "power_w_median": load.get("power_w_median"), "throttle": load.get("throttle"),
            })
        controls_in_order = []
        for label in ("start", "middle", "end"):
            bf16 = (ctl_raw.get(f"{host}-{label}") or {}).get("phases", {}).get("bf16")
            if bf16 is not None:
                controls_in_order.append({
                    "control": label,
                    "bf16_sm_mhz_median": bf16.get("sm_mhz_median"),
                    "bf16_tflops": bf16.get("tflops"),
                })
        out[host] = {
            "loads_counted": len(clocks), "host_median_sm_mhz": median, "outliers": outliers,
            "controls": controls_in_order,
        }
    return out


def raw_controls() -> dict[str, dict]:
    """The control files as read, keyed by file stem."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE / "control").glob("*.json"))}


# Markdown


def f(value: float | int | None, digits: int = 1) -> str:
    """A number for a table cell."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value:.{digits}f}"


def active_label(value: bool | None) -> str:
    """The manifest's batch_invariant_active for a table cell."""
    return "not recorded" if value is None else str(value).lower()


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def a1_table(summary_cells: dict, model: str, conc: int) -> list[str]:
    """One A1 table at a concurrency."""
    rows = []
    for workload in ("scene", "chat"):
        for engine in ENGINES:
            cells = summary_cells["A1"][model][engine]
            c = cells[f"{workload}-c{conc}"]
            if c is None:
                rows.append([workload, engine, "-", missing_status(model, engine, "A1")] + ["-"] * 13)
                continue
            rows.append([
                workload, engine, cells["host"], f(c["n"]), f(c["failed"]), f(c["dropped"]), f(c["e2e_p50"]),
                f(c["e2e_p90"]), f(c["e2e_p99"]), f(c["e2e_mean"]), f(c["tpot_p50"], 2), f(c["tpot_p99"], 2),
                f(c["tpot_mean"], 2),
                f(c["served_rps"], 3), f(c["tok_s"]), f(c["reply_mean"]), f(c["tokens_per_verify"], 3),
            ])
    header = [
        "workload", "engine", "host", "N", "failed", "dropped", "e2e p50", "e2e p90", "e2e p99", "e2e mean", "TPOT p50",
        "TPOT p99", "TPOT mean", "req/s", "tok/s", "reply tokens", "tokens per verify",
    ]
    return table(header, rows)


STABILITY_NOTE = (
    "Distinct answers: how many different answer texts (by content digest) a pass gave to its N requests; most "
    "common answer: how many requests got the pass's most frequent one. Identical, in the identical/N column, also "
    "needs the same token count and finish reason. An identical share means little when a pass gives most requests "
    "one answer."
)
COST_NOTES = {
    "gemma": (
        "vLLM's other mode is `vllm-bi-nodg`, which also turns DeepGEMM off (CUTLASS block FP8): its ratio is the "
        "cost of the configuration in which its invariance holds, not of the invariance switch alone."
    ),
    "qwen": (
        "SGLang's other mode on Qwen3.8 also runs the triton attention backend (flashinfer in its default mode): "
        "its ratio includes that change."
    ),
}


def pair(c: dict, key: str) -> str:
    """A cell's p1 and p2 counts as "a / b", "-" when missing."""
    a, b = c.get(f"{key}_p1"), c.get(f"{key}_p2")
    return "-" if a is None or b is None else f"{a} / {b}"


def markdown(cells: dict, ctl: dict) -> str:
    """The summary tables."""
    out = [
        "# Benchmark summary",
        "",
        "Numbers are derived from the published run files by summarize.py. Latencies are in milliseconds, TPOT is "
        "milliseconds per output token, throughput is in output tokens per second. A dash is a value the file does "
        "not hold. The host column names the host whose files a row's numbers come from.",
        "",
    ]
    for model in MODELS:
        title = MODEL_TITLE[model]
        out += [f"## {title}", ""]
        for conc in (1, 16, 64):
            out += [f"### {title} A1, {conc} in flight (ms, tokens)", "", *a1_table(cells, model, conc)]
            if model == "qwen" and conc == 64 and cells["A1"]["qwen"]["sglang"]["scene-c64"]:
                out += [
                    "SGLang ran at 48 running requests (a named deviation: its fp32 mamba state does not fit 64), so "
                    "at 64 in flight 16 requests queue in its server.",
                    "",
                ]
        rows = []
        for workload in ("scene", "chat"):
            b = cells["bound"][model][workload]
            rows.append([
                workload, f(b["median_ms"]), f(b["value_ms"]), f(b["value_in_file_ms"]),
                ", ".join(f"{e} {f(v)}" for e, v in b["engines"].items()) or "-",
            ])
        out += [
            f"### {title} bound (from the A1 runs on host {BOUND_FROM[model][0]}; ms)", "",
            *table(
                ["workload", "median of e2e p50", "bound (2 x median)", "bound in file", "e2e p50 per engine"], rows
            ),
        ]
        rows = []
        for engine in ENGINES:
            a = cells["A2"][model][engine]
            if a is None:
                rows.append([engine, "-", missing_status(model, engine, "A2")] + ["-"] * 6)
                continue
            best = f(a["best_rate"], 2) if a["best_rate"] is not None else f"none at or above {f(a['low'], 2)}"
            rows.append([
                engine, a["host"], best, str(a["confirmed"]), f(a["first_failing_rate"], 2), f(a["low"], 2),
                f(a["high"], 2), f(a["probes_checked"]), f(a["probe_verdict_mismatches"]),
            ])
        out += [
            f"### {title} A2 highest rate within the bound (requests per second)", "",
            *table(["engine", "host", "best rate", "confirmed", "first failing", "low", "high", "probes checked",
                    "mismatches"], rows),
        ]
        rows = []
        for arm in C_ARMS:
            c = cells["C"][model].get(arm)
            if c is None:
                continue
            tail = [f(c["fp8_kernel"]), active_label(c["batch_invariant_active"]), c["status"]]
            if c["n"] is None:
                rows.append([arm, f(c["host"]), "-", "-", "-", "-", "-", "-", *tail])
                continue
            rows.append([
                arm, c["host"], f"{c['identical']}/{c['n']}", f(c["pct"]), f(c["ci_lo_pct"]), f(c["ci_hi_pct"]),
                pair(c, "distinct"), pair(c, "top"), *tail,
            ])
        out += [
            f"### {title} stability: answers identical between two passes (percent, 95% Wilson interval)", "",
            *table(["arm", "host", "identical/N", "%", "interval low", "interval high", "distinct answers p1 / p2",
                    "most common answer p1 / p2", "fp8 kernel", "active", "status"], rows),
            STABILITY_NOTE, "",
        ]
        rows = []
        for engine in ENGINES:
            for workload in ("scene", "chat"):
                c = cells["cost"][model][engine][workload]
                if c["ratio"] is None:
                    rows.append([engine, f(c["host"]), c["source"], c["other_arm"], workload, c["status"]] + ["-"] * 8)
                    continue
                rows.append([
                    engine, c["host"], c["source"], c["other_arm"], workload, c["status"], f(c["default_tok_s"]),
                    f(c["other_tok_s"]), f(c["ratio"], 3), f(c["default_tpot_p50"], 2), f(c["other_tpot_p50"], 2),
                    f(c["default_served_rps"], 3), f(c["other_served_rps"], 3),
                    f"{f(c['default_reply_mean'])} / {f(c['other_reply_mean'])}",
                ])
        note = (
            "family I, both modes from fresh servers on one host" if model == "gemma"
            else "A1 against the C pass-1 load of the other mode, both from one host"
        )
        out += [
            f"### {title} cost of invariance at 64 in flight ({note}; other mode = the engine's non-default mode)", "",
            *table(
                ["engine", "host", "source", "other mode", "workload", "status", "default tok/s", "other tok/s",
                 "ratio", "default TPOT p50", "other TPOT p50", "default req/s", "other req/s",
                 "reply tokens default / other"],
                rows,
            ),
            COST_NOTES[model], "",
        ]
        if model == "gemma":
            rows = []
            for arm, k in cells["C_vllm_kernels"].items():
                rows.append([
                    arm, f(k["host"]), k["deep_gemm"], f(k["fp8_kernel"]), active_label(k["batch_invariant_active"]),
                    "-" if k["n"] is None else f"{k['identical']}/{k['n']}", f(k["pct"]), f(k["ci_lo_pct"]),
                    f(k["ci_hi_pct"]), pair(k, "distinct"), pair(k, "top"),
                ])
            out += [
                "### vLLM's batch invariance on Gemma 4, by FP8 kernel (percent, 95% Wilson interval)", "",
                *table(["arm", "host", "DeepGEMM", "fp8 kernel", "active", "identical/N", "%", "interval low",
                        "interval high", "distinct answers p1 / p2", "most common answer p1 / p2"], rows),
                STABILITY_NOTE, "",
            ]
        rows = []
        for engine in ENGINES:
            d = cells["D"][model][engine]
            if d is None:
                rows.append([engine, "-", missing_status(model, engine, "D"), "-", "-", "-"])
                continue
            rows.append([
                engine, d["host"], f(d["cold_ready_s"]), f(d["cold_first_answer_s"]), f(d["warm_ready_s"]),
                f(d["warm_first_answer_s"]),
            ])
        out += [
            f"### {title} start-up (seconds)", "",
            *table(["engine", "host", "cold ready", "cold first answer", "warm ready", "warm first answer"], rows),
        ]
        rows = []
        for engine in ENGINES:
            s = cells["S"][model][engine]
            if s is None:
                rows.append([engine, "-", missing_status(model, engine, "D")] + ["-"] * 6)
                continue
            keys = ("requests", "stop", "valid", "invalid", "duplicate_keys", "length", "errors")
            if s["status"] != "ok":
                rows.append([engine, s["host"], s["status"]] + ["-"] * 6)
                continue
            rows.append([engine, s["host"]] + [f(s[k]) for k in keys])
        out += [
            f"### {title} validity of structured replies (requests)", "",
            *table(["engine", "host", "requests", "stop", "valid", "invalid", "duplicate keys", "length", "errors"],
                   rows),
        ]
        rows = []
        for arm in B_ARMS:
            b = cells["B"][model][arm]
            if b is None:
                rows.append([arm, "-", missing_status(model, arm, "B")] + ["-"] * 5)
                continue
            rows.append([
                arm, b["host"], "ok", f(b["c1_tpot_p50"], 2), f(b["c1_e2e_p50"]), f(b["c1_tokens_per_verify"], 3),
                f(b["c64_served_rps"], 3), f(b["c64_tpot_p50"], 2),
            ])
        out += [
            f"### {title} vLLM's own draft counts (vLLM's settings, not the headline draft count; scene; ms)",
            "",
            *table(["arm", "host", "status", "c=1 TPOT p50", "c=1 e2e p50", "c=1 tokens per verify", "c=64 req/s",
                    "c=64 TPOT p50"], rows),
        ]
    rows = []
    for engine in ("vllm", "sglang"):
        q = cells["Q"][engine]
        if q is None:
            rows.append([engine, "not run"] + ["-"] * 5)
            continue
        rows.append([
            engine, "ok", f(q["q_tpot_p50"], 2), f(q["p1_tpot_p50"], 2), f(q["host_ratio"], 4),
            f"{100 * q['deviation']:.2f}%", q["verdict"],
        ])
    out += [
        "## Vendor-image check", "",
        "### Gemma scene at one request in flight (hosts q-vllm and q-sglang against p1; TPOT in ms; pass when "
        "|deviation| <= 5%)", "",
        *table(["engine", "status", "q TPOT p50", "p1 TPOT p50", "host ratio", "deviation", "verdict"], rows),
        "## GPU controls", "",
        "### Start, middle and end controls (TFLOPS, GB/s, MHz, W; SM clock and power: bf16 phase medians)", "",
    ]
    rows = [
        [name, f(c["bf16_tflops"]), f(c["fp8_tflops"]), f(c["copy_gbs"]), f(c["sm_mhz_median_bf16"], 0),
         f(c["power_w_median_bf16"])]
        for name, c in ctl.items()
    ]
    out += table(["control", "bf16 TFLOPS", "fp8 TFLOPS", "copy GB/s", "SM MHz", "power W"], rows)
    out += ["## llm-mini-infer's grammar compile cost (CPU, host p2)", ""]
    for model in MODELS:
        g = cells["grammar_cost"][model]
        if g is None:
            out += [f"### {MODEL_TITLE[model]}", "", "not run", ""]
            continue
        scalar = [k for k in g if k != "masks_p50_p99"]
        masks = g["masks_p50_p99"] or {}
        sub = sorted({k for v in masks.values() for k in v})
        out += [
            f"### {MODEL_TITLE[model]} (ms unless the name says otherwise)", "",
            *table(["field", "value"], [[k, json.dumps(g[k])] for k in scalar]),
            "Mask times, p50 / p99 (ms):", "",
            *table(["case", *sub], [[case, *[" / ".join(f(x, 3) for x in v.get(k, [])) or "-" for k in sub]]
                                     for case, v in masks.items()]),
        ]
    out += ["## Clock check: loads more than 3% from their host's median SM clock", ""]
    for host, c in cells["clock_check"].items():
        out += [f"### Host {host} (median of {c['loads_counted']} loads: {f(c['host_median_sm_mhz'], 0)} MHz)", ""]
        if c["outliers"]:
            out += table(
                ["unit", "load", "SM MHz", "diff %", "power W", "throttle"],
                [[o["unit"], o["load"], f(o["sm_mhz_median"], 0), f(o["diff_pct"], 2), f(o["power_w_median"]),
                  json.dumps(o["throttle"])] for o in c["outliers"]],
            )
        else:
            out += ["No load differs by more than 3%.", ""]
        if c["controls"]:
            out += table(
                ["control", "bf16 SM MHz", "bf16 TFLOPS"],
                [[x["control"], f(x["bf16_sm_mhz_median"], 0), f(x["bf16_tflops"])] for x in c["controls"]],
            )
    rows = [
        [key, str(m["status"]), (m["status_line"] or "-")[:120]]
        for key, m in cells["_manifest_status"].items()
        if m["status"] != "ok"
    ]
    if rows:
        out += ["## Manifests whose status is not ok", "", *table(["manifest", "status", "status line"], rows)]
    return "\n".join(out)


def main() -> int:
    a1 = a1_cells()
    bounds = bound_cells()
    ctl = controls()
    c = c_cells()
    cells = {
        "A1": a1,
        "bound": bounds,
        "A2": a2_cells(bounds),
        "C": c,
        "C_vllm_kernels": c_vllm_kernels(c),
        "cost": cost_cells(),
        "D": d_cells(),
        "S": s_cells(),
        "B": b_cells(),
        "Q": q_cells(a1, ctl),
    }
    mans = manifest_summary()
    grammar = grammar_cost()
    check = clock_check(raw_controls())
    summary = {
        "cells": cells, "manifests": mans, "clocks": clock_summary(), "controls": ctl, "grammar_cost": grammar,
        "clock_check": check,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    text = markdown({**cells, "_manifest_status": mans, "grammar_cost": grammar, "clock_check": check}, ctl)
    (HERE / "summary.md").write_text(text + "\n", encoding="utf-8")
    for warning in WARNINGS:
        print("warning:", warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

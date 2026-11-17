"""The chart and its rendering with the pinned helm (tests/deploy/test_helm.py,
tests/deploy/test_deploy_contract.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from .paths import DATA, ROOT

sys.path.insert(0, str(ROOT / "tools"))
from fetch_tools import path as tool_path

HELM = ROOT / "deploy" / "helm"
VALUES = DATA / "chart-values.yaml"
HELM_BIN = tool_path("helm")


def render(chart: str, values: Path | None = None, *sets: str, release: str | None = None) -> list[dict]:
    cmd = [str(HELM_BIN), "template", release or chart, str(HELM / chart), "--namespace", "inference"]
    if values:
        cmd += ["-f", str(values)]
    for s in sets:
        cmd += ["--set", s]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return [d for d in yaml.safe_load_all(out.stdout) if d]


def kind(docs: list[dict], k: str) -> list[dict]:
    return [d for d in docs if d["kind"] == k]

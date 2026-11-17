#!/usr/bin/env python3
"""Checks that the published benchmark reports hold nothing private or identifying.

Usage: python3 bench/reports/check.py [DIR ...]

With no directory it checks every directory next to this script. It prints one line per finding, `path: rule: where`,
and exits 1 if there is any; otherwise it prints `N files checked, no findings` and exits 0. A finding for the `token`
rule never prints the matched text.

A `.json` file is parsed and walked: every key and every string value is checked, and `where` is the JSON path. Any
other file is read as UTF-8 text, line by line, and `where` is the line number. A file that is not UTF-8 text, or a
`.json` file that is not valid JSON, is a finding too.

Rules: path, ip, host, pod, token, short-id, date, manifest-key (see the constants below), plus `encoding` and
`json` for unreadable files. Standard library only.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PATH_COMPONENTS = (
    "Users", "home", "root", "workspace", "tmp", "private", "var", "opt", "cache", "etc", "mnt", "srv", "proc", "sys",
    "dev", "usr", "bin", "lib", "run", "data", "models",
)
# An absolute path starts at a `/` that is not the tail of a word, a host or a URL (`scheme://host/...`).
PATH_RE = re.compile(r"(?<![\w.~:/-])/(?:" + "|".join(PATH_COMPONENTS) + r")(?![\w-]|\.\w)|(?<!\w)[A-Za-z]:\\")

IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")
IPV4_OK = {"127.0.0.1", "0.0.0.0"}
IPV6_RE = re.compile(r"(?<![\w:])[0-9A-Fa-f:]{2,}(?![\w:])")
IPV6_OK = {"::1"}

HOST_RE = re.compile(r"(?<![\w.-])[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+(?![\w-])", re.ASCII)
TLDS = frozenset(["com", "net", "org", "io", "ai", "dev", "app", "cloud", "internal", "local"])
HOST_BAD_SUFFIXES = ("runpod.net", "runpod.io", ".internal", ".local")
HOST_ALLOWED = (
    "huggingface.co", "github.com", "ghcr.io", "docs.vllm.ai", "docs.sglang.ai", "download.pytorch.org", "pypi.org",
    "files.pythonhosted.org", "flashinfer.ai", "githubusercontent.com",
)

POD_RE = re.compile(r"(?<!\w)(?=[a-z0-9]{14}(?!\w))(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{14}(?!\w)")

TOKEN_RES = [
    re.compile(p)
    for p in (
        r"hf_[A-Za-z0-9]{20,}",
        r"gh[pousr]_[A-Za-z0-9]{20,}",
        r"github_pat_\w{20,}",
        r"rpa_[A-Za-z0-9]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"Bearer [A-Za-z0-9._-]{16,}",
    )
] + [re.compile(r"(api[_-]?key|token|secret|password)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._-]{12,}", re.IGNORECASE)]

SHORT_ID_RE = re.compile(r"(?<!\w)(?=[0-9a-f]{7,12}(?!\w))(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,12}(?!\w)")
SHORT_ID_EXEMPT_KEY = "answers"

DATE_RE = re.compile(r"(?<![\d-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")
WHEN_KEY = "when"
WHEN_RE = re.compile(r"\d{4}-\d{2}")

MANIFEST_KEYS = {
    "schema", "arm", "family", "engine", "model", "drafter", "revisions", "k", "k_source", "batch_invariant",
    "batch_invariant_source", "argv", "env_names", "deviations", "status", "status_line", "versions", "source",
    "resolved", "smoke", "startup", "loads", "host", "client", "order", "when", "image", "commit", "notes",
    "env_inherited",
}
MANIFEST_NESTED = {
    "resolved": {"kv_tokens", "max_running_requests", "prefix_cache", "attention_backend", "fp8_kernel",
                 "chunk_tokens", "memory_fraction", "batch_invariant_active", "spec", "context", "cuda_graphs",
                 "model_runner", "kv_dtype", "notes"},
    "smoke": {"ok", "failures", "scene_prompt_tokens", "scene_completion_tokens", "scene_cached_tokens_second",
              "scene_valid", "scene_compact", "chat_prompt_tokens", "chat_completion_tokens", "reasoning_free",
              "verify_steps", "seconds"},
    "startup": {"ready_s", "first_answer_s", "cold", "wiped", "evicted", "stopped"},
    "host": {"gpu", "power_limit_w", "power_max_w", "sm_clock_max_mhz", "memory_mib", "driver", "cuda", "cpu",
             "cpus", "cpu_quota", "kernel"},
    "client": {"seed", "warmup_requests", "warmup_s", "duration_s", "tail_s", "timeout_s", "python"},
}


@dataclass(frozen=True)
class Finding:
    path: Path
    rule: str
    where: str

    def format(self, base: Path | None = None) -> str:
        path = self.path
        if base is not None:
            with contextlib.suppress(ValueError):
                path = path.relative_to(base)
        return f"{path}: {self.rule}: {self.where}"


def _host_findings(text: str) -> bool:
    for m in HOST_RE.finditer(text):
        name = m.group(0).lower()
        labels = name.split(".")
        if labels[-1] not in TLDS or not labels[-1].isalpha():
            continue
        if name.endswith(HOST_BAD_SUFFIXES):
            return True
        if len(labels) >= 3 and not any(name == a or name.endswith("." + a) for a in HOST_ALLOWED):
            return True
    return False


def _ip_findings(text: str) -> bool:
    for m in IPV4_RE.finditer(text):
        s = m.group(0)
        if s not in IPV4_OK and all(int(p) <= 255 for p in s.split(".")):
            return True
    for m in IPV6_RE.finditer(text):
        s = m.group(0)
        if s.count(":") < 2 or s in IPV6_OK:
            continue
        groups = s.split(":")
        if "::" in s:
            if s.count("::") != 1:
                continue
            parts = [g for g in groups if g]
            if not parts or len(parts) > 7:
                continue
        else:
            parts = groups
            if len(parts) != 8:
                continue
        if all(1 <= len(g) <= 4 for g in parts):
            return True
    return False


def scan_text(text: str, short_id: bool = True) -> list[str]:
    """The rules that `text` breaks, each once, in a fixed order."""
    rules = []
    if PATH_RE.search(text):
        rules.append("path")
    if _ip_findings(text):
        rules.append("ip")
    if _host_findings(text):
        rules.append("host")
    if POD_RE.search(text):
        rules.append("pod")
    if any(r.search(text) for r in TOKEN_RES):
        rules.append("token")
    if short_id and SHORT_ID_RE.search(text):
        rules.append("short-id")
    if DATE_RE.search(text):
        rules.append("date")
    return rules


def _walk(node: object, where: str, path: Path, out: list[Finding], short_id: bool = True) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_where = f"{where}.{key}"
            for rule in scan_text(str(key), short_id):
                out.append(Finding(path, rule, key_where + " (key)"))
            if key == WHEN_KEY and not (isinstance(value, str) and WHEN_RE.fullmatch(value)):
                out.append(Finding(path, "date", key_where))
            _walk(value, key_where, path, out, short_id and key != SHORT_ID_EXEMPT_KEY)
            # A secret may sit in a value whose key names it.
            if isinstance(value, str) and any(r.search(f"{key}: {value}") for r in TOKEN_RES):
                out.append(Finding(path, "token", key_where))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{where}[{i}]", path, out, short_id)
    elif isinstance(node, str):
        for rule in scan_text(node, short_id):
            out.append(Finding(path, rule, where))


def _check_manifest(data: object, path: Path, out: list[Finding]) -> None:
    if not isinstance(data, dict):
        out.append(Finding(path, "manifest-key", "$"))
        return
    for key, value in data.items():
        if key not in MANIFEST_KEYS:
            out.append(Finding(path, "manifest-key", f"$.{key}"))
        elif key in MANIFEST_NESTED and isinstance(value, dict):
            for sub in value:
                if sub not in MANIFEST_NESTED[key]:
                    out.append(Finding(path, "manifest-key", f"$.{key}.{sub}"))


def check_file(path: Path, root: Path | None = None) -> list[Finding]:
    """The findings in one file. `root` makes the manifest test (a `manifests` folder) relative to it."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(path, "encoding", "file")]
    out: list[Finding] = []
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError:
            return [Finding(path, "json", "file")]
        _walk(data, "$", path, out)
        parts = path.relative_to(root).parts if root is not None else path.parts
        if "manifests" in parts[:-1]:
            _check_manifest(data, path, out)
    else:
        for n, line in enumerate(text.splitlines(), 1):
            for rule in scan_text(line):
                out.append(Finding(path, rule, f"line {n}"))
    seen = set()
    unique = []
    for f in out:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def check_tree(root: Path) -> list[Finding]:
    """The findings in every file under `root`."""
    findings: list[Finding] = []
    for p in _files(root):
        findings.extend(check_file(p, root))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    here = Path(__file__).resolve().parent
    dirs = [Path(a) for a in args] or sorted(p for p in here.iterdir() if p.is_dir() and p.name != "__pycache__")
    findings: list[Finding] = []
    count = 0
    for d in dirs:
        count += len(_files(d))
        d = d.resolve()
        for f in check_tree(d):
            findings.append(f)
            print(f.format(d.parent))
    if findings:
        return 1
    print(f"{count} files checked, no findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())

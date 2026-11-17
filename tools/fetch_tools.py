#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The command-line tools the repo's scripts run: kind, helm, kubeconform, kubectl and gh at pinned versions, fetched
into `tools/bin/<os>-<arch>/` (git-ignored), each download checked against its pinned sha256 before anything is
unpacked. Nothing is installed outside the repo: no Homebrew, no ~/.local/bin.

    uv run tools/fetch_tools.py                          # all five, for this machine
    uv run tools/fetch_tools.py helm kubeconform         # some of them
    uv run tools/fetch_tools.py --platform linux-amd64   # another platform's (CI's)
    uv run tools/fetch_tools.py --check                  # verify what is there; fetch nothing

Platforms: darwin-arm64 (Apple Silicon) and linux-amd64 (CI). Scripts find a tool with `tool(name)`, which fetches it
on first use, or `path(name)`, None unless the pinned, unmodified copy is there (the tests skip then). Docker stays
the machine's own.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "tools" / "bin"
PLATFORMS = ("darwin-arm64", "linux-amd64")

# name -> version, and per platform: (url, sha256 of the download, the binary's path inside a .tar.gz or .zip, or None)
PINS: dict[str, dict] = {
    "kind": {
        "version": "v0.33.0",
        "darwin-arm64": ("https://github.com/kubernetes-sigs/kind/releases/download/v0.33.0/kind-darwin-arm64",
                         "0c8c7dbe5e23594a198b786c4bc13dacc101fa6196b0cb0b23a1ca44e61f4b4f", None),
        "linux-amd64": ("https://github.com/kubernetes-sigs/kind/releases/download/v0.33.0/kind-linux-amd64",
                        "aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d", None),
    },
    "helm": {
        "version": "v3.22.0",
        "darwin-arm64": ("https://get.helm.sh/helm-v3.22.0-darwin-arm64.tar.gz",
                         "4c9982a6cdeb458b60258df66b55398ca5b19293f6877faffe2909ad6f23dfe0", "darwin-arm64/helm"),
        "linux-amd64": ("https://get.helm.sh/helm-v3.22.0-linux-amd64.tar.gz",
                        "1e4ab49e429626cf6c6958d914248b78c9730803c2751b87627e171dc800e7bb", "linux-amd64/helm"),
    },
    "kubeconform": {
        "version": "v0.8.0",
        "darwin-arm64": ("https://github.com/yannh/kubeconform/releases/download/v0.8.0/kubeconform-darwin-arm64.tar.gz",
                         "f84f4dfbebf4a6b0b230385fa065a39ea35e02608c2b50d025dcf64775a69d67", "kubeconform"),
        "linux-amd64": ("https://github.com/yannh/kubeconform/releases/download/v0.8.0/kubeconform-linux-amd64.tar.gz",
                        "9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883", "kubeconform"),
    },
    "kubectl": {
        "version": "v1.37.1",
        "darwin-arm64": ("https://dl.k8s.io/release/v1.37.1/bin/darwin/arm64/kubectl",
                         "fd65982c97ddad3106754b69ffa196d0e543aa591930ae52aed1adfb92f8c77f", None),
        "linux-amd64": ("https://dl.k8s.io/release/v1.37.1/bin/linux/amd64/kubectl",
                        "65691ff77eb6fa44c908b77a1082c9f092c3b9733b5cefabec0d1104890e21a8", None),
    },
    "gh": {
        "version": "v2.102.0",
        "darwin-arm64": ("https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_macOS_arm64.zip",
                         "da922c20d1792e5b2cbf375593d7a658acf034c12c84e007e71c76ef959c337e",
                         "gh_2.102.0_macOS_arm64/bin/gh"),
        "linux-amd64": ("https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz",
                        "bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386",
                        "gh_2.102.0_linux_amd64/bin/gh"),
    },
}


def this_platform() -> str:
    system = platform.system().lower()
    machine = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
    return f"{system}-{machine}"


def folder(plat: str | None = None) -> Path:
    return BIN / (plat or this_platform())


def _stamp(name: str, plat: str) -> Path:
    return folder(plat) / f".{name}.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(name: str, plat: str | None = None) -> str:
    """'ok', 'absent', 'stale' (not the pinned download) or 'modified' (the binary is not what was fetched)."""
    plat = plat or this_platform()
    exe, stamp = folder(plat) / name, _stamp(name, plat)
    if not exe.exists() or not stamp.exists():
        return "absent"
    if plat not in PINS[name]:
        return "stale"
    info = json.loads(stamp.read_text())
    if info.get("version") != PINS[name]["version"] or info.get("download_sha256") != PINS[name][plat][1]:
        return "stale"
    return "ok" if _sha256(exe) == info.get("binary_sha256") else "modified"


def path(name: str) -> Path | None:
    """The repo's copy of a tool for this machine, pinned and unmodified, or None."""
    p = folder() / name
    return p if p.exists() and status(name) == "ok" else None


def fetch(name: str, plat: str | None = None, log=print) -> Path:
    """Download the pinned release, check its sha256, unpack the binary into tools/bin/<platform>/."""
    plat = plat or this_platform()
    if plat not in PLATFORMS:
        raise SystemExit(f"no pins for {plat} (have: {', '.join(PLATFORMS)})")
    url, want, member = PINS[name][plat]
    dest = folder(plat)
    dest.mkdir(parents=True, exist_ok=True)
    log(f"fetching {name} {PINS[name]['version']} ({plat}) from {url}")
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        dl = Path(tmp) / "download"
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": "llm-mini-infer fetch_tools"})
        with urllib.request.urlopen(req, timeout=120) as r, dl.open("wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
        got = h.hexdigest()
        if got != want:
            raise SystemExit(f"{name}: sha256 mismatch for {url}: got {got}, pinned {want}; nothing installed")
        exe = Path(tmp) / name
        if member is None:
            dl.rename(exe)
        elif url.endswith(".zip"):
            with zipfile.ZipFile(dl) as z, z.open(member) as src, exe.open("wb") as f:
                shutil.copyfileobj(src, f)
        else:
            with tarfile.open(dl) as tar:
                src = tar.extractfile(tar.getmember(member))
                if src is None:
                    raise SystemExit(f"{name}: {member} is not a file in {url}")
                with exe.open("wb") as f:
                    shutil.copyfileobj(src, f)
        exe.chmod(0o755)
        info = {"version": PINS[name]["version"], "url": url, "download_sha256": got, "binary_sha256": _sha256(exe)}
        os.replace(exe, dest / name)
        _stamp(name, plat).write_text(json.dumps(info, indent=1) + "\n")
    return dest / name


@functools.cache
def tool(name: str) -> str:
    """The repo's copy of a tool, fetched (pinned, checked) the first time it is needed; a path to run (checked once
    per process: a script may run kubectl hundreds of times)."""
    p = path(name)
    if p is None:
        p = fetch(name, log=lambda m: print(f"[fetch_tools] {m}", file=sys.stderr, flush=True))
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"the tools (default: all of {', '.join(PINS)})")
    ap.add_argument("--platform", choices=PLATFORMS, default=None, help="default: this machine's")
    ap.add_argument("--check", action="store_true", help="report each tool's state; fetch nothing")
    args = ap.parse_args()
    plat = args.platform or this_platform()
    names = args.names or list(PINS)
    if unknown := [n for n in names if n not in PINS]:
        ap.error(f"unknown tools {unknown}; pinned: {', '.join(PINS)}")
    bad = 0
    for name in names:
        state = status(name, plat)
        if args.check:
            print(f"{name:12} {PINS[name]['version']:8} {plat}: {state}")
            bad += state != "ok"
            continue
        if state != "ok":
            fetch(name, plat)
        print(f"{name:12} {PINS[name]['version']:8} {folder(plat).relative_to(ROOT) / name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

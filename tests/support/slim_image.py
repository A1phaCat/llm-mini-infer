"""What the image tests (tests/deploy/test_image_build.py, tests/deploy/test_image_checks.py, tests/deploy/test_helm.py)
share about the slim image: uv.lock as the image installs it (deploy/Dockerfile: `uv sync --locked --no-dev` with the
extras below, on linux x86_64 and Ubuntu's CPython 3.12, less the packages it names with --no-install-package), the
records of the GPU-validated stack it keeps (the image's own freeze, the inventory's AOT modules: tests/data/image/),
and the Dockerfile's arguments. Standard library and `packaging` only (the image checks' dependency group)."""

from __future__ import annotations

import json
import re
import tomllib

from packaging.markers import Marker

from .paths import DATA as TESTS_DATA
from .paths import ROOT

# The public repository: the image's source label, the chart's and the examples' default image
REPOSITORY = "deftinfer/llm-mini-infer"
SOURCE = f"https://github.com/{REPOSITORY}"
IMAGE_REPOSITORY = f"ghcr.io/{REPOSITORY}"
STACK_LABEL = "io.github.deftinfer.llm-mini-infer.stack"
LOCK = ROOT / "uv.lock"
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "deploy" / "Dockerfile"
DATA = TESTS_DATA / "image"  # the stack's records (its README)
FREEZE = DATA / "freeze.txt"
INVENTORY = DATA / "inventory.json"
EXTRAS = ("cuda", "images", "image")  # pyproject's: the CUDA runner, image inputs, the image's other imports
# The image's interpreter and platform, for the lock's markers
IMAGE_ENV = {"sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux", "os_name": "posix",
             "python_version": "3.12", "python_full_version": "3.12.3", "implementation_name": "cpython",
             "implementation_version": "3.12.3", "platform_python_implementation": "CPython",
             "platform_release": "", "platform_version": "", "extra": ""}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(extras: tuple[str, ...] = EXTRAS, env: dict[str, str] = IMAGE_ENV) -> dict[str, str]:
    """name -> version of every locked distribution the project's dependencies and `extras` reach in `env`, the
    project itself included."""
    lock = tomllib.loads(LOCK.read_text())
    by_name: dict[str, list[dict]] = {}
    for p in lock["package"]:
        by_name.setdefault(p["name"], []).append(p)
    (root,) = [p for p in lock["package"] if p["source"].get("editable") == "." or p["source"].get("virtual") == "."]

    def entry(dep: dict) -> dict:
        found = [p for p in by_name[dep["name"]]
                 if dep.get("version", p["version"]) == p["version"] and dep.get("source", p["source"]) == p["source"]]
        if len(found) != 1:
            raise ValueError(f"{dep}: {len(found)} locked entries match")
        return found[0]

    out: dict[str, str] = {root["name"]: root["version"]}
    todo = [*root.get("dependencies", []), *[d for e in extras for d in root.get("optional-dependencies", {})[e]]]
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    while todo:
        dep = todo.pop()
        if "marker" in dep and not Marker(dep["marker"]).evaluate(env):
            continue
        pkg = entry(dep)
        key = (pkg["name"], pkg["version"], tuple(dep.get("extra", ())))
        if key in seen:
            continue
        seen.add(key)
        if out.setdefault(pkg["name"], pkg["version"]) != pkg["version"]:
            raise ValueError(f"{pkg['name']}: {out[pkg['name']]} and {pkg['version']} both reached")
        todo += pkg.get("dependencies", [])
        for e in dep.get("extra", ()):
            todo += pkg.get("optional-dependencies", {}).get(e, [])
    return out


def not_installed() -> set[str]:
    """The locked distributions the image leaves out: the Dockerfile's `uv sync --no-install-package` names (the deps
    and project stages name the same ones)."""
    text = re.sub(r"\\\n", "", DOCKERFILE.read_text())
    syncs = [set(re.findall(r"--no-install-package (\S+)", line)) for line in text.splitlines()
             if "uv sync" in line and not line.lstrip().startswith("#")]
    assert syncs and all(s == syncs[0] for s in syncs), syncs
    return {norm(n) for n in syncs[0]}


def installed() -> dict[str, str]:
    """name -> version of every distribution the image installs: the closure less what it leaves out."""
    left_out = not_installed()
    return {n: v for n, v in closure().items() if n not in left_out}


def freeze() -> dict[str, str]:
    """name -> version of the freeze's `name==version` lines (names normalized)."""
    out = {}
    for line in FREEZE.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            out[norm(name)] = version.strip()
    return out


def version() -> str:
    """pyproject's version: the package's, the release tag's (v<version>), the chart's appVersion and the image's."""
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def dockerfile_arg(name: str) -> str:
    """An ARG's default in deploy/Dockerfile, continuation lines joined."""
    text = re.sub(r"\\\n", "", DOCKERFILE.read_text())
    (value,) = re.findall(rf"^ARG {name}=(.*)$", text, re.M)
    return value.strip('"')


def kernels() -> dict[str, str]:
    """Our kernel libraries' recorded bytes (the Dockerfile's KERNELS_SHA256): path under /opt/llm-mini-infer/kernels ->
    sha256."""
    return {e.split(":", 1)[1]: e.split(":", 1)[0] for e in dockerfile_arg("KERNELS_SHA256").split()}


def aot_modules() -> dict[str, str]:
    """FlashInfer's AOT modules llm-mini-infer loaded in the image (the inventory, every run): path inside site-packages
    -> sha256."""
    inv = json.loads(INVENTORY.read_text())
    return {p: d for run in inv["runs"].values() for p, d in run["aot_sha256"].items()}

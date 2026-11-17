"""Committed goldens of GPU runs' bits (tests/data/golden/): a run's digest recorded per GPU and per set of tiny
weights, so that a new torch, CUDA, cuBLAS, FlashInfer or kernel build shows as a digest that moved or one that did not.

An entry's key is `<name>/<gpu>/<weights>`: the GPU's architecture as nvcc names it and its SM count (`gpu`:
"sm_120-188sm"; FlashInfer's prefill plans and cuBLAS's heuristics read the SM count, so two GPUs of one architecture
can differ), and the weights the start of a sha256 over the generated checkpoints' tensors (`weights_digest`), which
moves only when the tiny models do. Its value is the run's digests (`bits_digest`: the hidden states', the logits', the
images' encoder rows') and, as metadata only, the stack that recorded it (`versions`): keyed by the package versions,
every stack move would find no entry and could never compare. `check` compares a run with its entry; with
`--refresh-golden` (tests/conftest.py) it writes or replaces the entry instead. A missing entry fails: a new GPU or new
tiny weights are recorded deliberately, on that GPU, its other tests passing in the same run. `python -m
tests.support.golden <file>` merges the entries of a golden file fetched from a GPU into the committed file of its name
(`merge`). tests/repo/test_repro_harness.py checks this module on the CPU."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from .paths import DATA, ROOT

GOLDEN_DIR = DATA / "golden"
PARTS = {"hidden": "hidden states", "logits": "logits", "images": "images' encoder rows"}  # a digest's parts, named


def gpu() -> str:
    """The current CUDA device as the keys name it: its architecture as nvcc names it and its SM count
    ("sm_120-188sm")."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}-{torch.cuda.get_device_properties().multi_processor_count}sm"


def weights_digest(dirs: Iterable[Path]) -> str:
    """sha256 over the safetensors files in `dirs`, in that order and each directory's files in name order: each
    tensor's name, dtype, shape and bytes, by name, read from the file's own header. File names, the header's metadata
    and the configuration files stay out."""
    h = hashlib.sha256()
    for d in dirs:
        for f in sorted(Path(d).glob("*.safetensors")):
            data = f.read_bytes()
            n = int.from_bytes(data[:8], "little")
            header = json.loads(data[8:8 + n])
            header.pop("__metadata__", None)
            for name in sorted(header):
                t = header[name]
                a, b = t["data_offsets"]
                h.update(f"{name} {t['dtype']} {t['shape']}\n".encode())
                h.update(data[8 + n + a:8 + n + b])
    return h.hexdigest()


def bits_digest(hidden: dict[Any, np.ndarray], logits: dict[Any, tuple[np.ndarray, np.ndarray]],
                images: dict[Any, np.ndarray]) -> dict[str, str]:
    """sha256s of a run's captures (`forced.Bits`', keyed by position): "hidden", each key's hidden state; "logits",
    each key's token ids and logits; "images", each key's encoder rows (none: the digest of nothing); each in key
    order, each array with its dtype and shape."""
    out = {}
    for part, rows in (("hidden", {k: (v,) for k, v in hidden.items()}), ("logits", logits),
                       ("images", {k: (v,) for k, v in images.items()})):
        h = hashlib.sha256()
        for k in sorted(rows):
            h.update(f"{k}\n".encode())
            for a in rows[k]:
                a = np.ascontiguousarray(a)
                h.update(f"{a.dtype.str} {a.shape}\n".encode())
                h.update(a.tobytes())
        out[part] = h.hexdigest()
    return out


def key(name: str, device: str, weights: str) -> str:
    return f"{name}/{device}/{weights[:16]}"


def versions() -> dict[str, str]:
    """The stack a run used: torch, its CUDA and cuBLAS, FlashInfer, Pillow, the GPU's name and driver (NVML's, where
    present), and each kernel library's build key and the start of its sha256 (`kernels.build`, as loaded)."""
    import torch

    from llm_mini_infer.kernels import build

    out = {"torch": torch.__version__, "cuda": torch.version.cuda or "none"}
    for name, dist in (("cublas", "nvidia-cublas"), ("flashinfer", "flashinfer-python"), ("pillow", "pillow")):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[name] = metadata.version(dist)
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
    with contextlib.suppress(Exception):  # nvidia-ml-py, FlashInfer's dependency
        import pynvml

        pynvml.nvmlInit()
        try:
            out["driver"] = str(pynvml.nvmlSystemGetDriverVersion())
        finally:
            pynvml.nvmlShutdown()
    for name, stats in (("kernels", build.stats), ("cutlass", build.stats_cutlass)):
        if path := stats.get("path"):
            lib = Path(str(path))
            out[name] = f"{lib.parent.name} sha256:{hashlib.sha256(lib.read_bytes()).hexdigest()[:16]}"
    return out


def _write(path: Path, golden: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")


def merge(src: Path, dest: Path) -> list[str]:
    """The entries of the golden file `src` (fetched from a GPU) into `dest`, replacing those of the same keys and
    keeping the others (entries recorded on other GPUs, or committed since the GPU's tree was taken). Returns the keys
    written."""
    new = json.loads(Path(src).read_text())
    golden = json.loads(dest.read_text()) if dest.exists() else {}
    _write(dest, {**golden, **new})
    return sorted(new)


def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check(path: Path, entry: str, digest: dict[str, str], *, refresh: bool, stack: dict[str, str], test: str) -> None:
    """Compare a run's `digest` with the golden `entry` of the file at `path`, or with `refresh` write it there (with
    `stack`, `versions()`, as metadata; the file's other entries kept). `test`: what to rerun to record it."""
    golden = json.loads(path.read_text()) if path.exists() else {}
    if refresh:
        golden[entry] = {**digest, "versions": stack}
        _write(path, golden)
        return
    record = (f"run `python3 -m pytest {test} --refresh-golden` on this GPU (its other tests must pass in the same "
              f"run) and commit {_shown(path)} (docs/development.md, \"The GPU suite\")")
    rec = golden.get(entry)
    if rec is None:
        pytest.fail(f"no golden for {entry} in {_shown(path)}: a new GPU (architecture or SM count) or new tiny "
                    f"weights are recorded deliberately. To record it, {record}.", pytrace=False)
    moved = [PARTS.get(p, p) for p in sorted(set(digest) | (set(rec) - {"versions"})) if rec.get(p) != digest.get(p)]
    if moved:
        what = moved[0] if len(moved) == 1 else f"{', '.join(moved[:-1])} and {moved[-1]}"
        was = ", ".join(f"{k} {v}" for k, v in sorted(rec.get("versions", {}).items()))
        now = ", ".join(f"{k} {v}" for k, v in sorted(stack.items()))
        pytest.fail(f"{entry}: the run's {what} differ from the golden, recorded with {was}; this run: {now}. If the "
                    f"change is meant to move them, {record}.", pytrace=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m tests.support.golden",
                                 description="Merge golden files fetched from a GPU into the committed ones of their "
                                             "names (tests/data/golden/).")
    ap.add_argument("files", nargs="+", type=Path)
    for f in ap.parse_args().files:
        dest = GOLDEN_DIR / f.name
        for k in merge(f, dest):
            print(f"{_shown(dest)}: {k}")

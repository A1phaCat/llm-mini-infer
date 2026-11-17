"""Model weights from Hugging Face: `model.path` takes a repo id, e.g. `Qwen/Qwen3.8-27B-FP8`.

`model.path` and `speculative.model` (Gemma 4's assistant) each take a local directory (a mounted checkpoint) or a repo
id (`org/name`). For a repo id, `resolve()` fetches the snapshot with huggingface_hub into the Hugging Face cache
(`HF_HOME`: on Kubernetes a volume, in the image `/cache/huggingface`) and points that setting at its local snapshot;
the launcher calls it once, before the image pool, the engine and the frontends start, so all of them read the same
local directory. The served name stays the repo id (the model as given; `config.derive`).

- **Revision:** `model.revision` and `speculative.revision` (a branch, a tag or a commit); default `main`.
- **A cached snapshot is reused:** huggingface_hub downloads only the files the cache lacks. With `HF_HUB_OFFLINE=1`
  nothing goes to the network: the cached snapshot is used if it is complete, else startup fails with
  huggingface_hub's message.
- **Fast downloads:** `HF_XET_HIGH_PERFORMANCE=1` unless set otherwise (hf_xet's high-performance mode: every CPU
  and the most bandwidth). huggingface_hub's progress bars are off (`HF_HUB_DISABLE_PROGRESS_BARS`: they would flood
  a container's log); a line every 15 s says how far the download is.
- **The token** comes from `HF_TOKEN` (huggingface_hub reads it from the environment), never from a configuration file,
  and is never logged.
- **Files:** what the server reads (`*.json`, `*.safetensors`, `*.jinja`, `*.txt`): no README, images or `.bin`
  weights. A snapshot without `config.json`, or missing a shard its safetensors index lists, fails startup.
- **Logged:** the repo, the revision and its commit, the snapshot's size, what was downloaded and how fast.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("llm_mini_infer.hub")

PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt"]
_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}")
_resolved: dict[tuple[str, str | None], str] = {}  # (repo, revision) -> snapshot, once per process


class HubError(RuntimeError):
    pass


def is_repo_id(path: str) -> bool:
    """`org/name`, and not an existing local path (a local directory always wins)."""
    return bool(path) and _REPO_ID.fullmatch(path) is not None and not Path(path).exists()


def resolve(cfg: Any) -> None:
    """Each of `model.path` and `speculative.model` given as a repo id: fetch its snapshot (or find it in the cache),
    point the setting at it and record `org/name@commit` in `cfg.model_source` or `cfg.speculative_source`. A local
    directory is left as it is. Once per process: a second call (the entrypoint checks the container first, then the
    launcher starts) reuses the first's snapshot."""
    m = cfg.model
    if is_repo_id(m.path):
        key = (m.path, m.revision)
        path = _resolved.get(key)
        if path is None:
            path = _resolved[key] = download(m.path, m.revision)
        cfg.model_source = f"{m.path}@{Path(path).name}"
        m.path = path
    sp = getattr(cfg, "speculative", None)
    if sp is not None and sp.model and is_repo_id(sp.model):
        key = (sp.model, sp.revision)
        path = _resolved.get(key)
        if path is None:
            path = _resolved[key] = download(sp.model, sp.revision)
        cfg.speculative_source = f"{sp.model}@{Path(path).name}"
        sp.model = path


def cache_dir() -> Path:
    """The Hugging Face hub cache this process uses (`HF_HUB_CACHE`, else `$HF_HOME/hub`)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def download(repo: str, revision: str | None) -> str:
    """The local snapshot of `repo` at `revision`, downloaded into the Hugging Face cache unless complete there."""
    # hf_xet reads it once, when it is first imported (huggingface_hub imports it at the first xet download), so set
    # it before that
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        disable_progress_bars()
    cache = Path(constants.HF_HUB_CACHE)
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    before = _bytes(repo_dir / "blobs")
    rev = revision or "main"
    offline = constants.HF_HUB_OFFLINE
    log.info("model %s (revision %s): %s", repo, rev,
             f"looking for it in the Hugging Face cache {cache} (HF_HUB_OFFLINE)" if offline else
             f"fetching it into the Hugging Face cache {cache} (only what the cache lacks; xet high performance "
             f"{'on' if _true('HF_XET_HIGH_PERFORMANCE') else 'off'}; token: "
             f"{'HF_TOKEN' if os.environ.get('HF_TOKEN') else 'none'})")
    done = threading.Event()
    t0 = time.monotonic()
    watcher = threading.Thread(target=_progress, args=(repo, repo_dir / "blobs", before, t0, done), daemon=True)
    watcher.start()
    try:
        path = snapshot_download(repo, revision=revision, allow_patterns=PATTERNS)
    except Exception as exc:  # huggingface_hub's own message: the repo, the revision, offline, a gated repo, ...
        raise HubError(f"model {repo} (revision {rev}): {type(exc).__name__}: {exc}") from None
    finally:
        done.set()
        watcher.join(1)
    elapsed = time.monotonic() - t0
    snapshot = Path(path)
    _check_complete(repo, snapshot)
    total = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())  # through the symlinks to the blobs
    got = max(0, _bytes(repo_dir / "blobs") - before)
    log.info("model %s (revision %s, commit %s): %.2f GB in %s; %s", repo, rev, snapshot.name, total / 1e9, snapshot,
             f"downloaded {got / 1e9:.2f} GB in {elapsed:.1f} s ({got / 1e6 / max(elapsed, 1e-3):.0f} MB/s)" if got
             else f"all of it from the cache ({elapsed:.1f} s)")
    return path


def _true(var: str) -> bool:
    return os.environ.get(var, "").upper() in {"1", "ON", "YES", "TRUE"}


def _bytes(blobs: Path) -> int:
    """Bytes under a repo's blob folder, downloads in progress (`.incomplete`) included."""
    try:
        return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
    except OSError:
        return 0


def _progress(repo: str, blobs: Path, before: int, t0: float, done: threading.Event) -> None:
    while not done.wait(15.0):
        got = _bytes(blobs) - before
        if got > 0:
            log.info("model %s: %.2f GB downloaded so far (%.0f s)", repo, got / 1e9, time.monotonic() - t0)


def _check_complete(repo: str, snapshot: Path) -> None:
    if not (snapshot / "config.json").is_file():
        raise HubError(f"model {repo}: the snapshot {snapshot} has no config.json")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        missing = sorted(s for s in shards if not (snapshot / s).is_file())
        if missing:
            raise HubError(f"model {repo}: the snapshot {snapshot} lacks {len(missing)} of the {len(shards)} shards "
                           f"its index lists (first: {missing[0]})")

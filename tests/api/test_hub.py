"""Model weights from Hugging Face (`llm_mini_infer.hub`): `model.path` takes a repo id, e.g. Qwen/Qwen3.8-27B-FP8.

The real huggingface_hub downloads from a fake hub (`tests/support/fake_hub.py`, through `HF_ENDPOINT`). huggingface_hub
reads its environment when imported, so every case resolves in a fresh process, as the launcher does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ..support.api_util import Server
from ..support.fake_hub import FakeHub, Repo, auth_digest
from ..support.fixtures import QWEN, require_qwen
from ..support.paths import ROOT

REPO = "acme/tiny-model"
FILES = {
    "config.json": b'{"model_type": "tiny"}',
    "generation_config.json": b"{}",
    "tokenizer.json": b'{"version": "1.0"}',
    "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                                              "b": "model-00002-of-00002.safetensors"}}).encode(),
    "model-00001-of-00002.safetensors": os.urandom(4096),
    "model-00002-of-00002.safetensors": os.urandom(2048),
    "README.md": b"# not downloaded",
    "pytorch_model.bin": b"not downloaded either",
}
WANTED = sorted(f for f in FILES if not f.endswith((".md", ".bin")))
TOKEN = "hf_test_" + "x" * 30  # a made-up token: the test checks it is sent and never printed

RESOLVE = """
import json, logging, sys
from llm_mini_infer import config, hub
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s %(message)s")
cfg = config.load(None, json.loads(sys.argv[1]))
try:
    hub.resolve(cfg)
except hub.HubError as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(0)
print(json.dumps({"path": cfg.model.path, "source": cfg.model_source}))
"""


def resolve(hub: FakeHub, home: Path, model: dict, **env: str) -> tuple[dict, str]:
    """Resolve `model` (the `model` group) in a fresh process against the fake hub: (its result, its log)."""
    e = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE")}
    e.update(HF_ENDPOINT=hub.url, HF_HOME=str(home), HF_HUB_DISABLE_TELEMETRY="1", **env)
    p = subprocess.run([sys.executable, "-c", RESOLVE, json.dumps({f"model.{k}": v for k, v in model.items()})],
                       cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr


def test_a_repo_id_downloads_what_the_server_reads_then_reuses_the_cache(tmp_path) -> None:
    with FakeHub({REPO: Repo(FILES)}) as hub:
        out, logs = resolve(hub, tmp_path, {"path": REPO})
        snap = Path(out["path"])
        assert out["source"] == f"{REPO}@{hub.repos[REPO].commit}" and snap.name == hub.repos[REPO].commit
        assert sorted(p.name for p in snap.iterdir()) == WANTED  # no README, no .bin weights
        assert all((snap / f).read_bytes() == FILES[f] for f in WANTED)
        assert sorted(hub.downloads()) == WANTED
        assert "downloaded 0.00 GB" in logs and "MB/s" in logs  # the download's size and time are logged
        n = len(hub.requests)
        again, logs = resolve(hub, tmp_path, {"path": REPO})  # a new process, the cache warm
        assert again == out and sorted(hub.downloads()) == WANTED  # nothing fetched twice
        assert all("/resolve/" not in p for _, p, _ in hub.requests[n:])
        assert "all of it from the cache" in logs


def test_offline_uses_a_complete_cached_snapshot_and_fails_clearly_without_one(tmp_path) -> None:
    with FakeHub({REPO: Repo(FILES)}) as hub:
        out, _ = resolve(hub, tmp_path / "warm", {"path": REPO})
        n = len(hub.requests)
        offline, logs = resolve(hub, tmp_path / "warm", {"path": REPO}, HF_HUB_OFFLINE="1")
        assert offline == out and len(hub.requests) == n  # not one request
        assert "HF_HUB_OFFLINE" in logs
        cold, _ = resolve(hub, tmp_path / "cold", {"path": REPO}, HF_HUB_OFFLINE="1")
        assert "HF_HUB_OFFLINE" in cold["error"] and len(hub.requests) == n


QUIET = """
import logging
from huggingface_hub.utils import _http
from llm_mini_infer.logutil import setup_logging
setup_logging("INFO", "test")
client = type(_http.get_session()).__module__.split(".")[0]
print(client, logging.getLogger(client).getEffectiveLevel(),
      logging.getLogger(client.replace("httpx", "httpcore")).getEffectiveLevel())
"""


def test_the_hubs_http_client_logs_no_line_per_request() -> None:
    """huggingface_hub's HTTP client (httpx2 on huggingface-hub 2, the pin) logs every request of a download at INFO:
    every process's logging keeps it and its transport at WARNING (`llm_mini_infer.hub` logs the download itself).
    Another client is a change of the names `logutil.setup_logging` quiets."""
    e = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    p = subprocess.run([sys.executable, "-c", QUIET], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-3000:]
    client, level, transport = p.stdout.split()
    assert client == "httpx2" and int(level) >= logging.WARNING and int(transport) >= logging.WARNING


def test_a_revision_pins_the_commit_and_the_served_name_stays_the_repo_id(tmp_path) -> None:
    from llm_mini_infer import config

    commit = hashlib.sha1(b"pinned").hexdigest()
    with FakeHub({REPO: Repo(FILES, commit=commit)}) as hub:
        out, _ = resolve(hub, tmp_path, {"path": REPO, "revision": commit})
        assert out["source"] == f"{REPO}@{commit}"
        assert not any("/revision/" in p for _, p, _ in hub.requests)  # a commit needs no lookup
        bad, _ = resolve(hub, tmp_path, {"path": REPO, "revision": "no-such-branch"})
        assert "no-such-branch" in bad["error"] and "RevisionNotFound" in bad["error"]
    cfg = config.load(None, {"model.path": str(require_qwen())})
    cfg.model_source = f"Qwen/Qwen3.8-27B-FP8@{commit}"  # as resolve() leaves it
    config.derive(cfg)
    assert cfg.model.served_model_name == "Qwen/Qwen3.8-27B-FP8"  # the model as given
    assert "repo id" in cfg.reasons["model.served_model_name"]


def test_the_token_comes_from_hf_token_is_sent_and_never_printed(tmp_path) -> None:
    with FakeHub({REPO: Repo(FILES, token=TOKEN)}) as hub:
        refused, logs = resolve(hub, tmp_path, {"path": REPO})
        assert "error" in refused and "token: none" in logs
        out, logs = resolve(hub, tmp_path, {"path": REPO}, HF_TOKEN=TOKEN)
        assert "path" in out and "token: HF_TOKEN" in logs
        assert auth_digest(f"Bearer {TOKEN}") in {a for _, _, a in hub.requests}
        assert TOKEN not in logs and TOKEN not in json.dumps(out)


def test_a_snapshot_missing_a_shard_of_its_index_fails_startup(tmp_path) -> None:
    files = {k: v for k, v in FILES.items() if k != "model-00002-of-00002.safetensors"}
    with FakeHub({REPO: Repo(files)}) as hub:
        out, _ = resolve(hub, tmp_path, {"path": REPO})
        assert "lacks 1 of the 2 shards" in out["error"]


def test_a_local_directory_is_never_taken_for_a_repo_id(tmp_path, monkeypatch) -> None:
    from llm_mini_infer import hub

    monkeypatch.chdir(tmp_path)
    (tmp_path / "acme" / "local").mkdir(parents=True)
    assert hub.is_repo_id("Qwen/Qwen3.8-27B-FP8") and not hub.is_repo_id("acme/local")
    assert not hub.is_repo_id("/models/qwen") and not hub.is_repo_id("models") and not hub.is_repo_id("a/b/c")


def qwen_repo() -> Repo:
    """The 27B's configuration and tokenizer files (no weights: the fake runner needs none)."""
    model = require_qwen()
    return Repo({p.name: p.read_bytes() for p in model.iterdir() if p.is_file()})


@pytest.mark.parametrize("via", ["cli", "entrypoint"])
def test_the_server_serves_a_repo_id_under_its_name(tmp_path, via) -> None:
    """The launcher downloads before anything starts; the entrypoint checks the container first and the launcher reuses
    its snapshot (one revision lookup in all)."""
    with FakeHub({"Qwen/Qwen3.8-27B-FP8": qwen_repo()}) as hub:
        env = {"HF_ENDPOINT": hub.url, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_DISABLE_TELEMETRY": "1"}
        srv = Server(QWEN, tmp_path, env=env,
                     extra={"model": {"path": "Qwen/Qwen3.8-27B-FP8", "served_model_name": None},
                            "observability": {"log_level": "info"}},
                     entrypoint=via == "entrypoint")
        srv.start(timeout=120)
        try:
            models = json.loads(srv.get("/v1/models")[1])["data"][0]
            assert models["id"] == models["root"] == "Qwen/Qwen3.8-27B-FP8"
            info = json.loads(srv.get("/server_info")[1])
            assert info["config"]["model_source"] == f"Qwen/Qwen3.8-27B-FP8@{hub.repos['Qwen/Qwen3.8-27B-FP8'].commit}"
            assert info["config"]["model"]["path"].endswith(hub.repos["Qwen/Qwen3.8-27B-FP8"].commit)
            status, body = srv.post("/v1/chat/completions", {"model": "Qwen/Qwen3.8-27B-FP8", "max_tokens": 4,
                                                             "messages": [{"role": "user", "content": "hi"}]})
            assert status == 200, body
            assert sum("/revision/" in p for _, p, _ in hub.requests) == 1
        finally:
            assert srv.stop() == 0
        log = srv.log_path.read_text()
        assert "downloaded" in log and "Qwen/Qwen3.8-27B-FP8" in log


def test_a_failed_download_stops_the_server_with_the_reason(tmp_path) -> None:
    with FakeHub({}) as hub:
        env = {"HF_ENDPOINT": hub.url, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_DISABLE_TELEMETRY": "1"}
        srv = Server(QWEN, tmp_path, env=env, extra={"model": {"path": "acme/missing"}})
        try:
            with pytest.raises(RuntimeError, match="server exited"):
                srv.start(timeout=60)
        finally:
            srv.stop()
        deadline = time.monotonic() + 10
        while srv.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert srv.proc.returncode == 1
        assert "acme/missing" in srv.log_path.read_text() and "RepositoryNotFound" in srv.log_path.read_text()


DRAFTER = "acme/tiny-model-assistant"
DRAFTER_FILES = {"config.json": b'{"model_type": "gemma4_assistant"}', "model.safetensors": os.urandom(1024)}
RESOLVE_BOTH = RESOLVE.replace('print(json.dumps({"path": cfg.model.path, "source": cfg.model_source}))',
                               'print(json.dumps({"path": cfg.model.path, "drafter": cfg.speculative.model, '
                               '"source": cfg.speculative_source}))')


def test_the_drafter_model_downloads_like_the_model(tmp_path) -> None:
    """`speculative.model` (Gemma 4's assistant) takes a repo id too, fetched into the same cache."""
    e = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE")}
    with FakeHub({REPO: Repo(FILES), DRAFTER: Repo(DRAFTER_FILES)}) as hub:
        e.update(HF_ENDPOINT=hub.url, HF_HOME=str(tmp_path), HF_HUB_DISABLE_TELEMETRY="1")
        p = subprocess.run([sys.executable, "-c", RESOLVE_BOTH, json.dumps({"model.path": REPO,
                                                                           "speculative.model": DRAFTER})],
                           cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-3000:]
        out = json.loads(p.stdout.strip().splitlines()[-1])
        drafter = Path(out["drafter"])
        assert drafter.name == hub.repos[DRAFTER].commit and (drafter / "model.safetensors").is_file()
        assert out["source"] == f"{DRAFTER}@{hub.repos[DRAFTER].commit}"
        assert Path(out["path"]).name == hub.repos[REPO].commit

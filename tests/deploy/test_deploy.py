"""The container: the entrypoint's preflight names the compose setting to fix before the server would fail later with
a less helpful message; kernels prebuilt in the image are found beside the cache volume, and a lookup that finds none
names why."""

from __future__ import annotations

import yaml

from llm_mini_infer import entrypoint
from llm_mini_infer.kernels import build

from ..support.fixtures import fresh_kernel_lookup, require_qwen
from ..support.paths import ROOT


def write(tmp_path, **groups) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(groups))
    return str(path)


def test_preflight_names_the_missing_mount_and_accepts_a_good_config(tmp_path, monkeypatch) -> None:
    model = require_qwen()
    for var in entrypoint.CACHES:
        monkeypatch.setenv(var, str(tmp_path / "cache" / var))
    bad = write(tmp_path, model={"path": str(tmp_path / "nothing")})
    problems = entrypoint.preflight(bad, [])
    assert len(problems) == 1 and "LLM_MINI_INFER_MODEL_DIR" in problems[0]
    good = write(tmp_path, model={"path": str(model)}, runner={"kind": "fake"})
    assert entrypoint.preflight(good, ["--set", "dev.images.arena_mb=1"]) == []
    assert entrypoint.preflight(good, ["--set=model.path=/absent"])[0].startswith("model.path /absent")
    # The extra arguments count as `llm-mini-infer serve` counts them: --model, --config and the other named flags too,
    # every --set last
    assert entrypoint.preflight(bad, ["--model", str(model), "--set", "runner.kind=fake"]) == []
    cfg = entrypoint.load(bad, ["--model", "/a", "--port", "1", "--set=model.path=/b", "--config", good])
    assert (cfg.model.path, cfg.server.port, cfg.runner.kind) == ("/b", 1, "fake")


def test_preflight_warns_of_an_unwritable_cache_the_image_writes(tmp_path, monkeypatch, caplog) -> None:
    """An unwritable JIT cache on the volume is a warning (it would not persist across starts); the kernel cache is
    none of those: the image prebuilds our kernels and never writes it."""
    import logging

    model = require_qwen()
    for var in entrypoint.CACHES:
        monkeypatch.setenv(var, str(tmp_path / "cache" / var))
    (tmp_path / "file").write_text("")
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "file" / "kernels"))  # mkdir fails
    good = write(tmp_path, model={"path": str(model)}, runner={"kind": "fake"})
    with caplog.at_level(logging.WARNING, logger="llm_mini_infer.entrypoint"):
        assert entrypoint.preflight(good, []) == []
        monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "file" / "fi"))
        assert entrypoint.preflight(good, []) == []
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1 and warned[0].startswith("FLASHINFER_WORKSPACE_BASE=") and "not writable" in warned[0]


def test_preflight_fetches_a_drafter_given_by_repo_id_beside_a_mounted_checkpoint(tmp_path, monkeypatch) -> None:
    """A mounted checkpoint with its assistant given by repo id (compose's example: Gemma 4 31B in FP8): the
    preflight fetches the drafter as the launcher would (hub.resolve takes both), instead of refusing it as a
    directory without config.json."""
    from llm_mini_infer import hub

    from ..support.config_cases import _assistant, _gemma

    target, assistant = _gemma(tmp_path / "target"), _assistant(tmp_path / "assistant")
    fetched = []

    def download(repo: str, revision: str | None) -> str:
        fetched.append(repo)
        return str(assistant)

    monkeypatch.setattr(hub, "_resolved", {})
    monkeypatch.setattr(hub, "cache_dir", lambda: tmp_path / "hf")
    monkeypatch.setattr(hub, "download", download)
    for var in entrypoint.CACHES:
        monkeypatch.setenv(var, str(tmp_path / "cache" / var))
    cfg = write(tmp_path, model={"path": str(target)}, runner={"kind": "fake"},
                speculative={"model": "google/gemma-4-31B-it-assistant"})
    assert entrypoint.preflight(cfg, []) == []
    assert fetched == ["google/gemma-4-31B-it-assistant"]


def test_prebuilt_libraries_are_found_beside_the_cache(tmp_path, monkeypatch) -> None:
    """The image prebuilds its kernels under $LLM_MINI_INFER_KERNEL_PREBUILT; a cache volume mounted over
    $LLM_MINI_INFER_KERNEL_CACHE neither hides them nor shadows them with a stale build (its key differs)."""
    cache, pre = tmp_path / "cache", tmp_path / "pre"
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(cache))
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_PREBUILT", f"{tmp_path / 'absent'}:{pre}")
    want = cache / "120-abc" / build.LIB_NAME
    assert build.prebuilt(want) == want  # neither has it: built into the cache
    (pre / "120-abc").mkdir(parents=True)
    (pre / "120-abc" / build.LIB_NAME).write_bytes(b"x")
    assert build.prebuilt(want) == pre / "120-abc" / build.LIB_NAME
    want.parent.mkdir(parents=True)
    want.write_bytes(b"y")
    assert build.prebuilt(want) == want  # the cache's own wins


def test_the_kernel_lookup_names_why_it_found_no_library(tmp_path, monkeypatch) -> None:
    """A CUDA runner needs the kernel library (on CUDA its sampler has no torch fallback), so the lookup records why
    it found none and `require` fails with it: without nvcc nothing is searched (the library's key holds nvcc's
    version); then no GPU; for CUTLASS no headers, or no SM 12.x GPU. A failed build or load stays recorded after its
    error is raised (named without the missing-nvcc hint), and LLM_MINI_INFER_KERNELS=0 (a debug switch) is named as
    such."""
    import pytest

    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_PREBUILT", str(tmp_path / "pre"))
    monkeypatch.setattr(build, "find_nvcc", lambda: None)
    fresh_kernel_lookup(monkeypatch)
    assert build.load() is None and build.load_cutlass() is None
    state = build.stats["state"]
    assert "nvcc not found" in state and str(tmp_path / "cache") in state and str(tmp_path / "pre") in state
    assert "nvcc not found" in build.stats_cutlass["state"]
    with pytest.raises(RuntimeError,
                       match=r"the CUDA runner needs llm-mini-infer's CUDA kernel library.*nvcc not found.*"
                             r"install nvcc \(CUDA_HOME\)"):
        build.require("the CUDA runner")

    monkeypatch.setattr(build, "find_nvcc", lambda: str(tmp_path / "nvcc"))
    monkeypatch.setattr(build, "local_arch", lambda: None)
    fresh_kernel_lookup(monkeypatch)
    assert build.load() is None and "no CUDA GPU" in build.stats["state"]
    monkeypatch.setattr(build, "local_arch", lambda: "90")
    monkeypatch.setattr(build, "cutlass_root", lambda: None)
    assert build.load_cutlass() is None and "CUTLASS 4 headers" in build.stats_cutlass["state"]
    monkeypatch.setattr(build, "cutlass_root", lambda: tmp_path)
    fresh_kernel_lookup(monkeypatch)
    assert build.load_cutlass() is None and "no SM 12.x GPU (this one is sm_90)" in build.stats_cutlass["state"]

    def failed_build(*args, **kw):
        raise RuntimeError("nvcc failed on sample.cu: an error")

    lib = tmp_path / "cache" / "120-x" / build.LIB_NAME
    monkeypatch.setattr(build, "library_path", lambda archs=None: lib)
    monkeypatch.setattr(build, "build", failed_build)
    fresh_kernel_lookup(monkeypatch)
    with pytest.raises(RuntimeError, match=r"nvcc failed on sample\.cu"):
        build.load()
    assert build.stats["state"] == "failed to build: nvcc failed on sample.cu: an error"
    with pytest.raises(RuntimeError, match="which is unavailable: failed to build: nvcc failed") as exc:
        build.require("the CUDA runner")  # a later lookup names it too
    assert "install nvcc" not in str(exc.value)  # nvcc is there

    def no_nvcc(*args, **kw):
        raise RuntimeError("nvcc not found (set CUDA_HOME)")

    # The image on another GPU: its prebuilt directory has no library for it, and no nvcc builds one. The first
    # lookup's failed build comes out of `require` named, with the image's limit, not bare
    monkeypatch.setattr(build, "build", no_nvcc)
    monkeypatch.setattr(build, "local_arch", lambda: "90")
    fresh_kernel_lookup(monkeypatch)
    with pytest.raises(RuntimeError,
                       match=r"needs llm-mini-infer's CUDA kernel library.*failed to build: nvcc not found.*"
                             r"sm_90.*sm_120 alone"):
        build.require("the CUDA runner")
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"not a shared library")
    fresh_kernel_lookup(monkeypatch)
    with pytest.raises(OSError):
        build.load()
    assert build.stats["state"].startswith("failed to load: ")
    with pytest.raises(RuntimeError, match="which is unavailable: failed to load: ") as exc:
        build.require("the CUDA runner")
    assert "install nvcc" not in str(exc.value)
    fresh_kernel_lookup(monkeypatch)
    monkeypatch.setenv("LLM_MINI_INFER_KERNELS", "0")
    with pytest.raises(RuntimeError, match=r"disabled \(LLM_MINI_INFER_KERNELS=0\)\. Unset LLM_MINI_INFER_KERNELS"):
        build.require("the CUDA runner")


def test_the_first_lookup_of_a_failed_cutlass_build_names_the_library(tmp_path, monkeypatch) -> None:
    """A failed CUTLASS build raises its own error on the first lookup (`load_cutlass`): a block-FP8 checkpoint's
    runtime names the FP8 GEMM library and the setting that does without it there too, as on a later lookup, and
    `cutlass_available` is False, not an error."""
    import pytest
    import torch

    from llm_mini_infer.kernels import fp8
    from llm_mini_infer.runner.cuda.fp8 import Fp8Runtime

    def failed_build(*args, **kw):
        raise RuntimeError("nvcc failed on gemm.cu")

    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("LLM_MINI_INFER_KERNEL_PREBUILT", raising=False)
    monkeypatch.setattr(build, "cutlass_library_path", lambda archs=None: tmp_path / "cache" / "c" / "lib.so")
    monkeypatch.setattr(build, "build_cutlass", failed_build)
    want = (r"^the FP8 GEMM library is unavailable: failed to build: nvcc failed on gemm\.cu \(SM 12\.x, nvcc and "
            r"CUTLASS 4 headers; or dev\.runner\.cuda\.fp8_gemm\)$")
    fresh_kernel_lookup(monkeypatch)
    for _ in range(2):  # the first lookup, then a later one
        with pytest.raises(RuntimeError, match=want):
            Fp8Runtime(True, cuda=False, kernels=True, dtype=torch.bfloat16)
    fresh_kernel_lookup(monkeypatch)
    assert fp8.cutlass_available() is False


def test_a_lookup_that_fails_to_locate_or_bind_its_library_names_it(tmp_path, monkeypatch) -> None:
    """Any error of the first lookup is recorded before it is raised, so a later lookup returns None and `require`
    names it: a search that raises (`library_path` runs nvcc for its version), and a library that loads without one
    of its functions; for either library."""
    import pytest

    def search_fails(archs=None):
        raise RuntimeError("nvcc --version exited with 1")

    class NoFunctions:  # a library that loads and lacks every function
        def __init__(self, path: str) -> None:
            pass

        def __getattr__(self, name: str):
            raise AttributeError(f"undefined symbol: {name}")

    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("LLM_MINI_INFER_KERNEL_PREBUILT", raising=False)
    for path_of, load, stats, require in (
            ("library_path", build.load, lambda: build.stats, lambda: build.require("the CUDA runner")),
            ("cutlass_library_path", build.load_cutlass, lambda: build.stats_cutlass,
             lambda: build.require_cutlass("dev.runner.cuda.fp8_gemm"))):
        fresh_kernel_lookup(monkeypatch)
        monkeypatch.setattr(build, path_of, search_fails)
        with pytest.raises(RuntimeError, match="nvcc --version exited"):
            load()
        assert stats()["state"] == "failed to locate: nvcc --version exited with 1"
        assert load() is None  # a later lookup: the first one's state, not its error
        with pytest.raises(RuntimeError, match="unavailable: failed to locate: nvcc --version exited with 1"):
            require()
        lib = tmp_path / "cache" / path_of / "lib.so"
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"")
        monkeypatch.setattr(build, path_of, lambda archs=None, lib=lib: lib)
        monkeypatch.setattr(build.ctypes, "CDLL", NoFunctions)
        fresh_kernel_lookup(monkeypatch)
        with pytest.raises(AttributeError, match="undefined symbol"):
            load()
        assert stats()["state"].startswith("failed to bind: undefined symbol: ")
        assert load() is None
        with pytest.raises(RuntimeError, match="unavailable: failed to bind: undefined symbol"):
            require()


def test_the_image_config_leaves_the_workload_to_the_model_and_compose_outlasts_its_drain() -> None:
    """deploy/config.yaml, the image's configuration for any model (the model and its own settings come from the
    chart's values, the host check Job's --set or compose's command; tests/deploy/test_deploy_contract.py): the CUDA
    runner, the default, named explicitly (the fake runner answers with random tokens), seed 0 and guidance with compact
    JSON; the workload's settings left to their defaults (the model's context and image budget, the engine's images per
    request, keep-alive and drain). Compose's stop period outlasts the drain."""

    from llm_mini_infer import config as cfgmod

    root = ROOT
    cfg = cfgmod.load(str(root / "deploy" / "config.yaml"), {})
    default = cfgmod.load(None, {})
    assert cfg.runner.kind == "cuda" and cfg.model.served_model_name is None
    assert cfg.runner.cuda.fp8_cold_prefill == "reuse"  # the precise prefix's default, as the chart's
    im = cfg.images
    assert (im.min_pixels, im.max_pixels, im.max_soft_tokens, cfg.speculative.model) == (None, None, None, None)
    assert cfg.scheduler.seed == 0
    assert (cfg.structured_output.backend, cfg.structured_output.disable_any_whitespace) == ("guidance", True)
    assert cfg.model.max_model_len is None and im.max_per_request == default.images.max_per_request
    for key in ("keep_alive_s", "drain_grace_s", "drain_timeout_s"):
        assert getattr(cfg.server, key) == getattr(default.server, key), key
    compose = yaml.safe_load((root / "deploy" / "docker-compose.yaml").read_text())["services"]["llm-mini-infer"]
    assert int(compose["stop_grace_period"].rstrip("s")) >= cfg.server.drain_grace_s + cfg.server.drain_timeout_s + 15
    assert compose["init"] is True and compose["restart"] == "unless-stopped"
    assert any(v.endswith(f":{cfg.model.path}:ro") for v in compose["volumes"])  # the checkpoint where it is read


def test_the_entrypoints_lines_have_every_processs_format(tmp_path) -> None:
    """The entrypoint logs as every process does (`logutil.setup_logging`): `LEVEL MM-DD HH:MM:SS.mmm [entrypoint]`,
    milliseconds included (here a failed check, which exits 2)."""
    import os
    import re
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["LLM_MINI_INFER_CONFIG"] = str(tmp_path / "missing.yaml")
    p = subprocess.run([sys.executable, "-c", "from llm_mini_infer import entrypoint; entrypoint.main(['serve'])"],
                       env=env, capture_output=True, text=True, timeout=60)
    lines = p.stderr.splitlines()
    assert p.returncode == 2 and lines, p.stderr[-2000:]
    assert all(re.match(r"(INFO|WARNING|ERROR) \d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \[entrypoint\] ", ln) for ln in lines)


def test_every_value_the_chart_reads_is_declared() -> None:
    """Each `.Values.<path>` the chart's templates read is in values.yaml (docs/operations.md: every setting is
    there, with its default), the overrides of its names included."""
    import re

    chart = ROOT / "deploy" / "helm" / "llm-mini-infer"
    values = yaml.safe_load((chart / "values.yaml").read_text())
    read = {m for f in (chart / "templates").iterdir()
            for m in re.findall(r"\.Values\.(\w+(?:\.\w+)*)", f.read_text())}
    missing = []
    for path in sorted(read):
        node = values
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(path)
                break
            node = node[part]
    assert not missing, missing

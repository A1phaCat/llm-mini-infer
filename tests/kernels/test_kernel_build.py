"""The kernel build's parallelism (llm_mini_infer/kernels/build.py, `default_jobs`): min(8, usable CPUs, available
memory ÷ the library's budget), from fake /proc/meminfo and cgroup files; `--jobs` overrides it; one log line names the
choice. CPU only: nvcc is faked."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from llm_mini_infer.kernels import build

GiB = build.GiB


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake host: /proc/meminfo and an empty cgroup mount under tmp_path, 16 CPUs in the affinity mask."""
    cg = tmp_path / "cgroup"
    cg.mkdir()
    monkeypatch.setattr(build, "CGROUP", cg)
    monkeypatch.setattr(build, "MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)

    class Host:
        def meminfo(self, avail_kib: int) -> None:
            (tmp_path / "meminfo").write_text(f"MemTotal:       65000000 kB\nMemFree:        1000 kB\n"
                                              f"MemAvailable:   {avail_kib} kB\nBuffers:        0 kB\n")

        def v2(self, cpu: str = "max 100000", mem_max: str = "max", current: int = 0, inactive: int = 0) -> None:
            (cg / "cpu.max").write_text(cpu + "\n")
            (cg / "memory.max").write_text(mem_max + "\n")
            (cg / "memory.current").write_text(f"{current}\n")
            (cg / "memory.stat").write_text(f"anon 1\nfile {inactive}\nactive_file 0\ninactive_file {inactive}\n")

        def v1(self, quota: int = -1, period: int = 100000, limit: int = 9223372036854771712, usage: int = 0,
               inactive: int = 0) -> None:
            (cg / "cpu").mkdir(exist_ok=True)
            (cg / "cpu" / "cpu.cfs_quota_us").write_text(f"{quota}\n")
            (cg / "cpu" / "cpu.cfs_period_us").write_text(f"{period}\n")
            (cg / "memory").mkdir(exist_ok=True)
            (cg / "memory" / "memory.limit_in_bytes").write_text(f"{limit}\n")
            (cg / "memory" / "memory.usage_in_bytes").write_text(f"{usage}\n")
            (cg / "memory" / "memory.stat").write_text(f"cache 1\ntotal_inactive_file {inactive}\n")

    return Host()


CUTLASS = build.JOB_BUDGET[build.CUTLASS_LIB_NAME]
KERNELS = build.JOB_BUDGET[build.LIB_NAME]


def test_the_budgets_are_measured_peaks() -> None:
    """Each library has a budget a job (its largest source's measured peak, plus a margin), CUTLASS's the larger."""
    assert set(build.JOB_BUDGET) == {build.LIB_NAME, build.CUTLASS_LIB_NAME}
    assert GiB / 8 <= KERNELS < CUTLASS <= 8 * GiB


def test_a_big_host_takes_eight(host) -> None:
    host.meminfo(200 * 1024 * 1024)  # 200 GiB available
    jobs, why = build.default_jobs(CUTLASS, 5)
    assert jobs == 5 and "5 sources" in why  # no more jobs than sources
    jobs, why = build.default_jobs(KERNELS, 17)
    assert jobs == 8 and "16 CPUs (affinity mask (16 CPUs), no cgroup quota)" in why and "MemAvailable" in why


def test_memory_caps_the_jobs(host) -> None:
    """A builder with 7 GiB available (the 2-CPU, 8 GB GitHub runner, its VM's own use taken) compiles the CUTLASS
    sources one at a time, the fused ops as its CPUs allow."""
    host.meminfo(7 * 1024 * 1024)
    host.v2(cpu="200000 100000")
    assert build.default_jobs(CUTLASS, 5)[0] == max(1, min(2, int(7 * GiB // CUTLASS)))
    jobs, why = build.default_jobs(KERNELS, 17)
    assert jobs == min(2, int(7 * GiB // KERNELS)) and "cgroup v2 cpu.max 200000/100000" in why
    host.meminfo(100 * 1024)  # 100 MiB: still one job, never none
    assert build.default_jobs(CUTLASS, 5)[0] == 1 and build.default_jobs(KERNELS, 17)[0] == 1


def test_the_cgroup_v2_limit_less_its_working_set(host) -> None:
    """A container's memory.max binds before the host's MemAvailable: its limit less what it holds, its inactive file
    pages (reclaimable) not counted as held."""
    host.meminfo(100 * 1024 * 1024)
    host.v2(mem_max=str(16 * GiB), current=6 * GiB, inactive=2 * GiB)
    avail, why = build.available_memory()
    assert avail == 12 * GiB and why == "cgroup v2 memory.max 16.0 GiB"
    jobs, why = build.default_jobs(CUTLASS, 5)
    assert jobs == int(12 * GiB // CUTLASS) and "12.0 GiB available (cgroup v2 memory.max 16.0 GiB)" in why
    host.v2(mem_max="max", current=60 * GiB)  # no limit: the host's figure
    assert build.available_memory() == (100 * GiB, "MemAvailable")
    host.meminfo(3 * 1024 * 1024)
    host.v2(mem_max=str(16 * GiB))  # the host has less than the cgroup allows
    assert build.available_memory() == (3 * GiB, "MemAvailable")


def test_cgroup_v1(host) -> None:
    host.meminfo(100 * 1024 * 1024)
    host.v1(quota=350000, limit=10 * GiB, usage=3 * GiB, inactive=GiB)
    avail, why = build.available_memory()
    assert avail == 8 * GiB and why == "cgroup v1 memory.limit_in_bytes 10.0 GiB"
    jobs, why = build.default_jobs(KERNELS, 17)
    assert jobs == min(3, int(8 * GiB // KERNELS)) and "cgroup v1 cfs_quota_us 350000/100000" in why
    host.v1()  # unlimited (about 2^63) and no quota
    assert build.available_memory() == (100 * GiB, "MemAvailable")
    assert build.default_jobs(KERNELS, 17)[0] == 8


def test_a_fractional_quota_and_an_over_committed_cgroup(host) -> None:
    host.meminfo(100 * 1024 * 1024)
    host.v2(cpu="150000 100000", mem_max=str(4 * GiB), current=5 * GiB)  # over its limit (reclaim pending): 0 left
    assert build.available_memory() == (0, "cgroup v2 memory.max 4.0 GiB")
    jobs, why = build.default_jobs(KERNELS, 17)
    assert jobs == 1 and "1 CPUs (cgroup v2 cpu.max 150000/100000)" in why


def test_unreadable_memory_leaves_the_cpus(host) -> None:
    """Without /proc/meminfo or a cgroup limit (macOS) memory caps nothing."""
    assert build.available_memory() == (None, "no /proc/meminfo or cgroup memory limit")
    jobs, why = build.default_jobs(CUTLASS, 5)
    assert jobs == 5 and "memory unknown" in why


def _fake_nvcc(tmp_path: Path, monkeypatch) -> Path:
    """An nvcc that writes its source's name into the object (and the objects' names, in order, into the library),
    recording how many compiles ran at once."""
    script = tmp_path / "nvcc"
    script.write_text(f"""#!{sys.executable}
import os, sys, time
args = sys.argv[1:]
if "--version" in args:
    print("Cuda compilation tools, release 13.0, V13.0.88")
    sys.exit(0)
out = args[args.index("-o") + 1]
if "-shared" in args:
    assert "--strip-all" in args, args  # build.LINK_FLAGS
    objs = [a for a in args if a.endswith(".o")]
    open(out, "w").write("".join(open(o).read() for o in objs))
else:
    log = {str(tmp_path / 'running')!r}
    with open(log, "a") as f:
        f.write("+\\n")
    time.sleep(0.05)
    with open(log, "a") as f:
        f.write("-\\n")
    open(out, "w").write(os.path.basename(args[args.index("-c") + 1]) + "\\n")
""")
    script.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    (tmp_path / "cuda" / "bin").mkdir(parents=True)
    (tmp_path / "cuda" / "bin" / "nvcc").symlink_to(script)
    return tmp_path / "running"


def _most_at_once(running: Path) -> int:
    most = now = 0
    for line in running.read_text().split():
        now += 1 if line == "+" else -1
        most = max(most, now)
    running.unlink()
    return most


def test_jobs_change_no_output_and_the_choice_is_logged(host, tmp_path, monkeypatch, caplog) -> None:
    """The library links its objects in the sources' order at any job count; the default follows the cap, `jobs`
    overrides it, and one line names the choice."""
    running = _fake_nvcc(tmp_path, monkeypatch)
    src = tmp_path / "csrc"
    src.mkdir()
    for name in ("b.cu", "a.cu", "d.cu", "c.cu", "e.cu"):
        (src / name).write_text(f"// {name}\n")
    srcs = sorted(src.glob("*.cu"))
    nvcc = build.find_nvcc()
    host.meminfo(int(2.5 * KERNELS) // 1024)  # room for two jobs
    libs = []
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.kernels.build"):
        for jobs in (None, 1, 4):
            lib = tmp_path / f"out{jobs}" / build.LIB_NAME
            build._compile(nvcc, srcs, [], [], lib, jobs)
            libs.append(lib.read_bytes())
            assert _most_at_once(running) <= (jobs or 2)
    assert libs[0] == libs[1] == libs[2] == b"a.cu\nb.cu\nc.cu\nd.cu\ne.cu\n"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("building ")]
    assert len(lines) == 3
    assert lines[0].startswith(f"building {build.LIB_NAME}: 2 nvcc jobs (min of 8, 16 CPUs") and "a job" in lines[0]
    assert lines[1] == f"building {build.LIB_NAME}: 1 nvcc job (--jobs)"
    assert lines[2] == f"building {build.LIB_NAME}: 4 nvcc jobs (--jobs)"


def test_the_cli_takes_jobs(host, tmp_path, monkeypatch) -> None:
    """`python -m llm_mini_infer.kernels.build --jobs N` passes N to both libraries; 0 is the default; a negative count
    is refused."""
    seen = []
    monkeypatch.setattr(build, "build", lambda archs, force, jobs: seen.append(("kernels", jobs)) or "k")
    monkeypatch.setattr(build, "build_cutlass", lambda archs, force, jobs: seen.append(("cutlass", jobs)) or "c")
    monkeypatch.setattr(build, "record_nvcc_version", lambda: "v")
    for argv, want in ((["--jobs", "3"], 3), ([], None), (["--jobs", "0"], None)):
        monkeypatch.setattr(sys, "argv", ["build", "--arch", "120", *argv])
        build.main()
        assert seen[-2:] == [("kernels", want), ("cutlass", want)]
    monkeypatch.setattr(sys, "argv", ["build", "--jobs", "-1"])
    with pytest.raises(SystemExit):
        build.main()


def test_without_nvcc_the_lookup_keys_on_the_builds_record(tmp_path, monkeypatch) -> None:
    """The ahead-of-time build records `nvcc --version` beside its libraries; a runtime without nvcc (the image's)
    keys its lookup on that record, so it finds the libraries under the keys the build gave them, and with neither
    nvcc nor a record it searches nothing and says so."""
    _fake_nvcc(tmp_path, monkeypatch)
    cutlass = tmp_path / "cutlass"
    (cutlass / "include" / "cutlass").mkdir(parents=True)
    (cutlass / "include" / "cutlass" / "cutlass.h").write_text("")
    (cutlass / "include" / "cutlass" / "version.h").write_text("#define CUTLASS_MAJOR 4\n")
    monkeypatch.setenv("CUTLASS_HOME", str(cutlass))
    # the build's cache: the runtime's prebuilt
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "pre"))
    monkeypatch.delenv("LLM_MINI_INFER_KERNEL_PREBUILT", raising=False)

    def keys(cache: Path) -> list[Path]:
        return [p.relative_to(cache) for p in (build.library_path(["120"]), build.cutlass_library_path(["120"]))]

    built = keys(tmp_path / "pre")
    record = build.record_nvcc_version()
    assert record == tmp_path / "pre" / build.NVCC_VERSION_FILE
    assert record.read_text() == "Cuda compilation tools, release 13.0, V13.0.88\n"  # the fake's own answer

    monkeypatch.setattr(build, "find_nvcc", lambda: None)
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_PREBUILT", f"{tmp_path / 'empty'}:{tmp_path / 'pre'}")
    assert keys(tmp_path / "cache") == built and build.nvcc_version() == record.read_text()
    monkeypatch.setenv("LLM_MINI_INFER_KERNEL_PREBUILT", str(tmp_path / "empty"))
    assert build.library_path(["120"]) is None and build.cutlass_library_path(["120"]) is None
    monkeypatch.setattr(build, "local_arch", lambda: "120")
    assert "nor a prebuilt directory recording its version" in build._missing()
    with pytest.raises(RuntimeError, match="nvcc not found"):
        build.record_nvcc_version()


@functools.cache
def nvcc_compiles() -> bool:
    """nvcc that compiles a kernel: nvcc with its host compiler and cicc. The slim image's nvcc answers `--version`
    (the kernel cache's key, FlashInfer's import) and compiles nothing."""
    nvcc = build.find_nvcc()
    if nvcc is None:
        return False
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.cu"
        src.write_text("__global__ void probe(int *x) { *x = 1; }\n")
        try:
            out = subprocess.run([nvcc, "-c", "-arch=sm_120", str(src), "-o", str(Path(d) / "probe.o")],
                                 capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return out.returncode == 0


@pytest.mark.skipif(not nvcc_compiles(), reason="needs nvcc that compiles (with its host compiler and cicc)")
def test_two_builds_give_the_same_bytes(tmp_path, monkeypatch) -> None:
    """With nvcc: the fused ops' library built at one job and at the default (two caches, so two builds) is the same
    file byte for byte; unstripped, nvcc's process-id-named intermediates would make every build differ. Its launchers
    stay exported."""
    import ctypes
    import hashlib

    libs = []
    for jobs, name in ((1, "one"), (None, "default")):
        monkeypatch.setenv("LLM_MINI_INFER_KERNEL_CACHE", str(tmp_path / name))
        libs.append(build.build(["120"], force=True, jobs=jobs))
    a, b = (hashlib.sha256(p.read_bytes()).hexdigest() for p in libs)
    assert a == b, (a, b)
    lib = ctypes.CDLL(str(libs[0]))
    assert all(hasattr(lib, name) for name in build.SIGNATURES)

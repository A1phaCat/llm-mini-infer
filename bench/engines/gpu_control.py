#!/usr/bin/env python3
"""A two-minute GPU control: the same torch code under every engine's interpreter, so a difference between hosts, or
within one pod's life, shows as a number rather than as a doubt about an engine's result.

    python3 bench/engines/gpu_control.py --out control.json [--seconds 35]

Three phases of `--seconds` each, after a short warm-up, timed with CUDA events and sampled by nvidia-smi every 200 ms:
- `bf16`: a bf16 matmul of 8192 x 8192 x 8192 in a loop (TFLOP/s);
- `fp8`: the same shape through `torch._scaled_mm` with e4m3 inputs, per-tensor scales of 1 and a bf16 output
  (TFLOP/s). A refused call is recorded as the phase's `error`, and the other phases still run;
- `copy`: a device-to-device copy of a 2 GiB buffer in a loop (GB/s, a read and a write each counted once).

The sampler (`Sampler`) and the sample summary (`summarize`) are shared with bench/engines/drive.py, which samples
the GPU for the whole life of each unit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

N = 8192  # the matmul's side
COPY_BYTES = 2 * 2**30
SAMPLE_MS = 200
# nvidia-smi's query, with the clock-event field named as the driver knows it: the newer name first, then the older one,
# then none (the field is dropped when the driver refuses both)
BASE_FIELDS = ("timestamp", "power.draw", "clocks.sm", "clocks.mem", "memory.used", "temperature.gpu")
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

# A sample: (monotonic seconds, SM clock in MHz or None, power in W or None, event-reason mask or None)
Sample = tuple[float, float | None, float | None, int | None]


def parse_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_mask(text: str) -> int | None:
    try:
        return int(text.strip(), 0)
    except ValueError:
        return None


def parse_row(line: str, with_reasons: bool) -> tuple[float | None, float | None, int | None] | None:
    """One nvidia-smi CSV row of `BASE_FIELDS` (and the event-reason mask when `with_reasons`) as (SM MHz, watts,
    mask); None for a row that does not have the columns."""
    cols = [c.strip() for c in line.split(",")]
    if len(cols) != len(BASE_FIELDS) + (1 if with_reasons else 0):
        return None
    mask = parse_mask(cols[-1]) if with_reasons else None
    return parse_number(cols[2]), parse_number(cols[1]), mask


def summarize(samples: list[Sample], start: float, end: float) -> dict:
    """The median SM clock and power of the samples taken between `start` and `end` (monotonic seconds), and the
    distinct non-zero event-reason masks among them. A window without samples gives nulls and no masks."""
    window = [s for s in samples if start <= s[0] <= end]
    clocks = [s[1] for s in window if s[1] is not None]
    watts = [s[2] for s in window if s[2] is not None]
    masks = sorted({s[3] for s in window if s[3]})
    return {"sm_mhz_median": statistics.median(clocks) if clocks else None,
            "power_w_median": statistics.median(watts) if watts else None, "throttle": masks}


class Sampler:
    """nvidia-smi every `interval_ms` in a child process, read by a thread. `samples` holds the parsed rows with the
    reader's monotonic time; with `csv_path` each raw row is also written there, behind a monotonic time column.
    Without nvidia-smi (or with a driver that refuses the query) nothing is recorded and the owner carries on."""

    def __init__(self, interval_ms: int = SAMPLE_MS, csv_path: Path | None = None) -> None:
        self.interval_ms = interval_ms
        self.csv_path = csv_path
        self.samples: list[Sample] = []
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.fields: tuple[str, ...] = ()

    def _spawn(self, fields: tuple[str, ...]) -> subprocess.Popen | None:
        cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
               f"-lms={self.interval_ms}"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return None
        time.sleep(0.5)  # a refused query exits at once
        if proc.poll() is not None:
            return None
        return proc

    def start(self) -> bool:
        """Start sampling; False when no query worked."""
        for extra in (*((f,) for f in REASON_FIELDS), ()):
            fields = BASE_FIELDS + extra
            proc = self._spawn(fields)
            if proc is not None:
                self.proc, self.fields = proc, fields
                break
        else:
            return False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return True

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        with contextlib.ExitStack() as stack:
            out = None
            if self.csv_path is not None:
                out = stack.enter_context(open(self.csv_path, "w"))
                out.write("mono_s," + ",".join(self.fields) + "\n")
            for line in self.proc.stdout:
                now = time.monotonic()
                row = parse_row(line, len(self.fields) > len(BASE_FIELDS))
                if row is None:
                    continue
                self.samples.append((now, *row))
                if out is not None:
                    out.write(f"{now:.3f},{line.strip()}\n")
                    out.flush()

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=10)
            if self.proc.poll() is None:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=10)
        self.proc = None


def tflops(side: int, iters: int, seconds: float) -> float:
    """TFLOP/s of `iters` square matmuls of `side` (2 x side^3 operations each) in `seconds`."""
    return 2 * side**3 * iters / seconds / 1e12


def copy_gbs(nbytes: int, iters: int, seconds: float) -> float:
    """GB/s of `iters` copies of `nbytes`: each copy reads and writes the buffer, counted once each."""
    return 2 * nbytes * iters / seconds / 1e9


def run_phase(step, seconds: float, sampler: Sampler, torch) -> tuple[int, float, float, float]:
    """Call `step()` in chunks of 10 until `seconds` have passed, timing each chunk with CUDA events. Returns
    (calls, GPU seconds, start, end) with the times on the monotonic clock."""
    start = time.monotonic()
    calls, gpu_ms = 0, 0.0
    while time.monotonic() - start < seconds:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            step()
        b.record()
        b.synchronize()
        gpu_ms += a.elapsed_time(b)
        calls += 10
    return calls, gpu_ms / 1e3, start, time.monotonic()


def phase(make_step, rate, seconds: float, sampler: Sampler, torch) -> dict:
    """One phase: build its step, warm it up, run it, and report its rate (`rate(calls, gpu_seconds)`) with the
    samples taken during it. A phase that raises records the error."""
    try:
        step = make_step()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        calls, gpu_s, t0, t1 = run_phase(step, seconds, sampler, torch)
    except Exception as exc:  # a refused kernel: the other phases still run
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {**rate(calls, gpu_s), **summarize(sampler.samples, t0, t1), "seconds": round(t1 - t0, 2)}


def control(seconds: float) -> dict:
    import torch

    dev = torch.device("cuda")
    sampler = Sampler(SAMPLE_MS)
    sampler.start()
    try:
        def bf16_step():
            a = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
            b = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
            return lambda: torch.matmul(a, b)

        def fp8_step():
            a = torch.randn(N, N, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(N, N, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()  # column-major
            one = torch.ones((), device=dev, dtype=torch.float32)
            return lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

        def copy_step():
            src = torch.empty(COPY_BYTES, device=dev, dtype=torch.uint8)
            dst = torch.empty_like(src)
            return lambda: dst.copy_(src)

        phases = {
            "bf16": phase(bf16_step, lambda n, s: {"tflops": round(tflops(N, n, s), 1)}, seconds, sampler,
                          torch),
            "fp8": phase(fp8_step, lambda n, s: {"tflops": round(tflops(N, n, s), 1)}, seconds, sampler, torch),
            "copy": phase(copy_step, lambda n, s: {"gbs": round(copy_gbs(COPY_BYTES, n, s), 1)}, seconds,
                          sampler, torch),
        }
    finally:
        sampler.stop()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "phases": phases}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=35.0, help="per phase (default: %(default)s)")
    args = ap.parse_args(argv)
    report = control(args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

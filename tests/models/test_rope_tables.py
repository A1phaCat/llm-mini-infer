"""The RoPE tables that the engine builds on the CPU, bit for bit, under each ATen CPU capability the machine offers.

- **The vision towers' tables** (`models.common.rope_inv_freq`) are the same on every capability. They are the bits of
  an AVX-512 host, where ATen's `pow` takes glibc's scalar `powf` for all 18 values. ATen's vector route (AVX2, NEON)
  takes the first 16 through SLEEF, which gives Gemma 4's element 15 one ULP off.
- **Gemma 4's text tables** keep ATen's `pow`. Their 128 and 64 values fill whole vector groups (8, 16 or 32 values),
  so the vector routes agree. x86's `default` capability is scalar and gives other bits: a host with AVX2 runs it only
  when forced.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ..support.paths import DATA, ROOT
from ..support.rope_tables import qwen_table, words

# Gemma 4 31B's tower (theta 100, head_dim 72) as an AVX-512 host computes it
GEMMA = (0x3F800000, 0x3F463625, 0x3F1977CC, 0x3EEDA63C, 0x3EB800D6, 0x3E8E778F, 0x3E5C9D35, 0x3E2AD041, 0x3E044133,
         0x3DCCCCCD, 0x3D9E91B6, 0x3D758C7B, 0x3D3E1E95, 0x3D1333DF, 0x3CE3F280, 0x3CB07DC6, 0x3C88A69B, 0x3C539B86)
# Qwen3.8's tower (theta 1e4, head_dim 72): ATen's routes all agree on these
QWEN = (0x3F800000, 0x3F1977CC, 0x3EB800D6, 0x3E5C9D35, 0x3E044133, 0x3D9E91B6, 0x3D3E1E95, 0x3CE3F280, 0x3C88A69B,
        0x3C23D70A, 0x3BC47060, 0x3B6B8631, 0x3B0D3169, 0x3AA94938, 0x3A4AF7F3, 0x39F35A5C, 0x3991E2E1, 0x392EE9BF)
# sha256 of Gemma 4 31B's text tables on ATen's vector routes: sliding (128 values), full (64 and 192 zeros)
SLIDING_SHA = "cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15"
FULL_SHA = "71a80d617056a8be95eeeb385ebcc01c7fb2c73d5430a8ce862ed6c94179d076"

CHILD = "import json, sys; from tests.support.rope_tables import tables; print(json.dumps(tables(sys.argv[1])))"
RANK = {"DEFAULT": 0, "AVX2": 1, "AVX512": 2}  # ATen's x86 capabilities; its own choice is the highest the CPU has


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """google/gemma-4-31B-it's config.json (tests/data)."""
    d = tmp_path_factory.mktemp("g4")
    shutil.copy(DATA / "gemma-4-31b-it.config.json", d / "config.json")
    return d


def test_the_vision_tables_are_an_avx512_hosts(checkpoint) -> None:
    """Gemma 4's table in this process, whichever capability ATen picked here, and Qwen3.8's."""
    from llm_mini_infer.models.gemma4.vision import GemmaVisionConfig, vision_buffers

    assert words(vision_buffers(GemmaVisionConfig.from_dir(checkpoint))["inv_freq"]) == GEMMA
    assert words(qwen_table()) == QWEN


def child(checkpoint: Path, requested: str | None) -> dict:
    """`tables` in a fresh process, with ATEN_CPU_CAPABILITY at `requested` (None: unset, the CPU's own choice)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ATEN_CPU_CAPABILITY")}
    if requested:
        env["ATEN_CPU_CAPABILITY"] = requested
    p = subprocess.run([sys.executable, "-c", CHILD, str(checkpoint)], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def native(checkpoint) -> dict:
    return child(checkpoint, None)


@pytest.mark.parametrize("requested", [None, "default", "avx2", "avx512"], ids=["native", "default", "avx2", "avx512"])
def test_the_tables_under_each_cpu_capability(checkpoint, native, requested) -> None:
    """ATen reads ATEN_CPU_CAPABILITY once a process: a child per value. ATen runs an x86 value that it knows without
    checking the CPU, so a value above the CPU's own choice skips (its kernels would fault). A value that the build
    does not know (the macOS wheel knows only `default`) makes ATen warn and run another: that case skips too."""
    if requested and RANK.get(requested.upper(), 0) > RANK.get(native["ran"], 0):
        pytest.skip(f"ATen runs {native['ran']} here, not {requested}: this CPU does not offer it")
    got = child(checkpoint, requested) if requested else native
    ran = got["ran"]
    if requested and ran.lower() != requested:
        pytest.skip(f"ATen ran {ran}, not {requested}: this CPU does not offer it")
    assert tuple(got["gemma"]) == GEMMA
    assert tuple(got["qwen"]) == QWEN
    if ran in ("AVX2", "AVX512") or (ran == "DEFAULT" and platform.machine() in ("arm64", "aarch64")):
        assert (got["sliding"], got["full"]) == (SLIDING_SHA, FULL_SHA)

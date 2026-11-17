"""LICENSE and the files agree: every file it lists exists, adapted code carries its origin's copyright line, no
file with an origin's notice in its first 15 lines (under llm_mini_infer/, tools/, bench/ and tests/, less
tests/.fixtures and tests/data) is missing from it, the Gemma 4 fixtures' pins match, and its licence texts are
whole."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

from ..support.paths import ROOT

VLLM = "SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
MARKERS = ("Adapted from", "Vendored from", "SPDX-FileCopyrightText")
SOURCE = re.compile(r"\.(?:py|cuh?|cpp|hpp|h|sh)$")  # the source files a notice may head
SKIPPED_DIRS = {"__pycache__", "bin", "node_modules"}
HEAD_LINES = 15
LICENCES = {("above", "Apache License, Version 2.0"), ("below", "BSD 3-Clause License")}
ORIGINS = ("vLLM", "Hugging Face transformers", "SGLang", "NVIDIA CUTLASS", "Gemma 4", "Helm")


def notice() -> str:
    """LICENSE's text after the Apache terms."""
    return (ROOT / "LICENSE").read_text().split("END OF TERMS AND CONDITIONS", 1)[1]


def sections() -> dict[str, str]:
    """The origins' sections by name: each starts with `Name (https://url), Licence (above|below)`."""
    parts = re.split(r"\n\n\n(?=.+ \(https://\S+\), .+ \((?:above|below)\)\n)", notice())
    return {m.group(1): p.strip() for p in parts if (m := re.match(r"(.+?) \(https://\S+\), ", p.strip()))}


def bullets(section: str) -> list[tuple[list[str], str]]:
    """The section's `- <path>[, <path>]:` bullets: their files (a bare name takes the first path's directory) and the
    text under them."""
    out = []
    for block in re.split(r"\n(?=- )", section):
        if not block.startswith("- "):
            continue
        text = re.sub(r"\n\s+", " ", block[2:])
        head = re.match(r"((?:llm_mini_infer|tests)/[^:]+):(?:\s|$)", text)
        if not head:
            continue
        names = [n.strip() for n in head.group(1).split(",")]
        base = names[0].rsplit("/", 1)[0]
        files = [n if "/" in n else f"{base}/{n}" for n in names]
        out.append((files, block))
    return out


def listed() -> set[str]:
    return {f for s in sections().values() for files, _ in bullets(s) for f in files}


def head_text(rel: str, lines: int = 12) -> str:
    return "\n".join((ROOT / rel).read_text().splitlines()[:lines])


def tree_files(root: Path, prefix: str, skip: tuple[str, ...] = ()) -> list[tuple[str, Path]]:
    """The source files under `root` (a symlink is resolved) as (path from the repository's root, file); `skip` names
    folders of `root` left out (by their path from it)."""
    found = []
    base = root.resolve()
    for folder, dirs, names in os.walk(base):
        here = Path(folder).relative_to(base)
        dirs[:] = [d for d in dirs
                   if d not in SKIPPED_DIRS and not d.startswith(".") and (here / d).as_posix() not in skip]
        for name in names:
            path = Path(folder) / name
            if SOURCE.search(path.name):
                found.append((f"{prefix}/{path.relative_to(root.resolve()).as_posix()}", path))
    return found


def unlisted(files: list[tuple[str, Path]], known: set[str]) -> list[str]:
    """The files whose first lines (`HEAD_LINES`) hold an origin's marker and that `known` lacks."""
    bad = []
    for rel, path in files:
        top = "\n".join(path.read_text(errors="ignore").splitlines()[:HEAD_LINES])
        if rel not in known and any(m in top for m in MARKERS):
            bad.append(rel)
    return bad


def test_every_listed_file_exists() -> None:
    files = listed()
    assert len(files) >= 20
    for rel in sorted(files):
        assert (ROOT / rel).is_file(), rel


def test_the_files_adapted_from_vllm_carry_its_copyright_line() -> None:
    adapted = [f for files, _ in bullets(sections()["vLLM"]) for f in files if f.startswith("llm_mini_infer/")]
    assert len(adapted) >= 10 and any(a.endswith(".cu") for a in adapted)
    for rel in adapted:
        assert VLLM in head_text(rel), rel


def test_the_files_adapted_from_transformers_carry_the_copyright_line_under_their_bullet() -> None:
    checked = 0
    for files, block in bullets(sections()["Hugging Face transformers"]):
        line = re.search(r"^\s+(Copyright .+)$", block, re.M)
        assert line, block
        for rel in files:
            assert line.group(1) in head_text(rel), rel
            checked += 1
    assert checked >= 5


def test_no_file_with_an_origins_notice_is_missing_from_the_licence() -> None:
    files = [f for d in ("llm_mini_infer", "tools", "bench") for f in tree_files(ROOT / d, d)]
    files += tree_files(ROOT / "tests", "tests", skip=("data",))  # the dot folders (.fixtures) are skipped already
    assert len(files) > 200 and any(rel.startswith(("bench/", "tests/")) for rel, _ in files)
    assert unlisted(files, listed()) == []


@pytest.mark.parametrize("marker", MARKERS)
def test_the_reverse_check_finds_an_unlisted_file(marker: str, tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "d.py").write_text(f"# {marker} somewhere\n")
    (tmp_path / "e.py").write_text("\n" * 12 + f"# {marker} somewhere\n")
    (tmp_path / "f.py").write_text("\n" * 15 + f"# {marker} somewhere\n")
    (tmp_path / "a.py").write_text(f"# {marker} somewhere\n")
    (tmp_path / "b.py").write_text("# nothing\n")
    (tmp_path / "c.py").write_text(f"# {marker} somewhere\n")
    files = tree_files(tmp_path, "pkg", skip=("data",))
    assert sorted(unlisted(files, set())) == ["pkg/a.py", "pkg/c.py", "pkg/e.py"]
    assert unlisted(files, {"pkg/a.py", "pkg/c.py", "pkg/e.py"}) == []


def test_the_gemma_fixtures_match_their_pins() -> None:
    section = sections()["Gemma 4"]
    pins = (ROOT / "tests" / "scripts" / "fetch_fixtures.py").read_text()
    pinned = re.search(r'"google/gemma-4-31B-it", "([0-9a-f]{40})"', pins)
    assert pinned
    assert f"google/gemma-4-31B-it at commit\n  {pinned.group(1)}" in section
    sha = re.search(r"(tests/data/gemma-4-31b-it-fp8-block\.config\.json):.*?sha256 ([0-9a-f]{64})", section, re.S)
    assert sha and hashlib.sha256((ROOT / sha.group(1)).read_bytes()).hexdigest() == sha.group(2)


def test_the_licence_holds_the_texts_its_sections_name() -> None:
    """The Apache License 2.0 text first (GitHub's licence detector reads it, and ignores what follows its end of
    terms), then our notice and the origins' sections, each pointing to the Apache text above or to CUTLASS's BSD
    3-Clause text, which comes whole at the end. No NOTICE or LICENSES/ beside it."""
    apache = (ROOT / "LICENSE").read_text().split("END OF TERMS AND CONDITIONS", 1)[0]
    rest = notice()
    assert apache.lstrip().startswith("Apache License") and "Version 2.0, January 2004" in apache
    assert "\nllm-mini-infer\nCopyright 2026 The llm-mini-infer Authors\n" in rest
    named = re.findall(r"^(.+?) \(https://\S+\), (.+) \((above|below)\)$", rest, re.M)
    assert {n for n, _, _ in named} >= set(ORIGINS)
    assert all((where, text) in LICENCES for _, text, where in named)
    bsd = rest.rsplit("NVIDIA CUTLASS's licence: the BSD 3-Clause License\n", 1)[1]
    assert "SPDX-License-Identifier: BSD-3-Clause" in bsd and "Redistribution and use in source and binary forms" in bsd
    assert not (ROOT / "NOTICE").exists() and not (ROOT / "LICENSES").exists()

""".github/scripts/ci_changes.py, ci.yml's choice of whether to run the fuzz smoke, on a scratch git repository."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from ..support.paths import ROOT

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="no git (the GPU tests' image has none)")

CODE = '''"""A module."""


def f(x):
    """Doubles."""
    return 2 * x  # twice
'''


def test_the_fuzz_smoke_runs_for_code_and_skips_comments_and_docstrings(tmp_path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                              cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    def commit(files: dict[str, str]) -> str:
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", "c")
        return git("rev-parse", "HEAD")

    def fuzz(base: str, head: str) -> tuple[str, str]:
        r = subprocess.run([sys.executable, str(ROOT / ".github" / "scripts" / "ci_changes.py"), base, head],
                           cwd=tmp_path, capture_output=True, text=True, check=True)
        return r.stdout.strip(), r.stderr.strip()

    git("init", "-q")
    base = commit({"llm_mini_infer/m.py": CODE, "bench/b.py": "x = 1\n", "docs/a.md": "a\n"})
    comments = commit({"llm_mini_infer/m.py": CODE.replace("Doubles.", "Twice x.").replace("# twice", "# 2x")})
    assert fuzz(base, comments) == ("fuzz=false", "fuzz: no fuzz input changed beyond comments and docstrings")
    outside = commit({"bench/b.py": "x = 2\n", "docs/a.md": "b\n", "llm_mini_infer/notes.md": "c\n"})
    assert fuzz(comments, outside)[0] == "fuzz=false"  # bench/ and Markdown are no fuzz input
    literal = commit({"llm_mini_infer/m.py": CODE.replace("return 2 * x", "return 'x' * x")})
    assert fuzz(outside, literal) == ("fuzz=true", "fuzz: llm_mini_infer/m.py: its code changed")
    data = commit({"tests/data/fuzz_regressions/r.json": "{}\n"})
    assert fuzz(literal, data) == ("fuzz=true", "fuzz: tests/data/fuzz_regressions/r.json changed")
    assert fuzz("0" * 40, data)[0] == "fuzz=true"  # a branch's first push: nothing to compare with
    assert fuzz("f" * 40, data)[0] == "fuzz=true"  # a base the history lacks (a force push)
    broken = commit({"llm_mini_infer/m.py": "def f(:\n"})
    assert fuzz(data, broken)[0] == "fuzz=true"  # a file that no longer parses

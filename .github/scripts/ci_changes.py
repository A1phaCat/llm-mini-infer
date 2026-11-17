"""Whether a change needs ci.yml's fuzz smoke: prints `fuzz=true` or `fuzz=false` for $GITHUB_OUTPUT, and the reason to
stderr. Usage: python3 .github/scripts/ci_changes.py BASE HEAD (run on the runner's own Python: the standard library and
git).

The fuzz suites drive the scheduler and the cache over the fake runner, so they read `llm_mini_infer/` and `tests/`
alone, with the lock and this workflow. A Python file counts as unchanged when its syntax tree, without docstrings, is
the same: comments are not in the tree, and every other string stays, since a literal can change what the code does. Any
other file under those paths (fixtures, regression examples, kernel sources) counts as changed. Without a usable BASE
(a branch's first push, a force push, a history the checkout lacks) the smoke runs.
"""

from __future__ import annotations

import ast
import subprocess
import sys

INPUTS = ("llm_mini_infer/", "tests/", "pyproject.toml", "uv.lock", ".github/workflows/ci.yml")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def code(rev: str, path: str) -> str | None:
    """The file's syntax tree without docstrings, or None when it is absent or does not parse."""
    r = git("show", f"{rev}:{path}")
    if r.returncode:
        return None
    try:
        tree = ast.parse(r.stdout)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body and \
                isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def needs_fuzz(base: str, head: str) -> tuple[bool, str]:
    if not base.strip("0") or git("cat-file", "-e", f"{base}^{{commit}}").returncode:
        return True, f"no base commit to compare with ({base or 'none'})"
    r = git("diff", "--name-only", base, head)
    if r.returncode:
        return True, f"git diff failed: {r.stderr.strip()}"
    for path in r.stdout.split():
        if not path.startswith(INPUTS) or path.endswith(".md"):
            continue
        if not path.endswith(".py"):
            return True, f"{path} changed"
        before, after = code(base, path), code(head, path)
        if before is None or after is None or before != after:
            return True, f"{path}: its code changed"
    return False, "no fuzz input changed beyond comments and docstrings"


if __name__ == "__main__":
    fuzz, why = needs_fuzz(sys.argv[1], sys.argv[2])
    print(f"fuzz: {why}", file=sys.stderr)
    print(f"fuzz={'true' if fuzz else 'false'}")

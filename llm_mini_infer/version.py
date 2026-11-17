"""The server's version, and the engine marker every response carries (README.md, "The API").

In a mixed fleet (several engines behind one router or split), each response says which engine made it:
- `system_fingerprint` in every chat completion and every streamed chunk: `llm-mini-infer-<version>`;
  OpenAI's clients expose the field (`completion.system_fingerprint`);
- the `x-llm-engine` response header: `llm-mini-infer/<version>`.

The version is the package's, plus the image's build id (`LLM_MINI_INFER_BUILD`, the git commit deploy/Dockerfile
records) as semver build metadata when there is one: `<version>+<commit>`.
"""

from __future__ import annotations

import functools
import os

ENGINE_HEADER = b"x-llm-engine"


@functools.cache
def version() -> str:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        ver = pkg_version("llm-mini-infer")
    except PackageNotFoundError:
        ver = "unknown"
    build = os.environ.get("LLM_MINI_INFER_BUILD", "")
    if build and build != "unknown":
        ver += f"+{build}"
    return ver


def fingerprint() -> str:
    return f"llm-mini-infer-{version()}"


def engine() -> str:
    return f"llm-mini-infer/{version()}"

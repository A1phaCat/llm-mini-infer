# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/entrypoints/serve/middleware/authenticate.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm),
# and changed for llm-mini-infer.
"""API keys: vLLM's `--api-key` middleware (`vllm/entrypoints/serve/middleware/authenticate.py`), deny by default.

When a key is configured, every request needs `Authorization: Bearer <key>` but OPTIONS and the four open paths,
`OPEN_PATHS`: `/health`, `/ready`, `/live` and `/metrics`, for probes and scrapes. They match exactly, on the path the
router reads (`/health?ready=1` is `/health`; `/health/`, `/healthz` and `/metrics/x` are not open). The check comes
before routing, so without the key a path the server does not serve is a 401 too, and a route added later is guarded.
OPTIONS reaches no other handler (`App.refuse_options`). vLLM guards only the paths under `/v1`, `/v2`, `/inference`
and `/cohere`, and SGLang opens its probes by prefix. A missing or wrong key is a 401 with an OpenAI-style error body.

The key comes only from the environment, never from a configuration file: `LLM_MINI_INFER_API_KEY`, or `VLLM_API_KEY`,
so that a deployment that sets it keeps its secret unchanged; when both are set, either is accepted. Only SHA-256
digests are kept, compared in constant time. The key itself never reaches the configuration, `/server_info`, a log
line or an error message.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Mapping

ENV_VARS = ("LLM_MINI_INFER_API_KEY", "VLLM_API_KEY")
OPEN_PATHS = frozenset(("/health", "/ready", "/live", "/metrics"))


def client_headers(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The Authorization header for a client of this server started in the same environment (the host check's): the
    first key `ENV_VARS` set, as Bearer; none without one."""
    env = os.environ if environ is None else environ
    key = next((env[v] for v in ENV_VARS if env.get(v)), "")
    return {"authorization": f"Bearer {key}"} if key else {}


class ApiKeys:
    def __init__(self, environ: Mapping[str, str] | None = None):
        env = os.environ if environ is None else environ
        self.sources = [v for v in ENV_VARS if env.get(v)]
        self._digests = list(dict.fromkeys(hashlib.sha256(env[v].encode()).digest() for v in self.sources))

    @property
    def enabled(self) -> bool:
        return bool(self._digests)

    def guards(self, method: str, path: str) -> bool:
        """Whether this request needs a key: with one set, every request but OPTIONS and the open paths'."""
        return self.enabled and method != "OPTIONS" and path not in OPEN_PATHS

    def allows(self, headers: list[tuple[bytes, bytes]]) -> bool:
        """`Authorization: Bearer <key>` (the first Authorization header, the scheme in any case) matches a key."""
        value = next((v for n, v in headers if n == b"authorization"), None)
        if not value:
            return False
        scheme, _, token = value.partition(b" ")
        if scheme.lower() != b"bearer":
            return False
        # the header as starlette decodes it (latin-1), re-encoded as UTF-8: the bytes vLLM hashes (the same for
        # ASCII keys)
        digest = hashlib.sha256(token.decode("latin-1").encode()).digest()
        ok = False
        for d in self._digests:  # every digest, no early exit
            ok |= secrets.compare_digest(digest, d)
        return ok

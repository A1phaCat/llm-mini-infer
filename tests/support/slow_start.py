"""A frontend that starts `SLOW_START_S` (the environment's, default 60) seconds late, for the launcher's checks while
its frontends start (tests/api/test_supervise.py). `serve` runs it in place of `api.server.frontend_main`."""

from __future__ import annotations

import os
import time
from typing import Any


def frontend_main(*args: Any, **kwargs: Any) -> None:
    time.sleep(float(os.environ.get("SLOW_START_S", "60")))
    from llm_mini_infer.api.server import frontend_main as main

    main(*args, **kwargs)


def serve(config: str) -> None:
    """`llm-mini-infer serve --config <config>` with the slow frontends."""
    from llm_mini_infer import cli
    from llm_mini_infer.api import server

    server.frontend_main = frontend_main
    cli.main(["serve", "--config", config])

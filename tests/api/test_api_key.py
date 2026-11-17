"""The API key's startup line (`log_api_key`) and the test harness's handling of the key variables."""

from __future__ import annotations

import logging
import os
import subprocess

import pytest

from llm_mini_infer.api.app import log_api_key
from llm_mini_infer.api.auth import ApiKeys
from tests.support.api_util import Server

KEY = "sk-test-" + "z" * 24


def lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "llm_mini_infer.api.app"]


def test_the_startup_line_with_a_key_names_the_open_paths_and_the_variable_never_the_key(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm_mini_infer.api.app")
    log_api_key(ApiKeys({"VLLM_API_KEY": KEY}), "0.0.0.0")
    (rec,) = lines(caplog)
    text = rec.getMessage()
    assert rec.levelno == logging.INFO and KEY not in text
    for name in ("OPTIONS", "/health", "/ready", "/live", "/metrics", "VLLM_API_KEY"):
        assert name in text
    assert "LLM_MINI_INFER_API_KEY" not in text


def test_the_startup_line_names_both_variables_when_both_are_set(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm_mini_infer.api.app")
    log_api_key(ApiKeys({"LLM_MINI_INFER_API_KEY": KEY, "VLLM_API_KEY": KEY + "2"}), "127.0.0.1")
    text = lines(caplog)[0].getMessage()
    assert "LLM_MINI_INFER_API_KEY" in text and "VLLM_API_KEY" in text and KEY not in text


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.1.2.3", "example.org"])
def test_without_a_key_the_startup_line_warns_when_the_server_listens_beyond_loopback(caplog, host) -> None:
    caplog.set_level(logging.DEBUG, logger="llm_mini_infer.api.app")
    log_api_key(ApiKeys({}), host)
    (rec,) = lines(caplog)
    assert rec.levelno == logging.WARNING and "every path is open" in rec.getMessage()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_without_a_key_the_startup_line_is_informational_on_loopback(caplog, host) -> None:
    caplog.set_level(logging.DEBUG, logger="llm_mini_infer.api.app")
    log_api_key(ApiKeys({}), host)
    (rec,) = lines(caplog)
    assert rec.levelno == logging.INFO and "every path is open" in rec.getMessage()


def test_the_harness_drops_the_key_variables_unless_the_test_sets_them(monkeypatch, tmp_path) -> None:
    """Server.launch gives the server the tests' environment without a key, so a key in the developer's shell neither
    guards the tests' servers nor reaches them."""
    for name in ("LLM_MINI_INFER_API_KEY", "VLLM_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "inherited")
    seen: list[dict] = []

    class Stub:
        def __init__(self, cmd, **kw):
            seen.append(kw["env"])

    monkeypatch.setattr(subprocess, "Popen", Stub)
    model = tmp_path / "model"
    model.mkdir()
    Server(model, tmp_path).launch().log.close()
    Server(model, tmp_path, env={"VLLM_API_KEY": KEY}).launch().log.close()
    bare, keyed = seen
    assert "LLM_MINI_INFER_API_KEY" not in bare and "VLLM_API_KEY" not in bare and "HF_TOKEN" not in bare
    assert "LLM_MINI_INFER_API_KEY" not in keyed and keyed["VLLM_API_KEY"] == KEY
    assert os.environ["VLLM_API_KEY"] == "inherited"

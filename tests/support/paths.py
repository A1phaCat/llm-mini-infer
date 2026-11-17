"""The repository's paths for the tests and their tools. The standard library only: the image's checks import it with
pytest alone."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"  # the tests' committed data
# The model fixtures (tokenizers, chat templates, configuration files; no weights) and the chat-template corpora, in
# the git-ignored tests/.fixtures/: uv run tests/scripts/fetch_fixtures.py
FIXTURES = ROOT / "tests" / ".fixtures"
MODELS_DIR = FIXTURES / "models"
# Every tool's default output folder (git-ignored): results/<tool>/
RESULTS = ROOT / "results"

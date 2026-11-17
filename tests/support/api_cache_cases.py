"""The API cache tests' request settings and long system prompt (tests/api/test_api_cache.py)."""

from __future__ import annotations

KW = {"enable_thinking": False}
SYSTEM = " ".join(f"Rule {i}: label items by their main material, then by colour." for i in range(60))

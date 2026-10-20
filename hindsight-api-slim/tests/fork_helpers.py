"""Shared setup for the fork's tests (FORK.md: fork helpers live in fork-owned files, not in upstream ones).

- ``build_codex``: a ``CodexLLM`` built without reading a real Codex login.
- ``cost_env_cleared``: the body of an autouse fixture that removes every ``HINDSIGHT_API_LLM_COST_*``
  variable (plus any names given), optionally turns the flag on, and resets the settings caches around each test.
- ``enable_cost``: turns the cost flag on inside such a test.
"""

import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.llm_cost import load_cost_settings
from hindsight_api.engine.providers.codex_llm import CodexLLM

COST_ENABLED_ENV = "HINDSIGHT_API_LLM_COST_ENABLED"


def build_codex(model: str = "gpt-5.4-mini", **kwargs: Any) -> CodexLLM:
    """A ``CodexLLM`` on the ChatGPT backend URL, with a stub login (token, account) and no refresh token."""
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model=model,
            **kwargs,
        )


def _reset_caches() -> None:
    clear_config_cache()
    load_cost_settings.cache_clear()


def cost_env_cleared(monkeypatch: pytest.MonkeyPatch, *extra_names: str, enabled: bool = False) -> Iterator[None]:
    """Use as ``yield from cost_env_cleared(monkeypatch)`` in an autouse fixture; ``enabled`` turns the flag on."""
    for name in list(os.environ):
        if name.startswith("HINDSIGHT_API_LLM_COST_") or name in extra_names:
            monkeypatch.delenv(name)
    if enabled:
        monkeypatch.setenv(COST_ENABLED_ENV, "true")
    _reset_caches()
    yield
    _reset_caches()


def enable_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COST_ENABLED_ENV, "true")
    _reset_caches()

"""Live check of the Codex quota fetch against the real ChatGPT backend (fork: cost-accounting).

Runs only with ``HINDSIGHT_QUOTA_LIVE=1`` and a Codex login at ``$CODEX_HOME/auth.json``
(``~/.codex/auth.json`` when ``CODEX_HOME`` is unset).

WARNING: the fetch goes through the same token path as every Codex call. If the access token
is near expiry it is refreshed, and ``auth.json`` is rewritten with the new tokens. The quota
path itself never forces a refresh.

The test prints only the pass/fail outcome and window labels. No token, header or body is printed.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight_api.engine.llm_quota import LLMQuotaService, QuotaSettings
from hindsight_api.engine.providers.codex_llm import CodexLLM


def _codex_auth_file() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
    return base / "auth.json"


_LIVE_ENABLED = os.environ.get("HINDSIGHT_QUOTA_LIVE") == "1"
_HAS_LOGIN = _codex_auth_file().is_file()

pytestmark = [
    pytest.mark.skipif(
        not (_LIVE_ENABLED and _HAS_LOGIN),
        reason="set HINDSIGHT_QUOTA_LIVE=1 and log in to Codex (auth.json) to run",
    ),
    pytest.mark.asyncio,
]


async def test_codex_quota_live_snapshot():
    llm = CodexLLM(provider="openai-codex", api_key="", base_url="", model="gpt-6.1-sol")
    engine = SimpleNamespace(
        _llm_config=SimpleNamespace(
            provider="openai-codex",
            api_key="",
            base_url=llm.base_url,
            _provider_impl=llm,
        )
    )
    service = LLMQuotaService(engine, QuotaSettings(enabled=True))

    response = await service.fetch_all("live-check")

    assert len(response.providers) == 1
    provider = response.providers[0]
    assert provider.status == "ok", f"status={provider.status} reason={provider.reason}"
    assert provider.windows, "expected at least one usage window"
    now = datetime.now(UTC)
    for window in provider.windows:
        assert window.used_percent is not None
        assert 0 <= window.used_percent <= 100
        assert window.resets_at is not None
        assert window.resets_at > now
    print("codex quota windows:", [window.label for window in provider.windows])

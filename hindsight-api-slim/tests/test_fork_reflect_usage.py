"""Fork regression test: reflect_async must carry the agent's cached/thoughts tokens
into the response TokenUsage (#5487).

Adapted from akushonkamen/hindsight@470778940 (#5487), the reflect_async section of
tests/test_token_usage_cached_thoughts.py, moved into its own file so the fork does
not edit the upstream test module.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect.models import ReflectAgentResult, TokenUsageSummary
from hindsight_api.models import RequestContext


def _reflect_engine_with_agent(monkeypatch, agent_result) -> MemoryEngine:
    """A shell MemoryEngine whose I/O collaborators are stubbed and whose reflect
    agent is replaced by one returning ``agent_result`` — light enough to drive
    reflect_async all the way to its response assembly without a database."""
    engine = object.__new__(MemoryEngine)
    engine._reflect_llm_config = SimpleNamespace(provider="openai")
    engine._operation_validator = None
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=SimpleNamespace(
                llm_output_language=None,
                reflect_use_decision_model=False,
                mental_model_refresh_use_decision_model=False,
            )
        ),
        get_bank_config=AsyncMock(return_value={}),
    )
    engine._authenticate_tenant = AsyncMock()
    engine._tag_scope = AsyncMock(return_value=None)
    engine.ensure_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})
    engine._bank_freshness = AsyncMock(return_value={})
    engine._evidence_as_stored = AsyncMock(return_value=[])
    engine.list_directives = AsyncMock(return_value=SimpleNamespace(items=[]))
    engine._llm_for_reflect_operation = lambda _op: SimpleNamespace(with_config=lambda *_a, **_k: SimpleNamespace())

    class _Conn:
        async def fetchval(self, *_a, **_k):
            return 0

        async def fetch(self, *_a, **_k):
            return []

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_a):
            return False

    backend = MagicMock()
    backend.acquire.return_value = _Acquire()
    engine._get_backend = AsyncMock(return_value=backend)

    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.run_reflect_agent",
        AsyncMock(return_value=agent_result),
    )
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.get_config",
        lambda: SimpleNamespace(
            reflect_max_iterations=2,
            reflect_max_context_tokens=4096,
            reflect_wall_timeout=5,
            database_schema="public",
        ),
    )
    return engine


@pytest.mark.asyncio
async def test_reflect_async_propagates_agent_cached_and_thoughts_tokens(monkeypatch):
    """reflect_async converts ReflectAgentResult.usage into the response's TokenUsage —
    cached_tokens and thoughts_tokens must survive that conversion, not reset to 0."""
    agent_result = ReflectAgentResult(
        text="answer",
        usage=TokenUsageSummary(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=123,
            thoughts_tokens=45,
        ),
    )
    engine = _reflect_engine_with_agent(monkeypatch, agent_result)

    result = await engine.reflect_async("bank-1", "question", request_context=RequestContext())

    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.cached_tokens == 123  # was 0: dropped by the TokenUsage conversion (#5487)
    assert result.usage.thoughts_tokens == 45  # was 0: dropped by the TokenUsage conversion (#5487)

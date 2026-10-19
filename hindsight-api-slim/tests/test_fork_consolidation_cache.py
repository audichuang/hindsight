"""Fork: consolidation-cache — a batch-invariant schema for providers that send it as prompt text.

See ``FORK-CONSOLIDATION-CACHE.md``. The Codex provider appends the response schema to
``instructions`` when strict_schema is off; the per-batch ``enum`` / ``maxItems`` made
``instructions`` differ on every consolidation batch and the prompt cache missed.
"""

import inspect
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import LLM_STRATEGY_FAILOVER, LLMStrategyConfig
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.cacheable_schema import schema_is_prompt_text
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.llm_wrapper import ConfiguredLLMProvider, LLMCallResult, LLMProvider
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.providers.codex_llm import CodexLLM
from hindsight_api.engine.response_models import TokenUsage
from tests.codex_stream_stub import stub_codex_stream

_BATCH_A = ("11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")
_BATCH_B = ("33333333-3333-4333-8333-333333333333",)
_EMPTY_BATCH = '{"creates": [], "updates": [], "deletes": []}'


def _config(*, strict: bool = False) -> MagicMock:
    config = MagicMock()
    config.llm_output_language = None
    config.observations_mission = None
    config.consolidation_max_attempts = 1
    config.consolidation_llm_max_retries = None
    config.consolidation_max_completion_tokens = None
    config.llm_strict_schema_consolidation = strict
    config.llm_temperature_consolidation = 0.0
    config.llm_supports_max_items = True
    return config


def _batch(*ids: str) -> list[dict[str, Any]]:
    return [{"id": i, "text": f"fact {i}"} for i in ids]


class _Recorder:
    """An llm_config stand-in that parses ``reply`` with the response_format, as a provider does."""

    def __init__(self, provider: str | None, reply: str = _EMPTY_BATCH) -> None:
        if provider is not None:
            self.provider = provider
        self.reply = reply
        self.response_formats: list[Any] = []

    async def call(self, **kwargs: Any) -> LLMCallResult:
        response_format = kwargs["response_format"]
        self.response_formats.append(response_format)
        return LLMCallResult(content=response_format.model_validate(json.loads(self.reply)), usage=TokenUsage())


async def _run_two_batches(llm: Any, *, strict: bool = False) -> None:
    """Two batches with different fact ids and different remaining observation slots."""
    config = _config(strict=strict)
    for ids, slots in ((_BATCH_A, 3), (_BATCH_B, 1)):
        await _consolidate_batch_with_llm(
            llm_config=llm,
            memories=_batch(*ids),
            union_observations=[],
            union_source_facts={},
            config=config,
            remaining_observation_slots=slots,
            max_observations_per_scope=10,
        )


async def _schemas_for_two_batches(llm: _Recorder, *, strict: bool = False) -> list[dict[str, Any]]:
    await _run_two_batches(llm, strict=strict)
    return [rf.model_json_schema() for rf in llm.response_formats]


def test_consolidator_keeps_the_cacheable_schema_hook():
    source = inspect.getsource(consolidator._consolidate_batch_with_llm)
    assert source.count("# fork: consolidation-cache") == 3


@pytest.mark.parametrize(
    ("provider", "strict", "expected"),
    [
        ("openai-codex", False, True),
        ("openai-codex", True, False),
        ("anthropic", False, True),
        ("anthropic", True, False),
        ("xai-oauth", False, True),
        ("openai-responses", False, True),
        ("openai", False, True),
        ("openai", True, False),
        ("llamacpp", False, True),
        ("ollama", False, False),
        ("gemini", False, False),
        ("vertexai", False, False),
        ("bedrock", False, False),
        ("litellm", False, False),
        ("github-copilot", False, False),
        ("claude-code", False, False),
        ("cursor", False, False),
    ],
)
def test_schema_is_prompt_text(provider: str, strict: bool, expected: bool):
    assert schema_is_prompt_text(SimpleNamespace(provider=provider), _config(strict=strict)) is expected


def test_a_config_without_a_provider_name_keeps_the_constrained_schema():
    assert schema_is_prompt_text(object(), _config()) is False
    assert schema_is_prompt_text(AsyncMock(), _config()) is False


def _openai(model: str) -> LLMProvider:
    return LLMProvider(provider="openai", api_key="sk-test", base_url=None, model=model)


def test_a_configured_single_provider_is_seen_through_the_wrapper():
    assert schema_is_prompt_text(ConfiguredLLMProvider(_openai("solo"), None, None), _config()) is True


def test_a_multi_llm_chain_keeps_the_constrained_schema():
    # MultiLLMProvider forwards ``provider`` to its first member; a failover member may enforce the schema.
    chain = MultiLLMProvider([_openai("primary"), _openai("fallback")], LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))
    configured = ConfiguredLLMProvider(chain, None, None)
    assert configured.provider == "openai"
    assert schema_is_prompt_text(configured, _config()) is False


@pytest.mark.asyncio
async def test_prompt_text_provider_gets_one_schema_for_every_batch():
    first, second = await _schemas_for_two_batches(_Recorder("openai-codex"))
    assert first == second
    rendered = json.dumps(first)
    assert '"enum"' not in rendered and '"maxItems"' not in rendered


@pytest.mark.asyncio
async def test_only_the_per_batch_keywords_leave_upstreams_schema():
    upstream, cacheable = _Recorder(None), _Recorder("openai-codex")
    await _run_two_batches(upstream)
    await _run_two_batches(cacheable)
    expected = upstream.response_formats[0].model_json_schema()
    expected["properties"]["creates"].pop("maxItems")
    for name in ("_Create", "_Update"):
        ids = expected["$defs"][name]["properties"]["source_fact_ids"]
        ids["items"].pop("enum")
        assert ids["minItems"] == 1  # batch-invariant, so it stays
    assert cacheable.response_formats[0].model_json_schema() == expected


# A delete plus its replacement create, sent when the scope may have no new observation.
_DELETE_AND_REPLACE = json.dumps(
    {
        "creates": [{"text": "merged", "source_fact_ids": [_BATCH_A[0]]}],
        "updates": [],
        "deletes": [{"observation_id": "44444444-4444-4444-8444-444444444444"}],
    }
)


async def _single_batch(provider: str | None, *, slots: int) -> Any:
    return await _consolidate_batch_with_llm(
        llm_config=_Recorder(provider, reply=_DELETE_AND_REPLACE),
        memories=_batch(*_BATCH_A),
        union_observations=[],
        union_source_facts={},
        config=_config(),
        remaining_observation_slots=slots,
        max_observations_per_scope=10,
    )


@pytest.mark.parametrize("provider", ["openai-codex", None])
@pytest.mark.asyncio
async def test_an_over_cap_response_still_fails_the_batch(provider: str | None):
    # Upstream's max_length on creates rejects the whole response, so the delete is never
    # applied without its replacement. The batch-invariant schema keeps that validation.
    result = await _single_batch(provider, slots=0)
    assert result.failed
    assert result.creates == [] and result.deletes == []


@pytest.mark.asyncio
async def test_a_response_within_the_cap_is_applied():
    result = await _single_batch("openai-codex", slots=1)
    assert not result.failed
    assert len(result.creates) == 1 and len(result.deletes) == 1


@pytest.mark.parametrize(
    ("provider", "strict"), [("openai-codex", True), ("ollama", False), ("gemini", False), (None, False)]
)
@pytest.mark.asyncio
async def test_enforcing_provider_keeps_the_per_batch_constraints(provider: str | None, strict: bool):
    first, second = await _schemas_for_two_batches(_Recorder(provider), strict=strict)
    assert first != second
    assert '"enum"' in json.dumps(first)


def _build_codex() -> CodexLLM:
    with patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


class _CodexLane:
    """Forwards the consolidation call to a real CodexLLM, as the configured lane would."""

    provider = "openai-codex"

    def __init__(self, codex: CodexLLM) -> None:
        self._codex = codex

    async def call(self, *, messages: Any, response_format: Any, strict_schema: bool, **_: Any) -> LLMCallResult:
        return await self._codex.call(
            messages=messages, response_format=response_format, strict_schema=strict_schema, max_retries=0
        )


@pytest.mark.asyncio
async def test_codex_instructions_are_byte_identical_across_batches():
    codex = _build_codex()
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    with stub_codex_stream(codex, response) as mock_stream:
        with patch.object(codex, "_parse_sse_stream", new_callable=AsyncMock, return_value=_EMPTY_BATCH):
            await _run_two_batches(_CodexLane(codex))
    first, second = (c.kwargs["json"]["instructions"] for c in mock_stream.call_args_list)
    assert "You must respond with valid JSON matching this schema" in first
    assert first == second
    assert not any(fact_id in first for fact_id in _BATCH_A + _BATCH_B)

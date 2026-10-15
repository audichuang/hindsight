"""Anthropic usage folds cache reads and writes into input_tokens on every path.

Anthropic reports ``input_tokens`` as the uncached remainder; cache writes and
cache reads arrive as separate fields. The TokenUsage contract is that
``input_tokens`` is the total prompt and ``cached_tokens`` is a subset of it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import CacheCreation, Usage
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.llm_trace import current_response_usage

pytestmark = pytest.mark.asyncio

# Real SDK models, so a renamed field in the SDK fails these tests instead of passing on a stand-in.
USAGE = Usage(
    input_tokens=12,
    cache_creation_input_tokens=3000,
    cache_read_input_tokens=9000,
    output_tokens=250,
    cache_creation=CacheCreation(ephemeral_5m_input_tokens=3000, ephemeral_1h_input_tokens=0),
)
EXPECTED_INPUT = 12012
EXPECTED_CACHED = 9000


def _make_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-5",
        )
    provider._client = MagicMock()
    return provider


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(text: str, usage=USAGE):
    resp = MagicMock()
    resp.content = [_text_block(text)]
    resp.usage = usage
    resp.stop_reason = "end_turn"
    return resp


class _Answer(BaseModel):
    answer: str


class _SpanSpy:
    def __init__(self):
        self.calls: list[dict] = []

    def record_llm_call(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def span_spy():
    spy = _SpanSpy()
    tracing.register_span_recorder(spy)
    yield spy
    tracing.unregister_span_recorder(spy)


async def test_call_reports_total_input_and_cached(span_spy):
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_response('{"answer": "ok"}'))
    metrics = MagicMock()

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector", return_value=metrics):
        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            scope="memory",
        )

    assert result.usage.input_tokens == EXPECTED_INPUT
    assert result.usage.cached_tokens == EXPECTED_CACHED
    assert result.usage.output_tokens == 250
    assert metrics.record_llm_call.call_args.kwargs["cached_input_tokens"] == EXPECTED_CACHED
    assert metrics.record_llm_call.call_args.kwargs["input_tokens"] == EXPECTED_INPUT
    assert span_spy.calls[-1]["input_tokens"] == EXPECTED_INPUT
    assert span_spy.calls[-1]["cached_tokens"] == EXPECTED_CACHED


async def test_call_with_tools_reports_total_input_and_cached(span_spy):
    provider = _make_provider()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_1"
    tool_block.name = "structured_response"
    tool_block.input = {"answer": "ok"}
    resp = MagicMock()
    resp.content = [tool_block]
    resp.usage = USAGE
    resp.stop_reason = "tool_use"
    provider._client.messages.create = AsyncMock(return_value=resp)
    metrics = MagicMock()

    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector", return_value=metrics):
        result = await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "structured_response", "parameters": {"type": "object"}},
                }
            ],
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            scope="reflect",
        )

    assert result.input_tokens == EXPECTED_INPUT
    assert result.cached_tokens == EXPECTED_CACHED
    assert result.output_tokens == 250
    assert metrics.record_llm_call.call_args.kwargs["cached_input_tokens"] == EXPECTED_CACHED
    assert span_spy.calls[-1]["input_tokens"] == EXPECTED_INPUT
    assert span_spy.calls[-1]["cached_tokens"] == EXPECTED_CACHED


async def test_batch_translator_reports_total_prompt():
    provider = _make_provider()
    message = _response("{}")

    body = provider._translate_batch_message(message)

    assert body["usage"] == {
        "prompt_tokens": EXPECTED_INPUT,
        "completion_tokens": 250,
        "total_tokens": EXPECTED_INPUT + 250,
        "prompt_tokens_details": {"cached_tokens": EXPECTED_CACHED},
    }


async def test_uncached_response_without_cache_fields_keeps_input():
    provider = _make_provider()
    usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    usage.cache_creation_input_tokens = None
    provider._client.messages.create = AsyncMock(return_value=_response('{"answer": "ok"}', usage=usage))

    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Answer,
        scope="memory",
    )

    assert result.usage.input_tokens == 5
    assert result.usage.cached_tokens == 0


async def test_parse_failure_still_stashes_total_usage():
    provider = _make_provider()
    provider._client.messages.create = AsyncMock(return_value=_response("not json at all"))

    with pytest.raises(Exception):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Answer,
            scope="memory",
            max_retries=0,
        )

    usage = current_response_usage()
    assert usage is not None
    assert usage.input_tokens == EXPECTED_INPUT
    assert usage.cached_tokens == EXPECTED_CACHED

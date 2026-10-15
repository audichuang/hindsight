"""Codex reports real token usage from the ``response.completed`` event.

Before this fix the Codex provider traced zeros (tools) or a chars/4 estimate (``call``)
even though the backend sends a usage block on the terminal event. Reasoning tokens are
folded into ``output_tokens`` by the Responses API, so visible output is the remainder.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.llm_trace import current_response_usage
from hindsight_api.engine.providers.codex_llm import _STRUCTURED_TOOL_NAME, CodexLLM, CodexStreamError
from tests.codex_stream_stub import CodexReply, stub_codex_stream, stub_codex_stream_with

pytestmark = pytest.mark.asyncio

USAGE = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1000},
    "output_tokens": 90,
    "output_tokens_details": {"reasoning_tokens": 60},
    "total_tokens": 1290,
}
# Visible output is 90 - 60 = 30; the reasoning tokens are reported as thoughts.
EXPECTED_OUTPUT = 30
EXPECTED_CACHED = 1000
EXPECTED_THOUGHTS = 60
EXPECTED_INPUT = 1200

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall memories",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]
_MESSAGES = [{"role": "user", "content": "extract the facts from this text"}]


class _Facts(BaseModel):
    facts: list[str]


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _text_stream(text: str, usage: dict | None = USAGE) -> str:
    events: list[tuple[str, dict]] = [("response.text.delta", {"delta": text})]
    if usage is not None:
        events.append(("response.completed", {"type": "response.completed", "response": {"usage": usage}}))
    return _sse(*events)


def _tool_stream(name: str, arguments: dict, usage: dict | None = USAGE) -> str:
    events: list[tuple[str, dict]] = [
        (
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "status": "completed",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "call_id": "call_1",
                }
            },
        )
    ]
    if usage is not None:
        events.append(("response.completed", {"type": "response.completed", "response": {"usage": usage}}))
    return _sse(*events)


def _build_llm() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


class _SpanSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_llm_call(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.fixture
def span_spy():
    spy = _SpanSpy()
    tracing.register_span_recorder(spy)
    yield spy
    tracing.unregister_span_recorder(spy)


async def test_call_reports_real_usage_from_response_completed():
    llm = _build_llm()
    metrics = MagicMock()
    with (
        stub_codex_stream(llm, CodexReply(text=_text_stream('{"facts": ["a"]}'))),
        patch("hindsight_api.engine.providers.codex_llm.get_metrics_collector", return_value=metrics),
    ):
        result = await llm.call(messages=_MESSAGES, response_format=_Facts)

    usage = result.usage
    assert usage.input_tokens == EXPECTED_INPUT
    assert usage.output_tokens == EXPECTED_OUTPUT
    assert usage.cached_tokens == EXPECTED_CACHED
    assert usage.thoughts_tokens == EXPECTED_THOUGHTS
    assert usage.total_tokens == EXPECTED_INPUT + EXPECTED_OUTPUT

    kwargs = metrics.record_llm_call.call_args.kwargs
    assert kwargs["input_tokens"] == EXPECTED_INPUT
    assert kwargs["output_tokens"] == EXPECTED_OUTPUT
    assert kwargs["cached_input_tokens"] == EXPECTED_CACHED
    assert kwargs["thoughts_tokens"] == EXPECTED_THOUGHTS


async def test_call_with_tools_reports_real_usage_from_response_completed():
    llm = _build_llm()
    metrics = MagicMock()
    with (
        stub_codex_stream(llm, CodexReply(text=_tool_stream("recall", {"query": "x"}))),
        patch("hindsight_api.engine.providers.codex_llm.get_metrics_collector", return_value=metrics),
    ):
        result = await llm.call_with_tools(messages=_MESSAGES, tools=_TOOLS, max_retries=0)

    assert [tc.name for tc in result.tool_calls] == ["recall"]
    assert result.input_tokens == EXPECTED_INPUT
    assert result.output_tokens == EXPECTED_OUTPUT
    assert result.cached_tokens == EXPECTED_CACHED
    assert result.thoughts_tokens == EXPECTED_THOUGHTS

    kwargs = metrics.record_llm_call.call_args.kwargs
    assert kwargs["input_tokens"] == EXPECTED_INPUT
    assert kwargs["cached_input_tokens"] == EXPECTED_CACHED


async def test_call_without_response_completed_keeps_the_char_estimate():
    llm = _build_llm()
    content = '{"facts": ["a"]}'
    with stub_codex_stream(llm, CodexReply(text=_text_stream(content, usage=None))):
        result = await llm.call(messages=_MESSAGES, response_format=_Facts)

    assert result.usage.input_tokens == sum(len(m["content"]) for m in _MESSAGES) // 4
    assert result.usage.output_tokens == len(content) // 4
    assert result.usage.cached_tokens == 0
    assert result.usage.thoughts_tokens == 0


async def test_call_with_tools_without_response_completed_reports_zeros():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_tool_stream("recall", {"query": "x"}, usage=None))):
        result = await llm.call_with_tools(messages=_MESSAGES, tools=_TOOLS, max_retries=0)

    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cached_tokens == 0
    assert result.thoughts_tokens == 0


async def test_forced_structured_tool_reports_real_usage():
    llm = _build_llm()
    stream = _tool_stream(_STRUCTURED_TOOL_NAME, {"facts": ["a"]})
    with stub_codex_stream(llm, CodexReply(text=stream)):
        result = await llm.call(messages=_MESSAGES, response_format=_Facts, strict_schema=True)

    assert result.content == _Facts(facts=["a"])
    assert result.usage.input_tokens == EXPECTED_INPUT
    assert result.usage.output_tokens == EXPECTED_OUTPUT
    assert result.usage.cached_tokens == EXPECTED_CACHED
    assert result.usage.thoughts_tokens == EXPECTED_THOUGHTS


async def test_span_recorder_sees_the_same_numbers(span_spy):
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_text_stream('{"facts": ["a"]}'))):
        await llm.call(messages=_MESSAGES, response_format=_Facts)
    with stub_codex_stream(llm, CodexReply(text=_tool_stream("recall", {"query": "x"}))):
        await llm.call_with_tools(messages=_MESSAGES, tools=_TOOLS, max_retries=0)

    assert len(span_spy.calls) == 2
    for call in span_spy.calls:
        assert call["input_tokens"] == EXPECTED_INPUT
        assert call["output_tokens"] == EXPECTED_OUTPUT
        assert call["cached_tokens"] == EXPECTED_CACHED
        assert call["thoughts_tokens"] == EXPECTED_THOUGHTS


async def test_response_failed_with_usage_is_stashed_before_raising():
    llm = _build_llm()
    failed = _sse(
        (
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "usage": USAGE,
                    "error": {"code": "server_error", "message": "boom"},
                },
            },
        )
    )
    with stub_codex_stream(llm, CodexReply(text=failed)):
        with pytest.raises(CodexStreamError, match="server_error"):
            await llm.call(messages=_MESSAGES, response_format=_Facts, max_retries=0)

    usage = current_response_usage()
    assert usage is not None
    assert usage.input_tokens == EXPECTED_INPUT
    assert usage.output_tokens == EXPECTED_OUTPUT
    assert usage.cached_tokens == EXPECTED_CACHED


async def test_failed_retry_keeps_the_previous_attempt_usage():
    llm = _build_llm()
    calls = {"n": 0}

    def handler(_url, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Attempt 1: real usage, but the JSON does not parse, so the call retries.
            return CodexReply(text=_text_stream("not json"))
        # Attempt 2: the transport fails before any usage event arrives.
        return CodexReply(status_code=500, text="upstream exploded")

    with stub_codex_stream_with(llm, handler):
        with pytest.raises(UpstreamHTTPError):
            await llm.call(
                messages=_MESSAGES,
                response_format=_Facts,
                max_retries=1,
                initial_backoff=0,
                max_backoff=0,
            )

    usage = current_response_usage()
    assert usage is not None
    assert usage.input_tokens == EXPECTED_INPUT
    assert usage.output_tokens == EXPECTED_OUTPUT
    assert usage.cached_tokens == EXPECTED_CACHED


# --- usage parsing at the event boundary ----------------------------------------------------------


def test_usage_parser_ignores_extra_fields_and_reads_missing_as_zero():
    from hindsight_api.engine.providers.codex_llm import _usage_from_codex_completed

    event = {"type": "response.completed", "response": {"usage": {"input_tokens": 7, "unknown": [1]}, "id": "r"}}
    usage = _usage_from_codex_completed(event)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens) == (7, 0, 0, 0)


def test_usage_parser_treats_wrong_typed_counts_as_absent():
    from hindsight_api.engine.providers.codex_llm import _usage_from_codex_completed

    event = {
        "response": {
            "usage": {
                "input_tokens": "1200",
                "output_tokens": True,
                "input_tokens_details": "nope",
                "output_tokens_details": {"reasoning_tokens": None},
            }
        }
    }
    usage = _usage_from_codex_completed(event)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "event",
    [
        "not an object",
        {"response": None},
        {"response": "text"},
        {"response": {"usage": None}},
        {"response": {"usage": ["list"]}},
        {"type": "response.completed"},
    ],
)
def test_usage_parser_returns_none_without_a_usage_object(event):
    from hindsight_api.engine.providers.codex_llm import _usage_from_codex_completed

    assert _usage_from_codex_completed(event) is None


def test_usage_parser_clamps_visible_output_at_zero():
    from hindsight_api.engine.providers.codex_llm import _usage_from_codex_completed

    usage = _usage_from_codex_completed(
        {"response": {"usage": {"output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 9}}}}
    )
    assert usage is not None
    assert (usage.output_tokens, usage.thoughts_tokens) == (0, 9)


# --- cache write tokens (input_tokens_details.cache_write_tokens) -----------------------------------


def test_usage_model_reads_cache_write_tokens():
    from hindsight_api.engine.providers.codex_llm import _CodexUsage

    usage = _CodexUsage.model_validate(
        {"input_tokens": 30000, "input_tokens_details": {"cached_tokens": 10000, "cache_write_tokens": 20000}}
    )
    assert usage.input_tokens == 30000
    assert usage.input_tokens_details.cached_tokens == 10000
    assert usage.input_tokens_details.cache_write_tokens == 20000


def test_usage_parser_keeps_input_total_and_cached_with_cache_writes():
    from hindsight_api.engine.providers.codex_llm import _usage_from_codex_completed

    event = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 30000,
                "input_tokens_details": {"cached_tokens": 10000, "cache_write_tokens": 20000},
                "output_tokens": 100,
            }
        },
    }
    usage = _usage_from_codex_completed(event)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens) == (
        30000,
        100,
        10000,
        0,
    )


@pytest.mark.parametrize("bad", ["20000", None, True, False, [20000]])
def test_cache_write_tokens_non_int_reads_zero(bad):
    from hindsight_api.engine.providers.codex_llm import _CodexUsage

    usage = _CodexUsage.model_validate({"input_tokens_details": {"cached_tokens": 10000, "cache_write_tokens": bad}})
    assert usage.input_tokens_details.cache_write_tokens == 0
    assert usage.input_tokens_details.cached_tokens == 10000

"""The Codex provider sends the ``session-id`` header the ChatGPT backend routes its prompt cache on.

The official Codex CLI notes that "ChatGPT derives cache affinity from the Responses
session-id header" (codex-rs/core/src/client.rs) and sends it on every request, equal
to its ``prompt_cache_key`` for a root session. Hindsight sent none. The header now
carries the request's effective ``prompt_cache_key``, so the two never disagree,
including when an operator pins the key through ``extra_body``.

All deterministic: the upstream is a local stub server (``codex_stream_stub``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_trace import LLMTraceContext, reset_trace_context, set_trace_context
from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import CodexReply, stub_codex_stream, stub_codex_stream_with

_MESSAGES = [
    {"role": "system", "content": "You are a memory agent."},
    {"role": "user", "content": "What do you know?"},
]


def _build_llm(extra_body: dict[str, Any] | None = None) -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            extra_body=extra_body,
        )


@contextmanager
def _bound_trace() -> Iterator[None]:
    token = set_trace_context(
        LLMTraceContext(bank_id="bank-1", operation="reflect", trace_id="11111111-1111-1111-1111-111111111111")
    )
    try:
        yield
    finally:
        reset_trace_context(token)


async def _send_call(llm: CodexLLM) -> MagicMock:
    with (
        stub_codex_stream(llm, MagicMock(status_code=200)) as stream,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(messages=_MESSAGES, max_retries=0)
    return stream


async def _send_tools(llm: CodexLLM) -> MagicMock:
    with (
        stub_codex_stream(llm, MagicMock(status_code=200)) as stream,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await llm.call_with_tools(messages=_MESSAGES, tools=[], max_retries=0)
    return stream


_PATHS = pytest.mark.parametrize("send", [_send_call, _send_tools], ids=["call", "call_with_tools"])


@_PATHS
async def test_session_header_equals_the_prompt_cache_key(send):
    llm = _build_llm()
    with _bound_trace():
        stream = await send(llm)

    request = stream.call_args.kwargs
    assert request["headers"]["session-id"] == request["json"]["prompt_cache_key"]


@_PATHS
async def test_session_header_follows_an_operator_prompt_cache_key(send):
    llm = _build_llm({"prompt_cache_key": "operator-key"})
    with _bound_trace():
        stream = await send(llm)

    assert stream.call_args.kwargs["headers"]["session-id"] == "operator-key"


@_PATHS
async def test_no_session_header_without_a_key(send):
    llm = _build_llm()
    with patch("hindsight_api.engine.providers.codex_llm.cache_affinity_id", return_value=None):
        stream = await send(llm)

    assert "session-id" not in stream.call_args.kwargs["headers"]


def test_header_builder_adds_no_session_by_default():
    """The quota fetch builds headers with no request behind them; it must not carry a session."""
    assert "session-id" not in _build_llm()._build_request_headers()


async def test_session_header_survives_the_auth_refresh_retry():
    llm = _build_llm()
    replies = [CodexReply(401, "unauthorized"), CodexReply()]
    with (
        _bound_trace(),
        patch.object(llm, "_refresh_oauth_tokens", new_callable=AsyncMock),
        stub_codex_stream_with(llm, lambda url, **kwargs: replies.pop(0)) as stream,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await llm.call_with_tools(messages=_MESSAGES, tools=[], max_retries=0)

    assert stream.call_count == 2
    first, second = (c.kwargs["headers"]["session-id"] for c in stream.call_args_list)
    assert first == second == stream.call_args.kwargs["json"]["prompt_cache_key"]

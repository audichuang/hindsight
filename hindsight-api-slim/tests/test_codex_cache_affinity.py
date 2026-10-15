"""The Codex provider sends a stable ``prompt_cache_key`` so its prompt cache can hit.

It sent a fresh ``uuid4`` on every request, which put each call of one agent loop
on a different cache key: a production reflect refresh had 3.4% of 10.8M input
tokens cached although consecutive turns were exact prefix extensions. The key now
comes from ``cache_affinity_id`` -- the operation's trace id, else the first
message -- the derivation the OpenAI-compatible providers already use.

All deterministic: the upstream is a local stub server (``codex_stream_stub``).
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.cache_affinity import cache_affinity_id
from hindsight_api.engine.llm_trace import LLMTraceContext, reset_trace_context, set_trace_context
from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import stub_codex_stream

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
_TRACE_A = "11111111-1111-1111-1111-111111111111"
_TRACE_B = "22222222-2222-2222-2222-222222222222"
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
def _bound_trace(trace_id: str) -> Iterator[None]:
    """Bind an operation trace context, as ``ConfiguredLLMProvider`` does per call."""
    token = set_trace_context(LLMTraceContext(bank_id="bank-1", operation="reflect", trace_id=trace_id))
    try:
        yield
    finally:
        reset_trace_context(token)


def _ok() -> MagicMock:
    return MagicMock(status_code=200)


async def _call_payload(llm: CodexLLM, messages: list[dict[str, Any]]) -> dict[str, Any]:
    with (
        stub_codex_stream(llm, _ok()) as stream,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(messages=messages, max_retries=0)
    return stream.call_args.kwargs["json"]


async def _tools_payload(llm: CodexLLM, messages: list[dict[str, Any]]) -> dict[str, Any]:
    with (
        stub_codex_stream(llm, _ok()) as stream,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await llm.call_with_tools(messages=messages, tools=[], max_retries=0)
    return stream.call_args.kwargs["json"]


_PATHS = pytest.mark.parametrize("send", [_call_payload, _tools_payload], ids=["call", "call_with_tools"])


@_PATHS
async def test_key_is_the_operation_affinity_id(send):
    llm = _build_llm()
    with _bound_trace(_TRACE_A):
        payload = await send(llm, _MESSAGES)
        expected = cache_affinity_id(_MESSAGES)

    assert payload["prompt_cache_key"] == expected
    assert _HEX32.match(payload["prompt_cache_key"])


@_PATHS
async def test_key_is_stable_within_one_operation(send):
    llm = _build_llm()
    with _bound_trace(_TRACE_A):
        first = await send(llm, _MESSAGES)
        second = await send(llm, [{"role": "system", "content": "A different prompt."}])

    assert first["prompt_cache_key"] == second["prompt_cache_key"]


@_PATHS
async def test_key_differs_across_operations(send):
    llm = _build_llm()
    with _bound_trace(_TRACE_A):
        first = await send(llm, _MESSAGES)
    with _bound_trace(_TRACE_B):
        second = await send(llm, _MESSAGES)

    assert first["prompt_cache_key"] != second["prompt_cache_key"]


@_PATHS
async def test_untraced_key_is_stable_as_the_conversation_grows(send):
    """Untraced, the key hashes the original first message (the system prompt), not
    the Responses ``input`` list, so it holds across the turns of one loop."""
    llm = _build_llm()
    grown = [*_MESSAGES, {"role": "assistant", "content": "Searching."}, {"role": "user", "content": "And?"}]

    first = await send(llm, _MESSAGES)
    second = await send(llm, grown)

    assert first["prompt_cache_key"] == second["prompt_cache_key"] == cache_affinity_id(_MESSAGES)


@_PATHS
async def test_no_key_when_no_id_can_be_derived(send):
    llm = _build_llm()
    with patch("hindsight_api.engine.providers.codex_llm.cache_affinity_id", return_value=None):
        payload = await send(llm, _MESSAGES)

    assert "prompt_cache_key" not in payload


@_PATHS
async def test_operator_extra_body_key_wins(send):
    llm = _build_llm({"prompt_cache_key": "operator-key"})
    with _bound_trace(_TRACE_A):
        payload = await send(llm, _MESSAGES)

    assert payload["prompt_cache_key"] == "operator-key"


async def test_the_factory_built_provider_sends_the_trace_key():
    """Through ``create_llm_provider``: the branch config actually reaches."""
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        llm = create_llm_provider(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            reasoning_effort="low",
        )
    with _bound_trace(_TRACE_A):
        payload = await _tools_payload(llm, _MESSAGES)
        expected = cache_affinity_id(_MESSAGES)

    assert payload["prompt_cache_key"] == expected


async def test_one_operation_shares_a_key_through_the_engine_provider():
    """The decisive shape: ``LLMProvider.with_config`` binds the trace context the
    key comes from. Two calls of one configured operation share a key; another
    operation gets another."""
    from types import SimpleNamespace

    from hindsight_api.engine.llm_wrapper import LLMProvider

    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        provider = LLMProvider(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
        )
    impl = provider._provider_impl
    config = SimpleNamespace(llm_gemini_safety_settings=None)
    first_run = provider.with_config(config, bank_id="bank-1", operation="reflect")
    second_run = provider.with_config(config, bank_id="bank-1", operation="reflect")

    with (
        stub_codex_stream(impl, _ok()) as stream,
        patch.object(impl, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await first_run.call_with_tools(messages=_MESSAGES, tools=[], max_retries=0)
        await first_run.call_with_tools(
            messages=[*_MESSAGES, {"role": "user", "content": "more"}], tools=[], max_retries=0
        )
        await second_run.call_with_tools(messages=_MESSAGES, tools=[], max_retries=0)

    keys = [c.kwargs["json"]["prompt_cache_key"] for c in stream.call_args_list]
    assert keys[0] == keys[1] != keys[2]
    assert all(_HEX32.match(k) for k in keys)

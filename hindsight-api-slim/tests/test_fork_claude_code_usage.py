"""Real usage from the Claude Agent SDK ResultMessage / AssistantMessage (B4).

Before this, the claude-code provider reported a chars/4 estimate for every call,
even though the SDK reports the Anthropic usage dict on ResultMessage and on each
AssistantMessage. These tests cover the real-usage path, the estimate fallback,
and the tool-round break (#2966).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO
from hindsight_api.engine.llm_trace import current_response_usage

ANTHROPIC_USAGE = {
    "input_tokens": 10,
    "cache_creation_input_tokens": 200,
    "cache_read_input_tokens": 3000,
    "output_tokens": 50,
}


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing the SDK."""

    system_prompt: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    cwd: str | None = None
    setting_sources: list[str] | None = None


class _FakeAssistantMessage:
    def __init__(self, content: list[Any], usage: dict[str, Any] | None = None) -> None:
        self.content = content
        self.usage = usage  # the SDK's messages always carry the field, None when absent


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class _FakeResultMessage:
    def __init__(
        self,
        subtype: str,
        is_error: bool,
        result: str | None,
        usage: dict[str, Any] | None = None,
        total_cost_usd: float | None = None,
    ) -> None:
        self.subtype = subtype
        self.is_error = is_error
        self.result = result
        self.usage = usage  # the SDK's messages always carry the field, None when absent
        self.total_cost_usd = total_cost_usd


@dataclass
class _FakeSdkMcpTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Any


def _fake_create_sdk_mcp_server(name: str, version: str, tools=None):
    return {"name": name, "version": version, "tools": tools}


class _FakeMetrics:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _FakeSpanRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _install_fake_sdk(monkeypatch, client_cls=None, query_fn=None) -> None:
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", _FakeToolUseBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create_sdk_mcp_server)
    if client_cls is not None:
        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    if query_fn is not None:
        monkeypatch.setattr(claude_agent_sdk, "query", query_fn)


def _patch_observability(monkeypatch) -> tuple[_FakeMetrics, _FakeSpanRecorder]:
    import hindsight_api.engine.providers.claude_code_llm as module
    import hindsight_api.tracing as tracing

    metrics = _FakeMetrics()
    recorder = _FakeSpanRecorder()
    monkeypatch.setattr(module, "get_metrics_collector", lambda: metrics)
    monkeypatch.setattr(tracing, "get_span_recorder", lambda: recorder)
    return metrics, recorder


def _instantiate_provider():
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model="claude-haiku-4-5",
        reasoning_effort="low",
    )


_RECALL_TOOL = {
    "function": {
        "name": "recall",
        "description": "Search the memory bank.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
}

_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer briefly."}]


async def test_call_uses_result_message_usage(monkeypatch):
    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="Paris")])
        yield _FakeResultMessage(
            subtype="success", is_error=False, result="Paris", usage=ANTHROPIC_USAGE, total_cost_usd=0.01
        )

    _install_fake_sdk(monkeypatch, query_fn=fake_query)
    metrics, recorder = _patch_observability(monkeypatch)

    provider = _instantiate_provider()
    result = await provider.call(messages=_MESSAGES, max_retries=0, scope="test")

    assert result.content == "Paris"
    assert result.usage.input_tokens == 3210
    assert result.usage.output_tokens == 50
    assert result.usage.cached_tokens == 3000
    assert result.usage.total_tokens == 3260

    assert metrics.calls[0]["input_tokens"] == 3210
    assert metrics.calls[0]["output_tokens"] == 50
    assert metrics.calls[0]["cached_input_tokens"] == 3000

    assert recorder.calls[0]["input_tokens"] == 3210
    assert recorder.calls[0]["output_tokens"] == 50
    assert recorder.calls[0]["cached_tokens"] == 3000

    stashed = current_response_usage()
    assert stashed is not None
    assert (stashed.input_tokens, stashed.output_tokens, stashed.cached_tokens) == (3210, 50, 3000)


@pytest.mark.asyncio
async def test_call_result_usage_wins_over_assistant_usage(monkeypatch):
    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(
            content=[_FakeTextBlock(text="Paris")],
            usage={"input_tokens": 1, "output_tokens": 1},
        )
        yield _FakeResultMessage(subtype="success", is_error=False, result="Paris", usage=ANTHROPIC_USAGE)

    _install_fake_sdk(monkeypatch, query_fn=fake_query)
    _patch_observability(monkeypatch)

    result = await _instantiate_provider().call(messages=_MESSAGES, max_retries=0, scope="test")

    assert result.usage.input_tokens == 3210
    assert result.usage.output_tokens == 50


@pytest.mark.asyncio
async def test_call_without_usage_keeps_chars_per_four_estimate(monkeypatch):
    text = "Paris is the capital of France."

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text=text)])
        yield _FakeResultMessage(subtype="success", is_error=False, result=text)

    _install_fake_sdk(monkeypatch, query_fn=fake_query)
    metrics, recorder = _patch_observability(monkeypatch)

    result = await _instantiate_provider().call(messages=_MESSAGES, max_retries=0, scope="test")

    expected_input = len(_MESSAGES[0]["content"]) // 4
    expected_output = len(text) // 4
    assert result.usage.input_tokens == expected_input
    assert result.usage.output_tokens == expected_output
    assert result.usage.cached_tokens == 0
    assert result.usage.total_tokens == expected_input + expected_output
    assert metrics.calls[0]["input_tokens"] == expected_input
    assert metrics.calls[0]["cached_input_tokens"] == 0
    assert recorder.calls[0]["input_tokens"] == expected_input
    assert recorder.calls[0]["output_tokens"] == expected_output
    assert recorder.calls[0]["cached_tokens"] == 0

    stashed = current_response_usage()
    assert stashed is not None
    assert (stashed.input_tokens, stashed.output_tokens) == (expected_input, expected_output)


@pytest.mark.asyncio
async def test_call_with_tools_uses_result_message_usage(monkeypatch):
    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="No tools needed.")])
            yield _FakeResultMessage(subtype="success", is_error=False, result="ok", usage=ANTHROPIC_USAGE)

    _install_fake_sdk(monkeypatch, client_cls=_FakeClient)
    metrics, recorder = _patch_observability(monkeypatch)

    result = await _instantiate_provider().call_with_tools(
        messages=_MESSAGES,
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert result.input_tokens == 3210
    assert result.output_tokens == 50
    assert result.cached_tokens == 3000
    assert recorder.calls[0]["cached_tokens"] == 3000
    assert metrics.calls[0]["cached_input_tokens"] == 3000


@pytest.mark.asyncio
async def test_tool_round_break_uses_assistant_usage(monkeypatch):
    """The loop breaks on the tool call (#2966), so ResultMessage is never read.

    The AssistantMessage usage is the only real count and must be used.
    """
    assistant_usage = {
        "input_tokens": 5,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 400,
        "output_tokens": 9,
    }

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(
                content=[
                    _FakeToolUseBlock(id="tu_1", name="mcp__hindsight_tools__recall", input={"query": "x"}),
                ],
                usage=assistant_usage,
            )
            # Never reached: the provider stops consuming after the tool call.
            yield _FakeResultMessage(
                subtype="error_max_turns",
                is_error=True,
                result=None,
                usage={"input_tokens": 999, "output_tokens": 999},
            )

    _install_fake_sdk(monkeypatch, client_cls=_FakeClient)
    metrics, recorder = _patch_observability(monkeypatch)

    result = await _instantiate_provider().call_with_tools(
        messages=_MESSAGES,
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert result.finish_reason == "tool_calls"
    assert result.input_tokens == 425
    assert result.output_tokens == 9
    assert result.cached_tokens == 400
    assert recorder.calls[0]["input_tokens"] == 425
    assert recorder.calls[0]["cached_tokens"] == 400
    assert metrics.calls[0]["input_tokens"] == 425
    assert metrics.calls[0]["cached_input_tokens"] == 400


@pytest.mark.asyncio
async def test_tool_round_without_usage_keeps_chars_per_four_estimate(monkeypatch):
    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="plain answer text")])
            yield _FakeResultMessage(subtype="success", is_error=False, result="ok")

    _install_fake_sdk(monkeypatch, client_cls=_FakeClient)
    _patch_observability(monkeypatch)

    result = await _instantiate_provider().call_with_tools(
        messages=_MESSAGES,
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert result.input_tokens == len(_MESSAGES[0]["content"]) // 4
    assert result.output_tokens == len("plain answer text") // 4
    assert result.cached_tokens == 0

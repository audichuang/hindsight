"""Fork hooks that carry per-call usage markers into the llm_requests trace (cost-accounting).

Upstream files get only small marked hooks. ``llm_trace.LLMTraceRecorder.record_llm_call`` (H2)
writes ``llm_info.usage`` on every row once the feature is on. Each provider adapter passes its
``fork_cost_usage`` kwarg on its success span (H3-H5 for Anthropic, H14-H21 for the others).

The source-text tests fail if a hook disappears in a rebase. The end-to-end tests drive each
provider's real call path with a fake SDK or HTTP reply, into a real ``LLMTraceRecorder``, and
check the stored ``llm_info``. The feature-off tests check that each provider passes upstream's
exact kwargs and that no row carries a ``usage`` key.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import CacheCreation, Usage

from hindsight_api import tracing
from hindsight_api.config import clear_config_cache
from hindsight_api.engine import llm_cost
from hindsight_api.engine.llm_cost import load_cost_settings
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.llm_trace import LLMRequestRecord, LLMTraceRecorder
from hindsight_api.engine.providers import claude_code_llm as claude_module
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM
from tests.codex_stream_stub import CodexReply, stub_codex_stream
from tests.test_fork_claude_code_usage import (
    _RECALL_TOOL,
    ANTHROPIC_USAGE,
    _FakeAssistantMessage,
    _FakeResultMessage,
    _FakeTextBlock,
    _FakeToolUseBlock,
    _install_fake_sdk,
    _instantiate_provider,
)
from tests.test_fork_codex_usage import _MESSAGES as CODEX_MESSAGES
from tests.test_fork_codex_usage import _TOOLS as CODEX_TOOLS
from tests.test_fork_codex_usage import USAGE as CODEX_USAGE
from tests.test_fork_codex_usage import _build_llm, _Facts, _text_stream, _tool_stream

ENGINE_DIR = Path(__file__).resolve().parent.parent / "hindsight_api" / "engine"
PROVIDERS_DIR = ENGINE_DIR / "providers"
TRACE_SRC = (ENGINE_DIR / "llm_trace.py").read_text(encoding="utf-8")
ANTHROPIC_SRC = (PROVIDERS_DIR / "anthropic_llm.py").read_text(encoding="utf-8")
PYPROJECT = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

MARKER = "# fork: cost-accounting"
H2_IMPORT = "from .llm_cost import trace_usage_info  # fork: cost-accounting"
H2_UPDATE = "llm_info.update(trace_usage_info(_extra))  # fork: cost-accounting"
H3_IMPORT = "from hindsight_api.engine.llm_cost import anthropic_cache_write_kwargs  # fork: cost-accounting"
H4_H5_KWARGS = "**anthropic_cache_write_kwargs(response),  # fork: cost-accounting"
H7_DEPENDENCY = '"genai-prices>=0.1.12,<0.2",  # fork: cost-accounting'

# Each file's hook lines, with the exact number of times each appears (design2 section 3).
HOOK_LINES: dict[str, list[tuple[str, int]]] = {
    "llm_trace.py": [(H2_IMPORT, 1), (H2_UPDATE, 1)],
    "providers/claude_code_llm.py": [
        ("from hindsight_api.engine.llm_cost import claude_code_usage_kwargs  # fork: cost-accounting", 2),
        ("**claude_code_usage_kwargs(sdk_usage),  # fork: cost-accounting", 2),
    ],
    "providers/codex_llm.py": [
        ("from hindsight_api.engine.llm_cost import note_codex_usage  # fork: cost-accounting", 1),
        ("note_codex_usage(result, data)  # fork: cost-accounting", 1),
        ("from hindsight_api.engine.llm_cost import codex_usage_kwargs  # fork: cost-accounting", 2),
        ("**codex_usage_kwargs(token_usage),  # fork: cost-accounting", 1),
        ("**codex_usage_kwargs(usage),  # fork: cost-accounting", 1),
    ],
    "providers/openai_compatible_llm.py": [
        ("from hindsight_api.engine.llm_cost import openai_compatible_usage_kwargs  # fork: cost-accounting", 2),
        ("**openai_compatible_usage_kwargs(self.provider, self.base_url, response),  # fork: cost-accounting", 2),
    ],
    "providers/openai_responses_llm.py": [
        ("response: Any = None,  # fork: cost-accounting", 1),
        ("from hindsight_api.engine.llm_cost import openai_responses_usage_kwargs  # fork: cost-accounting", 1),
        ("**openai_responses_usage_kwargs(response),  # fork: cost-accounting", 1),
        ("response=response,  # fork: cost-accounting", 2),
    ],
}
# Marker-line totals per file: the hook lines above, nothing else.
MARKER_LINE_TOTALS = {
    "llm_trace.py": 2,
    "providers/claude_code_llm.py": 4,
    "providers/codex_llm.py": 6,
    "providers/openai_compatible_llm.py": 4,
    "providers/openai_responses_llm.py": 5,
}

# Anthropic usage: input_tokens is the uncached remainder; cache writes and reads are separate.
# Real SDK models, so a renamed field in the SDK fails these tests instead of passing on a stand-in.
USAGE = Usage(
    input_tokens=12,
    cache_creation_input_tokens=3000,
    cache_read_input_tokens=9000,
    output_tokens=250,
    cache_creation=CacheCreation(ephemeral_5m_input_tokens=3000, ephemeral_1h_input_tokens=0),
)
EXPECTED_INPUT = 12012


def _hook_source(relative: str) -> str:
    return (ENGINE_DIR / relative).read_text(encoding="utf-8")


# ── source-text hook existence ────────────────────────────────────────────────


def test_trace_recorder_hook_is_present_once():
    assert TRACE_SRC.count(H2_IMPORT) == 1
    assert TRACE_SRC.count(H2_UPDATE) == 1


def test_trace_hook_is_unguarded_and_runs_after_tool_calls_and_before_record():
    tool_line = 'llm_info["tool_calls"] = [tc.get("name", "") for tc in tool_calls]'
    assert 'if "cache_write_tokens" in _extra' not in TRACE_SRC
    assert TRACE_SRC.index(tool_line) < TRACE_SRC.index(H2_IMPORT) < TRACE_SRC.index(H2_UPDATE)
    assert TRACE_SRC.index(H2_UPDATE) < TRACE_SRC.index("record = LLMRequestRecord(")


@pytest.mark.parametrize("relative", sorted(HOOK_LINES))
def test_each_hook_line_is_present_with_its_exact_count(relative):
    source = _hook_source(relative)
    for line, expected in HOOK_LINES[relative]:
        assert source.count(line) == expected, f"{relative}: {line!r} x{source.count(line)} != {expected}"


@pytest.mark.parametrize("relative", sorted(MARKER_LINE_TOTALS))
def test_marker_line_total_matches_the_table(relative):
    lines = [line for line in _hook_source(relative).splitlines() if MARKER in line]
    assert len(lines) == MARKER_LINE_TOTALS[relative]
    listed = sum(expected for _line, expected in HOOK_LINES[relative])
    assert len(lines) == listed


@pytest.mark.parametrize("relative", sorted(HOOK_LINES))
def test_fork_imports_are_local_not_module_level(relative):
    # Local, not module level: the fork module stays out of the provider's import graph (FORK.md).
    for line in _hook_source(relative).splitlines():
        if "llm_cost import" in line and MARKER in line:
            assert line.startswith(" "), line


def test_anthropic_hooks_are_unchanged():
    assert ANTHROPIC_SRC.count(H3_IMPORT) == 2
    assert ANTHROPIC_SRC.count(H4_H5_KWARGS) == 2


def test_genai_prices_dependency_hook_is_present():
    assert PYPROJECT.count(H7_DEPENDENCY) == 1


# ── end to end: provider call -> span kwargs -> LLMTraceRecorder row ────────────


class _SpanSpy:
    """Keeps the raw kwargs a provider passes to ``record_llm_call``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_llm_call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _CapturingRecorder(LLMTraceRecorder):
    """The real trace recorder, with the DB write replaced by an in-memory list (fake pool)."""

    def __init__(self) -> None:
        super().__init__(pool_getter=lambda: None, schema_getter=lambda: "public", enabled=True, allowed_scopes=[])
        self.records: list[LLMRequestRecord] = []

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        self.records.append(record)


@dataclass
class _Traced:
    spy: _SpanSpy
    recorder: _CapturingRecorder


@pytest.fixture(autouse=True)
def _clean_cost_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("HINDSIGHT_API_LLM_COST_"):
            monkeypatch.delenv(name)
    clear_config_cache()
    load_cost_settings.cache_clear()
    yield
    clear_config_cache()
    load_cost_settings.cache_clear()


@pytest.fixture
def cost_on(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    clear_config_cache()
    load_cost_settings.cache_clear()


@pytest.fixture
def traced():
    spy = _SpanSpy()
    recorder = _CapturingRecorder()
    tracing.register_span_recorder(spy)
    tracing.register_span_recorder(recorder)
    yield _Traced(spy=spy, recorder=recorder)
    tracing.unregister_span_recorder(spy)
    tracing.unregister_span_recorder(recorder)


def _patch_claude_metrics(monkeypatch) -> None:
    monkeypatch.setattr(claude_module, "get_metrics_collector", lambda: MagicMock())


_CLAUDE_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer briefly."}]


async def _anthropic_drive() -> None:
    provider = _make_anthropic()
    provider._client.messages.create = AsyncMock(return_value=_anthropic_tool_response())
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector", return_value=MagicMock()):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "structured_response", "parameters": {"type": "object"}}}],
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            scope="reflect",
        )


def _make_anthropic():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(provider="anthropic", api_key="fake-key", base_url="", model="claude-sonnet-5")
    provider._client = MagicMock()
    return provider


def _anthropic_tool_response():
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_1"
    tool_block.name = "structured_response"
    tool_block.input = {"answer": "ok"}
    resp = MagicMock()
    resp.content = [tool_block]
    resp.usage = USAGE
    resp.stop_reason = "tool_use"
    return resp


async def _claude_call_drive(monkeypatch, *, result_usage=None, total_cost=None) -> None:
    async def fake_query(prompt: str, options):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="Paris")])
        yield _FakeResultMessage(
            subtype="success",
            is_error=False,
            result="Paris",
            usage=result_usage,
            total_cost_usd=total_cost,
        )

    _install_fake_sdk(monkeypatch, query_fn=fake_query)
    _patch_claude_metrics(monkeypatch)
    await _instantiate_provider().call(messages=_CLAUDE_MESSAGES, max_retries=0, scope="test")


async def _claude_tools_drive(monkeypatch, *, assistant_usage=None) -> None:
    class _FakeClient:
        def __init__(self, options) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(
                content=[_FakeToolUseBlock(id="t1", name="mcp__hindsight_tools__recall", input={"query": "x"})],
                usage=assistant_usage,
            )

    _install_fake_sdk(monkeypatch, client_cls=_FakeClient)
    _patch_claude_metrics(monkeypatch)
    await _instantiate_provider().call_with_tools(
        messages=_CLAUDE_MESSAGES,
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )


async def _codex_call_drive(monkeypatch, *, usage=CODEX_USAGE) -> None:
    llm = _build_llm()
    with (
        stub_codex_stream(llm, CodexReply(text=_text_stream('{"facts": ["a"]}', usage=usage))),
        patch("hindsight_api.engine.providers.codex_llm.get_metrics_collector", return_value=MagicMock()),
    ):
        await llm.call(messages=CODEX_MESSAGES, response_format=_Facts)


async def _codex_tools_drive(monkeypatch, *, usage=CODEX_USAGE) -> None:
    llm = _build_llm()
    with (
        stub_codex_stream(llm, CodexReply(text=_tool_stream("recall", {"query": "x"}, usage=usage))),
        patch("hindsight_api.engine.providers.codex_llm.get_metrics_collector", return_value=MagicMock()),
    ):
        await llm.call_with_tools(messages=CODEX_MESSAGES, tools=CODEX_TOOLS, max_retries=0)


_COMPAT_MESSAGES = [{"role": "user", "content": "extract"}]
_COMPAT_TOOLS = [
    {
        "type": "function",
        "function": {"name": "recall", "description": "Recall", "parameters": {"type": "object"}},
    }
]
# OpenRouter-shaped usage: cache-write detail and a provider-reported credit cost.
_OPENROUTER_USAGE = SimpleNamespace(
    prompt_tokens=100,
    completion_tokens=5,
    total_tokens=105,
    cost=0.0123,
    is_byok=False,
    prompt_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=50),
    completion_tokens_details=None,
)
_PLAIN_USAGE = SimpleNamespace(
    prompt_tokens=100,
    completion_tokens=5,
    total_tokens=105,
    prompt_tokens_details=None,
    completion_tokens_details=None,
)


def _compat_response(usage, content: str = '{"ok": true}') -> SimpleNamespace:
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    return SimpleNamespace(error=None, usage=usage, choices=[choice])


def _compat_llm(provider: str, base_url: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(provider=provider, api_key="test-key", base_url=base_url, model="gpt-4o-mini")


async def _compat_call_drive(monkeypatch, *, provider="openrouter", base_url="https://openrouter.ai/api/v1", usage):
    llm = _compat_llm(provider, base_url)
    llm._client.chat.completions.create = AsyncMock(return_value=_compat_response(usage))
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector", return_value=MagicMock()):
        await llm.call(messages=_COMPAT_MESSAGES, scope="memory", max_retries=0)


async def _compat_tools_drive(monkeypatch, *, provider="openrouter", base_url="https://openrouter.ai/api/v1", usage):
    llm = _compat_llm(provider, base_url)
    llm._client.chat.completions.create = AsyncMock(return_value=_compat_response(usage))
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector", return_value=MagicMock()):
        await llm.call_with_tools(
            messages=_COMPAT_MESSAGES,
            tools=_COMPAT_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_AUTO,
            max_retries=0,
            scope="reflect",
        )


# Responses usage: 400 cache-write tokens, and xAI-style ticks (37756000 ticks = $0.0037756).
_RESPONSES_USAGE = SimpleNamespace(
    input_tokens=1000,
    output_tokens=50,
    input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=400),
    output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    cost_in_usd_ticks=37756000,
)


def _responses_reply(usage) -> SimpleNamespace:
    return SimpleNamespace(output_text='{"ok": true}', output=[], usage=usage, status="completed")


def _responses_llm() -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(
        provider="openai-responses",
        api_key="sk-test",
        base_url="",
        model="gpt-5.6",
        reasoning_effort="high",
    )


async def _responses_call_drive(monkeypatch, *, usage=_RESPONSES_USAGE) -> None:
    llm = _responses_llm()
    llm._client.responses.create = AsyncMock(return_value=_responses_reply(usage))
    with patch(
        "hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector",
        return_value=MagicMock(),
    ):
        await llm.call(messages=_COMPAT_MESSAGES, scope="memory", max_retries=0)


async def _responses_tools_drive(monkeypatch, *, usage=_RESPONSES_USAGE) -> None:
    llm = _responses_llm()
    llm._client.responses.create = AsyncMock(return_value=_responses_reply(usage))
    with patch(
        "hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector",
        return_value=MagicMock(),
    ):
        await llm.call_with_tools(
            messages=_COMPAT_MESSAGES,
            tools=_COMPAT_TOOLS,
            tool_choice=LLM_TOOL_CHOICE_AUTO,
            max_retries=0,
            scope="reflect",
        )


# Upstream's exact span kwargs, read off each provider's record_llm_call with the feature off.
_CLAUDE_CALL_KEYS = {
    "provider",
    "model",
    "scope",
    "messages",
    "response_content",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "duration",
    "finish_reason",
    "error",
}
_CLAUDE_TOOLS_KEYS = _CLAUDE_CALL_KEYS | {"tool_calls"}
_CODEX_CALL_KEYS = {
    "provider",
    "model",
    "scope",
    "messages",
    "response_content",
    "input_tokens",
    "output_tokens",
    "duration",
    "finish_reason",
    "error",
    "cached_tokens",
    "thoughts_tokens",
}
_CODEX_TOOLS_KEYS = _CODEX_CALL_KEYS | {"tool_calls"}
_COMPAT_CALL_KEYS = _CODEX_CALL_KEYS | {"served_by"}
_COMPAT_TOOLS_KEYS = _COMPAT_CALL_KEYS | {"tool_calls"}
_RESPONSES_KEYS = _CODEX_CALL_KEYS | {"tool_calls"}

_V2_ESTIMATE = {"v": 2, "usage_source": "estimate"}

# (id, driver, expected llm_info.usage with the feature on, upstream span kwarg keys with it off)
PROVIDER_CASES = [
    pytest.param(
        "claude-code-call",
        lambda m: _claude_call_drive(m, result_usage=ANTHROPIC_USAGE, total_cost=0.01),
        {
            "v": 2,
            "cache_write_tokens": 200,
            "cache_write_5m_tokens": 200,
            "cache_write_1h_tokens": 0,
            "usage_source": "provider",
        },
        _CLAUDE_CALL_KEYS,
        id="claude-code-call",
    ),
    pytest.param(
        "claude-code-call-estimate",
        lambda m: _claude_call_drive(m),
        _V2_ESTIMATE,
        _CLAUDE_CALL_KEYS,
        id="claude-code-call-estimate",
    ),
    pytest.param(
        "claude-code-tools",
        lambda m: _claude_tools_drive(m, assistant_usage=ANTHROPIC_USAGE),
        {
            "v": 2,
            "cache_write_tokens": 200,
            "cache_write_5m_tokens": 200,
            "cache_write_1h_tokens": 0,
            "usage_source": "provider",
        },
        _CLAUDE_TOOLS_KEYS,
        id="claude-code-tools",
    ),
    pytest.param(
        "codex-call",
        lambda m: _codex_call_drive(m),
        {"v": 2, "usage_source": "provider"},  # CODEX_USAGE sends no cache_write_tokens: no breakdown, not 0
        _CODEX_CALL_KEYS,
        id="codex-call",
    ),
    pytest.param(
        "codex-tools-cache-write",
        lambda m: _codex_tools_drive(
            m,
            usage={**CODEX_USAGE, "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 300}},
        ),
        {
            "v": 2,
            "cache_write_tokens": 300,
            "cache_write_5m_tokens": 300,
            "cache_write_1h_tokens": 0,
            "usage_source": "provider",
        },
        _CODEX_TOOLS_KEYS,
        id="codex-tools-cache-write",
    ),
    pytest.param(
        "codex-call-without-usage",
        lambda m: _codex_call_drive(m, usage=None),
        _V2_ESTIMATE,
        _CODEX_CALL_KEYS,
        id="codex-call-without-usage",
    ),
    pytest.param(
        "openai-compatible-call",
        lambda m: _compat_call_drive(m, usage=_OPENROUTER_USAGE),
        {
            "v": 2,
            "cache_write_tokens": 50,
            "cache_write_5m_tokens": 50,
            "cache_write_1h_tokens": 0,
            "provider_cost_usd": 0.0123,
            "usage_source": "provider",
        },
        _COMPAT_CALL_KEYS,
        id="openai-compatible-call",
    ),
    pytest.param(
        "openai-compatible-tools",
        lambda m: _compat_tools_drive(m, usage=_OPENROUTER_USAGE),
        {
            "v": 2,
            "cache_write_tokens": 50,
            "cache_write_5m_tokens": 50,
            "cache_write_1h_tokens": 0,
            "provider_cost_usd": 0.0123,
            "usage_source": "provider",
        },
        _COMPAT_TOOLS_KEYS,
        id="openai-compatible-tools",
    ),
    pytest.param(
        "zai-coding-plan-call",
        lambda m: _compat_call_drive(
            m,
            provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4",
            usage=_PLAIN_USAGE,
        ),
        # _PLAIN_USAGE sends no cache-write field: no breakdown, not zero writes
        {"v": 2, "billing_hint": "plan", "usage_source": "provider"},
        _COMPAT_CALL_KEYS,
        id="zai-coding-plan-call",
    ),
    pytest.param(
        "openai-responses-call",
        lambda m: _responses_call_drive(m),
        {
            "v": 2,
            "cache_write_tokens": 400,
            "cache_write_5m_tokens": 400,
            "cache_write_1h_tokens": 0,
            "provider_cost_usd": 0.0037756,
            "usage_source": "provider",
        },
        _RESPONSES_KEYS,
        id="openai-responses-call",
    ),
    pytest.param(
        "openai-responses-tools",
        lambda m: _responses_tools_drive(m),
        {
            "v": 2,
            "cache_write_tokens": 400,
            "cache_write_5m_tokens": 400,
            "cache_write_1h_tokens": 0,
            "provider_cost_usd": 0.0037756,
            "usage_source": "provider",
        },
        _RESPONSES_KEYS,
        id="openai-responses-tools",
    ),
]


def _assert_usage_matches(actual: dict, expected: dict) -> None:
    """Exact dict equality, except the xAI-style cost, which is a float from ticks."""
    assert set(actual) == set(expected), actual
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value), key
        else:
            assert actual[key] == value, key


@pytest.mark.asyncio
async def test_anthropic_feature_on_records_v2_marker_and_breakdown(cost_on, traced):
    await _anthropic_drive()

    assert len(traced.recorder.records) == 1
    record = traced.recorder.records[0]
    assert record.input_tokens == EXPECTED_INPUT
    assert record.cached_tokens == 9000
    _assert_usage_matches(
        record.llm_info["usage"],
        {
            "v": 2,
            "cache_write_tokens": 3000,
            "cache_write_5m_tokens": 3000,
            "cache_write_1h_tokens": 0,
            "usage_source": "provider",
        },
    )


@pytest.mark.asyncio
async def test_anthropic_feature_off_leaves_llm_info_as_upstream_writes_it(traced):
    await _anthropic_drive()

    assert len(traced.recorder.records) == 1
    record = traced.recorder.records[0]
    # Upstream builds llm_info from finish_reason and tool_calls only; no usage key.
    assert record.llm_info == {"finish_reason": "tool_calls", "tool_calls": ["structured_response"]}
    assert record.input_tokens == EXPECTED_INPUT
    assert all("fork_cost_usage" not in call for call in traced.spy.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "drive", "expected", "_keys"), PROVIDER_CASES)
async def test_feature_on_each_provider_writes_the_v2_usage_marker(
    monkeypatch, cost_on, traced, case_id, drive, expected, _keys
):
    await drive(monkeypatch)

    assert len(traced.recorder.records) == 1, case_id
    _assert_usage_matches(traced.recorder.records[0].llm_info["usage"], expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "drive", "_expected", "keys"), PROVIDER_CASES)
async def test_feature_off_each_provider_passes_upstream_kwargs_and_no_usage(
    monkeypatch, traced, case_id, drive, _expected, keys
):
    await drive(monkeypatch)

    assert len(traced.spy.calls) == 1, case_id
    assert set(traced.spy.calls[0]) == keys, case_id
    assert "fork_cost_usage" not in traced.spy.calls[0]
    assert len(traced.recorder.records) == 1, case_id
    assert "usage" not in traced.recorder.records[0].llm_info


@pytest.mark.asyncio
async def test_feature_on_row_without_adapter_kwarg_gets_the_bare_marker(cost_on, traced):
    # Rows with no adapter (the Ollama native path, a recovered call) carry only the v2 marker.
    tracing.get_span_recorder().record_llm_call(
        provider="ollama", model="m", scope="memory", messages=[], response_content="x", input_tokens=1
    )

    assert traced.recorder.records[0].llm_info["usage"] == {"v": 2}


@pytest.mark.asyncio
async def test_feature_on_is_per_row_and_never_leaks_between_calls(monkeypatch, cost_on, traced):
    await _codex_call_drive(monkeypatch)
    await _codex_call_drive(monkeypatch, usage=None)

    assert [r.llm_info["usage"]["usage_source"] for r in traced.recorder.records] == ["provider", "estimate"]


def test_llm_cost_adapters_are_inert_with_feature_off():
    assert llm_cost.trace_usage_info({}) == {}
    assert llm_cost.trace_usage_info({"fork_cost_usage": llm_cost.CallUsage()}) == {}
    assert llm_cost.claude_code_usage_kwargs(None) == {}
    assert llm_cost.codex_usage_kwargs(object()) == {}
    assert llm_cost.openai_compatible_usage_kwargs("openai", None, _compat_response(_PLAIN_USAGE)) == {}
    assert llm_cost.openai_responses_usage_kwargs(_responses_reply(_RESPONSES_USAGE)) == {}

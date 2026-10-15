"""Live provider calls checked against the raw usage and a hand price formula (fork: cost-accounting).

Anthropic (``claude-haiku-5-5``, API key from ``ANTHROPIC_API_KEY``) and Codex (``gpt-6.1-sol``, the
local ChatGPT login) are called for real. Each trace row written through an ``LLMTraceRecorder`` on a
fake pool is checked against the provider's own usage block, and each price against a formula written
out in the test. Skipped unless ``HINDSIGHT_COST_LIVE=1``. Keys and tokens are never printed.

The Claude Code case also needs ``HINDSIGHT_COST_LIVE_CLAUDE_CODE=1``, because it spends subscription
quota. It compares the row's list price with the ``total_cost_usd`` the Agent SDK reports.
"""

import json
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.engine.llm_cost import RowCost, RowUsage, load_cost_settings, price_row, read_stored_usage
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.llm_trace import LLMTraceRecorder
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.codex_auth import default_codex_auth_file

ENV_LIVE = "HINDSIGHT_COST_LIVE"
ENV_CLAUDE_CODE = "HINDSIGHT_COST_LIVE_CLAUDE_CODE"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get(ENV_LIVE) != "1",
        reason=f"set {ENV_LIVE}=1 to run live provider calls",
    ),
]

ANTHROPIC_MODEL = "claude-haiku-5-5"
CODEX_MODEL = "gpt-6.1-sol"
CLAUDE_CODE_MODEL = "claude-haiku-5-5"
ENV_ENABLED = "HINDSIGHT_API_LLM_COST_ENABLED"
MILLION = 1_000_000

# About 10k tokens of fixed text: well above the cache minimum, and free of timestamps or UUIDs.
_SYSTEM_PROMPT = "\n".join(
    f"Rule {i:03d}: when summarizing a memory entry, keep the subject, the verb and the date, "
    "and drop filler words and repeated qualifiers so that the entry stays faithful to its source."
    for i in range(300)
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_note",
            "description": "Record a short note for the user.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read back the notes recorded so far.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass(frozen=True)
class CapturedRow:
    """One ``llm_requests`` INSERT as the fake pool received it."""

    provider: str
    model: str
    started_at: datetime
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    thoughts_tokens: int
    llm_info: dict[str, Any]


_INSERT_COLUMNS = (
    "id",
    "bank_id",
    "operation",
    "scope",
    "trace_id",
    "span_id",
    "parent_span_id",
    "provider",
    "model",
    "status",
    "started_at",
    "ended_at",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "thoughts_tokens",
    "total_tokens",
    "input",
    "output",
    "error",
    "llm_info",
    "metadata",
)


class _FakeConnection:
    def __init__(self, sink: list[CapturedRow]) -> None:
        self._sink = sink

    async def execute(self, sql: str, *args: Any) -> None:
        columns = dict(zip(_INSERT_COLUMNS, args, strict=True))
        self._sink.append(
            CapturedRow(
                provider=columns["provider"],
                model=columns["model"],
                started_at=columns["started_at"],
                input_tokens=columns["input_tokens"],
                output_tokens=columns["output_tokens"],
                cached_tokens=columns["cached_tokens"],
                thoughts_tokens=columns["thoughts_tokens"],
                llm_info=json.loads(columns["llm_info"]),
            )
        )


@dataclass
class FakePool:
    """Raw-pool shape that ``acquire_with_retry`` accepts (``acquire``/``release`` coroutines)."""

    rows: list[CapturedRow] = field(default_factory=list)

    async def acquire(self) -> _FakeConnection:
        return _FakeConnection(self.rows)

    async def release(self, conn: _FakeConnection) -> None:
        return None


class _SpanSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@dataclass
class Traced:
    pool: FakePool
    recorder: LLMTraceRecorder
    spy: _SpanSpy

    async def flush(self) -> CapturedRow:
        """Wait for the fire-and-forget write of the call just made; return the row it wrote."""
        before = len(self.pool.rows)
        await self.recorder._writes.drain_all(30.0)
        assert len(self.pool.rows) == before + 1, "expected exactly one trace row per call"
        return self.pool.rows[-1]


@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch) -> Iterator[Traced]:
    monkeypatch.setenv(ENV_ENABLED, "true")
    load_cost_settings.cache_clear()
    pool = FakePool()
    recorder = LLMTraceRecorder(
        pool_getter=lambda: pool,
        schema_getter=lambda: "public",
        enabled=True,
        allowed_scopes=[],
    )
    spy = _SpanSpy()
    tracing.register_span_recorder(recorder)
    tracing.register_span_recorder(spy)
    yield Traced(pool=pool, recorder=recorder, spy=spy)
    tracing.unregister_span_recorder(recorder)
    tracing.unregister_span_recorder(spy)
    load_cost_settings.cache_clear()


def _row_usage(row: CapturedRow) -> RowUsage:
    return RowUsage(
        provider=row.provider,
        model=row.model,
        started_at=row.started_at,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cached_tokens=row.cached_tokens,
        thoughts_tokens=row.thoughts_tokens,
        usage=read_stored_usage(row.llm_info.get("usage")),
    )


def _priced(row: CapturedRow) -> RowCost:
    return price_row(_row_usage(row), load_cost_settings())


def _rounded(value: float) -> float:
    """Money is stored rounded to 8 decimals, so the hand value is rounded the same way before comparing."""
    return round(value, 8)


def _n(value: int | None) -> int:
    return value or 0


# ── Anthropic ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_live_usage_and_price(traced: Traced) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    provider = LLMProvider(provider="anthropic", api_key=api_key, base_url="", model=ANTHROPIC_MODEL)
    impl = provider._provider_impl
    raw_usage: list[Any] = []
    create = impl._client.messages.create

    async def spy_create(*args: Any, **kwargs: Any) -> Any:
        response = await create(*args, **kwargs)
        raw_usage.append(response.usage)
        return response

    impl._client.messages.create = spy_create

    messages_system = {"role": "system", "content": _SYSTEM_PROMPT}
    await provider.call_with_tools(
        messages=[messages_system, {"role": "user", "content": "Store the note: the meeting moved to Thursday."}],
        tools=_TOOLS,
        tool_choice=LLM_TOOL_CHOICE_REQUIRED,
        max_completion_tokens=256,
    )
    write_row = await traced.flush()

    await provider.call_with_tools(
        messages=[messages_system, {"role": "user", "content": "Read back the notes recorded so far."}],
        tools=_TOOLS,
        tool_choice=LLM_TOOL_CHOICE_REQUIRED,
        max_completion_tokens=256,
    )
    read_row = await traced.flush()

    await provider.call(
        messages=[messages_system, {"role": "user", "content": "Reply with the single word: ready."}],
        max_completion_tokens=32,
    )
    plain_row = await traced.flush()

    assert len(raw_usage) == 3
    write_raw, read_raw, plain_raw = raw_usage
    # The first call writes the prefix, or reads it when an earlier run within 5 minutes already did.
    assert _n(write_raw.cache_creation_input_tokens) + _n(write_raw.cache_read_input_tokens) > 0
    # The second call repeats the prefix within the cache lifetime, so it must read it.
    assert _n(read_raw.cache_read_input_tokens) > 0

    for raw, row in ((write_raw, write_row), (read_raw, read_row), (plain_raw, plain_row)):
        writes = _n(raw.cache_creation_input_tokens)
        reads = _n(raw.cache_read_input_tokens)
        output = _n(raw.output_tokens)
        uncached = _n(raw.input_tokens)
        creation = getattr(raw, "cache_creation", None)
        write_5m = _n(getattr(creation, "ephemeral_5m_input_tokens", None)) if creation else writes
        write_1h = _n(getattr(creation, "ephemeral_1h_input_tokens", None)) if creation else 0

        assert row.input_tokens == uncached + writes + reads
        assert (row.cached_tokens or 0) == reads  # the recorder stores 0 as NULL
        assert row.output_tokens == output
        usage = row.llm_info["usage"]
        assert usage["v"] == 2
        assert usage["usage_source"] == "provider"
        assert usage["cache_write_tokens"] == writes
        assert usage["cache_write_5m_tokens"] + usage["cache_write_1h_tokens"] == writes

        # Haiku 5.5 per million tokens: input 0.10, 5m write 0.125, 1h write 0.20, read 0.01, output 0.50.
        hand = (uncached * 0.10 + write_5m * 0.125 + write_1h * 0.20 + reads * 0.01 + output * 0.50) / MILLION
        priced = _priced(row)
        assert priced.cost_usd == pytest.approx(_rounded(hand), abs=1e-9)

    spend = sum(_priced(row).cost_usd or 0.0 for row in (write_row, read_row, plain_row))
    assert spend < 0.01


# ── Codex ─────────────────────────────────────────────────────────────────────


def _tap_codex_stream(impl: Any, sink: list[dict[str, Any]]) -> None:
    """Pass every SSE line through unchanged and keep the usage block of each ``response.completed``."""
    original = impl._iter_sse_lines

    async def tee(response: Any) -> AsyncIterator[str]:
        event: str | None = None
        async for line in original(response):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and (
                    event == "response.completed" or data.get("type") == "response.completed"
                ):
                    usage = (data.get("response") or {}).get("usage")
                    if isinstance(usage, dict):
                        sink.append(usage)
            yield line

    impl._iter_sse_lines = tee


class _Verdict(BaseModel):
    verdict: str


@pytest.mark.asyncio
async def test_codex_live_usage_mapping_and_list_price(traced: Traced) -> None:
    if not default_codex_auth_file(None).exists():
        pytest.skip("no local Codex login (auth.json)")

    provider = LLMProvider(provider="openai-codex", api_key="", base_url="", model=CODEX_MODEL)
    usages: list[dict[str, Any]] = []
    _tap_codex_stream(provider._provider_impl, usages)

    await provider.call(messages=[{"role": "user", "content": "Reply with the single word: ready."}])
    plain_row = await traced.flush()

    await provider.call_with_tools(
        messages=[{"role": "user", "content": "Record the note: the meeting moved to Thursday."}],
        tools=_TOOLS,
        tool_choice=LLM_TOOL_CHOICE_REQUIRED,
    )
    tools_row = await traced.flush()

    await provider.call(
        messages=[{"role": "user", "content": "Return a verdict of 'ok'."}],
        response_format=_Verdict,
    )
    format_row = await traced.flush()

    assert len(usages) == 3
    for usage, row in zip(usages, (plain_row, tools_row, format_row), strict=True):
        reported_input = usage["input_tokens"]
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        reported_output = usage["output_tokens"]

        assert row.input_tokens == reported_input
        assert (row.cached_tokens or 0) == cached
        assert (row.thoughts_tokens or 0) == reasoning
        assert row.output_tokens == reported_output - reasoning

        priced = _priced(row)
        usage_info = row.llm_info["usage"]
        assert usage_info["v"] == 2
        assert usage_info["usage_source"] == "provider"
        assert isinstance(usage_info["cache_write_tokens"], int)
        assert usage_info["cache_write_tokens"] >= 0
        assert priced.billing == "plan"
        assert priced.cost_usd is None
        # gpt-6.1-sol per million tokens: uncached input 2, cached input 0.1, output 10 (reasoning included).
        hand = ((reported_input - cached) * 2 + cached * 0.1 + reported_output * 10) / MILLION
        assert priced.list_price_usd == pytest.approx(_rounded(hand), abs=1e-9)


# ── Claude Code ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get(ENV_CLAUDE_CODE) != "1",
    reason=f"set {ENV_CLAUDE_CODE}=1 to spend Claude Code subscription quota",
)
@pytest.mark.asyncio
async def test_claude_code_live_usage_and_list_price(traced: Traced, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = pytest.importorskip("claude_agent_sdk")
    result_type = sdk.ResultMessage
    original_query = sdk.query
    reported_costs: list[float] = []

    async def tee(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Pass every SDK message through unchanged and keep the cost each ResultMessage reports."""
        async for message in original_query(*args, **kwargs):
            if isinstance(message, result_type):
                cost = message.total_cost_usd
                if isinstance(cost, int | float):
                    reported_costs.append(float(cost))
            yield message

    # ClaudeCodeLLM imports ``query`` from the package on each call, so the patch reaches it.
    monkeypatch.setattr(sdk, "query", tee)

    provider = LLMProvider(provider="claude-code", api_key="", base_url="", model=CLAUDE_CODE_MODEL)
    messages = [
        {"role": "system", "content": "Answer with one word."},
        {"role": "user", "content": "Reply with the single word: ready."},
    ]
    await provider.call(messages=messages, max_completion_tokens=32)
    row = await traced.flush()

    assert len(reported_costs) == 1, "the SDK reported no total_cost_usd"
    reported = reported_costs[0]

    # The chars/4 estimate of the prompt the call would have stored without real usage.
    estimated_input = sum(len(str(m["content"])) for m in messages) // 4
    assert row.input_tokens != estimated_input
    assert row.llm_info["usage"]["usage_source"] == "provider"

    priced = _priced(row)
    assert priced.billing == "plan"
    assert priced.list_price_usd is not None
    # The CLI's total_cost_usd is its own estimate and is not a price oracle: on 2026-10-15 it billed
    # claude-haiku-5-5 cache writes at $6.25/MTok (an Opus-class rate) against the official $0.20/MTok
    # for 1h writes. Check the list price against the official Haiku 5.5 rates instead, and check the
    # reported figure is not recorded at all.
    usage = row.llm_info["usage"]
    write_5m = usage.get("cache_write_5m_tokens") or 0
    write_1h = usage.get("cache_write_1h_tokens") or 0
    reads = row.cached_tokens or 0
    uncached = row.input_tokens - reads - write_5m - write_1h
    assert uncached >= 0
    hand = (uncached * 0.10 + write_5m * 0.125 + write_1h * 0.20 + reads * 0.01 + row.output_tokens * 0.50) / MILLION
    assert priced.list_price_usd == pytest.approx(_rounded(hand), abs=1e-8)
    assert priced.provider_cost_usd is None
    assert reported > 0

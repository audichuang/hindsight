"""fork: attempt-usage — a Codex call's trace row carries the usage of every attempt.

Before this patch the row carried the last stash only: a runaway attempt (no
``response.completed``) left NULL tokens on the error row, a retry after a runaway or a
parse failure dropped the earlier attempts' tokens, and a chars/4 estimate on an error row
was not marked as one. Each test below fails with the hooks removed.
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.engine import llm_trace
from hindsight_api.engine.llm_trace import LLMRequestRecord, LLMTraceRecorder
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers import codex_llm
from hindsight_api.engine.providers.codex_llm import CodexLLM, CodexRunawayStreamError
from tests.codex_stream_stub import CodexReply, stub_codex_stream_with
from tests.fork_helpers import build_codex, cost_env_cleared, enable_cost

pytestmark = pytest.mark.asyncio

_MESSAGES = [
    {"role": "system", "content": "s" * 400},
    {"role": "user", "content": "u" * 3600},
]
_INPUT_ESTIMATE = 4000 // 4
_CEILING = 30_000  # small runaway ceiling so a test stream runs away quickly


class _Facts(BaseModel):
    facts: list[str]


class _CapturingRecorder(LLMTraceRecorder):
    def __init__(self) -> None:
        super().__init__(pool_getter=lambda: None, schema_getter=lambda: "public", enabled=True, allowed_scopes=[])
        self.records: list[LLMRequestRecord] = []

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        self.records.append(record)


@pytest.fixture(autouse=True)
def _clean_cost_env(monkeypatch):
    yield from cost_env_cleared(monkeypatch)


@pytest.fixture
def cost_on(monkeypatch):
    enable_cost(monkeypatch)


@pytest.fixture
def recorder():
    rec = _CapturingRecorder()
    tracing.register_span_recorder(rec)
    yield rec
    tracing.unregister_span_recorder(rec)


@pytest.fixture(autouse=True)
def _small_ceiling(monkeypatch):
    monkeypatch.setattr(codex_llm, "_MAX_SSE_BODY_CHARS", _CEILING)
    monkeypatch.setattr(codex_llm, "get_metrics_collector", lambda: MagicMock())


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _usage(input_tokens: int, output_tokens: int, cached: int = 0, reasoning: int = 0, writes: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes},
        "output_tokens": output_tokens + reasoning,
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def _completed(text: str, usage: dict | None) -> str:
    events: list[tuple[str, dict]] = [("response.text.delta", {"delta": text})]
    if usage is not None:
        events.append(("response.completed", {"type": "response.completed", "response": {"usage": usage}}))
    return _sse(*events) + "data: [DONE]\n\n"


_RUNAWAY_DELTA = "event: response.text.delta\ndata: " + json.dumps({"delta": "x" * 80}) + "\n\n"


def _runaway() -> str:
    """Deltas well past the ceiling, never completing."""
    return _RUNAWAY_DELTA * (2 * _CEILING // len(_RUNAWAY_DELTA))


# Delta events counted before the ceiling trips: the client reads in chunks, and the chunk
# that trips the guard is counted whole, so the count is at least every event below the
# ceiling and at most what one more read of the socket buffer can hold.
_DELTAS_PER_RUNAWAY = range(_CEILING // len(_RUNAWAY_DELTA), 2 * _CEILING // len(_RUNAWAY_DELTA) + 1)


def _wrapper(replies: list[str]) -> LLMProvider:
    llm = LLMProvider(provider="anthropic", api_key="test-key", base_url="", model="claude-x")
    llm._provider_impl = build_codex("gpt-x")
    llm.provider = "openai-codex"
    llm.model = "gpt-x"
    queue = list(replies)
    llm._fork_test_handler = lambda _url, **_kw: CodexReply(text=queue.pop(0))  # type: ignore[attr-defined]
    return llm


async def _call(llm: LLMProvider, *, response_format=None, max_retries: int = 1):
    with stub_codex_stream_with(llm._provider_impl, llm._fork_test_handler):  # type: ignore[attr-defined]
        return await llm.call(
            messages=_MESSAGES,
            response_format=response_format,
            scope="reflect",
            max_retries=max_retries,
            initial_backoff=0.0,
            max_backoff=0.0,
        )


def _usage_marker(record: LLMRequestRecord) -> dict:
    return record.llm_info.get("usage", {})


async def test_runaway_on_every_attempt_records_an_estimate(cost_on, recorder):
    llm = _wrapper([_runaway(), _runaway()])
    with pytest.raises(CodexRunawayStreamError):
        await _call(llm)
    [row] = recorder.records
    assert row.status == "error"
    assert row.input_tokens == 2 * _INPUT_ESTIMATE
    assert row.output_tokens is not None
    per_attempt = row.output_tokens / 2
    assert int(per_attempt) == per_attempt and per_attempt in _DELTAS_PER_RUNAWAY
    assert _usage_marker(row)["usage_source"] == "estimate"


async def test_runaway_then_success_adds_both_attempts(cost_on, recorder):
    llm = _wrapper([_runaway(), _completed("answer", _usage(50_000, 700, cached=40_000, reasoning=300))])
    result = await _call(llm)
    assert result.content == "answer"
    [row] = recorder.records
    assert row.status == "success"
    # The runaway sent the same messages as the attempt that completed: its input is that count.
    assert row.input_tokens == 2 * 50_000
    assert row.cached_tokens == 40_000
    assert row.thoughts_tokens == 300
    assert row.output_tokens - 700 in _DELTAS_PER_RUNAWAY
    assert _usage_marker(row)["usage_source"] == "estimate"


async def test_parse_failure_retry_adds_both_provider_usages(cost_on, recorder):
    llm = _wrapper(
        [
            _completed("not json", _usage(1000, 50, cached=200, writes=300)),
            _completed('{"facts": ["a"]}', _usage(1100, 60, cached=900, writes=500)),
        ]
    )
    await _call(llm, response_format=_Facts)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (2100, 110, 1100)
    assert _usage_marker(row)["usage_source"] == "provider"
    assert _usage_marker(row)["cache_write_tokens"] == 800


async def test_parse_failure_without_usage_marks_the_error_row_as_estimate(cost_on, recorder):
    llm = _wrapper([_completed("not json", None)])
    with pytest.raises(json.JSONDecodeError):
        await _call(llm, response_format=_Facts, max_retries=0)
    [row] = recorder.records
    assert row.status == "error"
    assert row.input_tokens == _INPUT_ESTIMATE  # the provider's own chars/4 stash, unchanged
    assert _usage_marker(row)["usage_source"] == "estimate"


async def test_one_attempt_with_provider_usage_is_left_alone(cost_on, recorder):
    llm = _wrapper([_completed("answer", _usage(1000, 50, cached=200, reasoning=10))])
    await _call(llm)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens, row.cached_tokens, row.thoughts_tokens) == (1000, 50, 200, 10)
    assert _usage_marker(row) == {
        "v": 2,
        "cache_write_tokens": 0,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "usage_source": "provider",
    }


async def test_inert_with_the_feature_off(recorder):
    llm = _wrapper([_runaway(), _completed("answer", _usage(1000, 50))])
    await _call(llm)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens) == (1000, 50)
    assert "usage" not in row.llm_info


async def test_concurrent_calls_keep_separate_ledgers(cost_on, recorder):
    import asyncio

    first = _wrapper([_runaway(), _completed("a", _usage(1200, 10))])
    second = _wrapper([_completed("b", _usage(3000, 20))])
    await asyncio.gather(_call(first), _call(second))
    rows = sorted(recorder.records, key=lambda r: r.input_tokens or 0)
    assert rows[0].input_tokens == 2 * 1200
    assert (rows[1].input_tokens, rows[1].output_tokens) == (3000, 20)
    assert _usage_marker(rows[1])["usage_source"] == "provider"


def test_a_success_row_counts_its_own_attempt_even_without_a_stash(cost_on, recorder):
    """A provider path that records success without stashing (the tool path after upstream #5693)
    must not have its real counts replaced by the delta estimate."""
    from hindsight_api.engine.llm_cost import CallUsage
    from hindsight_api.engine.providers.codex_output_cap import CodexAttemptScan

    token = llm_trace.set_response_usage(None)
    try:
        CodexAttemptScan(None).feed("event: response.output_text.delta\ndata: {}\n\n" * 7, 0)
        recorder.record_llm_call(
            provider="openai-codex",
            model="gpt-x",
            scope="reflect",
            messages=_MESSAGES,
            input_tokens=900,
            output_tokens=40,
            fork_cost_usage=CallUsage(cache_write_tokens=0, usage_source="provider"),
        )
    finally:
        llm_trace.reset_response_usage(token)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens) == (900, 40)
    assert _usage_marker(row)["usage_source"] == "provider"


# --- the hooks are still there ---------------------------------------------------------------


def _source(path: str) -> str:
    return (Path(__file__).resolve().parents[1] / "hindsight_api" / path).read_text()


def test_llm_trace_keeps_the_attempt_usage_hooks():
    source = _source("engine/llm_trace.py")
    assert "from .llm_attempt_usage import (  # fork: attempt-usage" in source
    assert "    open_call_ledger()  # fork: attempt-usage" in inspect.getsource(llm_trace.set_response_usage)
    assert "    close_call_ledger()  # fork: attempt-usage" in inspect.getsource(llm_trace.reset_response_usage)
    assert "    note_stashed_usage(usage)  # fork: attempt-usage" in inspect.getsource(llm_trace.stash_response_usage)
    assert "    @fold_attempt_usage  # fork: attempt-usage\n    def record_llm_call(" in source


def test_codex_keeps_the_attempt_meter_hooks():
    # One scan per attempt serves the meter and the output cap (FORK-CODEX-OUTPUT-CAP.md, C3/C4).
    body = inspect.getsource(CodexLLM._iter_sse_lines)
    assert "attempt = self._fork_attempt_scan()  # fork: codex-output-cap; fork: attempt-usage" in body
    assert (
        "            attempt.feed(text, seen_chars)  # fork: codex-output-cap; fork: attempt-usage\n"
        "            if seen_chars > _MAX_SSE_BODY_CHARS:"
    ) in body

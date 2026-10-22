"""fork: attempt-usage — a Codex call's trace row carries the usage of every attempt.

Before this patch the row carried the last stash only: a runaway attempt (no
``response.completed``) left NULL tokens on the error row, a retry after a runaway or a
parse failure dropped the earlier attempts' tokens, and a chars/4 estimate on an error row
was not marked as one. Each test below fails with the hooks removed.
"""

import contextvars
import inspect
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.config import clear_config_cache
from hindsight_api.engine import llm_trace
from hindsight_api.engine.llm_cost import load_cost_settings
from hindsight_api.engine.llm_trace import LLMRequestRecord, LLMTraceRecorder
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers import codex_llm
from hindsight_api.engine.providers.codex_llm import CodexLLM, CodexRunawayStreamError
from tests.codex_stream_stub import CodexReply, stub_codex_stream, stub_codex_stream_with

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
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        impl = CodexLLM(
            provider="openai-codex", api_key="ignored", base_url="https://chatgpt.com/backend-api", model="gpt-x"
        )
    llm._provider_impl = impl
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
    # Each attempt trips on its own chunk boundary, so the two counts may differ; more than one attempt's
    # maximum proves both were counted.
    assert _DELTAS_PER_RUNAWAY.stop - 1 < row.output_tokens <= 2 * (_DELTAS_PER_RUNAWAY.stop - 1)
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
    from hindsight_api.engine.llm_attempt_usage import begin_codex_attempt
    from hindsight_api.engine.llm_cost import CallUsage

    token = llm_trace.set_response_usage(None)
    try:
        begin_codex_attempt().feed("event: response.output_text.delta\ndata: {}\n\n" * 7)
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
    body = inspect.getsource(CodexLLM._iter_sse_lines)
    assert "attempt = begin_codex_attempt()  # fork: attempt-usage" in body
    assert (
        "            attempt.feed(text)  # fork: attempt-usage\n            if seen_chars > _MAX_SSE_BODY_CHARS:"
        in body
    )


def test_codex_call_keeps_the_stash_alias_hook():
    """Upstream's call() stashes a copy of the parsed usage; the alias keeps the cost note on that copy."""
    body = inspect.getsource(CodexLLM.call)
    assert "from hindsight_api.engine.llm_cost import alias_codex_usage_note  # fork: attempt-usage" in body
    assert "stash_response_usage(\n                    alias_codex_usage_note(  # fork: attempt-usage" in body


def _failed(usage: dict | None) -> str:
    response: dict = {"status": "failed", "error": {"code": "server_error", "message": "boom"}}
    if usage is not None:
        response["usage"] = usage
    return _sse(("response.failed", {"type": "response.failed", "response": response}))


async def test_a_failed_attempt_counts_the_usage_its_failure_event_reports(cost_on, recorder):
    llm = _wrapper([_failed(_usage(50000, 700, cached=40000, writes=100)), _completed("answer", _usage(1000, 50))])
    await _call(llm)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (51000, 750, 40000)
    assert _usage_marker(row)["usage_source"] == "provider"
    assert _usage_marker(row)["cache_write_tokens"] == 100


async def test_an_error_row_keeps_the_usage_of_its_failure_event(cost_on, recorder):
    llm = _wrapper([_failed(_usage(50000, 700, cached=40000))])
    with pytest.raises(codex_llm.CodexStreamError, match="server_error"):
        await _call(llm, max_retries=0)
    [row] = recorder.records
    assert row.status == "error"
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (50000, 700, 40000)


def test_failure_usage_is_not_stashed_with_the_feature_off():
    from hindsight_api.engine.llm_attempt_usage import stash_codex_event_usage
    from hindsight_api.engine.llm_trace import current_response_usage

    event = {"type": "response.failed", "response": {"status": "failed", "usage": _usage(50000, 700)}}
    assert stash_codex_event_usage("response.failed", event) is None
    assert current_response_usage() is None


def test_codex_parsers_keep_the_event_usage_hook():
    for parser in (CodexLLM._parse_sse_stream, CodexLLM._parse_sse_tool_stream):
        body = inspect.getsource(parser)
        assert (
            "from hindsight_api.engine.llm_attempt_usage import stash_codex_event_usage  # fork: attempt-usage" in body
        )
        assert (
            "usage = stash_codex_event_usage(event_type, data) or usage  # fork: attempt-usage\n"
            "                    _raise_on_sse_failure(event_type, data)\n"
            "                    usage = _usage_from_completed_event(event_type, data) or usage" in body
        )


# --- usage that arrives before the stream fails to end, and a truncated reply's usage ----------------

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


def _codex() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex", api_key="ignored", base_url="https://chatgpt.com/backend-api", model="gpt-x"
        )


def _drop_after_completed(impl: CodexLLM):
    """The real line reader (so the attempt meter runs), with the connection dropped right after ``response.completed``."""
    original = impl._iter_sse_lines

    async def dropping(response):
        async for line in original(response):
            yield line
            if line.startswith("data: ") and "response.completed" in line:
                raise aiohttp.ClientPayloadError("connection dropped before the body ended")

    return dropping


@pytest.mark.parametrize("path", ["call", "call_with_tools"])
async def test_completed_usage_survives_a_connection_dropped_after_it(path, cost_on):
    """``response.completed`` is charged even when the body then fails to finish; the stash keeps it."""
    llm = _codex()
    with (
        stub_codex_stream(llm, CodexReply(text=_completed("answer", _usage(50000, 700, cached=40000, reasoning=30)))),
        patch.object(llm, "_iter_sse_lines", _drop_after_completed(llm)),
        pytest.raises(aiohttp.ClientPayloadError),
    ):
        if path == "call":
            await llm.call(messages=_MESSAGES, response_format=_Facts, max_retries=0)
        else:
            await llm.call_with_tools(messages=_MESSAGES, tools=_TOOLS, max_retries=0)

    usage = llm_trace.current_response_usage()
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens) == (
        50000,
        700,
        40000,
        30,
    )


async def test_an_error_row_keeps_the_usage_of_a_completed_event_before_the_drop(cost_on, recorder):
    llm = _wrapper([_completed("answer", _usage(50000, 700, cached=40000))])
    with patch.object(llm._provider_impl, "_iter_sse_lines", _drop_after_completed(llm._provider_impl)):
        with pytest.raises(aiohttp.ClientPayloadError):
            await _call(llm, max_retries=0)
    [row] = recorder.records
    assert row.status == "error"
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (50000, 700, 40000)
    assert _usage_marker(row).get("usage_source") != "estimate"


async def test_a_dropped_attempt_counts_its_completed_usage_on_the_retry_row(cost_on, recorder):
    llm = _wrapper([_completed("answer", _usage(50000, 700, cached=40000)), _completed("answer", _usage(1000, 50))])
    original = llm._provider_impl._iter_sse_lines
    dropping = _drop_after_completed(llm._provider_impl)
    attempts = []

    def first_drops(response):
        attempts.append(response)
        return dropping(response) if len(attempts) == 1 else original(response)

    with patch.object(llm._provider_impl, "_iter_sse_lines", first_drops):
        await _call(llm, max_retries=1)
    [row] = recorder.records
    assert row.status == "success"
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (51000, 750, 40000)
    assert _usage_marker(row)["usage_source"] == "provider"


def _incomplete(text: str, usage: dict) -> str:
    return _sse(
        ("response.text.delta", {"delta": text}),
        ("response.incomplete", {"type": "response.incomplete", "response": {"status": "incomplete", "usage": usage}}),
    )


async def test_a_truncated_reply_reports_the_usage_of_its_incomplete_event(cost_on):
    """An incomplete response is charged; its usage, not the chars/4 estimate, is the call's usage."""
    llm = _codex()
    with stub_codex_stream(llm, CodexReply(text=_incomplete('{"facts": ["a"]}', _usage(50000, 700, 40000, 30)))):
        result = await llm.call(messages=_MESSAGES, response_format=_Facts)
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cached_tokens,
        result.usage.thoughts_tokens,
    ) == (50000, 700, 40000, 30)


async def test_a_truncated_reply_row_carries_provider_usage(cost_on, recorder):
    llm = _wrapper([_incomplete("answer", _usage(50000, 700, cached=40000, writes=100))])
    await _call(llm, max_retries=0)
    [row] = recorder.records
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (50000, 700, 40000)
    assert _usage_marker(row)["usage_source"] == "provider"
    assert _usage_marker(row)["cache_write_tokens"] == 100


async def test_a_truncated_reply_keeps_the_char_estimate_with_the_feature_off():
    llm = _codex()
    with stub_codex_stream(llm, CodexReply(text=_incomplete('{"facts": ["a"]}', _usage(50000, 700)))):
        result = await llm.call(messages=_MESSAGES, response_format=_Facts)
    assert result.usage.input_tokens == _INPUT_ESTIMATE


def _stash_in_a_fresh_context(event_type: str | None, data: dict) -> tuple[object, object]:
    """Run the hook in its own context, so its stash cannot leak into later tests on this worker."""
    from hindsight_api.engine.llm_attempt_usage import stash_codex_event_usage

    def run() -> tuple[object, object]:
        return stash_codex_event_usage(event_type, data), llm_trace.current_response_usage()

    return contextvars.copy_context().run(run)


@pytest.mark.parametrize(
    "event_type",
    [None, "response.output_text.delta"],
    ids=["no-event-line", "stale-event-line"],
)
def test_an_event_named_only_in_its_data_is_stashed(cost_on, event_type):
    """The parsers keep the last `event:` name across events, so data.type alone must also count."""
    data = {"type": "response.completed", "response": {"usage": _usage(50000, 700)}}
    _, stashed = _stash_in_a_fresh_context(event_type, data)
    assert stashed is not None
    assert (stashed.input_tokens, stashed.output_tokens) == (50000, 700)


@pytest.mark.parametrize("flag", ["on", "off"])
@pytest.mark.parametrize("bad_type", [[], {}, 7], ids=["list", "dict", "int"])
def test_a_malformed_event_type_never_raises(monkeypatch, flag, bad_type):
    """Odd metadata in a data-only event must not abort a stream, with the feature on or off."""
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true" if flag == "on" else "false")
    clear_config_cache()
    load_cost_settings.cache_clear()
    returned, stashed = _stash_in_a_fresh_context(None, {"type": bad_type, "response": {"usage": _usage(1, 1)}})
    assert returned is None and stashed is None


def test_a_delta_event_never_reads_the_cost_settings(monkeypatch):
    """The hook runs on every SSE data line; only a terminal event may reach the flag check."""
    from hindsight_api.engine import llm_attempt_usage

    reads: list[None] = []

    def counting() -> bool:  # the hook swallows exceptions, so count instead of raising
        reads.append(None)
        return True

    monkeypatch.setattr(llm_attempt_usage, "_cost_enabled", counting)
    delta = {"type": "response.output_text.delta", "delta": "x"}
    assert llm_attempt_usage.stash_codex_event_usage("response.output_text.delta", delta) is None
    assert reads == []

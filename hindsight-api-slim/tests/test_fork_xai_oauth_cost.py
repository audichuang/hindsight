"""Fork hook H22: xAI OAuth provider-reported cost (cost-accounting, D5).

Upstream ``_ChatUsage`` gains ``cost_in_usd_ticks``; both ``XaiOAuthLLM`` span call sites forward
``xai_oauth_usage_kwargs`` through the ``fork_usage`` keyword of ``_record_span``. With the feature off the
span kwargs must match upstream exactly. The source-text test fails if a hook is lost in a rebase.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from hindsight_api import tracing
from hindsight_api.config import clear_config_cache
from hindsight_api.engine.llm_cost import load_cost_settings
from hindsight_api.engine.llm_trace import LLMRequestRecord, LLMTraceRecorder
from hindsight_api.engine.providers.xai_oauth_llm import _ChatCompletion
from tests.test_xai_oauth_llm import _make_llm, _ok_reply

PROVIDER_SRC = (
    Path(__file__).resolve().parent.parent / "hindsight_api" / "engine" / "providers" / "xai_oauth_llm.py"
).read_text(encoding="utf-8")

H22_TICKS_FIELD = (
    "    cost_in_usd_ticks: object = None  # fork: cost-accounting (tolerant: a bad value never fails the parse)"
)
H22_RECORD_PARAM = "        fork_usage: dict[str, Any] | None = None,  # fork: cost-accounting"
H22_FORWARD = "                **(fork_usage or {}),  # fork: cost-accounting"
H22_IMPORT = "                from hindsight_api.engine.llm_cost import xai_oauth_usage_kwargs  # fork: cost-accounting"
H22_TICKS_LOCAL = "                fork_ticks = completion.usage.cost_in_usd_ticks if completion.usage else None  # fork: cost-accounting"
H22_CALL_KWARG = "                    fork_usage=xai_oauth_usage_kwargs(fork_ticks),  # fork: cost-accounting"

TICKS = 123_450_000_000  # 12.345 USD at 1e10 ticks per USD

# Keys upstream passes to record_llm_call from _record_span; feature off must send exactly these.
UPSTREAM_SPAN_KEYS = {
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
    "tool_calls",
}


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


class _KwargsRecorder:
    """Stands in for a span recorder and keeps the kwargs of every record_llm_call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _CapturingRecorder(LLMTraceRecorder):
    """Keeps the built trace records in memory instead of writing them to the DB."""

    def __init__(self) -> None:
        super().__init__(pool_getter=lambda: None, schema_getter=lambda: "public", enabled=True, allowed_scopes=[])
        self.records: list[LLMRequestRecord] = []

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        self.records.append(record)


def _usage_body(**extra: Any) -> dict[str, Any]:
    return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, **extra}


# ── source-text hook presence ─────────────────────────────────────────────────


def test_h22_hook_lines_are_present_once_each():
    assert PROVIDER_SRC.count(H22_TICKS_FIELD) == 1
    assert PROVIDER_SRC.count(H22_RECORD_PARAM) == 1
    assert PROVIDER_SRC.count(H22_FORWARD) == 1
    assert PROVIDER_SRC.count(H22_IMPORT) == 2
    assert PROVIDER_SRC.count(H22_TICKS_LOCAL) == 2
    assert PROVIDER_SRC.count(H22_CALL_KWARG) == 2


def test_h22_marker_count_is_exactly_the_nine_hook_lines():
    # _ChatUsage field, _record_span param, forward kwarg, and per call site: import, ticks, kwarg.
    assert PROVIDER_SRC.count("# fork: cost-accounting") == 9


def test_h22_import_is_local_inside_the_call_paths():
    # Local, not module level: the fork module stays out of the provider's import graph (J7).
    for line in PROVIDER_SRC.splitlines():
        if "hindsight_api.engine.llm_cost import xai_oauth_usage_kwargs" in line:
            assert line.startswith(" " * 16), line


def test_h22_call_sites_pass_fork_usage_after_the_parsed_completion():
    # Each call site must read the parsed completion (the variable in scope), not a raw body.
    assert PROVIDER_SRC.count('completion = await self._request_completion(body, body["messages"])') == 2
    assert PROVIDER_SRC.count(H22_TICKS_LOCAL) == 2


# ── parsing ───────────────────────────────────────────────────────────────────


def test_chat_completion_parses_cost_in_usd_ticks():
    completion = _ChatCompletion.model_validate(
        {"choices": [], "usage": _usage_body(cost_in_usd_ticks=TICKS)},
    )

    assert completion.usage is not None
    assert completion.usage.cost_in_usd_ticks == TICKS


@pytest.mark.parametrize("bad", [1.5, "abc", "12abc", 1e20, True, [1], {"a": 1}])
def test_chat_completion_with_a_bad_cost_still_parses(bad):
    # A 200 body whose optional cost field is the wrong type must not fail the whole completion (that path
    # is retried as an unparseable body and burns plan quota). The bad value is simply not read as a cost.
    completion = _ChatCompletion.model_validate({"choices": [], "usage": _usage_body(cost_in_usd_ticks=bad)})

    assert completion.usage is not None
    assert completion.usage.prompt_tokens == _usage_body()["prompt_tokens"]


def test_chat_completion_without_cost_leaves_the_field_none():
    completion = _ChatCompletion.model_validate({"choices": [], "usage": _usage_body()})

    assert completion.usage is not None
    assert completion.usage.cost_in_usd_ticks is None


# ── span kwargs ───────────────────────────────────────────────────────────────


@pytest.fixture
def span_kwargs():
    recorder = _KwargsRecorder()
    tracing.register_span_recorder(recorder)
    yield recorder
    tracing.unregister_span_recorder(recorder)


@pytest.mark.asyncio
async def test_feature_off_sends_no_fork_cost_usage(tmp_path, monkeypatch, span_kwargs):
    reply = _ok_reply("ok", usage=_usage_body(cost_in_usd_ticks=TICKS))
    llm = _make_llm(tmp_path, monkeypatch, replies=[reply])

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert len(span_kwargs.calls) == 1
    kwargs = span_kwargs.calls[0]
    assert "fork_cost_usage" not in kwargs
    # Identical to upstream: the same keyword set, nothing added by the hook.
    assert set(kwargs) == UPSTREAM_SPAN_KEYS


@pytest.mark.asyncio
async def test_feature_on_sends_provider_cost_on_the_span(tmp_path, monkeypatch, span_kwargs):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    clear_config_cache()
    load_cost_settings.cache_clear()
    reply = _ok_reply("ok", usage=_usage_body(cost_in_usd_ticks=TICKS))
    llm = _make_llm(tmp_path, monkeypatch, replies=[reply])

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert len(span_kwargs.calls) == 1
    usage = span_kwargs.calls[0]["fork_cost_usage"]
    assert usage.provider_cost_usd == TICKS / 1e10
    assert usage.usage_source == "provider"


@pytest.mark.asyncio
async def test_feature_on_trace_row_carries_provider_cost_usd(tmp_path, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    clear_config_cache()
    load_cost_settings.cache_clear()
    recorder = _CapturingRecorder()
    tracing.register_span_recorder(recorder)
    try:
        reply = _ok_reply("ok", usage=_usage_body(cost_in_usd_ticks=TICKS))
        llm = _make_llm(tmp_path, monkeypatch, replies=[reply])

        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    finally:
        tracing.unregister_span_recorder(recorder)

    assert len(recorder.records) == 1
    assert recorder.records[0].llm_info["usage"]["provider_cost_usd"] == TICKS / 1e10


@pytest.mark.asyncio
async def test_feature_on_without_ticks_sends_no_provider_cost(tmp_path, monkeypatch, span_kwargs):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    clear_config_cache()
    load_cost_settings.cache_clear()
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply("ok", usage=_usage_body())])

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    usage = span_kwargs.calls[0]["fork_cost_usage"]
    assert usage.provider_cost_usd is None

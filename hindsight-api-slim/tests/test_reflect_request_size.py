"""The reflect context budget bounds what the provider is sent, in the provider's tokens.

``max_context_tokens`` used to be compared against a local count of the message
text only. It missed the tool definitions and the provider tokenizer's gap to the
local encoding, so with the budget at 100K, Claude Haiku reflect turns went out at
~120K. These tests drive the agent with a fake provider whose tokenizer counts
``_GAP`` times the local encoding and that reports that size back as usage, the way
real providers do, and assert no request it receives is over the budget.

All deterministic: no network, a fake provider, the real local tokenizer.
"""

import itertools
import json
import math
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect import agent as agent_module
from hindsight_api.engine.reflect.agent import _count_messages_tokens, run_reflect_agent
from hindsight_api.engine.reflect.request_size import RequestSizer
from hindsight_api.engine.reflect.tokenization import count_prompt_tokens
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult, TokenUsage

#: Provider tokens per local token. Production measured >= 18% for Claude Haiku
#: against o200k_base.
_GAP = 1.22
#: The budget the reference deployment runs with.
_BUDGET = 95_000

_ids = itertools.count()


def _tool_tokens(tools: list[dict[str, Any]]) -> int:
    return count_prompt_tokens(json.dumps(tools, ensure_ascii=False))


class _GappedProvider:
    """A provider whose tokenizer counts ``_GAP`` x the local encoding.

    Every request's provider-side size is recorded, and reported back as its
    ``input_tokens``. In free turns it keeps asking for one large recall, so
    the conversation grows until reflect stops it; ``prose_on_turn`` makes that
    turn answer with text instead.
    """

    provider = "anthropic"
    model = "gapped-test-model"
    _provider_impl = None  # no incremental prompt cache

    def __init__(
        self, *, report_usage: bool = True, prose_on_turn: int | None = None, bad_document: bool = False
    ) -> None:
        self.bad_document = bad_document
        self.report_usage = report_usage
        self.prose_on_turn = prose_on_turn
        self.tool_requests: list[tuple[str, int]] = []
        self.last_tool_tokens = 0
        self.text_requests: list[int] = []
        self.text_prompts: list[list[dict[str, Any]]] = []

    @staticmethod
    def size(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        local = _count_messages_tokens(messages) + (_tool_tokens(tools) if tools else 0)
        return math.ceil(local * _GAP)

    async def call_with_tools(self, *, messages, tools, tool_choice, **_kwargs) -> LLMToolCallResult:
        real = self.size(messages, tools)
        name = tool_choice.function_name or tool_choice.mode.value
        self.tool_requests.append((name, real))
        self.last_tool_tokens = _tool_tokens(tools) if tools else 0
        usage = {"input_tokens": real, "output_tokens": 40} if self.report_usage else {}
        if self.prose_on_turn is not None and len(self.tool_requests) == self.prose_on_turn:
            return LLMToolCallResult(content="I have enough.", finish_reason="stop", **usage)
        if name == "done" and self.bad_document:
            calls = [LLMToolCall(id=f"d{next(_ids)}", name="done", arguments={"document": {"sections": "nope"}})]
        elif name == "done":
            calls = [LLMToolCall(id=f"d{next(_ids)}", name="done", arguments={"answer": "From done."})]
        elif name in ("auto", "required"):  # a free turn; upstream #5832 sends it as required
            calls = [LLMToolCall(id=f"c{next(_ids)}", name="recall", arguments={"query": "more", "max_tokens": 16000})]
        else:
            calls = [LLMToolCall(id=f"c{next(_ids)}", name=name, arguments={"query": "q"})]
        return LLMToolCallResult(tool_calls=calls, finish_reason="tool_calls", **usage)

    async def call(self, *, messages, **_kwargs) -> LLMCallResult:
        real = self.size(messages)
        self.text_requests.append(real)
        self.text_prompts.append(messages)
        return LLMCallResult(
            content="Synthesized answer.",
            usage=TokenUsage(input_tokens=real, output_tokens=10, total_tokens=real + 10),
        )


async def _recall(query: str, max_tokens: int, max_chunk_tokens: int) -> dict[str, Any]:
    """Recall that fills whatever token budget it is given, ~500 local tokens per memory."""
    return {
        "memories": [{"id": f"mem-{next(_ids)}", "content": "fact " * 500} for _ in range(max(1, max_tokens // 500))]
    }


def _callbacks(**overrides: Any) -> dict[str, Any]:
    return {
        "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "search_observations_fn": AsyncMock(return_value={"observations": []}),
        "recall_fn": AsyncMock(side_effect=_recall),
        "expand_fn": AsyncMock(return_value={"memories": []}),
        **overrides,
    }


async def _reflect(
    provider: _GappedProvider, *, max_iterations: int = 10, context: str | None = None, **overrides: Any
):
    include_observations = overrides.pop("include_observations", False)
    return await run_reflect_agent(
        llm_config=provider,
        bank_id="test-bank",
        query="What do you know?",
        bank_profile={"name": "Test", "mission": "Testing"},
        max_context_tokens=_BUDGET,
        max_iterations=max_iterations,
        context=context,
        include_observations=include_observations,
        **_callbacks(**overrides),
    )


def _all_sizes(provider: _GappedProvider) -> list[int]:
    return [size for _, size in provider.tool_requests] + provider.text_requests


# ── RequestSizer ──────────────────────────────────────────────────────────────


def _tools() -> list[dict[str, Any]]:
    return get_reflect_tools(include_mental_models=True, include_observations=True, include_recall=True)


def test_estimate_is_the_local_count_until_usage_is_reported():
    sizer = RequestSizer(_tools())
    assert sizer.estimate(80_000) == 80_000
    sizer.observe(80_000, 0)
    assert sizer.estimate(80_000) == 80_000


def test_reported_usage_counts_tools_and_the_tokenizer_gap():
    """Production scale: 80K local messages reported at 1.2x; 10K more fires the guard the old count missed."""
    tools = _tools()
    sizer = RequestSizer(tools)
    sizer.observe(80_000, math.ceil(1.2 * (80_000 + _tool_tokens(tools))))
    assert 90_000 < _BUDGET <= sizer.estimate(90_000)
    assert sizer.estimate(90_000) == math.ceil((90_000 + _tool_tokens(tools)) * sizer.ratio)


def test_a_report_below_the_local_count_still_counts_the_tools():
    tools = _tools()
    sizer = RequestSizer(tools)
    sizer.observe(5_000, 10)  # MockLLM reports a flat 10
    assert sizer.ratio == 1.0
    assert sizer.estimate(5_000) == 5_000 + _tool_tokens(tools)


def test_the_largest_ratio_seen_is_kept():
    tools = _tools()
    sizer = RequestSizer(tools)
    local = 10_000 - _tool_tokens(tools)
    sizer.observe(local, 13_000)
    sizer.observe(local, 11_000)
    assert sizer.ratio == pytest.approx(1.3)


def test_an_implausible_report_is_capped():
    tools = _tools()
    sizer = RequestSizer(tools)
    sizer.observe(1_000, 10_000_000)
    assert sizer.ratio == 3.0


def test_to_local_converts_a_provider_allowance():
    tools = _tools()
    sizer = RequestSizer(tools)
    sizer.observe(10_000 - _tool_tokens(tools), 12_500)
    assert sizer.to_local(_BUDGET) == 76_000


# ── the agent loop ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_request_exceeds_the_budget_in_provider_tokens():
    """The loop stops retrieving before a turn would go over, and synthesis fits too.

    Before the fix the turn after ~85.7K was sent at ~105.8K: its local message
    count (~86K) was still under 95K.
    """
    provider = _GappedProvider()

    result = await _reflect(provider)

    assert max(_all_sizes(provider)) <= _BUDGET
    # It used the budget rather than stopping early.
    assert max(size for _, size in provider.tool_requests) > 0.8 * _BUDGET
    assert [c.scope for c in result.llm_trace][-1] == "final"


@pytest.mark.asyncio
async def test_closing_done_on_the_last_iteration_respects_the_budget():
    """The last iteration asks for ``done`` without passing the loop's guard."""
    provider = _GappedProvider()

    result = await _reflect(provider, max_iterations=7)

    assert max(_all_sizes(provider)) <= _BUDGET
    assert "done" not in [name for name, _ in provider.tool_requests]
    assert "closing_done" not in [c.scope for c in result.llm_trace]
    assert [c.scope for c in result.llm_trace][-1] == "final"


@pytest.mark.asyncio
async def test_closing_done_after_a_prose_stop_respects_the_budget(monkeypatch):
    """A prose stop asks for ``done`` on the conversation plus a closing prompt the guard never saw."""
    monkeypatch.setattr(agent_module, "build_done_request_prompt", lambda *_a, **_k: "pad " * 100_000)
    provider = _GappedProvider(prose_on_turn=3)

    result = await _reflect(provider)

    assert max(_all_sizes(provider)) <= _BUDGET
    assert "done" not in [name for name, _ in provider.tool_requests]
    assert [c.scope for c in result.llm_trace][-1] == "final"


@pytest.mark.asyncio
async def test_forced_recall_is_compacted_to_fit_in_provider_tokens():
    """#4563 at production scale: observations fill the budget only once the gap is counted."""
    observations = [{"id": f"obs-{i}", "text": f"Observation {i}: " + "obs " * 500} for i in range(156)]
    provider = _GappedProvider()
    recall_fn = AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "Newest correction"}]})

    result = await _reflect(
        provider,
        include_observations=True,
        search_observations_fn=AsyncMock(return_value={"observations": observations}),
        recall_fn=recall_fn,
    )

    assert max(_all_sizes(provider)) <= _BUDGET
    recall_fn.assert_awaited_once()
    assert provider.tool_requests[1][0] == "recall"
    # The model saw a compacted conversation; the answer still reads every observation.
    assert "Observation 0" in str(provider.text_prompts)
    assert [c.scope for c in result.llm_trace][-1] == "final"


@pytest.mark.asyncio
async def test_a_provider_without_usage_keeps_the_local_estimate():
    """No report, nothing learned: the guard compares the local message count, as before."""
    provider = _GappedProvider(report_usage=False)

    await _reflect(provider)

    # The last turn it sent was under the budget by local count, which is all it knew: the
    # messages only, since the tool definitions are not counted before a usage report.
    _, last = provider.tool_requests[-1]
    assert last / _GAP - provider.last_tool_tokens < _BUDGET + 1


@pytest.mark.asyncio
async def test_a_document_re_ask_over_the_budget_fails_and_says_why(monkeypatch):
    """A refused document is never re-asked past the ceiling, and the error names the budget."""
    asks = itertools.count()
    # The first closing ask is small; the re-ask carries a rejected payload that is not.
    monkeypatch.setattr(
        agent_module,
        "build_done_request_prompt",
        lambda *_a, **_k: "pad " * 100_000 if next(asks) else "Give your answer.",
    )
    provider = _GappedProvider(prose_on_turn=3, bad_document=True)

    with pytest.raises(agent_module.DocumentSectionsInvalidError, match="exceed the context budget"):
        await _reflect(provider)

    assert max(_all_sizes(provider)) <= _BUDGET


@pytest.mark.asyncio
async def test_synthesis_history_budget_leaves_room_for_the_prompt_around_it(monkeypatch):
    """Additional context sits outside the history budget, so the budget is what is left after it."""
    context = "background " * 35_000
    budgets: list[int] = []
    real_split = agent_module.split_context_history

    def spy(history, max_context_tokens):
        budgets.append(max_context_tokens)
        return real_split(history, max_context_tokens)

    monkeypatch.setattr(agent_module, "split_context_history", spy)
    provider = _GappedProvider()

    await _reflect(provider, context=context)

    (budget,) = budgets
    # In provider tokens, the history budget plus the context it sits beside fits the limit.
    assert math.ceil((budget + count_prompt_tokens(context)) * _GAP) <= _BUDGET
    assert max(_all_sizes(provider)) <= _BUDGET

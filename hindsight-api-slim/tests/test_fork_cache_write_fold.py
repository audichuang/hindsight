"""Missing-total cache writes through the pricing path (fork: cost-accounting, audit follow-up).

The provider fold tests are upstream-bound (test_fork_anthropic_fold.py, test_fork_claude_code_usage.py).
This file covers what those cannot: the folded input and the write count the cost hook records must price
together, with a non-negative uncached remainder.
"""

from datetime import UTC, datetime

import pytest

from hindsight_api.engine.llm_cost import (
    FORK_USAGE_KWARG,
    CostSettings,
    RowUsage,
    StoredUsage,
    claude_code_usage_kwargs,
    read_stored_usage,
    summarize_costs,
)
from hindsight_api.engine.providers.claude_code_llm import _usage_from_sdk_usage
from tests.fork_helpers import cost_env_cleared


@pytest.fixture(autouse=True)
def _cost_on(monkeypatch):
    yield from cost_env_cleared(monkeypatch, enabled=True)


def test_missing_total_fold_prices_without_negative_uncached_input():
    raw = {
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000},
    }
    folded = _usage_from_sdk_usage(raw)
    assert folded is not None
    call = claude_code_usage_kwargs(raw)[FORK_USAGE_KWARG]
    row = RowUsage(
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        input_tokens=folded.input_tokens,
        output_tokens=folded.output_tokens,
        cached_tokens=folded.cached_tokens,
        thoughts_tokens=None,
        usage=read_stored_usage(call.to_llm_info()["usage"]) or StoredUsage(),
    )

    summary = summarize_costs(
        [row],
        CostSettings.from_env({"HINDSIGHT_API_LLM_COST_ENABLED": "true"}),
        bank_id="bank-1",
        period=None,
        start="2026-10-01",
        end=None,
        truncated=False,
    )

    assert summary.unpriced_calls == 0
    (model,) = summary.models
    assert model.cache_write_tokens == 1000
    assert model.input_tokens == 1100  # the folded total: 100 uncached + 1000 written
    assert summary.cost_usd > 0

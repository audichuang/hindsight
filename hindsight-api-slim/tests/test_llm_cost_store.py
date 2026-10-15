"""Fork tests: the cost store's window rules (cost-accounting). Pure functions, no database."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from hindsight_api.engine.llm_cost_store import (
    CostQuery,
    LLMCostError,
    _to_stored,
    check_bank_readable,
    resolve_window,
)
from hindsight_api.extensions import OperationValidationError

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
END = datetime(2026, 10, 10, 0, 0, tzinfo=UTC)
START = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)


def test_period_alone_is_the_last_n_days_up_to_now():
    window = resolve_window(CostQuery(period="7d"), NOW)
    assert window.start == NOW - timedelta(days=7)
    assert window.end is None
    assert window.period == "7d"


def test_start_alone_runs_from_start_to_now():
    window = resolve_window(CostQuery(period="7d", start=START), NOW)
    assert window.start == START
    assert window.end is None
    assert window.period is None


def test_end_alone_is_the_period_that_ends_at_end_date():
    # Regression (P7): end_date without start_date used to start at now minus the period, which
    # could leave the window empty or inverted (a 400 for an end in the past).
    window = resolve_window(CostQuery(period="7d", end=END), NOW)
    assert window.start == END - timedelta(days=7)
    assert window.end == END
    assert window.period is None


def test_both_bounds_are_used_exactly():
    window = resolve_window(CostQuery(period="30d", start=START, end=END), NOW)
    assert (window.start, window.end, window.period) == (START, END, None)


def test_end_alone_in_the_past_is_not_an_error():
    past_end = NOW - timedelta(days=400)
    window = resolve_window(CostQuery(period="1d", end=past_end), NOW)
    assert window.end == past_end
    assert window.start == past_end - timedelta(days=1)


@pytest.mark.parametrize(
    "query",
    [
        CostQuery(period="2d"),
        CostQuery(period="7d", start=END, end=START),
        CostQuery(period="7d", start=END, end=END),
    ],
)
def test_bad_period_or_inverted_bounds_are_400(query):
    with pytest.raises(LLMCostError) as raised:
        resolve_window(query, NOW)
    assert raised.value.status_code == 400


def test_cost_query_defaults_to_utc_days():
    query = CostQuery(period="7d")
    assert query.tz is UTC
    assert query.tz_name == "UTC"


def _record(**overrides):
    record = {
        "id": uuid4(),
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "started_at": datetime(2026, 10, 1, 17, 30, tzinfo=UTC),
        "input_tokens": 2010,
        "output_tokens": 300,
        "cached_tokens": 0,
        "thoughts_tokens": None,
    }
    record.update(overrides)
    return record


def test_to_stored_reads_the_usage_marker():
    stored = _to_stored(
        _record(),
        {"v": 2, "cache_write_tokens": 2000, "cache_write_5m_tokens": 2000, "cache_write_1h_tokens": 0},
    )
    assert stored.usage.usage is not None
    assert stored.usage.usage.cache_write_tokens == 2000
    assert stored.usage.usage.cache_write_5m_tokens == 2000


def test_to_stored_without_a_marker_has_no_usage():
    stored = _to_stored(_record(), None)
    assert stored.usage.usage is None


def test_to_stored_normalises_a_one_hour_only_marker():
    # D8: a row with only the 1h write and no total must not price a negative 5m write.
    stored = _to_stored(_record(), {"v": 2, "cache_write_1h_tokens": 500})
    assert stored.usage.usage.cache_write_1h_tokens == 500
    assert stored.usage.usage.cache_write_5m_tokens == 0
    assert stored.usage.usage.cache_write_tokens == 500


class _FakeMemory:
    """Stands in for MemoryEngine: a bank lookup and the whole-bank scope refusal, nothing else."""

    def __init__(self, *, exists: bool, refuse: Exception | None = None) -> None:
        self._exists = exists
        self._refuse = refuse
        self.refused_with: list[str] = []

    async def get_bank_profile(self, bank_id, request_context):
        return {"bank_id": bank_id} if self._exists else None

    async def _refuse_bank_wide_if_scoped(self, bank_id, request_context, what):
        self.refused_with.append(what)
        if self._refuse is not None:
            raise self._refuse


@pytest.mark.asyncio
async def test_check_bank_readable_passes_for_a_whole_bank_caller():
    memory = _FakeMemory(exists=True)
    await check_bank_readable(memory, "bank-a", object(), "read the server LLM quota")
    assert memory.refused_with == ["read the server LLM quota"]


@pytest.mark.asyncio
async def test_check_bank_readable_unknown_bank_is_404_before_the_scope_check():
    memory = _FakeMemory(exists=False)
    with pytest.raises(LLMCostError) as raised:
        await check_bank_readable(memory, "missing", object(), "read the bank's LLM request costs")
    assert raised.value.status_code == 404
    assert raised.value.detail == "Bank 'missing' not found"
    assert memory.refused_with == []


@pytest.mark.asyncio
async def test_check_bank_readable_scoped_caller_is_403_with_the_action_text():
    refusal = OperationValidationError("A tag-scoped caller can't read the bank's LLM request costs", status_code=403)
    memory = _FakeMemory(exists=True, refuse=refusal)
    with pytest.raises(OperationValidationError) as raised:
        await check_bank_readable(memory, "bank-a", object(), "read the bank's LLM request costs")
    assert raised.value.status_code == 403
    assert memory.refused_with == ["read the bank's LLM request costs"]

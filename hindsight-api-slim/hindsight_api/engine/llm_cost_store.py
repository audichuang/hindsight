"""Fork: LLM cost store (cost-accounting). Reads ``llm_requests`` and prices the rows for the cost endpoints.

``api/llm_cost_routes.py`` only parses parameters and maps errors. Everything that touches the engine
or the database lives here: bank access checks, the window rules, the SQL and the pricing hand-off.
PostgreSQL only; on Oracle ``llm_requests`` does not exist, so the store answers with no rows.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING, Any
from uuid import UUID

from hindsight_api.engine.db import ResultRow
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.llm_cost import (
    CostSettings,
    LLMCostSummaryResponse,
    LLMRowCostResponse,
    RowUsage,
    price_row,
    read_stored_usage,
    summarize_costs,
)
from hindsight_api.engine.llm_trace import call_type_clause
from hindsight_api.engine.schema import _is_oracle, fq_table

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.models import RequestContext

ROW_CAP = 200_000  # the summary reads one row past this to know whether it truncated
_PERIOD_DAYS = {"1d": 1, "7d": 7, "30d": 30}
_COLUMNS = (
    "id, provider, model, started_at, input_tokens, output_tokens, cached_tokens, thoughts_tokens, "
    "llm_info->'usage' AS usage"
)


_READ_ACTION = "read the bank's LLM request costs"


class LLMCostError(Exception):
    """A request the store refuses. The route maps it to ``status_code`` with ``detail``."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class CostQuery:
    period: str
    start: datetime | None = None
    end: datetime | None = None
    operation: str | None = None
    tz: tzinfo = UTC  # day buckets are local calendar dates in this zone
    tz_name: str = "UTC"  # the zone name as requested, echoed in the response


@dataclass(frozen=True)
class CostWindow:
    start: datetime
    end: datetime | None
    period: str | None  # None when explicit bounds replace the period


@dataclass(frozen=True)
class _StoredRow:
    row_id: str
    usage: RowUsage


def resolve_window(query: CostQuery, now: datetime) -> CostWindow:
    """The window a query covers.

    ``period`` alone: the last N days up to now. ``start`` alone: from start up to now.
    ``end`` alone: the N days that end at ``end``. Both: exactly [start, end). The bounds
    replace the period, so a window with explicit bounds reports no period.
    """
    if query.period not in _PERIOD_DAYS:
        raise LLMCostError(400, "period must be one of 1d, 7d, 30d")
    period_length = timedelta(days=_PERIOD_DAYS[query.period])
    if query.start is not None:
        start = query.start
    elif query.end is not None:
        start = query.end - period_length
    else:
        return CostWindow(start=now - period_length, end=None, period=query.period)
    if query.end is not None and query.end <= start:
        raise LLMCostError(400, "end_date must be after start_date")
    return CostWindow(start=start, end=query.end, period=None)


async def check_bank_readable(
    memory: "MemoryEngine", bank_id: str, request_context: "RequestContext", action: str
) -> None:
    """Whole-bank read check shared by the cost store and the quota route.

    Unknown bank: 404. A tag-scoped caller is refused with an ``OperationValidationError`` (403)
    that names ``action``.
    """
    if await memory.get_bank_profile(bank_id, request_context=request_context) is None:
        raise LLMCostError(404, f"Bank '{bank_id}' not found")
    await memory._refuse_bank_wide_if_scoped(bank_id, request_context, action)


def _to_stored(record: ResultRow, raw_usage: object) -> _StoredRow:
    return _StoredRow(
        row_id=str(record["id"]),
        usage=RowUsage(
            provider=record["provider"],
            model=record["model"],
            started_at=record["started_at"],
            input_tokens=record["input_tokens"],
            output_tokens=record["output_tokens"],
            cached_tokens=record["cached_tokens"],
            thoughts_tokens=record["thoughts_tokens"],
            usage=read_stored_usage(raw_usage),
        ),
    )


class LLMCostStore:
    """Prices a bank's LLM request rows. Access checks match ``llm_request_stats``."""

    def __init__(self, memory: "MemoryEngine", settings: CostSettings) -> None:
        self._memory = memory
        self._settings = settings

    async def summarize(
        self, bank_id: str, request_context: "RequestContext", query: CostQuery
    ) -> LLMCostSummaryResponse:
        await check_bank_readable(self._memory, bank_id, request_context, _READ_ACTION)
        window = resolve_window(query, datetime.now(UTC))
        if _is_oracle():
            return summarize_costs(
                [],
                self._settings,
                bank_id=bank_id,
                period=window.period,
                start=window.start.isoformat(),
                end=window.end.isoformat() if window.end else None,
                truncated=False,
                tz=query.tz,
                tz_name=query.tz_name,
            )
        where = ["bank_id = $1", "started_at >= $2"]
        params: list[Any] = [bank_id, window.start]
        if window.end is not None:
            params.append(window.end)
            where.append(f"started_at < ${len(params)}")
        if query.operation:
            params.append(query.operation)
            where.append(f"operation = ${len(params)}")
        stored = await self._fetch(" AND ".join(where), params, limit=ROW_CAP + 1)
        truncated = len(stored) > ROW_CAP
        # Pricing is CPU-bound (about 30 us a row), so it runs in a worker thread, not on the event loop.
        return await asyncio.to_thread(
            summarize_costs,
            [item.usage for item in stored[:ROW_CAP]],
            self._settings,
            bank_id=bank_id,
            period=window.period,
            start=window.start.isoformat(),
            end=window.end.isoformat() if window.end else None,
            truncated=truncated,
            tz=query.tz,
            tz_name=query.tz_name,
        )

    async def row_costs(
        self, bank_id: str, request_context: "RequestContext", ids: Sequence[UUID]
    ) -> LLMRowCostResponse:
        await check_bank_readable(self._memory, bank_id, request_context, _READ_ACTION)
        if _is_oracle():
            return LLMRowCostResponse(bank_id=bank_id, items={})
        stored = await self._fetch("bank_id = $1 AND id = ANY($2::uuid[])", [bank_id, list(ids)], limit=None)
        return LLMRowCostResponse(
            bank_id=bank_id,
            items={item.row_id: price_row(item.usage, self._settings) for item in stored},
        )

    async def _fetch(self, where_sql: str, params: list[Any], limit: int | None) -> list[_StoredRow]:
        """LLM-call rows for one bank as pricer inputs, newest first.

        ``llm_requests`` also holds embedding, rerank and decision-model rows (told apart by scope since
        upstream #5835); they are not LLM calls, are not costed, and must not use up ``ROW_CAP``.
        The connection is held only for the query and the JSON decode; the conversion to
        ``RowUsage`` happens after it is released, so a 200k-row read does not pin a pool connection.
        """
        limit_sql = "" if limit is None else f" LIMIT {limit}"
        where = f"({where_sql}) AND {call_type_clause('llm')}"
        sql = f"SELECT {_COLUMNS} FROM {fq_table('llm_requests')} WHERE {where} ORDER BY started_at DESC{limit_sql}"
        backend = await self._memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            records = await conn.fetch(sql, *params)
            decoded = [(record, conn.parse_json(record["usage"])) for record in records]
        return [_to_stored(record, raw_usage) for record, raw_usage in decoded]

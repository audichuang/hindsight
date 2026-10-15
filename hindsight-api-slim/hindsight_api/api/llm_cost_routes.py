"""Fork: LLM cost endpoints (cost-accounting).

Registered only when ``HINDSIGHT_API_LLM_COST_ENABLED`` is on. The routes are
excluded from the OpenAPI schema, so the upstream client surface does not change. They are added with
``app.get`` rather than a separate ``APIRouter`` so they keep the app's ``UnknownParamsRoute`` route class.

This module only parses query parameters and maps errors. The reads and the pricing live in
``engine/llm_cost_store.py`` and ``engine/llm_cost.py``.
"""

from collections.abc import Callable
from datetime import tzinfo
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query

from hindsight_api.engine.llm_cost import (
    CostSettings,
    LLMCostSummaryResponse,
    LLMRowCostResponse,
    load_cost_settings,
    parse_timezone,
)
from hindsight_api.engine.llm_cost_store import CostQuery, LLMCostError, LLMCostStore
from hindsight_api.extensions import AuthenticationError, OperationValidationError
from hindsight_api.models import RequestContext

_MAX_ROW_IDS = 500


def _parse_ids(raw: str) -> list[UUID]:
    parts = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not parts:
        raise HTTPException(status_code=400, detail="ids must list at least one llm request id")
    if len(parts) > _MAX_ROW_IDS:
        raise HTTPException(status_code=400, detail=f"at most {_MAX_ROW_IDS} ids per request")
    try:
        return [UUID(part) for part in parts]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated UUIDs")


def _parse_timezone(raw: str) -> tzinfo:
    try:
        return parse_timezone(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _as_http_error(exc: Exception, where: str) -> HTTPException:
    """Map a store or access error to its HTTP status; anything else is logged and returned as a 500."""
    # Local import: http.py imports this module from inside _register_routes, so a module-level
    # import back into http.py would be a cycle.
    from hindsight_api.api.http import _internal_error  # noqa: PLC0415

    if isinstance(exc, LLMCostError):
        return HTTPException(status_code=exc.status_code, detail=exc.detail)
    if isinstance(exc, OperationValidationError):
        return HTTPException(status_code=exc.status_code, detail=exc.reason)
    return _internal_error(exc, where)


def register_llm_cost_routes(app: FastAPI, get_request_context: Callable[..., RequestContext]) -> None:
    """Register the cost routes on ``app`` when the feature is enabled. A bad env var raises here."""
    cost_settings = load_cost_settings()
    if cost_settings.enabled:
        _register_cost_routes(app, get_request_context, cost_settings)


def _register_cost_routes(
    app: FastAPI, get_request_context: Callable[..., RequestContext], settings: CostSettings
) -> None:
    from hindsight_api.api.http import _parse_iso_datetime  # noqa: PLC0415

    def _store() -> LLMCostStore:
        return LLMCostStore(app.state.memory, settings)

    @app.get(
        "/v1/default/banks/{bank_id}/llm-requests/cost",
        include_in_schema=False,
        response_model=LLMCostSummaryResponse,
    )
    async def api_llm_request_cost(
        bank_id: str,
        period: str = Query("7d", description="Time period: 1d, 7d, or 30d"),
        operation: str | None = Query(None, description="Filter by operation"),
        start_date: str | None = Query(None, description="ISO start; overrides period"),
        end_date: str | None = Query(None, description="ISO end; overrides period"),
        tz: str = Query("UTC", description="IANA time zone for per-day buckets"),
        request_context: RequestContext = Depends(get_request_context),
    ) -> LLMCostSummaryResponse:
        """Estimated LLM spend for a bank over a window, with per-model and per-day totals."""
        where = f"GET /v1/default/banks/{bank_id}/llm-requests/cost"
        try:
            zone = _parse_timezone(tz)
            query = CostQuery(
                period=period,
                start=_parse_iso_datetime(start_date, "start_date"),
                end=_parse_iso_datetime(end_date, "end_date"),
                operation=operation,
                tz=zone,
                tz_name=tz,
            )
            return await _store().summarize(bank_id, request_context, query)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            raise _as_http_error(e, where)

    @app.get(
        "/v1/default/banks/{bank_id}/llm-requests/cost/rows",
        include_in_schema=False,
        response_model=LLMRowCostResponse,
    )
    async def api_llm_request_cost_rows(
        bank_id: str,
        ids: str = Query(..., description="Comma-separated llm request ids (at most 500)"),
        request_context: RequestContext = Depends(get_request_context),
    ) -> LLMRowCostResponse:
        """Per-row cost for the given llm request ids. Unknown ids are omitted."""
        where = f"GET /v1/default/banks/{bank_id}/llm-requests/cost/rows"
        try:
            return await _store().row_costs(bank_id, request_context, _parse_ids(ids))
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            raise _as_http_error(e, where)


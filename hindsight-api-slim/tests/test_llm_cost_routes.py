"""Fork tests: the LLM cost endpoints (cost-accounting) on the pg0 test database."""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.api.unknown_params import UnknownParamsRoute
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.llm_cost import (
    ENV_ENABLED,
    CostSettings,
    RowUsage,
    StoredUsage,
    load_cost_settings,
    price_row,
    summarize_costs,
)
from hindsight_api.engine.schema import fq_table

COST_PATH = "/v1/default/banks/{bank_id}/llm-requests/cost"
ROWS_PATH = "/v1/default/banks/{bank_id}/llm-requests/cost/rows"


@pytest.fixture
def bank_id():
    return f"llm_cost_test_{uuid4().hex}"


@asynccontextmanager
async def _client(memory, monkeypatch, *, enabled: bool):
    """A client on an app built with the cost feature on or off. Settings are cached per process, so clear them."""
    if enabled:
        monkeypatch.setenv(ENV_ENABLED, "true")
    else:
        monkeypatch.delenv(ENV_ENABLED, raising=False)
    load_cost_settings.cache_clear()
    try:
        app = create_app(memory, initialize_memory=False)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client
    finally:
        load_cost_settings.cache_clear()


@pytest_asyncio.fixture
async def cost_client(memory, monkeypatch):
    async with _client(memory, monkeypatch, enabled=True) as pair:
        yield pair


async def _insert(
    memory,
    bank_id: str,
    *,
    provider: str,
    model: str,
    started_at: datetime,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    thoughts_tokens: int = 0,
    operation: str = "retain",
    usage: dict | None = None,
    status: str = "success",
    scope: str | None = None,
) -> str:
    row_id = uuid4()
    llm_info = {"usage": usage} if usage is not None else {}
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            f"INSERT INTO {fq_table('llm_requests')} "
            "(id, bank_id, operation, status, provider, model, started_at, input_tokens, output_tokens, "
            "cached_tokens, thoughts_tokens, llm_info, scope) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13)",
            row_id,
            bank_id,
            operation,
            status,
            provider,
            model,
            started_at,
            input_tokens,
            output_tokens,
            cached_tokens,
            thoughts_tokens,
            json.dumps(llm_info),
            scope,
        )
    return str(row_id)


async def _seed(memory, bank_id: str) -> dict[str, str]:
    """Three rows: a Haiku cache write, a plan-billed Codex call, and a provider with no pricing."""
    now = datetime.now(UTC)
    haiku = await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=now,
        input_tokens=2010,
        output_tokens=300,
        usage={"v": 1, "cache_write_tokens": 2000, "cache_write_5m_tokens": 2000, "cache_write_1h_tokens": 0},
    )
    codex = await _insert(
        memory,
        bank_id,
        provider="openai-codex",
        model="gpt-6.1-sol",
        started_at=now,
        input_tokens=1000,
        output_tokens=100,
        cached_tokens=200,
        thoughts_tokens=50,
    )
    unmapped = await _insert(
        memory,
        bank_id,
        provider="mystery-cloud",
        model="some-model",
        started_at=now,
        input_tokens=10,
        output_tokens=5,
    )
    return {"haiku": haiku, "codex": codex, "unmapped": unmapped}


async def _create_bank(client, bank_id: str) -> None:
    await client.put(f"/v1/default/banks/{bank_id}", json={"name": "Cost Bank"})


@pytest.mark.asyncio
async def test_summary_matches_summarize_costs(cost_client, memory, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    await _seed(memory, bank_id)

    response = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d"})
    assert response.status_code == 200
    body = response.json()

    # Hand check, from the formula in price_row's tests: 2000 write at 1.25 per Mtok, 10 uncached at 1.00,
    # 300 output at 5.00 per Mtok, so the Haiku row alone is 0.000401 USD.
    haiku_models = [m for m in body["models"] if m["model"] == "claude-haiku-5-5"]
    assert len(haiku_models) == 1
    assert haiku_models[0]["cache_write_tokens"] == 2000
    assert haiku_models[0]["cost_usd"] == pytest.approx(0.000401)

    expected = summarize_costs(
        [
            RowUsage(
                provider="anthropic",
                model="claude-haiku-5-5",
                started_at=datetime.now(UTC),
                input_tokens=2010,
                output_tokens=300,
                cached_tokens=0,
                thoughts_tokens=0,
                usage=StoredUsage(cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0),
            ),
            RowUsage(
                provider="openai-codex",
                model="gpt-6.1-sol",
                started_at=datetime.now(UTC),
                input_tokens=1000,
                output_tokens=100,
                cached_tokens=200,
                thoughts_tokens=50,
                usage=None,
            ),
            RowUsage(
                provider="mystery-cloud",
                model="some-model",
                started_at=datetime.now(UTC),
                input_tokens=10,
                output_tokens=5,
                cached_tokens=None,
                thoughts_tokens=None,
                usage=None,
            ),
        ],
        CostSettings.from_env({ENV_ENABLED: "true"}),
        bank_id=bank_id,
        period="1d",
        start="",
        end=None,
        truncated=False,
    )
    assert body["models"] == expected.model_dump(mode="json")["models"]
    assert body["by_day"] == expected.model_dump(mode="json")["by_day"]
    assert body["cost_usd"] == pytest.approx(expected.cost_usd)
    assert body["plan_list_price_usd"] == pytest.approx(expected.plan_list_price_usd)
    assert body["unpriced_calls"] == expected.unpriced_calls == 1
    assert body["truncated"] is False


@pytest.mark.asyncio
async def test_summary_reads_llm_calls_only(cost_client, memory, bank_id):
    """Embedding, rerank and decision-model rows share llm_requests since upstream #5835; they are not priced."""
    _, client = cost_client
    await _create_bank(client, bank_id)
    await _seed(memory, bank_id)
    now = datetime.now(UTC)
    for scope, provider, model in (
        ("embedding_document", "openai", "qwen/qwen3-embedding-4b"),
        ("rerank", "cohere", "rerank-v4.0-fast"),
        ("decision_consolidation_triage", "typesafe", "jev"),
    ):
        await _insert(
            memory,
            bank_id,
            provider=provider,
            model=model,
            started_at=now,
            input_tokens=500,
            output_tokens=0,
            scope=scope,
        )

    response = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d"})
    assert response.status_code == 200
    body = response.json()
    assert body["unpriced_calls"] == 1  # the seed's own unmapped LLM row, nothing else
    assert {m["model"] for m in body["models"]} == {"claude-haiku-5-5", "gpt-6.1-sol", "some-model"}


@pytest.mark.asyncio
async def test_rows_endpoint_gives_no_cost_line_to_an_embedding_row(cost_client, memory, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    ids = await _seed(memory, bank_id)
    embedding = await _insert(
        memory,
        bank_id,
        provider="openai",
        model="text-embedding-3-small",
        started_at=datetime.now(UTC),
        input_tokens=500,
        output_tokens=0,
        scope="embedding_query",
    )

    response = await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": f"{ids['haiku']},{embedding}"})
    assert response.status_code == 200
    assert set(response.json()["items"]) == {ids["haiku"]}


@pytest.mark.asyncio
async def test_window_and_operation_filters(cost_client, memory, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    now = datetime.now(UTC)
    await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=now - timedelta(days=10),
        input_tokens=100,
        output_tokens=10,
        operation="retain",
    )
    await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=now,
        input_tokens=100,
        output_tokens=10,
        operation="reflect",
    )

    recent = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "7d"})
    assert [m["calls"] for m in recent.json()["models"]] == [1]

    explicit = await client.get(
        COST_PATH.format(bank_id=bank_id),
        params={
            "start_date": (now - timedelta(days=11)).isoformat(),
            "end_date": (now - timedelta(days=9)).isoformat(),
        },
    )
    assert explicit.status_code == 200
    assert [m["calls"] for m in explicit.json()["models"]] == [1]
    assert explicit.json()["period"] is None

    by_operation = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "30d", "operation": "retain"})
    assert [m["calls"] for m in by_operation.json()["models"]] == [1]


@pytest.mark.asyncio
async def test_end_date_alone_reads_the_period_before_it(cost_client, memory, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    now = datetime.now(UTC)
    await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=now - timedelta(days=20),
        input_tokens=100,
        output_tokens=10,
    )
    end = now - timedelta(days=15)

    response = await client.get(COST_PATH.format(bank_id=bank_id), params={"end_date": end.isoformat()})

    # Regression: end_date alone used to start at now minus the period, which put the window after end.
    # The window is [end - 7d, end] = [now-22d, now-15d], so only the row from 20 days ago is in it.
    assert response.status_code == 200
    assert response.json()["period"] is None
    assert [m["calls"] for m in response.json()["models"]] == [1]


@pytest.mark.asyncio
async def test_invalid_period_and_inverted_window_are_400(cost_client, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    assert (await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "2d"})).status_code == 400
    now = datetime.now(UTC)
    inverted = await client.get(
        COST_PATH.format(bank_id=bank_id),
        params={"start_date": now.isoformat(), "end_date": (now - timedelta(days=1)).isoformat()},
    )
    assert inverted.status_code == 400


@pytest.mark.asyncio
async def test_unknown_bank_is_404(cost_client, bank_id):
    _, client = cost_client
    assert (await client.get(COST_PATH.format(bank_id=bank_id))).status_code == 404
    assert (await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": str(uuid4())})).status_code == 404


@pytest.mark.asyncio
async def test_rows_endpoint_prices_requested_ids(cost_client, memory, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    ids = await _seed(memory, bank_id)
    missing = str(uuid4())

    response = await client.get(
        ROWS_PATH.format(bank_id=bank_id),
        params={"ids": f"{ids['haiku']},{ids['codex']},{missing}"},
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert set(items) == {ids["haiku"], ids["codex"]}
    assert items[ids["haiku"]]["cost_usd"] == pytest.approx(0.000401)
    assert items[ids["codex"]]["billing"] == "plan"
    assert items[ids["codex"]]["cost_usd"] is None
    assert items[ids["codex"]]["list_price_usd"] is not None


@pytest.mark.asyncio
async def test_rows_endpoint_rejects_bad_ids(cost_client, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)
    too_many = ",".join(str(uuid4()) for _ in range(501))
    assert (await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": too_many})).status_code == 400
    assert (await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": "not-a-uuid"})).status_code == 400
    assert (await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": " , "})).status_code == 400


@pytest.mark.asyncio
async def test_routes_absent_and_hidden_from_openapi_when_disabled(memory, monkeypatch, bank_id):
    async with _client(memory, monkeypatch, enabled=False) as (app, client):
        await _create_bank(client, bank_id)
        assert (await client.get(COST_PATH.format(bank_id=bank_id))).status_code == 404
        assert (await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": str(uuid4())})).status_code == 404
        assert COST_PATH not in app.openapi()["paths"]


@pytest.mark.asyncio
async def test_routes_hidden_from_openapi_when_enabled(memory, monkeypatch):
    async with _client(memory, monkeypatch, enabled=True) as (app, _):
        paths = app.openapi()["paths"]
        assert COST_PATH not in paths
        assert ROWS_PATH not in paths


@pytest.mark.asyncio
async def test_rows_endpoint_routes_are_unknown_params_routes(cost_client):
    app, _ = cost_client

    cost_routes = [
        r for r in app.routes if getattr(r, "path", "").startswith("/v1/default/banks/{bank_id}/llm-requests/cost")
    ]
    assert len(cost_routes) == 2
    assert all(isinstance(r, UnknownParamsRoute) for r in cost_routes)


def test_fork_hook_in_http_is_present():
    """Fails if a conflict resolution drops the H6 hook that registers the cost routes."""
    source = (Path(__file__).parents[1] / "hindsight_api" / "api" / "http.py").read_text()
    assert (
        "# fork: cost-accounting\n    from hindsight_api.api.llm_cost_routes import register_llm_cost_routes\n"
        in source
    )
    assert "    register_llm_cost_routes(app, get_request_context)\n" in source


def test_haiku_cache_write_row_prices_to_hand_value():
    settings = CostSettings.from_env({ENV_ENABLED: "true"})
    cost = price_row(
        RowUsage(
            provider="anthropic",
            model="claude-haiku-5-5",
            started_at=datetime.now(UTC),
            input_tokens=2010,
            output_tokens=300,
            cached_tokens=0,
            thoughts_tokens=0,
            usage=StoredUsage(cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0),
        ),
        settings,
    )
    assert cost.cost_usd == pytest.approx(0.000401)


@pytest.mark.asyncio
async def test_rows_endpoint_never_prices_another_banks_row(cost_client, memory, bank_id):
    """Bank isolation: bank A asking for bank B's llm request id gets nothing back."""
    _, client = cost_client
    other_bank = f"{bank_id}_other"
    await _create_bank(client, bank_id)
    await _create_bank(client, other_bank)
    foreign = await _insert(
        memory,
        other_bank,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime.now(UTC),
        input_tokens=2010,
        output_tokens=300,
    )

    response = await client.get(ROWS_PATH.format(bank_id=bank_id), params={"ids": foreign})

    assert response.status_code == 200
    assert response.json()["items"] == {}
    own = await client.get(ROWS_PATH.format(bank_id=other_bank), params={"ids": foreign})
    assert foreign in own.json()["items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [COST_PATH, ROWS_PATH])
async def test_tag_scoped_caller_is_refused_on_both_routes(cost_client, memory, monkeypatch, bank_id, path):
    """The fork routes use the same bank-wide refusal as llm-requests/stats: a scope refusal is a 403."""
    from hindsight_api.extensions import OperationValidationError

    _, client = cost_client
    await _create_bank(client, bank_id)
    row_id = await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime.now(UTC),
        input_tokens=10,
        output_tokens=1,
    )

    async def refuse(bank, request_context, what):
        raise OperationValidationError(f"A tag-scoped caller can't {what}", status_code=403)

    monkeypatch.setattr(memory, "_refuse_bank_wide_if_scoped", refuse)
    params = {"ids": row_id} if path == ROWS_PATH else {}
    response = await client.get(path.format(bank_id=bank_id), params=params)

    assert response.status_code == 403
    assert "tag-scoped caller" in response.json()["detail"]


@pytest.mark.asyncio
async def test_summary_truncates_to_the_newest_rows_at_the_cap(cost_client, memory, monkeypatch, bank_id):
    """Past the cap the summary prices only the newest rows and says it truncated."""
    import hindsight_api.engine.llm_cost_store as store

    monkeypatch.setattr(store, "ROW_CAP", 2)
    _, client = cost_client
    await _create_bank(client, bank_id)
    now = datetime.now(UTC)
    for minutes_ago, input_tokens in [(0, 1000), (1, 2000), (2, 4000)]:
        await _insert(
            memory,
            bank_id,
            provider="anthropic",
            model="claude-haiku-5-5",
            started_at=now - timedelta(minutes=minutes_ago),
            input_tokens=input_tokens,
            output_tokens=0,
        )

    response = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is True
    (group,) = body["models"]
    assert group["calls"] == 2
    assert group["input_tokens"] == 3000  # the newest two rows: 1000 + 2000, not the oldest 4000


def test_router_holds_no_database_access():
    """The router parses parameters and maps errors; SQL and engine internals live in llm_cost_store."""
    source = (Path(__file__).parents[1] / "hindsight_api" / "api" / "llm_cost_routes.py").read_text()
    for forbidden in ("acquire_with_retry", "fq_table", "_get_backend", "_refuse_bank_wide_if_scoped", "conn."):
        assert forbidden not in source, forbidden
    assert "memory: Any" not in source


@pytest.mark.asyncio
async def test_tz_outside_the_iana_database_is_400(cost_client, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)

    response = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d", "tz": "Mars/Base"})

    assert response.status_code == 400
    assert "IANA time zone" in response.json()["detail"]


@pytest.mark.asyncio
async def test_tz_is_echoed_and_defaults_to_utc(cost_client, bank_id):
    _, client = cost_client
    await _create_bank(client, bank_id)

    default = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d"})
    taipei = await client.get(COST_PATH.format(bank_id=bank_id), params={"period": "1d", "tz": "Asia/Taipei"})

    assert default.json()["tz"] == "UTC"
    assert taipei.status_code == 200
    assert taipei.json()["tz"] == "Asia/Taipei"


@pytest.mark.asyncio
async def test_day_bucket_follows_the_requested_tz(cost_client, memory, bank_id):
    """17:30Z on Oct 1 is still Oct 1 in UTC and already Oct 2 in Asia/Taipei (UTC+8)."""
    _, client = cost_client
    await _create_bank(client, bank_id)
    await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime(2026, 10, 1, 17, 30, tzinfo=UTC),
        input_tokens=10,
        output_tokens=1,
    )
    window = {"period": "7d", "start_date": "2026-09-30T00:00:00+00:00", "end_date": "2026-10-05T00:00:00+00:00"}

    utc = await client.get(COST_PATH.format(bank_id=bank_id), params=window)
    taipei = await client.get(COST_PATH.format(bank_id=bank_id), params={**window, "tz": "Asia/Taipei"})

    assert [day["day"] for day in utc.json()["by_day"]] == ["2026-10-01"]
    assert [day["day"] for day in taipei.json()["by_day"]] == ["2026-10-02"]
    assert taipei.json()["tz"] == "Asia/Taipei"


@pytest.mark.asyncio
async def test_usage_marker_is_read_back_per_row(cost_client, memory, bank_id):
    from hindsight_api.engine.llm_cost_store import LLMCostStore

    _, client = cost_client
    await _create_bank(client, bank_id)
    row_id = await _insert(
        memory,
        bank_id,
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime.now(UTC),
        input_tokens=2010,
        output_tokens=300,
        usage={"v": 2, "cache_write_tokens": 2000, "cache_write_5m_tokens": 2000, "cache_write_1h_tokens": 0},
    )

    stored = await LLMCostStore(memory, CostSettings.from_env({ENV_ENABLED: "true"}))._fetch(
        "bank_id = $1 AND id = $2", [bank_id, UUID(row_id)], limit=None
    )

    (item,) = stored
    assert item.usage.usage is not None
    assert item.usage.usage.cache_write_tokens == 2000

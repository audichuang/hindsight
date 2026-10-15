"""Fork tests: the LLM quota route (cost-accounting) on the pg0 test database.

The route is server-wide, so the tests stub the snapshot builder rather than reach a provider. The
cost and quota flags are independent: each test sets both explicitly.
"""

from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.llm_cost import ENV_ENABLED as ENV_COST_ENABLED
from hindsight_api.engine.llm_cost import load_cost_settings
from hindsight_api.engine.llm_quota import (
    ENV_QUOTA_CLAUDE_OAUTH_TOKEN,
    ENV_QUOTA_ENABLED,
    LLMQuotaResponse,
    LLMQuotaService,
    ProviderQuota,
    load_quota_settings,
)
from tests.test_llm_cost_routes import _create_bank

QUOTA_PATH = "/v1/default/banks/{bank_id}/llm-requests/quota"
COST_PATH = "/v1/default/banks/{bank_id}/llm-requests/cost"


@pytest.fixture
def bank_id():
    return f"llm_quota_test_{uuid4().hex}"


@asynccontextmanager
async def _app_client(memory, monkeypatch, *, cost: bool, quota: bool):
    """A client on an app built with the cost and quota flags set as given. Both settings are cached, so clear them."""
    if cost:
        monkeypatch.setenv(ENV_COST_ENABLED, "true")
    else:
        monkeypatch.delenv(ENV_COST_ENABLED, raising=False)
    if quota:
        monkeypatch.setenv(ENV_QUOTA_ENABLED, "true")
    else:
        monkeypatch.delenv(ENV_QUOTA_ENABLED, raising=False)
    monkeypatch.delenv(ENV_QUOTA_CLAUDE_OAUTH_TOKEN, raising=False)
    load_cost_settings.cache_clear()
    load_quota_settings.cache_clear()
    try:
        app = create_app(memory, initialize_memory=False)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client
    finally:
        load_cost_settings.cache_clear()
        load_quota_settings.cache_clear()


@pytest_asyncio.fixture
async def quota_client(memory, monkeypatch):
    async with _app_client(memory, monkeypatch, cost=False, quota=True) as pair:
        yield pair


def _quota_routes(app) -> list:
    return [r for r in app.routes if getattr(r, "path", "") == QUOTA_PATH]


@pytest.mark.asyncio
async def test_route_absent_when_quota_is_off(memory, monkeypatch, bank_id):
    async with _app_client(memory, monkeypatch, cost=True, quota=False) as (app, client):
        await _create_bank(client, bank_id)
        assert _quota_routes(app) == []
        assert (await client.get(QUOTA_PATH.format(bank_id=bank_id))).status_code == 404
        # The cost routes are still registered: the two flags are independent.
        assert any(getattr(r, "path", "") == COST_PATH for r in app.routes)


@pytest.mark.asyncio
async def test_route_present_when_only_quota_is_on(memory, monkeypatch, bank_id):
    async with _app_client(memory, monkeypatch, cost=False, quota=True) as (app, client):
        await _create_bank(client, bank_id)
        assert len(_quota_routes(app)) == 1
        assert not any(getattr(r, "path", "") == COST_PATH for r in app.routes)
        assert (await client.get(QUOTA_PATH.format(bank_id=bank_id))).status_code == 200


@pytest.mark.asyncio
async def test_unknown_bank_is_404(quota_client, bank_id):
    _, client = quota_client
    assert (await client.get(QUOTA_PATH.format(bank_id=bank_id))).status_code == 404


@pytest.mark.asyncio
async def test_tag_scoped_caller_is_refused(quota_client, memory, monkeypatch, bank_id):
    from hindsight_api.extensions import OperationValidationError

    _, client = quota_client
    await _create_bank(client, bank_id)

    async def refuse(bank, request_context, what):
        raise OperationValidationError(f"A tag-scoped caller can't {what}", status_code=403)

    monkeypatch.setattr(memory, "_refuse_bank_wide_if_scoped", refuse)
    response = await client.get(QUOTA_PATH.format(bank_id=bank_id))

    assert response.status_code == 403
    assert "tag-scoped caller" in response.json()["detail"]


@pytest.mark.asyncio
async def test_response_shape_from_stubbed_snapshot(quota_client, monkeypatch, bank_id):
    _, client = quota_client
    await _create_bank(client, bank_id)
    seen: list[str] = []

    async def fake_fetch_all(self, bank_id_arg: str) -> LLMQuotaResponse:
        seen.append(bank_id_arg)
        return LLMQuotaResponse(
            bank_id=bank_id_arg,
            providers=[ProviderQuota(provider="openai-codex", lanes=["default"], status="unsupported", reason=None)],
        )

    monkeypatch.setattr(LLMQuotaService, "fetch_all", fake_fetch_all)
    response = await client.get(QUOTA_PATH.format(bank_id=bank_id))

    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "server"
    assert body["bank_id"] == bank_id
    assert body["providers"][0]["provider"] == "openai-codex"
    assert body["providers"][0]["status"] == "unsupported"
    assert seen == [bank_id]


@pytest.mark.asyncio
async def test_service_is_built_once_and_reused(quota_client, monkeypatch, bank_id):
    app, client = quota_client
    await _create_bank(client, bank_id)
    built: list[object] = []
    real_init = LLMQuotaService.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        real_init(self, *args, **kwargs)

    async def empty_fetch(self, bank_id_arg: str) -> LLMQuotaResponse:
        return LLMQuotaResponse(bank_id=bank_id_arg, providers=[])

    monkeypatch.setattr(LLMQuotaService, "__init__", counting_init)
    monkeypatch.setattr(LLMQuotaService, "fetch_all", empty_fetch)
    assert (await client.get(QUOTA_PATH.format(bank_id=bank_id))).status_code == 200
    assert (await client.get(QUOTA_PATH.format(bank_id=bank_id))).status_code == 200
    assert len(built) == 1


@pytest.mark.asyncio
async def test_quota_route_is_not_in_openapi(quota_client):
    app, _ = quota_client
    assert QUOTA_PATH not in app.openapi()["paths"]


@pytest.mark.asyncio
async def test_bad_quota_env_fails_startup(memory, monkeypatch):
    monkeypatch.setenv(ENV_QUOTA_ENABLED, "maybe")
    load_quota_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match=ENV_QUOTA_ENABLED):
            create_app(memory, initialize_memory=False)
    finally:
        monkeypatch.delenv(ENV_QUOTA_ENABLED, raising=False)
        load_quota_settings.cache_clear()

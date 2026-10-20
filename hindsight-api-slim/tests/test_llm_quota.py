"""Quota snapshot: parsers, target discovery, cache, service error mapping (fork: cost-accounting).

Every HTTP call goes through an injected fake; time comes from an injected clock. The secret
tokens below must never appear in a model dump, a repr or an error reading.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from hindsight_api.engine import llm_quota
from hindsight_api.engine.llm_quota import (
    HttpResult,
    LLMQuotaService,
    QuotaCache,
    QuotaReading,
    QuotaSettings,
    collect_quota_targets,
    load_quota_settings,
    parse_claude_usage,
    parse_codex_usage,
    parse_zai_usage,
)
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.providers.codex_auth import CodexRefreshExpiredError
from hindsight_api.engine.providers.codex_llm import CodexLLM

FETCHED = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
CLAUDE_TOKEN = "sk-ant-oat01-ENVSECRET-0123456789"
LANE_CLAUDE_TOKEN = "sk-ant-oat01-LANESECRET-9876543210"
CODEX_TOKEN = "codex-access-SECRET-abcdef"
ZAI_KEY = "zai-key-SECRET-555"
ALL_SECRETS = (CLAUDE_TOKEN, LANE_CLAUDE_TOKEN, CODEX_TOKEN, ZAI_KEY)


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHttp:
    """Routes by URL. A responder returns an HttpResult or raises."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def __call__(self, url: str, headers) -> HttpResult:
        self.calls.append((url, dict(headers)))
        return self._responder(url)


def _ok(payload: object) -> HttpResult:
    return HttpResult(status=200, body=json.dumps(payload))


def _assert_no_secret(*values: object) -> None:
    for value in values:
        text = value if isinstance(value, str) else repr(value)
        for secret in ALL_SECRETS:
            assert secret not in text


# --- Settings -----------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["true", "1", "yes", "TRUE", " Yes "])
def test_settings_enabled_truthy(raw):
    settings = QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_ENABLED": raw})
    assert settings.enabled is True


@pytest.mark.parametrize("raw", ["false", "0", "no", ""])
def test_settings_enabled_falsy(raw):
    assert QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_ENABLED": raw}).enabled is False


def test_settings_default_off_and_bad_bool_rejected():
    assert QuotaSettings.from_env({}).enabled is False
    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_COST_QUOTA_ENABLED"):
        QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_ENABLED": "maybe"})


def test_settings_token_is_secret_and_hidden():
    settings = QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_CLAUDE_OAUTH_TOKEN": CLAUDE_TOKEN})
    assert settings.claude_oauth_token is not None
    assert settings.claude_oauth_token.get_secret_value() == CLAUDE_TOKEN
    _assert_no_secret(settings.model_dump(), settings.model_dump_json(), repr(settings), str(settings))


def test_settings_blank_token_is_none():
    settings = QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_CLAUDE_OAUTH_TOKEN": "  "})
    assert settings.claude_oauth_token is None


def test_unknown_quota_key_rejected_by_name_only():
    env = {"HINDSIGHT_API_LLM_COST_QUOTA_BOGUS": CLAUDE_TOKEN}
    with pytest.raises(ValueError) as info:
        QuotaSettings.from_env(env)
    assert "HINDSIGHT_API_LLM_COST_QUOTA_BOGUS" in str(info.value)
    assert CLAUDE_TOKEN not in str(info.value)


def test_cost_keys_are_not_rejected_by_quota_settings():
    settings = QuotaSettings.from_env({"HINDSIGHT_API_LLM_COST_ENABLED": "true"})
    assert settings.enabled is False


def test_load_quota_settings_is_cached(monkeypatch):
    calls = []

    def fake_env() -> dict[str, str]:
        calls.append(1)
        return {"HINDSIGHT_API_LLM_COST_QUOTA_ENABLED": "true"}

    monkeypatch.setattr(llm_quota, "llm_cost_env", fake_env)
    load_quota_settings.cache_clear()
    try:
        assert load_quota_settings().enabled is True
        assert load_quota_settings().enabled is True
        assert len(calls) == 1
    finally:
        load_quota_settings.cache_clear()


# --- Codex parser -------------------------------------------------------------------------------


def _codex_payload(**overrides) -> dict:
    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "limit_reached": False,
            "primary_window": {
                "used_percent": 37,
                "limit_window_seconds": 18000,
                "reset_at": 1_781_000_000,
                "reset_after_seconds": 1200,
            },
            "secondary_window": {
                "used_percent": 12,
                "limit_window_seconds": 604800,
                "reset_at": 1_781_500_000,
                "reset_after_seconds": 90000,
            },
        },
        "credits": {"has_credits": True, "unlimited": False, "balance": "12.5"},
        "rate_limit_reset_credits": {"available_count": 2},
    }
    payload.update(overrides)
    return payload


def test_codex_primary_and_secondary_labels_and_reset():
    reading = parse_codex_usage(_codex_payload(), FETCHED)
    assert reading.status == "ok"
    assert reading.plan == "plus"
    assert [w.label for w in reading.windows] == ["5h", "Week"]
    assert reading.windows[0].window_seconds == 18000
    assert reading.windows[0].used_percent == 37.0
    assert reading.windows[0].resets_at == datetime.fromtimestamp(1_781_000_000, UTC)
    assert reading.windows[0].scope == "account"


def test_codex_weekly_only_in_primary_window_is_labelled_by_duration():
    payload = _codex_payload()
    payload["rate_limit"]["primary_window"] = {
        "used_percent": 5,
        "limit_window_seconds": 604800,
        "reset_at": 1_781_500_000,
    }
    payload["rate_limit"]["secondary_window"] = None
    reading = parse_codex_usage(payload, FETCHED)
    assert [w.label for w in reading.windows] == ["Week"]


def test_codex_missing_durations_use_positional_fallback():
    payload = _codex_payload()
    payload["rate_limit"]["primary_window"] = {"used_percent": 1}
    payload["rate_limit"]["secondary_window"] = {"used_percent": 2}
    reading = parse_codex_usage(payload, FETCHED)
    assert [w.label for w in reading.windows] == ["5h", "Week"]
    assert all(w.window_seconds is None for w in reading.windows)


def test_codex_reset_after_seconds_only_is_relative_to_fetch_time():
    payload = _codex_payload()
    payload["rate_limit"]["primary_window"] = {
        "used_percent": 9,
        "limit_window_seconds": 18000,
        "reset_after_seconds": 3600,
    }
    reading = parse_codex_usage(payload, FETCHED)
    assert reading.windows[0].resets_at == FETCHED + timedelta(hours=1)


def test_codex_credits_string_balance_and_banked_resets():
    reading = parse_codex_usage(_codex_payload(), FETCHED)
    assert reading.credits is not None
    assert reading.credits.balance == 12.5
    assert reading.credits.unlimited is False
    assert reading.credits.banked_resets == 2


def test_codex_without_credits_shows_no_balance_or_unlimited():
    # Hermes rule: a plan account without credits reports has_credits=false with a "0" balance. That is not a
    # balance to show, and unlimited is only meaningful with credits.
    payload = _codex_payload(
        credits={"has_credits": False, "unlimited": True, "balance": "0"},
        rate_limit_reset_credits=None,
    )
    reading = parse_codex_usage(payload, FETCHED)
    assert reading.credits is None
    assert reading.windows  # the rate-limit windows are unaffected


def test_codex_banked_resets_show_without_credits():
    payload = _codex_payload(credits={"has_credits": False, "balance": "0"})
    reading = parse_codex_usage(payload, FETCHED)
    assert reading.credits is not None
    assert reading.credits.balance is None
    assert reading.credits.unlimited is False
    assert reading.credits.banked_resets == 2


def test_codex_unparseable_balance_becomes_none():
    payload = _codex_payload(credits={"balance": "not-a-number"})
    reading = parse_codex_usage(payload, FETCHED)
    assert reading.credits is not None
    assert reading.credits.balance is None


def test_codex_limit_reached_sets_detail():
    payload = _codex_payload()
    payload["rate_limit"]["limit_reached"] = True
    assert parse_codex_usage(payload, FETCHED).detail == "limit reached"


def test_codex_clamps_out_of_range_percent():
    payload = _codex_payload()
    payload["rate_limit"]["primary_window"]["used_percent"] = 250
    payload["rate_limit"]["secondary_window"]["used_percent"] = -4
    reading = parse_codex_usage(payload, FETCHED)
    assert [w.used_percent for w in reading.windows] == [100.0, 0.0]


def test_codex_malformed_shape_is_malformed_response():
    reading = parse_codex_usage(["not", "an", "object"], FETCHED)
    assert reading.status == "error"
    assert reading.reason == "malformed_response"


def test_codex_extra_keys_are_ignored():
    reading = parse_codex_usage(_codex_payload(brand_new_field={"x": 1}), FETCHED)
    assert reading.status == "ok"


# --- Claude parser ------------------------------------------------------------------------------


def _claude_payload(**overrides) -> dict:
    payload = {
        "five_hour": {"utilization": 37.0, "resets_at": "2026-10-15T15:00:00+00:00"},
        "seven_day": {"utilization": 8, "resets_at": "2026-10-20T00:00:00Z"},
        "seven_day_opus": None,
        "seven_day_sonnet": {"utilization": 61.5, "resets_at": "2026-10-20T00:00:00+00:00"},
        "extra_usage": {
            "is_enabled": True,
            "monthly_limit": 5000,
            "used_credits": 1234,
            "utilization": 24.68,
            "currency": "USD",
        },
    }
    payload.update(overrides)
    return payload


def test_claude_windows_and_scopes():
    reading = parse_claude_usage(_claude_payload(), FETCHED)
    assert reading.status == "ok"
    labels = [(w.label, w.scope) for w in reading.windows]
    assert labels == [("5h", "account"), ("Week", "account"), ("Week (Sonnet)", "model")]
    assert reading.windows[0].used_percent == 37.0
    assert reading.windows[0].resets_at == datetime(2026, 10, 15, 15, 0, tzinfo=UTC)
    assert reading.windows[0].window_seconds == 18000


def test_claude_extra_usage_converts_minor_units():
    reading = parse_claude_usage(_claude_payload(), FETCHED)
    assert reading.budget is not None
    assert reading.budget.used == pytest.approx(12.34)
    assert reading.budget.limit == pytest.approx(50.00)
    assert reading.budget.currency == "USD"


def test_claude_disabled_extra_usage_has_no_budget():
    payload = _claude_payload(extra_usage={"is_enabled": False, "monthly_limit": 5000, "used_credits": 1})
    assert parse_claude_usage(payload, FETCHED).budget is None


def test_claude_clamps_utilization():
    payload = _claude_payload(five_hour={"utilization": 180, "resets_at": None})
    assert parse_claude_usage(payload, FETCHED).windows[0].used_percent == 100.0


def test_claude_malformed_is_malformed_response():
    reading = parse_claude_usage(_claude_payload(five_hour={"resets_at": "not a date"}), FETCHED)
    assert reading.reason == "malformed_response"


# --- Z.ai parser --------------------------------------------------------------------------------


def _zai_payload(**overrides) -> dict:
    payload = {
        "code": 200,
        "success": True,
        "msg": "ok",
        "data": {
            "planName": "Coding Lite",
            "limits": [
                {
                    "type": "TOKENS_LIMIT",
                    "unit": 3,
                    "number": 5,
                    "percentage": 42,
                    "nextResetTime": 1_781_000_000_000,
                },
                {
                    "type": "TIME_LIMIT",
                    "unit": 5,
                    "number": 1,
                    "percentage": 7,
                    "nextResetTime": "2026-11-01T00:00:00Z",
                },
            ],
        },
    }
    payload.update(overrides)
    return payload


def test_zai_tokens_and_time_limits():
    reading = parse_zai_usage(_zai_payload(), FETCHED)
    assert reading.status == "ok"
    assert reading.plan == "Coding Lite"
    assert [w.label for w in reading.windows] == ["Tokens (5h)", "Monthly"]
    assert reading.windows[0].used_percent == 42.0
    assert reading.windows[0].window_seconds == 5 * 3600
    assert reading.windows[0].resets_at == datetime.fromtimestamp(1_781_000_000, UTC)
    assert reading.windows[1].resets_at == datetime(2026, 11, 1, tzinfo=UTC)


def test_zai_numeric_string_reset_is_epoch_milliseconds():
    payload = _zai_payload()
    payload["data"]["limits"][0]["nextResetTime"] = "1781000000000"
    assert parse_zai_usage(payload, FETCHED).windows[0].resets_at == datetime.fromtimestamp(1_781_000_000, UTC)


def test_zai_plan_falls_back_to_plan_key():
    payload = _zai_payload()
    payload["data"] = {"plan": "Pro", "limits": []}
    assert parse_zai_usage(payload, FETCHED).plan == "Pro"


def test_zai_unsuccessful_payload_never_surfaces_the_provider_msg():
    reading = parse_zai_usage(_zai_payload(success=False, code=401, msg="x" * 500), FETCHED)
    assert reading.status == "error"
    assert reading.reason == "provider_error"
    assert reading.detail == "provider reported an error"


def test_zai_code_not_200_is_error_even_when_success_true():
    reading = parse_zai_usage(_zai_payload(code=500), FETCHED)
    assert reading.reason == "provider_error"


def test_zai_clamps_percentage():
    payload = _zai_payload()
    payload["data"]["limits"][0]["percentage"] = -1
    assert parse_zai_usage(payload, FETCHED).windows[0].used_percent == 0.0


# --- Target discovery ---------------------------------------------------------------------------


def _codex_member(auth_file: str = "/codex-home/auth.json", base_url: str = "https://chatgpt.com/backend-api"):
    impl = MagicMock(spec=CodexLLM)
    impl._auth_file = Path(auth_file)
    impl.base_url = base_url
    member = SimpleNamespace(provider="openai-codex", api_key="", base_url=base_url, _provider_impl=impl)
    return member, impl


def _claude_member(api_key: str = "", provider: str = "claude-code"):
    return SimpleNamespace(provider=provider, api_key=api_key, base_url="", _provider_impl=None)


def _zai_member(base_url: str = "https://api.z.ai/api/coding/paas/v4", api_key: str = ZAI_KEY):
    return SimpleNamespace(provider="zai", api_key=api_key, base_url=base_url, _provider_impl=None)


def _memory(**lanes):
    return SimpleNamespace(**lanes)


def _settings(token: str | None = None) -> QuotaSettings:
    env = {"HINDSIGHT_API_LLM_COST_QUOTA_CLAUDE_OAUTH_TOKEN": token} if token else {}
    return QuotaSettings.from_env(env)


def test_codex_lane_detected_by_isinstance_and_labelled_with_chain_index():
    member, _ = _codex_member()
    chain = MultiLLMProvider.__new__(MultiLLMProvider)
    chain._members = [_zai_member(base_url="https://api.z.ai/api/paas/v4"), member]
    targets = collect_quota_targets(_memory(_retain_llm_config=chain), _settings())
    codex = [t for t in targets if t.provider == "openai-codex"]
    assert len(codex) == 1
    assert codex[0].lanes == ("retain[2]",)
    assert codex[0].fetch is not None


def test_same_codex_auth_file_is_deduplicated_across_lanes():
    member_a, _ = _codex_member("/shared/auth.json")
    member_b, _ = _codex_member("/shared/auth.json")
    targets = collect_quota_targets(_memory(_llm_config=member_a, _reflect_llm_config=member_b), _settings())
    assert len(targets) == 1
    assert targets[0].lanes == ("default", "reflect")


def test_distinct_codex_profiles_stay_separate():
    member_a, _ = _codex_member("/profile-a/auth.json")
    member_b, _ = _codex_member("/profile-b/auth.json")
    chain = MultiLLMProvider.__new__(MultiLLMProvider)
    chain._members = [member_a, member_b]
    targets = collect_quota_targets(_memory(_llm_config=chain), _settings())
    assert [t.lanes for t in targets] == [("default[1]",), ("default[2]",)]


def test_claude_token_order_env_then_lane_key_then_none():
    lane = _claude_member(api_key=LANE_CLAUDE_TOKEN)
    with_env = collect_quota_targets(_memory(_llm_config=lane), _settings(CLAUDE_TOKEN))
    assert with_env[0].provider == "claude-code"
    assert with_env[0].status is None

    with_lane = collect_quota_targets(_memory(_llm_config=lane), _settings())
    assert with_lane[0].status is None
    assert with_lane[0].fetch is not None

    no_token = collect_quota_targets(_memory(_llm_config=_claude_member(api_key="sk-ant-api03-x")), _settings())
    assert no_token[0].status == "unavailable"
    assert no_token[0].reason == "no_credentials"
    assert no_token[0].fetch is None


def test_claude_env_token_and_lane_token_with_same_value_dedupe():
    targets = collect_quota_targets(
        _memory(_llm_config=_claude_member(api_key=CLAUDE_TOKEN), _retain_llm_config=_claude_member()),
        _settings(CLAUDE_TOKEN),
    )
    assert len(targets) == 1
    assert targets[0].lanes == ("default", "retain")


def test_zai_coding_lane_is_fetchable_paas_lane_is_not_a_plan():
    targets = collect_quota_targets(
        _memory(
            _llm_config=_zai_member(),
            _retain_llm_config=_zai_member(base_url="https://api.z.ai/api/paas/v4"),
        ),
        _settings(),
    )
    by_reason = {t.reason: t for t in targets if t.status == "unsupported"}
    assert by_reason["not_a_plan_endpoint"].lanes == ("retain",)
    coding = [t for t in targets if t.fetch is not None]
    assert len(coding) == 1
    assert coding[0].lanes == ("default",)


def test_unsupported_plan_providers_are_listed_and_metered_providers_are_not():
    targets = collect_quota_targets(
        _memory(
            _llm_config=SimpleNamespace(provider="cursor", api_key="", base_url="", _provider_impl=None),
            _retain_llm_config=SimpleNamespace(provider="openai", api_key="sk-x", base_url="", _provider_impl=None),
            _reflect_llm_config=SimpleNamespace(provider="ollama", api_key="", base_url="", _provider_impl=None),
        ),
        _settings(),
    )
    assert [(t.provider, t.status, t.reason) for t in targets] == [("cursor", "unsupported", "no_usage_endpoint")]


def test_absent_lane_attributes_are_skipped():
    assert collect_quota_targets(_memory(), _settings()) == []


# --- Cache --------------------------------------------------------------------------------------


def _counting_fetch(readings: list[QuotaReading]):
    calls = {"n": 0}

    async def fetch() -> QuotaReading:
        calls["n"] += 1
        return readings[min(calls["n"] - 1, len(readings) - 1)]

    return fetch, calls


OK_READING = QuotaReading(status="ok", fetched_at=FETCHED)
ERR_READING = QuotaReading(status="error", reason="http_error", detail="HTTP 500", fetched_at=FETCHED)


async def test_cache_serves_success_within_ttl_then_refetches():
    clock = FakeClock()
    cache = QuotaCache(clock)
    fetch, calls = _counting_fetch([OK_READING])
    await cache.get("k", fetch)
    clock.advance(59)
    await cache.get("k", fetch)
    assert calls["n"] == 1
    clock.advance(2)
    await cache.get("k", fetch)
    assert calls["n"] == 2


async def test_cache_failure_backoff_doubles_to_cap_and_serves_stale_last_good():
    clock = FakeClock()
    cache = QuotaCache(clock)
    fetch, calls = _counting_fetch([OK_READING, ERR_READING])
    await cache.get("k", fetch)
    clock.advance(61)
    stale = await cache.get("k", fetch)  # fetch fails; first backoff is 60 s
    assert calls["n"] == 2
    assert stale.status == "ok"
    assert stale.stale is True
    assert (await cache.get("k", fetch)).stale is True  # inside backoff: no new fetch
    assert calls["n"] == 2

    for _ in range(6):
        clock.advance(1000)  # always past the backoff, so another fetch happens
        before = calls["n"]
        await cache.get("k", fetch)
        assert calls["n"] == before + 1
    # The cap holds: 900 s after the last failure a fetch still happens.
    clock.advance(901)
    before = calls["n"]
    await cache.get("k", fetch)
    assert calls["n"] == before + 1


async def test_cache_backoff_sequence_is_60_120_240_480_900():
    clock = FakeClock()
    cache = QuotaCache(clock)
    fetch, calls = _counting_fetch([ERR_READING])
    expected_gaps = [60, 120, 240, 480, 900, 900]
    await cache.get("k", fetch)
    for gap in expected_gaps:
        clock.advance(gap - 1)
        before = calls["n"]
        await cache.get("k", fetch)
        assert calls["n"] == before, f"refetched before the {gap}s backoff elapsed"
        clock.advance(1)
        await cache.get("k", fetch)
        assert calls["n"] == before + 1


async def test_cache_error_without_last_good_returns_the_error():
    clock = FakeClock()
    cache = QuotaCache(clock)
    fetch, _ = _counting_fetch([ERR_READING])
    result = await cache.get("k", fetch)
    assert result.status == "error"
    assert result.reason == "http_error"


async def test_cache_single_flight_under_concurrency():
    clock = FakeClock()
    cache = QuotaCache(clock)
    calls = {"n": 0}

    async def slow_fetch() -> QuotaReading:
        calls["n"] += 1
        await asyncio.sleep(0)
        return OK_READING

    results = await asyncio.gather(*(cache.get("k", slow_fetch) for _ in range(5)))
    assert calls["n"] == 1
    assert all(r.status == "ok" for r in results)


async def test_cache_keys_are_independent():
    clock = FakeClock()
    cache = QuotaCache(clock)
    fetch_a, calls_a = _counting_fetch([OK_READING])
    fetch_b, calls_b = _counting_fetch([OK_READING])
    await cache.get("a", fetch_a)
    await cache.get("b", fetch_b)
    assert calls_a["n"] == 1
    assert calls_b["n"] == 1


async def test_cache_fetch_exception_becomes_internal_error_reading():
    clock = FakeClock()
    cache = QuotaCache(clock)

    async def boom() -> QuotaReading:
        raise RuntimeError("kaboom")

    result = await cache.get("k", boom)
    assert result.status == "error"
    assert result.reason == "internal_error"


# --- Service: fetch mapping ---------------------------------------------------------------------


def _service(
    memory, responder, clock: FakeClock | None = None, token: str | None = CLAUDE_TOKEN
) -> tuple[LLMQuotaService, FakeHttp]:
    http = FakeHttp(responder)
    service = LLMQuotaService(memory, _settings(token), http_get=http, cache=QuotaCache(clock or FakeClock()))
    return service, http


def _only(response) -> object:
    assert len(response.providers) == 1
    return response.providers[0]


async def test_claude_fetch_maps_success_and_sends_bearer_with_beta_headers():
    service, http = _service(
        _memory(_llm_config=_claude_member()),
        lambda url: _ok(_claude_payload()),
    )
    response = await service.fetch_all("bank-1")
    provider = _only(response)
    assert provider.status == "ok"
    assert provider.lanes == ["default"]
    assert [w.label for w in provider.windows] == ["5h", "Week", "Week (Sonnet)"]
    url, headers = http.calls[0]
    assert url == "https://api.anthropic.com/api/oauth/usage"
    assert headers["Authorization"] == f"Bearer {CLAUDE_TOKEN}"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["anthropic-version"] == "2023-06-01"
    _assert_no_secret(response.model_dump(), response.model_dump_json(), repr(response))


async def test_claude_403_with_user_profile_is_token_scope_unavailable():
    service, _ = _service(
        _memory(_llm_config=_claude_member()),
        lambda url: HttpResult(status=403, body='{"error":"missing user:profile scope"}'),
    )
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.status == "unavailable"
    assert provider.reason == "token_scope"


async def test_claude_401_is_auth_failed():
    service, _ = _service(_memory(_llm_config=_claude_member()), lambda url: HttpResult(status=401, body="nope"))
    provider = _only(await service.fetch_all("bank-1"))
    assert (provider.status, provider.reason) == ("error", "auth_failed")


async def test_plain_403_without_user_profile_is_auth_failed():
    service, _ = _service(_memory(_llm_config=_claude_member()), lambda url: HttpResult(status=403, body="forbidden"))
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "auth_failed"


async def test_server_error_is_http_error_with_status_only_detail():
    service, _ = _service(_memory(_llm_config=_claude_member()), lambda url: HttpResult(status=500, body="secret body"))
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "http_error"
    assert provider.detail == "HTTP 500"
    _assert_no_secret(repr(provider))
    assert "secret body" not in repr(provider)


async def test_invalid_json_is_malformed_response():
    service, _ = _service(_memory(_llm_config=_claude_member()), lambda url: HttpResult(status=200, body="<html>"))
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "malformed_response"


async def test_timeout_is_timeout():
    def responder(url):
        raise TimeoutError()

    service, _ = _service(_memory(_llm_config=_claude_member()), responder)
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "timeout"


async def test_unexpected_transport_error_is_internal_error_and_does_not_raise():
    def responder(url):
        raise ConnectionResetError(f"reset while calling with {CLAUDE_TOKEN}")

    service, _ = _service(_memory(_llm_config=_claude_member()), responder)
    response = await service.fetch_all("bank-1")
    provider = _only(response)
    assert provider.reason == "internal_error"
    _assert_no_secret(repr(response), response.model_dump_json())


async def test_no_credentials_claude_lane_makes_no_request():
    service, http = _service(_memory(_llm_config=_claude_member(api_key="")), lambda url: _ok({}), token=None)
    provider = _only(await service.fetch_all("bank-1"))
    assert (provider.status, provider.reason) == ("unavailable", "no_credentials")
    assert http.calls == []


async def test_zai_coding_fetch_uses_origin_and_bearer():
    service, http = _service(_memory(_llm_config=_zai_member()), lambda url: _ok(_zai_payload()))
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.status == "ok"
    assert provider.plan == "Coding Lite"
    url, headers = http.calls[0]
    assert url == "https://api.z.ai/api/monitor/usage/quota/limit"
    assert headers["Authorization"] == f"Bearer {ZAI_KEY}"
    _assert_no_secret(repr(provider), provider.model_dump_json())


async def test_zai_error_envelope_maps_to_provider_error():
    service, _ = _service(
        _memory(_llm_config=_zai_member()),
        lambda url: _ok(_zai_payload(success=False, code=401, msg="token expired")),
    )
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.status == "error"
    assert provider.reason == "provider_error"
    assert provider.detail == "provider reported an error"


async def test_zai_error_text_that_echoes_the_key_never_reaches_the_response():
    # A Z.ai error envelope can echo the configured key; it must not reach the cached or returned snapshot.
    service, _ = _service(
        _memory(_llm_config=_zai_member()),
        lambda url: _ok(_zai_payload(success=False, code=401, msg=f"Invalid API key {ZAI_KEY}")),
    )
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "provider_error"
    _assert_no_secret(repr(provider), provider.model_dump_json())


async def test_codex_fetch_uses_wham_usage_and_drops_content_type():
    member, impl = _codex_member()
    impl._ensure_fresh_token = AsyncMock()
    impl._build_request_headers = MagicMock(
        return_value={
            "Authorization": f"Bearer {CODEX_TOKEN}",
            "Content-Type": "application/json",
            "ChatGPT-Account-ID": "acct",
        }
    )
    service, http = _service(_memory(_llm_config=member), lambda url: _ok(_codex_payload()))
    response = await service.fetch_all("bank-1")
    provider = _only(response)
    assert provider.status == "ok"
    impl._ensure_fresh_token.assert_awaited_once()
    url, headers = http.calls[0]
    assert url == "https://chatgpt.com/backend-api/wham/usage"
    assert "Content-Type" not in headers
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == f"Bearer {CODEX_TOKEN}"
    _assert_no_secret(response.model_dump_json(), repr(response))


async def test_codex_url_without_backend_api_uses_codex_usage_path():
    member, impl = _codex_member(base_url="https://example.test/codex")
    impl.base_url = "https://example.test/codex"
    impl._ensure_fresh_token = AsyncMock()
    impl._build_request_headers = MagicMock(return_value={"Authorization": "Bearer x"})
    service, http = _service(_memory(_llm_config=member), lambda url: _ok(_codex_payload()))
    await service.fetch_all("bank-1")
    assert http.calls[0][0] == "https://example.test/api/codex/usage"


async def test_codex_refresh_expired_is_auth_failed_without_http():
    member, impl = _codex_member()
    impl._ensure_fresh_token = AsyncMock(side_effect=CodexRefreshExpiredError("expired"))
    service, http = _service(_memory(_llm_config=member), lambda url: _ok(_codex_payload()))
    provider = _only(await service.fetch_all("bank-1"))
    assert provider.reason == "auth_failed"
    assert http.calls == []


async def test_codex_header_build_failure_is_internal_error_without_token_in_output():
    member, impl = _codex_member()
    impl._ensure_fresh_token = AsyncMock()
    impl._build_request_headers = MagicMock(side_effect=RuntimeError(f"bad {CODEX_TOKEN}"))
    service, http = _service(_memory(_llm_config=member), lambda url: _ok(_codex_payload()))
    response = await service.fetch_all("bank-1")
    provider = _only(response)
    assert provider.reason == "internal_error"
    assert http.calls == []
    _assert_no_secret(repr(response), response.model_dump_json())


async def test_unsupported_and_unavailable_targets_are_not_fetched():
    service, http = _service(
        _memory(
            _llm_config=SimpleNamespace(provider="github-copilot", api_key="", base_url="", _provider_impl=None),
            _retain_llm_config=_zai_member(base_url="https://api.z.ai/api/paas/v4"),
        ),
        lambda url: _ok({}),
    )
    response = await service.fetch_all("bank-1")
    assert {p.status for p in response.providers} == {"unsupported"}
    assert {p.reason for p in response.providers} == {"no_usage_endpoint", "not_a_plan_endpoint"}
    assert http.calls == []


async def test_snapshot_result_is_server_scoped_and_never_names_a_secret():
    service, _ = _service(_memory(_llm_config=_claude_member()), lambda url: _ok(_claude_payload()))
    response = await service.fetch_all("bank-xyz")
    assert response.bank_id == "bank-xyz"
    assert response.scope == "server"
    _assert_no_secret(response.model_dump_json())


async def test_service_repeat_call_is_served_from_cache():
    clock = FakeClock()
    service, http = _service(_memory(_llm_config=_claude_member()), lambda url: _ok(_claude_payload()), clock)
    await service.fetch_all("bank-1")
    await service.fetch_all("bank-1")
    assert len(http.calls) == 1


async def test_fetch_all_with_no_lanes_returns_empty_providers():
    service, _ = _service(_memory(), lambda url: _ok({}))
    response = await service.fetch_all("bank-1")
    assert response.providers == []


def test_response_models_reject_bad_status_values():
    with pytest.raises(ValidationError):
        llm_quota.ProviderQuota.model_validate({"provider": "x", "lanes": [], "status": "great"})


async def test_trickling_provider_is_cut_off_by_the_total_deadline():
    """A provider that never finishes (each read inside the per-phase timeout) must not stall /quota."""

    async def never_finishes(url, headers):
        await asyncio.sleep(3600)

    service = LLMQuotaService(
        _memory(_llm_config=_claude_member()),
        _settings(CLAUDE_TOKEN),
        http_get=never_finishes,
        cache=QuotaCache(FakeClock(), fetch_timeout_seconds=0.05),
    )
    first, second = await asyncio.wait_for(
        asyncio.gather(service.fetch_all("bank-1"), service.fetch_all("bank-1")), timeout=2
    )
    assert (_only(first).status, _only(first).reason) == ("error", "timeout")
    # The second page load waited on the same key's lock and got the backed-off reading, not a second hang.
    assert _only(second).reason == "timeout"


async def test_codex_token_refresh_outlives_the_deadline():
    """The deadline must not cancel a Codex refresh mid-rotation: the rotated refresh token would be lost."""
    member, impl = _codex_member()
    refreshed = asyncio.Event()

    async def slow_refresh():
        await asyncio.sleep(0.2)
        refreshed.set()

    impl._ensure_fresh_token = slow_refresh
    impl._build_request_headers = MagicMock(return_value={"Authorization": "Bearer x"})
    http = FakeHttp(lambda url: _ok(_codex_payload()))
    service = LLMQuotaService(
        _memory(_llm_config=member),
        _settings(),
        http_get=http,
        cache=QuotaCache(FakeClock(), fetch_timeout_seconds=0.05),
    )
    provider = _only(await service.fetch_all("bank-1"))
    assert (provider.status, provider.reason) == ("error", "timeout")
    assert not refreshed.is_set()
    await asyncio.wait_for(refreshed.wait(), timeout=2)
    assert http.calls == []  # the probe itself was abandoned; only the refresh finished


async def test_shutdown_drain_waits_for_a_refresh_the_deadline_left_running():
    """Engine shutdown closes the loop's HTTP sessions; a refresh still rotating tokens must finish first."""
    from hindsight_api.engine.llm_quota import _DETACHED_REFRESHES, drain_detached_refreshes

    member, impl = _codex_member()
    refreshed = asyncio.Event()

    async def slow_refresh():
        await asyncio.sleep(0.2)
        refreshed.set()

    impl._ensure_fresh_token = slow_refresh
    impl._build_request_headers = MagicMock(return_value={"Authorization": "Bearer x"})
    service = LLMQuotaService(
        _memory(_llm_config=member),
        _settings(),
        http_get=FakeHttp(lambda url: _ok(_codex_payload())),
        cache=QuotaCache(FakeClock(), fetch_timeout_seconds=0.05),
    )
    assert _only(await service.fetch_all("bank-1")).reason == "timeout"
    assert not refreshed.is_set()
    await drain_detached_refreshes(timeout=2)
    assert refreshed.is_set()
    assert not _DETACHED_REFRESHES


async def test_shutdown_drain_is_bounded_and_leaves_a_stuck_refresh_alone():
    from hindsight_api.engine.llm_quota import drain_detached_refreshes

    member, impl = _codex_member()
    stuck = asyncio.Event()

    async def never_finishes():
        await stuck.wait()

    impl._ensure_fresh_token = never_finishes
    service = LLMQuotaService(
        _memory(_llm_config=member),
        _settings(),
        http_get=FakeHttp(lambda url: _ok(_codex_payload())),
        cache=QuotaCache(FakeClock(), fetch_timeout_seconds=0.05),
    )
    await service.fetch_all("bank-1")
    await asyncio.wait_for(drain_detached_refreshes(timeout=0.05), timeout=2)
    stuck.set()  # let the background task end so the test leaves nothing running
    await drain_detached_refreshes(timeout=2)

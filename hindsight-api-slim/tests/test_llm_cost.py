"""Fork LLM cost accounting (fork: cost-accounting): deterministic pricing, settings and adapter tests.

Every expected value is a hand calculation kept next to its constant. Rates are genai-prices
0.1.12 USD per million tokens (MTok) and the formula is written in each comment, so a pin bump that
moves a price fails here loudly instead of silently.
"""

import logging
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from anthropic.types import CacheCreation, Usage
from openai.types import CompletionUsage

import hindsight_api
from hindsight_api.config import llm_cost_env
from hindsight_api.engine import llm_cost
from hindsight_api.engine.llm_cost import (
    FORK_USAGE_KWARG,
    CallUsage,
    CostSettings,
    CustomPrice,
    ModelKey,
    RowUsage,
    StoredUsage,
    anthropic_cache_write_kwargs,
    claude_code_usage_kwargs,
    codex_usage_kwargs,
    load_cost_settings,
    note_codex_usage,
    openai_compatible_usage_kwargs,
    openai_responses_usage_kwargs,
    parse_timezone,
    price_row,
    read_stored_usage,
    summarize_costs,
    trace_usage_info,
    xai_oauth_usage_kwargs,
)
from tests.fork_helpers import cost_env_cleared

PACKAGE_ROOT = Path(hindsight_api.__file__).parent
ENV_PREFIX = "HINDSIGHT_API_LLM_COST_"
JAN_1 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
DEFAULT = CostSettings.from_env({})
# Approx tolerance for values the response models round to 8 decimals.
TOL = 1e-9


@pytest.fixture(autouse=True)
def _clean_cost_env(monkeypatch):
    """No feature variable leaks in from the shell, and the settings cache is fresh for every test."""
    yield from cost_env_cleared(monkeypatch)


def _marked(**fields) -> StoredUsage:
    """A row marker as a v2 writer leaves it."""
    return StoredUsage(v=2, **fields)


def _row(**overrides) -> RowUsage:
    fields = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "started_at": JAN_1,
        "input_tokens": 2010,
        "output_tokens": 300,
        "cached_tokens": 0,
        "thoughts_tokens": 0,
        "usage": _marked(),
    }
    fields.update(overrides)
    return RowUsage(**fields)


def _enable(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")


def _usage_of(kwargs: dict) -> CallUsage:
    return kwargs[FORK_USAGE_KWARG]


def _completion(**usage_fields) -> SimpleNamespace:
    usage = CompletionUsage.model_validate(
        {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105, **usage_fields}
    )
    return SimpleNamespace(usage=usage)


# --- Settings: defaults and validation ---------------------------------------------------------


def test_defaults_when_nothing_is_set():
    assert DEFAULT.enabled is False
    assert DEFAULT.billing_overrides == {}
    assert DEFAULT.legacy_before is None
    assert llm_cost.DEFAULT_BILLING["openai-codex"] == "plan"
    assert llm_cost.DEFAULT_BILLING["xai-oauth"] == "plan"
    assert llm_cost.DEFAULT_BILLING["claude-code"] == "plan"
    assert llm_cost.DEFAULT_BILLING["github-copilot"] == "plan"
    assert llm_cost.DEFAULT_BILLING["cursor"] == "plan"
    assert llm_cost.DEFAULT_BILLING["nous"] == "plan"
    assert llm_cost.DEFAULT_BILLING["zai"] == "plan"  # the default base URL is the Coding Plan endpoint
    for local in ("ollama", "lmstudio", "llamacpp", "mock", "none"):
        assert llm_cost.DEFAULT_BILLING[local] == "local"
    assert "anthropic" not in llm_cost.DEFAULT_BILLING  # metered by default
    assert llm_cost.DEFAULT_PROVIDER_MAP["openai-responses"] == "openai"
    assert dict(DEFAULT.provider_map) == llm_cost.DEFAULT_PROVIDER_MAP
    assert dict(DEFAULT.model_aliases) == {}
    assert dict(DEFAULT.custom_prices) == {}


def test_blank_values_mean_unset():
    settings = CostSettings.from_env(
        {
            "HINDSIGHT_API_LLM_COST_ENABLED": "   ",
            "HINDSIGHT_API_LLM_COST_BILLING": "",
            "HINDSIGHT_API_LLM_COST_PROVIDER_MAP": " ",
            "HINDSIGHT_API_LLM_COST_MODEL_ALIASES": "",
            "HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": "  ",
            "HINDSIGHT_API_LLM_COST_LEGACY_BEFORE": "  ",
        }
    )
    assert settings == DEFAULT


@pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", " Yes "])
def test_enabled_accepts_truthy_spellings(raw):
    assert CostSettings.from_env({"HINDSIGHT_API_LLM_COST_ENABLED": raw}).enabled is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "NO"])
def test_enabled_accepts_falsy_spellings(raw):
    assert CostSettings.from_env({"HINDSIGHT_API_LLM_COST_ENABLED": raw}).enabled is False


@pytest.mark.parametrize(
    ("variable", "raw"),
    [
        ("HINDSIGHT_API_LLM_COST_ENABLED", "maybe"),
        ("HINDSIGHT_API_LLM_COST_BILLING", "zai"),  # no '='
        ("HINDSIGHT_API_LLM_COST_BILLING", "zai="),  # no kind
        ("HINDSIGHT_API_LLM_COST_BILLING", "zai=plan,,ollama=local"),  # empty entry
        ("HINDSIGHT_API_LLM_COST_BILLING", "zai=bogus"),  # unknown kind
        ("HINDSIGHT_API_LLM_COST_BILLING", "zai=plan,zai=local"),  # duplicate provider
        ("HINDSIGHT_API_LLM_COST_PROVIDER_MAP", "zai=no-such-provider"),
        ("HINDSIGHT_API_LLM_COST_PROVIDER_MAP", "zai"),
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", "not json"),
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", "[]"),  # not an object
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", '{"xai/grok-4.7-build-fast": 1}'),  # non-string value
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", '{"nosplash": "x-ai/grok-4.7"}'),  # key without '/'
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", '{"xai/grok-4.7-build-fast": "x-ai"}'),  # target without '/'
        ("HINDSIGHT_API_LLM_COST_MODEL_ALIASES", '{"xai/grok-4.7-build-fast": "x-ai/no-such-model"}'),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", "not json"),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": {"input_mtok": 1}}'),  # missing output
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": {"input_mtok": 1, "output_mtok": 1, "x": 2}}'),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": {"input_mtok": -1, "output_mtok": 1}}'),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": {"input_mtok": true, "output_mtok": 1}}'),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": {"input_mtok": NaN, "output_mtok": 1}}'),
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"zai/glm-5.3": 3}'),  # price is not an object
        ("HINDSIGHT_API_LLM_COST_CUSTOM_PRICES", '{"nosplash": {"input_mtok": 1, "output_mtok": 1}}'),
        ("HINDSIGHT_API_LLM_COST_LEGACY_BEFORE", "2026-10-01T00:00:00"),  # naive: no UTC offset
        ("HINDSIGHT_API_LLM_COST_LEGACY_BEFORE", "last tuesday"),
    ],
)
def test_bad_value_raises_naming_its_variable(variable, raw):
    with pytest.raises(ValueError, match=variable):
        CostSettings.from_env({variable: raw})


def test_unknown_cost_key_is_rejected_by_name_without_its_value():
    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_COST_ENABELD") as info:
        CostSettings.from_env({"HINDSIGHT_API_LLM_COST_ENABELD": "secret-value-xyz"})
    assert "secret-value-xyz" not in str(info.value)


def test_quota_sub_prefix_is_not_rejected_by_cost_settings():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_QUOTA_ENABLED": "true"})
    assert settings == DEFAULT


def test_legacy_before_is_an_aware_timestamp():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_LEGACY_BEFORE": "2026-10-01T13:00:00Z"})
    assert settings.legacy_before == datetime(2026, 10, 1, 13, 0, tzinfo=UTC)


def test_valid_overrides_merge_over_defaults():
    settings = CostSettings.from_env(
        {
            "HINDSIGHT_API_LLM_COST_BILLING": "zai=metered, ollama=plan",
            "HINDSIGHT_API_LLM_COST_PROVIDER_MAP": "vertexai=google",
            "HINDSIGHT_API_LLM_COST_MODEL_ALIASES": '{"xai/grok-4.7-build-fast": "x-ai/grok-4.7"}',
            "HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": (
                '{"zai/glm-5.3": {"input_mtok": 1.5, "output_mtok": 2, "cache_read_mtok": 0.1}}'
            ),
        }
    )
    assert settings.billing_overrides == {"zai": "metered", "ollama": "plan"}
    assert settings.provider_map["vertexai"] == "google"
    assert settings.provider_map["anthropic"] == "anthropic"
    assert settings.model_aliases == {
        ModelKey("xai", "grok-4.7-build-fast"): ModelKey("x-ai", "grok-4.7"),
    }
    assert settings.custom_prices == {
        ModelKey("zai", "glm-5.3"): CustomPrice(
            input_mtok=Decimal("1.5"),
            output_mtok=Decimal("2"),
            cache_read_mtok=Decimal("0.1"),
        ),
    }


def test_model_key_splits_on_the_first_slash():
    settings = CostSettings.from_env(
        {"HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": '{"openrouter/meta/llama-3": {"input_mtok": 1, "output_mtok": 1}}'}
    )
    assert ModelKey("openrouter", "meta/llama-3") in settings.custom_prices


def test_load_cost_settings_reads_only_through_config_accessor(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "yes")
    assert load_cost_settings().enabled is True
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "no")
    assert load_cost_settings().enabled is True  # cached until cleared
    load_cost_settings.cache_clear()
    assert load_cost_settings().enabled is False


def test_load_cost_settings_failure_is_not_cached(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "maybe")
    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_COST_ENABLED"):
        load_cost_settings()
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    assert load_cost_settings().enabled is True


@pytest.mark.parametrize("raw", ["UTC", "utc", "Utc"])
def test_parse_timezone_utc_in_any_case_needs_no_database(raw):
    assert parse_timezone(raw) is UTC


def test_parse_timezone_accepts_iana_names():
    assert parse_timezone("Asia/Taipei") == ZoneInfo("Asia/Taipei")


@pytest.mark.parametrize("raw", ["Mars/Base", ""])
def test_parse_timezone_rejects_unknown_names_with_value_error(raw):
    with pytest.raises(ValueError, match="IANA"):
        parse_timezone(raw)


# --- Write path: adapters ----------------------------------------------------------------------


def test_anthropic_kwargs_with_ttl_split(monkeypatch):
    _enable(monkeypatch)
    response = SimpleNamespace(
        usage=Usage(
            input_tokens=10,
            output_tokens=3,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=2000,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=1500, ephemeral_1h_input_tokens=500),
        )
    )
    usage = _usage_of(anthropic_cache_write_kwargs(response))
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (2000, 1500, 500)
    assert usage.usage_source == "provider"


def test_anthropic_kwargs_without_cache_creation_count_writes_as_5m(monkeypatch):
    _enable(monkeypatch)
    response = SimpleNamespace(usage=Usage(input_tokens=10, output_tokens=3, cache_creation_input_tokens=2000))
    usage = _usage_of(anthropic_cache_write_kwargs(response))
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (2000, 2000, 0)


def test_anthropic_kwargs_none_fields_become_known_zeros(monkeypatch):
    _enable(monkeypatch)
    # An uncached call: the SDK sends None for the cache fields.
    response = SimpleNamespace(
        usage=Usage(input_tokens=10, output_tokens=3, cache_creation_input_tokens=None, cache_creation=None)
    )
    usage = _usage_of(anthropic_cache_write_kwargs(response))
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (0, 0, 0)


def test_anthropic_kwargs_magicmock_usage_with_real_input_tokens_gives_zeros(monkeypatch):
    # Upstream-test compatibility: MagicMock responses from older tests still get a known-zero breakdown.
    _enable(monkeypatch)
    response = MagicMock()
    response.usage = MagicMock(input_tokens=5)
    usage = _usage_of(anthropic_cache_write_kwargs(response))
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (0, 0, 0)


def test_anthropic_kwargs_empty_when_accounting_is_off():
    # Off means no extraction at all: the provider hook then adds nothing to the trace call.
    response = SimpleNamespace(usage=Usage(input_tokens=10, output_tokens=3, cache_creation_input_tokens=2000))
    assert anthropic_cache_write_kwargs(response) == {}


def test_kwargs_empty_and_warn_once_when_settings_are_invalid(monkeypatch, caplog):
    monkeypatch.setattr(llm_cost, "_settings_error_logged", False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "maybe")
    response = SimpleNamespace(usage=Usage(input_tokens=10, output_tokens=3, cache_creation_input_tokens=2000))
    with caplog.at_level(logging.WARNING, logger=llm_cost.logger.name):
        assert anthropic_cache_write_kwargs(response) == {}  # the provider call path must not raise
        assert anthropic_cache_write_kwargs(response) == {}
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1  # a broken setting is reported once per process, not on every provider call
    assert "HINDSIGHT_API_LLM_COST_ENABLED" in warnings[0].getMessage()


def test_anthropic_kwargs_empty_when_usage_is_missing_or_unusable(monkeypatch):
    _enable(monkeypatch)
    assert anthropic_cache_write_kwargs(SimpleNamespace(usage=None)) == {}
    assert anthropic_cache_write_kwargs(object()) == {}
    assert anthropic_cache_write_kwargs(MagicMock()) == {}  # usage.input_tokens is a MagicMock
    assert anthropic_cache_write_kwargs(SimpleNamespace(usage=SimpleNamespace(input_tokens=True))) == {}


def test_claude_code_dict_usage_is_provider_counts(monkeypatch):
    _enable(monkeypatch)
    raw = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 3000,
        "output_tokens": 50,
        "cache_creation": {"ephemeral_5m_input_tokens": 150, "ephemeral_1h_input_tokens": 50},
    }
    usage = _usage_of(claude_code_usage_kwargs(raw))
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (200, 150, 50)
    assert usage.usage_source == "provider"
    assert usage.provider_cost_usd is None  # the CLI's total_cost_usd is an estimate and is never recorded


def test_claude_code_without_usage_is_an_estimate(monkeypatch):
    _enable(monkeypatch)
    usage = _usage_of(claude_code_usage_kwargs(None))
    assert usage.usage_source == "estimate"
    assert usage.cache_write_tokens is None
    assert usage.provider_cost_usd is None


def test_claude_code_unreadable_usage_and_cost_fall_back_to_estimate(monkeypatch):
    _enable(monkeypatch)
    usage = _usage_of(claude_code_usage_kwargs({"input_tokens": "lots"}))
    assert usage.usage_source == "estimate"


def _codex_event(cache_write_tokens: int) -> dict:
    """A raw Codex ``response.completed`` event reporting ``cache_write_tokens``."""
    return {"response": {"usage": {"input_tokens_details": {"cache_write_tokens": cache_write_tokens}}}}


def test_codex_note_is_tied_to_the_stashed_object(monkeypatch):
    _enable(monkeypatch)
    stash = SimpleNamespace(input_tokens=100000, cached_tokens=10000)
    note_codex_usage(stash, _codex_event(20000))
    usage = _usage_of(codex_usage_kwargs(stash))
    assert usage.cache_write_tokens == 20000
    assert usage.usage_source == "provider"
    # A different object (a stale or unrelated stash) never receives the breakdown.
    other = _usage_of(codex_usage_kwargs(SimpleNamespace(input_tokens=1, cached_tokens=0)))
    assert other.cache_write_tokens is None
    assert other.usage_source == "estimate"


@pytest.mark.parametrize(
    "details, expected",
    [
        ({"cached_tokens": 10, "cache_write_tokens": 0}, 0),
        ({"cached_tokens": 10, "cache_write_tokens": 7}, 7),
        ({"cached_tokens": 10}, None),
        ({"cached_tokens": 10, "cache_write_tokens": "7"}, None),
    ],
)
def test_codex_cache_writes_tells_a_reported_zero_from_none_reported(details, expected):
    event = {"type": "response.completed", "response": {"usage": {"input_tokens": 50, "input_tokens_details": details}}}
    assert llm_cost._codex_cache_writes(event) == expected
    assert llm_cost._codex_cache_writes({"response": "not-a-dict"}) is None


def test_codex_cache_writes_ignores_malformed_sibling_fields():
    # A malformed sibling block does not hide a reported count, and a negative count reads as zero.
    for sibling in (None, "x", 3):
        usage = {"input_tokens": 50, "cost_details": sibling, "input_tokens_details": {"cache_write_tokens": 11}}
        assert llm_cost._codex_cache_writes({"response": {"usage": usage}}) == 11
    usage = {"input_tokens": 50, "input_tokens_details": {"cache_write_tokens": -2}}
    assert llm_cost._codex_cache_writes({"response": {"usage": usage}}) == 0


def test_codex_cache_writes_reads_the_explicit_zero_the_chatgpt_backend_sends():
    """Shape of a real ``response.completed`` usage block from the ChatGPT backend (gpt-6.1-sol, 2026-10-20).

    The backend sends ``cache_write_tokens: 0`` explicitly, also on the call that created the cache, so the
    row records a reported zero (a breakdown), not a missing one.
    """
    usage = {
        "input_tokens": 6841,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 6846,
        "attribution": {"request_fields": {"instructions": {"input_tokens": 6826, "cache_write_tokens": 0}}},
    }
    assert llm_cost._codex_cache_writes({"type": "response.completed", "response": {"usage": usage}}) == 0


def test_codex_note_is_a_no_op_with_the_feature_off():
    stash = SimpleNamespace(input_tokens=1, cached_tokens=0)
    note_codex_usage(stash, _codex_event(99))
    assert codex_usage_kwargs(stash) == {}


def test_codex_gpt_6_1_sol_write_is_priced_at_the_write_rate(monkeypatch):
    # uncached 70000 x 2.00 + cache read 10000 x 0.10 + cache write 20000 x 2.50 + output 100 x 10.00, per MTok
    # = 0.140000 + 0.001000 + 0.050000 + 0.001000 = 0.192
    _enable(monkeypatch)
    stash = SimpleNamespace(input_tokens=100000, cached_tokens=10000)
    note_codex_usage(stash, _codex_event(20000))
    info = trace_usage_info(codex_usage_kwargs(stash))
    row = RowUsage(
        provider="openai-codex",
        model="gpt-6.1-sol",
        started_at=JAN_1,
        input_tokens=100000,
        output_tokens=100,
        cached_tokens=10000,
        thoughts_tokens=0,
        usage=read_stored_usage(info["usage"]),
    )
    cost = price_row(row, DEFAULT)
    assert cost.billing == "plan"
    assert cost.list_price_usd == pytest.approx(0.192, abs=TOL)
    assert cost.has_cache_write_breakdown is True


def test_openai_compatible_xai_ticks_are_provider_cost(monkeypatch):
    _enable(monkeypatch)
    response = _completion(cost_in_usd_ticks=37756000)  # 37756000 / 1e10 = 0.0037756 USD
    usage = _usage_of(openai_compatible_usage_kwargs("openai", "https://api.x.ai/v1", response))
    assert usage.provider_cost_usd == pytest.approx(0.0037756, abs=TOL)
    assert usage.usage_source == "provider"
    assert usage.billing_hint is None


def test_openai_compatible_openrouter_cost_and_byok_upstream_cost(monkeypatch):
    _enable(monkeypatch)
    plain = _usage_of(openai_compatible_usage_kwargs("openrouter", None, _completion(cost=0.0042)))
    assert plain.provider_cost_usd == pytest.approx(0.0042, abs=TOL)
    # BYOK: `cost` is only OpenRouter's fee; the upstream inference cost is added.
    byok = _completion(cost=0.001, is_byok=True, cost_details={"upstream_inference_cost": 0.02})
    assert _usage_of(openai_compatible_usage_kwargs("openrouter", None, byok)).provider_cost_usd == pytest.approx(
        0.021, abs=TOL
    )


def test_openai_compatible_cache_write_fallbacks(monkeypatch):
    _enable(monkeypatch)
    details = _completion(prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 300})
    assert _usage_of(openai_compatible_usage_kwargs("openrouter", None, details)).cache_write_tokens == 300
    top = _completion(cache_creation_input_tokens=125)
    assert _usage_of(openai_compatible_usage_kwargs("openai", None, top)).cache_write_tokens == 125
    zero = _completion(prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0})
    assert _usage_of(openai_compatible_usage_kwargs("openrouter", None, zero)).cache_write_tokens == 0
    plain = _usage_of(openai_compatible_usage_kwargs("openai", None, _completion()))
    assert plain.cache_write_tokens is None  # no write field sent: no breakdown, not "no writes"
    assert "cache_write_tokens" not in plain.to_llm_info()["usage"]


def test_openai_responses_without_a_cache_write_field_has_no_breakdown(monkeypatch):
    _enable(monkeypatch)
    absent = SimpleNamespace(usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=10)))
    assert _usage_of(openai_responses_usage_kwargs(absent)).cache_write_tokens is None
    zero = SimpleNamespace(usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cache_write_tokens=0)))
    assert _usage_of(openai_responses_usage_kwargs(zero)).cache_write_tokens == 0


@pytest.mark.parametrize(
    ("base_url", "hint"),
    [
        ("https://api.z.ai/api/coding/paas/v4", "plan"),
        ("https://api.z.ai/api/paas/v4", "metered"),
        (None, None),
    ],
)
def test_zai_billing_hint_follows_the_endpoint(monkeypatch, base_url, hint):
    _enable(monkeypatch)
    usage = _usage_of(openai_compatible_usage_kwargs("zai", base_url, _completion()))
    assert usage.billing_hint == hint


def test_billing_hint_is_only_written_for_zai(monkeypatch):
    _enable(monkeypatch)
    usage = _usage_of(openai_compatible_usage_kwargs("openai", "https://api.z.ai/api/coding/paas/v4", _completion()))
    assert usage.billing_hint is None


def test_openai_compatible_without_usage_is_an_estimate(monkeypatch):
    _enable(monkeypatch)
    usage = _usage_of(openai_compatible_usage_kwargs("openai", None, SimpleNamespace(usage=None)))
    assert usage.usage_source == "estimate"


def test_openai_responses_cache_write_and_ticks(monkeypatch):
    _enable(monkeypatch)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens_details=SimpleNamespace(cache_write_tokens=300),
            cost_in_usd_ticks=10_000_000_000,  # 1 USD
        )
    )
    usage = _usage_of(openai_responses_usage_kwargs(response))
    assert usage.cache_write_tokens == 300
    assert usage.provider_cost_usd == pytest.approx(1.0, abs=TOL)


def test_xai_oauth_ticks_are_provider_cost(monkeypatch):
    _enable(monkeypatch)
    usage = _usage_of(xai_oauth_usage_kwargs(37756000))
    assert usage.provider_cost_usd == pytest.approx(0.0037756, abs=TOL)
    assert usage.cache_write_tokens == 0
    assert _usage_of(xai_oauth_usage_kwargs("x")).provider_cost_usd is None


def test_every_adapter_is_inert_without_env():
    response = _completion(cost=1.0, cost_in_usd_ticks=10**10)
    adapters = [
        lambda: anthropic_cache_write_kwargs(SimpleNamespace(usage=Usage(input_tokens=1, output_tokens=1))),
        lambda: claude_code_usage_kwargs({"input_tokens": 1}),
        lambda: codex_usage_kwargs(object()),
        lambda: openai_compatible_usage_kwargs("zai", "https://api.z.ai/api/coding/paas/v4", response),
        lambda: openai_responses_usage_kwargs(response),
        lambda: xai_oauth_usage_kwargs(10**10),
    ]
    for adapter in adapters:
        assert adapter() == {}
    assert trace_usage_info({FORK_USAGE_KWARG: CallUsage(cache_write_tokens=1)}) == {}
    assert trace_usage_info({}) == {}


@pytest.mark.parametrize("garbage", [None, "text", 42, MagicMock(), object()])
def test_adapters_never_raise_on_garbage(monkeypatch, garbage):
    _enable(monkeypatch)
    anthropic_cache_write_kwargs(garbage)
    claude_code_usage_kwargs(garbage)
    codex_usage_kwargs(garbage)
    openai_compatible_usage_kwargs("zai", garbage, garbage)
    openai_compatible_usage_kwargs("openai", None, SimpleNamespace(usage=garbage))
    openai_responses_usage_kwargs(SimpleNamespace(usage=garbage))
    xai_oauth_usage_kwargs(garbage)
    note_codex_usage(garbage, _codex_event(1))


# --- Write path: the marker ---------------------------------------------------------------------


def test_trace_usage_info_writes_a_v2_marker_for_every_row(monkeypatch):
    _enable(monkeypatch)
    assert trace_usage_info({"thoughts_tokens": 4}) == {"usage": {"v": 2}}  # no breakdown, still marked
    assert trace_usage_info(
        {
            FORK_USAGE_KWARG: CallUsage(
                cache_write_tokens=2000, cache_write_5m_tokens=1500, cache_write_1h_tokens=500, usage_source="provider"
            )
        }
    ) == {
        "usage": {
            "v": 2,
            "cache_write_tokens": 2000,
            "cache_write_5m_tokens": 1500,
            "cache_write_1h_tokens": 500,
            "usage_source": "provider",
        }
    }


def test_trace_usage_info_keeps_a_reported_zero(monkeypatch):
    _enable(monkeypatch)
    usage = trace_usage_info({FORK_USAGE_KWARG: CallUsage(cache_write_tokens=0)})["usage"]
    assert usage == {"v": 2, "cache_write_tokens": 0, "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}


def test_trace_usage_info_never_raises_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(llm_cost, "_settings_error_logged", False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "maybe")
    extra = {FORK_USAGE_KWARG: CallUsage(cache_write_tokens=3000)}
    with caplog.at_level(logging.WARNING, logger=llm_cost.logger.name):
        assert trace_usage_info(extra) == {}
        assert trace_usage_info(extra) == {}
        assert trace_usage_info(extra) == {}
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "HINDSIGHT_API_LLM_COST_ENABLED" in warnings[0].getMessage()


def test_call_usage_dumps_the_persisted_shape_in_order():
    usage = CallUsage(
        cache_write_tokens=2000,
        cache_write_5m_tokens=1500,
        cache_write_1h_tokens=500,
        provider_cost_usd=0.5,
        billing_hint="plan",
        usage_source="provider",
    )
    info = usage.to_llm_info()["usage"]
    assert list(info) == [
        "v",
        "cache_write_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
        "provider_cost_usd",
        "billing_hint",
        "usage_source",
    ]
    assert info["v"] == 2


def test_call_usage_omits_unknown_fields_instead_of_writing_zeros():
    info = CallUsage(usage_source="estimate").to_llm_info()
    assert info == {"usage": {"v": 2, "usage_source": "estimate"}}


def test_call_usage_counts_bad_values_as_absent_and_ignores_unknown_keys():
    usage = CallUsage.model_validate(
        {
            "cache_write_tokens": "lots",
            "cache_write_5m_tokens": None,
            "cache_write_1h_tokens": True,
            "provider_cost_usd": float("nan"),
            "billing_hint": "bogus",
            "usage_source": "guess",
            "extra": 9,
        }
    )
    assert usage == CallUsage()  # None (no breakdown) is not a 0, and nothing else survives
    assert usage.cache_write_tokens is None


def test_call_usage_defaults_five_minute_writes_to_the_total_without_a_split():
    usage = CallUsage(cache_write_tokens=700)
    assert usage.cache_write_5m_tokens == 700
    assert usage.cache_write_1h_tokens == 0


def test_call_usage_one_hour_only_has_no_negative_five_minute_write():
    usage = CallUsage(cache_write_1h_tokens=500)
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (500, 0, 500)


def test_call_usage_negative_five_minute_is_clamped():
    usage = CallUsage(cache_write_tokens=100, cache_write_5m_tokens=-5, cache_write_1h_tokens=None)
    assert (usage.cache_write_tokens, usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (100, 100, 0)


def test_call_usage_provider_cost_must_be_finite_and_non_negative():
    assert CallUsage(provider_cost_usd=-1).provider_cost_usd is None
    assert CallUsage(provider_cost_usd=float("inf")).provider_cost_usd is None
    assert CallUsage(provider_cost_usd=0.25).provider_cost_usd == 0.25


# --- Read path: the stored marker ---------------------------------------------------------------


def test_read_stored_usage_without_a_marker_is_none():
    assert read_stored_usage(None) is None
    assert read_stored_usage("not an object") is None


def test_stored_usage_reads_field_by_field():
    stored = read_stored_usage(
        {
            "v": 2,
            "cache_write_tokens": {"nested": 1},
            "provider_cost_usd": "1.0",
            "billing_hint": "bogus",
            "usage_source": "provider",
        }
    )
    assert stored is not None
    assert stored.v == 2
    assert stored.cache_write_tokens is None
    assert stored.provider_cost_usd is None
    assert stored.billing_hint is None
    assert stored.usage_source == "provider"


def test_stored_usage_reads_a_v1_marker():
    stored = read_stored_usage(
        {"v": 1, "cache_write_tokens": 40, "cache_write_5m_tokens": 40, "cache_write_1h_tokens": 0}
    )
    assert stored is not None
    assert (stored.v, stored.cache_write_tokens, stored.cache_write_1h_tokens) == (1, 40, 0)


def test_stored_usage_normalises_on_read():
    assert StoredUsage.model_validate({"cache_write_1h_tokens": 500}).cache_write_tokens == 500
    assert StoredUsage.model_validate({"cache_write_1h_tokens": 500}).cache_write_5m_tokens == 0
    partial = StoredUsage.model_validate({"cache_write_tokens": 100, "cache_write_1h_tokens": 300})
    assert (partial.cache_write_tokens, partial.cache_write_5m_tokens, partial.cache_write_1h_tokens) == (300, 0, 300)


def test_row_usage_normalises_a_naive_started_at_to_utc():
    row = RowUsage(
        provider="anthropic",
        model="claude-haiku-5-5",
        started_at=datetime(2026, 10, 1, 12, 0),
        input_tokens=10,
        output_tokens=1,
        cached_tokens=0,
        thoughts_tokens=0,
    )
    assert row.started_at == datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


# --- price_row: hand calculations --------------------------------------------------------------


def test_haiku_cache_write_call():
    # uncached 10 x 0.10 + cache write 2000 x 0.125 + output 300 x 0.50, all per MTok
    # = 0.000001 + 0.000250 + 0.000150 = 0.000401
    row = _row(
        input_tokens=2010,
        output_tokens=300,
        usage=_marked(cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0),
    )
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.000401, abs=TOL)
    assert cost.list_price_usd == pytest.approx(0.000401, abs=TOL)
    assert cost.billing == "metered"
    assert cost.reason is None
    assert cost.priced_as == "anthropic/claude-haiku-5-5"
    assert cost.cost_status == "estimated"
    assert cost.cost_source == "genai-prices"
    assert cost.has_cache_write_breakdown is True
    assert cost.legacy_input_repaired is False


def test_haiku_cache_read_call():
    # uncached 10 x 0.10 + cache read 2000 x 0.01 + output 300 x 0.50, all per MTok
    # = 0.000001 + 0.000020 + 0.000150 = 0.000171
    row = _row(input_tokens=2010, output_tokens=300, cached_tokens=2000)
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.000171, abs=TOL)
    assert cost.has_cache_write_breakdown is False


def test_haiku_one_hour_cache_write():
    # 2000 x 0.20 (1h write rate) = 0.000400
    row = _row(
        input_tokens=2000,
        output_tokens=0,
        usage=_marked(cache_write_tokens=2000, cache_write_5m_tokens=0, cache_write_1h_tokens=2000),
    )
    assert price_row(row, DEFAULT).cost_usd == pytest.approx(0.0004, abs=TOL)


def test_haiku_long_context_tier():
    # input 150000 x 0.50 (>100k tier) + output 1000 x 2.50 (>100k tier), per MTok
    # = 0.075000 + 0.002500 = 0.0775
    row = _row(input_tokens=150000, output_tokens=1000)
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.0775, abs=TOL)


def test_gpt_codex_is_plan_with_a_list_price():
    # list = (10000 - 4000) x 2.00 + 4000 x 0.10 + (500 + 100) x 10.00, all per MTok
    # = 0.012000 + 0.000400 + 0.006000 = 0.0184
    row = _row(
        provider="openai-codex",
        model="gpt-6.1-sol",
        input_tokens=10000,
        output_tokens=500,
        cached_tokens=4000,
        thoughts_tokens=100,
    )
    cost = price_row(row, DEFAULT)
    assert cost.billing == "plan"
    assert cost.cost_usd is None
    assert cost.reason == "subscription_plan"
    assert cost.cost_status == "included"
    assert cost.cost_source == "genai-prices"
    assert cost.list_price_usd == pytest.approx(0.0184, abs=TOL)
    assert cost.priced_as == "openai/gpt-6.1-sol"


def test_plan_provider_keeps_the_quote_reason_for_an_unknown_model():
    cost = price_row(_row(provider="openai-codex", model="nope-model"), DEFAULT)
    assert cost.billing == "plan"
    assert cost.cost_usd is None
    assert cost.list_price_usd is None
    assert cost.reason == "unknown_model"
    assert cost.priced_as is None


def test_plan_provider_without_an_alias_reports_unknown_model():
    # xai-oauth needs an alias for grok-4.7-build-fast; without one the row is unknown, not a plan row.
    cost = price_row(_row(provider="xai-oauth", model="grok-4.7-build-fast", input_tokens=100), DEFAULT)
    assert cost.billing == "plan"
    assert cost.reason == "unknown_model"


def test_plan_provider_summary_reports_the_unknown_model_reason():
    rows = [_row(provider="openai-codex", model="nope-model", started_at=JAN_1)]
    summary = summarize_costs(rows, DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)
    (group,) = summary.models
    assert group.unpriced_calls == 1
    assert group.unpriced_reason == "unknown_model"


def test_plan_provider_list_price_falls_back_to_the_provider_cost():
    row = _row(
        provider="xai-oauth",
        model="grok-unlisted",
        usage=_marked(provider_cost_usd=0.4, usage_source="provider"),
    )
    cost = price_row(row, DEFAULT)
    assert cost.billing == "plan"
    assert cost.cost_usd is None
    assert cost.cost_status == "included"
    assert cost.cost_source == "provider"
    assert cost.list_price_usd == pytest.approx(0.4, abs=TOL)
    assert cost.provider_cost_usd == pytest.approx(0.4, abs=TOL)
    assert cost.reason == "subscription_plan"


def test_zai_glm_is_plan_by_default_without_a_hint():
    # Historical zai rows carry no hint; the default base URL is the Coding Plan endpoint.
    cost = price_row(_row(provider="zai", model="glm-5.3", input_tokens=1000, output_tokens=100), DEFAULT)
    assert cost.billing == "plan"
    assert cost.cost_usd is None
    assert cost.cost_status == "included"
    assert cost.list_price_usd == pytest.approx(0.00184, abs=TOL)  # 1000 x 1.40 + 100 x 4.40, per MTok


def test_zai_metered_hint_prices_as_metered():
    row = _row(
        provider="zai",
        model="glm-5.3",
        input_tokens=1000,
        output_tokens=100,
        usage=_marked(billing_hint="metered"),
    )
    cost = price_row(row, DEFAULT)
    assert cost.billing == "metered"
    assert cost.cost_usd == pytest.approx(0.00184, abs=TOL)
    assert cost.cost_status == "estimated"


def test_zai_plan_hint_is_plan():
    row = _row(
        provider="zai", model="glm-5.3", input_tokens=1000, output_tokens=100, usage=_marked(billing_hint="plan")
    )
    assert price_row(row, DEFAULT).billing == "plan"


def test_explicit_billing_override_beats_a_plan_hint():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_BILLING": "zai=metered"})
    row = _row(
        provider="zai", model="glm-5.3", input_tokens=1000, output_tokens=100, usage=_marked(billing_hint="plan")
    )
    assert price_row(row, settings).billing == "metered"


def test_explicit_billing_override_beats_a_metered_hint():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_BILLING": "zai=plan"})
    row = _row(
        provider="zai", model="glm-5.3", input_tokens=1000, output_tokens=100, usage=_marked(billing_hint="metered")
    )
    assert price_row(row, settings).billing == "plan"


def test_provider_cost_is_actual_for_a_metered_row():
    row = _row(usage=_marked(provider_cost_usd=0.5, usage_source="provider"))
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.5, abs=TOL)
    assert cost.cost_status == "actual"
    assert cost.cost_source == "provider"
    assert cost.provider_cost_usd == pytest.approx(0.5, abs=TOL)
    # The list value stays the genai-prices quote: 2010 x 0.10 + 300 x 0.50, per MTok = 0.000351.
    assert cost.list_price_usd == pytest.approx(0.000351, abs=TOL)


def test_provider_cost_without_a_quote_is_its_own_list_price():
    row = _row(provider="openrouter", model="x/unlisted", usage=_marked(provider_cost_usd=0.25))
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.25, abs=TOL)
    assert cost.list_price_usd == pytest.approx(0.25, abs=TOL)
    assert cost.cost_status == "actual"
    assert cost.reason is None


def test_deepseek_cache_hit_is_priced_from_the_hit_count():
    # genai-prices 0.1.12 prices deepseek-chat by time of day: 0.135 / 0.035 / 0.55 per MTok outside
    # 00:30-16:30 UTC (and double that inside). 18:00 UTC is outside, so the base rates apply.
    # uncached 20000 x 0.135 + cache hit 80000 x 0.035 + output 500 x 0.55, per MTok
    # = 0.0027 + 0.0028 + 0.000275 = 0.005775 (without the cache-field fix all 100000 price as uncached: 0.013775)
    off_peak = datetime(2026, 10, 1, 18, 0, tzinfo=UTC)
    row = _row(
        provider="deepseek",
        model="deepseek-chat",
        started_at=off_peak,
        input_tokens=100000,
        output_tokens=500,
        cached_tokens=80000,
    )
    assert price_row(row, DEFAULT).cost_usd == pytest.approx(0.005775, abs=TOL)
    uncached = _row(
        provider="deepseek", model="deepseek-chat", started_at=off_peak, input_tokens=100000, output_tokens=500
    )
    assert price_row(uncached, DEFAULT).cost_usd == pytest.approx(0.013775, abs=TOL)


def test_local_provider_is_included_at_zero():
    cost = price_row(_row(provider="ollama", model="llama3.2", input_tokens=500, output_tokens=50), DEFAULT)
    assert cost.billing == "local"
    assert cost.cost_usd == 0.0
    assert cost.list_price_usd is None
    assert cost.reason == "local"
    assert cost.cost_status == "included"
    assert cost.cost_source == "none"


def test_unknown_model_is_null_and_unknown_status():
    cost = price_row(_row(model="no-such-model"), DEFAULT)
    assert cost.cost_usd is None
    assert cost.list_price_usd is None
    assert cost.reason == "unknown_model"
    assert cost.cost_status == "unknown"
    assert cost.cost_source == "none"


def test_unmapped_provider_is_null():
    cost = price_row(_row(provider="vertexai", model="gemini-3-pro"), DEFAULT)
    assert cost.cost_usd is None
    assert cost.reason == "unmapped_provider"


def test_row_without_any_token_column_is_not_priced():
    row = _row(input_tokens=None, output_tokens=None, cached_tokens=None, thoughts_tokens=None)
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd is None
    assert cost.reason == "no_usage"


def test_negative_tokens_are_invalid_usage():
    cost = price_row(_row(input_tokens=-5), DEFAULT)
    assert cost.cost_usd is None
    assert cost.reason == "invalid_usage"


def test_custom_price_is_estimated_from_the_custom_source():
    settings = CostSettings.from_env(
        {"HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": '{"anthropic/claude-haiku-5-5": {"input_mtok": 1, "output_mtok": 2}}'}
    )
    cost = price_row(_row(input_tokens=1000, output_tokens=100), settings)
    assert cost.cost_status == "estimated"
    assert cost.cost_source == "custom"


def test_legacy_row_is_repaired_and_flagged():
    # Pre-B1 row: input 12 is the uncached prompt only, cache read 9000. Repaired input is 9012.
    # uncached 12 x 0.10 + cache read 9000 x 0.01 = 0.0000012 + 0.0000900 = 0.0000912
    row = _row(input_tokens=12, output_tokens=0, cached_tokens=9000, usage=None)
    cost = price_row(row, DEFAULT)
    assert cost.cost_usd == pytest.approx(0.0000912, abs=TOL)
    assert cost.legacy_input_repaired is True


def test_legacy_error_row_is_repaired_too():
    row = _row(input_tokens=12, output_tokens=0, cached_tokens=9000, usage=None)
    assert price_row(row, DEFAULT).legacy_input_repaired is True


def test_marked_rows_are_not_repaired():
    row = _row(input_tokens=2010, output_tokens=0, cached_tokens=2000, usage=_marked(billing_hint=None))
    assert price_row(row, DEFAULT).legacy_input_repaired is False
    v1 = _row(input_tokens=12, output_tokens=0, cached_tokens=9000, usage=StoredUsage(v=1))
    assert price_row(v1, DEFAULT).legacy_input_repaired is False


def test_unmarked_non_anthropic_row_is_not_repaired():
    row = _row(provider="openai", model="gpt-6.1-sol", input_tokens=100, output_tokens=0, cached_tokens=500, usage=None)
    cost = price_row(row, DEFAULT)
    assert cost.legacy_input_repaired is False


def test_legacy_before_excludes_later_rows():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_LEGACY_BEFORE": "2026-10-01T13:00:00Z"})
    before = _row(input_tokens=12, output_tokens=0, cached_tokens=9000, usage=None)
    after = _row(
        input_tokens=12,
        output_tokens=0,
        cached_tokens=9000,
        usage=None,
        started_at=datetime(2026, 10, 1, 14, 0, tzinfo=UTC),
    )
    assert price_row(before, settings).legacy_input_repaired is True
    assert price_row(after, settings).legacy_input_repaired is False


def test_unmarked_plan_rows_are_flagged_as_estimates():
    assert price_row(_row(provider="claude-code", usage=None), DEFAULT).tokens_estimated is True
    assert price_row(_row(provider="openai-codex", model="gpt-6.1-sol", usage=None), DEFAULT).tokens_estimated is True


def test_estimate_source_is_flagged_and_provider_counts_are_not():
    estimate = _row(provider="claude-code", usage=_marked(usage_source="estimate"))
    provider = _row(provider="claude-code", usage=_marked(usage_source="provider"))
    assert price_row(estimate, DEFAULT).tokens_estimated is True
    assert price_row(provider, DEFAULT).tokens_estimated is False


def test_unmarked_codex_row_after_the_cutover_is_not_an_estimate():
    settings = CostSettings.from_env({"HINDSIGHT_API_LLM_COST_LEGACY_BEFORE": "2026-10-01T13:00:00Z"})
    later = _row(
        provider="openai-codex",
        model="gpt-6.1-sol",
        usage=None,
        started_at=datetime(2026, 10, 1, 14, 0, tzinfo=UTC),
    )
    assert price_row(later, settings).tokens_estimated is False


def test_alias_reprices_under_the_target_model():
    # xai/grok-4.7-build-fast is priced as x-ai/grok-4.7: input 1000 x 2.00 + output 100 x 6.00
    settings = CostSettings.from_env(
        {"HINDSIGHT_API_LLM_COST_MODEL_ALIASES": '{"xai/grok-4.7-build-fast": "x-ai/grok-4.7"}'}
    )
    cost = price_row(_row(provider="xai", model="grok-4.7-build-fast", input_tokens=1000, output_tokens=100), settings)
    assert cost.cost_usd == pytest.approx(0.0026, abs=TOL)
    assert cost.priced_as == "x-ai/grok-4.7"


def test_custom_price_beats_alias():
    # custom: input 1000 x 1.00 + output 100 x 2.00 = 0.001000 + 0.000200 = 0.0012
    settings = CostSettings.from_env(
        {
            "HINDSIGHT_API_LLM_COST_MODEL_ALIASES": '{"xai/grok-4.7-build-fast": "x-ai/grok-4.7"}',
            "HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": (
                '{"xai/grok-4.7-build-fast": {"input_mtok": 1, "output_mtok": 2}}'
            ),
        }
    )
    cost = price_row(_row(provider="xai", model="grok-4.7-build-fast", input_tokens=1000, output_tokens=100), settings)
    assert cost.cost_usd == pytest.approx(0.0012, abs=TOL)
    assert cost.priced_as == "custom"


def test_custom_price_cache_rates_fall_back_to_input_rate():
    # uncached 10 x 1.00 + cache read 2000 x 0.50 (explicit) + output 300 x 2.00, per MTok
    # = 0.000010 + 0.001000 + 0.000600 = 0.001610
    settings = CostSettings.from_env(
        {
            "HINDSIGHT_API_LLM_COST_CUSTOM_PRICES": (
                '{"anthropic/claude-haiku-5-5": {"input_mtok": 1, "output_mtok": 2, "cache_read_mtok": 0.5}}'
            )
        }
    )
    cost = price_row(_row(input_tokens=2010, output_tokens=300, cached_tokens=2000), settings)
    assert cost.cost_usd == pytest.approx(0.00161, abs=TOL)


def test_started_at_is_passed_as_the_request_timestamp(monkeypatch):
    import genai_prices

    seen: list[datetime] = []
    real_calc_price = genai_prices.calc_price

    def spy(*args, **kwargs):
        seen.append(kwargs["genai_request_timestamp"])
        return real_calc_price(*args, **kwargs)

    monkeypatch.setattr(genai_prices, "calc_price", spy)
    moment = datetime(2026, 3, 4, 5, 6, tzinfo=timezone(timedelta(hours=8)))
    price_row(_row(started_at=moment), DEFAULT)
    assert seen == [moment]


# --- summarize_costs ---------------------------------------------------------------------------


def _summary_rows() -> list[RowUsage]:
    return [
        # 2026-10-01 23:30 UTC: Haiku cache write, 0.000401
        _row(
            started_at=datetime(2026, 10, 1, 23, 30, tzinfo=UTC),
            input_tokens=2010,
            output_tokens=300,
            usage=_marked(cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0),
        ),
        # 2026-10-02 07:30 in Taipei is 2026-10-01 23:30 UTC. Haiku cache read, 0.000171.
        # No write breakdown: counts toward rows_without_breakdown for anthropic.
        _row(
            started_at=datetime(2026, 10, 2, 7, 30, tzinfo=timezone(timedelta(hours=8))),
            input_tokens=2010,
            output_tokens=300,
            cached_tokens=2000,
        ),
        # 2026-10-02 UTC: Codex plan, notional list value 0.003 (1000 x 2.00 + 100 x 10.00 per MTok)
        _row(
            started_at=datetime(2026, 10, 2, 0, 0, tzinfo=UTC),
            provider="openai-codex",
            model="gpt-6.1-sol",
            input_tokens=1000,
            output_tokens=100,
        ),
        # Local model with no price: cost 0, no list value
        _row(provider="ollama", model="llama3.2", input_tokens=500, output_tokens=50),
        _row(model="no-such-model"),  # unknown_model
        _row(
            provider="zai",
            model="glm-5.3",
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            thoughts_tokens=None,
        ),  # no_usage
    ]


def test_summary_groups_days_and_totals():
    summary = summarize_costs(
        _summary_rows(),
        DEFAULT,
        bank_id="bank-a",
        period="7d",
        start="2026-09-24T00:00:00+00:00",
        end=None,
        truncated=False,
    )

    assert summary.bank_id == "bank-a"
    assert summary.currency == "USD"
    assert summary.tz == "UTC"
    assert summary.pricing_source.startswith("genai-prices ")
    assert summary.truncated is False
    assert summary.unpriced_calls == 2
    assert summary.cost_usd == pytest.approx(0.000572, abs=TOL)  # 0.000401 + 0.000171 (+0 local)
    assert summary.plan_list_price_usd == pytest.approx(0.003, abs=TOL)
    # Haiku x2 estimated; codex and ollama and zai included; the unknown model unknown.
    assert summary.status_counts.model_dump() == {"actual": 0, "estimated": 2, "included": 3, "unknown": 1}

    models = {(m.provider, m.model): m for m in summary.models}
    haiku = models[("anthropic", "claude-haiku-5-5")]
    assert haiku.calls == 2
    assert haiku.priced_calls == 2
    assert haiku.cost_usd == pytest.approx(0.000572, abs=TOL)
    assert haiku.input_tokens == 4020
    assert haiku.cache_read_tokens == 2000
    assert haiku.cache_write_tokens == 2000
    assert haiku.output_tokens == 600
    assert haiku.rows_without_breakdown == 1
    assert haiku.legacy_rows_repaired == 0
    assert haiku.tokens_estimated_rows == 0
    assert haiku.provider_cost_usd is None
    assert haiku.provider_cost_calls == 0

    codex = models[("openai-codex", "gpt-6.1-sol")]
    assert codex.billing == "plan"
    assert codex.cost_usd is None
    assert codex.list_price_usd == pytest.approx(0.003, abs=TOL)
    assert codex.rows_without_breakdown == 1  # codex is a cache-write provider, and this row has no breakdown
    assert codex.status_counts.included == 1

    ollama = models[("ollama", "llama3.2")]
    assert ollama.billing == "local"
    assert ollama.cost_usd == 0.0
    assert ollama.unpriced_calls == 0
    assert ollama.rows_without_breakdown == 0  # local providers never owe a cache-write breakdown

    unknown = models[("anthropic", "no-such-model")]
    assert unknown.unpriced_calls == 1
    assert unknown.unpriced_reason == "unknown_model"
    assert unknown.cost_usd is None
    assert unknown.status_counts.unknown == 1

    days = {day.day: day for day in summary.by_day}
    assert list(days) == ["2026-10-01", "2026-10-02"]
    assert days["2026-10-01"].cost_usd == pytest.approx(0.000572, abs=TOL)
    assert days["2026-10-01"].plan_list_price_usd == 0.0  # metered traffic does not count as plan value
    assert days["2026-10-02"].cost_usd == 0.0
    assert days["2026-10-02"].plan_list_price_usd == pytest.approx(0.003, abs=TOL)


def test_day_buckets_use_the_requested_local_zone():
    # 2026-10-01 17:30 UTC is 2026-10-02 01:30 in Taipei.
    row = _row(started_at=datetime(2026, 10, 1, 17, 30, tzinfo=UTC), input_tokens=2010, output_tokens=300)
    utc = summarize_costs([row], DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)
    taipei = summarize_costs(
        [row],
        DEFAULT,
        bank_id="b",
        period=None,
        start="s",
        end=None,
        truncated=False,
        tz=ZoneInfo("Asia/Taipei"),
        tz_name="Asia/Taipei",
    )
    assert [day.day for day in utc.by_day] == ["2026-10-01"]
    assert [day.day for day in taipei.by_day] == ["2026-10-02"]
    assert taipei.tz == "Asia/Taipei"


def test_summary_sums_provider_reported_cost_per_model():
    rows = [
        _row(usage=_marked(provider_cost_usd=0.5, usage_source="provider")),
        _row(provider="openrouter", model="x/unlisted", usage=_marked(provider_cost_usd=0.25)),
    ]
    summary = summarize_costs(rows, DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)
    assert summary.cost_usd == pytest.approx(0.75, abs=TOL)
    models = {(m.provider, m.model): m for m in summary.models}
    openrouter = models[("openrouter", "x/unlisted")]
    assert openrouter.provider_cost_usd == pytest.approx(0.25, abs=TOL)
    assert openrouter.provider_cost_calls == 1
    assert openrouter.status_counts.actual == 1
    anthropic = models[("anthropic", "claude-haiku-5-5")]
    assert anthropic.provider_cost_usd == pytest.approx(0.5, abs=TOL)


def test_summary_counts_estimated_token_rows():
    rows = [_row(provider="claude-code", usage=None), _row(provider="ollama", model="llama3.2", usage=None)]
    summary = summarize_costs(rows, DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)
    models = {(m.provider, m.model): m for m in summary.models}
    assert models[("claude-code", "claude-haiku-5-5")].tokens_estimated_rows == 1
    assert models[("ollama", "llama3.2")].tokens_estimated_rows == 0


def test_empty_summary_has_zero_totals():
    summary = summarize_costs([], DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=True)
    assert summary.models == []
    assert summary.by_day == []
    assert summary.cost_usd == 0.0
    assert summary.plan_list_price_usd == 0.0
    assert summary.unpriced_calls == 0
    assert summary.truncated is True
    assert summary.status_counts.model_dump() == {"actual": 0, "estimated": 0, "included": 0, "unknown": 0}


def test_summary_counts_legacy_repairs():
    rows = [_row(input_tokens=12, output_tokens=0, cached_tokens=9000, usage=None)]
    summary = summarize_costs(rows, DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)
    assert summary.models[0].legacy_rows_repaired == 1
    assert summary.models[0].input_tokens == 9012


def test_summary_splits_one_model_into_plan_and_metered_groups():
    # A zai model can carry plan rows (/coding/) and metered rows (/paas/) at once. Each group holds one
    # billing kind, so the metered cost is billed and the plan row only contributes list price.
    # Haiku 4.5, 1000 in / 100 out: 1000 x $1/MTok + 100 x $5/MTok = $0.0015 (metered and plan list alike).
    rows = [
        _row(model="claude-haiku-4-5", input_tokens=1000, output_tokens=100, usage=_marked(billing_hint="plan")),
        _row(model="claude-haiku-4-5", input_tokens=1000, output_tokens=100, usage=_marked(billing_hint="metered")),
    ]
    summary = summarize_costs(rows, DEFAULT, bank_id="b", period=None, start="s", end=None, truncated=False)

    assert summary.cost_usd == pytest.approx(0.0015, abs=TOL)
    assert summary.plan_list_price_usd == pytest.approx(0.0015, abs=TOL)
    assert summary.status_counts.model_dump() == {"actual": 0, "estimated": 1, "included": 1, "unknown": 0}

    by_billing = {m.billing: m for m in summary.models if m.model == "claude-haiku-4-5"}
    assert set(by_billing) == {"metered", "plan"}
    metered, plan = by_billing["metered"], by_billing["plan"]
    assert metered.calls == 1 and metered.cost_usd == pytest.approx(0.0015, abs=TOL)
    assert metered.list_price_usd == pytest.approx(0.0015, abs=TOL)
    assert metered.status_counts.model_dump() == {"actual": 0, "estimated": 1, "included": 0, "unknown": 0}
    assert plan.calls == 1 and plan.cost_usd is None
    assert plan.list_price_usd == pytest.approx(0.0015, abs=TOL)
    assert plan.status_counts.model_dump() == {"actual": 0, "estimated": 0, "included": 1, "unknown": 0}

    # Per-model money agrees with the overall totals.
    assert sum(m.cost_usd or 0.0 for m in summary.models) == pytest.approx(summary.cost_usd, abs=TOL)
    assert sum(m.list_price_usd or 0.0 for m in summary.models if m.billing == "plan") == pytest.approx(
        summary.plan_list_price_usd, abs=TOL
    )


# --- Dataset guard -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_id", "model_ref"),
    [("anthropic", "claude-haiku-5-5"), ("openai", "gpt-6.1-sol"), ("x-ai", "grok-4.7"), ("zai", "glm-5.3")],
)
def test_dataset_still_prices_the_deployed_models(provider_id, model_ref):
    from genai_prices import Usage, calc_price

    assert calc_price(Usage(input_tokens=0), model_ref, provider_id=provider_id) is not None


def test_every_default_provider_map_target_exists_in_the_dataset():
    from genai_prices.data_snapshot import find_provider_by_id, get_snapshot

    providers = get_snapshot().providers
    for provider, pricing_provider in llm_cost.DEFAULT_PROVIDER_MAP.items():
        assert find_provider_by_id(providers, pricing_provider) is not None, provider


# --- Source and hook guards --------------------------------------------------------------------


def test_llm_cost_module_never_reads_the_process_environment():
    source = (PACKAGE_ROOT / "engine" / "llm_cost.py").read_text()
    assert "import os" not in source
    assert "os.environ" not in source
    assert "os.getenv" not in source


def test_config_keeps_the_llm_cost_hook():
    source = (PACKAGE_ROOT / "config.py").read_text()
    assert "# fork: cost-accounting\ndef llm_cost_env()" in source


def test_llm_cost_env_returns_only_the_feature_prefix(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABLED", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_BILLING", "zai=plan")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_FOREGROUND_ORDER", "a")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "b")
    env = llm_cost_env()
    assert env["HINDSIGHT_API_LLM_COST_ENABLED"] == "true"
    assert env["HINDSIGHT_API_LLM_COST_BILLING"] == "zai=plan"
    assert all(name.startswith(ENV_PREFIX) for name in env)

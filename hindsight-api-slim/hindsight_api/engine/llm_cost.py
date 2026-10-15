"""Fork LLM cost accounting (fork: cost-accounting).

Prices every ``llm_requests`` row at read time, so no money is stored and a genai-prices bump reprices
history. This module is the only parser of the ``HINDSIGHT_API_LLM_COST_*`` variables; they are read
through ``config.llm_cost_env()``.

Write path. Provider adapters in this module return keyword arguments for ``record_llm_call`` under
``FORK_USAGE_KWARG`` (a ``CallUsage``), built from the raw provider usage by tolerant Pydantic models.
Every adapter returns ``{}`` with the feature off and never raises. ``trace_usage_info`` turns the kwarg
into the ``llm_info.usage`` v2 marker, which every traced row carries while the feature is on.

Read path. ``read_stored_usage`` reads a stored marker into ``StoredUsage``; ``RowUsage`` carries it into
``price_row`` (one row) and ``summarize_costs`` (totals and local-day buckets). An Anthropic row without a
marker is a legacy row (before B1 the stored input excluded cache reads) and is repaired. Each row has a
cost status: ``actual`` (provider-reported), ``estimated`` (genai-prices or a custom price), ``included``
(plan or local) or ``unknown``. See FORK-COST-ACCOUNTING.md at the repo root.
"""

import importlib.metadata
import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from decimal import Decimal
from functools import cache
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from ..config import llm_cost_env  # the only env reader; see config.py

logger = logging.getLogger(__name__)

BillingKind = Literal["metered", "plan", "local"]
BillingHint = Literal["plan", "metered"]
CostStatus = Literal["actual", "estimated", "included", "unknown"]
CostSource = Literal["provider", "genai-prices", "custom", "none"]
UsageSource = Literal["provider", "estimate"]
UnpricedReason = Literal[
    "subscription_plan", "local", "unmapped_provider", "unknown_model", "no_usage", "invalid_usage"
]

USAGE_VERSION = 2
FORK_USAGE_KWARG = "fork_cost_usage"

ENV_PREFIX = "HINDSIGHT_API_LLM_COST_"
ENV_ENABLED = "HINDSIGHT_API_LLM_COST_ENABLED"
ENV_BILLING = "HINDSIGHT_API_LLM_COST_BILLING"
ENV_PROVIDER_MAP = "HINDSIGHT_API_LLM_COST_PROVIDER_MAP"
ENV_MODEL_ALIASES = "HINDSIGHT_API_LLM_COST_MODEL_ALIASES"
ENV_CUSTOM_PRICES = "HINDSIGHT_API_LLM_COST_CUSTOM_PRICES"
ENV_LEGACY_BEFORE = "HINDSIGHT_API_LLM_COST_LEGACY_BEFORE"
# Quota settings share the prefix but are validated by llm_quota.py.
QUOTA_ENV_PREFIX = "HINDSIGHT_API_LLM_COST_QUOTA_"
_KNOWN_ENV = frozenset(
    {ENV_ENABLED, ENV_BILLING, ENV_PROVIDER_MAP, ENV_MODEL_ALIASES, ENV_CUSTOM_PRICES, ENV_LEGACY_BEFORE}
)

_BILLING_KINDS: dict[str, BillingKind] = {"metered": "metered", "plan": "plan", "local": "local"}
_TRUE_VALUES = frozenset({"true", "1", "yes"})
_FALSE_VALUES = frozenset({"false", "0", "no", ""})
_MILLION = Decimal(1_000_000)
_TICKS_PER_USD = 10_000_000_000  # xAI cost_in_usd_ticks (docs.x.ai cost tracking)

DEFAULT_BILLING: dict[str, BillingKind] = {
    **{
        provider: "plan"
        for provider in ("openai-codex", "xai-oauth", "claude-code", "github-copilot", "cursor", "nous", "zai")
    },
    **{provider: "local" for provider in ("ollama", "lmstudio", "llamacpp", "mock", "none")},
}
DEFAULT_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "claude-code": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "openai-responses": "openai",
    "xai": "x-ai",
    "xai-oauth": "x-ai",
    "zai": "zai",
    "openrouter": "openrouter",
    "gemini": "google",
    "groq": "groq",
    "deepseek": "deepseek",
    "minimax": "minimax",
    "fireworks": "fireworks",
    "github-copilot": "github-copilot",
    "cursor": "cursor",
}
_CUSTOM_PRICE_KEYS = frozenset(
    {"input_mtok", "output_mtok", "cache_read_mtok", "cache_write_mtok", "cache_write_1h_mtok"}
)
_CUSTOM_PRICE_REQUIRED = frozenset({"input_mtok", "output_mtok"})
# Providers whose cache writes are billed separately, so a row without a write breakdown is a gap.
_CACHE_WRITE_PROVIDERS = frozenset(
    {"anthropic", "claude-code", "openai-codex", "openai", "openai-responses", "openrouter"}
)
_LEGACY_PROVIDER = "anthropic"  # pre-B1 rows: input excluded cache reads, and no marker was written
_ESTIMATE_PROVIDERS = frozenset({"claude-code", "openai-codex"})  # pre-B4/B2 rows: chars/4 estimates
_ZAI_PROVIDER = "zai"


@dataclass(frozen=True)
class ModelKey:
    provider: str
    model: str


@dataclass(frozen=True)
class CustomPrice:
    """USD per million tokens. Unset cache rates fall back to the input rate."""

    input_mtok: Decimal
    output_mtok: Decimal
    cache_read_mtok: Decimal | None = None
    cache_write_mtok: Decimal | None = None
    cache_write_1h_mtok: Decimal | None = None


@dataclass(frozen=True)
class CostSettings:
    """Validated feature settings. ``billing_overrides`` holds only explicit env entries; defaults are
    ``DEFAULT_BILLING`` and apply after a row's own billing hint (see ``_billing_of``)."""

    enabled: bool
    billing_overrides: Mapping[str, BillingKind]
    provider_map: Mapping[str, str]
    model_aliases: Mapping[ModelKey, ModelKey]
    custom_prices: Mapping[ModelKey, CustomPrice]
    legacy_before: datetime | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "CostSettings":
        """Parse and validate every feature variable. Raises ValueError naming the variable at fault."""
        _reject_unknown_keys(env)
        return cls(
            enabled=_parse_enabled(env.get(ENV_ENABLED, "")),
            billing_overrides=_parse_billing(env.get(ENV_BILLING, "")),
            provider_map=_parse_provider_map(env.get(ENV_PROVIDER_MAP, "")),
            model_aliases=_parse_model_aliases(env.get(ENV_MODEL_ALIASES, "")),
            custom_prices=_parse_custom_prices(env.get(ENV_CUSTOM_PRICES, "")),
            legacy_before=_parse_legacy_before(env.get(ENV_LEGACY_BEFORE, "")),
        )


@cache
def load_cost_settings() -> CostSettings:
    """Settings from the environment. Cached; tests call ``load_cost_settings.cache_clear()``."""
    return CostSettings.from_env(llm_cost_env())


def parse_timezone(raw: str) -> tzinfo:
    """An IANA zone name. ``UTC`` in any case needs no tz database. Raises ValueError otherwise."""
    if raw.upper() == "UTC":
        return UTC
    try:
        return ZoneInfo(raw)
    except (LookupError, ValueError, OSError) as exc:  # ZoneInfoNotFoundError is a KeyError
        raise ValueError("tz must be an IANA time zone name such as Asia/Taipei, or UTC") from exc


# --- Parsing -----------------------------------------------------------------------------------


def _reject_unknown_keys(env: Mapping[str, str]) -> None:
    """A misspelt setting must not silently do nothing. The message names the key, never its value."""
    for name in sorted(env):
        if name.startswith(ENV_PREFIX) and not name.startswith(QUOTA_ENV_PREFIX) and name not in _KNOWN_ENV:
            raise ValueError(f"{name} is not a known LLM cost setting")


def _parse_enabled(raw: str) -> bool:
    value = raw.strip().lower()
    if value in _FALSE_VALUES:
        return False
    if value in _TRUE_VALUES:
        return True
    raise ValueError(f"{ENV_ENABLED} must be one of true/false/1/0/yes/no, got {raw!r}")


def _parse_pairs(raw: str, variable: str) -> dict[str, str]:
    """``a=x,b=y`` into a dict. Blank means unset. Empty entries, missing '=' and duplicates raise."""
    pairs: dict[str, str] = {}
    if not raw.strip():
        return pairs
    for entry in raw.split(","):
        key, separator, value = entry.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not separator or not value:
            raise ValueError(f"{variable}: expected provider=value, got {entry.strip()!r}")
        if key in pairs:
            raise ValueError(f"{variable}: duplicate entry for {key!r}")
        pairs[key] = value
    return pairs


def _parse_json_object(raw: str, variable: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{variable} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{variable} must be a JSON object")
    return parsed


def _parse_billing(raw: str) -> dict[str, BillingKind]:
    """Explicit operator entries only. An entry overrides both the row's billing hint and the default."""
    overrides: dict[str, BillingKind] = {}
    for provider, kind in _parse_pairs(raw, ENV_BILLING).items():
        resolved = _BILLING_KINDS.get(kind)
        if resolved is None:
            raise ValueError(f"{ENV_BILLING}: kind for {provider!r} must be metered, plan or local, got {kind!r}")
        overrides[provider] = resolved
    return overrides


def _parse_legacy_before(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{ENV_LEGACY_BEFORE} must be an ISO-8601 timestamp with a UTC offset") from exc
    if moment.tzinfo is None:
        raise ValueError(f"{ENV_LEGACY_BEFORE} must include a UTC offset, for example 2026-10-01T00:00:00Z")
    return moment


def _require_pricing_provider(provider_id: str, variable: str) -> None:
    from genai_prices.data_snapshot import find_provider_by_id, get_snapshot

    if find_provider_by_id(get_snapshot().providers, provider_id) is None:
        raise ValueError(f"{variable}: genai-prices has no provider {provider_id!r}")


def _parse_provider_map(raw: str) -> dict[str, str]:
    provider_map = dict(DEFAULT_PROVIDER_MAP)
    for provider, pricing_provider in _parse_pairs(raw, ENV_PROVIDER_MAP).items():
        _require_pricing_provider(pricing_provider, ENV_PROVIDER_MAP)
        provider_map[provider] = pricing_provider
    return provider_map


def _split_model_key(raw: str, variable: str) -> ModelKey:
    """Split ``provider/model`` on the first slash; model ids may themselves contain slashes."""
    provider, separator, model = raw.partition("/")
    if not separator or not provider or not model:
        raise ValueError(f"{variable}: {raw!r} must be 'provider/model'")
    return ModelKey(provider=provider, model=model)


def _require_price_target(target: ModelKey, variable: str) -> None:
    from genai_prices import Usage, calc_price

    _require_pricing_provider(target.provider, variable)
    try:
        calc_price(Usage(input_tokens=0), target.model, provider_id=target.provider)
    except LookupError as exc:
        raise ValueError(f"{variable}: genai-prices has no model {target.model!r} for {target.provider!r}") from exc


def _parse_model_aliases(raw: str) -> dict[ModelKey, ModelKey]:
    aliases: dict[ModelKey, ModelKey] = {}
    for key, target in _parse_json_object(raw, ENV_MODEL_ALIASES).items():
        if not isinstance(target, str):
            raise ValueError(f"{ENV_MODEL_ALIASES}: value for {key!r} must be a string")
        pricing_target = _split_model_key(target, ENV_MODEL_ALIASES)
        _require_price_target(pricing_target, ENV_MODEL_ALIASES)
        aliases[_split_model_key(key, ENV_MODEL_ALIASES)] = pricing_target
    return aliases


def _mtok(value: object, variable: str, key: str, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{variable}: {field_name} for {key!r} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{variable}: {field_name} for {key!r} must be finite and non-negative")
    return Decimal(str(value))


def _optional_mtok(spec: Mapping[str, Any], field_name: str, key: str) -> Decimal | None:
    if spec.get(field_name) is None:
        return None
    return _mtok(spec[field_name], ENV_CUSTOM_PRICES, key, field_name)


def _parse_custom_prices(raw: str) -> dict[ModelKey, CustomPrice]:
    prices: dict[ModelKey, CustomPrice] = {}
    for key, spec in _parse_json_object(raw, ENV_CUSTOM_PRICES).items():
        model_key = _split_model_key(key, ENV_CUSTOM_PRICES)
        if not isinstance(spec, dict):
            raise ValueError(f"{ENV_CUSTOM_PRICES}: price for {key!r} must be a JSON object")
        unknown = set(spec) - _CUSTOM_PRICE_KEYS
        if unknown:
            raise ValueError(f"{ENV_CUSTOM_PRICES}: unknown field(s) {sorted(unknown)} for {key!r}")
        missing = _CUSTOM_PRICE_REQUIRED - set(spec)
        if missing:
            raise ValueError(f"{ENV_CUSTOM_PRICES}: {key!r} is missing {sorted(missing)}")
        prices[model_key] = CustomPrice(
            input_mtok=_mtok(spec["input_mtok"], ENV_CUSTOM_PRICES, key, "input_mtok"),
            output_mtok=_mtok(spec["output_mtok"], ENV_CUSTOM_PRICES, key, "output_mtok"),
            cache_read_mtok=_optional_mtok(spec, "cache_read_mtok", key),
            cache_write_mtok=_optional_mtok(spec, "cache_write_mtok", key),
            cache_write_1h_mtok=_optional_mtok(spec, "cache_write_1h_mtok", key),
        )
    return prices


# --- Boundary parsing (tolerant) ---------------------------------------------------------------


def _is_real_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_true(value: object) -> bool:
    return value is True


def _optional_count(value: object) -> int | None:
    """A real int (not bool) is kept; anything else, including None, is None (no breakdown)."""
    return value if _is_real_int(value) else None


def _optional_cost(value: object) -> float | None:
    """A finite, non-negative real number in USD; anything else is None (dropped)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _optional_billing_hint(value: object) -> object:
    return value if value in ("plan", "metered") else None


def _optional_usage_source(value: object) -> object:
    return value if value in ("provider", "estimate") else None


OptionalCount = Annotated[int | None, BeforeValidator(_optional_count)]
"""Use for every optional count. NOT ``CountField | None``: a 0-defaulting validator turns None into 0."""
OptionalCost = Annotated[float | None, BeforeValidator(_optional_cost)]
OptionalBillingHint = Annotated[BillingHint | None, BeforeValidator(_optional_billing_hint)]
OptionalUsageSource = Annotated[UsageSource | None, BeforeValidator(_optional_usage_source)]


def _safe_attr(obj: object, name: str) -> object:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _as_mapping(obj: object, names: tuple[str, ...]) -> dict[str, object]:
    """A provider usage object as a plain dict: ``model_dump()`` when that returns a dict, otherwise
    attribute reads of ``names``. Never raises; anything unreadable is absent."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    dump = _safe_attr(obj, "model_dump")
    if callable(dump):
        try:
            dumped = dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped
    return {name: _safe_attr(obj, name) for name in names}


class _CacheWriteFields(BaseModel):
    """The cache-write fields shared by the write and read models, with the D8 normalisation."""

    model_config = ConfigDict(extra="ignore")

    cache_write_tokens: OptionalCount = None
    cache_write_5m_tokens: OptionalCount = None
    cache_write_1h_tokens: OptionalCount = None

    @model_validator(mode="after")
    def _normalise_cache_writes(self) -> "_CacheWriteFields":
        if (
            self.cache_write_tokens is None
            and self.cache_write_5m_tokens is None
            and self.cache_write_1h_tokens is None
        ):
            return self
        writes = _normalised_writes(self.cache_write_tokens, self.cache_write_5m_tokens, self.cache_write_1h_tokens)
        # Frozen models reject normal assignment; object.__setattr__ writes the same field storage pydantic reads.
        object.__setattr__(self, "cache_write_tokens", writes.total)
        object.__setattr__(self, "cache_write_5m_tokens", writes.five_minute)
        object.__setattr__(self, "cache_write_1h_tokens", writes.one_hour)
        return self


@dataclass(frozen=True)
class _CacheWrites:
    """The three normalised cache-write counts (D8)."""

    total: int
    five_minute: int
    one_hour: int


def _normalised_writes(total: int | None, five_minute: int | None, one_hour: int | None) -> _CacheWrites:
    """D8: no negative 5m write when the total is missing, and the total always covers both TTLs."""
    one_h = max(0, one_hour or 0)
    five_m = max(0, five_minute) if five_minute is not None else max(0, (total or 0) - one_h)
    all_writes = max(total or 0, five_m + one_h)
    return _CacheWrites(total=all_writes, five_minute=all_writes - one_h, one_hour=one_h)


class CallUsage(_CacheWriteFields):
    """What the adapter knows about one call beyond the token counts. Written as ``llm_info.usage``.

    A None cache-write count means "no breakdown" (absent from the row). Once any write count is set,
    all three are set. ``provider_cost_usd`` is a finite, non-negative USD amount or None.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    provider_cost_usd: OptionalCost = None
    billing_hint: OptionalBillingHint = None
    usage_source: OptionalUsageSource = None

    def to_llm_info(self) -> dict[str, Any]:
        """The ``llm_info`` fragment. Key order is the persisted order: v, then the fields in definition order."""
        return {"usage": {"v": USAGE_VERSION, **self.model_dump(exclude_none=True)}}


class StoredUsage(_CacheWriteFields):
    """``llm_info.usage`` as read back from a row. Every field is tolerant: a wrong type reads as absent."""

    model_config = ConfigDict(extra="ignore")

    v: OptionalCount = None
    provider_cost_usd: OptionalCost = None
    billing_hint: OptionalBillingHint = None
    usage_source: OptionalUsageSource = None


def read_stored_usage(raw: object) -> StoredUsage | None:
    """The stored marker. None when there is no ``llm_info.usage`` object (no marker); otherwise the
    fields that read cleanly, the rest absent."""
    if not isinstance(raw, dict):
        return None
    try:
        return StoredUsage.model_validate(raw)
    except ValidationError:
        return StoredUsage()


class _CacheDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cache_write_tokens: OptionalCount = None
    cache_creation_input_tokens: OptionalCount = None


class _CostDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    upstream_inference_cost: OptionalCost = None


class _UsageExtras(BaseModel):
    """Fork-only fields of an OpenAI-shaped usage block (chat completions and Responses)."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens_details: _CacheDetails = Field(default_factory=_CacheDetails)
    input_tokens_details: _CacheDetails = Field(default_factory=_CacheDetails)
    cache_creation_input_tokens: OptionalCount = None
    cost: OptionalCost = None  # OpenRouter credits, USD
    cost_in_usd_ticks: OptionalCount = None  # xAI
    is_byok: Annotated[bool, BeforeValidator(_is_true)] = False
    cost_details: _CostDetails = Field(default_factory=_CostDetails)


_USAGE_EXTRA_NAMES = (
    "prompt_tokens_details",
    "input_tokens_details",
    "cache_creation_input_tokens",
    "cost",
    "cost_in_usd_ticks",
    "is_byok",
    "cost_details",
)
_NESTED_NAMES = ("cache_write_tokens", "cache_creation_input_tokens", "upstream_inference_cost")


def _usage_extras(usage: object) -> _UsageExtras:
    fields = _as_mapping(usage, _USAGE_EXTRA_NAMES)
    for key in ("prompt_tokens_details", "input_tokens_details", "cost_details"):
        fields[key] = _as_mapping(fields.get(key), _NESTED_NAMES)
    return _UsageExtras.model_validate(fields)


class _UsageBlock(BaseModel):
    """Anthropic-shaped usage (Messages API, and the Claude Agent SDK's raw dict)."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: OptionalCount = None
    cache_creation_input_tokens: OptionalCount = None


class _CacheCreationBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ephemeral_5m_input_tokens: OptionalCount = None
    ephemeral_1h_input_tokens: OptionalCount = None


_USAGE_BLOCK_NAMES = ("input_tokens", "cache_creation_input_tokens", "cache_creation")
_CREATION_NAMES = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _anthropic_breakdown(usage: object) -> CallUsage | None:
    """The cache-write split of an Anthropic-shaped usage block. None when there is no real input count."""
    fields = _as_mapping(usage, _USAGE_BLOCK_NAMES)
    block = _UsageBlock.model_validate(fields)
    if block.input_tokens is None:
        return None
    creation = _CacheCreationBlock.model_validate(_as_mapping(fields.get("cache_creation"), _CREATION_NAMES))
    return CallUsage(
        cache_write_tokens=block.cache_creation_input_tokens or 0,  # a reported None is an uncached call
        cache_write_5m_tokens=creation.ephemeral_5m_input_tokens,
        cache_write_1h_tokens=creation.ephemeral_1h_input_tokens,
        usage_source="provider",
    )


def _first_positive(*values: int | None) -> int:
    for value in values:
        if value is not None and value > 0:
            return value
    return 0


def _ticks_cost(ticks: int | None) -> float | None:
    return None if ticks is None else _optional_cost(ticks / _TICKS_PER_USD)


def _billing_hint_for(provider: str, base_url: object) -> BillingHint | None:
    """zai only: ``plan`` on the Coding Plan endpoint (``/coding/`` in the path), else ``metered``.

    None (no hint) when the base URL is unknown; the default for zai is then the plan billing.
    """
    if provider != _ZAI_PROVIDER or not isinstance(base_url, str) or not base_url:
        return None
    return "plan" if "/coding/" in urlsplit(base_url).path else "metered"


# --- Write-path adapters -----------------------------------------------------------------------

_settings_error_logged = False


def _settings_or_none() -> CostSettings | None:
    """The settings, or None when they do not parse. Never raises; a broken setting logs one warning per process."""
    global _settings_error_logged
    try:
        return load_cost_settings()
    except Exception as exc:
        if not _settings_error_logged:
            _settings_error_logged = True
            logger.warning("LLM cost accounting is off; invalid HINDSIGHT_API_LLM_COST_* settings: %s", exc)
        return None


def _cost_enabled() -> bool:
    settings = _settings_or_none()
    return settings is not None and settings.enabled


def _guarded(build: Callable[[], CallUsage | None]) -> dict[str, CallUsage]:
    """Adapter wrapper: ``{}`` with the feature off; on a parse failure, an empty breakdown; never raises."""
    if not _cost_enabled():
        return {}
    try:
        usage = build()
    except Exception as exc:
        logger.debug("LLM cost usage not read: %s", type(exc).__name__)
        usage = CallUsage()
    return {} if usage is None else {FORK_USAGE_KWARG: usage}


def anthropic_cache_write_kwargs(response: Any) -> dict[str, CallUsage]:
    """Keyword arguments for ``record_llm_call``, read off an Anthropic response.

    ``{}`` with the feature off or when the response has no usable usage block.
    """
    return _guarded(lambda: _anthropic_breakdown(getattr(response, "usage", None)))


def claude_code_usage_kwargs(raw_usage: object) -> dict[str, CallUsage]:
    """Claude Agent SDK usage. A raw usage dict is provider counts; None means the chars/4 estimate.

    The SDK's ``total_cost_usd`` is deliberately not recorded: it is the CLI's own client-side
    estimate, not a provider bill, and it is not trustworthy (live check 2026-10-15: it priced
    claude-haiku-5-5 cache writes at $6.25/MTok against the official $0.20/MTok).
    """

    def build() -> CallUsage:
        breakdown = _anthropic_breakdown(raw_usage) if isinstance(raw_usage, dict) else None
        return CallUsage(usage_source="estimate") if breakdown is None else breakdown

    return _guarded(build)


_codex_pending: ContextVar[tuple[object, CallUsage] | None] = ContextVar("fork_codex_usage", default=None)


class _CodexCostResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: _UsageExtras | None = None


class _CodexCostEvent(BaseModel):
    """The part of a Codex ``response.completed`` event the cost hook reads: ``response.usage``."""

    model_config = ConfigDict(extra="ignore")

    response: _CodexCostResponse | None = None


def _codex_cache_writes(event: object) -> int:
    """``input_tokens_details.cache_write_tokens`` of a raw Codex event; 0 when it is absent or unreadable.

    Read here because the provider's own usage model (upstream's) does not carry the field.
    """
    try:
        parsed = _CodexCostEvent.model_validate(event)
    except ValidationError:
        return 0
    usage = parsed.response.usage if parsed.response is not None else None
    writes = usage.input_tokens_details.cache_write_tokens if usage is not None else None
    return writes or 0


def note_codex_usage(result: object, event: object) -> None:
    """Remember the cache-write split of the usage object a Codex call just built (D3), read from the raw event.

    ``codex_usage_kwargs`` hands the breakdown back only for that same object, so a stale value never
    reaches another call. A no-op with the feature off.
    """
    if not _cost_enabled():
        return
    try:
        writes = _codex_cache_writes(event)
        _codex_pending.set((result, CallUsage(cache_write_tokens=writes, usage_source="provider")))
    except Exception as exc:
        logger.debug("LLM cost Codex usage not noted: %s", type(exc).__name__)
        _codex_pending.set(None)


def codex_usage_kwargs(usage: object) -> dict[str, CallUsage]:
    """Keyword arguments for a Codex span record. The noted breakdown only when it belongs to ``usage``."""

    def build() -> CallUsage:
        pending = _codex_pending.get()
        if pending is not None and pending[0] is usage:
            return pending[1]
        return CallUsage(usage_source="estimate")

    return _guarded(build)


def openai_compatible_usage_kwargs(provider: str, base_url: str | None, response: object) -> dict[str, CallUsage]:
    """Cache writes, provider-reported cost and the zai billing hint of an OpenAI-compatible completion."""

    def build() -> CallUsage:
        hint = _billing_hint_for(provider, base_url)
        usage = getattr(response, "usage", None)
        if usage is None:
            return CallUsage(billing_hint=hint, usage_source="estimate")
        extras = _usage_extras(usage)
        return CallUsage(
            cache_write_tokens=_first_positive(
                extras.prompt_tokens_details.cache_write_tokens,
                extras.prompt_tokens_details.cache_creation_input_tokens,
                extras.cache_creation_input_tokens,
            ),
            provider_cost_usd=_openai_cost(extras),
            billing_hint=hint,
            usage_source="provider",
        )

    return _guarded(build)


def _openai_cost(extras: _UsageExtras) -> float | None:
    """xAI ticks win by field presence; else OpenRouter ``cost``, plus the upstream cost for BYOK."""
    if extras.cost_in_usd_ticks is not None:
        return _ticks_cost(extras.cost_in_usd_ticks)
    if extras.cost is None:
        return None
    total = extras.cost
    upstream = extras.cost_details.upstream_inference_cost
    if extras.is_byok and upstream is not None:
        total += upstream
    return _optional_cost(total)


def openai_responses_usage_kwargs(response: object) -> dict[str, CallUsage]:
    """Cache writes and xAI-style cost of an OpenAI Responses call."""

    def build() -> CallUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return CallUsage(usage_source="estimate")
        extras = _usage_extras(usage)
        return CallUsage(
            cache_write_tokens=_first_positive(extras.input_tokens_details.cache_write_tokens),
            provider_cost_usd=_ticks_cost(extras.cost_in_usd_ticks),
            usage_source="provider",
        )

    return _guarded(build)


def xai_oauth_usage_kwargs(cost_in_usd_ticks: object) -> dict[str, CallUsage]:
    """xAI OAuth (plan) cost, from ``usage.cost_in_usd_ticks``. xAI has no cache-write concept, so zero."""

    def build() -> CallUsage:
        return CallUsage(
            cache_write_tokens=0,
            provider_cost_usd=_ticks_cost(_optional_count(cost_in_usd_ticks)),
            usage_source="provider",
        )

    return _guarded(build)


def trace_usage_info(extra: Mapping[str, Any]) -> dict[str, Any]:
    """The ``llm_info`` fragment for one trace row. Never raises.

    ``{}`` with the feature off. With it on, every row gets the v2 marker, plus the breakdown when an
    adapter produced one. A broken setting logs one warning per process and keeps the row without a marker.
    """
    if not _cost_enabled():
        return {}
    try:
        usage = extra.get(FORK_USAGE_KWARG)
        if isinstance(usage, CallUsage):
            return usage.to_llm_info()
    except Exception as exc:
        logger.debug("LLM cost usage not traced: %s", type(exc).__name__)
    return {"usage": {"v": USAGE_VERSION}}


# --- Pricing -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowUsage:
    """One ``llm_requests`` row as the pricer sees it. ``usage`` is None when the row has no marker."""

    provider: str | None
    model: str | None
    started_at: datetime
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    thoughts_tokens: int | None
    usage: StoredUsage | None = None

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None:
            object.__setattr__(self, "started_at", self.started_at.replace(tzinfo=UTC))


class RowCost(BaseModel):
    billing: BillingKind
    cost_usd: float | None  # billed (metered), 0 (local), None (plan or unpriced)
    list_price_usd: float | None  # API-equivalent at public rates; the custom price when one applies
    reason: UnpricedReason | None
    priced_as: str | None  # "anthropic/claude-haiku-5-5", or "custom"
    cost_status: CostStatus
    cost_source: CostSource
    provider_cost_usd: float | None = None  # what the provider reported for this call, when it did
    legacy_input_repaired: bool = False
    has_cache_write_breakdown: bool = False
    tokens_estimated: bool = False


class CostStatusCounts(BaseModel):
    actual: int = 0
    estimated: int = 0
    included: int = 0
    unknown: int = 0


class ModelCostTotals(BaseModel):
    provider: str | None
    model: str | None
    billing: BillingKind
    calls: int
    priced_calls: int
    unpriced_calls: int
    unpriced_reason: UnpricedReason | None  # the most frequent reason among unpriced calls
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    thoughts_tokens: int
    rows_without_breakdown: int
    legacy_rows_repaired: int
    tokens_estimated_rows: int
    cost_usd: float | None  # None for plan billing
    list_price_usd: float | None
    provider_cost_usd: float | None  # sum over the rows that carried one; None when none did
    provider_cost_calls: int
    status_counts: CostStatusCounts


class DayCostTotals(BaseModel):
    day: str  # local calendar date in the requested time zone, ISO
    cost_usd: float  # metered and local spend that day
    plan_list_price_usd: float  # notional list value of that day's plan traffic only


class LLMCostSummaryResponse(BaseModel):
    bank_id: str
    period: str | None
    start: str
    end: str | None
    currency: Literal["USD"] = "USD"
    pricing_source: str  # e.g. "genai-prices 0.1.12"
    truncated: bool
    tz: str = "UTC"  # the zone name as requested; day buckets are local dates in it
    models: list[ModelCostTotals]
    by_day: list[DayCostTotals]
    cost_usd: float  # metered and local total
    plan_list_price_usd: float  # notional value of plan traffic
    unpriced_calls: int
    status_counts: CostStatusCounts = Field(default_factory=CostStatusCounts)


class LLMRowCostResponse(BaseModel):
    bank_id: str
    items: dict[str, RowCost]  # keyed by llm_requests.id


@dataclass(frozen=True)
class _Tokens:
    input_tokens: int  # total prompt, after legacy repair
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int
    output_tokens: int  # visible output only
    thoughts_tokens: int
    has_breakdown: bool
    legacy_repaired: bool
    tokens_estimated: bool


@dataclass(frozen=True)
class _Quote:
    value: Decimal | None
    reason: UnpricedReason | None
    priced_as: str | None
    source: CostSource


@dataclass(frozen=True)
class _Priced:
    billing: BillingKind
    cost: Decimal | None
    list_price: Decimal | None
    reason: UnpricedReason | None
    priced_as: str | None
    cost_status: CostStatus
    cost_source: CostSource
    provider_cost: Decimal | None
    tokens: _Tokens


def _billing_of(row: RowUsage, settings: CostSettings) -> BillingKind:
    """Resolution: explicit env override > the row's billing hint > DEFAULT_BILLING > metered."""
    provider = row.provider
    if provider is None:
        return "metered"
    override = settings.billing_overrides.get(provider)
    if override is not None:
        return override
    if row.usage is not None and row.usage.billing_hint is not None:
        return row.usage.billing_hint
    return DEFAULT_BILLING.get(provider, "metered")


def _before_cutover(row: RowUsage, settings: CostSettings) -> bool:
    return settings.legacy_before is None or row.started_at < settings.legacy_before


def _tokens(row: RowUsage, settings: CostSettings) -> _Tokens:
    """Token counts after the D2 legacy repair. A marked row is trusted as stored."""
    input_tokens = _count(row.input_tokens)
    cache_read = _count(row.cached_tokens)
    usage = row.usage
    has_breakdown = usage is not None and usage.cache_write_tokens is not None
    cache_write = _count(usage.cache_write_tokens) if usage is not None else 0
    one_hour = _count(usage.cache_write_1h_tokens) if usage is not None else 0
    before_cutover = _before_cutover(row, settings)
    legacy = row.provider == _LEGACY_PROVIDER and usage is None and before_cutover and cache_read > 0
    if legacy:
        input_tokens += cache_read  # the stored input excluded cache reads; writes were never stored
    estimated = (usage is None and row.provider in _ESTIMATE_PROVIDERS and before_cutover) or (
        usage is not None and usage.usage_source == "estimate"
    )
    return _Tokens(
        input_tokens=input_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cache_write_1h_tokens=one_hour,
        output_tokens=_count(row.output_tokens),
        thoughts_tokens=_count(row.thoughts_tokens),
        has_breakdown=has_breakdown,
        legacy_repaired=legacy,
        tokens_estimated=estimated,
    )


def _count(value: object) -> int:
    """A token count: a real int (not bool) is kept; anything else, or absent, reads as 0."""
    return value if _is_real_int(value) else 0


def _usage_kwargs(tokens: _Tokens) -> dict[str, int]:
    kwargs = {
        "input_tokens": tokens.input_tokens,
        "cache_read_tokens": tokens.cache_read_tokens,
        "output_tokens": tokens.output_tokens + tokens.thoughts_tokens,  # reasoning bills at the output rate
    }
    if tokens.has_breakdown:
        kwargs["cache_write_tokens"] = tokens.cache_write_tokens
        kwargs["cache_write_5m_tokens"] = tokens.cache_write_tokens - tokens.cache_write_1h_tokens
        kwargs["cache_write_1h_tokens"] = tokens.cache_write_1h_tokens
    return kwargs


def _custom_value(price: CustomPrice, tokens: _Tokens) -> Decimal:
    def rate(value: Decimal | None, fallback: Decimal) -> Decimal:
        return fallback if value is None else value

    cache_write_rate = rate(price.cache_write_mtok, price.input_mtok)
    one_hour_rate = rate(price.cache_write_1h_mtok, cache_write_rate)
    uncached = tokens.input_tokens - tokens.cache_read_tokens - tokens.cache_write_tokens
    five_min_writes = tokens.cache_write_tokens - tokens.cache_write_1h_tokens
    priced = (
        Decimal(uncached) * price.input_mtok
        + Decimal(tokens.cache_read_tokens) * rate(price.cache_read_mtok, price.input_mtok)
        + Decimal(five_min_writes) * cache_write_rate
        + Decimal(tokens.cache_write_1h_tokens) * one_hour_rate
        + Decimal(tokens.output_tokens + tokens.thoughts_tokens) * price.output_mtok
    )
    return priced / _MILLION


def _genai_quote(row: RowUsage, tokens: _Tokens, target: ModelKey) -> _Quote:
    from genai_prices import Usage, calc_price

    try:
        usage = Usage(**_usage_kwargs(tokens))
        calc = calc_price(usage, target.model, provider_id=target.provider, genai_request_timestamp=row.started_at)
    except LookupError:
        return _Quote(None, "unknown_model", None, "none")
    except ValueError:
        return _Quote(None, "invalid_usage", None, "none")
    return _Quote(calc.total_price, None, f"{target.provider}/{target.model}", "genai-prices")


def _quote(row: RowUsage, tokens: _Tokens, settings: CostSettings) -> _Quote:
    """The list value of the row's tokens at the rate that applied when it was made, and where it came from."""
    if (
        row.input_tokens is None
        and row.output_tokens is None
        and row.cached_tokens is None
        and row.thoughts_tokens is None
    ):
        return _Quote(None, "no_usage", None, "none")
    if row.provider is None:
        return _Quote(None, "unmapped_provider", None, "none")
    if row.model is None:
        return _Quote(None, "unknown_model", None, "none")
    counts = (
        tokens.input_tokens,
        tokens.cache_read_tokens,
        tokens.cache_write_tokens,
        tokens.cache_write_1h_tokens,
        tokens.output_tokens,
        tokens.thoughts_tokens,
    )
    if min(counts) < 0:
        return _Quote(None, "invalid_usage", None, "none")
    key = ModelKey(provider=row.provider, model=row.model)
    custom = settings.custom_prices.get(key)
    if custom is not None:
        return _Quote(_custom_value(custom, tokens), None, "custom", "custom")
    target = settings.model_aliases.get(key)
    if target is None:
        pricing_provider = settings.provider_map.get(row.provider)
        if pricing_provider is None:
            return _Quote(None, "unmapped_provider", None, "none")
        target = ModelKey(provider=pricing_provider, model=row.model)
    return _genai_quote(row, tokens, target)


def _decimal_or_none(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _price(row: RowUsage, settings: CostSettings) -> _Priced:
    """The D5/D6 table: provider cost wins for metered rows; plan and local rows are included, never billed."""
    billing = _billing_of(row, settings)
    tokens = _tokens(row, settings)
    quote = _quote(row, tokens, settings)
    provider_cost = _decimal_or_none(row.usage.provider_cost_usd if row.usage is not None else None)
    if billing == "local":
        return _Priced(
            billing, Decimal(0), quote.value, "local", quote.priced_as, "included", "none", provider_cost, tokens
        )
    if billing == "plan":
        # Plan traffic has no per-token bill. The list value is the notional API-equivalent: the quote,
        # else what the provider reported. A failed quote keeps its own reason when nothing else is known.
        if quote.value is not None:
            list_price, list_source = quote.value, quote.source
        elif provider_cost is not None:
            list_price, list_source = provider_cost, "provider"
        else:
            list_price, list_source = None, "none"
        reason = "subscription_plan" if list_price is not None else (quote.reason or "subscription_plan")
        return _Priced(
            billing, None, list_price, reason, quote.priced_as, "included", list_source, provider_cost, tokens
        )
    if provider_cost is not None:
        list_price = quote.value if quote.value is not None else provider_cost
        return _Priced(
            billing, provider_cost, list_price, None, quote.priced_as, "actual", "provider", provider_cost, tokens
        )
    if quote.value is not None:
        return _Priced(
            billing, quote.value, quote.value, None, quote.priced_as, "estimated", quote.source, None, tokens
        )
    return _Priced(billing, None, None, quote.reason, quote.priced_as, "unknown", "none", None, tokens)


def _money(value: Decimal | None) -> float | None:
    return None if value is None else float(round(value, 8))


def _row_cost(priced: _Priced) -> RowCost:
    return RowCost(
        billing=priced.billing,
        cost_usd=_money(priced.cost),
        list_price_usd=_money(priced.list_price),
        reason=priced.reason,
        priced_as=priced.priced_as,
        cost_status=priced.cost_status,
        cost_source=priced.cost_source,
        provider_cost_usd=_money(priced.provider_cost),
        legacy_input_repaired=priced.tokens.legacy_repaired,
        has_cache_write_breakdown=priced.tokens.has_breakdown,
        tokens_estimated=priced.tokens.tokens_estimated,
    )


def price_row(row: RowUsage, settings: CostSettings) -> RowCost:
    return _row_cost(_price(row, settings))


# --- Summary -----------------------------------------------------------------------------------


def _status_counts(statuses: Mapping[CostStatus, int]) -> CostStatusCounts:
    return CostStatusCounts(
        actual=statuses.get("actual", 0),
        estimated=statuses.get("estimated", 0),
        included=statuses.get("included", 0),
        unknown=statuses.get("unknown", 0),
    )


@dataclass
class _Group:
    provider: str | None
    model: str | None
    billing: BillingKind
    calls: int = 0
    priced_calls: int = 0
    unpriced_calls: int = 0
    reasons: Counter[UnpricedReason] = field(default_factory=Counter)
    statuses: Counter[CostStatus] = field(default_factory=Counter)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    rows_without_breakdown: int = 0
    legacy_rows_repaired: int = 0
    tokens_estimated_rows: int = 0
    cost: Decimal = Decimal(0)
    cost_seen: bool = False
    list_price: Decimal = Decimal(0)
    list_seen: bool = False
    provider_cost: Decimal = Decimal(0)
    provider_cost_calls: int = 0

    def add(self, priced: _Priced) -> None:
        tokens = priced.tokens
        self.calls += 1
        if priced.cost is None and priced.list_price is None:
            self.unpriced_calls += 1
            if priced.reason is not None:
                self.reasons[priced.reason] += 1
        else:
            self.priced_calls += 1
        self.statuses[priced.cost_status] += 1
        self.input_tokens += tokens.input_tokens
        self.cache_read_tokens += tokens.cache_read_tokens
        self.cache_write_tokens += tokens.cache_write_tokens
        self.output_tokens += tokens.output_tokens
        self.thoughts_tokens += tokens.thoughts_tokens
        if not tokens.has_breakdown and self.provider in _CACHE_WRITE_PROVIDERS:
            self.rows_without_breakdown += 1
        self.legacy_rows_repaired += 1 if tokens.legacy_repaired else 0
        self.tokens_estimated_rows += 1 if tokens.tokens_estimated else 0
        if priced.cost is not None:
            self.cost += priced.cost
            self.cost_seen = True
        if priced.list_price is not None:
            self.list_price += priced.list_price
            self.list_seen = True
        if priced.provider_cost is not None:
            self.provider_cost += priced.provider_cost
            self.provider_cost_calls += 1

    def totals(self) -> ModelCostTotals:
        most_common = self.reasons.most_common(1)
        return ModelCostTotals(
            provider=self.provider,
            model=self.model,
            billing=self.billing,
            calls=self.calls,
            priced_calls=self.priced_calls,
            unpriced_calls=self.unpriced_calls,
            unpriced_reason=most_common[0][0] if most_common else None,
            input_tokens=self.input_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            output_tokens=self.output_tokens,
            thoughts_tokens=self.thoughts_tokens,
            rows_without_breakdown=self.rows_without_breakdown,
            legacy_rows_repaired=self.legacy_rows_repaired,
            tokens_estimated_rows=self.tokens_estimated_rows,
            cost_usd=None if self.billing == "plan" or not self.cost_seen else _money(self.cost),
            list_price_usd=_money(self.list_price) if self.list_seen else None,
            provider_cost_usd=_money(self.provider_cost) if self.provider_cost_calls else None,
            provider_cost_calls=self.provider_cost_calls,
            status_counts=_status_counts(self.statuses),
        )


@dataclass
class _DayAccumulator:
    cost: Decimal = Decimal(0)
    plan_list_price: Decimal = Decimal(0)


def _local_day(moment: datetime, zone: tzinfo) -> str:
    return moment.astimezone(zone).date().isoformat()


def summarize_costs(
    rows: Iterable[RowUsage],
    settings: CostSettings,
    *,
    bank_id: str,
    period: str | None,
    start: str,
    end: str | None,
    truncated: bool,
    tz: tzinfo = UTC,
    tz_name: str = "UTC",
) -> LLMCostSummaryResponse:
    # Billing is resolved per row (override > hint > default), so one model can carry plan and metered
    # rows at once (e.g. a zai model called through both /coding/ and /paas/). Each group holds one kind.
    groups: dict[tuple[str | None, str | None, BillingKind], _Group] = {}
    days: dict[str, _DayAccumulator] = {}
    statuses: Counter[CostStatus] = Counter()
    for row in rows:
        priced = _price(row, settings)
        key = (row.provider, row.model, priced.billing)
        group = groups.get(key)
        if group is None:
            group = _Group(provider=row.provider, model=row.model, billing=priced.billing)
            groups[key] = group
        group.add(priced)
        statuses[priced.cost_status] += 1
        day = days.setdefault(_local_day(row.started_at, tz), _DayAccumulator())
        if priced.cost is not None:
            day.cost += priced.cost
        if priced.billing == "plan" and priced.list_price is not None:
            day.plan_list_price += priced.list_price

    ordered = sorted(groups.values(), key=lambda group: (group.provider or "", group.model or "", group.billing))
    billed = sum((group.cost for group in ordered if group.billing != "plan"), Decimal(0))
    plan_list = sum((group.list_price for group in ordered if group.billing == "plan"), Decimal(0))
    return LLMCostSummaryResponse(
        bank_id=bank_id,
        period=period,
        start=start,
        end=end,
        pricing_source=f"genai-prices {importlib.metadata.version('genai-prices')}",
        truncated=truncated,
        tz=tz_name,
        models=[group.totals() for group in ordered],
        by_day=[
            DayCostTotals(
                day=day, cost_usd=float(round(acc.cost, 8)), plan_list_price_usd=float(round(acc.plan_list_price, 8))
            )
            for day, acc in sorted(days.items())
        ],
        cost_usd=float(round(billed, 8)),
        plan_list_price_usd=float(round(plan_list, 8)),
        unpriced_calls=sum(group.unpriced_calls for group in ordered),
        status_counts=_status_counts(statuses),
    )

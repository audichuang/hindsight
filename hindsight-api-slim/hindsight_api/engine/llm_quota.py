"""Subscription quota snapshot for the server's LLM lanes (fork: cost-accounting).

Reads the plan usage windows that a subscription login exposes (Codex, Claude Code OAuth,
Z.ai coding plans) for the lanes this server is configured with. The payload describes the
server, not a bank, so the route checks bank readability and then returns server-wide data.

Nothing here raises into a request: every fetch ends as a ``ProviderQuota`` status. Tokens,
headers and response bodies are never logged or returned; the only text returned from a
provider is a fixed phrase per reason (never the provider's own text, which can echo a key). See
FORK-COST-ACCOUNTING.md at the repo root.

The env variables live under the ``HINDSIGHT_API_LLM_COST_QUOTA_`` prefix and are read through
``config.llm_cost_env()``, the same accessor as the cost feature. This module validates its own keys.
"""

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import cache, partial
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from hindsight_api.config import llm_cost_env
from hindsight_api.engine.aiohttp_session import LoopLocalSession, per_phase_timeout
from hindsight_api.engine.multi_llm import MultiLLMProvider

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.engine.providers.codex_llm import CodexLLM
    from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

# --- Settings ------------------------------------------------------------------------------------

ENV_QUOTA_PREFIX = "HINDSIGHT_API_LLM_COST_QUOTA_"
ENV_QUOTA_ENABLED = "HINDSIGHT_API_LLM_COST_QUOTA_ENABLED"
ENV_QUOTA_CLAUDE_OAUTH_TOKEN = "HINDSIGHT_API_LLM_COST_QUOTA_CLAUDE_OAUTH_TOKEN"

_KNOWN_QUOTA_ENV = frozenset({ENV_QUOTA_ENABLED, ENV_QUOTA_CLAUDE_OAUTH_TOKEN})
_TRUE_VALUES = frozenset({"1", "true", "yes"})
_FALSE_VALUES = frozenset({"", "0", "false", "no"})

CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
CLAUDE_OAUTH_TOKEN_PREFIX = "sk-ant-oat"
UNSUPPORTED_PLAN_PROVIDERS = frozenset({"xai-oauth", "github-copilot", "cursor", "nous"})


class QuotaSettings(BaseModel):
    """Quota feature settings. The token is a SecretStr and never shows in repr or dumps."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    claude_oauth_token: SecretStr | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "QuotaSettings":
        """Parse the quota variables. Unknown ``HINDSIGHT_API_LLM_COST_QUOTA_*`` keys raise, named by key only."""
        unknown = sorted(k for k in env if k.startswith(ENV_QUOTA_PREFIX) and k not in _KNOWN_QUOTA_ENV)
        if unknown:
            raise ValueError(f"unknown LLM quota variable(s): {', '.join(unknown)}")
        raw_enabled = env.get(ENV_QUOTA_ENABLED, "").strip().lower()
        if raw_enabled in _TRUE_VALUES:
            enabled = True
        elif raw_enabled in _FALSE_VALUES:
            enabled = False
        else:
            raise ValueError(f"{ENV_QUOTA_ENABLED} must be one of true/false/1/0/yes/no")
        raw_token = env.get(ENV_QUOTA_CLAUDE_OAUTH_TOKEN, "").strip()
        return cls(enabled=enabled, claude_oauth_token=SecretStr(raw_token) if raw_token else None)


@cache
def load_quota_settings() -> QuotaSettings:
    """Settings from the environment. Cached; tests call ``load_quota_settings.cache_clear()``."""
    return QuotaSettings.from_env(llm_cost_env())


# --- Response models -----------------------------------------------------------------------------

QuotaStatus = Literal["ok", "unsupported", "unavailable", "error"]
QuotaReason = Literal[
    "no_credentials",
    "token_scope",
    "auth_failed",
    "http_error",
    "malformed_response",
    "timeout",
    "provider_error",
    "not_a_plan_endpoint",
    "no_usage_endpoint",
    "internal_error",
]


class QuotaWindow(BaseModel):
    label: str
    used_percent: float | None  # clamped to 0..100
    resets_at: datetime | None  # aware UTC
    window_seconds: int | None
    scope: Literal["account", "model"] = "account"


class QuotaCredits(BaseModel):
    """Codex credits and banked reset credits."""

    balance: float | None
    unlimited: bool = False
    banked_resets: int = 0


class QuotaBudget(BaseModel):
    """Claude extra usage, converted from minor units to major units."""

    used: float
    limit: float
    currency: str = "USD"


class ProviderQuota(BaseModel):
    provider: str
    lanes: list[str]
    status: QuotaStatus
    reason: QuotaReason | None = None
    detail: str | None = None  # short; never contains credentials (e.g. "HTTP 500", a Z.ai msg)
    plan: str | None = None
    windows: list[QuotaWindow] = []
    credits: QuotaCredits | None = None
    budget: QuotaBudget | None = None
    fetched_at: datetime | None = None
    stale: bool = False


class LLMQuotaResponse(BaseModel):
    bank_id: str
    scope: Literal["server"] = "server"
    providers: list[ProviderQuota]


class QuotaReading(BaseModel):
    """Internal result of one provider fetch or parse. Converted to ProviderQuota by the service."""

    status: Literal["ok", "error", "unavailable"]
    reason: QuotaReason | None = None
    detail: str | None = None
    plan: str | None = None
    windows: list[QuotaWindow] = []
    credits: QuotaCredits | None = None
    budget: QuotaBudget | None = None
    fetched_at: datetime | None = None
    stale: bool = False


# --- HTTP ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: str


class HttpGet(Protocol):
    async def __call__(self, url: str, headers: Mapping[str, str]) -> HttpResult: ...


# Shared like every other outbound call: LoopLocalSession follows HTTPS_PROXY/HTTP_PROXY/NO_PROXY, so the
# quota probe reaches the provider the same way the LLM calls for the same credential do.
_QUOTA_SESSION = LoopLocalSession(timeout=per_phase_timeout(10))
# Total per-target deadline, enforced by QuotaCache around the whole fetch (the session's 10 s is per phase).
QUOTA_FETCH_TIMEOUT_SECONDS = 10.0


async def default_http_get(url: str, headers: Mapping[str, str]) -> HttpResult:
    """GET through the shared session with a 10 s per-phase timeout."""
    async with _QUOTA_SESSION.get().get(url, headers=dict(headers)) as response:
        body = await response.text()
        return HttpResult(status=response.status, body=body)


# --- Boundary payload models (tolerant: extra keys ignored, missing keys are None) ----------------


class _CodexWindow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    used_percent: float | None = None
    limit_window_seconds: int | None = None
    reset_at: float | None = None
    reset_after_seconds: float | None = None


class _CodexRateLimit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    primary_window: _CodexWindow | None = None
    secondary_window: _CodexWindow | None = None
    limit_reached: bool | None = None


class _CodexCredits(BaseModel):
    model_config = ConfigDict(extra="ignore")
    has_credits: bool | None = None
    unlimited: bool | None = None
    balance: float | str | None = None


class _CodexResetCredits(BaseModel):
    model_config = ConfigDict(extra="ignore")
    available_count: int | None = None


class _CodexUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rate_limit: _CodexRateLimit | None = None
    plan_type: str | None = None
    credits: _CodexCredits | None = None
    rate_limit_reset_credits: _CodexResetCredits | None = None


class _ClaudeWindow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    utilization: float | None = None
    resets_at: datetime | None = None


class _ClaudeExtraUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    is_enabled: bool | None = None
    monthly_limit: float | None = None
    used_credits: float | None = None
    utilization: float | None = None
    currency: str | None = None


class _ClaudeUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    five_hour: _ClaudeWindow | None = None
    seven_day: _ClaudeWindow | None = None
    seven_day_opus: _ClaudeWindow | None = None
    seven_day_sonnet: _ClaudeWindow | None = None
    extra_usage: _ClaudeExtraUsage | None = None


class _ZaiLimit(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    type: str | None = None
    unit: int | None = None
    number: int | None = None
    percentage: float | None = None
    next_reset_time: float | str | None = Field(default=None, alias="nextResetTime")


class _ZaiData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    limits: list[_ZaiLimit] = []
    plan_name: str | None = Field(default=None, alias="planName")
    plan: str | None = None


class _ZaiUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    code: int | None = None
    success: bool | None = None
    msg: str | None = None
    data: _ZaiData | None = None


# --- Parsing helpers -----------------------------------------------------------------------------

_ZAI_UNIT_SECONDS = {1: 86400, 3: 3600, 5: 60}
_ZAI_UNIT_SUFFIX = {1: "d", 3: "h", 5: "m"}


def _clamp(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return max(0.0, min(100.0, value))


def _epoch_seconds(seconds: float, *, fallback: datetime | None = None) -> datetime | None:
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError, ValueError):
        return fallback


def _aware(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _balance(value: float | str | None) -> float | None:
    if value is None or isinstance(value, float):
        return value
    try:
        return float(value)
    except ValueError:
        return None


def _failed(reason: QuotaReason, fetched_at: datetime, detail: str | None = None) -> QuotaReading:
    return QuotaReading(status="error", reason=reason, detail=detail, fetched_at=fetched_at)


def _codex_label(seconds: int | None, position: Literal["primary", "secondary"]) -> str:
    if seconds is None:
        return "5h" if position == "primary" else "Week"
    if seconds == 18000:
        return "5h"
    if seconds == 604800:
        return "Week"
    if seconds < 86400:
        return f"{max(1, seconds // 3600)}h"
    return f"{seconds // 86400}d"


def _codex_window(window: _CodexWindow, position: Literal["primary", "secondary"], fetched_at: datetime) -> QuotaWindow:
    resets_at: datetime | None = None
    if window.reset_at is not None:
        resets_at = _epoch_seconds(window.reset_at)
    elif window.reset_after_seconds is not None:
        resets_at = fetched_at + timedelta(seconds=window.reset_after_seconds)
    return QuotaWindow(
        label=_codex_label(window.limit_window_seconds, position),
        used_percent=_clamp(window.used_percent),
        resets_at=resets_at,
        window_seconds=window.limit_window_seconds,
    )


def parse_codex_usage(payload: object, fetched_at: datetime) -> QuotaReading:
    """Codex ``/wham/usage`` payload. Malformed input yields an error reading, never an exception."""
    try:
        data = _CodexUsage.model_validate(payload)
    except ValidationError:
        return _failed("malformed_response", fetched_at)

    windows: list[QuotaWindow] = []
    detail: str | None = None
    rate_limit = data.rate_limit
    if rate_limit is not None:
        if rate_limit.limit_reached:
            detail = "limit reached"
        for position, window in (("primary", rate_limit.primary_window), ("secondary", rate_limit.secondary_window)):
            if window is not None:
                windows.append(_codex_window(window, position, fetched_at))

    # Hermes rule: a balance (and "unlimited") is shown only when the account has credits. A plan
    # account without credits reports has_credits=false with a "0" balance, which is not a balance to show.
    has_credits = data.credits is not None and data.credits.has_credits is True
    balance = _balance(data.credits.balance) if has_credits and data.credits is not None else None
    if balance is not None and not (math.isfinite(balance) and balance >= 0):
        balance = None
    banked_resets = (data.rate_limit_reset_credits.available_count or 0) if data.rate_limit_reset_credits else 0
    credits: QuotaCredits | None = None
    if has_credits or banked_resets > 0:
        credits = QuotaCredits(
            balance=balance,
            unlimited=has_credits and bool(data.credits is not None and data.credits.unlimited),
            banked_resets=banked_resets,
        )
    return QuotaReading(
        status="ok",
        plan=data.plan_type,
        detail=detail,
        windows=windows,
        credits=credits,
        fetched_at=fetched_at,
    )


_CLAUDE_WINDOWS: tuple[tuple[str, str, int, Literal["account", "model"]], ...] = (
    ("five_hour", "5h", 18000, "account"),
    ("seven_day", "Week", 604800, "account"),
    ("seven_day_opus", "Week (Opus)", 604800, "model"),
    ("seven_day_sonnet", "Week (Sonnet)", 604800, "model"),
)


def parse_claude_usage(payload: object, fetched_at: datetime) -> QuotaReading:
    """Claude ``/api/oauth/usage`` payload. Extra usage arrives in minor units (cents)."""
    try:
        data = _ClaudeUsage.model_validate(payload)
    except ValidationError:
        return _failed("malformed_response", fetched_at)

    windows: list[QuotaWindow] = []
    for field_name, label, seconds, scope in _CLAUDE_WINDOWS:
        window: _ClaudeWindow | None = getattr(data, field_name)
        if window is None:
            continue
        windows.append(
            QuotaWindow(
                label=label,
                used_percent=_clamp(window.utilization),
                resets_at=_aware(window.resets_at) if window.resets_at is not None else None,
                window_seconds=seconds,
                scope=scope,
            )
        )

    budget: QuotaBudget | None = None
    extra = data.extra_usage
    if extra is not None and extra.is_enabled and extra.monthly_limit is not None and extra.used_credits is not None:
        budget = QuotaBudget(
            used=extra.used_credits / 100,
            limit=extra.monthly_limit / 100,
            currency=extra.currency or "USD",
        )
    return QuotaReading(status="ok", windows=windows, budget=budget, fetched_at=fetched_at)


def _zai_reset(value: float | str | None) -> datetime | None:
    """Z.ai nextResetTime is epoch milliseconds (number or numeric string) or an ISO string."""
    if value is None:
        return None
    if isinstance(value, float):
        return _epoch_seconds(value / 1000)
    try:
        return _epoch_seconds(float(value) / 1000)
    except ValueError:
        pass
    try:
        return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _zai_window(limit: _ZaiLimit) -> QuotaWindow:
    if limit.type == "TIME_LIMIT":
        label = "Monthly"
        seconds = None
    else:
        suffix = _ZAI_UNIT_SUFFIX.get(limit.unit or -1, "")
        label = f"Tokens ({limit.number}{suffix})" if limit.number is not None else "Tokens"
        unit_seconds = _ZAI_UNIT_SECONDS.get(limit.unit or -1)
        seconds = unit_seconds * limit.number if unit_seconds and limit.number else None
    return QuotaWindow(
        label=label,
        used_percent=_clamp(limit.percentage),
        resets_at=_zai_reset(limit.next_reset_time),
        window_seconds=seconds,
    )


def parse_zai_usage(payload: object, fetched_at: datetime) -> QuotaReading:
    """Z.ai ``/api/monitor/usage/quota/limit`` payload. ``success`` must be true and ``code`` 200."""
    try:
        data = _ZaiUsage.model_validate(payload)
    except ValidationError:
        return _failed("malformed_response", fetched_at)

    if data.success is not True or data.code != 200:
        # The provider's own msg is never surfaced: a Z.ai error envelope can echo the configured key.
        return _failed("provider_error", fetched_at, detail="provider reported an error")

    plan_data = data.data
    if plan_data is None:
        return QuotaReading(status="ok", fetched_at=fetched_at)
    windows = [_zai_window(limit) for limit in plan_data.limits]
    return QuotaReading(
        status="ok",
        plan=plan_data.plan_name or plan_data.plan,
        windows=windows,
        fetched_at=fetched_at,
    )


# --- Targets -------------------------------------------------------------------------------------

FetchFn = Callable[[HttpGet, datetime], Awaitable[QuotaReading]]


@dataclass(frozen=True)
class QuotaTarget:
    """One provider credential the server can ask about. ``key`` is internal and never returned."""

    provider: str
    key: str
    lanes: tuple[str, ...]
    fetch: FetchFn | None
    status: Literal["unsupported", "unavailable"] | None = None
    reason: QuotaReason | None = None


async def _run_fetch(
    http_get: HttpGet,
    url: str,
    headers: Mapping[str, str],
    fetched_at: datetime,
    parse: Callable[[object, datetime], QuotaReading],
) -> QuotaReading:
    """GET, map HTTP and transport failures to a reading, then parse. Never raises."""
    try:
        result = await http_get(url, headers)
    except TimeoutError:
        return _failed("timeout", fetched_at)
    except Exception as exc:
        logger.warning("LLM quota fetch failed before a response: %s", type(exc).__name__)
        return _failed("internal_error", fetched_at)

    if result.status in (401, 403):
        if result.status == 403 and "user:profile" in result.body:
            return QuotaReading(status="unavailable", reason="token_scope", fetched_at=fetched_at)
        return _failed("auth_failed", fetched_at)
    if not 200 <= result.status < 300:
        return _failed("http_error", fetched_at, detail=f"HTTP {result.status}")
    try:
        payload = json.loads(result.body)
    except ValueError:
        return _failed("malformed_response", fetched_at)
    try:
        return parse(payload, fetched_at)
    except Exception:
        return _failed("internal_error", fetched_at)


# Refresh tasks that may outlive a cancelled quota fetch; held here so they are not garbage-collected mid-run.
# Process-global and reachable from any loop's thread, so add/discard go through a threading lock (never iterated).
_DETACHED_REFRESHES: set[asyncio.Task[None]] = set()
_DETACHED_REFRESHES_LOCK = threading.Lock()


def _settle_detached_refresh(task: asyncio.Task[None]) -> None:
    with _DETACHED_REFRESHES_LOCK:
        _DETACHED_REFRESHES.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.debug("Codex token refresh started by the quota probe failed: %s", type(task.exception()).__name__)


async def _refresh_shielded(impl: "CodexLLM") -> None:
    """The lane's proactive token refresh, shielded from the quota deadline.

    The refresh rotates the refresh token: cancelling it between the provider's answer and the
    in-memory update would leave the live Codex lane holding a refresh token that no longer works.
    When the deadline fires, the refresh runs to completion in the background and the fetch reports
    ``timeout``; the next Codex call sees the new tokens.
    """
    task = asyncio.create_task(impl._ensure_fresh_token())
    with _DETACHED_REFRESHES_LOCK:
        _DETACHED_REFRESHES.add(task)
    task.add_done_callback(_settle_detached_refresh)
    await asyncio.shield(task)


async def _fetch_codex(impl: object, http_get: HttpGet, fetched_at: datetime) -> QuotaReading:
    from hindsight_api.engine.providers.codex_auth import CodexRefreshExpiredError
    from hindsight_api.engine.providers.codex_llm import CodexLLM

    if not isinstance(impl, CodexLLM):
        return _failed("internal_error", fetched_at)
    try:
        await _refresh_shielded(impl)
        headers = {k: v for k, v in impl._build_request_headers().items() if k != "Content-Type"}
    except CodexRefreshExpiredError:
        return _failed("auth_failed", fetched_at)
    except Exception:
        return _failed("internal_error", fetched_at)
    headers["Accept"] = "application/json"

    base = impl.base_url.rstrip("/").removesuffix("/codex")
    url = base + "/wham/usage" if "/backend-api" in base else base + "/api/codex/usage"
    return await _run_fetch(http_get, url, headers, fetched_at, parse_codex_usage)


async def _fetch_claude(token: str, http_get: HttpGet, fetched_at: datetime) -> QuotaReading:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20",
    }
    return await _run_fetch(http_get, CLAUDE_USAGE_URL, headers, fetched_at, parse_claude_usage)


async def _fetch_zai(origin: str, api_key: str, http_get: HttpGet, fetched_at: datetime) -> QuotaReading:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    return await _run_fetch(http_get, origin + "/api/monitor/usage/quota/limit", headers, fetched_at, parse_zai_usage)


def _digest(secret: str) -> str:
    """Opaque dedupe key component; the secret itself is never stored in a key."""
    return hashlib.sha256(secret.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class _LaneMember:
    """One LLM member the engine is configured with, and the lane label it serves."""

    label: str
    member: object


def _lane_members(memory: object) -> list[_LaneMember]:
    """Every lane member the engine is configured with. Chain members are ``lane[n]``, 1-based."""
    lane_attrs = (
        ("default", "_llm_config"),
        ("retain", "_retain_llm_config"),
        ("reflect", "_reflect_llm_config"),
        ("consolidation", "_consolidation_llm_config"),
        ("mental_model_refresh", "_mental_model_refresh_llm_override"),
    )
    found: list[_LaneMember] = []
    for label, attr in lane_attrs:
        lane = getattr(memory, attr, None)
        if lane is None:
            continue
        if isinstance(lane, MultiLLMProvider):
            found.extend(
                _LaneMember(label=f"{label}[{index}]", member=member)
                for index, member in enumerate(lane.members, start=1)
            )
        else:
            found.append(_LaneMember(label=label, member=lane))
    return found


def _classify(member: object, env_token: str, codex_cls: type) -> QuotaTarget | None:
    provider = str(getattr(member, "provider", "") or "")
    impl = getattr(member, "_provider_impl", None)
    if isinstance(impl, codex_cls):
        return QuotaTarget(
            provider=provider,
            key=f"codex:{impl._auth_file}",
            lanes=(),
            fetch=partial(_fetch_codex, impl),
        )
    if provider == "claude-code":
        lane_key = getattr(member, "api_key", None)
        lane_token = lane_key if isinstance(lane_key, str) and lane_key.startswith(CLAUDE_OAUTH_TOKEN_PREFIX) else ""
        token = env_token or lane_token
        if not token:
            return QuotaTarget(
                provider=provider,
                key="claude-code:no-credentials",
                lanes=(),
                fetch=None,
                status="unavailable",
                reason="no_credentials",
            )
        return QuotaTarget(
            provider=provider,
            key=f"claude:{_digest(token)}",
            lanes=(),
            fetch=partial(_fetch_claude, token),
        )
    if provider == "zai":
        parts = urlsplit(str(getattr(member, "base_url", "") or ""))
        origin = f"{parts.scheme}://{parts.netloc}"
        if "/coding/" not in parts.path.rstrip("/") + "/":
            return QuotaTarget(
                provider=provider,
                key=f"zai-paas:{origin}",
                lanes=(),
                fetch=None,
                status="unsupported",
                reason="not_a_plan_endpoint",
            )
        api_key = str(getattr(member, "api_key", "") or "")
        return QuotaTarget(
            provider=provider,
            key=f"zai:{origin}:{_digest(api_key)}",
            lanes=(),
            fetch=partial(_fetch_zai, origin, api_key),
        )
    if provider in UNSUPPORTED_PLAN_PROVIDERS:
        return QuotaTarget(
            provider=provider,
            key=f"unsupported:{provider}",
            lanes=(),
            fetch=None,
            status="unsupported",
            reason="no_usage_endpoint",
        )
    return None


def collect_quota_targets(memory: object, settings: QuotaSettings) -> list[QuotaTarget]:
    """Quota targets for the engine's lanes, deduplicated by credential and labelled with every lane using it.

    Metered API-key and local providers are not listed. Per-bank and per-operation overrides are not walked.
    """
    from hindsight_api.engine.providers.codex_llm import CodexLLM

    env_token = settings.claude_oauth_token.get_secret_value() if settings.claude_oauth_token is not None else ""
    targets: dict[str, QuotaTarget] = {}
    lanes: dict[str, list[str]] = {}
    for lane in _lane_members(memory):
        target = _classify(lane.member, env_token, CodexLLM)
        if target is None:
            continue
        targets.setdefault(target.key, target)
        used_by = lanes.setdefault(target.key, [])
        if lane.label not in used_by:
            used_by.append(lane.label)
    return [replace(target, lanes=tuple(lanes[target.key])) for target in targets.values()]


# --- Cache ---------------------------------------------------------------------------------------


@dataclass
class _Slot:
    current: QuotaReading | None = None
    last_good: QuotaReading | None = None
    expires_at: float = 0.0
    failures: int = 0
    retry_at: float = 0.0


class QuotaCache:
    """In-process cache per target key. Success is valid for ``ttl_seconds``; failures back off
    exponentially to ``backoff_cap_seconds`` and serve the last good reading marked stale."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        *,
        ttl_seconds: float = 60.0,
        backoff_base_seconds: float = 60.0,
        backoff_cap_seconds: float = 900.0,
        fetch_timeout_seconds: float = QUOTA_FETCH_TIMEOUT_SECONDS,
    ) -> None:
        self.clock = clock
        self._fetch_timeout = fetch_timeout_seconds
        self._ttl = ttl_seconds
        self._backoff_base = backoff_base_seconds
        self._backoff_cap = backoff_cap_seconds
        self._slots: dict[str, _Slot] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str, fetch: Callable[[], Awaitable[QuotaReading]]) -> QuotaReading:
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            slot = self._slots.setdefault(key, _Slot())
            now = self.clock()
            if slot.current is not None and slot.failures == 0 and now < slot.expires_at:
                return slot.current
            if slot.failures > 0 and now < slot.retry_at:
                return self._degraded(slot)
            try:
                # One wall-clock deadline per target, covering the Codex token refresh, the GET and the body.
                # The session's own timeout is per phase (each socket read), so a provider that trickles
                # bytes would otherwise hold this key's lock, and the whole /quota gather, indefinitely.
                async with asyncio.timeout(self._fetch_timeout):
                    reading = await fetch()
            except TimeoutError:
                reading = _failed("timeout", datetime.fromtimestamp(now, UTC))
            except Exception:
                reading = _failed("internal_error", datetime.fromtimestamp(now, UTC))
            if reading.status == "ok":
                slot.current = reading
                slot.last_good = reading
                slot.expires_at = now + self._ttl
                slot.failures = 0
                return reading
            slot.failures += 1
            slot.retry_at = now + min(self._backoff_base * 2 ** (slot.failures - 1), self._backoff_cap)
            slot.current = reading
            return self._degraded(slot)

    @staticmethod
    def _degraded(slot: _Slot) -> QuotaReading:
        if slot.last_good is not None:
            return slot.last_good.model_copy(update={"stale": True})
        assert slot.current is not None  # set before any failure is recorded
        return slot.current


# --- Service -------------------------------------------------------------------------------------


class LLMQuotaService:
    """Builds the server-wide quota snapshot. Created once at route registration."""

    def __init__(
        self,
        memory: "MemoryEngine",
        settings: QuotaSettings,
        http_get: HttpGet | None = None,
        cache: QuotaCache | None = None,
    ) -> None:
        self._memory: "MemoryEngine" = memory
        self._settings = settings
        self._http_get: HttpGet = http_get or default_http_get
        self._cache = cache or QuotaCache()

    async def snapshot(self, bank_id: str, request_context: "RequestContext") -> LLMQuotaResponse:
        """Bank readability check, then the server-wide snapshot. Only the bank check can raise."""
        from hindsight_api.engine.llm_cost_store import check_bank_readable

        await check_bank_readable(self._memory, bank_id, request_context, "read the server LLM quota")
        return await self.fetch_all(bank_id)

    async def fetch_all(self, bank_id: str) -> LLMQuotaResponse:
        """Fetch every target concurrently (no bank check). Nothing raises."""
        try:
            targets = collect_quota_targets(self._memory, self._settings)
        except Exception:
            logger.warning("LLM quota target discovery failed")
            targets = []
        providers = await asyncio.gather(*(self._one(target) for target in targets))
        return LLMQuotaResponse(bank_id=bank_id, providers=list(providers))

    async def _one(self, target: QuotaTarget) -> ProviderQuota:
        try:
            if target.fetch is None:
                return ProviderQuota(
                    provider=target.provider,
                    lanes=list(target.lanes),
                    status=target.status or "unavailable",
                    reason=target.reason,
                )
            fetch = target.fetch
            now = datetime.fromtimestamp(self._cache.clock(), UTC)
            reading = await self._cache.get(target.key, lambda: fetch(self._http_get, now))
            return ProviderQuota(
                provider=target.provider,
                lanes=list(target.lanes),
                status=reading.status,
                reason=reading.reason,
                detail=reading.detail,
                plan=reading.plan,
                windows=reading.windows,
                credits=reading.credits,
                budget=reading.budget,
                fetched_at=reading.fetched_at,
                stale=reading.stale,
            )
        except Exception:
            logger.warning("LLM quota provider %s failed unexpectedly", target.provider)
            return ProviderQuota(
                provider=target.provider,
                lanes=list(target.lanes),
                status="error",
                reason="internal_error",
            )

"""Fork-only lane routing for the reranker failover chain (fork: rerank-routing).

Upstream tries one failover chain in member order on every call. This module adds two lanes
(foreground / background) with their own member order, a per-member in-flight cap that skips a
busy member instead of queueing, an optional HTTP 429 cooldown, a per-lane reranker candidate
cap, and per-member per-lane candidate caps (a capped member reranks only the RRF head and the rest
keep RRF order behind it). Everything lives here so upstream files only carry a few hooks marked ``# fork: rerank-routing``;
see FORK-RERANK-ROUTING.md at the repo root.
"""

import asyncio
import collections
import contextlib
import contextvars
import email.utils
import functools
import logging
import math
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Generic, TypeVar

from ..config import reranker_routing_env  # the only env reader; see config.py
from .cross_encoder import CrossEncoderModel, MultiCrossEncoder, RerankTimeoutError, _served_provider
from .remote_retry import status_code_of

logger = logging.getLogger(__name__)

LANE_FOREGROUND = "foreground"
LANE_BACKGROUND = "background"
LANES = (LANE_FOREGROUND, LANE_BACKGROUND)

ENV_FOREGROUND_ORDER = "HINDSIGHT_API_RERANKER_FOREGROUND_ORDER"
ENV_BACKGROUND_ORDER = "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER"
ENV_FOREGROUND_MAX_CANDIDATES = "HINDSIGHT_API_RERANKER_FOREGROUND_MAX_CANDIDATES"
ENV_BACKGROUND_MAX_CANDIDATES = "HINDSIGHT_API_RERANKER_BACKGROUND_MAX_CANDIDATES"
ENV_COOLDOWN_ENABLED = "HINDSIGHT_API_RERANKER_COOLDOWN_ENABLED"
ENV_COOLDOWN_DEFAULT_SECONDS = "HINDSIGHT_API_RERANKER_COOLDOWN_DEFAULT_SECONDS"
ENV_COOLDOWN_PER_MINUTE_SECONDS = "HINDSIGHT_API_RERANKER_COOLDOWN_PER_MINUTE_SECONDS"
ENV_COOLDOWN_MAX_SECONDS = "HINDSIGHT_API_RERANKER_COOLDOWN_MAX_SECONDS"

DEFAULT_COOLDOWN_DEFAULT_SECONDS = 30.0
DEFAULT_COOLDOWN_PER_MINUTE_SECONDS = 60.0
DEFAULT_COOLDOWN_MAX_SECONDS = 86400.0
# Upper bound for every cooldown duration env var. Keeps the until-time finite and well inside datetime's range.
MAX_COOLDOWN_ENV_SECONDS = 7 * 86400.0

_ORDER_ENV = {LANE_FOREGROUND: ENV_FOREGROUND_ORDER, LANE_BACKGROUND: ENV_BACKGROUND_ORDER}


DEFAULT_BACKGROUND_QUEUE_MAX = 2  # when QUEUE_SECONDS > 0 and QUEUE_MAX is unset: how many background requests may wait for one member. Bounds how many worker slots can sit blocked on one member.
QUEUE_POLL_INITIAL_SECONDS = (
    0.05  # a waiter re-checks its ticket this often at first, doubling up to QUEUE_POLL_MAX_SECONDS
)
QUEUE_POLL_MAX_SECONDS = 0.5
# A ticket not polled for this long belongs to a waiter whose loop died without unwinding; it is dropped so it cannot block the head of the queue forever.
QUEUE_STALE_TICKET_SECONDS = 60.0
_QUEUE_ACQUIRED = "acquired"  # internal verdict of one poll: the slot was taken
_QUEUE_WAITING = "waiting"  # internal verdict of one poll: still queued, sleep and poll again


class _QueueTicket:
    """One background request waiting for a member's in-flight slot. Compared by identity (no __eq__): deque.remove and `in` must match this object itself."""

    __slots__ = ("last_seen",)

    def __init__(self, last_seen: float) -> None:
        self.last_seen = last_seen  # monotonic time of the waiter's last poll


def member_env_name(index: int, suffix: str) -> str:
    """Env var ``HINDSIGHT_API_RERANKER_<suffix>`` for member 0, ``HINDSIGHT_API_RERANKER_<index>_<suffix>`` otherwise (same scheme as RerankerMemberConfig)."""
    if index == 0:
        return f"HINDSIGHT_API_RERANKER_{suffix}"
    return f"HINDSIGHT_API_RERANKER_{index}_{suffix}"


def max_inflight_env_name(index: int) -> str:
    """Env var that caps in-flight calls for member ``index`` (same scheme as RerankerMemberConfig)."""
    return member_env_name(index, "MAX_INFLIGHT")


def queue_seconds_env_name(index: int) -> str:
    """Env var for how long a background request may wait for member ``index`` at its in-flight cap."""
    return member_env_name(index, "BACKGROUND_QUEUE_SECONDS")


def queue_max_env_name(index: int) -> str:
    """Env var for how many background requests may wait for member ``index`` at once."""
    return member_env_name(index, "BACKGROUND_QUEUE_MAX")


def member_max_candidates_env_name(index: int, lane: str) -> str:
    """Env var for the candidate cap member ``index`` applies in ``lane``. Always numbered, member 0 included: the unnumbered name is v1's lane-wide cap (ENV_FOREGROUND_MAX_CANDIDATES / ENV_BACKGROUND_MAX_CANDIDATES)."""
    return f"HINDSIGHT_API_RERANKER_{index}_{lane.upper()}_MAX_CANDIDATES"


# The lane of the current rerank call. Set by the worker's background entry point and inherited
# by child tasks, so nested recalls (consolidation, mental-model refresh) pick it up without plumbing.
_rerank_lane: contextvars.ContextVar[str] = contextvars.ContextVar("hindsight_rerank_lane", default=LANE_FOREGROUND)


def current_lane() -> str:
    """The rerank lane the current context runs in."""
    return _rerank_lane.get()


@contextlib.contextmanager
def background_lane() -> Iterator[None]:
    """Run the enclosed block in the background lane, restoring the previous lane on exit."""
    token = _rerank_lane.set(LANE_BACKGROUND)
    try:
        yield
    finally:
        _rerank_lane.reset(token)


def rerank_background_lane(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Run the decorated coroutine (MemoryEngine.execute_task) in the background lane; nested recalls inherit it."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with background_lane():
            return await func(*args, **kwargs)

    # Marker read by the hook-presence tests; a rebase that drops the decorator fails them.
    wrapper._rerank_background_lane = True  # type: ignore[attr-defined]
    return wrapper


def parse_lane_order(raw: str | None, member_count: int, env_name: str) -> tuple[int, ...]:
    """Parse a comma-separated member order. Unset means natural order; subsets are allowed."""
    if raw is None:
        return tuple(range(member_count))
    if not raw.strip():
        raise ValueError(f"Invalid {env_name}={raw!r}: empty; list member indexes such as 0,1,2 or unset it")

    order: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"Invalid {env_name}={raw!r}: empty entry; check for stray or doubled commas")
        try:
            index = int(item)
        except ValueError:
            raise ValueError(f"Invalid {env_name}={raw!r}: {item!r} is not a member index") from None
        if index < 0 or index >= member_count:
            raise ValueError(
                f"Invalid {env_name}={raw!r}: member {index} does not exist "
                f"(configured members are 0..{member_count - 1})"
            )
        if index in order:
            raise ValueError(f"Invalid {env_name}={raw!r}: member {index} is listed twice")
        order.append(index)
    return tuple(order)


def _parse_non_negative_int(raw: str | None, env_name: str) -> int:
    """Parse an int >= 0; unset or blank means 0."""
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"Invalid {env_name}={raw!r}: expected a non-negative integer") from None
    if value < 0:
        raise ValueError(f"Invalid {env_name}={raw!r}: expected a non-negative integer")
    return value


def _parse_optional_non_negative_int(raw: str | None, env_name: str) -> int | None:
    """Parse an int >= 0 like _parse_non_negative_int, but unset or blank means None so an explicit 0 stays distinguishable."""
    if raw is None or not raw.strip():
        return None
    return _parse_non_negative_int(raw, env_name)


def _parse_non_negative_float(
    raw: str | None, env_name: str, default: float, *, ceiling: float | None = MAX_COOLDOWN_ENV_SECONDS
) -> float:
    """Parse a finite number of seconds >= 0; unset or blank means ``default``. inf and nan always fail. Cooldown durations pass the MAX_COOLDOWN_ENV_SECONDS ceiling because their until-times are datetimes; background queue waits pass ``ceiling=None`` because they only use monotonic arithmetic."""
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value >= 0 and (ceiling is None or value <= ceiling)):
        bound = (
            "a finite number of seconds, 0 or more" if ceiling is None else f"a number of seconds from 0 to {ceiling:g}"
        )
        raise ValueError(f"Invalid {env_name}={raw!r}: expected {bound}")
    return value


def _parse_bool(raw: str | None) -> bool:
    """Parse a boolean flag; unset is false."""
    if raw is None:
        return False
    return raw.strip().lower() in ("true", "1", "yes")


_T = TypeVar("_T")


@dataclass(frozen=True)
class LaneValues(Generic[_T]):
    """One value per rerank lane. ``for_lane`` rejects an unknown lane with ValueError instead of a late KeyError."""

    foreground: _T
    background: _T

    def for_lane(self, lane: str) -> _T:
        if lane == LANE_FOREGROUND:
            return self.foreground
        if lane == LANE_BACKGROUND:
            return self.background
        raise ValueError(f"unknown rerank lane {lane!r}; expected one of {LANES}")


@dataclass(frozen=True)
class Cooldown:
    """A member's cooldown from a 429: the epoch time it may be tried again, and a short body-free reason for logs."""

    until: float
    reason: str


@dataclass(frozen=True)
class AcquireResult:
    """Outcome of trying one member for a request: the skip reason (None when the slot was taken) and the seconds spent waiting in the background queue."""

    skip_reason: str | None
    waited_seconds: float


@dataclass(frozen=True)
class RoutingSettings:
    """Resolved routing env for a chain of N members.

    ``configured=False`` means routing is off and the plain upstream chain is used.
    """

    lane_orders: LaneValues[tuple[int, ...]]
    max_inflight: tuple[int, ...]
    cooldown_enabled: bool
    cooldown_default_seconds: float
    cooldown_per_minute_seconds: float
    cooldown_max_seconds: float
    # v2: per-member background queue (index-aligned with members) and per-member, per-lane candidate caps.
    queue_seconds: tuple[float, ...]
    queue_max: tuple[
        int, ...
    ]  # effective: the explicit value, else DEFAULT_BACKGROUND_QUEUE_MAX when queue_seconds > 0, else 0
    queue_max_explicit: tuple[int | None, ...]  # raw parse result, kept for startup warnings
    member_max_candidates: LaneValues[tuple[int, ...]]  # per lane, index-aligned with members; 0 = no member cap
    configured: bool

    @classmethod
    def from_env(cls, member_count: int, environ: Mapping[str, str] | None = None) -> "RoutingSettings":
        """Read and validate the routing env vars for ``member_count`` members."""
        env = reranker_routing_env() if environ is None else environ
        lane_orders = LaneValues(
            foreground=parse_lane_order(env.get(ENV_FOREGROUND_ORDER), member_count, ENV_FOREGROUND_ORDER),
            background=parse_lane_order(env.get(ENV_BACKGROUND_ORDER), member_count, ENV_BACKGROUND_ORDER),
        )
        max_inflight = tuple(
            _parse_non_negative_int(env.get(max_inflight_env_name(i)), max_inflight_env_name(i))
            for i in range(member_count)
        )
        queue_seconds = tuple(
            _parse_non_negative_float(env.get(queue_seconds_env_name(i)), queue_seconds_env_name(i), 0.0, ceiling=None)
            for i in range(member_count)
        )
        queue_max_explicit = tuple(
            _parse_optional_non_negative_int(env.get(queue_max_env_name(i)), queue_max_env_name(i))
            for i in range(member_count)
        )
        queue_max = tuple(
            explicit if explicit is not None else (DEFAULT_BACKGROUND_QUEUE_MAX if seconds > 0 else 0)
            for explicit, seconds in zip(queue_max_explicit, queue_seconds, strict=True)
        )

        def member_caps(lane: str) -> tuple[int, ...]:
            names = [member_max_candidates_env_name(i, lane) for i in range(member_count)]
            return tuple(_parse_non_negative_int(env.get(name), name) for name in names)

        member_max_candidates = LaneValues(
            foreground=member_caps(LANE_FOREGROUND), background=member_caps(LANE_BACKGROUND)
        )
        cooldown_enabled = _parse_bool(env.get(ENV_COOLDOWN_ENABLED))
        configured = (
            any(env.get(_ORDER_ENV[lane]) is not None for lane in LANES)
            or any(max_inflight)
            or cooldown_enabled
            or any(seconds > 0 for seconds in queue_seconds)
            or any(explicit for explicit in queue_max_explicit if explicit is not None)
            or any(any(member_max_candidates.for_lane(lane)) for lane in LANES)
        )
        return cls(
            lane_orders=lane_orders,
            max_inflight=max_inflight,
            cooldown_enabled=cooldown_enabled,
            cooldown_default_seconds=_parse_non_negative_float(
                env.get(ENV_COOLDOWN_DEFAULT_SECONDS), ENV_COOLDOWN_DEFAULT_SECONDS, DEFAULT_COOLDOWN_DEFAULT_SECONDS
            ),
            cooldown_per_minute_seconds=_parse_non_negative_float(
                env.get(ENV_COOLDOWN_PER_MINUTE_SECONDS),
                ENV_COOLDOWN_PER_MINUTE_SECONDS,
                DEFAULT_COOLDOWN_PER_MINUTE_SECONDS,
            ),
            cooldown_max_seconds=_parse_non_negative_float(
                env.get(ENV_COOLDOWN_MAX_SECONDS), ENV_COOLDOWN_MAX_SECONDS, DEFAULT_COOLDOWN_MAX_SECONDS
            ),
            queue_seconds=queue_seconds,
            queue_max=queue_max,
            queue_max_explicit=queue_max_explicit,
            member_max_candidates=member_max_candidates,
            configured=configured,
        )


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def next_utc_midnight(now: float) -> float:
    """OpenRouter resets per-day free quotas at 00:00 UTC."""
    today = datetime.fromtimestamp(now, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return (today + timedelta(days=1)).timestamp()


_BODY_RESET_RE = re.compile(
    r"x-ratelimit-reset\W+(\d+(?:\.\d+)?)", re.IGNORECASE
)  # OpenRouter echoes its limit headers in error.metadata.headers


def _error_text(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    return f"{'' if body is None else body} {exc}"


def _response_headers(exc: BaseException) -> dict[str, str]:
    # UpstreamHTTPError carries .headers (the OpenRouter / Cohere-compatible path); httpx-based errors carry .response.headers.
    headers = getattr(exc, "headers", None)
    if headers is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "items"):
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def _reset_value_to_delay(value: float, now: float) -> float:
    """X-RateLimit-Reset is epoch milliseconds (OpenRouter), epoch seconds, or a delay in seconds."""
    if value > 1e11:
        return value / 1000.0 - now
    if value > 1e9:
        return value - now
    return value


def reset_delay_from(exc: BaseException, now: float) -> float | None:
    """Seconds until the upstream says the limit resets, or None."""
    headers = _response_headers(exc)
    candidates: list[float] = []
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            candidates.append(float(retry_after))
        except ValueError:
            try:
                candidates.append(email.utils.parsedate_to_datetime(retry_after).timestamp() - now)
            except (TypeError, ValueError, IndexError, OverflowError):
                pass
    reset = headers.get("x-ratelimit-reset")
    if reset:
        try:
            candidates.append(_reset_value_to_delay(float(reset), now))
        except ValueError:
            pass
    if not candidates:
        match = _BODY_RESET_RE.search(_error_text(exc))
        if match:
            candidates.append(_reset_value_to_delay(float(match.group(1)), now))
    return next((delay for delay in candidates if delay > 0), None)


def _http_carrier(exc: BaseException) -> BaseException:
    """The exception that carries the HTTP status: exc itself, or the upstream error it wraps. TEI re-raises UpstreamHTTPError as RuntimeError, so look at most two levels down the cause/context chain."""
    current: BaseException | None = exc
    for _ in range(3):
        if current is None:
            break
        if status_code_of(current) is not None:
            return current
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return exc


def rate_limit_cooldown(exc: BaseException, now: float, settings: RoutingSettings) -> Cooldown | None:
    """For an HTTP 429, the epoch time the member may be tried again and why; None otherwise. An upstream reset time wins; then the OpenRouter body markers."""
    carrier = _http_carrier(exc)
    if status_code_of(carrier) != 429:
        return None
    delay = reset_delay_from(carrier, now)
    if delay is not None:
        return Cooldown(now + min(max(delay, 1.0), settings.cooldown_max_seconds), "reset header")
    text = _error_text(carrier).lower()
    if "per-day" in text:
        return Cooldown(next_utc_midnight(now), "per-day quota")
    if "per-min" in text:
        return Cooldown(now + settings.cooldown_per_minute_seconds, "per-minute limit")
    return Cooldown(now + settings.cooldown_default_seconds, "rate limited")


class RerankerRateLimitedError(Exception):
    """A member answered HTTP 429 and is cooling down. Raised in place of the provider error so the member's own retry stops at once: it carries no status code, so remote_retry treats it as permanent, and retrying a spent quota only adds latency before the chain moves on."""

    def __init__(self, index: int, until: float) -> None:
        super().__init__(f"reranker member {index} is rate limited until {_iso(until)}")
        self.index = index
        self.until = until


class RerankerUnavailableError(RuntimeError):
    """Every member in the lane order was skipped (in-flight cap, cooldown, queue_full or queue_timeout) and none was tried."""


@functools.lru_cache(maxsize=16)
def _lane_caps(raw_foreground: str | None, raw_background: str | None) -> LaneValues[int]:
    """Parse the per-lane candidate caps; cached on the raw strings so env changes are picked up."""
    return LaneValues(
        foreground=_parse_non_negative_int(raw_foreground, ENV_FOREGROUND_MAX_CANDIDATES),
        background=_parse_non_negative_int(raw_background, ENV_BACKGROUND_MAX_CANDIDATES),
    )


def lane_max_candidates() -> int:
    """The current lane's reranker candidate cap, or 0 to keep the budget-tier cap."""
    env = reranker_routing_env()
    caps = _lane_caps(env.get(ENV_FOREGROUND_MAX_CANDIDATES), env.get(ENV_BACKGROUND_MAX_CANDIDATES))
    return caps.for_lane(current_lane())


def lane_max_candidates_override(func: Callable[..., int]) -> Callable[..., int]:
    """Wrap _resolve_reranker_max_candidates: a per-lane cap > 0 replaces the budget-tier result."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> int:
        cap = lane_max_candidates()
        return cap if cap > 0 else func(*args, **kwargs)

    return wrapper


def _failure_label(exc: BaseException) -> str:
    """Short, body-free reason for logs: the HTTP status, or the exception class. Never the message."""
    if isinstance(exc, RerankerRateLimitedError):
        return "HTTP 429"
    status = status_code_of(_http_carrier(exc))
    return f"HTTP {status}" if status is not None else type(exc).__name__


class RerankerMemberCapReached(RerankTimeoutError):
    """A member answered, but only for the RRF head of the candidates it was capped to in this lane.

    Raised only by RoutedMultiCrossEncoder._predict, after the member succeeded on a truncated send. It is never raised by a member: a member's own RerankTimeoutError (e.g. LocalSTCrossEncoder) is still a member failure that the chain's ``except Exception`` fails over on, as upstream does. This is raised after the chain's member loop, so it never reaches that handler.

    It reuses upstream's partial-result channel. ``scores`` is exactly as long as the full candidate list: the head scores, then ``None`` for every tail candidate. CrossEncoderReranker.rerank normalizes the head as if it had been scored alone, then maps each ``None`` to raw = norm = weight = 0.0 and appends those candidates after the sorted head in input (RRF) order. The message carries no query or document text.
    """

    def __init__(self, scores: list[float | None], *, index: int, provider: str, lane: str, sent: int) -> None:
        # Not calling RerankTimeoutError.__init__: its message reports an exhausted budget, which this is not.
        Exception.__init__(
            self,
            f"reranker member {index} ({provider}) is capped at {sent}/{len(scores)} candidates in the {lane} lane",
        )
        self.scores = scores
        self.timeout = 0.0
        self.model_name = provider
        self.index = index
        self.provider = provider
        self.lane = lane
        self.sent = sent


class RoutedMultiCrossEncoder(MultiCrossEncoder):
    """Failover chain that walks the current lane's member order, skipping a member at its in-flight cap or cooling down after a 429. Background requests may wait FIFO for a member at its cap when its QUEUE_SECONDS > 0; foreground never waits. A member with a per-lane candidate cap (v2) is sent only the RRF head of the candidates and the rest come back as an unscored tail (see RerankerMemberCapReached). Gate state is per chain; the engine builds one chain per process, so both lanes share it, and it is safe across threads and event loops (all shared state sits under self._gate_lock)."""

    def __init__(
        self,
        members: list[CrossEncoderModel],
        settings: RoutingSettings,
        *,
        clock: Callable[[], float] = time.time,  # noqa: TID251 - cooldowns are logged as wall-clock deadlines
        monotonic: Callable[[], float] = time.monotonic,
        queue_poll_initial: float = QUEUE_POLL_INITIAL_SECONDS,
        queue_poll_max: float = QUEUE_POLL_MAX_SECONDS,
    ) -> None:
        super().__init__(members)
        if len(settings.max_inflight) != len(members) or len(settings.queue_seconds) != len(members):
            raise ValueError("RoutingSettings were built for a different number of members")
        self._settings = settings
        self._clock = clock  # wall clock, for cooldowns
        self._monotonic = monotonic  # for queue deadlines and staleness
        self._queue_poll_initial = queue_poll_initial
        self._queue_poll_max = queue_poll_max
        self._gate_lock = threading.Lock()
        self._inflight = [0] * len(members)
        self._cooldown_until = [0.0] * len(members)
        self._queues: list[collections.deque[_QueueTicket]] = [collections.deque() for _ in members]
        if settings.cooldown_enabled:
            # Members on the shared predict() path retry a 429 inside it; intercepting _predict turns it into a non-retryable error so a spent quota costs one request, not a retry budget.
            for index, member in enumerate(members):
                if type(member).predict is CrossEncoderModel.predict:
                    setattr(member, "_predict", self._intercept_rate_limit(index, member._predict))

    def _intercept_rate_limit(
        self, index: int, predict: Callable[[list[tuple[str, str]]], Awaitable[list[float]]]
    ) -> Callable[[list[tuple[str, str]]], Awaitable[list[float]]]:
        async def intercepted(pairs: list[tuple[str, str]]) -> list[float]:
            try:
                return await predict(pairs)
            except Exception as exc:
                until = self._enter_cooldown(index, exc)
                if until is None:
                    raise
                raise RerankerRateLimitedError(index, until) from exc

        return intercepted

    def _enter_cooldown(self, index: int, exc: BaseException) -> float | None:
        """Start (or extend) member index's cooldown when exc is a 429; return the until-time, or None if exc is not a 429."""
        now = self._clock()
        cooldown = rate_limit_cooldown(exc, now, self._settings)
        if cooldown is None:
            return None
        with self._gate_lock:
            previous = self._cooldown_until[index]
            self._cooldown_until[index] = max(previous, cooldown.until)
        if cooldown.until > previous:
            logger.warning(
                "Reranker member %d (%s) hit a rate limit (%s); cooling down until %s",
                index,
                self._members[index].provider_name,
                cooldown.reason,
                _iso(cooldown.until),
            )
        return max(previous, cooldown.until)

    def _purge_stale(self, index: int, now: float) -> int:
        """Drop tickets of member ``index`` not polled for QUEUE_STALE_TICKET_SECONDS, in place. Call only while holding self._gate_lock; returns the count so the caller can log once after releasing it."""
        queue = self._queues[index]
        stale = [ticket for ticket in queue if now - ticket.last_seen > QUEUE_STALE_TICKET_SECONDS]
        for ticket in stale:
            queue.remove(ticket)
        return len(stale)

    def _log_stale(self, index: int, dropped: int) -> None:
        if dropped:
            logger.warning(
                "Reranker member %d (%s) dropped %d background queue ticket(s) not polled for %.0f s",
                index,
                self._members[index].provider_name,
                dropped,
                QUEUE_STALE_TICKET_SECONDS,
            )

    def _try_acquire(self, index: int, lane: str) -> str | None:
        """Claim an in-flight slot; return the skip reason instead when the member is cooling down or full.

        No barging: a background arrival never overtakes queued background waiters, even when a slot is momentarily free between their polls. Foreground never waits, so it may take a free slot ahead of them; its alternative is the next member or rrf.
        """
        now = self._clock()
        ended = False
        dropped = 0
        with self._gate_lock:
            until = self._cooldown_until[index]
            if until > now:
                return "cooldown"
            if until:
                self._cooldown_until[index] = 0.0
                ended = True
            cap = self._settings.max_inflight[index]
            if cap:
                dropped = self._purge_stale(index, self._monotonic())
            if cap and (self._inflight[index] >= cap or (lane == LANE_BACKGROUND and self._queues[index])):
                reason = "inflight"
            else:
                self._inflight[index] += 1
                reason = None
        self._log_stale(index, dropped)
        if ended:
            logger.info("Reranker member %d (%s) cooldown ended", index, self._members[index].provider_name)
        return reason

    async def _wait_in_queue(self, index: int, deadline: float) -> str | None:
        """Wait FIFO for member ``index``'s in-flight slot until the monotonic ``deadline``. Returns None once the slot is taken (the caller then owns it and must release it), or the skip reason: queue_full, queue_timeout or cooldown.

        Invariants: the slot is taken in the same critical section that pops the ticket. The caller enters its try/finally release with no await in between, so neither cancellation nor timeout can leak a slot or a queue position. CancelledError is a BaseException, so only the finally here (never ``except Exception``) removes the ticket.
        """
        queue = self._queues[index]
        queue_max = self._settings.queue_max[index]
        ticket = _QueueTicket(self._monotonic())
        with self._gate_lock:
            dropped = self._purge_stale(index, ticket.last_seen)
            full = len(queue) >= queue_max
            if not full:
                queue.append(ticket)
        self._log_stale(index, dropped)
        if full:
            return "queue_full"
        delay = self._queue_poll_initial
        try:
            while True:
                now_m = self._monotonic()
                now_w = self._clock()
                # Checked before the head/slot test, so a poll at or after the deadline never takes a slot that freed up late.
                expired = now_m >= deadline
                dropped = 0
                with self._gate_lock:
                    ticket.last_seen = now_m
                    if self._cooldown_until[index] > now_w:
                        verdict = "cooldown"
                    else:
                        dropped = self._purge_stale(index, now_m)
                        verdict = _QUEUE_WAITING
                        if expired:
                            verdict = "queue_timeout"
                        elif ticket not in queue and len(queue) >= queue_max:
                            verdict = (
                                "queue_full"  # our ticket was purged while we were blocked, and the queue is full again
                            )
                        else:
                            if ticket not in queue:
                                queue.append(ticket)
                            cap = self._settings.max_inflight[index]
                            if queue[0] is ticket and (not cap or self._inflight[index] < cap):
                                queue.popleft()
                                self._inflight[index] += 1
                                verdict = _QUEUE_ACQUIRED
                self._log_stale(index, dropped)
                if verdict == _QUEUE_ACQUIRED:
                    return None
                if verdict != _QUEUE_WAITING:
                    return verdict
                # Sleep on this waiter's own loop; nothing loop-bound is shared with other threads or loops.
                await asyncio.sleep(min(delay, deadline - now_m))
                delay = min(delay * 2, self._queue_poll_max)
        finally:
            with self._gate_lock:
                with contextlib.suppress(ValueError):
                    queue.remove(ticket)

    async def _acquire(self, index: int, lane: str) -> AcquireResult:
        """Try member ``index`` for ``lane``. Only a background request blocked by the in-flight cap waits (when the member has QUEUE_SECONDS > 0 and a non-zero QUEUE_MAX)."""
        reason = self._try_acquire(index, lane)
        if (
            reason == "inflight"
            and lane == LANE_BACKGROUND
            and self._settings.queue_seconds[index] > 0
            and self._settings.queue_max[index] > 0
        ):
            started = self._monotonic()
            reason = await self._wait_in_queue(index, started + self._settings.queue_seconds[index])
            return AcquireResult(reason, self._monotonic() - started)
        return AcquireResult(reason, 0.0)

    def _release(self, index: int) -> None:
        with self._gate_lock:
            self._inflight[index] = max(0, self._inflight[index] - 1)

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs with the first member of the current lane's order that is free and answers usably."""
        lane = current_lane()
        started = time.perf_counter()
        skipped: list[str] = []
        queue_wait = 0.0
        last_exc: BaseException | None = None
        for index in self._settings.lane_orders.for_lane(lane):
            member = self._members[index]
            outcome = await self._acquire(index, lane)
            queue_wait += outcome.waited_seconds
            if outcome.skip_reason is not None:
                skipped.append(f"{index}:{outcome.skip_reason}")
                continue
            # The head is the first <cap> pairs, which are in RRF order. It is computed from the full `pairs` for every member, so a member that is skipped or fails never shrinks the list the next member gets. rrf ignores its cap: its scores are constant, so the list is only reseeded from RRF.
            cap = self._settings.member_max_candidates.for_lane(lane)[index]
            send = pairs[:cap] if cap and cap < len(pairs) and member.provider_name != "rrf" else pairs
            try:
                if not self._ready[index]:
                    await self._ensure_member_ready(index)
                scores = await member.predict(send)
                if len(scores) != len(send):
                    raise RuntimeError(f"returned {len(scores)} scores for {len(send)} pairs")
                # ±inf would reach ScoredResult unsanitized (upstream's NaN sanitizer does not cover it), so it fails this member over instead. NaN is left to upstream's sanitizer.
                if any(math.isinf(score) for score in scores):
                    raise RuntimeError("returned non-finite scores")
            except Exception as e:  # noqa: BLE001 - re-raised below if no member answers
                last_exc = e
                if self._settings.cooldown_enabled and not isinstance(e, RerankerRateLimitedError):
                    self._enter_cooldown(index, e)  # members that override predict() bypass the interceptor
                skipped.append(f"{index}:failed({_failure_label(e)})")
                continue
            finally:
                self._release(index)
            self._active = index
            _served_provider.set(member.provider_name)
            self._log_call(
                lane,
                f"{index}:{member.provider_name}",
                len(pairs),
                started,
                skipped,
                sent=len(send),
                queue_wait=queue_wait,
            )
            if len(send) < len(pairs):
                # Raised here, not inside the try: the slot is already released (finally above), the failover `except Exception` never sees it, and _served_provider is set first because rerank() reads it after its own except block.
                raise RerankerMemberCapReached(
                    list(scores) + [None] * (len(pairs) - len(send)),
                    index=index,
                    provider=member.provider_name,
                    lane=lane,
                    sent=len(send),
                )
            return scores
        self._log_call(lane, "none", len(pairs), started, skipped, sent=0, queue_wait=queue_wait)
        if last_exc is not None:
            raise last_exc
        raise RerankerUnavailableError(
            f"No reranker member available in the {lane} lane ({', '.join(skipped) or 'empty order'})"
        )

    def _log_call(
        self,
        lane: str,
        served: str,
        candidates: int,
        started: float,
        skipped: list[str],
        *,
        sent: int,
        queue_wait: float,
    ) -> None:
        # One line per call. Query and document text are never logged.
        level = logging.WARNING if served == "none" or any(":failed(" in s for s in skipped) else logging.INFO
        logger.log(
            level,
            "Reranker routing: lane=%s served=%s candidates=%d sent=%d latency_ms=%.0f queue_wait_ms=%.0f skipped=[%s]",
            lane,
            served,
            candidates,
            sent,
            (time.perf_counter() - started) * 1000,
            queue_wait * 1000,
            ", ".join(skipped),
        )


def validate_routing_env(member_count: int) -> None:
    """Startup check for chains that never reach routed_chain (fork hook in create_cross_encoder_from_env). A bad value fails here; a single member keeps the bare provider and only warns that the order, in-flight and cooldown env cannot apply to it."""
    settings = RoutingSettings.from_env(member_count)
    lane_max_candidates()
    if member_count == 1 and settings.configured:
        logger.warning(
            "Reranker routing: a single reranker member has nothing to route; the lane order, "
            "MAX_INFLIGHT, COOLDOWN and BACKGROUND_QUEUE_* env vars and the per-member "
            "FOREGROUND/BACKGROUND_MAX_CANDIDATES are ignored (the lane-wide FOREGROUND/BACKGROUND_MAX_CANDIDATES still apply)"
        )


def routing_warnings(settings: RoutingSettings, provider_names: list[str]) -> list[str]:
    """Human-readable startup warnings for settings that will never take effect. Each names the env var at fault; nothing here fails startup."""
    warnings: list[str] = []
    background_order = settings.lane_orders.background
    for i, provider in enumerate(provider_names):
        seconds = settings.queue_seconds[i]
        explicit = settings.queue_max_explicit[i]
        if seconds > 0 and settings.max_inflight[i] == 0:
            warnings.append(
                f"{queue_seconds_env_name(i)} is set on member {i}, but {max_inflight_env_name(i)} is unset or 0 "
                "(no in-flight cap), so nothing ever waits"
            )
        if seconds > 0 and explicit == 0:
            warnings.append(
                f"{queue_max_env_name(i)}=0 disables waiting on member {i} although {queue_seconds_env_name(i)} is set"
            )
        if explicit and seconds == 0:
            warnings.append(
                f"{queue_max_env_name(i)} is ignored on member {i} because {queue_seconds_env_name(i)} is unset or 0"
            )
        if seconds > 0 and i not in background_order:
            warnings.append(
                f"{queue_seconds_env_name(i)} is set on member {i}, which is not in {_ORDER_ENV[LANE_BACKGROUND]}; it is never used"
            )
        if provider == "rrf":
            for lane in LANES:
                if settings.member_max_candidates.for_lane(lane)[i] > 0:
                    warnings.append(
                        f"{member_max_candidates_env_name(i, lane)} is ignored on member {i}: the rrf passthrough does not rerank"
                    )
    return warnings


def routed_chain(members: list[CrossEncoderModel], *, environ: Mapping[str, str] | None = None) -> MultiCrossEncoder:
    """Build the failover chain for create_cross_encoder_from_env (fork hook). With no routing env set this is upstream's MultiCrossEncoder, unchanged. Validates the routing env (and the per-lane candidate caps) here so a bad value fails at startup."""
    settings = RoutingSettings.from_env(len(members), environ)
    lane_max_candidates()  # validate the per-lane candidate caps at startup
    if not settings.configured:
        return MultiCrossEncoder(members)
    caps = {i: cap for i, cap in enumerate(settings.max_inflight) if cap}
    queue = {
        i: f"{seconds:g}s/max {settings.queue_max[i]}"
        for i, seconds in enumerate(settings.queue_seconds)
        if seconds > 0
    }
    member_caps = {
        lane: {i: cap for i, cap in enumerate(settings.member_max_candidates.for_lane(lane)) if cap} for lane in LANES
    }
    member_caps = {lane: found for lane, found in member_caps.items() if found}
    logger.info(
        "Reranker routing: foreground order=%s background order=%s max_inflight=%s cooldown=%s queue=%s member_caps=%s",
        list(settings.lane_orders.foreground),
        list(settings.lane_orders.background),
        caps or "unlimited",
        "on" if settings.cooldown_enabled else "off",
        queue or "off",
        member_caps or "none",
    )
    unused = [
        i
        for i in range(len(members))
        if i not in settings.lane_orders.foreground and i not in settings.lane_orders.background
    ]
    if unused:
        logger.warning("Reranker routing: member(s) %s are in no lane order and will never serve", unused)
    for warning in routing_warnings(settings, [member.provider_name for member in members]):
        logger.warning("Reranker routing: %s", warning)
    return RoutedMultiCrossEncoder(members, settings)

"""Fork lane-based reranker routing (fork: rerank-routing): lanes, per-lane order, in-flight caps, 429 cooldown, per-lane candidate cap.

Section A: lanes, env parsing, RoutingSettings, per-lane candidate cap.
Section B: cooldown computation (429 classification, reset headers, error types).
Section C: the routed chain (lane order, in-flight gate, cooldown, 429 interception, per-call log).
"""

import asyncio
import email.utils
import logging
import math
import threading
import time
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.cross_encoder import (
    CrossEncoderModel,
    MultiCrossEncoder,
    RerankTimeoutError,
    RRFPassthroughCrossEncoder,
)
from hindsight_api.engine.remote_retry import RetryPolicy, is_transient_remote_error, status_code_of
from hindsight_api.engine.reranker_routing import (
    DEFAULT_BACKGROUND_QUEUE_MAX,
    ENV_BACKGROUND_MAX_CANDIDATES,
    ENV_BACKGROUND_ORDER,
    ENV_COOLDOWN_DEFAULT_SECONDS,
    ENV_COOLDOWN_ENABLED,
    ENV_COOLDOWN_MAX_SECONDS,
    ENV_COOLDOWN_PER_MINUTE_SECONDS,
    ENV_FOREGROUND_MAX_CANDIDATES,
    ENV_FOREGROUND_ORDER,
    LANE_BACKGROUND,
    LANE_FOREGROUND,
    Cooldown,
    LaneValues,
    RerankerMemberCapReached,
    RerankerRateLimitedError,
    RerankerUnavailableError,
    RoutedMultiCrossEncoder,
    RoutingSettings,
    _QueueTicket,
    background_lane,
    current_lane,
    lane_max_candidates,
    lane_max_candidates_override,
    max_inflight_env_name,
    member_env_name,
    member_max_candidates_env_name,
    next_utc_midnight,
    parse_lane_order,
    queue_max_env_name,
    queue_seconds_env_name,
    rate_limit_cooldown,
    rerank_background_lane,
    routed_chain,
    routing_warnings,
    validate_routing_env,
)
from hindsight_api.engine.search.reranking import CrossEncoderReranker, apply_combined_scoring
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult

# ── lanes ─────────────────────────────────────────────────────────────


def test_lane_defaults_to_foreground():
    assert current_lane() == LANE_FOREGROUND


def test_background_lane_sets_and_resets():
    with background_lane():
        assert current_lane() == LANE_BACKGROUND
    assert current_lane() == LANE_FOREGROUND


def test_background_lane_resets_when_body_raises():
    with pytest.raises(RuntimeError):
        with background_lane():
            assert current_lane() == LANE_BACKGROUND
            raise RuntimeError("boom")
    assert current_lane() == LANE_FOREGROUND


async def _report_lane() -> str:
    return current_lane()


async def test_background_lane_is_inherited_by_child_tasks():
    with background_lane():
        assert await asyncio.create_task(_report_lane()) == LANE_BACKGROUND
        gathered = await asyncio.gather(_report_lane(), _report_lane())
        assert gathered == [LANE_BACKGROUND, LANE_BACKGROUND]
    # A task created outside the block copies the foreground context.
    assert await asyncio.create_task(_report_lane()) == LANE_FOREGROUND


async def test_rerank_background_lane_decorator():
    @rerank_background_lane
    async def run_task() -> str:
        """Docstring kept by functools.wraps."""
        return await asyncio.create_task(_report_lane())

    assert await run_task() == LANE_BACKGROUND
    assert current_lane() == LANE_FOREGROUND
    assert run_task._rerank_background_lane is True
    assert run_task.__name__ == "run_task"
    assert run_task.__doc__ == "Docstring kept by functools.wraps."


async def test_rerank_background_lane_decorator_resets_on_exception():
    @rerank_background_lane
    async def failing_task() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await failing_task()
    assert current_lane() == LANE_FOREGROUND


# ── lane order parsing ────────────────────────────────────────────────


def test_parse_lane_order_valid():
    assert parse_lane_order("2, 0,1,3", 4, "X") == (2, 0, 1, 3)
    assert parse_lane_order("2,3", 4, "X") == (2, 3)
    assert parse_lane_order(None, 4, "X") == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("", "empty"),
        ("  ", "empty"),
        (",", "empty entry"),
        ("0,,1", "empty entry"),
        ("a", "not a member index"),
        ("4", "does not exist"),
        ("-1", "does not exist"),
        ("0,1,0", "listed twice"),
    ],
)
def test_parse_lane_order_rejects(raw, fragment):
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_BACKGROUND_ORDER") as exc:
        parse_lane_order(raw, 4, ENV_BACKGROUND_ORDER)
    assert fragment in str(exc.value)


# ── RoutingSettings ───────────────────────────────────────────────────


def test_settings_unset_is_not_configured():
    settings = RoutingSettings.from_env(4, environ={})
    assert settings.configured is False
    assert settings.lane_orders == LaneValues(foreground=(0, 1, 2, 3), background=(0, 1, 2, 3))
    assert settings.max_inflight == (0, 0, 0, 0)
    assert settings.cooldown_enabled is False
    assert settings.cooldown_default_seconds == 30.0
    assert settings.cooldown_per_minute_seconds == 60.0
    assert settings.cooldown_max_seconds == 86400.0


def test_settings_configured_fields():
    settings = RoutingSettings.from_env(
        4,
        environ={
            ENV_FOREGROUND_ORDER: "0,1,2,3",
            ENV_BACKGROUND_ORDER: "2,0,1,3",
            "HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "1",
            ENV_COOLDOWN_ENABLED: "true",
            ENV_COOLDOWN_DEFAULT_SECONDS: "5",
        },
    )
    assert settings.configured is True
    assert settings.lane_orders.foreground == (0, 1, 2, 3)
    assert settings.lane_orders.background == (2, 0, 1, 3)
    assert settings.max_inflight == (0, 0, 1, 0)
    assert settings.cooldown_enabled is True
    assert settings.cooldown_default_seconds == 5.0


def test_settings_only_max_inflight_is_configured():
    settings = RoutingSettings.from_env(3, environ={"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "2"})
    assert settings.configured is True
    assert settings.lane_orders.background == (0, 1, 2)
    assert settings.max_inflight == (2, 0, 0)


def test_settings_only_cooldown_enabled_is_configured():
    settings = RoutingSettings.from_env(2, environ={ENV_COOLDOWN_ENABLED: "1"})
    assert settings.configured is True
    assert settings.cooldown_enabled is True


def test_settings_only_order_is_configured():
    settings = RoutingSettings.from_env(2, environ={ENV_BACKGROUND_ORDER: "1,0"})
    assert settings.configured is True
    assert settings.lane_orders.background == (1, 0)
    assert settings.lane_orders.foreground == (0, 1)


def test_settings_cooldown_enabled_flags():
    for raw, expected in (("yes", True), ("TRUE", True), ("0", False), ("no", False), ("", False)):
        assert RoutingSettings.from_env(1, environ={ENV_COOLDOWN_ENABLED: raw}).cooldown_enabled is expected


# ── max inflight env names ────────────────────────────────────────────


def test_max_inflight_env_names():
    assert max_inflight_env_name(0) == "HINDSIGHT_API_RERANKER_MAX_INFLIGHT"
    assert max_inflight_env_name(3) == "HINDSIGHT_API_RERANKER_3_MAX_INFLIGHT"


def test_max_inflight_rejects_invalid_values():
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_MAX_INFLIGHT"):
        RoutingSettings.from_env(2, environ={"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "x"})
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT"):
        RoutingSettings.from_env(3, environ={"HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "-2"})


def test_max_inflight_blank_counts_as_unset():
    settings = RoutingSettings.from_env(2, environ={"HINDSIGHT_API_RERANKER_1_MAX_INFLIGHT": ""})
    assert settings.max_inflight == (0, 0)
    assert settings.configured is False


# ── per-lane candidate cap ────────────────────────────────────────────


def test_lane_max_candidates(monkeypatch):
    monkeypatch.delenv(ENV_FOREGROUND_MAX_CANDIDATES, raising=False)
    monkeypatch.delenv(ENV_BACKGROUND_MAX_CANDIDATES, raising=False)
    assert lane_max_candidates() == 0
    with background_lane():
        assert lane_max_candidates() == 0

    monkeypatch.setenv(ENV_BACKGROUND_MAX_CANDIDATES, "40")
    monkeypatch.setenv(ENV_FOREGROUND_MAX_CANDIDATES, "120")
    assert lane_max_candidates() == 120
    with background_lane():
        assert lane_max_candidates() == 40


def test_lane_max_candidates_invalid_value(monkeypatch):
    monkeypatch.setenv(ENV_BACKGROUND_MAX_CANDIDATES, "abc")
    with pytest.raises(ValueError, match=ENV_BACKGROUND_MAX_CANDIDATES):
        with background_lane():
            lane_max_candidates()


def test_lane_max_candidates_override_passthrough_and_override(monkeypatch):
    monkeypatch.delenv(ENV_FOREGROUND_MAX_CANDIDATES, raising=False)
    monkeypatch.delenv(ENV_BACKGROUND_MAX_CANDIDATES, raising=False)

    @lane_max_candidates_override
    def resolve(config, budget):
        return 300

    assert resolve(object(), None) == 300

    monkeypatch.setenv(ENV_BACKGROUND_MAX_CANDIDATES, "40")
    with background_lane():
        assert resolve(object(), None) == 40
    assert resolve(object(), None) == 300


# ── cooldown ──────────────────────────────────────────────────────────

NOW = datetime(2026, 10, 13, 15, 30, tzinfo=UTC).timestamp()
MIDNIGHT = datetime(2026, 10, 14, tzinfo=UTC).timestamp()
SETTINGS = RoutingSettings.from_env(2, environ={})
PER_DAY_BODY = '{"error":{"message":"Rate limit exceeded: free-models-per-day-high-balance"}}'
PER_MIN_BODY = '{"error":{"message":"Rate limit exceeded: free-models-per-min"}}'


def _http_error(status: int, body: str = "", headers: dict[str, str] | None = None) -> UpstreamHTTPError:
    return UpstreamHTTPError(status, body, headers or {}, "https://openrouter.ai/api/v1/rerank")


def test_next_utc_midnight():
    assert next_utc_midnight(NOW) == MIDNIGHT
    # Exactly at midnight means the following midnight.
    assert next_utc_midnight(MIDNIGHT) == MIDNIGHT + 86400


def test_per_day_429_cools_down_until_utc_midnight():
    assert rate_limit_cooldown(_http_error(429, PER_DAY_BODY), NOW, SETTINGS) == Cooldown(MIDNIGHT, "per-day quota")


def test_per_minute_429_cools_down_60s():
    assert rate_limit_cooldown(_http_error(429, PER_MIN_BODY), NOW, SETTINGS) == Cooldown(NOW + 60, "per-minute limit")


def test_other_429_uses_default():
    assert rate_limit_cooldown(_http_error(429, "slow down"), NOW, SETTINGS) == Cooldown(NOW + 30, "rate limited")
    custom = RoutingSettings.from_env(2, environ={ENV_COOLDOWN_DEFAULT_SECONDS: "5"})
    assert rate_limit_cooldown(_http_error(429, "slow down"), NOW, custom) == Cooldown(NOW + 5, "rate limited")


def test_non_429_is_not_a_cooldown():
    assert rate_limit_cooldown(_http_error(500, PER_DAY_BODY), NOW, SETTINGS) is None
    assert rate_limit_cooldown(_http_error(401, PER_DAY_BODY), NOW, SETTINGS) is None
    assert rate_limit_cooldown(RuntimeError("per-day quota"), NOW, SETTINGS) is None


def test_retry_after_seconds_header_wins():
    error = _http_error(429, PER_DAY_BODY, {"Retry-After": "12"})
    assert rate_limit_cooldown(error, NOW, SETTINGS) == Cooldown(NOW + 12, "reset header")


def test_retry_after_http_date():
    header = email.utils.format_datetime(datetime.fromtimestamp(NOW + 90, UTC), usegmt=True)
    cooldown = rate_limit_cooldown(_http_error(429, PER_DAY_BODY, {"Retry-After": header}), NOW, SETTINGS)
    assert cooldown.until == pytest.approx(NOW + 90, abs=1)
    assert cooldown.reason == "reset header"


def test_x_ratelimit_reset_epoch_ms():
    error = _http_error(429, PER_DAY_BODY, {"X-RateLimit-Reset": str(int((NOW + 300) * 1000))})
    cooldown = rate_limit_cooldown(error, NOW, SETTINGS)
    assert cooldown.until == pytest.approx(NOW + 300, abs=1)
    assert cooldown.reason == "reset header"


def test_body_metadata_reset():
    body = (
        '{"error":{"message":"Rate limit exceeded: free-models-per-day","metadata":{"headers":'
        '{"X-RateLimit-Reset":"' + str(int(MIDNIGHT * 1000)) + '"}}}}'
    )
    cooldown = rate_limit_cooldown(_http_error(429, body), NOW, SETTINGS)
    assert cooldown.until == pytest.approx(MIDNIGHT, abs=1)
    assert cooldown.reason == "reset header"


def test_past_reset_is_ignored():
    error = _http_error(429, PER_MIN_BODY, {"X-RateLimit-Reset": str(int(NOW - 100))})
    assert rate_limit_cooldown(error, NOW, SETTINGS) == Cooldown(NOW + 60, "per-minute limit")


def test_header_delay_is_clamped():
    capped = RoutingSettings.from_env(2, environ={ENV_COOLDOWN_MAX_SECONDS: "3600"})
    assert rate_limit_cooldown(_http_error(429, "", {"Retry-After": "999999"}), NOW, capped) == Cooldown(
        NOW + 3600, "reset header"
    )
    assert rate_limit_cooldown(_http_error(429, "", {"Retry-After": "0.2"}), NOW, SETTINGS) == Cooldown(
        NOW + 1.0, "reset header"
    )


def test_rate_limited_error_is_not_retryable():
    error = RerankerRateLimitedError(1, MIDNIGHT)
    assert status_code_of(error) is None
    assert is_transient_remote_error(error) is False
    assert "member 1" in str(error)


# ── routed chain ──────────────────────────────────────────────────────

PAIRS = [("q", "doc a"), ("q", "doc b")]


class _Member(CrossEncoderModel):
    """Overrides predict() directly, so it opts out of the shared retry (like a test double)."""

    def __init__(
        self,
        name: str,
        *,
        error: BaseException | None = None,
        gate: asyncio.Event | None = None,
        score: float = 0.9,
    ) -> None:
        self._name = name
        self.error = error
        self.gate = gate
        self.score = score
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return [self.score] * len(pairs)


class _RetryingMember(CrossEncoderModel):
    """Implements _predict, so it goes through the shared predict() retry and the cooldown interceptor."""

    retry_policy = RetryPolicy(max_retries=3, initial_backoff=0.0, max_backoff=0.0, budget_seconds=5.0)

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "openrouter-like"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        raise UpstreamHTTPError(429, "Rate limit exceeded: free-models-per-min", {}, "http://x/rerank")


def _chain(members, env, now, *, mono=None, poll_initial=0.001, poll_max=0.005) -> RoutedMultiCrossEncoder:
    return RoutedMultiCrossEncoder(
        members,
        RoutingSettings.from_env(len(members), environ=env),
        clock=lambda: now[0],
        monotonic=(lambda: mono[0]) if mono else time.monotonic,
        queue_poll_initial=poll_initial,
        queue_poll_max=poll_max,
    )


# ── routed_chain builder ──────────────────────────────────────────────


def test_routed_chain_without_env_is_the_upstream_chain():
    assert type(routed_chain([_Member("a"), _Member("b")], environ={})) is MultiCrossEncoder


def test_routed_chain_with_env_is_routed():
    members = [_Member("a"), _Member("b")]
    chain = routed_chain(members, environ={"HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "1,0"})
    assert isinstance(chain, RoutedMultiCrossEncoder)
    with pytest.raises(ValueError, match="BACKGROUND_ORDER"):
        routed_chain(members, environ={"HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "0,5"})


# ── lane routing ──────────────────────────────────────────────────────


async def test_natural_order_when_only_inflight_is_set():
    m0, m1 = _Member("m0"), _Member("m1")
    chain = _chain([m0, m1], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "5"}, [NOW])
    await chain.predict(PAIRS)
    with background_lane():
        await chain.predict(PAIRS)
    assert m0.calls == 2
    assert m1.calls == 0


async def test_lanes_pick_different_first_members():
    m0, m1, m2 = _Member("m0"), _Member("m1"), _Member("m2")
    env = {
        "HINDSIGHT_API_RERANKER_FOREGROUND_ORDER": "0,1,2",
        "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "2,0,1",
    }
    chain = _chain([m0, m1, m2], env, [NOW])

    await chain.predict(PAIRS)
    assert m0.calls == 1
    assert m2.calls == 0

    with background_lane():
        await chain.predict(PAIRS)
    assert m2.calls == 1
    assert chain.provider_name == "m2"


async def test_background_lane_reaches_nested_tasks():
    m0, m2 = _Member("m0"), _Member("m2")
    env = {"HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "1,0"}
    chain = _chain([m0, _Member("m1"), m2], env, [NOW])
    with background_lane():
        await asyncio.create_task(chain.predict(PAIRS))
        await asyncio.gather(chain.predict(PAIRS), chain.predict(PAIRS))
    assert m2.calls == 0
    assert chain._members[1].calls == 3
    assert m0.calls == 0


# ── in-flight gate ────────────────────────────────────────────────────


async def _spin_until(member: _Member) -> None:
    for _ in range(100):
        if member.calls:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"{member.provider_name} was never called")


async def test_busy_member_is_skipped_immediately():
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("rrf-like")
    chain = _chain([m0, m1], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "1"}, [NOW])

    first = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    assert chain._inflight[0] == 1

    second = await asyncio.wait_for(chain.predict(PAIRS), 1.0)
    assert second == [0.9, 0.9]
    assert m1.calls == 1
    assert m0.calls == 1

    gate.set()
    await first
    assert chain._inflight == [0, 0]


async def test_inflight_released_on_error():
    m0 = _Member("a", error=RuntimeError("boom"))
    m1 = _Member("b")
    chain = _chain([m0, m1], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "1"}, [NOW])

    assert await chain.predict(PAIRS) == [0.9, 0.9]
    assert chain._inflight == [0, 0]
    await chain.predict(PAIRS)
    assert m0.calls == 2


async def test_inflight_released_on_cancel():
    gate = asyncio.Event()
    m0 = _Member("a", gate=gate)
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "1"}, [NOW])

    task = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    assert chain._inflight[0] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chain._inflight[0] == 0


async def test_all_skipped_without_rrf_raises_unavailable():
    gate0, gate1 = asyncio.Event(), asyncio.Event()
    m0 = _Member("a", gate=gate0)
    m1 = _Member("b", gate=gate1)
    env = {
        "HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "1",
        "HINDSIGHT_API_RERANKER_1_MAX_INFLIGHT": "1",
    }
    chain = _chain([m0, m1], env, [NOW])

    first = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    second = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m1)

    with pytest.raises(RerankerUnavailableError, match="foreground"):
        await chain.predict(PAIRS)

    gate0.set()
    gate1.set()
    await first
    await second
    assert chain._inflight == [0, 0]


async def test_all_failed_reraises_last_error():
    chain = _chain(
        [_Member("a", error=RuntimeError("a")), _Member("b", error=ValueError("b"))],
        {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "3"},
        [NOW],
    )
    with pytest.raises(ValueError):
        await chain.predict(PAIRS)


# ── cooldown ──────────────────────────────────────────────────────────

COOLDOWN_ON = {"HINDSIGHT_API_RERANKER_COOLDOWN_ENABLED": "true"}


async def test_per_day_429_skips_member_until_utc_midnight():
    now = [NOW]
    m0 = _Member("a", error=_http_error(429, "free-models-per-day-high-balance"))
    m1 = _Member("b")
    chain = _chain([m0, m1], COOLDOWN_ON, now)

    await chain.predict(PAIRS)
    assert m0.calls == 1
    assert chain._cooldown_until[0] == MIDNIGHT

    await chain.predict(PAIRS)
    assert m0.calls == 1

    now[0] = MIDNIGHT + 1
    m0.error = None
    await chain.predict(PAIRS)
    assert m0.calls == 2
    assert chain._cooldown_until[0] == 0.0


async def test_per_minute_429_cooldown_expires_after_60s():
    now = [NOW]
    m0 = _Member("a", error=_http_error(429, "free-models-per-min"))
    chain = _chain([m0, _Member("b")], COOLDOWN_ON, now)

    await chain.predict(PAIRS)
    assert m0.calls == 1

    now[0] = NOW + 59
    await chain.predict(PAIRS)
    assert m0.calls == 1

    now[0] = NOW + 61
    m0.error = None
    await chain.predict(PAIRS)
    assert m0.calls == 2


async def test_cooldown_disabled_keeps_calling_the_rate_limited_member():
    m0 = _Member("a", error=_http_error(429, "free-models-per-day-high-balance"))
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "9"}, [NOW])

    await chain.predict(PAIRS)
    await chain.predict(PAIRS)
    assert m0.calls == 2


async def test_cooldown_stops_the_member_retry():
    retrying = _RetryingMember()
    chain = _chain([retrying, _Member("b")], COOLDOWN_ON, [NOW])
    assert await chain.predict(PAIRS) == [0.9, 0.9]
    assert retrying.calls == 1
    assert chain.provider_name == "b"

    upstream = MultiCrossEncoder([_RetryingMember(), _Member("b")])
    assert await upstream.predict(PAIRS) == [0.9, 0.9]
    assert upstream._members[0].calls == 4


async def test_rate_limited_members_fall_through_to_rrf():
    m0 = _Member("a", error=_http_error(429, "free-models-per-day-high-balance"))
    m1 = _Member("b", error=_http_error(429, "free-models-per-day-high-balance"))
    chain = _chain([m0, m1, RRFPassthroughCrossEncoder()], COOLDOWN_ON, [NOW])

    assert len(await chain.predict(PAIRS)) == 2
    assert len(await chain.predict(PAIRS)) == 2
    assert m0.calls == 1
    assert m1.calls == 1


# ── per-call log ──────────────────────────────────────────────────────


def _routing_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "Reranker routing: lane=" in r.getMessage()]


async def test_one_log_line_per_call_with_fields(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    m0 = _Member("a", error=_http_error(429, "free-models-per-day-high-balance"))
    m1 = _Member("b")
    env = {**COOLDOWN_ON, "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "0,1"}
    chain = _chain([m0, m1], env, [NOW])

    with background_lane():
        await chain.predict(PAIRS)
    cooldown = [r for r in caplog.records if "cooling down until 2026-10-14T00:00:00Z" in r.getMessage()]
    assert len(cooldown) == 1
    assert cooldown[0].levelno == logging.WARNING

    caplog.clear()
    with background_lane():
        await chain.predict(PAIRS)
    records = _routing_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert records[0].levelno == logging.INFO
    assert "lane=background" in message
    assert "served=1:b" in message
    assert "candidates=2" in message
    assert "sent=2" in message
    assert "queue_wait_ms=0" in message
    assert "latency_ms=" in message
    assert "skipped=[0:cooldown]" in message
    assert "doc a" not in caplog.text


async def test_failed_member_label_in_log(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    chain = _chain(
        [_Member("a", error=RuntimeError("boom")), _Member("b")], {"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "2"}, [NOW]
    )

    await chain.predict(PAIRS)
    records = _routing_records(caplog)
    assert len(records) == 1
    assert "0:failed(RuntimeError)" in records[0].getMessage()
    assert records[0].levelno == logging.WARNING
    assert "boom" not in caplog.text


# ── upstream hooks (guards against losing a hook in a rebase) ─────────


async def test_execute_task_runs_in_the_background_lane():
    from types import SimpleNamespace

    from hindsight_api.engine.memory_engine import MemoryEngine

    seen = []

    async def _execute_task(task_dict):
        seen.append(current_lane())

    fake = SimpleNamespace(_execute_task=_execute_task)
    await MemoryEngine.execute_task(fake, {"type": "consolidation", "bank_id": "bank-1"})

    assert seen == ["background"]
    assert current_lane() == "foreground"
    assert getattr(MemoryEngine.execute_task, "_rerank_background_lane", False) is True


def test_resolve_reranker_max_candidates_honours_the_lane_cap(monkeypatch):
    from types import SimpleNamespace

    from hindsight_api.engine.memory_engine import Budget, _resolve_reranker_max_candidates

    cfg = SimpleNamespace(
        reranker_max_candidates=300,
        reranker_max_candidates_low=50,
        reranker_max_candidates_mid=0,
        reranker_max_candidates_high=0,
    )
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_BACKGROUND_MAX_CANDIDATES", "40")
    monkeypatch.delenv("HINDSIGHT_API_RERANKER_FOREGROUND_MAX_CANDIDATES", raising=False)

    assert _resolve_reranker_max_candidates(cfg, Budget.LOW) == 50
    assert _resolve_reranker_max_candidates(cfg, Budget.MID) == 300
    with background_lane():
        assert _resolve_reranker_max_candidates(cfg, Budget.LOW) == 40
        assert _resolve_reranker_max_candidates(cfg, Budget.MID) == 40


def test_factory_routes_when_configured(monkeypatch):
    import os
    from unittest.mock import patch

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "1,0",
    }
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            encoder = create_cross_encoder_from_env()
    finally:
        clear_config_cache()

    assert isinstance(encoder, RoutedMultiCrossEncoder)
    assert encoder._settings.lane_orders.background == (1, 0)


def test_factory_is_upstream_when_not_configured(monkeypatch):
    import os
    from unittest.mock import patch

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    for name in (
        ENV_FOREGROUND_ORDER,
        ENV_BACKGROUND_ORDER,
        ENV_COOLDOWN_ENABLED,
        "HINDSIGHT_API_RERANKER_MAX_INFLIGHT",
        "HINDSIGHT_API_RERANKER_1_MAX_INFLIGHT",
        "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS",
        "HINDSIGHT_API_RERANKER_1_BACKGROUND_QUEUE_SECONDS",
        "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_MAX",
        "HINDSIGHT_API_RERANKER_1_BACKGROUND_QUEUE_MAX",
        "HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES",
        "HINDSIGHT_API_RERANKER_1_FOREGROUND_MAX_CANDIDATES",
        "HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES",
        "HINDSIGHT_API_RERANKER_1_BACKGROUND_MAX_CANDIDATES",
    ):
        monkeypatch.delenv(name, raising=False)
    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "rrf",
    }
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            encoder = create_cross_encoder_from_env()
    finally:
        clear_config_cache()

    assert type(encoder) is MultiCrossEncoder


def test_factory_rejects_a_bad_background_order(monkeypatch):
    import os
    from unittest.mock import patch

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "0,7",
    }
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_BACKGROUND_ORDER"):
                create_cross_encoder_from_env()
    finally:
        clear_config_cache()


def test_routing_module_reads_env_only_through_config():
    """reranker_routing.py never touches os.environ; config.py's reranker_routing_env() is the one reader (the config guard test relies on it)."""
    from pathlib import Path

    import hindsight_api.engine.reranker_routing as module

    source = Path(module.__file__).read_text()
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "import os" not in source


def test_config_keeps_the_routing_env_hook(monkeypatch):
    from pathlib import Path

    import hindsight_api.config as config

    text = Path(config.__file__).read_text()
    assert "# fork: rerank-routing\ndef reranker_routing_env()" in text

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_9_MAX_INFLIGHT", "3")
    monkeypatch.setenv("HINDSIGHT_API_UNRELATED", "x")
    env = config.reranker_routing_env()
    assert env["HINDSIGHT_API_RERANKER_9_MAX_INFLIGHT"] == "3"
    assert "HINDSIGHT_API_UNRELATED" not in env


def test_settings_default_environment_comes_from_config(monkeypatch):
    monkeypatch.setenv(ENV_BACKGROUND_ORDER, "1,0")
    settings = RoutingSettings.from_env(2)
    assert settings.lane_orders.background == (1, 0)
    assert settings.configured is True


# ── startup validation of routing env (fork hook: create_cross_encoder_from_env) ──


@pytest.mark.parametrize(
    "env_name", [ENV_COOLDOWN_DEFAULT_SECONDS, ENV_COOLDOWN_PER_MINUTE_SECONDS, ENV_COOLDOWN_MAX_SECONDS]
)
@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "1e20", "604801", "-1", "soon"])
def test_cooldown_durations_reject_non_finite_and_huge_values(env_name, raw):
    with pytest.raises(ValueError, match=env_name):
        RoutingSettings.from_env(2, environ={env_name: raw})


def test_cooldown_durations_accept_the_upper_bound():
    settings = RoutingSettings.from_env(
        2, environ={ENV_COOLDOWN_DEFAULT_SECONDS: "604800", ENV_COOLDOWN_MAX_SECONDS: "0"}
    )
    assert settings.cooldown_default_seconds == 604800.0
    assert settings.cooldown_max_seconds == 0.0


# ── single member: routing env is validated, but the bare provider is kept ──


def _build_single(monkeypatch, extra: dict[str, str]):
    import os
    from unittest.mock import patch

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

    env = {"HINDSIGHT_API_RERANKER_PROVIDER": "rrf", **extra}
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            return create_cross_encoder_from_env()
    finally:
        clear_config_cache()


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        ({ENV_BACKGROUND_MAX_CANDIDATES: "abc"}, ENV_BACKGROUND_MAX_CANDIDATES),
        ({ENV_FOREGROUND_MAX_CANDIDATES: "-1"}, ENV_FOREGROUND_MAX_CANDIDATES),
        ({ENV_BACKGROUND_ORDER: "0,99"}, ENV_BACKGROUND_ORDER),
        ({"HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "x"}, "HINDSIGHT_API_RERANKER_MAX_INFLIGHT"),
        ({ENV_COOLDOWN_ENABLED: "true", ENV_COOLDOWN_DEFAULT_SECONDS: "inf"}, ENV_COOLDOWN_DEFAULT_SECONDS),
        ({"HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS": "nan"}, "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS"),
        (
            {"HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES": "-1"},
            "HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES",
        ),
    ],
)
def test_single_member_fails_at_startup_on_bad_routing_env(monkeypatch, extra, fragment):
    for name in (
        ENV_BACKGROUND_MAX_CANDIDATES,
        ENV_FOREGROUND_MAX_CANDIDATES,
        ENV_BACKGROUND_ORDER,
        "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS",
        "HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match=fragment):
        _build_single(monkeypatch, extra)


def test_single_member_with_valid_routing_env_keeps_the_bare_provider(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hindsight_api.engine.reranker_routing")
    encoder = _build_single(
        monkeypatch,
        {
            ENV_BACKGROUND_ORDER: "0",
            "HINDSIGHT_API_RERANKER_MAX_INFLIGHT": "2",
            ENV_COOLDOWN_ENABLED: "true",
            ENV_BACKGROUND_MAX_CANDIDATES: "40",
        },
    )
    assert isinstance(encoder, RRFPassthroughCrossEncoder)
    assert not isinstance(encoder, RoutedMultiCrossEncoder)
    assert any("single reranker member" in r.getMessage() for r in caplog.records)


def test_single_member_without_routing_env_does_not_warn(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hindsight_api.engine.reranker_routing")
    for name in (ENV_FOREGROUND_ORDER, ENV_BACKGROUND_ORDER, ENV_COOLDOWN_ENABLED, ENV_BACKGROUND_MAX_CANDIDATES):
        monkeypatch.delenv(name, raising=False)
    encoder = _build_single(monkeypatch, {})
    assert isinstance(encoder, RRFPassthroughCrossEncoder)
    assert not any("single reranker member" in r.getMessage() for r in caplog.records)


def test_validate_routing_env_accepts_unset_env_for_any_count(monkeypatch):
    for name in (ENV_FOREGROUND_ORDER, ENV_BACKGROUND_ORDER, ENV_COOLDOWN_ENABLED, ENV_BACKGROUND_MAX_CANDIDATES):
        monkeypatch.delenv(name, raising=False)
    validate_routing_env(1)
    validate_routing_env(4)


# ── a TEI-style 429: RuntimeError wrapping UpstreamHTTPError (TEI re-raises it this way) ──


def _wrapped_tei_429(body: str = PER_MIN_BODY) -> RuntimeError:
    try:
        try:
            raise _http_error(429, body)
        except UpstreamHTTPError as e:
            raise RuntimeError(f"TEI rerank request failed: {e}")
    except RuntimeError as wrapped:
        return wrapped


def test_wrapped_429_is_classified_through_the_context_chain():
    wrapped = _wrapped_tei_429()
    assert status_code_of(wrapped) is None
    assert rate_limit_cooldown(wrapped, NOW, SETTINGS) == Cooldown(NOW + 60, "per-minute limit")
    assert rate_limit_cooldown(RuntimeError("plain failure"), NOW, SETTINGS) is None


def test_wrapped_429_is_not_followed_more_than_two_levels():
    deep = ValueError("top")
    level = deep
    for name in ("one", "two"):
        level.__cause__ = RuntimeError(name)
        level = level.__cause__
    level.__cause__ = _http_error(429, PER_MIN_BODY)
    assert rate_limit_cooldown(deep, NOW, SETTINGS) is None


async def test_wrapped_429_falls_back_then_skips_until_cooldown_ends(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    tei = _Member("tei-like", error=_wrapped_tei_429())
    fallback = _Member("fallback", score=0.5)
    now = [NOW]
    chain = _chain([tei, fallback], {ENV_COOLDOWN_ENABLED: "true"}, now)

    assert await chain.predict(PAIRS) == [0.5, 0.5]
    assert tei.calls == 1
    assert any("0:failed(HTTP 429)" in r.getMessage() for r in caplog.records)

    now[0] = NOW + 10
    assert await chain.predict(PAIRS) == [0.5, 0.5]
    assert tei.calls == 1

    now[0] = NOW + 61
    assert await chain.predict(PAIRS) == [0.5, 0.5]
    assert tei.calls == 2


# ── v2 settings ───────────────────────────────────────────────────────

_V2_ENV_NAMES = (
    "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS",
    "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_MAX",
    "HINDSIGHT_API_RERANKER_1_BACKGROUND_QUEUE_SECONDS",
    "HINDSIGHT_API_RERANKER_1_BACKGROUND_QUEUE_MAX",
    "HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES",
    "HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES",
    "HINDSIGHT_API_RERANKER_1_FOREGROUND_MAX_CANDIDATES",
    "HINDSIGHT_API_RERANKER_1_BACKGROUND_MAX_CANDIDATES",
)


def test_member_env_names_for_queue_and_caps():
    assert member_env_name(0, "X") == "HINDSIGHT_API_RERANKER_X"
    assert member_env_name(2, "X") == "HINDSIGHT_API_RERANKER_2_X"
    assert queue_seconds_env_name(0) == "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS"
    assert queue_seconds_env_name(2) == "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS"
    assert queue_max_env_name(0) == "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_MAX"
    assert queue_max_env_name(2) == "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX"
    assert member_max_candidates_env_name(0, LANE_FOREGROUND) == "HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES"
    assert member_max_candidates_env_name(2, LANE_BACKGROUND) == "HINDSIGHT_API_RERANKER_2_BACKGROUND_MAX_CANDIDATES"
    assert DEFAULT_BACKGROUND_QUEUE_MAX == 2


def test_v2_unset_defaults():
    settings = RoutingSettings.from_env(4, environ={})
    assert settings.queue_seconds == (0.0, 0.0, 0.0, 0.0)
    assert settings.queue_max == (0, 0, 0, 0)
    assert settings.queue_max_explicit == (None, None, None, None)
    assert settings.member_max_candidates == LaneValues(foreground=(0, 0, 0, 0), background=(0, 0, 0, 0))
    assert settings.configured is False


def test_queue_seconds_alone_is_configured_with_default_queue_max():
    settings = RoutingSettings.from_env(4, environ={"HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "1800"})
    assert settings.configured is True
    assert settings.queue_seconds == (0.0, 0.0, 1800.0, 0.0)
    assert settings.queue_max == (0, 0, DEFAULT_BACKGROUND_QUEUE_MAX, 0)
    assert settings.queue_max_explicit == (None, None, None, None)


def test_explicit_queue_max_overrides_the_default():
    settings = RoutingSettings.from_env(
        4,
        environ={
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "1800",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "4",
        },
    )
    assert settings.queue_max == (0, 0, 4, 0)
    assert settings.queue_max_explicit[2] == 4


def test_explicit_zero_queue_max_disables_waiting_and_is_still_validated():
    settings = RoutingSettings.from_env(
        4,
        environ={
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "1800",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "0",
        },
    )
    assert settings.queue_max[2] == 0
    assert settings.queue_max_explicit[2] == 0
    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX"):
        RoutingSettings.from_env(4, environ={"HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "-1"})


def test_member_cap_alone_is_configured():
    settings = RoutingSettings.from_env(4, environ={"HINDSIGHT_API_RERANKER_2_FOREGROUND_MAX_CANDIDATES": "100"})
    assert settings.configured is True
    assert settings.member_max_candidates.foreground == (0, 0, 100, 0)
    assert settings.member_max_candidates.background == (0, 0, 0, 0)


def test_member_zero_cap_is_not_the_lane_wide_cap():
    settings = RoutingSettings.from_env(
        2,
        environ={ENV_FOREGROUND_MAX_CANDIDATES: "50", ENV_BACKGROUND_MAX_CANDIDATES: "300"},
    )
    assert settings.member_max_candidates == LaneValues(foreground=(0, 0), background=(0, 0))
    assert settings.configured is False


def test_member_zero_cap_uses_the_numbered_name():
    settings = RoutingSettings.from_env(2, environ={"HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES": "40"})
    assert settings.member_max_candidates.background == (40, 0)
    assert settings.configured is True


@pytest.mark.parametrize(
    "name",
    [
        "HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS",
        "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS",
    ],
)
@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "-1", "soon"])
def test_queue_seconds_rejects_bad_values(name, raw):
    with pytest.raises(ValueError, match=name):
        RoutingSettings.from_env(4, environ={name: raw})


@pytest.mark.parametrize("raw", ["604801", "1e20"])
def test_queue_seconds_has_no_cooldown_ceiling(raw):
    # Queue waits only use monotonic arithmetic, so any finite value >= 0 is fine; the 7-day cap is for cooldowns.
    settings = RoutingSettings.from_env(4, environ={queue_seconds_env_name(2): raw})
    assert settings.queue_seconds[2] == float(raw)


def test_lane_values_for_lane_rejects_an_unknown_lane():
    values = LaneValues(foreground=1, background=2)
    assert values.for_lane(LANE_FOREGROUND) == 1
    assert values.for_lane(LANE_BACKGROUND) == 2
    with pytest.raises(ValueError, match="unknown rerank lane"):
        values.for_lane("interactive")


@pytest.mark.parametrize("raw", ["-1", "x", "1.5"])
def test_queue_max_rejects_bad_values(raw):
    name = "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX"
    with pytest.raises(ValueError, match=name):
        RoutingSettings.from_env(4, environ={name: raw})


@pytest.mark.parametrize("raw", ["-1", "abc", "2.5"])
@pytest.mark.parametrize("lane", [LANE_FOREGROUND, LANE_BACKGROUND])
@pytest.mark.parametrize("index", [0, 2])
def test_member_caps_reject_bad_values(index, lane, raw):
    name = member_max_candidates_env_name(index, lane)
    with pytest.raises(ValueError, match=name):
        RoutingSettings.from_env(4, environ={name: raw})


def test_blank_v2_values_count_as_unset():
    settings = RoutingSettings.from_env(
        4,
        environ={
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "  ",
            "HINDSIGHT_API_RERANKER_2_FOREGROUND_MAX_CANDIDATES": "",
            "HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES": " ",
        },
    )
    assert settings.configured is False
    assert settings.queue_max_explicit == (None, None, None, None)


# ── routing_warnings ──────────────────────────────────────────────────

_PROVIDERS_4 = ["openrouter", "openrouter", "cohere", "rrf"]


def _warnings(environ: dict[str, str], providers: list[str] = _PROVIDERS_4) -> list[str]:
    settings = RoutingSettings.from_env(len(providers), environ=environ)
    return routing_warnings(settings, providers)


def test_warning_queue_seconds_without_inflight_cap():
    warnings = _warnings({"HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "10"})
    assert len(warnings) == 1
    assert "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS" in warnings[0]
    assert "MAX_INFLIGHT" in warnings[0]


def test_warning_explicit_zero_queue_max_with_queue_seconds():
    warnings = _warnings(
        {
            "HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "1",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "10",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "0",
        }
    )
    assert len(warnings) == 1
    assert "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX" in warnings[0]


def test_warning_queue_max_without_queue_seconds():
    warnings = _warnings(
        {
            "HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "1",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "4",
        }
    )
    assert len(warnings) == 1
    assert "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX" in warnings[0]
    assert "ignored" in warnings[0]


def test_warning_queue_seconds_on_member_outside_background_order():
    warnings = _warnings(
        {
            "HINDSIGHT_API_RERANKER_BACKGROUND_ORDER": "0,1,3",
            "HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "1",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "10",
        }
    )
    assert len(warnings) == 1
    assert "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS" in warnings[0]
    assert "BACKGROUND_ORDER" in warnings[0]


def test_warning_member_cap_on_rrf_passthrough():
    warnings = _warnings({"HINDSIGHT_API_RERANKER_1_FOREGROUND_MAX_CANDIDATES": "100"})
    assert len(warnings) == 0
    warnings = _warnings(
        {
            "HINDSIGHT_API_RERANKER_3_FOREGROUND_MAX_CANDIDATES": "100",
            "HINDSIGHT_API_RERANKER_3_BACKGROUND_MAX_CANDIDATES": "50",
        }
    )
    assert len(warnings) == 2
    assert "HINDSIGHT_API_RERANKER_3_FOREGROUND_MAX_CANDIDATES" in warnings[0]
    assert "HINDSIGHT_API_RERANKER_3_BACKGROUND_MAX_CANDIDATES" in warnings[1]


def test_no_warnings_for_the_ttnas_example():
    warnings = _warnings(
        {
            ENV_FOREGROUND_ORDER: "0,1,2,3",
            ENV_BACKGROUND_ORDER: "2,0,1,3",
            "HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT": "1",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS": "1800",
            "HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX": "4",
            "HINDSIGHT_API_RERANKER_2_FOREGROUND_MAX_CANDIDATES": "100",
        }
    )
    assert warnings == []


# ── v2 settings: single member keeps the bare provider ────────────────


def test_single_member_queue_seconds_only_keeps_bare_provider_and_warns(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hindsight_api.engine.reranker_routing")
    for name in _V2_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    encoder = _build_single(monkeypatch, {"HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS": "10"})
    assert isinstance(encoder, RRFPassthroughCrossEncoder)
    assert not isinstance(encoder, RoutedMultiCrossEncoder)
    assert any("single reranker member" in r.getMessage() for r in caplog.records)


def test_validate_routing_env_accepts_unset_v2_env_for_any_count(monkeypatch):
    for name in _V2_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    validate_routing_env(1)
    validate_routing_env(4)


# ── v2 background queue ───────────────────────────────────────────────

QUEUE_ENV = {
    max_inflight_env_name(0): "1",
    queue_seconds_env_name(0): "30",
    ENV_BACKGROUND_ORDER: "0,1",
}


def _background_task(coro) -> asyncio.Task:
    """Create a task in the background lane (a task copies the context at creation)."""
    with background_lane():
        return asyncio.create_task(coro)


async def _spin_until_true(predicate) -> None:
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached within about 2 s")


class _RecordingMember(_Member):
    """Records the document text of the first pair of every call, in call order."""

    def __init__(self, name: str, *, gate: asyncio.Event | None = None) -> None:
        super().__init__(name, gate=gate)
        self.served: list[str] = []

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.served.append(pairs[0][1])
        return await super().predict(pairs)


class _ThreadSafeMember(CrossEncoderModel):
    """Counts concurrent calls under a threading.Lock, so it can be shared by loops on several threads."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "threadsafe"

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._guard:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return [0.9] * len(pairs)
        finally:
            with self._guard:
                self.active -= 1


async def test_background_waits_and_acquires_when_the_slot_frees(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], QUEUE_ENV, [NOW])

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    waiter = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)
    assert m1.calls == 0

    gate.set()
    assert await asyncio.wait_for(waiter, 2.0) == [0.9, 0.9]
    await holder
    assert m0.calls == 2
    assert m1.calls == 0
    assert chain._inflight == [0, 0]
    assert len(chain._queues[0]) == 0

    background = [r.getMessage() for r in _routing_records(caplog) if "lane=background" in r.getMessage()]
    assert len(background) == 1
    assert "served=0:" in background[0]
    assert "queue_wait_ms=" in background[0]


async def test_background_queue_timeout_moves_to_the_next_member(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    mono = [0.0]
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], {**QUEUE_ENV, queue_seconds_env_name(0): "10"}, [NOW], mono=mono)

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    waiter = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)

    mono[0] = 11.0
    assert await asyncio.wait_for(waiter, 2.0) == [0.9, 0.9]
    assert m1.calls == 1
    assert len(chain._queues[0]) == 0
    assert any("0:queue_timeout" in r.getMessage() for r in _routing_records(caplog))

    gate.set()
    await holder
    assert chain._inflight == [0, 0]


async def test_poll_at_the_deadline_does_not_take_a_freed_slot(caplog):
    # The holder finishes and frees the slot in the same step that moves the clock past the waiter's deadline.
    # The waiter must time out, not acquire the slot that just freed.
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    mono = [0.0]
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], {**QUEUE_ENV, queue_seconds_env_name(0): "10"}, [NOW], mono=mono)

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    waiter = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)

    mono[0] = 11.0
    gate.set()
    assert await asyncio.wait_for(waiter, 2.0) == [0.9, 0.9]
    await holder

    assert m0.calls == 1  # only the holder ever reached the Mac
    assert m1.calls == 1
    assert chain._inflight == [0, 0]
    assert len(chain._queues[0]) == 0
    assert any("0:queue_timeout" in r.getMessage() for r in _routing_records(caplog))


async def test_background_queue_full_skips_immediately(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], {**QUEUE_ENV, queue_max_env_name(0): "1"}, [NOW])

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    first = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)

    with background_lane():
        second = await asyncio.wait_for(chain.predict(PAIRS), 1.0)
    assert second == [0.9, 0.9]
    assert m1.calls == 1
    assert any("skipped=[0:queue_full]" in r.getMessage() for r in _routing_records(caplog))
    assert len(chain._queues[0]) == 1
    assert not first.done()

    gate.set()
    assert await asyncio.wait_for(first, 2.0) == [0.9, 0.9]
    await holder
    assert m0.calls == 2
    assert len(chain._queues[0]) == 0


async def test_foreground_never_waits_in_the_queue(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], QUEUE_ENV, [NOW])

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    assert await asyncio.wait_for(chain.predict(PAIRS), 1.0) == [0.9, 0.9]
    assert m1.calls == 1
    assert len(chain._queues[0]) == 0
    assert any("skipped=[0:inflight]" in r.getMessage() for r in _routing_records(caplog))

    gate.set()
    await holder


async def test_cooldown_entered_during_a_wait_moves_the_waiter_on(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    now = [NOW]
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], {**QUEUE_ENV, **COOLDOWN_ON}, now)

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    waiter = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)

    chain._enter_cooldown(0, _http_error(429, PER_MIN_BODY))
    assert await asyncio.wait_for(waiter, 2.0) == [0.9, 0.9]
    assert m1.calls == 1
    assert len(chain._queues[0]) == 0
    assert any("skipped=[0:cooldown]" in r.getMessage() for r in _routing_records(caplog))

    gate.set()
    await holder
    assert m0.calls == 1


async def test_cancelling_a_waiter_removes_its_ticket_without_leaking():
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], QUEUE_ENV, [NOW])

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    waiter = _background_task(chain.predict(PAIRS))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert len(chain._queues[0]) == 0
    assert chain._inflight[0] == 1  # still the holder's slot, not the waiter's

    gate.set()
    await holder
    assert chain._inflight == [0, 0]

    with background_lane():
        assert await asyncio.wait_for(chain.predict(PAIRS), 1.0) == [0.9, 0.9]
    assert m0.calls == 2
    assert m1.calls == 0


async def test_background_waiters_are_served_in_arrival_order():
    gate = asyncio.Event()
    m0 = _RecordingMember("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], QUEUE_ENV, [NOW])

    holder = asyncio.create_task(chain.predict([("q", "hold")]))
    await _spin_until_true(lambda: m0.calls == 1)
    first = _background_task(chain.predict([("q", "A")]))
    await _spin_until_true(lambda: len(chain._queues[0]) == 1)
    second = _background_task(chain.predict([("q", "B")]))
    await _spin_until_true(lambda: len(chain._queues[0]) == 2)

    gate.set()
    await asyncio.wait_for(asyncio.gather(holder, first, second), 2.0)
    assert m0.served == ["hold", "A", "B"]
    assert m1.calls == 0
    assert chain._inflight == [0, 0]


def test_background_arrival_does_not_barge_past_queued_waiters():
    mono = [1000.0]
    chain = _chain([_Member("a"), _Member("b")], QUEUE_ENV, [NOW], mono=mono)
    chain._queues[0].append(_QueueTicket(1000.0))

    # A slot is free, but a background arrival must queue behind the waiter; foreground may take it.
    assert chain._try_acquire(0, LANE_BACKGROUND) == "inflight"
    assert chain._inflight == [0, 0]
    assert chain._try_acquire(0, LANE_FOREGROUND) is None
    chain._release(0)
    assert chain._inflight == [0, 0]


async def test_stale_ticket_is_dropped_with_one_warning(caplog):
    caplog.set_level(logging.WARNING, logger="hindsight_api.engine.reranker_routing")
    mono = [1000.0]
    m0 = _Member("mac")
    m1 = _Member("fallback")
    chain = _chain([m0, m1], QUEUE_ENV, [NOW], mono=mono)
    chain._queues[0].append(_QueueTicket(1000.0 - 120))

    with background_lane():
        assert await chain.predict(PAIRS) == [0.9, 0.9]
    assert m0.calls == 1
    assert m1.calls == 0
    assert len(chain._queues[0]) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "dropped 1 background queue ticket" in warnings[0].getMessage()


def test_queue_ticket_compares_by_identity():
    first, second = _QueueTicket(1.0), _QueueTicket(1.0)
    queue = [first]
    assert second not in queue
    assert first in queue


async def test_background_queue_is_safe_across_threads_and_loops():
    member = _ThreadSafeMember()
    fallback = _Member("fallback")
    env = {
        max_inflight_env_name(0): "1",
        queue_seconds_env_name(0): "30",
        queue_max_env_name(0): "50",
        ENV_BACKGROUND_ORDER: "0,1",
    }
    chain = _chain([member, fallback], env, [NOW])
    chain._ready = [True, True]  # skip the upstream per-member asyncio.Lock, which must not be shared across loops
    errors: list[BaseException] = []

    async def main() -> None:
        with background_lane():
            results = await asyncio.gather(*(chain.predict(PAIRS) for _ in range(5)))
        assert results == [[0.9, 0.9]] * 5

    def run() -> None:
        try:
            asyncio.run(main())
        except BaseException as exc:  # collected so a failure in a thread fails the test
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert member.max_active == 1
    assert member.calls == 10
    assert fallback.calls == 0
    assert chain._inflight == [0, 0]
    assert all(len(queue) == 0 for queue in chain._queues)


async def test_background_busy_member_is_skipped_immediately_without_queue_env(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    gate = asyncio.Event()
    m0 = _Member("mac", gate=gate)
    m1 = _Member("fallback")
    chain = _chain([m0, m1], {max_inflight_env_name(0): "1", ENV_BACKGROUND_ORDER: "0,1"}, [NOW])

    holder = asyncio.create_task(chain.predict(PAIRS))
    await _spin_until(m0)
    with background_lane():
        assert await asyncio.wait_for(chain.predict(PAIRS), 1.0) == [0.9, 0.9]
    assert m1.calls == 1
    assert len(chain._queues[0]) == 0
    assert any("skipped=[0:inflight]" in r.getMessage() for r in _routing_records(caplog))

    gate.set()
    await holder
    assert chain._inflight == [0, 0]


# ── v2 per-member per-lane candidate cap ──────────────────────────────

FIVE = [f"d{i}" for i in range(5)]


def _docs(n: int) -> list[tuple[str, str]]:
    return [("q", f"d{i}") for i in range(n)]


def _make_candidates(n: int, *, dated: bool = False) -> list[MergedCandidate]:
    """n merged candidates in RRF order. dated=True dates the first three 2000-01-01 and the rest today."""
    candidates = []
    for i in range(n):
        occurred = None
        if dated:
            occurred = datetime(2000, 1, 1, tzinfo=UTC) if i < 3 else datetime.now(UTC)
        candidates.append(
            MergedCandidate(
                retrieval=RetrievalResult(
                    id=str(uuid.uuid4()),
                    text=f"Document {i}",
                    fact_type="world",
                    occurred_start=occurred,
                    occurred_end=None,
                ),
                rrf_score=1.0 / (i + 1),
            )
        )
    return candidates


class _ListMember(CrossEncoderModel):
    """Returns configured scores (a list, or a function of len(pairs)) and records the documents of every call it receives."""

    def __init__(
        self,
        name: str,
        *,
        scores: list[float] | None | object = None,
        error: BaseException | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self._name = name
        self.scores = scores
        self.error = error
        self.gate = gate
        self.received: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.received.append([doc for _, doc in pairs])
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        if self.scores is None:
            return [0.9] * len(pairs)
        if callable(self.scores):
            return self.scores(len(pairs))
        return list(self.scores)


def _reranker(chain) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(cross_encoder=chain)
    reranker._initialized = True
    return reranker


def _ids(candidates: list[MergedCandidate]) -> list[str]:
    return [candidate.retrieval.id for candidate in candidates]


CAP0_FG = {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": "2"}


async def test_cap_sends_only_the_rrf_head_and_raises_a_full_length_partial_result(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    m0 = _ListMember("capped", scores=[0.3, 0.8])
    m1 = _ListMember("fallback")
    chain = _chain([m0, m1], CAP0_FG, [NOW])

    with pytest.raises(RerankerMemberCapReached) as exc:
        await chain.predict(_docs(5))
    assert exc.value.scores == [0.3, 0.8, None, None, None]
    assert m0.received == [["d0", "d1"]]
    assert m1.received == []
    assert isinstance(exc.value, RerankTimeoutError)
    assert "d0" not in str(exc.value)
    assert (exc.value.index, exc.value.sent, exc.value.lane) == (0, 2, LANE_FOREGROUND)
    line = _routing_records(caplog)[-1].getMessage()
    assert "candidates=5" in line
    assert "sent=2" in line


async def test_cap_is_per_lane():
    # A foreground-only cap leaves the background lane uncapped, and the reverse.
    foreground_only = _ListMember("capped", scores=lambda n: [0.5] * n)
    chain = _chain([foreground_only, _ListMember("fallback")], CAP0_FG, [NOW])
    with background_lane():
        assert await chain.predict(_docs(5)) == [0.5] * 5
    assert foreground_only.received[-1] == FIVE
    with pytest.raises(RerankerMemberCapReached):
        await chain.predict(_docs(5))
    assert foreground_only.received[-1] == FIVE[:2]

    background_only = _ListMember("capped", scores=lambda n: [0.5] * n)
    env = {"HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES": "2"}
    chain = _chain([background_only, _ListMember("fallback")], env, [NOW])
    assert await chain.predict(_docs(5)) == [0.5] * 5
    with background_lane(), pytest.raises(RerankerMemberCapReached):
        await chain.predict(_docs(5))
    assert background_only.received[-1] == FIVE[:2]


@pytest.mark.parametrize("cap", ["5", "9"])
async def test_cap_at_or_above_the_candidate_count_is_not_a_cap(cap):
    m0 = _ListMember("capped", scores=lambda n: [0.4] * n)
    chain = _chain([m0, _ListMember("fallback")], {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": cap}, [NOW])
    assert await chain.predict(_docs(5)) == [0.4] * 5
    assert m0.received == [FIVE]


async def test_next_member_gets_the_full_list_and_its_own_cap_applies():
    m0 = _ListMember("failing", error=RuntimeError("boom"))
    m1 = _ListMember("uncapped", scores=lambda n: [0.1] * n)
    chain = _chain([m0, m1], CAP0_FG, [NOW])
    # m0 was sent the head and failed; m1 is uncapped and must receive all five.
    assert await chain.predict(_docs(5)) == [0.1] * 5
    assert m0.received == [FIVE[:2]]
    assert m1.received == [FIVE]

    m1_capped = _ListMember("capped-next", scores=[0.4, 0.6, 0.2])
    chain = _chain(
        [_ListMember("failing", error=RuntimeError("boom")), m1_capped],
        {"HINDSIGHT_API_RERANKER_1_FOREGROUND_MAX_CANDIDATES": "3"},
        [NOW],
    )
    with pytest.raises(RerankerMemberCapReached) as exc:
        await chain.predict(_docs(5))
    assert m1_capped.received == [FIVE[:3]]
    assert exc.value.scores == [0.4, 0.6, 0.2, None, None]


async def test_member_skipped_for_inflight_passes_the_full_list_on():
    gate = asyncio.Event()
    m0 = _ListMember("capped", scores=[0.3, 0.8], gate=gate)
    m1 = _ListMember("uncapped", scores=lambda n: [0.1] * n)
    env = {max_inflight_env_name(0): "1", **CAP0_FG}
    chain = _chain([m0, m1], env, [NOW])

    holder = asyncio.create_task(chain.predict(_docs(5)))
    await _spin_until_true(lambda: m0.received)
    assert await asyncio.wait_for(chain.predict(_docs(5)), 1.0) == [0.1] * 5
    assert m1.received == [FIVE]

    gate.set()
    with pytest.raises(RerankerMemberCapReached):
        await holder
    assert chain._inflight == [0, 0]


async def test_rrf_member_ignores_its_cap():
    m0 = _ListMember("failing", error=RuntimeError("boom"))
    chain = _chain(
        [m0, RRFPassthroughCrossEncoder()],
        {"HINDSIGHT_API_RERANKER_1_FOREGROUND_MAX_CANDIDATES": "2"},
        [NOW],
    )
    assert await chain.predict(_docs(5)) == [0.5] * 5
    assert chain.provider_name == "rrf"


async def test_member_timeout_error_is_still_a_failover_not_a_cap():
    m0 = _ListMember("local", error=RerankTimeoutError([0.1, None], 0.1, "local"))
    m1 = _ListMember("fallback", scores=lambda n: [0.2] * n)
    chain = _chain([m0, m1], {}, [NOW])
    assert await chain.predict(_docs(5)) == [0.2] * 5
    assert m1.received == [FIVE]


async def test_rerank_puts_the_rrf_tail_after_the_scored_head():
    candidates = _make_candidates(6)
    ids = _ids(candidates)
    m0 = _ListMember("capped", scores=[0.2, 0.9, 0.5])
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": "3"}, [NOW])

    result = await _reranker(chain).rerank("q", candidates)

    assert _ids([r.candidate for r in result.results]) == [ids[1], ids[2], ids[0], ids[3], ids[4], ids[5]]
    head_min = min(r.cross_encoder_score_normalized for r in result.results[:3])
    for r in result.results[3:]:
        assert r.cross_encoder_score == r.cross_encoder_score_normalized == r.weight == 0.0
        assert r.cross_encoder_score_normalized < head_min
    for r in result.results:
        assert math.isfinite(r.cross_encoder_score)
        assert math.isfinite(r.cross_encoder_score_normalized)
        assert math.isfinite(r.weight)
    assert result.provider_name == "capped"


async def test_logit_head_is_sigmoided_and_the_tail_stays_at_zero():
    candidates = _make_candidates(6)
    ids = _ids(candidates)
    m0 = _ListMember("capped", scores=[-2.0, 3.0, -0.5])
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": "3"}, [NOW])

    result = await _reranker(chain).rerank("q", candidates)

    assert _ids([r.candidate for r in result.results]) == [ids[1], ids[2], ids[0], ids[3], ids[4], ids[5]]
    assert result.results[0].cross_encoder_score == 3.0
    assert result.results[0].cross_encoder_score_normalized == pytest.approx(1 / (1 + math.exp(-3.0)))
    assert result.results[1].cross_encoder_score_normalized == pytest.approx(1 / (1 + math.exp(0.5)))
    assert result.results[2].cross_encoder_score_normalized == pytest.approx(1 / (1 + math.exp(2.0)))
    for r in result.results[3:]:
        assert r.cross_encoder_score == r.cross_encoder_score_normalized == r.weight == 0.0
    assert all(0.0 < r.cross_encoder_score_normalized < 1.0 for r in result.results[:3])


async def test_combined_scoring_keeps_the_tail_last():
    candidates = _make_candidates(6, dated=True)
    ids = _ids(candidates)
    m0 = _ListMember("capped", scores=[0.2, 0.9, 0.5])
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": "3"}, [NOW])

    result = await _reranker(chain).rerank("q", candidates)
    apply_combined_scoring(result.results, now=datetime.now(UTC))
    result.results.sort(key=lambda r: r.weight, reverse=True)

    assert _ids([r.candidate for r in result.results[3:]]) == ids[3:]
    assert all(r.weight == 0.0 for r in result.results[3:])
    assert all(r.weight > 0.0 for r in result.results[:3])


async def test_reranker_floor_drops_the_unscored_tail():
    candidates = _make_candidates(6)
    ids = _ids(candidates)
    m0 = _ListMember("capped", scores=[0.2, 0.9, 0.5])
    chain = _chain([m0, _Member("b")], {"HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES": "3"}, [NOW])

    result = await _reranker(chain).rerank("q", candidates)

    # A floor above zero keeps only the scored head; a floor of exactly zero keeps the tail as well.
    kept = [r for r in result.results if r.cross_encoder_score_normalized >= 0.01]
    assert _ids([r.candidate for r in kept]) == [ids[1], ids[2], ids[0]]
    assert len([r for r in result.results if r.cross_encoder_score_normalized >= 0.0]) == 6


class _FixedScoresMember(_Member):
    """Returns exactly the scores it was given, whatever it is sent."""

    def __init__(self, name: str, scores: list[float]) -> None:
        super().__init__(name)
        self.scores = scores

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return list(self.scores)


async def test_infinite_score_fails_the_member_over(caplog):
    caplog.set_level(logging.INFO, logger="hindsight_api.engine.reranker_routing")
    m0 = _FixedScoresMember("bad", [math.inf, 0.5])
    m1 = _Member("good")
    chain = _chain([m0, m1], {ENV_FOREGROUND_ORDER: "0,1"}, [NOW])

    assert await chain.predict(PAIRS) == [0.9, 0.9]
    assert m1.calls == 1
    assert chain._inflight == [0, 0]
    assert any("0:failed(RuntimeError)" in r.getMessage() for r in _routing_records(caplog))

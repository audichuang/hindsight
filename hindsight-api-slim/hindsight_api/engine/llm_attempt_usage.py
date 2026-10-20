# fork: attempt-usage
"""Per-attempt usage for Codex calls, folded into the one trace row each call writes.

Upstream keeps one usage slot per LLM call: each provider stash overwrites the last
("the last attempt wins"), and a Codex attempt that ends without ``response.completed``
(a runaway stream, a deadline) stashes nothing at all. So a call whose first attempt ran
away and whose second succeeded is traced with the second attempt's tokens only, and a
call that ran away on every attempt is traced with NULL tokens.

This module keeps a ledger of a Codex call's attempts and, when the trace row is written,
replaces its token counts with the sum over the attempts:

- an attempt that stashed usage counts that usage (the last stash of the attempt);
- an attempt that received deltas but stashed nothing counts an estimate: for input, the
  provider-counted input of another attempt of the call (same messages), else the messages at
  chars/4; for output, one token per delta event received;
- an attempt that received no delta event counts nothing (the request never ran).

The row is marked ``usage_source="estimate"`` whenever any counted attempt was estimated
(``FORK-COST-ACCOUNTING.md``). A call with one attempt and provider usage is left exactly
as upstream writes it.

The ledger lives for one ``LLMProvider`` call: ``set_response_usage`` (the wrapper's
per-call reset) opens it and ``reset_response_usage`` closes it. Only Codex opens
attempts (in ``_iter_sse_lines``), so every other provider is unaffected. Everything is
inert with ``HINDSIGHT_API_LLM_COST_ENABLED`` off, and no hook raises.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Protocol

from .sse_delta_scan import DeltaEventScanner

logger = logging.getLogger(__name__)


class _Usage(Protocol):
    """The fields of ``llm_trace.LLMResponseUsage`` read here (not imported: llm_trace imports this module)."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    thoughts_tokens: int


@dataclass
class _Attempt:
    """One request to the backend within a call."""

    delta_events: int = 0
    usage: _Usage | None = None
    estimated: bool = False
    cache_write_tokens: int | None = None


@dataclass
class _Ledger:
    token: Token | None = None
    attempts: list[_Attempt] = field(default_factory=list)


_ledger: ContextVar[_Ledger | None] = ContextVar("fork_attempt_usage_ledger", default=None)


def _cost_enabled() -> bool:
    from .llm_cost import _cost_enabled as enabled

    return enabled()


# --- call scope (hooked into llm_trace.set_response_usage / reset_response_usage) ---------------


def open_call_ledger() -> None:
    """Start a ledger for the call that is starting. A no-op with the feature off."""
    try:
        if not _cost_enabled():
            return
        ledger = _Ledger()
        ledger.token = _ledger.set(ledger)
    except Exception as exc:
        logger.debug("attempt-usage ledger not opened: %s", type(exc).__name__)


def close_call_ledger() -> None:
    """End the current call's ledger, restoring the one around it (if any)."""
    ledger = _ledger.get()
    if ledger is None or ledger.token is None:
        return
    try:
        _ledger.reset(ledger.token)
    except ValueError:
        # Opened in another context (a copied context in a child task): just drop it here.
        _ledger.set(None)


# --- attempts (hooked into codex_llm._iter_sse_lines) --------------------------------------------


class AttemptMeter:
    """Counts the delta events of one Codex attempt from the raw body. Inert when no ledger is open.

    Fed every decoded chunk as it arrives, before the size guard can abandon the stream, so the
    chunk that trips the guard is counted too. Lines are split and matched by the scanner the
    output cap uses (``sse_delta_scan``), so both count the same events.
    """

    __slots__ = ("_attempt", "_scanner")

    def __init__(self, attempt: _Attempt | None) -> None:
        self._attempt = attempt
        self._scanner = DeltaEventScanner()

    def feed(self, text: str) -> None:
        if self._attempt is None:
            return
        self._attempt.delta_events += self._scanner.feed(text)


def begin_codex_attempt() -> AttemptMeter:
    """Record that a Codex request is being streamed; its lines go through the returned meter."""
    ledger = _ledger.get()
    if ledger is None:
        return AttemptMeter(None)
    attempt = _Attempt()
    ledger.attempts.append(attempt)
    return AttemptMeter(attempt)


# --- usage (hooked into llm_trace.stash_response_usage) ------------------------------------------


def note_stashed_usage(usage: _Usage | None) -> None:
    """Attach a stash to the current attempt (the last stash of an attempt wins)."""
    ledger = _ledger.get()
    if ledger is None or not ledger.attempts or usage is None:
        return
    try:
        from .llm_cost import FORK_USAGE_KWARG, codex_usage_kwargs

        attempt = ledger.attempts[-1]
        attempt.usage = usage
        # Codex notes the breakdown of every usage object it parsed from the stream; anything
        # else (its chars/4 fallback) comes back as an estimate. Read now, while it is the latest note.
        breakdown = codex_usage_kwargs(usage).get(FORK_USAGE_KWARG)
        attempt.estimated = breakdown is None or breakdown.usage_source != "provider"
        attempt.cache_write_tokens = breakdown.cache_write_tokens if breakdown is not None else None
    except Exception as exc:
        logger.debug("attempt-usage stash not noted: %s", type(exc).__name__)


# --- the trace row (decorates LLMTraceRecorder.record_llm_call) ----------------------------------


@dataclass(frozen=True)
class _Totals:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    thoughts_tokens: int
    estimated: bool
    cache_write_tokens: int | None


def _message_chars(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else message
        total += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
    return total


def _totals(attempts: list[_Attempt], messages: object) -> _Totals | None:
    counted = [a for a in attempts if a.usage is not None or a.delta_events > 0]
    if not counted:
        return None
    # Every attempt of a call sends the same messages, so an attempt with provider counts gives the
    # runaway attempts' input exactly; chars/4 is the fallback (a lower bound: provider counts run
    # ~1.5x chars/4 on the reference deployment's refresh inputs). Cached tokens are not copied: whether
    # the runaway attempt hit the prompt cache is unknown, so its input is priced uncached.
    provider_inputs = [a.usage.input_tokens for a in counted if a.usage is not None and not a.estimated]
    input_estimate = provider_inputs[-1] if provider_inputs else _message_chars(messages) // 4
    input_tokens = output_tokens = cached_tokens = thoughts_tokens = 0
    write_counts: list[int] = []
    estimated = False
    for attempt in counted:
        if attempt.usage is None:
            input_tokens += input_estimate
            output_tokens += attempt.delta_events
            estimated = True
            continue
        input_tokens += attempt.usage.input_tokens or 0
        output_tokens += attempt.usage.output_tokens or 0
        cached_tokens += attempt.usage.cached_tokens or 0
        thoughts_tokens += attempt.usage.thoughts_tokens or 0
        estimated = estimated or attempt.estimated
        if attempt.cache_write_tokens is not None:
            write_counts.append(attempt.cache_write_tokens)
    if len(counted) == 1 and not estimated:
        return None  # one attempt with provider counts: the row is already right
    return _Totals(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        thoughts_tokens=thoughts_tokens,
        estimated=estimated,
        cache_write_tokens=sum(write_counts) if write_counts else None,
    )


@dataclass
class _RowUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    thoughts_tokens: int


def _with_success_row_as_last(attempts: list[_Attempt], kwargs: dict[str, Any], call_usage: object) -> list[_Attempt]:
    """A success row is recorded by the provider from the attempt that succeeded: its counts are that
    attempt's, whether or not the provider stashed them (a path that never stashes still counts)."""
    from .llm_cost import CallUsage

    last = attempts[-1]
    breakdown = call_usage if isinstance(call_usage, CallUsage) else None
    usage = _RowUsage(
        input_tokens=kwargs.get("input_tokens") or 0,
        output_tokens=kwargs.get("output_tokens") or 0,
        cached_tokens=kwargs.get("cached_tokens") or 0,
        thoughts_tokens=kwargs.get("thoughts_tokens") or 0,
    )
    replaced = _Attempt(
        delta_events=last.delta_events,
        usage=usage,
        estimated=breakdown.usage_source == "estimate" if breakdown is not None else last.estimated,
        cache_write_tokens=breakdown.cache_write_tokens if breakdown is not None else last.cache_write_tokens,
    )
    return [*attempts[:-1], replaced]


def _folded_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    ledger = _ledger.get()
    if ledger is None or not ledger.attempts:
        return kwargs
    from .llm_cost import FORK_USAGE_KWARG, CallUsage

    attempts = ledger.attempts
    if kwargs.get("error") is None:
        attempts = _with_success_row_as_last(attempts, kwargs, kwargs.get(FORK_USAGE_KWARG))
    totals = _totals(attempts, kwargs.get("messages"))
    if totals is None:
        return kwargs

    base = kwargs.get(FORK_USAGE_KWARG)
    fields = base.model_dump(exclude_none=True) if isinstance(base, CallUsage) else {}
    for name in ("cache_write_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens"):
        fields.pop(name, None)
    if totals.cache_write_tokens is not None:
        fields["cache_write_tokens"] = totals.cache_write_tokens
    fields["usage_source"] = "estimate" if totals.estimated else "provider"
    return {
        **kwargs,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "cached_tokens": totals.cached_tokens,
        "thoughts_tokens": totals.thoughts_tokens,
        FORK_USAGE_KWARG: CallUsage(**fields),
    }


def fold_attempt_usage(record: Callable[..., None]) -> Callable[..., None]:
    """Decorate ``record_llm_call`` so the row carries every attempt's usage, not the last one's."""

    @functools.wraps(record)
    def wrapper(self: object, *args: Any, **kwargs: Any) -> None:
        if not args:  # every caller passes keywords (tracing's fan-out does); anything else is left alone
            try:
                kwargs = _folded_kwargs(kwargs)
            except Exception as exc:
                logger.debug("attempt-usage not folded: %s", type(exc).__name__)
        record(self, *args, **kwargs)

    return wrapper

# fork: codex-output-cap, attempt-usage
"""One scan per Codex attempt: the output-delta cap and the attempt meter (``FORK-CODEX-OUTPUT-CAP.md``).

``CodexLLM`` never sends a token limit, so a backend that wedges into runaway generation is
bounded only by the raw-body ceiling (``_MAX_SSE_BODY_CHARS``) and the wall-clock deadline. The
body ceiling counts characters of the SSE envelope, about 287 per output token on the ChatGPT
backend, and that envelope grows with the reply, so it is only a loose proxy for output tokens.
The cap counts the delta events themselves: one per output token, measured on the same backend.

Every ``event: response.<...>.delta`` line counts (``sse_delta_scan``): each carries a piece of
model-generated output (text, refusal, tool arguments, reasoning summary), and each grows without
bound when generation runs away. Reasoning tokens are not streamed, so nothing here sees them.

The same count feeds the attempt meter of ``llm_attempt_usage`` (the output estimate of an attempt
that never completed), so a chunk is split into lines once for both. The meter records only when a
call ledger is open (``HINDSIGHT_API_LLM_COST_ENABLED`` on); the cap does not depend on that flag.

The cap raises the body ceiling's ``CodexRunawayStreamError``, so it is handled as that one is:
``call()`` retries it up to ``max_retries``; ``call_with_tools`` raises it at once. With
``HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS`` unset and no ledger open, ``feed`` returns at once.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from ...config import ENV_CODEX_MAX_OUTPUT_DELTAS, _parse_optional_positive_int, codex_output_cap_env
from ..llm_attempt_usage import open_codex_attempt
from ..sse_delta_scan import DeltaEventScanner
from . import codex_llm


def max_output_deltas_from_env() -> int | None:
    """The cap: a positive integer, or None when unset or empty. Anything else raises ``ValueError``.

    Called when a ``CodexLLM`` is built (at startup for every configured Codex lane), so a typo
    stops the server instead of silently leaving the cap off. ``0`` is an error too.
    """
    return _parse_optional_positive_int(ENV_CODEX_MAX_OUTPUT_DELTAS, codex_output_cap_env())


def attempt_scan_factory() -> Callable[[], CodexAttemptScan]:
    """What ``CodexLLM`` keeps: a fresh scan per streamed request (each retry counts from zero)."""
    return partial(CodexAttemptScan, max_output_deltas_from_env())


class CodexAttemptScan:
    """Counts the output delta events of one streamed request, fed every decoded chunk as it arrives.

    Fed before the body-ceiling check, so the meter counts the chunk that trips either guard. When one
    chunk crosses both the cap and the body ceiling, the cap stays silent and the ceiling's error is the
    one raised. The whole chunk is counted before the check, so the stream is abandoned at the chunk
    that crosses the cap (the parser never sees that chunk's lines).
    """

    __slots__ = ("_attempt", "_cap", "_count", "_scanner")

    def __init__(self, cap: int | None) -> None:
        self._cap = cap
        self._attempt = open_codex_attempt()
        self._count = 0
        self._scanner = DeltaEventScanner()

    def feed(self, text: str, seen_chars: int) -> None:
        if self._cap is None and self._attempt is None:
            return
        deltas = self._scanner.feed(text)
        if not deltas:
            return
        self._count += deltas
        if self._attempt is not None:
            self._attempt.delta_events += deltas
        if self._cap is not None and self._count > self._cap and seen_chars <= codex_llm._MAX_SSE_BODY_CHARS:
            raise codex_llm.CodexRunawayStreamError(
                f"Codex response exceeded {self._cap} output delta events ({ENV_CODEX_MAX_OUTPUT_DELTAS}) "
                "without completing; abandoning the stream"
            )

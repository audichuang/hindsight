# fork: codex-output-cap
"""A ceiling on how many output delta events one Codex attempt may stream (``FORK-CODEX-OUTPUT-CAP.md``).

``CodexLLM`` never sends a token limit, so a backend that wedges into runaway generation is
bounded only by the raw-body ceiling (``_MAX_SSE_BODY_CHARS``) and the wall-clock deadline. The
body ceiling counts characters of the SSE envelope, about 287 per output token on the ChatGPT
backend, and that envelope grows with the reply (the sequence number gains digits), so a
character ceiling is only a loose proxy for output tokens. This counts the delta events
themselves: one per output token, measured on the same backend.

Which events count: every ``event: response.<...>.delta`` line. In the Responses API every
``*.delta`` event carries a piece of model-generated output (visible text, refusal text, tool
call arguments, reasoning summary or reasoning text, custom tool input), and every one of them
grows without bound when generation runs away: a tool call whose arguments never close is as
much a runaway as a text reply that never ends. Non-generating events (``*.added``, ``*.done``,
``response.created``/``in_progress``/``completed``) are not counted. Tool-argument deltas were
measured at one per output token too. Reasoning tokens are not streamed (the backend sends them
encrypted at the end), so neither this cap nor the body ceiling sees them.

Lines are split and matched by ``sse_delta_scan``, the scanner the attempt meter of
``llm_attempt_usage`` also uses, so a cap of N stops an attempt the meter counts at N+1 output
tokens or more. The cap keeps its own scanner instance because the meter counts only with
``HINDSIGHT_API_LLM_COST_ENABLED`` on, and a safety bound must not depend on that flag.

The error is the body ceiling's ``CodexRunawayStreamError``, so it is handled as that one is:
``call()`` retries it up to ``max_retries``; ``call_with_tools`` raises it at once (upstream retries
only auth failures there) and the reflect agent moves on to its next iteration.

Inert by default: with ``HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS`` unset (or empty), ``feed`` returns
at once and the stream is handled exactly as upstream handles it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import ENV_CODEX_MAX_OUTPUT_DELTAS, codex_output_cap_env
from ..sse_delta_scan import DeltaEventScanner
from .codex_llm import CodexRunawayStreamError


def parse_max_output_deltas(raw: str | None) -> int | None:
    """The cap from the raw env value: a positive integer, or None when unset or empty.

    Anything else raises ``ValueError``: a typo must fail the server at startup rather than
    silently leave the cap off. ``0`` is an error too; unset the variable to turn the cap off.
    """
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    if not (value.isascii() and value.isdigit()) or int(value) < 1:
        raise ValueError(f"{ENV_CODEX_MAX_OUTPUT_DELTAS} must be a positive integer (got {raw!r})")
    return int(value)


class _OutputDeltaCounter:
    """Counts the output delta events of one attempt from the raw body and stops it past the cap.

    Fed every decoded chunk as it arrives. Lines are split and matched by ``sse_delta_scan``, the
    scanner the attempt meter also uses, with the line-break rule of the stream parser (a lone
    ``\\r`` ends a line), so a body cannot reach the parser in lines the counter does not see. The
    whole chunk is counted before the check, so the stream is abandoned at the chunk that crosses
    the cap (the parser never sees that chunk's lines).
    """

    __slots__ = ("_cap", "_count", "_scanner")

    def __init__(self, cap: int | None) -> None:
        self._cap = cap
        self._count = 0
        self._scanner = DeltaEventScanner()

    def feed(self, text: str) -> None:
        if self._cap is None:
            return
        self._count += self._scanner.feed(text)
        if self._count > self._cap:
            raise CodexRunawayStreamError(
                f"Codex response exceeded {self._cap} output delta events ({ENV_CODEX_MAX_OUTPUT_DELTAS}) "
                "without completing; abandoning the stream"
            )


@dataclass(frozen=True)
class CodexOutputCap:
    """The parsed setting, resolved once per ``CodexLLM`` (so a bad value fails its construction)."""

    max_output_deltas: int | None

    @classmethod
    def from_env(cls) -> CodexOutputCap:
        return cls(parse_max_output_deltas(codex_output_cap_env()))

    def begin_attempt(self) -> _OutputDeltaCounter:
        """A fresh counter for one streamed request (each retry starts again from zero)."""
        return _OutputDeltaCounter(self.max_output_deltas)

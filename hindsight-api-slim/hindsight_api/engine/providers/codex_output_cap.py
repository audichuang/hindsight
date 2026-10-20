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
``response.created``/``in_progress``/``completed``) are not counted. It is the same definition
the attempt meter of ``llm_attempt_usage`` uses for its output estimate, so a cap of N stops an
attempt the meter would have counted at N+ output tokens. Reasoning tokens are not streamed (the
backend sends them encrypted at the end), so neither counter sees them.

The cap is independent of ``HINDSIGHT_API_LLM_COST_ENABLED``: the attempt meter counts only for
cost accounting, and a safety bound must not depend on that flag.

Inert by default: with ``HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS`` unset (or empty), ``feed`` returns
at once and the stream is handled exactly as upstream handles it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import ENV_CODEX_MAX_OUTPUT_DELTAS, codex_output_cap_env
from .codex_llm import CodexRunawayStreamError

# An event-name line is far shorter than this; a longer partial line only needs its start.
_LINE_PREFIX_KEPT = 256


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

    Fed every decoded chunk as it arrives. Lines are rejoined across chunk boundaries, so an
    event-name line split between two chunks is counted once. The whole chunk is counted before
    the check, so the stream is abandoned at the chunk that crosses the cap (the parser never
    sees that chunk's lines).
    """

    __slots__ = ("_cap", "_carry", "_count")

    def __init__(self, cap: int | None) -> None:
        self._cap = cap
        self._carry = ""
        self._count = 0

    def feed(self, text: str) -> None:
        if self._cap is None:
            return
        lines = (self._carry + text).split("\n")
        self._carry = lines.pop()[:_LINE_PREFIX_KEPT]
        for line in lines:
            if line.startswith("event: response.") and line.rstrip().endswith(".delta"):
                self._count += 1
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

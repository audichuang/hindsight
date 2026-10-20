# fork: attempt-usage; fork: codex-output-cap
"""Count the output delta events in a Codex SSE body as it arrives in chunks.

Used once per Codex attempt by ``providers/codex_output_cap.CodexAttemptScan``, whose count
feeds both the output cap (runaway guard) and the attempt meter of ``llm_attempt_usage``
(output-token estimate for cost), so both see exactly the same events. Lines are split by the rule ``CodexLLM._iter_sse_lines`` uses, ``str.splitlines``: a
lone ``\\r`` ends a line as well as ``\\n`` and ``\\r\\n``. Splitting on ``\\n`` only (what both
counters did first) let a body with CR-only line ends reach the parser intact while counting
no event at all, so the cap could be bypassed.

A ``\\r\\n`` split across two chunks ends the line at the ``\\r``; the ``\\n`` that opens the next
chunk then yields an empty line, which is never an event line, so nothing is counted twice.
"""

from __future__ import annotations

# An event-name line is far shorter than this; a longer partial line only needs its start.
_LINE_PREFIX_KEPT = 256


def is_output_delta_event(line: str) -> bool:
    """``event: response.<...>.delta``: every such event carries model-generated output."""
    return line.startswith("event: response.") and line.rstrip().endswith(".delta")


class DeltaEventScanner:
    """Counts output delta event lines across chunks, rejoining a line split between two chunks."""

    __slots__ = ("_carry",)

    def __init__(self) -> None:
        self._carry = ""

    def feed(self, text: str) -> int:
        """The number of output delta event lines completed by ``text``."""
        if not text:
            return 0
        joined = self._carry + text
        lines = joined.splitlines()
        # The last character ends a line exactly when ``str.splitlines`` alone breaks on it, so the
        # break set is the parser's own (``codex_llm._LINE_BREAK_RE`` lists the same characters).
        if joined[-1:].splitlines() == [""]:
            self._carry = ""
        else:
            self._carry = lines.pop()[:_LINE_PREFIX_KEPT]
        return sum(1 for line in lines if is_output_delta_event(line))

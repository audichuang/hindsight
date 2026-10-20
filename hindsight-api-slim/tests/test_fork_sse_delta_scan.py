"""fork: attempt-usage, codex-output-cap — both delta counters split lines the way the stream parser does.

``CodexLLM._iter_sse_lines`` ends a line on ``\\n``, ``\\r\\n`` or a lone ``\\r`` (``str.splitlines``).
The output cap and the attempt meter first split on ``\\n`` only, so a body with CR-only line ends
reached the parser intact while neither counted a single delta: the cap could be bypassed and the
meter's runaway estimate was 0. Both now read one count per attempt (``CodexAttemptScan``).
"""

import json
from collections.abc import AsyncIterator

import pytest

from hindsight_api.engine import llm_attempt_usage, llm_trace
from hindsight_api.engine.providers.codex_llm import CodexLLM, CodexRunawayStreamError
from hindsight_api.engine.sse_delta_scan import DeltaEventScanner
from tests.codex_stream_stub import CodexReply, stub_codex_stream
from tests.fork_helpers import build_codex, cost_env_cleared, enable_cost

CAP_ENV = "HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS"
_BREAKS = {"lf": "\n", "cr": "\r", "crlf": "\r\n"}


def _body(deltas: int, eol: str | list[str]) -> str:
    """``deltas`` text-delta events plus non-delta events, every line ended by ``eol`` (cycled if a list)."""
    lines: list[str] = ["event: response.created", "data: {}", ""]
    for i in range(deltas):
        lines += ["event: response.output_text.delta", "data: " + json.dumps({"delta": "x", "n": i}), ""]
    lines += ["event: response.output_text.done", "data: {}", ""]
    ends = eol if isinstance(eol, list) else [eol]
    return "".join(line + ends[i % len(ends)] for i, line in enumerate(lines))


def _chunks(body: str, how: str) -> list[str]:
    if how == "whole":
        return [body]
    assert how == "every-7"
    return [body[i : i + 7] for i in range(0, len(body), 7)]


def _split_crlf(body: str) -> list[str]:
    """Cut every ``\\r\\n`` between its ``\\r`` and its ``\\n``."""
    parts = body.split("\r\n")
    out = [parts[0] + "\r"]
    for piece in parts[1:-1]:
        out.append("\n" + piece + "\r")
    out.append("\n" + parts[-1])
    return out


class _FakeContent:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [c.encode() for c in chunks]

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[str]) -> None:
        self.content = _FakeContent(chunks)


async def _drain(llm: CodexLLM, chunks: list[str]) -> list[str]:
    return [line async for line in llm._iter_sse_lines(_FakeResponse(chunks))]  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    yield from cost_env_cleared(monkeypatch, CAP_ENV)


_CASES = [
    ("lf", "whole"),
    ("cr", "whole"),
    ("crlf", "whole"),
    ("cr", "every-7"),
    ("crlf", "every-7"),
    ("mixed", "whole"),
    ("mixed", "every-7"),
    ("crlf", "crlf-split"),
]


def _eol(style: str) -> str | list[str]:
    return ["\n", "\r", "\r\n"] if style == "mixed" else _BREAKS[style]


def _case_chunks(deltas: int, style: str, how: str) -> list[str]:
    body = _body(deltas, _eol(style))
    return _split_crlf(body) if how == "crlf-split" else _chunks(body, how)


# --- the output cap --------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("style", "how"), _CASES)
async def test_the_cap_sees_every_line_ending_the_parser_sees(monkeypatch, style, how):
    monkeypatch.setenv(CAP_ENV, "3")
    llm = build_codex("m")
    lines = await _drain(llm, _case_chunks(3, style, how))
    # The parser got all three deltas, so the counter must agree there were exactly three...
    assert lines.count("event: response.output_text.delta") == 3
    # ...and a fourth crosses the cap whatever the line ends and chunk boundaries.
    with pytest.raises(CodexRunawayStreamError, match="exceeded 3 output delta events"):
        await _drain(llm, _case_chunks(4, style, how))


@pytest.mark.asyncio
async def test_a_cr_only_reply_cannot_bypass_the_cap_through_call(monkeypatch):
    # The reviewer's reproduction: with cap 1, two deltas separated by LF raised but the same
    # events separated by CR completed as a success.
    monkeypatch.setenv(CAP_ENV, "1")
    llm = build_codex("m")
    body = _body(2, "\r") + "event: response.completed\rdata: {}\r\rdata: [DONE]\r\r"
    try:
        with stub_codex_stream(llm, CodexReply(text=body)), pytest.raises(CodexRunawayStreamError):
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    finally:
        await llm.cleanup()


# --- the attempt meter -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("style", "how"), _CASES)
async def test_the_meter_counts_every_line_ending_the_parser_sees(monkeypatch, style, how):
    enable_cost(monkeypatch)
    llm = build_codex("m")
    token = llm_trace.set_response_usage(None)  # opens the call's ledger
    try:
        await _drain(llm, _case_chunks(5, style, how))
        ledger = llm_attempt_usage._ledger.get()
        assert ledger is not None
        assert [a.delta_events for a in ledger.attempts] == [5]
    finally:
        llm_trace.reset_response_usage(token)


# --- the shared scanner ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_scan_feeds_the_meter_and_the_cap(monkeypatch):
    # One count per attempt: the meter records exactly the deltas the cap saw, including the chunk that trips it.
    monkeypatch.setenv(CAP_ENV, "3")
    enable_cost(monkeypatch)
    llm = build_codex("m")
    token = llm_trace.set_response_usage(None)
    try:
        with pytest.raises(CodexRunawayStreamError):
            await _drain(llm, [_body(4, "\n")])
        ledger = llm_attempt_usage._ledger.get()
        assert ledger is not None
        assert [a.delta_events for a in ledger.attempts] == [4]
    finally:
        llm_trace.reset_response_usage(token)


def test_a_crlf_split_between_chunks_counts_once():
    scanner = DeltaEventScanner()
    assert scanner.feed("event: response.output_text.delta\r") == 1
    assert scanner.feed("\ndata: {}\r\n\r\n") == 0

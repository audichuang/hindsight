"""fork: codex-output-cap — a Codex attempt is stopped after N output delta events (FORK-CODEX-OUTPUT-CAP.md).

The body-size ceiling counts characters of the SSE envelope; this cap counts the delta events
themselves (one per output token on the ChatGPT backend). It is off unless
``HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS`` is set, and it does not depend on the cost flag.
"""

import inspect
import json
from unittest.mock import patch

import pytest

from hindsight_api.engine.providers import codex_llm
from hindsight_api.engine.providers.codex_llm import CodexLLM, CodexRunawayStreamError
from hindsight_api.engine.providers.codex_output_cap import CodexAttemptScan, max_output_deltas_from_env
from tests.codex_stream_stub import CodexReply, stub_codex_stream, stub_codex_stream_with

ENV = "HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS"
_MESSAGES = [{"role": "user", "content": "list the claims"}]


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"


def _text_delta(n: int) -> str:
    return _event("response.output_text.delta", {"sequence_number": n, "delta": "x"})


def _completed(output_tokens: int) -> str:
    usage = {"input_tokens": 10, "output_tokens": output_tokens}
    return _event("response.completed", {"response": {"usage": usage}}) + "data: [DONE]\n\n"


def _text_reply(deltas: int) -> str:
    """A complete reply of ``deltas`` text deltas, preceded by events that are not output deltas."""
    head = (
        _event("response.created", {})
        + _event("response.in_progress", {})
        + _event("response.output_item.added", {"item": {"type": "message"}})
        + _event("response.content_part.added", {})
    )
    tail = (
        _event("response.output_text.done", {"text": "x" * deltas})
        + _event("response.content_part.done", {})
        + _event(
            "response.output_item.done",
            {"item": {"type": "message", "content": [{"type": "output_text", "text": "x" * deltas}]}},
        )
    )
    return head + "".join(_text_delta(i) for i in range(deltas)) + tail + _completed(deltas)


def _tool_reply(deltas: int) -> str:
    """A complete tool call whose arguments arrive as ``deltas`` argument deltas."""
    pieces = ['{"q":"', *("a" * (deltas - 2)), '"}']  # deltas >= 2
    args = "".join(pieces)
    item = {"type": "function_call", "status": "completed", "name": "search", "arguments": args, "call_id": "c1"}
    return (
        _event("response.output_item.added", {"item": {"type": "function_call"}})
        + "".join(_event("response.function_call_arguments.delta", {"delta": p}) for p in pieces)
        + _event("response.function_call_arguments.done", {"arguments": args})
        + _event("response.output_item.done", {"item": item})
        + _completed(deltas)
    )


def _build_llm() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


async def _call(body: str) -> str:
    llm = _build_llm()
    try:
        with stub_codex_stream(llm, CodexReply(text=body)):
            result = await llm.call(messages=_MESSAGES, max_retries=0)
        return result.content
    finally:
        await llm.cleanup()


async def _call_with_tools(body: str) -> int:
    llm = _build_llm()
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    try:
        with stub_codex_stream(llm, CodexReply(text=body)):
            result = await llm.call_with_tools(messages=_MESSAGES, tools=tools, max_retries=0)
        return len(result.tool_calls)
    finally:
        await llm.cleanup()


# --- the setting -----------------------------------------------------------------------------------


def test_unset_or_empty_means_no_cap(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert max_output_deltas_from_env() is None
    monkeypatch.setenv(ENV, "")
    assert max_output_deltas_from_env() is None


@pytest.mark.parametrize(("raw", "cap"), [("25000", 25000), (" 25000 ", 25000)])
def test_a_positive_integer_is_the_cap(monkeypatch, raw, cap):
    # Parsed by config._parse_optional_positive_int, the rule of the other integer settings (int()).
    monkeypatch.setenv(ENV, raw)
    assert max_output_deltas_from_env() == cap


@pytest.mark.parametrize("raw", ["0", "-5", "25k", "2.5", "1e4", "abc", "  "])
def test_a_bad_value_fails_provider_construction(monkeypatch, raw):
    # The provider is built when the server starts its LLM lanes, so a typo stops startup
    # instead of silently leaving the cap off.
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=ENV):
        _build_llm()


# --- behaviour -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_the_setting_a_long_reply_is_not_capped(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert await _call(_text_reply(500)) == "x" * 500


@pytest.mark.asyncio
async def test_more_deltas_than_the_cap_abandon_the_stream(monkeypatch):
    monkeypatch.setenv(ENV, "100")
    with pytest.raises(CodexRunawayStreamError, match="exceeded 100 output delta events"):
        await _call(_text_reply(101))


@pytest.mark.asyncio
async def test_exactly_the_cap_completes(monkeypatch):
    monkeypatch.setenv(ENV, "100")
    assert await _call(_text_reply(100)) == "x" * 100


@pytest.mark.asyncio
async def test_events_that_are_not_output_deltas_are_not_counted(monkeypatch):
    # 100 text deltas plus many more non-delta events (and a keep-alive comment and data lines
    # mentioning ".delta"): only the deltas count, so a cap of 100 is not crossed.
    monkeypatch.setenv(ENV, "100")
    noise = "".join(
        _event("response.in_progress", {"note": "event: response.output_text.delta"}) + ": keep-alive\n\n"
        for _ in range(300)
    )
    body = _text_reply(100)
    assert await _call(noise + body) == "x" * 100


@pytest.mark.asyncio
async def test_call_with_tools_is_capped_on_argument_deltas(monkeypatch):
    monkeypatch.setenv(ENV, "50")
    assert await _call_with_tools(_tool_reply(50)) == 1
    with pytest.raises(CodexRunawayStreamError, match="exceeded 50 output delta events"):
        await _call_with_tools(_tool_reply(51))


@pytest.mark.asyncio
async def test_each_attempt_counts_from_zero(monkeypatch):
    # Two 60-delta attempts under a cap of 100: the retry is not charged for the first attempt.
    monkeypatch.setenv(ENV, "100")
    llm = _build_llm()
    try:
        with stub_codex_stream(llm, CodexReply(text=_text_reply(60))) as mock:
            assert (await llm.call(messages=_MESSAGES, max_retries=0)).content == "x" * 60
            assert (await llm.call(messages=_MESSAGES, max_retries=0)).content == "x" * 60
        assert mock.call_count == 2
    finally:
        await llm.cleanup()


def test_an_event_line_split_across_chunks_counts_once():
    scan = CodexAttemptScan(2)
    for piece in ("event: response.output_te", "xt.delta\ndata: {}\n\nevent: resp", "onse.output_text.delta\r\n"):
        scan.feed(piece, 0)
    with pytest.raises(CodexRunawayStreamError):
        scan.feed("data: {}\n\nevent: response.function_call_arguments.delta\n", 0)


def test_the_cap_does_not_depend_on_the_cost_flag(monkeypatch):
    # The attempt meter of llm_attempt_usage counts only with the cost flag on; this counter
    # must count with it off.
    monkeypatch.delenv("HINDSIGHT_API_LLM_COST_ENABLED", raising=False)
    scan = CodexAttemptScan(1)
    with pytest.raises(CodexRunawayStreamError):
        scan.feed(_text_delta(0) + _text_delta(1), 0)


class _OneChunk:
    """A response whose whole body arrives as one chunk."""

    def __init__(self, body: str) -> None:
        self.content = self
        self._body = body.encode()

    async def iter_any(self):
        yield self._body


@pytest.mark.asyncio
@pytest.mark.parametrize(("ceiling_offset", "message"), [(-1, "characters without completing"), (0, "output delta")])
async def test_a_chunk_past_both_guards_raises_the_body_ceiling(monkeypatch, ceiling_offset, message):
    # The scan is fed before the body-ceiling check (so the meter counts the chunk), but when one chunk
    # crosses both guards the ceiling's error is the one raised, as when they were two counters.
    monkeypatch.setenv(ENV, "1")
    body = _text_reply(5)
    monkeypatch.setattr(codex_llm, "_MAX_SSE_BODY_CHARS", len(body) + ceiling_offset)
    llm = _build_llm()
    with pytest.raises(CodexRunawayStreamError, match=message):
        [line async for line in llm._iter_sse_lines(_OneChunk(body))]  # type: ignore[arg-type]


def _first_then(first: str, then: str):
    replies = iter([first])
    return lambda _url, **_kw: CodexReply(text=next(replies, then))


@pytest.mark.asyncio
async def test_call_retries_a_capped_attempt_and_recovers(monkeypatch):
    # FORK-CODEX-OUTPUT-CAP.md: call() handles a cap trip as it handles the body ceiling, a
    # transient error retried with a fresh counter.
    monkeypatch.setenv(ENV, "50")
    llm = _build_llm()
    try:
        with (
            patch("asyncio.sleep"),
            stub_codex_stream_with(llm, _first_then(_text_reply(51), _text_reply(50))) as mock,
        ):
            result = await llm.call(messages=_MESSAGES, max_retries=1)
        assert result.content == "x" * 50
        assert mock.call_count == 2
    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_call_with_tools_raises_a_capped_attempt_at_once(monkeypatch):
    # FORK-CODEX-OUTPUT-CAP.md: call_with_tools retries only auth failures (upstream), so a cap trip
    # is raised on the first attempt, as the body ceiling is; the reflect agent's next iteration
    # is the retry. If upstream adds a runaway retry here, update the document with this test.
    monkeypatch.setenv(ENV, "50")
    llm = _build_llm()
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    try:
        with stub_codex_stream_with(llm, _first_then(_tool_reply(51), _tool_reply(50))) as mock:
            with pytest.raises(CodexRunawayStreamError):
                await llm.call_with_tools(messages=_MESSAGES, tools=tools, max_retries=1)
        assert mock.call_count == 1
    finally:
        await llm.cleanup()


# --- hooks -----------------------------------------------------------------------------------------


def test_codex_keeps_the_output_cap_hooks():
    init = inspect.getsource(CodexLLM.__init__)
    assert "self._fork_attempt_scan = attempt_scan_factory()  # fork: codex-output-cap" in init
    body = inspect.getsource(CodexLLM._iter_sse_lines)
    assert "attempt = self._fork_attempt_scan()  # fork: codex-output-cap, attempt-usage" in body
    # Right after the character count, before the character ceiling (the meter must count the chunk
    # that trips either guard) and before any line of the chunk is handed to the parser.
    assert (
        "            seen_chars += len(text)\n"
        "            attempt.feed(text, seen_chars)  # fork: codex-output-cap, attempt-usage\n"
        "            if seen_chars > _MAX_SSE_BODY_CHARS:"
    ) in body
    assert body.index("attempt.feed(text, seen_chars)") < body.index("if not _LINE_BREAK_RE.search(text):")


def test_codex_keeps_the_8_mib_ceiling():
    source = inspect.getsource(codex_llm)
    assert "_MAX_SSE_BODY_CHARS = 8 * 1024 * 1024  # fork: codex-output-cap" in source


def test_config_keeps_the_output_cap_env_hook():
    from hindsight_api import config

    assert config.ENV_CODEX_MAX_OUTPUT_DELTAS == ENV
    assert "# fork: codex-output-cap" in inspect.getsource(config)

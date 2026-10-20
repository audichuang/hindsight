"""The Codex stream ceiling keeps long legitimate replies and still cuts off runaways by size.

The stream guard counts the raw SSE body. Measured on the ChatGPT backend, each output
token arrives as one delta event of about 287 characters, so the old 4 MiB ceiling cut a
reply off at ~14.6k output tokens, which split-synthesis map calls of a mental-model
refresh reach in normal operation: in one week no successful map call there went above
~13.6k output tokens, and 32 of ~1100 failed as ``CodexRunawayStreamError``. ``CodexLLM``
never sends ``max_completion_tokens``, so nothing else bounds such a reply.

The ceiling must also stay low enough to trip before the per-attempt deadline at the
observed 43-66 tokens/s: a 16 MiB ceiling (~58k deltas) took 880-1350 s to fill, past
the refresh lane's 900 s deadline, so a runaway ran to the deadline on every attempt.
"""

import json

import pytest

from hindsight_api.engine.providers import codex_llm
from hindsight_api.engine.providers.codex_llm import CodexRunawayStreamError
from tests.codex_stream_stub import CodexReply, stub_codex_stream
from tests.fork_helpers import build_codex

pytestmark = pytest.mark.asyncio

# The shape of one ``response.output_text.delta`` event from the backend: one token of
# text inside an envelope of fixed fields and random padding.
_ENVELOPE = {
    "type": "response.output_text.delta",
    "sequence_number": 123456,
    "item_id": "msg_0123456789abcdef0123456789abcdef0123456789abcdef",
    "output_index": 1,
    "content_index": 0,
    "logprobs": [],
    "obfuscation": "AbCdEfGhIjKlMnOpQrStUv",
}
_TOKENS = 20_000  # a long claims list, above the old cut-off and well under the new one
_RUNAWAY_DELTAS = 30_500  # just past the ceiling (a small unread tail), well under a 16 MiB one
_DELTA_EVENT = "event: response.text.delta\ndata: " + json.dumps({**_ENVELOPE, "delta": " tok"}) + "\n\n"


def _long_reply() -> str:
    completed = (
        "event: response.completed\ndata: "
        + json.dumps(
            {"type": "response.completed", "response": {"usage": {"input_tokens": 50_000, "output_tokens": _TOKENS}}}
        )
        + "\n\n"
    )
    return _DELTA_EVENT * _TOKENS + completed + "data: [DONE]\n\n"


async def test_a_20k_token_reply_completes():
    llm = build_codex()
    body = _long_reply()
    # The reply is bigger than the old ceiling; it is a real reply, not a runaway.
    assert 4 * 1024 * 1024 < len(body) < 8 * 1024 * 1024
    try:
        with stub_codex_stream(llm, CodexReply(text=body)):
            result = await llm.call(messages=[{"role": "user", "content": "list the claims"}], max_retries=0)
        assert result.content == " tok" * _TOKENS
        assert result.usage.output_tokens == _TOKENS
    finally:
        await llm.cleanup()


async def test_a_runaway_is_cut_off_by_size_before_16_mib():
    llm = build_codex()
    # A stream that never completes. Under a 16 MiB ceiling it would end on its own and
    # be returned as a reply; it has to be abandoned by size instead.
    body = _DELTA_EVENT * _RUNAWAY_DELTAS
    assert 8 * 1024 * 1024 < len(body) < 16 * 1024 * 1024
    try:
        with stub_codex_stream(llm, CodexReply(text=body)), pytest.raises(CodexRunawayStreamError):
            await llm.call(messages=[{"role": "user", "content": "list the claims"}], max_retries=0)
    finally:
        await llm.cleanup()


def test_the_ceiling_trips_before_the_refresh_deadline_at_observed_speed():
    # ~287 characters per delta event and 43 output tokens/s (the p50 of successful long
    # refresh calls) must fill the ceiling within the 900 s per-attempt deadline.
    assert codex_llm._MAX_SSE_BODY_CHARS / 287 / 43 < 900

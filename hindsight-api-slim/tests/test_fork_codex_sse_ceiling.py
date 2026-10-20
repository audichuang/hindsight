"""A long but legitimate Codex reply is not abandoned as a runaway.

The stream guard counts the raw SSE body, and each delta event wraps about one token in
a few hundred characters of envelope (inferred from the reference deployment, not
measured on the wire). At the old 4 MiB ceiling a reply was cut off at roughly 13-17k
output tokens, which split-synthesis map calls of a mental-model refresh reach in normal
operation: in one week no successful map call there went above ~13.6k output tokens, and
32 of ~1100 failed as ``CodexRunawayStreamError``. ``CodexLLM`` never sends
``max_completion_tokens``, so nothing else bounds such a reply.
"""

import json
from unittest.mock import patch

import pytest

from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import CodexReply, stub_codex_stream

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


def _long_reply() -> str:
    event = "event: response.text.delta\ndata: " + json.dumps({**_ENVELOPE, "delta": " tok"}) + "\n\n"
    completed = (
        "event: response.completed\ndata: "
        + json.dumps(
            {"type": "response.completed", "response": {"usage": {"input_tokens": 50_000, "output_tokens": _TOKENS}}}
        )
        + "\n\n"
    )
    return event * _TOKENS + completed + "data: [DONE]\n\n"


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


async def test_a_20k_token_reply_completes():
    llm = _build_llm()
    body = _long_reply()
    # The reply is bigger than the old ceiling; it is a real reply, not a runaway.
    assert len(body) > 4 * 1024 * 1024
    try:
        with stub_codex_stream(llm, CodexReply(text=body)):
            result = await llm.call(messages=[{"role": "user", "content": "list the claims"}], max_retries=0)
        assert result.content == " tok" * _TOKENS
        assert result.usage.output_tokens == _TOKENS
    finally:
        await llm.cleanup()

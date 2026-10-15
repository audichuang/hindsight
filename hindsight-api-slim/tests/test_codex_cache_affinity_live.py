"""Live check that consecutive Codex calls of one operation read the backend prompt cache.

Runs only with ``HINDSIGHT_LIVE_CODEX=1`` and a Codex login at ``$CODEX_HOME/auth.json``
(``~/.codex/auth.json`` when ``CODEX_HOME`` is unset). The model is
``HINDSIGHT_LIVE_CODEX_MODEL`` (default ``gpt-6.1-sol``). Four small requests at most.

WARNING: the calls go through the normal Codex token path. If the access token is near
expiry it is refreshed, and ``auth.json`` is rewritten with the new tokens.

Only token counts are printed. No token, header or body is printed.
"""

import asyncio
import os
import uuid
from pathlib import Path

import pytest

from hindsight_api.engine.llm_trace import LLMTraceContext, reset_trace_context, set_trace_context
from hindsight_api.engine.providers.codex_llm import CodexLLM


def _codex_auth_file() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
    return base / "auth.json"


_LIVE_ENABLED = os.environ.get("HINDSIGHT_LIVE_CODEX") == "1"
_MODEL = os.environ.get("HINDSIGHT_LIVE_CODEX_MODEL", "gpt-6.1-sol")

pytestmark = [
    pytest.mark.skipif(
        not (_LIVE_ENABLED and _codex_auth_file().is_file()),
        reason="set HINDSIGHT_LIVE_CODEX=1 and log in to Codex (auth.json) to run",
    ),
    pytest.mark.asyncio,
]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up an archive note by number.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "integer"}},
                "required": ["note"],
            },
        },
    }
]


def _system_prompt(nonce: str) -> str:
    """~3K tokens of stable text: the backend caches only prefixes past 1024 tokens.

    The nonce leads, so no earlier run's cache entry can serve this run's first call;
    a hit on the second call can only come from the first.
    """
    lines = [f"Nonce {nonce}. You are a terse assistant. Archive notes follow."]
    lines += [
        f"Note {i}: shelf {i % 37} holds volume {i * 7 % 101}, catalogued in year {1900 + i}." for i in range(150)
    ]
    return "\n".join(lines)


async def test_second_call_of_one_operation_reads_the_cache():
    llm = CodexLLM(provider="openai-codex", api_key="", base_url="", model=_MODEL, reasoning_effort="low")
    messages = [
        {"role": "system", "content": _system_prompt(uuid.uuid4().hex)},
        {"role": "user", "content": "Reply with the single word: ready."},
    ]
    token = set_trace_context(LLMTraceContext(bank_id="live", operation="reflect", trace_id=str(uuid.uuid4())))
    try:
        first = await llm.call_with_tools(messages=messages, tools=_TOOLS, max_retries=0)
        grown = [
            *messages,
            {"role": "assistant", "content": first.content or "ready"},
            {"role": "user", "content": "Reply with the single word: done."},
        ]
        # The cache entry is written asynchronously; give it a few seconds to land.
        for _ in range(3):
            second = await llm.call_with_tools(messages=grown, tools=_TOOLS, max_retries=0)
            if second.cached_tokens > 0:
                break
            await asyncio.sleep(3)
    finally:
        reset_trace_context(token)
        await llm.cleanup()

    print(
        f"first: input={first.input_tokens} cached={first.cached_tokens}; "
        f"second: input={second.input_tokens} cached={second.cached_tokens}"
    )
    assert first.input_tokens > 1024, "the prefix is too short for the backend to cache"
    assert second.cached_tokens > 0

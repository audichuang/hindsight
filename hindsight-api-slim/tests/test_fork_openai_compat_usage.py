"""Cache-read fields of OpenAI-compatible usage blocks (B5).

Standard ``prompt_tokens_details.cached_tokens`` wins when positive. Provider
fallbacks are DeepSeek ``prompt_cache_hit_tokens``, ``cache_read_input_tokens`` and
Kimi's top-level ``cached_tokens``. ``prompt_tokens`` stays the total prompt.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types import CompletionUsage

from hindsight_api.engine.providers.openai_compatible_llm import (
    _cached_prompt_tokens,
    _usage_from_openai_response,
    visible_token_usage,
)


def _response(usage: object) -> SimpleNamespace:
    return SimpleNamespace(usage=usage)


def _usage(**fields: object) -> CompletionUsage:
    base: dict[str, object] = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
    base.update(fields)
    return CompletionUsage.model_validate(base)


def test_deepseek_hit_tokens_are_cached_and_prompt_stays_total():
    usage = _usage(
        prompt_tokens=100000,
        completion_tokens=500,
        total_tokens=100500,
        prompt_cache_hit_tokens=80000,
        prompt_cache_miss_tokens=20000,
    )
    counts = visible_token_usage(_response(usage))
    assert counts.cached_tokens == 80000
    assert counts.input_tokens == 100000


def test_deepseek_usage_flows_into_llm_response_usage():
    usage = _usage(
        prompt_tokens=100000,
        completion_tokens=500,
        total_tokens=100500,
        prompt_cache_hit_tokens=80000,
        prompt_cache_miss_tokens=20000,
    )
    result = _usage_from_openai_response(_response(usage))
    assert result.cached_tokens == 80000
    assert result.input_tokens == 100000


def test_kimi_top_level_cached_tokens():
    usage = _usage(cached_tokens=42)
    assert _cached_prompt_tokens(usage) == 42
    assert visible_token_usage(_response(usage)).cached_tokens == 42


def test_cache_read_input_tokens_fallback():
    usage = _usage(cache_read_input_tokens=33)
    assert _cached_prompt_tokens(usage) == 33


def test_fallback_order_hit_then_read_then_top_level():
    usage = _usage(prompt_cache_hit_tokens=7, cache_read_input_tokens=8, cached_tokens=9)
    assert _cached_prompt_tokens(usage) == 7
    usage = _usage(cache_read_input_tokens=8, cached_tokens=9)
    assert _cached_prompt_tokens(usage) == 8


def test_standard_details_wins_when_positive():
    usage = _usage(
        prompt_tokens_details={"cached_tokens": 11},
        prompt_cache_hit_tokens=80000,
        cached_tokens=42,
    )
    assert _cached_prompt_tokens(usage) == 11
    assert visible_token_usage(_response(usage)).cached_tokens == 11


def test_fallback_used_when_details_cached_is_zero():
    usage = _usage(prompt_tokens_details={"cached_tokens": 0}, prompt_cache_hit_tokens=5)
    assert _cached_prompt_tokens(usage) == 5


def test_bool_and_non_positive_fallbacks_are_ignored():
    usage = _usage(prompt_cache_hit_tokens=True, cache_read_input_tokens=0, cached_tokens=-3)
    assert _cached_prompt_tokens(usage) == 0


def test_no_usage_yields_zero():
    assert _cached_prompt_tokens(None) == 0
    assert visible_token_usage(_response(None)).cached_tokens == 0


def test_no_cache_fields_yields_zero():
    assert visible_token_usage(_response(_usage())).cached_tokens == 0


def test_magicmock_usage_is_unchanged():
    # Auto-created MagicMock attributes are not real ints, so they never feed a fallback.
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 5
    usage.total_tokens = 105
    usage.prompt_tokens_details.cached_tokens = 0
    usage.completion_tokens_details = None
    counts = visible_token_usage(_response(usage))
    assert counts.cached_tokens == 0
    assert counts.input_tokens == 100
    assert counts.output_tokens == 5


def test_magicmock_usage_with_explicit_details_still_wins():
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 5
    usage.total_tokens = 105
    usage.prompt_tokens_details.cached_tokens = 9
    usage.completion_tokens_details = None
    assert visible_token_usage(_response(usage)).cached_tokens == 9


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"prompt_cache_hit_tokens": 80000}, 80000),
        ({"cache_read_input_tokens": 12}, 12),
        ({"cached_tokens": 3}, 3),
        ({}, 0),
    ],
)
def test_usage_from_openai_response_carries_same_cached_value(fields, expected):
    usage = _usage(**fields)
    result = _usage_from_openai_response(_response(usage))
    assert result.cached_tokens == expected
    assert result.cached_tokens == visible_token_usage(_response(usage)).cached_tokens

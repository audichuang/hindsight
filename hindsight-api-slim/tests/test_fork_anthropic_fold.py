"""Cache writes when the Anthropic usage total is missing (upstream-bound fix to B1).

The fold must count the TTL breakdown (cache_creation.ephemeral_5m/1h) when the reported
cache_creation_input_tokens total is missing or smaller, so input_tokens covers every write.
"""

from types import SimpleNamespace

from hindsight_api.engine.providers.anthropic_llm import _usage_from_anthropic_response


def _response(**usage):
    return SimpleNamespace(usage=SimpleNamespace(**usage))


def test_nested_writes_count_when_the_total_is_missing():
    folded = _usage_from_anthropic_response(
        _response(
            input_tokens=100,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=None,
            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=1000),
        )
    )

    assert folded.input_tokens == 1100  # 100 uncached + 1000 written
    assert folded.cached_tokens == 0


def test_a_reported_total_larger_than_the_breakdown_still_wins():
    folded = _usage_from_anthropic_response(
        _response(
            input_tokens=100,
            output_tokens=5,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=1500,
            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=1000),
        )
    )

    assert folded.input_tokens == 1800  # 100 + 1500 written + 200 read
    assert folded.cached_tokens == 200

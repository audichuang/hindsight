"""Keep consolidation's response schema batch-invariant where it is prompt text (fork: consolidation-cache).

``_build_response_model`` pins ``source_fact_ids`` to the batch's ids (``enum``) and caps
``creates`` (``maxItems``) per batch. A provider that appends the schema to the system
prompt then sends a different prefix on every batch and its prompt cache misses. Those
calls render upstream's schema without those two per-batch values but still validate
with upstream's constrained model, so an over-cap response fails exactly as upstream.
Design and measurements: ``FORK-CONSOLIDATION-CACHE.md``.
"""

from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from ...config import HindsightConfig

_M = TypeVar("_M", bound=BaseModel)

# Providers whose non-strict structured output appends the JSON schema to the system
# prompt / instructions and enforces at most "valid JSON" (``json_object``), not the
# schema. Native ``ollama`` is absent: it sends the schema as ``format`` and enforces it.
_SCHEMA_IN_SYSTEM_PROMPT_UNLESS_STRICT = frozenset(
    {
        "openai-codex",
        "anthropic",
        "xai-oauth",
        "openai-responses",
        # OpenAICompatibleLLM and the providers built on it
        "openai",
        "groq",
        "ollama-cloud",
        "lmstudio",
        "minimax",
        "deepseek",
        "volcano",
        "openrouter",
        "requesty",
        "zai",
        "opencode-go",
        "atlas",
        "meta",
        "fireworks",
        "nous",
        "llamacpp",
    }
)


def schema_is_prompt_text(llm_config: Any, config: "HindsightConfig") -> bool:
    """True when consolidation's response schema reaches the model as system-prompt text only.

    ``llm_config`` is a ``ConfiguredLLMProvider``, which forwards attribute reads to
    the provider it wraps. A multi-LLM chain forwards ``provider`` to its first member
    only, so a chain (it exposes ``members``) keeps upstream's constrained schema.
    """
    if config.llm_strict_schema_consolidation:
        return False
    if getattr(llm_config, "members", None) is not None:
        return False
    provider = getattr(llm_config, "provider", None)
    return isinstance(provider, str) and provider in _SCHEMA_IN_SYSTEM_PROMPT_UNLESS_STRICT


def _drop_per_batch_keywords(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove the two values ``_build_response_model`` varies per batch; keep the rest of upstream's schema."""
    creates = schema.get("properties", {}).get("creates")
    if isinstance(creates, dict):
        creates.pop("maxItems", None)
    for definition in schema.get("$defs", {}).values():
        source_fact_ids = definition.get("properties", {}).get("source_fact_ids")
        if isinstance(source_fact_ids, dict) and isinstance(source_fact_ids.get("items"), dict):
            source_fact_ids["items"].pop("enum", None)
    return schema


def cacheable_response_model(response_model: type[_M], llm_config: Any, config: "HindsightConfig") -> type[_M]:
    """The consolidation hook: ``response_model`` unchanged unless its schema is prompt text.

    When it is, the result validates with ``response_model`` but renders its JSON schema without the
    per-batch fact-id ``enum`` and ``maxItems``. Providers render the schema with
    ``response_format.model_json_schema(...)`` and parse with ``response_format.model_validate(...)``;
    only the former changes. The ``enum`` is schema-only upstream too, the creates cap (``max_length``)
    keeps rejecting an over-cap response, and batch-invariant hints such as ``minItems: 1`` stay in the schema.
    """
    if not schema_is_prompt_text(llm_config, config):
        return response_model

    parent: Any = response_model  # a runtime class; static checkers reject a TypeVar as a base

    class _BatchInvariantSchema(parent):
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            schema = _drop_per_batch_keywords(super().model_json_schema(*args, **kwargs))
            schema["title"] = response_model.__name__
            return schema

    return cast("type[_M]", _BatchInvariantSchema)

# fork: prompt-tone
"""Plain-voice prompt rewrites (FORK-PROMPT-TONE.md): inert when off, every rewrite matching when on."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hindsight_api.config import ENV_PROMPT_TONE
from hindsight_api.engine import prompt_tone
from hindsight_api.engine.prompt_tone import DONE_TOOL_REWRITES, REWRITES, parse_enabled
from hindsight_api.engine.reflect.prompts import (
    build_directives_section,
    build_final_system_prompt,
    build_system_prompt_for_tools,
)
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools
from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema

_ENGINE = Path(prompt_tone.__file__).parent
_DIRECTIVES = [{"name": "No Competitors", "content": "Never mention competitor names."}]
_BANK = {"name": "TestBank", "mission": ""}


@pytest.fixture
def tone(monkeypatch):
    """Set the switch for one test; the cached value is cleared before and after."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(ENV_PROMPT_TONE, raising=False)
        else:
            monkeypatch.setenv(ENV_PROMPT_TONE, value)
        prompt_tone.prompt_tone_enabled.cache_clear()

    yield _set
    prompt_tone.prompt_tone_enabled.cache_clear()


def _retain_config(mode: str) -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    config.llm_supports_string_pattern = False
    return config


def _all_prompt_text() -> str:
    """Every prompt the rewrites target, built through the real (hooked) builders."""
    return "\n".join(
        [
            build_system_prompt_for_tools(
                bank_profile=_BANK, has_mental_models=True, include_observations=True, directives=_DIRECTIVES
            ),
            build_final_system_prompt(directives=_DIRECTIVES),
            _build_extraction_prompt_and_schema(_retain_config("concise")).system_prompt,
            _build_extraction_prompt_and_schema(_retain_config("verbose")).system_prompt,
        ]
    )


def _done_tool(answer_as_document: bool = False) -> dict:
    tools = get_reflect_tools(
        directive_rules=["Never mention competitor names."], answer_as_document=answer_as_document
    )
    return next(t for t in tools if t["function"]["name"] == "done")


@pytest.mark.parametrize("raw", [None, "", "  ", "false", "0", "FALSE"])
def test_parse_off(raw):
    assert parse_enabled(raw) is False


@pytest.mark.parametrize("raw", ["true", "1", "True", " TRUE "])
def test_parse_on(raw):
    assert parse_enabled(raw) is True


@pytest.mark.parametrize("raw", ["yes", "on", "2", "ture"])
def test_parse_rejects_a_typo(raw):
    with pytest.raises(ValueError, match=ENV_PROMPT_TONE):
        parse_enabled(raw)


def test_off_is_byte_for_byte_upstream(tone):
    tone(None)
    assert build_system_prompt_for_tools(
        bank_profile=_BANK, has_mental_models=True, include_observations=True, directives=_DIRECTIVES
    ) == build_system_prompt_for_tools.__wrapped__(
        bank_profile=_BANK, has_mental_models=True, include_observations=True, directives=_DIRECTIVES
    )
    assert build_final_system_prompt(directives=_DIRECTIVES) == build_final_system_prompt.__wrapped__(
        directives=_DIRECTIVES
    )
    assert build_directives_section(_DIRECTIVES) == build_directives_section.__wrapped__(_DIRECTIVES)
    text = _all_prompt_text()
    assert prompt_tone.rewrite_text(text) is text


def test_every_rewrite_still_matches_upstream_text(tone):
    """A rewording upstream turns its rewrite into a silent no-op; this is where it shows."""
    tone(None)
    off = _all_prompt_text()
    missing = [old for old, _ in REWRITES if old not in off]
    assert not missing, f"upstream no longer contains: {missing}"


def test_on_applies_every_rewrite(tone):
    tone("true")
    on = _all_prompt_text()
    for old, new in REWRITES:
        assert old not in on
        if new:
            assert new in on
    for marker in ("will be REJECTED", "NON-CONVERSATIONAL", "⚠️", "MUST follow in ALL"):
        assert marker not in on


def test_on_keeps_the_directives_themselves(tone):
    tone("true")
    for prompt in (
        build_system_prompt_for_tools(
            bank_profile=_BANK, has_mental_models=False, include_observations=False, directives=_DIRECTIVES
        ),
        build_final_system_prompt(directives=_DIRECTIVES),
    ):
        assert "## DIRECTIVES (MANDATORY)" in prompt
        assert "## REMINDER: MANDATORY DIRECTIVES" in prompt
        assert prompt.count("Never mention competitor names.") == 2


def test_done_tool_off_is_upstream(tone):
    tone(None)
    params = _done_tool()["function"]["parameters"]
    assert "directive_compliance" in params["properties"]
    assert "directive_compliance" in params["required"]


def test_done_tool_on_drops_the_unread_compliance_field(tone):
    tone("true")
    function = _done_tool()["function"]
    params = function["parameters"]
    assert "directive_compliance" not in params["properties"]
    assert "directive_compliance" not in params["required"]
    assert "answer" in params["required"]
    assert "REJECTED" not in function["description"]
    assert "Never mention competitor names." in params["properties"]["answer"]["description"]
    for old, new in DONE_TOOL_REWRITES:
        assert old not in repr(function)
        assert new in function["description"] or new in params["properties"]["answer"]["description"]


def test_done_tool_on_document_variant(tone):
    tone("true")
    params = _done_tool(answer_as_document=True)["function"]["parameters"]
    assert "document" in params["required"]
    assert "directive_compliance" not in params["required"]
    assert "directive_compliance" not in params["properties"]


def test_done_tool_on_does_not_mutate_upstream_cache(tone):
    tone("true")
    _done_tool()
    tone(None)
    assert "directive_compliance" in _done_tool()["function"]["parameters"]["properties"]


@pytest.mark.parametrize(
    "relpath, count",
    [
        ("reflect/prompts.py", 4),
        ("reflect/tools_schema.py", 2),
        ("retain/fact_extraction.py", 2),
    ],
)
def test_upstream_files_keep_the_hooks(relpath, count):
    assert (_ENGINE / relpath).read_text().count("# fork: prompt-tone") == count


def test_config_keeps_the_env_hook():
    source = (_ENGINE.parent / "config.py").read_text()
    assert "# fork: prompt-tone" in source
    assert "def prompt_tone_env()" in source

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
    """Set the switch for one test (monkeypatch restores the environment afterwards)."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(ENV_PROMPT_TONE, raising=False)
        else:
            monkeypatch.setenv(ENV_PROMPT_TONE, value)

    return _set


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


def _prompts() -> dict[str, str]:
    """Every prompt the rewrites target, built through the real (hooked) builders."""
    return {
        "agent": build_system_prompt_for_tools(
            bank_profile=_BANK, has_mental_models=True, include_observations=True, directives=_DIRECTIVES
        ),
        "final": build_final_system_prompt(directives=_DIRECTIVES),
        "concise": _build_extraction_prompt_and_schema(_retain_config("concise")).system_prompt,
        "verbose": _build_extraction_prompt_and_schema(_retain_config("verbose")).system_prompt,
    }


_REFLECT_BOTH = ("agent", "final")
#: Which prompt each rewrite targets, by its position in REWRITES. Each target must hold the key once.
_TARGETS: tuple[tuple[str, ...], ...] = (
    _REFLECT_BOTH,  # directives section header line
    _REFLECT_BOTH,  # directives section closing lines
    _REFLECT_BOTH,  # reminder header line
    _REFLECT_BOTH,  # reminder REJECTED lines
    _REFLECT_BOTH,  # grounding rule
    ("agent",),  # non-conversational, agent prompt
    ("final",),  # non-conversational, final prompt
    ("concise",),  # SELECTIVITY
    ("verbose",),  # COREFERENCE
    ("verbose",),  # FACT_KIND
    ("verbose",),  # ⚠️ MUST set fact_kind
    ("verbose",),  # TEMPORAL HANDLING
    ("verbose",),  # ⚠️ IMPORTANT Event Date
    ("verbose",),  # ⚠️ CRITICAL assistant facts
)


def test_every_rewrite_has_a_target():
    assert len(_TARGETS) == len(REWRITES)


def _done_tool_with(rules: list[str], **kwargs) -> dict:
    tools = get_reflect_tools(directive_rules=rules, **kwargs)
    return next(t for t in tools if t["function"]["name"] == "done")


def _done_tool(answer_as_document: bool = False) -> dict:
    return _done_tool_with(["Never mention competitor names."], answer_as_document=answer_as_document)


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
    assert get_reflect_tools(directive_rules=["x"]) == get_reflect_tools.__wrapped__(directive_rules=["x"])
    for text in _prompts().values():
        assert prompt_tone.rewrite_text(text) is text


@pytest.mark.parametrize("index", range(len(REWRITES)))
def test_each_rewrite_still_matches_its_upstream_prompt(tone, index):
    """A rewording upstream turns its rewrite into a silent no-op; this is where it shows.

    Checked per prompt, so rewording one of two prompts that share a sentence still fails.
    """
    tone(None)
    prompts = _prompts()
    old, _ = REWRITES[index]
    for name in _TARGETS[index]:
        assert prompts[name].count(old) == 1, f"{name} prompt no longer contains {old!r}"


@pytest.mark.parametrize("index", range(len(REWRITES)))
def test_each_rewrite_applies_in_its_prompt(tone, index):
    tone("true")
    prompts = _prompts()
    old, new = REWRITES[index]
    for name in _TARGETS[index]:
        assert old not in prompts[name]
        if new:
            assert prompts[name].count(new) == 1


def test_on_leaves_no_shouting_markers(tone):
    tone("true")
    on = "\n".join(_prompts().values())
    for marker in ("will be REJECTED", "NON-CONVERSATIONAL", "⚠️", "MUST follow in ALL"):
        assert marker not in on


@pytest.mark.parametrize("old", [old for old, _ in REWRITES if "\n" not in old.strip("\n")])
def test_on_leaves_caller_text_alone(tone, old):
    """A directive, mission or custom retain instruction that quotes an upstream sentence keeps it."""
    tone("true")
    quoted = f"Quote this literally: {old.strip()}"
    directives = [{"name": "Quote", "content": quoted}]
    agent = build_system_prompt_for_tools(
        bank_profile={"name": "TestBank", "mission": quoted},
        has_mental_models=False,
        include_observations=False,
        directives=directives,
    )
    assert agent.count(quoted) >= 3  # mission, directives section, reminder
    assert quoted in build_final_system_prompt(mission=quoted, directives=directives)
    config = _retain_config("custom")
    config.retain_custom_instructions = quoted
    assert quoted in _build_extraction_prompt_and_schema(config).system_prompt
    done = _done_tool_with([quoted])["function"]["parameters"]["properties"]["answer"]["description"]
    assert quoted in done


@pytest.mark.parametrize(
    "mission",
    [
        "  Quote literally: {sentence}  ",
        'Answer as JSON {{"note": "{sentence}"}}',
        'Answer as JSON {{{{"note": "{sentence}"}}}}',
    ],
)
def test_on_leaves_a_rendered_mission_alone(tone, mission):
    """The builders strip the mission and escape its braces; the quoted sentence still survives."""
    tone("true")
    sentence = REWRITES[4][0]  # the grounding rule, present in both reflect prompts
    mission = mission.format(sentence=sentence)
    agent = build_system_prompt_for_tools(
        bank_profile={"name": "TestBank", "mission": mission}, has_mental_models=False, include_observations=False
    )
    final = build_final_system_prompt(mission=mission)
    for prompt in (agent, final):
        assert prompt.count(sentence) >= 1  # the mission's copy; upstream's own copy is rewritten
        assert REWRITES[4][1] in prompt


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


@pytest.mark.parametrize("answer_ready", [False, True])
@pytest.mark.parametrize("language", [None, "French"])
@pytest.mark.parametrize("answer_as_document", [False, True])
def test_done_tool_on_keeps_the_directives_in_every_variant(tone, answer_as_document, language, answer_ready):
    tone("true")
    params = _done_tool_with(
        ["Never mention competitor names."],
        answer_as_document=answer_as_document,
        llm_output_language=language,
        answer_ready=answer_ready,
    )["function"]["parameters"]
    field = "document" if answer_as_document else "answer"
    assert field in params["required"]
    assert "directive_compliance" not in params["required"]
    assert "directive_compliance" not in params["properties"]
    assert "Never mention competitor names." in params["properties"][field]["description"]


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


def test_config_keeps_the_env_hooks():
    source = (_ENGINE.parent / "config.py").read_text()
    assert source.count("# fork: prompt-tone") == 3
    assert "def prompt_tone_env()" in source
    assert "    validate_prompt_tone_env()  # fork: prompt-tone" in source


@pytest.mark.parametrize("value, enabled", [("true", True), ("false", False)])
def test_dotenv_loaded_after_import_takes_effect(tmp_path, monkeypatch, value, enabled):
    """The entry points import everything before loading .env; the switch must follow the .env."""
    from hindsight_api.config import load_dotenv_for_entrypoint

    monkeypatch.setenv(ENV_PROMPT_TONE, "")
    (tmp_path / ".env").write_text(f"{ENV_PROMPT_TONE}={value}\n")
    monkeypatch.chdir(tmp_path)
    load_dotenv_for_entrypoint()
    assert prompt_tone.prompt_tone_enabled() is enabled


def test_dotenv_typo_fails_the_entry_point(tmp_path, monkeypatch):
    from hindsight_api.config import load_dotenv_for_entrypoint

    monkeypatch.setenv(ENV_PROMPT_TONE, "")
    (tmp_path / ".env").write_text(f"{ENV_PROMPT_TONE}=ture\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=ENV_PROMPT_TONE):
        load_dotenv_for_entrypoint()


def test_inherited_typo_overridden_by_dotenv_does_not_block_startup(tmp_path, monkeypatch):
    """Hooks run at import, before the .env; only the value in effect after it is validated."""
    from hindsight_api.config import load_dotenv_for_entrypoint

    monkeypatch.setenv(ENV_PROMPT_TONE, "ture")
    assert build_final_system_prompt() == build_final_system_prompt.__wrapped__()  # no raise, off
    (tmp_path / ".env").write_text(f"{ENV_PROMPT_TONE}=false\n")
    monkeypatch.chdir(tmp_path)
    load_dotenv_for_entrypoint()
    assert prompt_tone.prompt_tone_enabled() is False

# fork: prompt-tone
"""Plain-voice rewrites of a few upstream prompt instructions (``FORK-PROMPT-TONE.md``).

Several reflect and retain instructions are written at full volume (``CRITICAL``, ``MUST``,
``NEVER``, ``⚠️``) and the reflect directive block threatens the model with a rejection that
no code performs. Current models follow instructions closely and over-apply that register.
This patch rewrites those specific sentences at normal volume, keeping every reason, and
drops the ``directive_compliance`` field of the directive-aware ``done`` tool, which nothing
reads.

The rewrites are exact-string replacements applied to what the upstream builders return, so
upstream's text stays the source and each replacement either matches it word for word or
does nothing. ``tests/test_fork_prompt_tone.py`` builds the real prompts and fails when a
replacement stops matching, which is how a rewording upstream is noticed after a rebase.

Inert by default: with ``HINDSIGHT_API_PROMPT_TONE`` unset, every hook returns upstream's own
object unchanged.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable
from typing import Any, TypeVar

from ..config import ENV_PROMPT_TONE, prompt_tone_env

_F = TypeVar("_F", bound=Callable[..., Any])

#: Upstream sentence -> plain-voice replacement. Each key is matched verbatim.
REWRITES: tuple[tuple[str, str], ...] = (
    # reflect: directives section (build_directives_section)
    (
        "These are hard rules you MUST follow in ALL responses:",
        "Follow these rules in every response:",
    ),
    (
        "NEVER violate these directives, even if other context suggests otherwise.\n"
        "IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.",
        "They take precedence over any other context. Apply them without commenting on them in the answer.",
    ),
    # reflect: directives reminder (build_directives_reminder). Nothing rejects an answer, and
    # the no-commentary rule is already stated once in the section above.
    (
        "Before responding, ensure your answer complies with ALL of these directives:",
        "Before responding, check your answer against these directives:",
    ),
    (
        "\n\nYour response will be REJECTED if it violates any directive above.\n"
        "Do NOT include any commentary about how you handled directives - just follow them.",
        "",
    ),
    # reflect: grounding rule opening both the agent and the final-synthesis system prompts
    (
        "CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.",
        "Use only information from retrieved tool results; do not make up names, people, events, or entities.",
    ),
    # reflect: non-conversational rule, agent prompt
    (
        "- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, "
        "or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.",
        "- The user cannot reply to this answer, so make it complete and self-contained: no follow-up questions, "
        "offers of further help, or suggested next steps.",
    ),
    # reflect: non-conversational rule, final-synthesis prompt
    (
        "CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer to search again, "
        'suggest alternatives, or end with anything like "Would you like me to..." or "Let me know if...". '
        "The user cannot reply. Your answer must be complete and self-contained.",
        "The user cannot reply to this answer, so make it complete and self-contained: do not ask follow-up "
        'questions, offer to search again, suggest alternatives, or close with lines like "Would you like me '
        'to..." or "Let me know if...".',
    ),
    # retain: fact-extraction headings and warnings (concise and verbose modes)
    ("SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)\n", "SELECTIVITY\n"),
    ("COREFERENCE RESOLUTION (CRITICAL)\n", "COREFERENCE RESOLUTION\n"),
    ("FACT_KIND CLASSIFICATION (CRITICAL FOR TEMPORAL HANDLING)\n", "FACT_KIND CLASSIFICATION\n"),
    (
        "⚠️ MUST set fact_kind correctly - this determines whether occurred_start/end are set!\n",
        "fact_kind decides whether occurred_start/end are set.\n",
    ),
    (
        "TEMPORAL HANDLING (CRITICAL - USE EVENT DATE AS REFERENCE)\n",
        "TEMPORAL HANDLING (EVENT DATE IS THE REFERENCE)\n",
    ),
    (
        '⚠️ IMPORTANT: Use the "Event Date" provided in the input as your reference point!\n',
        'Use the "Event Date" provided in the input as your reference point.\n',
    ),
    (
        "  ⚠️ CRITICAL for assistant facts: ALWAYS capture the user's request/question in the fact!\n",
        "  For assistant facts, capture the user's request/question in the fact.\n",
    ),
)

#: Directive-aware ``done`` tool (reflect/tools_schema.py): description and answer field.
DONE_TOOL_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "Signal completion with your final answer. IMPORTANT: You must confirm directive compliance before "
        "submitting. Your answer will be REJECTED if it violates any directive.",
        "Signal completion with your final answer. The answer must comply with the directives listed in the "
        "answer field.",
    ),
    (
        "MANDATORY: Your answer MUST comply with ALL directives:\n",
        "The answer must comply with these directives:\n",
    ),
)

_COMPLIANCE_FIELD = "directive_compliance"


def parse_enabled(raw: str | None) -> bool:
    """The switch from the raw env value. Unset or empty is off; a typo raises ``ValueError``."""
    if raw is None or not raw.strip():
        return False
    value = raw.strip().lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"{ENV_PROMPT_TONE} must be true/false/1/0 (got {raw!r})")


@functools.cache
def prompt_tone_enabled() -> bool:
    """Read once per process. Tests call ``prompt_tone_enabled.cache_clear()``."""
    return parse_enabled(prompt_tone_env())


def rewrite_text(text: str, rewrites: tuple[tuple[str, str], ...] = REWRITES) -> str:
    """Apply each rewrite that matches. With the switch off, ``text`` itself is returned."""
    if not prompt_tone_enabled():
        return text
    for old, new in rewrites:
        text = text.replace(old, new)
    return text


def soften_prompt(func: _F) -> _F:
    """Decorate a prompt builder that returns a ``str``; its result goes through ``rewrite_text``."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return rewrite_text(func(*args, **kwargs))

    return wrapper  # type: ignore[return-value]


def soften_done_tool(func: _F) -> _F:
    """Decorate the directive-aware ``done`` tool builder.

    With the switch on, the unread ``directive_compliance`` field goes away and the description
    and answer field are rewritten. The directives themselves stay listed on the answer field.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool = func(*args, **kwargs)
        if not prompt_tone_enabled():
            return tool
        tool = copy.deepcopy(tool)
        function = tool["function"]
        function["description"] = rewrite_text(function["description"], DONE_TOOL_REWRITES)
        params = function["parameters"]
        answer = params["properties"].get("answer")
        if answer is not None:
            answer["description"] = rewrite_text(answer["description"], DONE_TOOL_REWRITES)
        params["properties"].pop(_COMPLIANCE_FIELD, None)
        params["required"] = [name for name in params.get("required", []) if name != _COMPLIANCE_FIELD]
        return tool

    return wrapper  # type: ignore[return-value]

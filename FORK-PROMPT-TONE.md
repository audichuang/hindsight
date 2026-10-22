# Fork: prompt tone (plain-voice rewrites of a few upstream instructions)

Some reflect and retain instructions are written at full volume: `CRITICAL`, `MUST`, `NEVER`, `⚠️`. The reflect directive block also tells the model its answer "will be REJECTED", but no code rejects anything, and the directive-aware `done` tool requires a `directive_compliance` field that no code reads. Current models follow instructions closely, and a prompt written at that volume gets over-applied. This patch rewrites those specific sentences in a normal voice, keeping every reason. With the switch on, it also drops the unread field.

It is off unless `HINDSIGHT_API_PROMPT_TONE` is set. With the variable unset, every hook returns upstream's own object and behaviour is byte-for-byte upstream.

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `HINDSIGHT_API_PROMPT_TONE` | `true`/`1` turns the rewrites on server-wide (every bank, every LLM lane). `false`/`0`, empty or unset leaves them off. Any other value raises `ValueError`. `load_dotenv_for_entrypoint` validates it right after applying the `.env`, so a typo fails the API server, the worker and the admin CLI at startup. The switch is read on every prompt build (one dict lookup), never cached, so a `.env` loaded after the modules were imported still takes effect. | unset |

Turn it on only after comparing retain and reflect output with it off and on, on the lanes the deployment runs (see "Rollout").

## What changes

`engine/prompt_tone.py` holds two tables of exact upstream strings and their replacements.

- `REWRITES` applies to the agent system prompt, the final-synthesis system prompt, the directives section and the fact-extraction template. In retain, concise mode matches only the `SELECTIVITY` heading and verbose mode the six other retain keys; the custom and verbatim templates contain none of them.
  - Directive section and reminder: rewritten without `MUST`/`NEVER`/`IMPORTANT`. The reminder loses the "REJECTED" threat. It also loses its no-commentary line, which the section above already states.
  - The grounding rule ("use only information from retrieved tool results") that opens both reflect system prompts.
  - Both non-conversational rules. The reason (the user cannot reply) stays and the volume goes.
  - Fact-extraction headings and warnings: `SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)`, `(CRITICAL)` suffixes, `⚠️ MUST` / `⚠️ IMPORTANT` / `⚠️ CRITICAL` lines. The verbose mode's `NEVER SUMMARIZE` / `VERBOSITY REQUIREMENT` are that mode's specification and stay.
- `DONE_TOOL_REWRITES` applies to the directive-aware `done` tool. The description and the answer field drop `IMPORTANT`/`MANDATORY` and the rejection threat, and `directive_compliance` is removed from `properties` and `required`. The directives stay listed on the field the model fills in: the answer field, or in document mode (which has no answer field, so upstream listed them only in `directive_compliance`) the document field.

The headings `## DIRECTIVES (MANDATORY)` and `## REMINDER: MANDATORY DIRECTIVES` are left alone. Upstream tests and prose refer to them by name, and they label a section rather than shout an instruction.

## Why this design

- **Exact-string replacement, not copies of the prompts.** Upstream's text stays the source. A rewrite either matches upstream's sentence word for word or does nothing, so the patch can never ship a stale copy of a prompt that upstream has since improved. `test_each_rewrite_still_matches_its_upstream_prompt` builds each real prompt with the switch off and fails when a key is no longer in the prompt it targets (each target is checked on its own, so rewording one of two prompts that share a sentence still fails). That is how an upstream rewording shows up after a rebase.
- **Caller text is never rewritten.** A directive, bank mission or context that quotes one of these sentences keeps it. On the retain side the rewrite runs on the template before the custom instructions are filled in. The reflect builders interpolate caller values themselves, so `@soften_prompt` collects every string in the builder's arguments and masks those that contain a rewrite key while the rewrites run. The done tool masks the directive rules the same way. `test_on_leaves_caller_text_alone` passes every key through a directive, a mission and custom instructions.
- **Decorators and one wrapped value.** The reflect builders get `@soften_prompt`. `get_reflect_tools` gets `@soften_reflect_tools`, which deep-copies the done tool before editing, so upstream's object is never mutated. It sits on the public function rather than on `_build_done_tool_with_directives` because the document, `answer_ready` and output-language variants are derived after that builder returns. Fact extraction gets one `base_prompt = rewrite_text(base_prompt)` right before the template is filled. No function bodies change apart from that line and the validation call in `load_dotenv_for_entrypoint`.
- **Applied where the prompt is assembled, so the prompt preview matches.** `prompt_preview.py` calls the same builders. With the switch on, the preview shows the text the model receives.
- **Caches follow the text.** The Gemini context cache fingerprint hashes the system instruction itself (`gemini_cache.fingerprint`), and Anthropic/OpenAI prefix caches are byte-keyed. Turning the switch on starts fresh caches. It never serves a stale one.
- **Server-wide, not per bank.** It is a property of the prompt text the deployment ships, not a bank behaviour, so it is a static switch like the other fork patches.

## Upstream files edited directly (deliberate)

These edits are not hooks. They are plain text changes to upstream files and will conflict when upstream touches the same lines. They were kept here on the deployment owner's decision. They change no runtime behaviour, so they are not behind the switch.

| File | Change |
|---|---|
| `CLAUDE.md` | The `/code-review` rule was stated twice, the second time as `MANDATORY:`. It is now one sentence. |
| `skills/hindsight-cloud/SKILL.md`, `skills/hindsight-local/SKILL.md`, `skills/hindsight-self-hosted/SKILL.md` | `## IMPORTANT: When to …` headings become `## When to …`. `**Always recall** before:` becomes `Recall before:` (the same list then repeats it as a best practice). |
| `hindsight-docs/static/get-skill` | The same two edits in the copies of those skills that the installer embeds (local and cloud modes). |

On a rebase conflict in any of them, take upstream's side and re-apply the edit only if upstream's text still has the same wording. If upstream has rewritten the passage, drop the edit.

## Not done

- **Anthropic forced `tool_choice` for strict structured output.** `AnthropicLLM` sends `tool_choice={"type": "tool", ...}` when `strict_schema` is on, and current Claude models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject that with a 400. The reference deployment never turns strict schema on (`HINDSIGHT_API_LLM_STRICT_SCHEMA` unset), so that path is unreachable here and the official setting covers it. It belongs upstream as a defect report, not in the fork.
- **`CLAIMS_SYSTEM_PROMPT`** (reflect split-synthesis evidence extraction) has the same "MUST ONLY … NEVER make up" sentence. It is a module constant passed straight to the LLM call, with no builder to decorate. Rewriting it would need a hook at the call site, and that was not worth a new anchor.

## Hooks

Each line is marked `# fork: prompt-tone`.

| # | File | Hook |
|---|---|---|
| T1 | `hindsight-api-slim/hindsight_api/config.py` | `ENV_PROMPT_TONE` and the accessor `prompt_tone_env()` at the end of the file, after the codex-output-cap accessor; and in `load_dotenv_for_entrypoint`, after `clear_config_cache()`, a local import of `validate_prompt_tone_env` and its call |
| T2 | `hindsight-api-slim/hindsight_api/engine/reflect/prompts.py` | `from ..prompt_tone import soften_prompt`, first line of the relative-import block (ruff's isort order, before `from ..prompt_utils import ...`) |
| T3 | same file | `@soften_prompt` on `build_directives_section` |
| T4 | same file | `@soften_prompt` on `build_system_prompt_for_tools` |
| T5 | same file | `@soften_prompt` on `build_final_system_prompt` |
| T6 | `hindsight-api-slim/hindsight_api/engine/reflect/tools_schema.py` | `from ..prompt_tone import soften_reflect_tools` before `from .prompts import NO_IDS_IN_TEXT_RULE`, and `@soften_reflect_tools` on `get_reflect_tools` |
| T7 | `hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py`, `_build_extraction_prompt_and_schema` | a local import of `rewrite_text` and `base_prompt = rewrite_text(base_prompt)` right before `prompt = base_prompt.format(...)` |

`build_directives_reminder` is only ever called inside the two decorated system-prompt builders, so it needs no hook of its own.

Fork-owned files: `hindsight-api-slim/hindsight_api/engine/prompt_tone.py`, `hindsight-api-slim/tests/test_fork_prompt_tone.py` (switch parsing and `.env` timing, byte-for-byte-off, every rewrite matching its own prompt, caller text left alone, all eight done-tool variants, hook existence), and this file.

## Maintaining the patch

After each rebase, run `tests/test_fork_prompt_tone.py`. When `test_each_rewrite_still_matches_its_upstream_prompt` names a missing sentence, read upstream's new wording. If upstream has already toned it down, delete the entry. Otherwise update the key to the new wording. `test_upstream_files_keep_the_hooks` and `test_config_keeps_the_env_hooks` fail if a hook is lost in a conflict resolution.

Drop an entry when upstream's own text no longer shouts. Drop the whole patch when none are left and upstream removes `directive_compliance`.

## Rollout

The switch changes what every lane's model reads, and there is no deterministic test for that. Before enabling it on the reference deployment:

1. Retain: extract the same documents with the switch off and on, on the retain lane, and compare fact counts, `fact_kind`/temporal fields and assistant-fact attribution.
2. Reflect: run the same questions, including a bank with directives, and compare directive compliance and grounding.
3. Enable the switch in the deployment's env, Deploy, and record the comparison in the deployment record.

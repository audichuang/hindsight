# Fork: a prompt-cacheable consolidation schema

Consolidation sends one structured-output call per batch. Upstream builds that call's response model per batch (`_build_response_model`): `source_fact_ids` is pinned to the batch's fact ids with an `enum` (#5273), and `creates` gets `maxItems` set to the scope's remaining observation slots. Providers that send the schema as text appended to the system prompt therefore send a different prefix on every batch, and their prompt cache misses on the whole prompt.

For those providers this patch keeps upstream's per-batch model for parsing and validation, but renders its schema without the two per-batch values: the fact-id `enum` and `creates.maxItems`. Everything batch-invariant (for example `source_fact_ids.minItems: 1`) stays. Providers that enforce the schema are unchanged.

There are no environment variables. `FORK.md` asks feature patches to be inert by default; this is a defect fix instead (reproduced on the deployed release, below), so the corrected behaviour is the default, as with the fork's other bug fixes.

## Why this design

Measured on the reference deployment (Codex backend, `gpt-6.1-sol`, strict schema off), before the patch:

| Operation | Cache hit (cached / input tokens) |
|---|---|
| refresh_mental_model | 56.7% |
| retain | 34.0% |
| consolidation | 0.9% |

The consolidation system prompt is byte-identical within a run, its calls are sequential (median gap about 5 s) and share one `prompt_cache_key` / `session-id`. The stored messages hide the cause: they are recorded before the provider appends the schema. The Codex provider puts `system + "You must respond with valid JSON matching this schema: …"` into `instructions`, and the `enum` makes that text differ on every batch. Controlled pairs against the Codex backend (second call's `cached_tokens`):

| Pair | Cached |
|---|---|
| non-strict, same fact ids (identical `instructions`) | 3712 |
| non-strict, different fact ids (only the `enum` differs) | 0 |
| strict forced tool, different fact ids | 0 |
| strict forced tool, same fact ids | 3328 |
| this patch, six batches with different ids and caps | 3456 on every batch |

A difference anywhere in `instructions` costs the whole cached prefix, not only what follows it.

Providers render the schema with `response_format.model_json_schema(...)` and parse with `response_format.model_validate(...)`. The patch redirects only the first:

- The fact-id `enum` is schema-only upstream as well (`json_schema_extra`; validation is `list[str]`). Where the schema is prompt text the model is not held to it, and an action citing no batch fact is still discarded and counted (`unresolved_actions`, logged as `cited no fact id from their batch`).
- The creates cap is real validation upstream (`max_length`), and stays so: an over-cap response fails the batch exactly as before, so a delete is never applied without the replacement create it came with. Only the `maxItems` text leaves the prompt; the capacity note in the user message (when slots are scarce) and post-call truncation are unchanged.

Where the schema is grammar-enforced the `enum` is what stops a small model from emitting a miscopied id (#5273), so those calls keep upstream's schema.

Upstream's own comment at the call site already intends this ("response_schema … kept OUT of the cached prefix"); that holds for Gemini, where the schema is a separate request field, but not for the providers below.

## Which calls change

`cacheable_schema.cacheable_response_model(response_model, llm_config, config)` applies the redirect when `schema_is_prompt_text` holds: strict schema is off for consolidation and `llm_config.provider` is one whose non-strict path appends the schema to the system prompt (or Codex `instructions`) and enforces at most `json_object`:

| Provider | Strict off | Strict on |
|---|---|---|
| `openai-codex`, `anthropic`, `xai-oauth`, `openai-responses` | per-batch values dropped | upstream (forced tool / `json_schema`) |
| `OpenAICompatibleLLM` and the providers built on it: `openai`, `groq`, `ollama-cloud`, `lmstudio`, `minimax`, `deepseek`, `volcano`, `openrouter`, `requesty`, `zai`, `opencode-go`, `atlas`, `meta`, `fireworks`, `nous`, `llamacpp` | per-batch values dropped | upstream |
| native `ollama` (`format=` schema, enforced), Gemini / Vertex AI (native `response_schema`), Bedrock, LiteLLM, GitHub Copilot | upstream | upstream |
| `claude-code`, `cursor` | upstream | upstream |
| multi-LLM chains | upstream | upstream |

- `claude-code` and `cursor` append the schema after the per-batch user content, so it is not in the cached prefix and there is nothing to gain.
- A multi-LLM chain forwards `provider` to its first member, but a failover or round-robin member may enforce the schema. A chain is recognised by its `members` attribute and keeps upstream's model.

## Known limitations

- Strict mode on Codex still misses: the per-batch `enum` is inside the forced tool definition, which is part of the prefix. Keeping enforcement was chosen over the cache there.
- The first call of each consolidation run cannot hit: `prompt_cache_key` is derived from the run's trace id. The system prompt is shared across runs and banks, so a run-independent key could hit it too; not done here.
- Expected ceiling is the system prompt plus schema (about 3.5K of about 12.5K input tokens per call); the facts and observations differ per batch.

## Verifying on a deployment

- `llm_requests` for `scope='consolidation'`: share of rows with `cached_tokens > 0` and `sum(cached_tokens) / sum(input_tokens)` before and after.
- `cited no fact id from their batch` warnings should stay rare (the reference deployment logged 2 in about 66 hours with the `enum`).

## Maintaining the patch (rebase and cherry-pick)

Files owned by the fork:

- `hindsight-api-slim/hindsight_api/engine/consolidation/cacheable_schema.py`
- `hindsight-api-slim/tests/test_fork_consolidation_cache.py`
- `FORK-CONSOLIDATION-CACHE.md` (this file)

Upstream hook, three lines marked `# fork: consolidation-cache` (`grep -rn "fork: consolidation-cache" hindsight-api-slim/hindsight_api`):

- `engine/consolidation/consolidator.py`, `_consolidate_batch_with_llm`, directly after `response_model = _build_response_model(...)`: a comment, a local import and `response_model = cacheable_response_model(response_model, llm_config, config)`.

`test_consolidator_keeps_the_cacheable_schema_hook` fails if the hook is lost; `test_codex_instructions_are_byte_identical_across_batches` fails if the Codex `instructions` differ between two batches; `test_an_over_cap_response_still_fails_the_batch` fails if the cap stops being validated.

After each rebase, re-check what the patch depends on:

- `_build_response_model`: `_drop_per_batch_keywords` removes exactly `creates.maxItems` and `source_fact_ids.items.enum`. A new per-batch value there would break the cache again (`test_prompt_text_provider_gets_one_schema_for_every_batch` fails); a renamed field would silently stop being stripped (`test_only_the_per_batch_keywords_leave_upstreams_schema` fails).
- Each listed provider still renders the schema with `model_json_schema` / `provider_json_schema` and parses with `model_validate`, and its non-strict path still appends the schema to the system prompt. A provider that starts sending it as a grammar-enforced field must leave the set in `cacheable_schema.py`; a new OpenAI-compatible provider should join it.
- `MultiLLMProvider` still exposes `members`.

# Fork: LLM cost accounting

This fork-only feature turns the per-call token rows in `llm_requests` into USD estimates per call, per model and per bank. It reuses the `genai-prices` package (pydantic, MIT) for price data and arithmetic. It stores no money: cost is computed when a cost endpoint is read, from the stored token counts, the row's timestamp and the provider-reported cost captured at write time.

It is inert unless `HINDSIGHT_API_LLM_COST_ENABLED=true`. With it unset, no route is registered, `llm_info` is written exactly as upstream writes it (no `usage` key), and the control-plane panel mounts nothing and requests nothing (see "Control-plane switch" below).

It depends on five bug-fix commits that carry no fork marker, because each fixes an upstream defect and should drop out when upstream fixes it:

- `fix(anthropic): count cache reads and writes in input_tokens on every path`: `input_tokens` includes cache reads and cache writes on every path (`call`, `call_with_tools`, batch translation), and the tool path records cache reads. This matches the documented `TokenUsage` contract (input includes the cached prefix; cached is a subset). When `cache_creation_input_tokens` is missing, the write count is taken from the TTL breakdown (`cache_creation.ephemeral_5m_input_tokens` plus `ephemeral_1h_input_tokens`) when that is larger, so the folded input always covers the writes the cost hook records. Regression tests: `tests/test_fork_anthropic_fold.py`.
- `fix(codex): record real usage from the Responses stream (#5541)`: real usage from `response.completed` instead of a chars/4 estimate or zeros. The usage block is parsed by a small Pydantic model at the event boundary: extra fields are ignored, a missing count is 0, and a wrong-typed count is treated as absent. The model also reads the cache-write count (`input_tokens_details.cache_write_tokens`, included in `input_tokens`, 0 when missing), but the cost feature does not use that field: the fork hook (H16) receives the raw event and reads the count with `llm_cost.py`'s own tolerant model, so a count the backend did not send is recorded as no breakdown, not as 0. Regression tests: `tests/test_fork_codex_usage.py`; the cost hook's own parsing is tested in `tests/test_llm_cost.py`.
- `fix(reflect): pass cached and thoughts tokens through to TokenUsage (#5487)`: the reflect response keeps `cached_tokens` and `thoughts_tokens`; adapted from akushonkamen/hindsight@470778940 (a contributor's fork; #5487 is open upstream).
- `fix(claude-code): record real usage from ResultMessage`: the Claude Agent SDK's `ResultMessage.usage` (and, on tool rounds that stop before the result, the last `AssistantMessage.usage`) is used instead of the chars/4 estimate, folded the same way as the Anthropic fix. The same TTL-breakdown rule as the Anthropic fix applies to a missing total. Quota errors keep their real counts. The estimate marker is read by the fork hook (H14, H15), whose lines carry the fork marker and are not part of this commit. Regression tests: `tests/test_fork_claude_code_usage.py`.
- `fix(openai-compatible): read provider cache fields`: `cached_tokens` also reads DeepSeek's `prompt_cache_hit_tokens`, `cache_read_input_tokens`, and Kimi's top-level `cached_tokens`, when `prompt_tokens_details.cached_tokens` is not a positive integer. Only real integers count, so a wrong-typed field is ignored. `prompt_tokens` stays the total prompt. Regression tests: `tests/test_fork_openai_compat_usage.py`.

## Why this design

- **Read-time pricing.** genai-prices models price history. Passing each row's `started_at` prices it at the rate that applied then. Price corrections apply retroactively with a package bump, and rows recorded before this patch are priced with no backfill.
- **genai-prices, not a hand-written table or the litellm price JSON.** It covers the models the reference deployment uses (Claude Haiku 5.5 with 5m and 1h cache writes and the >100k tier, GPT-6.1 Sol, Grok 4.7, GLM-5.3).
- **Provider-reported cost first.** For metered rows the cost the provider reports for the call (xAI `cost_in_usd_ticks`, OpenRouter `cost`, including BYOK upstream cost) is the billed amount, and the row is `actual`. genai-prices is used only when the provider reports nothing, and then the row is `estimated`.
- **Every traced row carries a usage marker while the feature is on.** `llm_info.usage` is written with at least `{"v": 2}` on every row, not only on Anthropic rows. That lets the legacy rule tell a folded post-B1 error row from a pre-B1 row that excluded cache reads, and it marks rows whose tokens are chars/4 estimates.
- **Write-time facts arrive as one kwarg.** Fork-only facts (cache-write split, provider cost, zai billing hint, usage source) reach the trace as one `fork_cost_usage` keyword on the provider's success `record_llm_call`. A per-provider adapter in `llm_cost.py` builds it from the raw provider object with tolerant Pydantic models. The adapter returns nothing with the feature off and never raises. Codex parses its usage inside the SSE loop, so its breakdown is kept in a context variable and handed back only for the identical usage object (identity check). A stale value therefore cannot reach another call.
- **No migration.** The new facts (cache-write tokens, provider cost, billing hint, usage source) are kept in `llm_info.usage` (JSONB). `llm_requests` is PostgreSQL-only upstream.
- **Hidden fork endpoints instead of new fields on `/llm-requests/stats`.** Extending the stats models would change OpenAPI and every generated client, and would send `"cost": null` with the feature off. The fork routes use `include_in_schema=False`, so generated files stay byte-identical.
- **Subscription providers are not billed per token.** Their cost is null with reason `subscription_plan`. A `list_price_usd` shows what the same traffic would cost at public API rates, labelled as notional. A provider-reported cost on a plan row is shown separately as `provider_cost_usd`. Each day of the window carries `cost_usd` (metered and local spend) and `plan_list_price_usd` (plan rows only, so a metered-only day shows 0 there).
- **Unknown is null, never 0.** An unmapped provider or unknown model is reported with a reason and excluded from totals. A row whose cost could not be determined has status `unknown`.

## Token semantics (llm_requests)

| column | meaning |
|---|---|
| input_tokens | total prompt: uncached + cache read + cache write |
| cached_tokens | cache-read tokens (subset of input) |
| output_tokens | visible output (Anthropic includes thinking) |
| thoughts_tokens | reasoning reported separately (billed at the output rate) |
| llm_info.usage | the v2 marker (below), written on every row with the feature on |

The `llm_info.usage` object, written only with the feature on:

```json
{"v": 2, "cache_write_tokens": 0, "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0,
 "provider_cost_usd": 0.0012, "billing_hint": "plan", "usage_source": "provider"}
```

- `v` is always present. Version 1 (the earlier Anthropic success-row shape) is read the same way and counts as marked.
- The three cache-write keys are written together, and only when an adapter ran for the row and the provider sent a cache-write field. They are 0 when the provider reported 0. Absent means no breakdown: error rows written by the wrapper, providers without an adapter, and calls whose usage block has no cache-write field (a missing field is never recorded as 0).
- `provider_cost_usd`, `billing_hint` and `usage_source` are written only when known.
- `billing_hint` is `plan` or `metered`, and is written only for `zai` (see Environment variables).
- `usage_source`: `provider` (real counts) or `estimate` (chars/4 estimate, or zeros when a Codex call's breakdown could not be matched). Absent on a marked row means provider counts.
- Cache-write counts come from each provider's own usage fields: Anthropic and claude-code (`cache_creation`, with the 5m and 1h split), Codex and `openai-responses` (`input_tokens_details.cache_write_tokens`), and OpenAI-compatible providers (`prompt_tokens_details.cache_write_tokens` or `cache_creation_input_tokens`, which OpenRouter sends). Upstream records no cache writes, so these reach the trace only through the fork hooks. The ChatGPT backend behind `openai-codex` does send `cache_write_tokens`, always as 0 so far (see Known limitations).
- Cache-write totals are normalised on write and on read: when the 5m count is missing it is the total minus the 1h count, and the total always covers both TTLs, so a negative 5m write cannot be priced. A naive `started_at` is read as UTC. Neither changes what an operator sees in normal operation.
- A Codex call with more than one attempt, or whose attempt ran away, carries the sum over its attempts (`FORK-ATTEMPT-USAGE.md`). Such a row is `estimate` when any attempt was estimated: a runaway attempt counts one output token per delta event and the input of another attempt (or chars/4), and the sum can mix provider and estimated counts. Its cache-write count is the sum over the attempts that reported one, so it can appear on an error row the wrapper wrote.

Pricing a row: `Usage(input_tokens=input, cache_read_tokens=cached, cache_write_tokens=..., output_tokens=output+thoughts)`. The cache-write split (5m and 1h) is passed when the row has one.

Legacy repair (D2): an `anthropic` row with no `llm_info.usage` object at all, `cached_tokens > 0`, and a start before `HINDSIGHT_API_LLM_COST_LEGACY_BEFORE` (when that is set) was written before the B1 fix, whose stored `input_tokens` excluded cache reads. It is priced as `input_tokens + cached_tokens` and counted in `legacy_rows_repaired`. Its cache writes were never stored, so the repaired cost is a lower bound. `claude-code` rows are never repaired, and pre-fix tool-path Anthropic rows recorded no cache tokens at all, so they stay undercounted.

A row from `claude-code` or `openai-codex` with no marker, started before the cutover, is a chars/4 estimate. It is counted in `tokens_estimated_rows` and is never repaired.

## Environment variables

All of them are read through `llm_cost_env()`, the one fork hook in `config.py` (end of file), and parsed and validated in `engine/llm_cost.py`. Invalid values fail with the variable's name. A blank value counts as unset. The API validates at startup, so a typo stops the server from starting instead of failing on the first call. The `hindsight-worker` entry point validates them at startup too (one hook in `worker/main.py`), so it fails the same way the API does. Every process that makes LLM calls needs the same `HINDSIGHT_API_LLM_COST_*` environment: the trace rows are written by whichever process made the call. A process with the feature off, or with settings the hooks cannot use, writes rows without a usage marker, and its Anthropic rows are then repaired a second time by the legacy rule unless `LEGACY_BEFORE` is set (see Known limitations). The provider and trace hooks never raise.

Unknown keys are rejected. Any `HINDSIGHT_API_LLM_COST_*` name that is not in the table below stops startup, and the error names the key but never its value. Keys under the `HINDSIGHT_API_LLM_COST_QUOTA_` prefix are not checked here; they belong to a separate feature with its own validation.

| Variable | Values | Default |
|---|---|---|
| `HINDSIGHT_API_LLM_COST_ENABLED` | `true/false/1/0/yes/no` (case-insensitive) | false |
| `HINDSIGHT_API_LLM_COST_BILLING` | `provider=metered\|plan\|local,...` | plan: openai-codex, xai-oauth, claude-code, github-copilot, cursor, nous, zai; local: ollama, lmstudio, llamacpp, mock, none; everything else metered |
| `HINDSIGHT_API_LLM_COST_PROVIDER_MAP` | `provider=pricing_provider_id,...` | built-in map: anthropic, claude-code, openai, openai-codex, openai-responses (to openai), xai and xai-oauth (to x-ai), zai, openrouter, gemini (to google), groq, deepseek, minimax, fireworks, github-copilot, cursor |
| `HINDSIGHT_API_LLM_COST_MODEL_ALIASES` | JSON `{"provider/model": "pricing_provider/model_ref"}`; each side splits on the first `/` | none |
| `HINDSIGHT_API_LLM_COST_CUSTOM_PRICES` | JSON `{"provider/model": {"input_mtok", "output_mtok", "cache_read_mtok"?, "cache_write_mtok"?, "cache_write_1h_mtok"?}}` (USD per million tokens) | none |
| `HINDSIGHT_API_LLM_COST_LEGACY_BEFORE` | ISO-8601 timestamp with a UTC offset (`Z` allowed); a naive value is rejected | unset |

`LEGACY_BEFORE` is the moment the B1 fix went live, for a deployment that ran B1 with the feature off for a while. Without it, every Anthropic row with no marker is treated as pre-B1, so the folded rows written with the feature off would be repaired a second time. With it, only rows started before that moment are repaired. A deployment that enables the feature in the same deploy as B1 leaves it unset.

Billing for a row is resolved in this order: an explicit `HINDSIGHT_API_LLM_COST_BILLING` entry for the provider, then the row's `billing_hint`, then `DEFAULT_BILLING`, then `metered`. Only the explicit entry is an operator setting; the defaults are not. `zai` is `plan` by default because the default base URL is the Coding Plan endpoint (`/api/coding/paas/v4`). New `zai` rows record a hint from the lane's base URL: `plan` when the path contains `/coding/`, otherwise `metered`. So a lane on the pay-per-token endpoint is metered from its first row. Historical `zai` rows have no hint and are `plan`. To price such history as metered, set `HINDSIGHT_API_LLM_COST_BILLING=zai=metered`; that entry also overrides the hints.

Control-plane switch: `HINDSIGHT_CP_LLM_COST_ENABLED` (`true/1/yes` turns it on, anything else is off; it is read by the control plane's `/api/version` route, not by the API). It must agree with the API's `HINDSIGHT_API_LLM_COST_ENABLED`. The panel and the per-call line read the flag from the features payload the page already loads, so a disabled deployment sends no cost request. If the flag is on but the API is off, the proxies return 404 and the panel hides itself, so a mismatch fails quietly.

Precedence per row for the list price: custom price, then alias, then genai-prices through the provider map. The map targets must exist in the genai-prices snapshot, and so must alias targets (checked at startup). Vertex is not mapped, so its rows report `unmapped_provider` unless `PROVIDER_MAP` maps it. A metered row with a provider-reported cost is billed at that cost whatever the list price is (see the cost status table below).

Example for the reference deployment (values to be confirmed by its operator):

```
HINDSIGHT_API_LLM_COST_ENABLED=true
HINDSIGHT_API_LLM_COST_MODEL_ALIASES={"xai-oauth/grok-4.7-build-fast": "x-ai/grok-4.7"}
# only for a pay-per-token GLM key on the /api/paas/v4 endpoint, for history rows:
# HINDSIGHT_API_LLM_COST_BILLING=zai=metered
```

## Endpoints (not in OpenAPI)

- `GET /v1/default/banks/{bank_id}/llm-requests/cost?period=1d|7d|30d&operation=&start_date=&end_date=&tz=`: per-model totals, per-day totals, metered cost, plan list-price value, unpriced counts, `pricing_source`, `truncated` when the window has more than 200k rows, and `status_counts` for the whole window. Window rules: `period` alone is the last N days up to now; `start_date` alone runs from start to now; `end_date` alone is the N days that end at `end_date`; both bounds are used as given (`end_date` must be after `start_date`).
- Time zone (`tz`): the route takes an IANA zone name, default `UTC`, and `LLMCostStore` and `summarize_costs` bucket each day as a local calendar date in that zone. An unknown name is HTTP 400. `UTC` in any case needs no tz database; other zones need one (see Known limitations). The response has a `tz` field that echoes the name as sent. The control plane sends the browser's zone (`Intl.DateTimeFormat`, falling back to `UTC`). The tz database comes from `tzdata`, which `pandas` pulls in through the `xls` and `xlsx` extras of `markitdown` (`uv.lock`); the API requests those extras, so a normal install has it. Without a tz database, every zone other than `UTC` answers 400.
- Each model entry in `models` adds `status_counts` (calls per cost status), `provider_cost_usd` (the sum over rows that carried one, null when none did), `provider_cost_calls`, and `tokens_estimated_rows`. `rows_without_breakdown` counts only rows from providers that bill cache writes separately and have no write split.
- `GET /v1/default/banks/{bank_id}/llm-requests/cost/rows?ids=...`: per-row cost for up to 500 comma-separated ids (the control plane's span dialog). Each row adds `cost_status`, `cost_source`, `provider_cost_usd`, `tokens_estimated` and `legacy_input_repaired`.

Cost status and source per row:

| billing | condition | cost_usd | cost_status | cost_source |
|---|---|---|---|---|
| metered | provider cost present | provider cost | actual | provider |
| metered | custom price | quote | estimated | custom |
| metered | genai-prices quote | quote | estimated | genai-prices |
| metered | no quote, no provider cost | null | unknown | none |
| plan | any | null | included | the source of `list_price_usd` (genai-prices, custom or provider), or none |
| local | any | 0 | included | none |

`status_counts` (per model and for the window) counts rows in each status. The panel labels `github-copilot` list prices as notional, because Copilot bills premium requests, not tokens.

Both endpoints use the same auth as `/llm-requests/stats`: unknown bank is 404, and tag-scoped callers are refused. On Oracle the summary is empty, because `llm_requests` exists only on PostgreSQL.

Layering: `api/llm_cost_routes.py` only parses parameters and maps errors. `engine/llm_cost_store.py` (`LLMCostStore`) owns the bank checks, the window rules, the SQL and the hand-off to the pricer. The pricing rules are in `engine/llm_cost.py`.

The control-plane proxies are `hindsight-control-plane/src/app/api/banks/[bankId]/llm-requests/cost/route.ts` and `.../cost/rows/route.ts`. Both delegate to `src/lib/fork/llm-cost-proxy.ts`, which uses cost-specific English messages (`Failed to get LLM cost data`). A 404 from the proxy still hides the panel without logging, as a fallback for a flag/API mismatch. With the flag off, the panel is not mounted and no request is made.

## Subscription quota

A read-only snapshot of the plan limits behind the server's LLM lanes: for each rolling window, the used percentage and the reset time, plus the Codex credit balance or the Claude extra-usage budget where the provider exposes one. Plan rows are not billed per token, so the cost panel cannot show how close a plan is to its limit; this section does. The payload describes the server, not a bank.

It is off by default and independent of `HINDSIGHT_API_LLM_COST_ENABLED`. The control plane shows the card inside the cost panel, so it needs the cost switch on as well (see "Control-plane gating").

Environment variables (validated in `engine/llm_quota.py`, read through `llm_cost_env()`):

- `HINDSIGHT_API_LLM_COST_QUOTA_ENABLED`: `true/false/1/0/yes/no` (case-insensitive), default `false`. Any other value stops startup and the error names the variable.
- `HINDSIGHT_API_LLM_COST_QUOTA_CLAUDE_OAUTH_TOKEN`: optional OAuth access token with the `user:profile` scope, used for the Claude usage endpoint. It is held as a secret and is never printed, logged or returned. An unknown `HINDSIGHT_API_LLM_COST_QUOTA_*` key stops startup; the error names the key only.

The cost settings skip the `HINDSIGHT_API_LLM_COST_QUOTA_` prefix (`llm_cost.py`), so the two features do not validate each other's keys.

### Endpoint

`GET /v1/default/banks/{bank_id}/llm-requests/quota` (not in OpenAPI, `include_in_schema=False`). It is registered by `register_llm_cost_routes` when the quota flag is on, and it works with the cost flag off. Auth is the same as the cost routes: an unknown bank is 404, and a tag-scoped caller is refused with 403.

The response is server-wide: `scope` is `"server"`, and every bank gets the same provider list. The bank id is only used for the readability check and echoed back.

```
{ "bank_id", "scope": "server",
  "providers": [ { "provider", "lanes", "status", "reason", "detail", "plan",
                   "windows": [ {"label", "used_percent", "resets_at", "window_seconds", "scope"} ],
                   "credits", "budget", "fetched_at", "stale" } ] }
```

- `status`: `ok`, `unsupported` (no usage source for this provider), `unavailable` (a credential is missing or lacks scope; no request was made, or the provider refused the scope), or `error` (the request or the response failed).
- `reason`: `no_credentials`, `token_scope`, `auth_failed`, `http_error` (`detail` is `HTTP <code>`), `malformed_response`, `timeout`, `provider_error` (`detail` is the fixed phrase `provider reported an error`; the Z.ai `msg` is never returned, because an error envelope can echo the configured key), `not_a_plan_endpoint`, `no_usage_endpoint`, `internal_error`. `reason` is null on `ok`.
- `credits` (Codex) is shown only when the account reports `has_credits: true`, as Hermes does: a balance, and `unlimited` only with credits. A plan account without credits reports `has_credits: false` with a `"0"` balance, which is not shown. `banked_resets` is shown whenever it is positive.
- `used_percent` is clamped to 0 to 100. `budget` values are converted from minor units (cents) to major units.

### Lanes queried

Only the lanes this server's engine is configured with: the default `_llm_config`, `_retain_llm_config`, `_reflect_llm_config`, `_consolidation_llm_config` and `_mental_model_refresh_llm_override`. Labels are `default`, `retain`, `reflect`, `consolidation` and `mental_model_refresh`. Chain members get a 1-based index, such as `retain[2]`. Targets are deduplicated by credential (the internal key is never returned), and one target lists every lane that uses it. Per-bank and per-operation overrides are not walked.

### Credential source per provider

- `openai-codex`: the Codex login the lane already holds (`$CODEX_HOME/auth.json`, default `~/.codex/auth.json`). The quota path calls the same proactive refresh every Codex call makes, and never forces one; the quota deadline cannot cancel it (see Requests). A refresh that has expired gives `auth_failed`. Without a usable login the lane does not start, so there is nothing to query.
- `claude-code`: the quota token variable if set, else the lane's API key when it starts with `sk-ant-oat`. With neither, the status is `unavailable`, reason `no_credentials`, and no request is sent. The CLI's own `claude auth login` store is not read. A setup token is inference-only, so the usage endpoint answers 403 and the status is `unavailable`, reason `token_scope`.
- `zai`: the lane's API key, but only when the lane's base URL path contains `/coding/`. On a pay-per-token endpoint the status is `unsupported`, reason `not_a_plan_endpoint`, and no request is sent.
- `xai-oauth`, `github-copilot`, `cursor`, `nous`: `unsupported`, reason `no_usage_endpoint`, no request.
- Every other provider (metered API keys, local models) is not listed. Their spend is in the cost panel.

### Requests (URLs and header names; no values)

- Codex: `{base}/wham/usage` when the base URL contains `/backend-api`, else `{base}/api/codex/usage`, where `base` is the lane's base URL without a trailing `/codex`. Headers: `Authorization`, `ChatGPT-Account-ID`, `User-Agent`, `Origin`, `originator` (the lane's own headers, without `Content-Type`), and `Accept`.
- Claude: `https://api.anthropic.com/api/oauth/usage`. Headers: `Authorization`, `Accept`, `anthropic-version`, `anthropic-beta`.
- Z.ai: `{scheme}://{host}/api/monitor/usage/quota/limit` (the lane's origin only). Headers: `Authorization`, `Accept`.

Each target has a 10 s total deadline (`QUOTA_FETCH_TIMEOUT_SECONDS`), enforced by `QuotaCache` around the whole fetch: the Codex token refresh, the GET and the body. The shared session's own 10 s timeout is per phase, so on its own a provider that trickles bytes could hold the target's lock, and with it every `/quota` request, indefinitely. The Codex token refresh is shielded from that deadline: the refresh rotates the refresh token, and cancelling it between the provider's answer and the in-memory update would leave the live Codex lane with a refresh token that no longer works. When the deadline fires during a refresh, the refresh finishes in the background, the reading is `timeout`, and the next call uses the new tokens. `MemoryEngine.close()` waits up to 15 s (`DETACHED_REFRESH_DRAIN_SECONDS`) for such refreshes before it closes the loop's HTTP sessions (hook H24), so a shutdown does not cut a rotation whose response is still arriving. Targets are fetched concurrently. HTTP 401 gives `auth_failed`. A 403 whose body mentions `user:profile` gives `unavailable`, reason `token_scope`; any other 403 gives `auth_failed`. A non-2xx status gives `error`, reason `http_error`. A timeout gives `timeout`. A body that is not JSON gives `malformed_response`. Nothing raises into the request. Logs carry the provider and the HTTP status only, never headers, tokens or bodies.

### Cache and backoff

The cache is in-process, one entry per credential, and it lives for the server's lifetime. A success is valid for 60 s. A failure backs off for 60 s, doubling with each consecutive failure up to 900 s. While backing off, the last good reading is returned with `stale: true`; with none, the error is returned. Each credential has its own lock, so concurrent page loads share one fetch.

### Control-plane gating

The card (`LLMQuotaCard`) is mounted by the cost panel and shows only when `features.llm_cost` is true, which comes from `HINDSIGHT_CP_LLM_COST_ENABLED`. The control plane's proxy is `.../llm-requests/quota/route.ts`, and it calls `src/lib/fork/llm-cost-proxy.ts`.

The Refresh button shows a busy label while its request runs. When a refresh brings back the reading already shown (same newest `fetched_at`), the card says the server re-reads each provider at most once a minute. That hint is not shown while any provider is backing off: during backoff the same reading comes back for up to 900 s, either the last good one marked `stale` or, when the provider has never read ok, the failed reading itself (not `ok`, with the `fetched_at` of the failure). A provider the server never reads has no `fetched_at` and does not suppress the hint. Reset countdowns use the card's own en/zh-TW templates, and credit balances are rounded to two decimals and grouped for the locale. The cost panel and the quota card show the proxy's `error` text next to their load-error message (`src/lib/fork/fork-request.ts`), except the proxy's own generic English text (`COST_LOAD_ERROR`, `QUOTA_LOAD_ERROR`, sent when the dataplane gave no reason), which would only repeat that message.

Trade-off: with the cost switch on and the quota switch off, each page view sends one quota request that the API answers with 404. The card then hides itself without logging, and the cost card is unaffected. The alternative is a new field in the API's `features` object, which would change the generated clients.

### Not read, and why

- `RateLimitEvent`: the Claude Agent SDK sends rate-limit messages during `claude-code` calls. They could feed the same windows passively, without a request. This is a possible future source and is not built.

## Known limitations

- Error rows (provider charged, local parsing failed) get the total input, because the Anthropic provider stashes its usage (`stash_response_usage(_usage_from_anthropic_response(response))`, `anthropic_llm.py` in `call()` and `call_with_tools()`) before it parses or validates the output, and the wrapper writes that stash on the error row. They have no cache-write split, so their write portion is priced at the base input rate. A provider that never stashes writes error rows with NULL tokens (`no_usage`). Codex runaway and deadline attempts stash nothing, but their error rows now carry estimated tokens (see `FORK-ATTEMPT-USAGE.md`), so they count as estimated, not as `no_usage`. A Codex attempt that received `response.completed`, `response.incomplete` or `response.failed` before it failed keeps that event's provider usage (A8 there).
- With the feature off after B1, Anthropic rows carry no marker. Set `LEGACY_BEFORE` to the B1 go-live time (see Environment variables), or those rows are repaired a second time.
- The feature must be on, with the same settings, in every process that makes LLM calls. If it is on in the API container only (or in the worker only), the other process writes unmarked rows, and its Anthropic rows are repaired a second time unless `LEGACY_BEFORE` is set. The worker validates the settings at startup, but it cannot know what the API container was given. Check both containers when you change these variables.
- Anthropic batch calls are not traced into `llm_requests` upstream, so they are not costed.
- Embedding and reranker calls are not LLM calls and are not costed.
- History is bounded by `HINDSIGHT_API_LLM_TRACE_RETENTION_DAYS` (upstream default 1 day).
- Plan-provider "list price" is notional. The UI labels it as an API list price, not a billed amount.
- Claude Code tool rounds: when the round ends at a tool call, the result message has not arrived, so `output_tokens` comes from the last assistant message and may be a partial count. The optional live test checks this.
- Claude Code `total_cost_usd` is not recorded. It is the CLI's own client-side estimate, not a provider bill, and a live check on 2026-10-15 showed it pricing `claude-haiku-5-5` cache writes at $6.25/MTok against the official $0.20/MTok (about 30x too high). The list price for Claude Code rows comes from genai-prices like any other plan row.
- OpenRouter BYOK: the `cost` field is only OpenRouter's fee. When `is_byok` is true, the `cost_details.upstream_inference_cost` is added to it.
- xAI cost is read by field presence (`cost_in_usd_ticks`, 1 USD = 10,000,000,000 ticks), not by provider name. Whether xAI OAuth responses carry the field is not verified; if they do not, no provider cost is recorded for them.
- Gemini tool-use prompt tokens (`toolUsePromptTokenCount`) do not apply: Hindsight configures no Gemini built-in tools, and function declarations are already counted in `promptTokenCount`.
- Not modelled (D13): service tiers (flex, priority, fast), Anthropic `usage.iterations`, web-search fees, `inference_geo`. The genai-prices Sonnet 5 price is used as shipped.
- Accuracy checks beyond the hand-calculated unit tests and the dataset guard: `tests/test_llm_cost_live.py` calls the real Anthropic and Codex APIs and checks each trace row against the provider's own usage block. It is skipped unless `HINDSIGHT_COST_LIVE=1` and credentials are present, so CI does not run it; the Claude Code case also needs `HINDSIGHT_COST_LIVE_CLAUDE_CODE=1`, because it spends subscription quota. `tests/test_llm_cost_replay.py` and `scripts/fork/llm_cost_replay.py` price a read-only CSV export of `llm_requests` through the same `summarize_costs`; the script's `--tz` option sets the per-day buckets.
- Day buckets in a non-UTC zone need a tz database. Without one, the route answers 400 for those zones (see Endpoints).
- The lower-bound note in the panel appears when a model of a provider that bills cache writes separately has rows without a cache-write breakdown (recorded before cost tracking, or the provider sent no cache-write field) or rows repaired for the legacy input convention. Their cost is a lower bound.
- Codex cache writes: the ChatGPT backend sends `input_tokens_details.cache_write_tokens`, but a live check on 2026-10-20 (gpt-6.1-sol, two calls with the same 6.8k-token prefix) showed it as 0 on both calls, including the first, after which the second read 6,656 cached tokens. Codex rows therefore record 0 writes, and the notional list price never includes the write premium genai-prices lists for the model. Whether the plan treats those writes as billable cannot be told from the usage block, and the lower-bound note does not cover it, because the backend did report a count.
- Summary pricing runs in a worker thread (`asyncio.to_thread`), so the event loop is no longer blocked for the whole pass. It is still CPU work under the GIL: a 200k-row window (about 6 s of pricing) shares the process with other requests through thread switching and slows them, rather than running in parallel. Moving it to a process pool would remove that, and was not done because the cap bounds the cost and the endpoint is read only from the panel.
- The rows endpoint (at most 500 rows) prices inline on the event loop.
- With the feature off, the hooks on the hot paths do one cached-settings check and return. Each Anthropic call runs `anthropic_cache_write_kwargs`, each traced row runs `trace_usage_info` in H2, and each claude-code, Codex, OpenAI-compatible, Responses and xAI OAuth call runs its adapter. The settings are cached, so nothing else is parsed. That is the accepted cost of the hooks; the alternative is an edit to each provider's call sites.
- Not done: exposing the flag through the API's `/v1/version` features object. That would add a field to `FeaturesInfo`, which the generated Python, Go and OpenAPI clients carry, so the generated surface would change. The control plane's own env var was used instead (see "Control-plane switch").
- The panel and the per-call line show a loading state only after the flag is on, before the first response for the bank. A flag/API mismatch shows that state briefly and then hides.

- Upstream's `chatgpt` provider (#5688, Sign in with ChatGPT) is in neither `DEFAULT_BILLING` nor `DEFAULT_PROVIDER_MAP`, so its rows read as metered and unmapped, and `llm_quota` does not query it. Add it when a lane uses it.
- `scripts/fork/llm_cost_replay.py` prices every row of an export. Since #5835 an `llm_requests` export also holds embedding, rerank and decision rows; filter them out before replaying.

## Refreshing prices

Bump `genai-prices` in `hindsight-api-slim/pyproject.toml`, run `uv lock`, run `tests/test_llm_cost.py` (its dataset guard fails if a model the deployment uses disappears), and note the version in the commit. The package's network auto-update is never enabled. genai-prices is MIT-licensed; nothing is vendored.

## Maintaining the patch (rebase and cherry-pick)

Files owned by the fork (new logic, no upstream edits):

- `hindsight-api-slim/hindsight_api/engine/llm_cost.py`
- `hindsight-api-slim/hindsight_api/engine/llm_cost_store.py`
- `hindsight-api-slim/hindsight_api/api/llm_cost_routes.py`
- `hindsight-api-slim/tests/test_llm_cost.py`, `test_llm_cost_store.py`, `test_llm_cost_routes.py`, `test_llm_cost_hooks.py`
- `hindsight-api-slim/tests/test_fork_anthropic_usage.py`, `test_fork_codex_usage.py`, `test_fork_reflect_usage.py`, `test_fork_claude_code_usage.py`, `test_fork_openai_compat_usage.py` (the bug-fix regression tests)
- `hindsight-api-slim/tests/test_fork_xai_oauth_cost.py` (the xai-oauth cost hook, H22)
- `hindsight-api-slim/tests/test_llm_cost_live.py` (real provider calls, gated by `HINDSIGHT_COST_LIVE=1`)
- `hindsight-api-slim/tests/test_llm_cost_replay.py` and `scripts/fork/llm_cost_replay.py` (the CSV replay tool and its test)
- `hindsight-control-plane/src/app/api/banks/[bankId]/llm-requests/cost/route.ts`, `.../cost/rows/route.ts`
- `hindsight-control-plane/src/lib/fork/llm-cost.ts`, `hindsight-control-plane/src/lib/fork/llm-cost-proxy.ts`, `hindsight-control-plane/src/lib/fork/fork-request.ts` (the error type both clients throw), `hindsight-control-plane/src/components/fork/llm-cost-panel.tsx`
- `hindsight-control-plane/tests/components/llm-cost.test.ts`, `llm-cost-proxy.test.ts`
- Subscription quota: `hindsight-api-slim/hindsight_api/engine/llm_quota.py`; `hindsight-api-slim/tests/test_llm_quota.py`, `test_llm_quota_routes.py`, `test_llm_quota_live.py` (real Codex fetch, gated by `HINDSIGHT_QUOTA_LIVE=1`); `hindsight-control-plane/src/app/api/banks/[bankId]/llm-requests/quota/route.ts`; `hindsight-control-plane/src/lib/fork/llm-quota.ts`; `hindsight-control-plane/src/components/fork/llm-quota-card.tsx`; `hindsight-control-plane/tests/components/llm-quota.test.ts`
- `FORK-COST-ACCOUNTING.md`

Upstream hooks. Each one is marked `fork: cost-accounting`:

| # | File | Hook |
|---|---|---|
| H1 | `hindsight-api-slim/hindsight_api/config.py` | `# fork: cost-accounting` above `def llm_cost_env()`, at end of file (pure append) |
| H2 | `hindsight-api-slim/hindsight_api/engine/llm_trace.py` | in `LLMTraceRecorder.record_llm_call`, after the `tool_calls` line of `llm_info`: a local import of `trace_usage_info`, then `llm_info.update(trace_usage_info(_extra))`, both marked. Runs on every row: with the feature off it returns `{}` and `llm_info` is unchanged; with it on, every row gets at least the v2 marker. Depends on the recorder keeping its `**_extra` parameter |
| H3 | `hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py` | in each of `call()` and `call_with_tools()`, a local import of `anthropic_cache_write_kwargs` just before the `span_recorder.record_llm_call(...)`. Local, so the fork module stays out of the provider's import graph |
| H4 | same file, `call()` | `**anthropic_cache_write_kwargs(response),  # fork: cost-accounting` in `span_recorder.record_llm_call(...)` |
| H5 | same file, `call_with_tools()` | the same kwargs line in its `span_recorder.record_llm_call(...)` |
| H6 | `hindsight-api-slim/hindsight_api/api/http.py` | in `_register_routes`, after the `/llm-requests/stats` handler: `# fork: cost-accounting`, a local import of `register_llm_cost_routes`, and `register_llm_cost_routes(app, get_request_context)`. Depends on `get_request_context` being that closure |
| H7 | `hindsight-api-slim/pyproject.toml` | `"genai-prices>=0.1.12,<0.2",  # fork: cost-accounting` in `dependencies` |
| H8 | `uv.lock` (repo root) | regenerated by `uv lock` (generated, no marker) |
| H9 | `hindsight-control-plane/src/components/llm-requests-view.tsx` | `import { LLMCostPanel, LLMCallCost } from "@/components/fork/llm-cost-panel"; // fork: cost-accounting` |
| H10 | same file, `LLMRequestsView` render | `{/* fork: cost-accounting */ <LLMCostPanel bankId={currentBank} />}` on one line, after `<LLMRequestChart>` |
| H11 | same file, `SpanDetails` | `{/* fork: cost-accounting */ <LLMCallCost entry={entry} />}` on one line, after the tokens block |
| H12 | `hindsight-control-plane/src/app/api/version/route.ts` | import of `llmCostPanelEnabled`, and `features.llm_cost = llmCostPanelEnabled(process.env.HINDSIGHT_CP_LLM_COST_ENABLED)` after the `access_key_auth` line |
| H13 | `hindsight-control-plane/src/lib/features-context.tsx` | `llm_cost: boolean` in the `Features` interface and `llm_cost: false` in `defaultFeatures` |
| H14 | `hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py`, `call()` | a local import of `claude_code_usage_kwargs` just before the span recorder call inside the span `try`, and `**claude_code_usage_kwargs(sdk_usage),  # fork: cost-accounting` as the last kwarg of that `record_llm_call`. `sdk_usage` is upstream's (#5693): the SDK usage dict, None when the run reported none, which the hook records as an estimate |
| H15 | same file, `call_with_tools()` | the same two lines in its `get_span_recorder().record_llm_call(...)` |
| H16 | `hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py`, `_usage_from_completed_event` | upstream's `return TokenUsage(...)` becomes `result = TokenUsage(...)` (marked), then a local import of `note_codex_usage`, `note_codex_usage(result, data)` (the raw event; the cost module reads the cache-write count from it) and `return result`. `call()` stashes a copy of it, which `FORK-ATTEMPT-USAGE.md` A7 aliases to this note |
| H17 | same file, `call()` span | a local import of `codex_usage_kwargs` and `**codex_usage_kwargs(token_usage),  # fork: cost-accounting` in `span_recorder.record_llm_call(...)` |
| H18 | same file, `call_with_tools()` span | the same two lines |
| H19 | `hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py`, `call()` success span | a local import of `openai_compatible_usage_kwargs` and `**openai_compatible_usage_kwargs(self.provider, self.base_url, response),  # fork: cost-accounting` in the `span_recorder.record_llm_call(...)` after `visible_token_usage` |
| H20 | same file, `call_with_tools()` span | the same two lines |
| H21 | `hindsight-api-slim/hindsight_api/engine/providers/openai_responses_llm.py` | `_record_success` gains a `response` parameter marked `# fork: cost-accounting`; inside it, a local import and `**openai_responses_usage_kwargs(response),` in its `record_llm_call`; both `self._record_success(` call sites pass `response=response,  # fork: cost-accounting` |
| H22 | `hindsight-api-slim/hindsight_api/engine/providers/xai_oauth_llm.py` | `cost_in_usd_ticks: object = None` on `_ChatUsage` (marked; deliberately untyped, so a wrong-typed value never fails the completion parse, and `xai_oauth_usage_kwargs` filters it); `_record_span` gains a `fork_usage` parameter (marked) and forwards `**(fork_usage or {})`; both `self._record_span(` call sites pass `fork_usage=xai_oauth_usage_kwargs(...)` with a local import above |
| H23 | `hindsight-api-slim/hindsight_api/worker/main.py`, `main()` | a local import of `load_cost_settings` and `load_cost_settings()` right after `load_dotenv_for_entrypoint()` (both marked): the worker validates `HINDSIGHT_API_LLM_COST_*` at startup, as the API does |
| H24 | `hindsight-api-slim/hindsight_api/engine/memory_engine.py`, `MemoryEngine.close()` | a local import of `drain_detached_refreshes` and `await drain_detached_refreshes()` (both marked), just before the comment above `close_loop_sessions()`. Depends on `close()` still closing the loop's sessions there; `test_engine_close_drains_quota_refreshes_before_closing_sessions` checks the order. With the feature off no quota fetch runs, `_DETACHED_REFRESHES` is empty and the drain returns at once |

The fork-owned modules `engine/llm_cost.py`, `src/components/fork/llm-cost-panel.tsx`, `src/lib/fork/llm-cost.ts` and `src/lib/fork/fork-request.ts` also carry a `fork: cost-accounting` header comment. The other fork files (`llm_cost_store.py`, `llm_cost_routes.py`) are fork files by path and have no marker. So `grep -rn "fork: cost-accounting" hindsight-api-slim/hindsight_api hindsight-api-slim/pyproject.toml hindsight-control-plane/src` returns every hook line in the table above, plus the header comment of the fork-owned modules (`llm_cost.py`, `llm-cost.ts`, `fork-request.ts`, `llm-cost-panel.tsx`) and the cost route files. Files of the subscription quota feature also carry the marker in their headers; they are not hook lines and are not in the table. Any hook line from the table that is missing from that output has been lost. The hook-existence tests in `tests/test_llm_cost.py` and `tests/test_llm_cost_hooks.py` fail if a hook is lost (H7 is checked in `tests/test_llm_cost_hooks.py`); `tests/test_llm_cost_routes.py` checks the H6 block.

After each rebase, diff `anthropic_llm.py`, `codex_llm.py`, `claude_code_llm.py`, `openai_compatible_llm.py`, `openai_responses_llm.py`, `xai_oauth_llm.py` and `memory_engine.py` (the reflect usage conversion) against the previous tag. If upstream fixed any of the bugs, drop that fix commit and keep the hooks. Re-anchor H4, H5 and H14-H22 on upstream's span record calls if they moved; the hook tests count the occurrences. Re-diff `LLMTraceRecorder.record_llm_call` and the `llm_requests` columns as well.

Upstream behaviour this patch depends on, to re-diff after each rebase (private names, so upstream can rename them without notice):

- `llm_cost_routes.py`: `get_request_context` (the closure H6 passes in), `load_cost_settings`, `app.state.memory`, and `http.py`'s `_parse_iso_datetime` and `_internal_error`. `OperationValidationError` is the 403 path.
- `llm_quota.py`: `MemoryEngine._llm_config`, `_retain_llm_config`, `_reflect_llm_config`, `_consolidation_llm_config` and `_mental_model_refresh_llm_override` (the lanes queried); `MultiLLMProvider.members`; `LLMProvider.provider`, `api_key`, `base_url` and `_provider_impl`; `CodexLLM._ensure_fresh_token` (shielded from the quota deadline, so it must finish a rotation it has started on its own; `CodexAuthManager.refresh_tokens` bounds it with its auth-store lock timeout and per-phase session timeout), `_build_request_headers`, `_auth_file` and `base_url`; `CodexRefreshExpiredError` (`providers/codex_auth.py`); `MemoryEngine.close()` calling `close_loop_sessions()` (H24 drains detached refreshes just before it); the Claude Code lane's `api_key`. Provider response shapes (Codex `/wham/usage`, Claude `/api/oauth/usage`, Z.ai quota) are not upstream names, but they are parsed by `llm_quota.py`'s models and need the same re-check when a provider changes them.
- `llm_cost_store.py`: `MemoryEngine._get_backend`, `MemoryEngine.get_bank_profile`, `MemoryEngine._refuse_bank_wide_if_scoped` (tag-scope refusal, same as `llm-requests/stats`), `acquire_with_retry` (`engine/db_utils.py`), `conn.parse_json` and `conn.fetch` on the pooled connection, `fq_table` and `_is_oracle` (`engine/schema.py`), and `ResultRow` (`engine/db`).
- `llm_requests` columns and `llm_info` JSONB (`engine/llm_trace.py`, `LLMTraceRecorder.record_llm_call` and its `**_extra` parameter). The summary and the per-row costs read LLM calls only, through upstream's `call_type_clause("llm")` (`llm_trace.py`): since #5835 the table also holds embedding, rerank and decision-model rows, which are not priced.
- Provider usage shapes and the span-recorder call sites (the cost adapters depend on them):
  - `anthropic_llm.py`: `_usage_from_anthropic_response` and the `stash_response_usage` calls before parsing.
  - `codex_llm.py`: `_usage_from_completed_event` (its `data` argument and the `TokenUsage` it returns), `call()`'s `token_usage` and `call_with_tools()`'s `usage` (both must be that same object when the stream reported usage), and the two codex span record sites. The raw `response.completed` usage shape is read by `llm_cost.py` (`_CodexCostEvent`).
  - `claude_code_llm.py`: `sdk_usage` in both message loops (`ResultMessage.usage`, and in the tool path also `AssistantMessage.usage` before the `break` on the first tool call), `_sdk_token_usage`, and the `claude-agent-sdk` message types.
  - `openai_compatible_llm.py`: `visible_token_usage`, `_cached_prompt_tokens`, and the `prompt_tokens_details` and `usage` attribute reads.
  - `openai_responses_llm.py`: `_record_success` and its two call sites.
  - `xai_oauth_llm.py`: `_ChatUsage`, `_record_span`, and its two call sites.
- `hindsight_api/api/http.py` `FeaturesInfo` is not touched; the control plane's `/api/version` route and `features-context.tsx` depend on the `features` object shape only.
- Control plane: `llm-requests-view.tsx` (`SpanDetails` and the `LLMRequestsView` render), `useFeatures` in `features-context.tsx`, and `src/app/api/version/route.ts`.

On a conflict in `uv.lock`, take upstream's lock and re-run `uv lock`, then run `uv pip check` in `hindsight-api-slim`. A stale uv cache can leave a pin that `uv lock --check` accepts while `uv pip check` reports a dependency the lock does not satisfy (for example `httpx2` requiring a newer `idna`). The image build runs `uv pip check` after `uv sync --locked`, so that failure stops the build.

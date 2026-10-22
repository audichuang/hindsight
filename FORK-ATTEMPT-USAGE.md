# Fork: attempt usage (Codex retries and runaway streams)

A Codex call can make several requests to the backend: a runaway stream is retried, and so is a reply that fails to parse. Upstream keeps one usage slot per call, and each stash overwrites it ("the last attempt wins"). A runaway attempt never sees `response.completed`, so it stashes nothing. As a result:

- an error row whose every attempt ran away has NULL tokens;
- a success row after a runaway or a parse failure carries the last attempt's tokens only;
- an error row holding Codex's chars/4 fallback is not marked as an estimate.

This patch keeps a ledger of a call's Codex attempts and writes the sum on the call's one trace row. It is part of the cost accounting (`FORK-COST-ACCOUNTING.md`) and runs only when `HINDSIGHT_API_LLM_COST_ENABLED` is on. It has no environment variables of its own. With the flag off, nothing is recorded and rows are written exactly as upstream writes them.

## What a row carries

Each attempt counts as follows:

| attempt | counted as | source |
|---|---|---|
| stashed provider usage (`response.completed`, `response.failed`) | that usage (the attempt's last stash) | provider |
| stashed Codex chars/4 fallback (stream ended without usage) | that estimate | estimate |
| no stash, at least one delta event received (runaway, deadline, dropped connection) | input: the provider-counted input of another attempt of the call (every attempt sends the same messages), else the call's messages at chars/4; cached: 0; output: one token per delta event | estimate |
| no stash, no delta event (failed before the backend streamed) | nothing | — |
| the attempt a success row is recorded from | the counts on that row | the row's own `usage_source` |

The row's `input_tokens`, `output_tokens`, `cached_tokens` and `thoughts_tokens` are the sums. `llm_info.usage.usage_source` is `estimate` when any counted attempt was estimated, otherwise `provider`. The cache-write count is the sum over the attempts that reported one. A call with one attempt and provider usage is left exactly as the provider recorded it.

The row's duration already covers every attempt (the wrapper times the whole call), so the sums match it.

One delta event per output token was measured on the ChatGPT backend (gpt-6.1-sol, 2026-10-20: 4,776 `response.output_text.delta` events for 4,782 visible output tokens). Reasoning summary deltas count as output, and encrypted reasoning is not streamed, so a runaway's reasoning tokens are not in the estimate (15 of 4,797 output tokens at low effort in that check; more at `high`).

The input estimate is exact when another attempt of the call has provider counts (a runaway followed by a success, the common case). When every counted attempt ran away, chars/4 is a lower bound: on the reference deployment's refresh calls (more than 100k characters of messages) provider input ran 1.24-1.75x chars/4 (p50 1.52), so the estimate is about a third low. The estimated attempt's cached tokens are 0, because whether it hit the prompt cache is unknown; its input is priced as uncached, an upper bound for that part of the cost.

The returned `TokenUsage`, the metrics counters and the OTel spans still carry the last attempt only. Only the `llm_requests` row is folded.

## Why this design

- **One row per call, not one per attempt.** The control plane and the cost endpoints count calls; extra rows would change their meaning and every upstream consumer of the table.
- **The ledger rides on the per-call usage reset.** The wrapper calls `set_response_usage(None)` when a call starts and `reset_response_usage` when it ends, so the ledger opens and closes there. No hook goes into `llm_wrapper.py`, whose `call` and `call_with_tools` change often upstream.
- **Attempts are opened by Codex only**, in `_iter_sse_lines`, which every Codex request streams through (`call` and `call_with_tools`). Other providers stash several times per attempt (claude-code does), so "one stash per attempt" cannot be assumed for them; without an open attempt their stashes are ignored and their rows are unchanged.
- **Delta events are counted from the raw body** as each chunk arrives, before the size guard can abandon the stream, so the chunk that trips the guard is counted too.
- **A success row is authoritative for its own attempt.** The provider records success from the attempt that succeeded, so the ledger uses the row's counts for that attempt even when the provider did not stash them.
- **Provider usage is told from the chars/4 fallback by the attempt's own note.** `begin_codex_attempt` clears the cost module's Codex note (`forget_codex_usage`), so a note present at stash time was made by `note_codex_usage` while this attempt's stream was parsed (H16 of the cost accounting). At stash time the ledger asks `codex_stash_breakdown(usage)`: the stash is provider usage when it is the noted object or carries the same four counts (since upstream #5693, `call()` stashes a copy of the parsed `TokenUsage`, so identity alone cannot tie them). A chars/4 fallback finds no note.

## Hooks

Each line is marked `# fork: attempt-usage`.

| # | File | Hook |
|---|---|---|
| A1 | `hindsight-api-slim/hindsight_api/engine/llm_trace.py` | the import of `close_call_ledger, fold_attempt_usage, note_stashed_usage, open_call_ledger` after `from .db_utils import acquire_with_retry` |
| A2 | same file, `set_response_usage` | `open_call_ledger()` before the `return` |
| A3 | same file, `stash_response_usage` | `note_stashed_usage(usage)` before the `set` |
| A4 | same file, `reset_response_usage` | `close_call_ledger()` before the `reset` |
| A5 | same file, `LLMTraceRecorder.record_llm_call` | the decorator `@fold_attempt_usage`. Every caller passes keywords (the tracing fan-out does); a positional call is passed through unfolded |
| A6 | `hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py`, `_iter_sse_lines` | a local import of `begin_codex_attempt`, `attempt = begin_codex_attempt()` before the decoder, and `attempt.feed(text)` right after `seen_chars += len(text)` |

Fork-owned files: `hindsight-api-slim/hindsight_api/engine/llm_attempt_usage.py`, `hindsight-api-slim/hindsight_api/engine/sse_delta_scan.py` (shared with `FORK-CODEX-OUTPUT-CAP.md`), `hindsight-api-slim/tests/test_fork_attempt_usage.py` (behaviour and hook-existence tests), and this file.

## Related fix: the Codex stream ceiling

Two commits with no marker (an upstream defect), `fix(codex): stop abandoning long replies as runaway streams` and `fix(codex): set the stream ceiling to 8 MiB so it trips before the deadline`, raise `_MAX_SSE_BODY_CHARS` in `codex_llm.py` from 4 MiB to 8 MiB. Drop both when upstream changes the guard. Their regression test is `tests/test_fork_codex_sse_ceiling.py`.

- The guard counts the raw SSE body. Measured on the ChatGPT backend (gpt-6.1-sol, 2026-10-20), each output token arrives as one delta event of about 287 characters, so 4 MiB ended replies at about 14.6k output tokens. Split-synthesis map calls of a mental-model refresh reach 13-14k in normal operation: in one week the longest successful one was 13.6k, and 31 failed as `exceeded 4194304 characters`. `CodexLLM` never sends `max_completion_tokens`, so nothing else bounds a Codex reply.
- 8 MiB is about 29k output tokens, roughly twice the longest reply seen. That distribution was truncated at 4 MiB, so the margin is a judgment, not a measurement.
- The ceiling must also trip before the per-attempt deadline. The deployment streams long refresh replies at 43-66 output tokens/s (p50-p95), so 8 MiB fills in about 440-680 s, inside the refresh lane's 900 s deadline. The first commit's 16 MiB (~58k tokens) needed 880-1350 s: the deadline, not the size, ended nearly every runaway, and a runaway call cost up to (retries + 1) x timeout plus backoff (2 x 900 s + 30 s with the deployment's `MENTAL_MODEL_REFRESH_LLM_MAX_RETRIES=1`) instead of 400-840 s. A runaway slower than ~32 tokens/s still reaches the deadline first.
- Map calls of one refresh run concurrently (`asyncio.gather`), so two runaway maps overlap in time rather than adding up.

The delta-count cap is a separate patch, `FORK-CODEX-OUTPUT-CAP.md` (`HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS`, off by default). It has its own counter in `_iter_sse_lines`, because the attempt meter here counts only with the cost flag on and a safety bound must not depend on it. Both count with the same scanner, `engine/sse_delta_scan.py` (every `event: response.<...>.delta` line, lines split as `_iter_sse_lines` splits them, a lone `\r` included). The order inside a chunk is: attempt meter, then the 8 MiB body ceiling, then the delta cap, so the meter counts the chunk that trips either guard. The delta cap is meant to trip first. The recommended 25000 is below the ~29k deltas that 8 MiB holds, so the body ceiling remains the second guard, for a backend whose envelope or event names change.

### Not done

- **`max_output_tokens` through the lane's extra body**: a live check (2026-10-20) showed the backend accepts it and ends the stream with `response.incomplete` (`reason: max_output_tokens`; the cap includes reasoning tokens). `CodexLLM` does not treat `response.incomplete` as a failure, so it returns the truncated reply as a success. Setting the cap without first handling that event would hand a map call a silently cut claims list.
- **Smaller map chunks** (`REFLECT_MAX_CONTEXT_TOKENS` is global and also bounds the agent loop and single-pass reflect; a map-only chunk size is new code) and **a terser map prompt** (upstream product behaviour, deliberately exhaustive since #3122) are left as suggestions for upstream.
- **Stalls**: a call that streams almost nothing until the deadline (one `reflect_tool_call` in a week) is bound only by the deadline; no size or count guard sees it.

## Maintaining the patch

After each rebase, re-diff `llm_trace.py` (`set_response_usage`, `stash_response_usage`, `reset_response_usage`, `LLMTraceRecorder.record_llm_call` and its keyword arguments), the wrapper's use of those three functions in `LLMProvider.call` and `call_with_tools` (set at call start, reset in `finally`), and `codex_llm.py` (`_iter_sse_lines`, the stash sites, and the retry loops in `call` and `call_with_tools`).

### Rebasing onto a release that contains upstream #5693

Upstream #5693 (`dcc3ee421`, "report real token usage for Codex, Claude Code and Anthropic") rewrites `codex_llm.py` around `_CodexStreamReply`. On `origin/main` at `c5f585f8f` (which contains #5693), `_iter_sse_lines` and the size guard are unchanged by it: the branch's whole `codex_llm.py` diff (A6 and both ceiling commits) applies with a plain `git apply --check`. What changes is how usage is stashed:

- `call()` stashes once per attempt, after the stream is parsed, a new `LLMResponseUsage` copied from `reply.usage` (or the chars/4 fallback when `reply.usage` is None). A runaway still raises before the stash, so the ledger's delta estimate still applies.
- The usage object that is stashed is no longer the object `_usage_from_completed_event` built, so an identity check sees every stash as an estimate. The v1.1.0 rebase resolved this inside the fork modules instead of re-anchoring H16 on the stash site (which would need `_CodexStreamReply` to carry the raw event, an edit to an upstream dataclass): the note is cleared at each attempt start and matched to the stash by its counts (see "Provider usage is told from the chars/4 fallback" above). If upstream ever stashes something other than a copy of the parsed counts, revisit this.
- `call_with_tools` no longer stashes. Its success row is still counted correctly (the success row is authoritative for its own attempt), but an earlier parse-failed attempt in that path has nothing to count. It has no parse retry today, so nothing is lost.
- `test_fork_attempt_usage.py::test_parse_failure_retry_adds_both_provider_usages` fails if the stash is no longer tied to the attempt's note, and `test_codex_keeps_the_attempt_meter_hooks` fails if A6 is lost.

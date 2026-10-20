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
- **Delta events are counted from the raw body** as each chunk arrives, before the size guard can abandon the stream, so the chunk that trips the guard is counted too. The count comes from the per-attempt scan that also enforces the output cap (`CodexAttemptScan`, `FORK-CODEX-OUTPUT-CAP.md`), so a chunk is split into lines once. `llm_attempt_usage.open_codex_attempt()` only records the attempt; it returns None when no ledger is open, and then nothing is recorded.
- **A success row is authoritative for its own attempt.** The provider records success from the attempt that succeeded, so the ledger uses the row's counts for that attempt even when the provider did not stash them.
- **Provider usage is told from the chars/4 fallback by identity.** At stash time the ledger asks `codex_usage_kwargs(usage)` whether this object is the one `note_codex_usage` saw parsed from the stream (H16 of the cost accounting).

## Hooks

Each line is marked `# fork: attempt-usage`.

| # | File | Hook |
|---|---|---|
| A1 | `hindsight-api-slim/hindsight_api/engine/llm_trace.py` | the import of `close_call_ledger, fold_attempt_usage, note_stashed_usage, open_call_ledger` after `from .db_utils import acquire_with_retry` |
| A2 | same file, `set_response_usage` | `open_call_ledger()` before the `return` |
| A3 | same file, `stash_response_usage` | `note_stashed_usage(usage)` before the `set` |
| A4 | same file, `reset_response_usage` | `close_call_ledger()` before the `reset` |
| A5 | same file, `LLMTraceRecorder.record_llm_call` | the decorator `@fold_attempt_usage`. Every caller passes keywords (the tracing fan-out does); a positional call is passed through unfolded |
| A6 | `hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py`, `_iter_sse_lines` | shared with the output cap: `attempt = self._fork_attempt_scan()` before the decoder and `attempt.feed(text, seen_chars)` right after `seen_chars += len(text)`, both marked `# fork: codex-output-cap, attempt-usage` (C3 and C4 of `FORK-CODEX-OUTPUT-CAP.md`; the scan is built in `CodexLLM.__init__`, C2) |

Fork-owned files: `hindsight-api-slim/hindsight_api/engine/llm_attempt_usage.py`, `hindsight-api-slim/hindsight_api/engine/providers/codex_output_cap.py` and `hindsight-api-slim/hindsight_api/engine/sse_delta_scan.py` (shared with `FORK-CODEX-OUTPUT-CAP.md`), `hindsight-api-slim/tests/test_fork_attempt_usage.py` (behaviour and hook-existence tests), and this file.

## The Codex stream ceiling

The 8 MiB raw-body ceiling (`_MAX_SSE_BODY_CHARS`) and the delta-count cap that complements it are described in `FORK-CODEX-OUTPUT-CAP.md`, together with what was left out (`max_output_tokens`, smaller map chunks, stalls). The attempt meter counts the chunk that trips either guard.

## Maintaining the patch

After each rebase, re-diff `llm_trace.py` (`set_response_usage`, `stash_response_usage`, `reset_response_usage`, `LLMTraceRecorder.record_llm_call` and its keyword arguments), the wrapper's use of those three functions in `LLMProvider.call` and `call_with_tools` (set at call start, reset in `finally`), and `codex_llm.py` (`_iter_sse_lines`, the stash sites, and the retry loops in `call` and `call_with_tools`).

### Rebasing onto a release that contains upstream #5693

Upstream #5693 (`dcc3ee421`, "report real token usage for Codex, Claude Code and Anthropic") rewrites `codex_llm.py` around `_CodexStreamReply`. On `origin/main` at `c5f585f8f` (which contains #5693), `_iter_sse_lines` and the size guard are unchanged by it: the branch's whole `codex_llm.py` diff (A6 and the output-cap hooks) applies with a plain `git apply --check`. What changes is how usage is stashed:

- `call()` stashes once per attempt, after the stream is parsed, a new `LLMResponseUsage` copied from `reply.usage` (or the chars/4 fallback when `reply.usage` is None). A runaway still raises before the stash, so the ledger's delta estimate still applies.
- The usage object that is stashed is no longer the object `_usage_from_completed_event` built. So the identity check in `note_stashed_usage` (through `codex_usage_kwargs`) sees every stash as an estimate unless H16 of the cost accounting is re-anchored to note the stashed object when `reply.usage is not None`: pass the `LLMResponseUsage` given to `stash_response_usage` as the first argument of `note_codex_usage`, and as the second whatever H16's current signature takes (see the H16 row of `FORK-COST-ACCOUNTING.md`). Since the cost-quota fixes that is the raw `response.completed` event, which the stash site of #5693 does not have: `_CodexStreamReply` has to keep it. Without the re-anchor, retried calls that never ran away are marked `estimate`.
- `call_with_tools` no longer stashes. Its success row is still counted correctly (the success row is authoritative for its own attempt), but an earlier parse-failed attempt in that path has nothing to count. It has no parse retry today, so nothing is lost.
- `test_fork_attempt_usage.py::test_parse_failure_retry_adds_both_provider_usages` fails if the H16 re-anchor is missing, and `test_codex_keeps_the_attempt_meter_hooks` fails if A6 is lost.

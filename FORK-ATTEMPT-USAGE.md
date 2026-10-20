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
| no stash, at least one delta event received (runaway, deadline, dropped connection) | input: the call's messages at chars/4; output: one token per delta event | estimate |
| no stash, no delta event (failed before the backend streamed) | nothing | — |
| the attempt a success row is recorded from | the counts on that row | the row's own `usage_source` |

The row's `input_tokens`, `output_tokens`, `cached_tokens` and `thoughts_tokens` are the sums. `llm_info.usage.usage_source` is `estimate` when any counted attempt was estimated, otherwise `provider`. The cache-write count is the sum over the attempts that reported one. A call with one attempt and provider usage is left exactly as the provider recorded it.

The row's duration already covers every attempt (the wrapper times the whole call), so the sums match it.

One delta event per output token is an approximation: the Responses stream sends one `*.delta` event per token in practice. Reasoning summary deltas count as output, and encrypted reasoning is not streamed, so a runaway's reasoning tokens are not in the estimate.

The returned `TokenUsage`, the metrics counters and the OTel spans still carry the last attempt only. Only the `llm_requests` row is folded.

## Why this design

- **One row per call, not one per attempt.** The control plane and the cost endpoints count calls; extra rows would change their meaning and every upstream consumer of the table.
- **The ledger rides on the per-call usage reset.** The wrapper calls `set_response_usage(None)` when a call starts and `reset_response_usage` when it ends, so the ledger opens and closes there. No hook goes into `llm_wrapper.py`, whose `call` and `call_with_tools` change often upstream.
- **Attempts are opened by Codex only**, in `_iter_sse_lines`, which every Codex request streams through (`call` and `call_with_tools`). Other providers stash several times per attempt (claude-code does), so "one stash per attempt" cannot be assumed for them; without an open attempt their stashes are ignored and their rows are unchanged.
- **Delta events are counted from the raw body** as each chunk arrives, before the size guard can abandon the stream, so the chunk that trips the guard is counted too.
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
| A6 | `hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py`, `_iter_sse_lines` | a local import of `begin_codex_attempt`, `attempt = begin_codex_attempt()` before the decoder, and `attempt.feed(text)` right after `seen_chars += len(text)` |

Fork-owned files: `hindsight-api-slim/hindsight_api/engine/llm_attempt_usage.py`, `hindsight-api-slim/tests/test_fork_attempt_usage.py` (behaviour and hook-existence tests), and this file.

## Related fix: the Codex stream ceiling

The commit `fix(codex): stop abandoning long replies as runaway streams` (no marker; an upstream defect) raises `_MAX_SSE_BODY_CHARS` in `codex_llm.py` from 4 MiB to 16 MiB. The guard counts the raw SSE body, an estimated ~300 characters per output token (inferred from the reference deployment's traces), so 4 MiB ended replies at roughly 13-17k output tokens, which split-synthesis map calls of a mental-model refresh reach in normal operation. `CodexLLM` never sends `max_completion_tokens`, so nothing else bounds a Codex reply. Its regression test is `tests/test_fork_codex_sse_ceiling.py`. Drop the commit when upstream changes the guard.

## Maintaining the patch

After each rebase, re-diff `llm_trace.py` (`set_response_usage`, `stash_response_usage`, `reset_response_usage`, `LLMTraceRecorder.record_llm_call` and its keyword arguments), the wrapper's use of those three functions in `LLMProvider.call` and `call_with_tools` (set at call start, reset in `finally`), and `codex_llm.py` (`_iter_sse_lines`, the stash sites, and the retry loops in `call` and `call_with_tools`).

### Rebasing onto a release that contains upstream #5693

Upstream #5693 (`dcc3ee421`, "report real token usage for Codex, Claude Code and Anthropic") rewrites `codex_llm.py` around `_CodexStreamReply`. As of `origin/main` at the time of writing, `_iter_sse_lines` and the size guard are unchanged by it, so A6 and the ceiling commit apply as they are (checked with `git apply --3way --check`). What changes is how usage is stashed:

- `call()` stashes once per attempt, after the stream is parsed, a new `LLMResponseUsage` copied from `reply.usage` (or the chars/4 fallback when `reply.usage` is None). A runaway still raises before the stash, so the ledger's delta estimate still applies.
- The usage object that is stashed is no longer the object `_usage_from_completed_event` built. So the identity check in `note_stashed_usage` (through `codex_usage_kwargs`) sees every stash as an estimate unless H16 of the cost accounting is re-anchored to note the stashed object: call `note_codex_usage(<the LLMResponseUsage passed to stash_response_usage>, <cache-write count>)` when `reply.usage is not None`. Without that, retried calls that never ran away are marked `estimate`.
- `call_with_tools` no longer stashes. Its success row is still counted correctly (the success row is authoritative for its own attempt), but an earlier parse-failed attempt in that path has nothing to count. It has no parse retry today, so nothing is lost.
- `test_fork_attempt_usage.py::test_parse_failure_retry_adds_both_provider_usages` fails if the H16 re-anchor is missing, and `test_codex_keeps_the_attempt_meter_hooks` fails if A6 is lost.

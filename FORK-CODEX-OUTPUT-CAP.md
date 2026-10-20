# Fork: Codex output cap (stop a runaway by delta count)

`CodexLLM` never sends a token limit to the ChatGPT backend, so a reply that runs away is bounded only by the raw-body ceiling `_MAX_SSE_BODY_CHARS` (8 MiB, see "Related fix" in `FORK-ATTEMPT-USAGE.md`) and the per-attempt wall-clock deadline. The body ceiling counts characters of the SSE stream, envelope included. This patch adds a bound on what the model generates instead: an attempt that streams more than N output delta events is abandoned with the same `CodexRunawayStreamError` the body ceiling raises.

It is off unless `HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS` is set. With it unset, behaviour is byte-for-byte upstream (plus the body-ceiling fix).

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS` | Most output delta events one Codex attempt may stream. Server-wide (every Codex lane). A positive integer. Unset or empty: no cap. Anything else (`0`, `-1`, `25k`, `2.5`) fails when a `CodexLLM` is built, which is at startup for every configured Codex lane. | unset |

A value set on a server with no Codex lane is never parsed, so a typo there goes unnoticed. That is harmless (nothing reads it) and keeps validation in the one hook that already runs for Codex.

### Recommended value: 25000

- **Above normal replies.** The longest successful split-synthesis map reply seen on the reference deployment was 13.6k output tokens (p99 13.1k). That distribution was truncated by the old 4 MiB ceiling at about 14.6k, so the true tail is not known. 25000 is 1.7x the truncation point and 1.8x the longest reply.
- **Before the deadline.** The refresh lane's per-attempt deadline is 900 s. Long refresh replies stream at 23-66 output tokens/s, mostly about 50. 25000 deltas take 500 s at 50 tokens/s, 580 s at 43 and 380 s at 66. A runaway slower than 27.8 tokens/s (25000 / 900) still reaches the deadline first. At the slowest speed observed, 23 tokens/s, that is the case: the deadline ends the attempt, as it does today.
- **Below the body ceiling, so this is the first guard.** 8 MiB at about 287 characters per text delta is about 29.2k deltas. The envelope grows as the reply grows (the sequence number gains digits), so the body ceiling trips somewhat earlier than that. For text replies, a delta cap at or above about 29k (30000, for example) would never fire before the body ceiling and would add nothing. The envelope differs by event type: tool-argument deltas measured about 271 characters each (live check below), so 8 MiB holds about 30.9k of them, and an event type with a smaller envelope could let a cap of about 29k fire first. 25000 is below the body ceiling for both measured types, with about 14% headroom for text deltas and 19% for tool-argument deltas.
- **What it gains over the body ceiling is small per attempt.** A text runaway is stopped about 4k deltas sooner than at 8 MiB, about 80 s at 50 tokens/s. Its main value is that it is measured in output tokens and does not move when the backend changes the envelope (a longer item id or padding, a new field), which can shift the body ceiling by thousands of tokens either way.

If successful replies are later seen near 20k, raise the cap and the body ceiling together: the body ceiling has to stay the looser of the two.

After enabling it, watch the logs for `output delta events (HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS)` (logged by `call()` as `Codex connection error ...` and by `call_with_tools` as `Codex tool call error: ...`), next to the `characters without completing` and `deadline` runaway messages. Trips on calls whose operation otherwise succeeds would mean the cap is cutting legitimate replies.

## Why this design

- **Count deltas, not characters.** One `*.delta` event per output token was measured on the ChatGPT backend (gpt-6.1-sol, 2026-10-20: 4,776 `response.output_text.delta` events for 4,782 visible output tokens). The characters per delta depend on the envelope (event line, JSON with sequence number, item id and obfuscation padding), about 287, and change with the reply length and with any backend change to the envelope. A delta count stays in output tokens.
- **Which events count: every `event: response.<...>.delta`.** In the Responses API every delta event carries model-generated output: visible text (`output_text`, and the legacy `text` / `content_part` names the parsers also read), refusal text, function call arguments, reasoning summary text, reasoning text, custom tool input. Each grows without bound when generation runs away. A tool call whose arguments never close is a runaway too, and `call_with_tools` (the reflect agent) would otherwise be uncapped. Events that do not carry new output (`*.added`, `*.done`, `response.created`, `in_progress`, `completed`, keep-alive comments) are not counted. The attempt meter of `FORK-ATTEMPT-USAGE.md` uses the same scanner (`engine/sse_delta_scan.py`) for its output estimate, so a cap of N stops an attempt the meter counts as N+1 output tokens or more. Reasoning summary deltas are a few per reply (one in a live check at low effort), so they barely move the count.
- **Reasoning tokens are not counted.** The backend sends reasoning encrypted at the end of the reply and does not stream it. Neither this cap nor the body ceiling sees it. A runaway inside reasoning streams nothing and is bound by the deadline (the same as the "stalls" item in `FORK-ATTEMPT-USAGE.md`).
- **Independent of the cost flag.** The attempt meter counts only with `HINDSIGHT_API_LLM_COST_ENABLED` on. A safety bound must not turn off with cost accounting, so this patch has its own counter instead of reading the meter. The two scan the same chunks, a few string operations per chunk. With the cap unset, its counter returns at once.
- **Per attempt.** Each streamed request starts from zero. A retry after a runaway gets the full cap again, as it gets a fresh deadline and a fresh body count.
- **Same line rule as the parser.** Lines are split the way `_iter_sse_lines` splits them (`str.splitlines`: `\n`, `\r\n` and a lone `\r`, a `\r\n` cut between two chunks counted once). A counter that split on `\n` only, as the first version did, saw no event at all in a body with CR-only line ends while the parser read every line, so the cap could be bypassed.
- **Same error as the body ceiling, and the same handling.** The cap raises `CodexRunawayStreamError` from `_iter_sse_lines`, where the body ceiling raises it, so each call path handles a delta-cap trip exactly as it already handles a body-ceiling trip or a deadline:
  - `call()` retries it with backoff up to `max_retries` (each attempt with a fresh counter), and `remote_retry` classifies it as transient.
  - `call_with_tools` does **not** retry it. Upstream retries only a 401/403 there (after a token refresh); every other error, including the body ceiling, the deadline and `CodexStreamError`, is raised at once, and the `LLMProvider` wrapper does not retry either. Its only caller is the reflect agent (`engine/reflect/agent.py`). An error on an agent iteration is logged and the loop moves to the next iteration; a second consecutive error, or an error on the last iteration, fails the reflect. A failed closing `done` call falls back to the standalone synthesis prompt. So one runaway tool call costs an iteration, and two in a row fail the operation, as with the body ceiling today.
  - Adding a runaway retry to `call_with_tools` was considered and left out. The agent's next iteration already is a retry, so a provider retry would only add one more attempt of up to the deadline per iteration, and it would change upstream behaviour for the body ceiling and the deadline as well. Upstream chose to raise non-auth errors at once there, so it is not a defect of the deployed release.
  - The message names the cap and the variable: `Codex response exceeded 25000 output delta events (HINDSIGHT_API_CODEX_MAX_OUTPUT_DELTAS) without completing; abandoning the stream`.
- **Order inside a chunk.** The attempt meter is fed first, then the body ceiling is checked, then the delta cap. The meter therefore counts the chunk that trips either guard. The cap counts the whole chunk before checking, so the stream is abandoned at the chunk that crosses N, and none of that chunk's lines reach the parser.
- **Validated in `CodexLLM.__init__`**, next to the deadline it complements. That is where a Codex lane is built at startup (the same pattern as rerank routing validating when the reranker chain is built). No hook in a server or worker `main`.
- **Not `max_output_tokens`.** The backend accepts it, but it ends the stream with `response.incomplete`, which `CodexLLM` returns as a successful, silently truncated reply (see "Not done" in `FORK-ATTEMPT-USAGE.md`). A runaway error is retried and visible. A truncated success is neither.

Live check (2026-10-20, gpt-6.1-sol, reasoning effort low, cap 50, a request for a 200-line list): the call raised `CodexRunawayStreamError` with the message above. The body that arrived held 51 delta events, 1 `response.reasoning_summary_text.delta` and 50 `response.output_text.delta`, so the stream was dropped at the 51st delta.

Tool-argument granularity (2026-10-20, gpt-6.1-sol, reasoning effort low, no cap, `call_with_tools` with the tool forced, asked for 150 structured claims): 3,631 `response.function_call_arguments.delta` events for 3,645 output tokens (no reasoning tokens), 0.996 deltas per token, about 4.9 argument characters per delta. The body was 985,400 characters, about 271 per delta. Tool arguments therefore count the same as text, one delta per token, and 25000 stops a tool call at about 25k output tokens. Normal reflect tool calls are far shorter than a split-synthesis map reply, so the cap does not cut them off.

## Hooks

Each line is marked `# fork: codex-output-cap`.

| # | File | Hook |
|---|---|---|
| C1 | `hindsight-api-slim/hindsight_api/config.py` | `ENV_CODEX_MAX_OUTPUT_DELTAS` and the accessor `codex_output_cap_env()` at the end of the file, after the cost-accounting accessor |
| C2 | `hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py`, `CodexLLM.__init__` | a local import of `CodexOutputCap` and `self._fork_output_cap = CodexOutputCap.from_env()` right after `self._request_timeout = self.timeout or _DEFAULT_CODEX_TIMEOUT` |
| C3 | same file, `_iter_sse_lines` | `deltas = self._fork_output_cap.begin_attempt()` after `attempt = begin_codex_attempt()` (attempt-usage A6) |
| C4 | same file, `_iter_sse_lines` | `deltas.feed(text)` after the `_MAX_SSE_BODY_CHARS` check's `raise` block, before `if not _LINE_BREAK_RE.search(text):` |

`call` and `call_with_tools` both read the body through `_iter_sse_lines` (`_parse_sse_stream` and `_parse_sse_tool_stream`), so C3 and C4 cover both.

Fork-owned files: `hindsight-api-slim/hindsight_api/engine/providers/codex_output_cap.py`, `hindsight-api-slim/hindsight_api/engine/sse_delta_scan.py` (the line scanner shared with the attempt meter), `hindsight-api-slim/tests/test_fork_codex_output_cap.py` (behaviour, setting and hook-existence tests), `hindsight-api-slim/tests/test_fork_sse_delta_scan.py` (LF, CR, CRLF, mixed and chunk-split line ends, for both the cap and the meter, through `_iter_sse_lines`), and this file.

## Maintaining the patch

After each rebase, re-diff `codex_llm.py`: `CodexLLM.__init__` around `_request_timeout`, `_iter_sse_lines` (the chunk loop and the body-ceiling check), that both parsers still iterate `_iter_sse_lines`, that `CodexRunawayStreamError` is still retried by `call()` and `call_with_tools`, and the SSE event names. `sse_delta_scan._LINE_BREAKS` mirrors `codex_llm._LINE_BREAK_RE`; if upstream changes how `_iter_sse_lines` splits lines, change the scanner with it. Re-check the 287 characters per delta and the ceiling arithmetic above if the backend's envelope changes.

`test_codex_keeps_the_output_cap_hooks` fails if C2, C3 or C4 is lost or moved out of order. `test_config_keeps_the_output_cap_env_hook` fails if C1 is lost. The attempt-usage test `test_codex_keeps_the_attempt_meter_hooks` requires `attempt.feed(text)` to be immediately followed by the body-ceiling check, which is why C4 sits after that check and not between them.

Drop the patch when upstream bounds Codex output itself (a token limit it handles, `response.incomplete` treated as a failure, or a delta-based guard), and keep the env name out of the deployment once it is gone.

### Rebasing onto a release that contains upstream #5693

Upstream #5693 (`dcc3ee421`) rewrites `codex_llm.py` around `_CodexStreamReply`. The anchors of this patch are unchanged by it. On `origin/main` at `c5f585f8f`:

- `self._request_timeout = self.timeout or _DEFAULT_CODEX_TIMEOUT` is in `CodexLLM.__init__` (line 377). C2 goes right after it.
- `_iter_sse_lines` (line 933) has the same body as before: `decoder`, `pending`, `seen_chars`, the `iter_any()` loop, the `_MAX_SSE_BODY_CHARS` check and then `if not _LINE_BREAK_RE.search(text):`. C3 and C4 go where the table says.
- `_parse_sse_stream` (line 973) and `_parse_sse_tool_stream` (line 1237) now return `_CodexStreamReply` and still iterate `self._iter_sse_lines(response)`. The cap raises before any parsing, so the reply type does not matter to it.
- This branch's whole `codex_llm.py` diff (the body-ceiling commits, attempt-usage A6, C2-C4) applies to it with a plain `git apply --check`. With the patch applied there, `test_fork_codex_output_cap.py`, `test_fork_codex_sse_ceiling.py`, `test_codex_stream_deadline.py` and `test_codex_stream_failure.py` pass. In `config.py` the only conflict is the end of the file, where the earlier fork accessors (rerank routing, cost accounting) are cherry-picked first. Append C1 after them.

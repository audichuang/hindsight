# Fork: lane-based reranker routing

This fork-only feature splits the reranker failover chain into two lanes, each with its own member order. It also adds a per-member in-flight cap, an optional background queue that lets a background request wait for a busy member, an optional HTTP 429 cooldown, a lane-wide candidate cap, and per-member per-lane candidate caps that let a slow member rerank only the head of the list. It is inert unless one of its env vars is set: with none set, the plain upstream `MultiCrossEncoder` is used and behaviour is unchanged. Routing applies only to chains of two or more members, because a single reranker has nothing to route. A single member always gets the bare provider, with no in-flight gate, queue, cooldown or lane order (a skip would turn a queued call into `RerankerUnavailableError`). Its routing env is still validated at startup, so a malformed value fails the build, and setting an order, in-flight, queue, cooldown or per-member cap var with one member logs a warning that those vars are ignored. The lane-wide candidate caps still apply to a single reranker.

## Why this design

Problem it solves, in the reference deployment: two free OpenRouter reranker accounts (quota counted per request, not per token: about 1000/day and 20/min each) ran out most days, and the overflow landed on one local GPU reranker that can only run one request at a time. With one shared chain, every caller started at member 0, exhausted accounts cost a round trip on every call, and the candidate cap had to be lowered for everyone (MID 300 -> 100) so the slow member could keep up.

Decisions, with the evidence behind them:

- **Two lanes instead of one chain.** Interactive callers need low latency; mental-model refresh does not. Splitting lets background traffic go to the free local member first and keeps the fast remote members for foreground.
- **Background queues for the local member instead of spilling when busy.** A simulation on real arrival times: with spill-on-busy, 40-60% of background reranks spilled to the remote accounts whether N was 100 or 300; with a queue of up to 30 min, N=300 spilled 0-200 per day and waited 12-29 min at p95. Refresh tolerates that wait.
- **300 candidates.** A 90-query production eval (82 scorable) with a non-truncating bge: hit@5 was 67% at N=100, 70% at 200, 74% at 300, with no query getting worse. Most rescued targets sat at RRF rank 240-300, so 200 rescues few. Remote members bill per request, so 300 costs them nothing extra.
- **Per-member caps, not only lane-wide ones.** The lane-wide foreground cap overrides every budget tier, LOW included. A per-member cap lets foreground send 300 to remote members but only 100 to the slow local one.
- **Cooldown.** The upstream chain has no circuit breaker; without a cooldown every call re-tries an exhausted account first.
- **MAX_INFLIGHT=1 for a single-GPU member.** Measured with llama.cpp: 4 parallel slots gave the same throughput as 1, with 4x the latency per request.
- **Accepted limitations instead of more hooks.** Strict head-before-tail ordering under strategy boosts would need a hook in the recall scoring hot path (see Per-member candidate cap). The fork keeps hooks out of hot, frequently changed upstream code; the limitation matches upstream's own timeout tail.

Local reranker runtime (bge-reranker-v2-m3, Apple Silicon, 24 GB): llama.cpp was chosen.

- oMLX 0.7.0 truncates bge input at 512 tokens with no setting to change it (long documents lose their tail), and sends a whole request as one batch (about 15 GB peak at N=300, which tripped its memory guard).
- llama.cpp does not truncate (verified with an answer placed at about 1,600 tokens), peaks at about 3 GB, and is a native service with nothing to maintain.
- Faster options were measured but not chosen: an oMLX dev build with a per-batch cache-clear wrapper (needs a private hook kept in step with dev releases), and sentence-transformers on MPS (needs a server wrapper written for auth, model name and length limits). TEI exceeded the memory budget at warm-up.
- Benchmark with real-length documents: synthetic documents of half the length made latency look about 2x better (N=300: 13 s synthetic vs 26 s real).

## Lanes

- **FOREGROUND** is everything not marked background: hook recall, API and MCP recall, interactive reflect, and synchronous API mental-model refreshes.
- **BACKGROUND** is every task run through `MemoryEngine.execute_task`: the worker poller, and `SyncTaskBackend` running inline in embedded mode. Every recall or reflect nested inside those tasks inherits the lane through a `contextvars.ContextVar` (`rerank_background_lane` / `background_lane()` in `reranker_routing.py`).
  - Today the tasks that rerank are `refresh_mental_model`, through the reflect agent's recall tools.
  - Consolidation's own recall uses `reranking="interleave"`, which does not call the cross-encoder, but its task is still marked, so it is covered if that changes.
  - Retain does no recall today. If it ever does, it is background too.
  - Webhook, import/export, clone and graph maintenance never rerank, so marking them is harmless.

## Environment variables

Members are still defined by the existing `HINDSIGHT_API_RERANKER_*` (member 0) and `HINDSIGHT_API_RERANKER_<n>_*` (member n) variables. The variables below are parsed and validated in `reranker_routing.py`. They are read through `reranker_routing_env()`, the one fork hook in `config.py` (the end of the file), so the `config.py` env-reader guard still holds and `reranker_routing.py` never touches `os.environ`.

| Variable | Values | Unset / default |
|---|---|---|
| `HINDSIGHT_API_RERANKER_FOREGROUND_ORDER` | comma-separated member indexes, e.g. `0,1,2,3` | natural order `0..N-1` (upstream behaviour) |
| `HINDSIGHT_API_RERANKER_BACKGROUND_ORDER` | comma-separated member indexes, e.g. `2,0,1,3` | natural order `0..N-1` (upstream behaviour) |
| `HINDSIGHT_API_RERANKER_MAX_INFLIGHT` (member 0) and `HINDSIGHT_API_RERANKER_<n>_MAX_INFLIGHT` | integer >= 0 | `0` = unlimited; a blank value counts as unset |
| `HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_SECONDS` (member 0) and `HINDSIGHT_API_RERANKER_<n>_BACKGROUND_QUEUE_SECONDS` | finite number of seconds, 0 or more (no 7-day ceiling; queue waits use monotonic time only) | `0` = no waiting (v1 behaviour) |
| `HINDSIGHT_API_RERANKER_BACKGROUND_QUEUE_MAX` (member 0) and `HINDSIGHT_API_RERANKER_<n>_BACKGROUND_QUEUE_MAX` | integer >= 0 | unset = `2` when QUEUE_SECONDS > 0, else `0`; an explicit `0` disables waiting |
| `HINDSIGHT_API_RERANKER_<n>_FOREGROUND_MAX_CANDIDATES` | integer >= 0 (`n` >= 0; member 0 is `HINDSIGHT_API_RERANKER_0_FOREGROUND_MAX_CANDIDATES`) | `0` = no member cap |
| `HINDSIGHT_API_RERANKER_<n>_BACKGROUND_MAX_CANDIDATES` | integer >= 0 (`n` >= 0; member 0 is `HINDSIGHT_API_RERANKER_0_BACKGROUND_MAX_CANDIDATES`) | `0` = no member cap |
| `HINDSIGHT_API_RERANKER_COOLDOWN_ENABLED` | `true` / `1` / `yes` | `false` |
| `HINDSIGHT_API_RERANKER_COOLDOWN_DEFAULT_SECONDS` | finite float, 0 to 604800 (7 days) | `30` |
| `HINDSIGHT_API_RERANKER_COOLDOWN_PER_MINUTE_SECONDS` | finite float, 0 to 604800 (7 days) | `60` |
| `HINDSIGHT_API_RERANKER_COOLDOWN_MAX_SECONDS` | finite float, 0 to 604800 (7 days) | `86400` (caps cooldowns derived from response headers) |
| `HINDSIGHT_API_RERANKER_FOREGROUND_MAX_CANDIDATES` | integer >= 0 | `0` = existing budget-tier logic; > 0 replaces it for **every** foreground budget |
| `HINDSIGHT_API_RERANKER_BACKGROUND_MAX_CANDIDATES` | integer >= 0 | `0` = existing budget-tier logic; > 0 replaces it for the background lane |

**Caution:** the unnumbered `HINDSIGHT_API_RERANKER_FOREGROUND_MAX_CANDIDATES` is a lane-wide cap. A value > 0 replaces the budget tiers for EVERY foreground budget, LOW included, so hook recall gets it too. The recommended setup leaves it unset and uses upstream `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_LOW` and `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID` instead. The lane-wide `BACKGROUND_MAX_CANDIDATES` has the same effect on the background lane.

Naming: per-member caps are always numbered, including member 0 (`..._0_...`), because the unnumbered member-0 name is the lane-wide cap above. Upstream ignores `_0_` names, since it scans members from `_1_PROVIDER`, so the numbered member-0 names are only read by this fork.

Order rules:

- An order is a subset of member indexes. Members not listed are never used by that lane.
- Startup fails with an error naming the variable for: a set but blank value, an empty entry such as `0,,1`, a non-integer, an index outside `0..N-1`, or a duplicate.
- Startup logs a warning naming any member that is in neither lane's order.

In-flight caps:

- A member at its cap is skipped immediately in the foreground lane. In the background lane it may queue instead, see [Background queue](#background-queue). Either way, the next member in the lane order is tried when the request is not served.
- The count is per process and shared by both lanes. The API and worker processes count separately.
- The count is released on success, error, timeout and cancellation.

## Background queue

- Only the background lane waits, and only when the member is at its `MAX_INFLIGHT` cap, its `BACKGROUND_QUEUE_SECONDS` is > 0 and its effective `BACKGROUND_QUEUE_MAX` is > 0. Otherwise the request skips at once, as in v1.
- A waiting request is served in strict FIFO order. When its turn comes it takes the slot and proceeds. If `QUEUE_SECONDS` passes first, it skips with `queue_timeout`.
- A background request that arrives while other background requests are queued for the same member joins the back of the queue, even if a slot is momentarily free. A new background request never overtakes queued ones.
- Foreground never waits. A foreground request may take a momentarily free slot ahead of queued background requests. Its alternative is the next member or rrf, so latency matters more than order for it.
- A queue is full when `QUEUE_MAX` requests are already waiting for that member. A background request that finds it full skips with `queue_full` and moves to the next member.
- A member that arrives in cooldown is skipped at once with `cooldown`. A cooldown that starts while a request is waiting (from any thread) ends the wait with `cooldown` on the waiter's next poll.
- Mechanism: a waiter polls under the same threading lock that guards the in-flight counters and sleeps with `asyncio.sleep` on its own event loop, starting at 0.05 s and doubling up to 0.5 s. Polling is used because the engine runs in more than one event loop (the sync wrappers use `asyncio.run`, and the worker has its own loop), so no future or event is shared between loops or threads. Wake-up latency is at most 0.5 s, which is small against a rerank that takes tens of seconds.
- Cancellation and timeouts remove the waiter from the queue and never leak a slot or a queue position. This includes the reflect wall timeout. A queue entry not polled for 60 s is dropped as abandoned, with one WARNING, so a waiter whose loop died cannot block the head of the queue.
- The queue is per process, like `MAX_INFLIGHT`.

### Does an 1800 s queue fit the refresh timeouts?

- A mental-model refresh is one reflect. Upstream bounds the whole reflect with `asyncio.wait_for` around the agent, using `HINDSIGHT_API_REFLECT_WALL_TIMEOUT`.
- The worker poller applies the same value as the wall ceiling for the whole `refresh_mental_model` task. That covers the reflect iterations plus the delta ops and the write, not only the reflect.
- The default is 300 s. At the default, any long wait cancels the refresh, and a 1800 s wait certainly does. The refresh is cancelled cleanly and fails.
- The reference deployment sets 3600 s (the source default is 300). One maximal 1800 s wait leaves about 1800 s for the request, the other LLM calls and the write. That is not enough when the LLM work is slow: there, refresh delta ops measured p50 565 s and mean 931 s, with a per-call LLM timeout of 900 s. A maximal wait plus two slow LLM calls already exceeds 3600 s, and the task is then cancelled as a wall timeout and retried. Two maximal waits in one refresh do not fit either: one iteration's tool calls run in parallel with `asyncio.gather`, and the iterations repeat, so a refresh can wait more than once.
- Rule (a sizing heuristic, not a guarantee): `BACKGROUND_QUEUE_SECONDS <= REFLECT_WALL_TIMEOUT / 2`. Keep `BACKGROUND_QUEUE_MAX` small enough that real waits stay far below `QUEUE_SECONDS`. Realistic FIFO waits are about k x 36 s at N=300 (k = queue position), so 1800 s is a backstop, not the normal limit.
- The LLM timeouts (`LLM_TIMEOUT`, `MENTAL_MODEL_REFRESH_LLM_TIMEOUT`) apply per LLM call. They do not include rerank waits, but they count toward the same wall ceiling.

## Worker-slot trade-off

- A waiting request keeps its task's worker slot. It takes no new slot, but it holds the one it has for longer.
- `BACKGROUND_QUEUE_MAX` bounds the background rerank requests waiting for one member, not worker-slot occupancy. With `MAX_INFLIGHT=1`, up to `1 + QUEUE_MAX` such requests are in flight or waiting on that member, and each one parks the refresh task that issued it. So up to `1 + QUEUE_MAX` refresh tasks can be parked on the local member path. Other refreshes still occupy slots doing fallback reranks or LLM work.
- Slot floors in the reference deployment (`WORKER_MAX_SLOTS=12`): consolidation 2, retain 1 and refresh 1. These are floors (`config.py`, around L946-951). The slot scheduler guarantees only the retain and consolidation floors. A running refresh consumes its refresh floor, and refresh can use that floor plus all 8 shared slots, so `QUEUE_MAX` does not bound refresh occupancy. Refresh LLM concurrency is 8, and one refresh's gathered tool calls can hold several queue positions at once.
- A background request beyond `QUEUE_MAX` spills to the next member in the background order.

## Per-member candidate cap

- The serving member scores only the first `<cap>` candidates it is sent, where the cap is `<n>_<LANE>_MAX_CANDIDATES` for the current lane. The member receives the full list, and the chain sends it only the head.
- The order is the order recall passes the candidates in. That is RRF fusion order: `reciprocal_rank_fusion` sorts by RRF score, and `trim_merged_candidates` keeps that order, or re-sorts by boosted RRF score when it trims.
- The candidates past the cap come back after all scored candidates, in that same order, with reranker score 0.0. They use upstream's `RerankTimeoutError` partial-result channel in `search/reranking.py`. The fork raises a subclass of it, `RerankerMemberCapReached`, after the member loop, so a member's own timeout is still a failure that fails over.
- Guarantees:
  - The tail scores are always finite (0.0). Member scores pass two checks: a member that returns ±inf fails over (`RuntimeError`), and NaN goes to upstream's sanitizer, which turns it into 0.0. The fork does not sanitize NaN itself.
  - The head is normalized as if it were scored alone.
  - Without `RECALL_STRATEGY_BOOSTS`, the tail stays after the head. Combined scoring is multiplicative, so the tail's 0.0 stays 0.0, and both sorts are stable.
  - A head candidate whose normalized score is exactly 0.0 ties with the tail and stays ahead only by the stable sort. That happens for a literal 0.0, or for a logit below about -709 that underflows to exactly 0.0 in `sigmoid` (`np.exp` underflow).
- Floors: the tail fails any `min_scores.reranker` > 0, like upstream's timeout tail. A floor of exactly 0 passes it. `min_scores.final` applies to weight 0.0. Typesafe (in `RANK_SCORE_PROVIDERS`) skips the reranker floor anyway.
- With `RECALL_STRATEGY_BOOSTS` set, neither head-before-tail nor tail input order is guaranteed. `apply_post_rerank_boost` (`recall_boost.py`) adds up to 0.5 per arm with no cap, and memory_engine re-sorts by weight afterwards. So a boosted tail candidate can pass any head candidate, and tail candidates can reorder among themselves. This is the same as upstream's `RerankTimeoutError` tail. Enforcing the order would need a hook after the sort in `memory_engine.py` (the hot recall path). The fork deliberately adds none, because each hook has to survive every rebase. A fork-local negative or sub-minimum tail score cannot fix it either: a negative tail flips upstream's [0, 1] passthrough check to sigmoid and breaks "head normalized as if scored alone", and additive boosts exceed any [0, 1] score anyway. The reference deployment sets no strategy boosts.
- rrf members ignore caps, because their scores are constant and the list is only reseeded from RRF.
- A skipped or failed member passes the full list to the next member, which applies its own cap.
- Each capped call logs a WARNING from `hindsight_api.engine.search.reranking`: `Reranking: reranker member <i> (<provider>) is capped at <K>/<N> candidates in the <lane> lane; ranking the remainder by RRF order`.

## Cooldown rules

- Only HTTP 429 starts a cooldown. Other errors do not.
- With `COOLDOWN_ENABLED=true`, a 429 from a member also stops that member's own retry loop, so it costs one request instead of a retry budget.
- With the cooldown off, upstream retry and failover behaviour applies exactly.
- Duration precedence:
  1. The `Retry-After` or `X-RateLimit-Reset` response header. `X-RateLimit-Reset` may be epoch milliseconds, epoch seconds, or a delta in seconds. The delay is clamped to `COOLDOWN_MAX_SECONDS`.
  2. OpenRouter's error-body metadata `X-RateLimit-Reset`.
  3. A body containing `per-day` cools down until the next 00:00 UTC, which is OpenRouter's daily reset.
  4. A body containing `per-min` cools down for `COOLDOWN_PER_MINUTE_SECONDS` (60 s).
  5. Anything else cools down for `COOLDOWN_DEFAULT_SECONDS` (30 s).
- A member in cooldown is skipped without a call until the cooldown ends.
- TEI: `RemoteTEICrossEncoder` re-raises its upstream 429 as a `RuntimeError` after its own retry loop. The cooldown logic follows the `__cause__`/`__context__` chain (at most two levels) to find the HTTP status, so a TEI 429 starts a cooldown once TEI's internal retries are spent. TEI's own retries are not shortened, and no TEI member is deployed today.
- Values for the three cooldown durations must be finite and at most 7 days; `inf`, `nan` and huge values fail at startup.
- If every non-rrf member is skipped or failed, the rrf member serves when it is in the order. With no rrf in the order, the last error is re-raised. If nothing was tried because every member was skipped, `RerankerUnavailableError` is raised.

## Logging

One INFO line per rerank call, from logger `hindsight_api.engine.reranker_routing`:

```
Reranker routing: lane=<lane> served=<idx>:<provider>|none candidates=N sent=K latency_ms=M queue_wait_ms=W skipped=[2:queue_full, 0:cooldown, 1:inflight, 3:failed(HTTP 429)]
```

- `candidates` is the number of candidates the rerank call received. `sent` is the number the serving member scored (less than `candidates` when a member cap applied, and 0 when `served=none`).
- `queue_wait_ms` is the time spent waiting in background queues, summed over the members tried.
- Skip reasons: `inflight` (at the in-flight cap, foreground or queue disabled), `queue_full`, `queue_timeout`, `cooldown` (in cooldown on arrival or while waiting), and `failed(<label>)`, where the label is the HTTP status or the exception class, never the message.
- The level is WARNING instead of INFO when any member failed or none served.
- State changes are logged as well: a cooldown being entered (with its until-time), a cooldown ending, a stale queue entry being dropped, and a startup summary line `Reranker routing: foreground order=... background order=... max_inflight=... cooldown=... queue=... member_caps=...`.
- Startup warnings, none of which fail startup:
  - `BACKGROUND_QUEUE_SECONDS` is set on a member whose `MAX_INFLIGHT` is unset or 0, so nothing ever waits.
  - `BACKGROUND_QUEUE_MAX=0` disables waiting on a member whose `BACKGROUND_QUEUE_SECONDS` is set.
  - `BACKGROUND_QUEUE_MAX` is set on a member whose `BACKGROUND_QUEUE_SECONDS` is unset or 0, so it is ignored.
  - `BACKGROUND_QUEUE_SECONDS` is set on a member that is not in `BACKGROUND_ORDER`, so it is never used.
  - A `<n>_<LANE>_MAX_CANDIDATES` is set on a member whose provider is `rrf`, so it is ignored.
  - A member in neither lane's order.
  - With a single member: the order, in-flight, queue, cooldown and per-member cap vars are ignored (the lane-wide caps still apply).
- No query text and no document text is ever logged.
- The API and worker processes each log their own state.

## Reference deployment example

The block below also depends on `HINDSIGHT_API_REFLECT_WALL_TIMEOUT=3600`. The source default is 300, and the queue sizing rule depends on it.

Members:

- 0 and 1: OpenRouter free reranker accounts.
- 2: a local bge-reranker-v2-m3 on one Apple Silicon GPU (llama.cpp `llama-server --rerank`, Cohere-compatible `/v1/rerank`). It processes one request at a time.
- 3: rrf.

Environment:

```
HINDSIGHT_API_REFLECT_WALL_TIMEOUT=3600
HINDSIGHT_API_RERANKER_FOREGROUND_ORDER=0,1,2,3
HINDSIGHT_API_RERANKER_BACKGROUND_ORDER=2,0,1,3
HINDSIGHT_API_RERANKER_MAX_CANDIDATES_LOW=50
HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID=300
HINDSIGHT_API_RERANKER_BACKGROUND_MAX_CANDIDATES=300
HINDSIGHT_API_RERANKER_2_MAX_INFLIGHT=1
HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_SECONDS=1800
HINDSIGHT_API_RERANKER_2_BACKGROUND_QUEUE_MAX=4
HINDSIGHT_API_RERANKER_2_FOREGROUND_MAX_CANDIDATES=100
HINDSIGHT_API_RERANKER_2_COHERE_TIMEOUT=120
HINDSIGHT_API_RERANKER_COOLDOWN_ENABLED=true
```

The local member's timeout must sit well above its measured p95 (about 36 s at N=300 with real-length documents); the upstream-style 30 s is too low.

What each line achieves:

- `FOREGROUND_ORDER=0,1,2,3` and `BACKGROUND_ORDER=2,0,1,3`: foreground tries the OpenRouter accounts first and the local member after them. Background tries the local member first, then falls back to OpenRouter.
- `MAX_CANDIDATES_LOW=50`: the hook (LOW budget) stays at 50 candidates.
- `MAX_CANDIDATES_MID=300`: foreground MID sends 300 candidates. OpenRouter (members 0 and 1) scores all of them.
- `2_FOREGROUND_MAX_CANDIDATES=100`: if foreground falls to the local member, the local member scores only the first 100 candidates in RRF order, and the rest come back as the tail.
- `BACKGROUND_MAX_CANDIDATES=300`: background sends 300 candidates to the local member first, and a background request that finds the local member busy queues for it.
- `FOREGROUND_MAX_CANDIDATES` (lane-wide) is deliberately unset. Setting it would replace the LOW and MID tiers above.
- `2_MAX_INFLIGHT=1`: the local member runs one request at a time.
- `2_BACKGROUND_QUEUE_SECONDS=1800` and `2_BACKGROUND_QUEUE_MAX=4`: up to four background requests wait for the local member, each for at most 30 minutes.
- `2_COHERE_TIMEOUT=120`: a single request to the local member may take up to 120 s. The measured p95 at N=300 is about 36 s.
- `COOLDOWN_ENABLED=true`: a 429 from OpenRouter stops that account's retries and cools it down.

Why `BACKGROUND_QUEUE_MAX=4`:

- FIFO wait at queue position k is about k service times. With about 36 s p95 per request at N=300, four waiters take about 4 x 36 s, roughly 2.5 min at p95.
- A wedged local member is bounded by 4 x 120 s x 2 attempts (`RERANKER_MAX_RETRIES=1`, so two attempts per member) = 16 min, which is below 1800 s. The 1800 s queue timeout is a backstop, not the normal limit.
- Up to 5 refresh tasks (1 in flight and 4 waiting) can be parked on the local member path at once. That bounds local-member requests, not slots: other refreshes can still occupy shared slots, and refresh can use its floor plus all 8 shared slots. Only the retain (1) and consolidation (2) floors are guaranteed by the slot scheduler.
- Raise toward 7 (refresh LLM concurrency of 8, minus one) if `skipped=[2:queue_full]` in the logs shows background requests spilling to OpenRouter and retain or consolidation latency is acceptable.

The 1800 s queue meets the sizing rule for the reference refresh budget (`REFLECT_WALL_TIMEOUT=3600`, so 1800 <= 3600 / 2). That rule is a heuristic, and the wall ceiling covers the whole refresh, not only the reflect. See [Does an 1800 s queue fit the refresh timeouts?](#does-an-1800-s-queue-fit-the-refresh-timeouts).

Adding an OpenRouter account: define the `HINDSIGHT_API_RERANKER_4_*` member variables. Members must stay contiguous, so if rrf must stay last, move rrf to the new last index. Then put the new index in the orders. No code change is needed.

## Maintaining the patch (rebase/cherry-pick)

Files owned by the fork:

- `hindsight-api-slim/hindsight_api/engine/reranker_routing.py`
- `hindsight-api-slim/tests/test_reranker_routing.py`
- `FORK-RERANK-ROUTING.md` (this file)

Upstream hooks, all marked `# fork: rerank-routing` (find them with `grep -rn "fork: rerank-routing" hindsight-api-slim/hindsight_api`). Version 2 added no upstream hooks. The per-member queue and cap logic lives entirely in `reranker_routing.py`:

- `engine/cross_encoder.py`, end of `create_cross_encoder_from_env`: a local import plus `validate_routing_env(len(chain))` on the single-member path, and a local import plus `return routed_chain([...])` in place of `return MultiCrossEncoder([...])`.
- `engine/memory_engine.py`, above `_resolve_reranker_max_candidates`: one import line and the `@lane_max_candidates_override` decorator.
- `engine/memory_engine.py`, above `@_bind_bank_id("task_dict", key="bank_id")` on `execute_task`: `@rerank_background_lane`.
- `config.py`, end of file: `reranker_routing_env()`, the raw `HINDSIGHT_API_RERANKER_*` mapping that `reranker_routing.py` parses. It is appended after `clear_config_cache`, so it is a pure append. Keep it there when resolving conflicts.

Conflict recipe: take upstream's side of the conflicted hunk, then re-insert the marked lines at the same anchors. The "upstream hooks" tests in `test_reranker_routing.py` fail if any hook is lost.

Per-member caps depend on upstream's partial-result handling in `CrossEncoderReranker.rerank` (`search/reranking.py`): it catches `RerankTimeoutError`, treats `exc.scores[i] is None` as unscored and uses 0.0 for it, appends unscored candidates after the scored ones in input order, and reads `_served_provider` after its `except` block. After each rebase, diff `search/reranking.py` (around the `RerankTimeoutError` handler) and `cross_encoder.RerankTimeoutError` against the previous upstream tag. The end-to-end cap tests in `test_reranker_routing.py` fail if that behaviour drifts.

`RoutedMultiCrossEncoder` overrides `MultiCrossEncoder._predict`. After each rebase, diff upstream `MultiCrossEncoder._predict` for new behaviour (for example, a circuit breaker) and port it into the override.

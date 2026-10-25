# Fork patches: how to write one that survives rebases

Branch `patched` is upstream's release tag plus a few fork commits, re-applied by cherry-pick onto each new upstream tag. Every line a patch puts inside an upstream file is a future merge conflict, so the rules below are about keeping that surface small and easy to re-apply.

Each feature patch has its own `FORK-<NAME>.md` at the repo root: what it does, its env vars, why it is designed that way, and how to maintain it. Bug-fix patches are small enough to explain in their commit message.

## Branches

Work lands in three steps: a `feature/<name>` branch off `develop`, a PR into `develop`, then `develop` into `patched` when it is released. `.github/workflows/fork-ci.yml` runs on every PR into `develop`, `patched` or a `ci-base/<tag>` branch and on every push to `develop`; merge only when it is green. It runs the checks that need no provider credentials (the api-slim suite without real-LLM markers, `hindsight-system-tests`, `ty`, lint with no diff, the OpenAPI spec, the control plane), so the real-LLM checks in "Before tagging" still run locally. Before every push run `scripts/fork/check.sh`: it runs CI's `ty`, lint and OpenAPI checks, the fork's own api-slim tests and CI's control-plane job in about four minutes, so a format, lint, i18n or build error never costs a CI round. The full api-slim suite and the system tests run in CI only. Upstream's `test.yml` only runs on PRs to `main` and does not run here. On an upstream rebase, rebuild `develop` from the new `patched`.

## Rules

- **New logic goes in new files.** Use a fork-owned module (e.g. `engine/reranker_routing.py`) and a fork-owned test file. Do not edit upstream test files; add fork tests next to them.
- **Upstream files get hooks only.** A hook is a few lines that call into the fork module, each marked `# fork: <name>` so `grep -rn "fork: <name>"` finds every one. Prefer a decorator or a wrapped return value over editing a function body. No reformatting, reordering, renaming or drive-by cleanup in upstream files.
- **Place hooks where upstream is quiet.** Before choosing an anchor, run `git diff <base tag> origin/main -- <file>` and avoid regions upstream has touched since the base. Avoid hot, frequently changed paths (recall scoring is one); if correctness there would need a hook, prefer documenting a limitation.
- **Check upstream first.** Search upstream main for an existing or in-progress equivalent (settings, circuit breakers, per-operation options). Use it if it exists.
- **Env vars go through `config.py`.** `tests/test_config_is_the_only_env_reader.py` forbids reading `os.environ` anywhere else. Append one accessor at the end of `config.py` that returns the raw vars, and parse them in the fork module.
- **Import the fork module locally inside upstream hooks** when a module-level import would either create an import cycle or expose fork classes to upstream tests that scan a module's `vars()` (e.g. `TestEveryRemoteRerankerRetries` in `tests/test_reranker_retry.py` flags any backend class in `cross_encoder`'s namespace that overrides `predict`).
- **Inert by default.** With none of the patch's env vars set, behaviour must be byte-for-byte upstream (return upstream's own object, not a subclass that "behaves the same"). Validate the new vars at startup so a typo fails the build instead of being ignored.
- **Test that the hooks exist.** Add fork tests that fail if a hook disappears after a conflict resolution (e.g. `test_config_keeps_the_routing_env_hook`).
- **TypeScript integrations follow the same rules.** The hook comment is `// fork: <name>`, the logic goes in a fork-owned `fork-<name>.ts` and the tests in `fork-<name>.test.ts` next to it (e.g. `hindsight-integrations/coding-agents/src/core/fork-survey-env.ts`). `scripts/fork/hooked-integrations.sh` lists every integration package whose non-test sources carry such a hook, and `check.sh` and Fork CI run that package's typecheck and full test suite, which nothing else runs.
- **No host information on the fork.** It is public: no host names, internal addresses or ops-repo paths in code, tests or docs. Describe a deployment as "the reference deployment".

## Before tagging

1. Run the patch's tests, the upstream tests around the hooked code, `ruff check`, `ruff format --check` and `ty check` on the fork module.
2. Prove the patch applies to current upstream: in a throwaway detached worktree of `origin/main`, `git apply --3way --check` the full diff, then remove the worktree.
3. Have the change reviewed (a second model or Codex). For a rebase, give the reviewers the output of `scripts/fork/review-bundle.sh`: the fork commits whose patch changed, the rebase's own commits, the upstream changes under them, and every fork test the rebase deleted or removed tests from. Reject review fixes that would add hooks to hot upstream paths; record the trade-off in the patch's `FORK-<NAME>.md` instead.
   A pure rebase skips this step when every cherry-pick applied without conflicts, each patch's diff is identical to the one last reviewed, and the upstream files each `FORK-<NAME>.md` names as dependencies are unchanged between the two tags. A resolved conflict or a changed dependency needs a review.

## Rebasing onto a new upstream tag

First read each `FORK-<NAME>.md` "Maintaining the patch" section: it lists the upstream behaviour the patch depends on. Diff those files between the two tags (`git diff --stat <old-tag> <new-tag> -- <files>`). Where they changed, run that patch's tests as soon as it is picked, even when the pick is clean: in v1.1.0 the attempt-usage and session-header patches picked cleanly and still broke, as their docs had predicted.

Then run `scripts/fork/rebase-onto.sh start <new-tag>` in a clean worktree. It queues the fork commits of `patched`, cherry-picks them onto the tag and stops at each conflict. The tag it checks out has no `scripts/fork/`, so it runs the rest from a copy and prints that copy's path. For a conflict, take upstream's side of the hunk, re-insert the marked hook lines at the same anchors, then run the copy with `continue`; commits you make at the stop (the resolution with a message saying what changed, plus any fix an earlier fork commit needs) count as that pick. A commit upstream already contains is `drop "<why>"`. The run ends by checking that every fork commit was picked, came out empty or was dropped with a reason. A hook added to fit upstream's new structure gets a `# fork: <name>` marker and an entry in its `FORK-<NAME>.md`.

A fork commit whose fix upstream has since made: take upstream's code and drop the fork's, but go through the fork tests that no longer pass one by one before deleting any. A test of a private helper that upstream replaced can go. A test whose behaviour upstream's version lacks guards something the fork still wants, so re-add that behaviour as a marked hook instead. In v1.1.0, upstream's #5693 usage parsers no longer read the usage of a failed Codex response or a Claude Code error result, and no longer read the cache-write TTL breakdown. Deleting those tests silently dropped the behaviour, and only review caught it.

Order the work so each slow run happens once:

1. While resolving conflicts, run only the fork tests and the upstream tests around each hook. `check.sh` runs exactly that set: the fork's test files plus the upstream test files that import or patch a module carrying a hook (`scripts/fork/hook-tests.py`).
2. Once they pass, run `scripts/fork/check.sh`, push the branch and open a PR into `ci-base/<new-tag>` (a branch at the upstream tag), so Fork CI runs the full api-slim suite and `hindsight-system-tests` on it.
3. Start the review ("Before tagging" step 3) at the same time; it finds regressions the suites do not, such as a table the fork reads gaining new kinds of rows.
4. Each review fix gets its targeted tests, `check.sh` and a CI run.

A laptop run of the full api-slim suite takes about 30 minutes against CI's 15 and adds failures CI does not have (S3 tests need Docker; fact-extraction tests need a real LLM; a few tests are timing-flaky under load). Only a failure CI also reports needs a rerun on the bare upstream tag. To run a suite locally anyway, use `uv sync --frozen --all-extras --no-extra local-llm --index-strategy unsafe-best-match` on macOS (llama-cpp-python does not build; a plain `uv sync` leaves out pg0) and keep the worktree free of a `.env` (a deployment's reranker settings leak into the routing tests' single-member cases). Leave the worktree alone while a suite runs in it.

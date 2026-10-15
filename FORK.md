# Fork patches: how to write one that survives rebases

Branch `patched` is upstream's release tag plus a few fork commits, re-applied by cherry-pick onto each new upstream tag. Every line a patch puts inside an upstream file is a future merge conflict, so the rules below are about keeping that surface small and easy to re-apply.

Each feature patch has its own `FORK-<NAME>.md` at the repo root: what it does, its env vars, why it is designed that way, and how to maintain it. Bug-fix patches are small enough to explain in their commit message.

## Branches

Work lands in three steps: a `feature/<name>` branch off `develop`, a PR into `develop`, then `develop` into `patched` when it is released. `.github/workflows/fork-ci.yml` runs on every PR into `develop` or `patched` and on every push to `develop`; merge only when it is green. It runs the checks that need no provider credentials (the api-slim suite without real-LLM markers, `ty`, lint with no diff, the OpenAPI spec, the control plane), so the real-LLM checks in "Before tagging" still run locally. Upstream's `test.yml` only runs on PRs to `main` and does not run here. On an upstream rebase, rebuild `develop` from the new `patched`.

## Rules

- **New logic goes in new files.** Use a fork-owned module (e.g. `engine/reranker_routing.py`) and a fork-owned test file. Do not edit upstream test files; add fork tests next to them.
- **Upstream files get hooks only.** A hook is a few lines that call into the fork module, each marked `# fork: <name>` so `grep -rn "fork: <name>"` finds every one. Prefer a decorator or a wrapped return value over editing a function body. No reformatting, reordering, renaming or drive-by cleanup in upstream files.
- **Place hooks where upstream is quiet.** Before choosing an anchor, run `git diff <base tag> origin/main -- <file>` and avoid regions upstream has touched since the base. Avoid hot, frequently changed paths (recall scoring is one); if correctness there would need a hook, prefer documenting a limitation.
- **Check upstream first.** Search upstream main for an existing or in-progress equivalent (settings, circuit breakers, per-operation options). Use it if it exists.
- **Env vars go through `config.py`.** `tests/test_config_is_the_only_env_reader.py` forbids reading `os.environ` anywhere else. Append one accessor at the end of `config.py` that returns the raw vars, and parse them in the fork module.
- **Import the fork module locally inside upstream hooks** when a module-level import would either create an import cycle or expose fork classes to upstream tests that scan a module's `vars()` (e.g. `TestEveryRemoteRerankerRetries` in `tests/test_reranker_retry.py` flags any backend class in `cross_encoder`'s namespace that overrides `predict`).
- **Inert by default.** With none of the patch's env vars set, behaviour must be byte-for-byte upstream (return upstream's own object, not a subclass that "behaves the same"). Validate the new vars at startup so a typo fails the build instead of being ignored.
- **Test that the hooks exist.** Add fork tests that fail if a hook disappears after a conflict resolution (e.g. `test_config_keeps_the_routing_env_hook`).
- **No host information on the fork.** It is public: no host names, internal addresses or ops-repo paths in code, tests or docs. Describe a deployment as "the reference deployment".

## Before tagging

1. Run the patch's tests, the upstream tests around the hooked code, `ruff check`, `ruff format --check` and `ty check` on the fork module.
2. Prove the patch applies to current upstream: in a throwaway detached worktree of `origin/main`, `git apply --3way --check` the full diff, then remove the worktree.
3. Have the change reviewed (a second model or Codex). Reject review fixes that would add hooks to hot upstream paths; record the trade-off in the patch's `FORK-<NAME>.md` instead.
   A pure rebase skips this step when every cherry-pick applied without conflicts, each patch's diff is identical to the one last reviewed, and the upstream files each `FORK-<NAME>.md` names as dependencies are unchanged between the two tags. A resolved conflict or a changed dependency needs a review.

## Rebasing onto a new upstream tag

Cherry-pick the fork commits onto the new tag in a clean worktree. For a conflict, take upstream's side of the hunk and re-insert the marked hook lines at the same anchors. Then read each `FORK-<NAME>.md` "Maintaining the patch" section: it lists the upstream behaviour the patch depends on and must be re-diffed.

Then test the rebased tree as a whole, not only the patch: the api-slim suite as CI's `test-api` job runs it (`uv sync --frozen --all-extras --index-strategy unsafe-best-match`, then `uv run pytest tests -m "not hs_llm_mat and not hs_llm_core"`), and `hindsight-system-tests`. Some tests fail on a laptop for environment reasons (no provider credentials, missing services), so rerun every failure on the bare upstream tag: only a failure the bare tag does not have belongs to the patch. On macOS add `--no-extra local-llm` (llama-cpp-python does not build); a plain `uv sync` leaves out pg0 and every database test errors. Keep the worktree free of a `.env`: a deployment's reranker settings leak into the routing tests' single-member cases.

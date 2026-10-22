#!/bin/bash
# Local gate before pushing a fork branch: what Fork CI's typecheck-and-lint job checks (ty, lint with
# no diff left, OpenAPI spec and generated docs up to date), the fork's own api-slim tests, and Fork CI's
# control-plane job when the control plane changed since <upstream-tag>. Fork CI runs the full suite and the
# system tests; this catches the mistakes that would otherwise cost a CI round.
#
# Usage: scripts/fork/check.sh [<upstream-tag>]
#   <upstream-tag>  base of the fork commits; its test files that the fork added or changed are run.
#                   Default: the newest upstream release tag reachable from HEAD.
# Needs a clean tree: lint.sh rewrites files, and any diff it leaves is the failure.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"

base=${1:-$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' --exclude '*-patch*' HEAD)}
if [ -n "$(git status --porcelain)" ]; then
  echo "check.sh: commit or stash first; lint.sh's diff is the failure signal" >&2
  exit 2
fi

# llama-cpp-python does not build on macOS (FORK.md); CI installs every extra on Linux.
extras=(--all-extras)
[ "$(uname -s)" = Darwin ] && extras+=(--no-extra local-llm)

step() { printf '\n== %s\n' "$1"; }
fail() { echo "FAILED: $1" >&2; exit 1; }

step "dependencies"
[ -d node_modules ] || npm ci --no-audit --no-fund --loglevel=error
(cd hindsight-api-slim && uv sync --frozen "${extras[@]}" --index-strategy unsafe-best-match --quiet)
(cd hindsight-dev && uv sync --frozen --index-strategy unsafe-best-match --quiet)

step "ty check"
(cd hindsight-api-slim && uv run ty check) || fail "ty check"

step "lint (must leave no diff)"
# Every integration, as CI lints them: locally lint.sh only lints integrations with uncommitted changes.
LINT_ALL_INTEGRATIONS=1 ./scripts/hooks/lint.sh || fail "lint.sh"
[ -z "$(git status --porcelain)" ] || { git status --short; fail "lint.sh changed the files above; review and commit them"; }

step "OpenAPI spec"
(cd hindsight-dev && uv run generate-openapi >/dev/null) || fail "generate-openapi"
git diff --exit-code --stat -- hindsight-docs/static/openapi.json \
  || fail "openapi.json is stale; commit the regenerated spec"

step "generated files (upstream's verify-generated-files, minus the Rust client)"
# generate-clients needs Rust; the fork's routes stay out of OpenAPI, so an unchanged openapi.json
# above already means the clients are unchanged.
./scripts/generate-bank-template-schema.sh >/dev/null || fail "generate-bank-template-schema"
./scripts/generate-docs-skill.sh >/dev/null || fail "generate-docs-skill"
[ -z "$(git status --porcelain)" ] || { git status --short; fail "generated files are stale or new; commit the files above"; }

step "fork rebase tooling"
for t in rebase-onto review-bundle; do bash "scripts/fork/$t.test.sh" >/dev/null || fail "scripts/fork/$t.test.sh"; done

step "fork tests (added or changed since $base)"
tests=()  # macOS ships bash 3.2: no mapfile
while IFS= read -r f; do tests+=("${f#hindsight-api-slim/}"); done \
  < <(git diff --name-only --diff-filter=AM "$base" HEAD -- 'hindsight-api-slim/tests/test_*.py')
if [ "${#tests[@]}" -eq 0 ]; then
  echo "none"
else
  (cd hindsight-api-slim && uv run pytest "${tests[@]}" -q -m "not hs_llm_mat and not hs_llm_core") \
    || fail "fork tests"
fi

step "control plane (Fork CI's control-plane job)"
if git diff --quiet "$base" HEAD -- hindsight-control-plane hindsight-clients/typescript; then
  echo "unchanged since $base"
else
  # The control plane imports the TypeScript SDK from its build output (gitignored dist/).
  npm run build --workspace=hindsight-clients/typescript >/dev/null || fail "TypeScript SDK build"
  # TZ=UTC as on the CI runner: upstream's relative-time tests assume a UTC day.
  TZ=UTC npm test --workspace=hindsight-control-plane >/dev/null || fail "control-plane tests (npm test --workspace=hindsight-control-plane)"
  npm run i18n:check --workspace=hindsight-control-plane >/dev/null || fail "i18n parity"
  (cd hindsight-control-plane && npx --yes knip@5 --no-progress --include files,dependencies,unlisted) || fail "knip"
  npm run build --workspace=hindsight-control-plane >/dev/null || fail "control-plane build"
  [ -z "$(git status --porcelain)" ] || { git status --short; fail "the control-plane steps changed the files above"; }
fi

printf '\ncheck.sh: OK (%d fork test files). Full suite and system tests run in Fork CI.\n' "${#tests[@]}"

#!/bin/bash
# Tests for review-bundle.sh in a throwaway repo: a pair whose rebase removed one file's patch, a message-only
# change, upstream changes to a path only the old side touched, and removed tests (multi-line it.each, a
# colocated test file).
# Usage: bash scripts/fork/review-bundle.test.sh
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/review-bundle.sh"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
failures=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; failures=$((failures + 1)); }

cd "$tmp"; git init -q -b main; git config user.email t@t; git config user.name t; git config commit.gpgsign false
for f in a b c d; do printf 'base\n' > $f.txt; done
git add -A; git commit -qm base; git tag v1.0.0
git checkout -q -b fork-old
for f in a b c; do printf 'fork\n' > $f.txt; done; git commit -qam "fork: abc"; abc=$(git rev-parse HEAD)
mkdir -p pkg/tests pkg/src
cat > pkg/tests/x.test.ts <<'TS'
it("keeps this", () => {});
it.each([
  ["a", 1],
  ["b", 2],
])("guards fork behaviour", (k, v) => {});
test.each([1])(
  "title on the next line",
  (n) => {},
);
TS
printf 'it("colocated", () => {});\n' > pkg/src/y.test.ts
git add -A; git commit -qm "fork: tests"; tests=$(git rev-parse HEAD)
printf 'fork\n' > d.txt; git commit -qam "fork: d"; d=$(git rev-parse HEAD)
oldhead=$(git rev-parse HEAD)

git checkout -q main; printf 'upstream\n' > a.txt; git commit -qam upstream; git tag v2.0.0
git checkout -q --detach v2.0.0
git cherry-pick -x "$abc" >/dev/null 2>&1 || true
git checkout -q v2.0.0 -- a.txt; git add -A; GIT_EDITOR=true git cherry-pick --continue >/dev/null   # a.txt patch gone
git cherry-pick -x "$tests" >/dev/null
printf 'it("keeps this", () => {});\n' > pkg/tests/x.test.ts; git rm -q pkg/src/y.test.ts
git commit -qam "fork: tests, trimmed"
git cherry-pick -x "$d" >/dev/null                                                                        # message only
bash "$script" v1.0.0 "$oldhead" v2.0.0 HEAD "$tmp/out" >/dev/null

grep -q 'fork: abc' "$tmp/out/range-diff.txt" && ok "pair that lost a file's patch is listed" || bad "pair that lost a file's patch"
grep -q 'fork: d' "$tmp/out/range-diff.txt" && bad "message-only change is listed" || ok "message-only change is left out"
grep -q '^+++ b/a.txt' "$tmp/out/upstream-touching.diff" && ok "old-side path a.txt in upstream-touching" || bad "old-side path a.txt"
grep -q 'removed: each: guards fork behaviour' "$tmp/out/fork-tests.txt" && ok "multi-line it.each removal named" || bad "multi-line it.each"
grep -q 'removed: each: title on the next line' "$tmp/out/fork-tests.txt" && ok "title-on-next-line removal named" || bad "title on next line"
grep -q 'removed: keeps this' "$tmp/out/fork-tests.txt" && bad "kept test reported removed" || ok "kept test not reported"
grep -q '^DELETED  pkg/src/y.test.ts' "$tmp/out/fork-tests.txt" && ok "colocated test file deletion" || bad "colocated test file"

[ "$failures" -eq 0 ] && echo "all passed" || { cat "$tmp/out/README.txt" "$tmp/out/fork-tests.txt"; echo "$failures failed"; exit 1; }

#!/bin/bash
# Tests for rebase-onto.sh in throwaway repos: a clean run, a conflict resolved and continued, a commit made by
# hand during a stop, a drop, a pick refused by local changes (which must stop, never count as empty), and a refused checkout.
# Usage: bash scripts/fork/rebase-onto.test.sh
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/rebase-onto.sh"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
failures=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; failures=$((failures + 1)); }

# v1.0.0 -- (fork) a, b, c on branch fork-ref; v2.0.0 = v1.0.0 + upstream change to a.txt (conflicts with fork commit "a").
make_repo() {
  local d="$tmp/$1"; mkdir -p "$d"; cd "$d"
  git init -q -b main; git config user.email t@t; git config user.name t; git config commit.gpgsign false
  printf 'base\n' > a.txt; printf 'base\n' > b.txt; printf 'base\n' > c.txt
  git add -A; git commit -qm base; git tag -a v1.0.0 -m v1.0.0
  git checkout -q -b fork-ref
  printf 'fork a\n' > a.txt; git commit -qam "fork: a"
  printf 'fork b\n' > b.txt; git commit -qam "fork: b"
  printf 'fork c\n' > c.txt; git commit -qam "fork: c"
  git checkout -q main
  if [ "$2" = conflict ]; then printf 'upstream a\n' > a.txt; else printf 'upstream d\n' > d.txt; git add d.txt; fi
  git commit -qam upstream; git tag -a v2.0.0 -m v2.0.0
}
copy() { echo "$(git rev-parse --absolute-git-dir)/fork-rebase/rebase-onto.sh"; }

# 1. Clean run: every commit picked; must end with done and a .last ledger.
make_repo clean noconflict
if bash "$script" start v2.0.0 fork-ref >"$tmp/out1" 2>&1 && grep -q '^done:' "$tmp/out1" \
   && [ "$(grep -c ' picked ' "$(git rev-parse --absolute-git-dir)/fork-rebase.last/ledger")" = 3 ]; then
  ok "clean run ends done with 3 picked"
else bad "clean run"; cat "$tmp/out1"; fi

# 2. Conflict resolved and continued: recorded picked, run completes.
make_repo cont conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
printf 'resolved\n' > a.txt; git add a.txt
if bash "$(copy)" continue >"$tmp/out2" 2>&1 && grep -q '^done:' "$tmp/out2" \
   && grep -q ' picked ' "$(git rev-parse --absolute-git-dir)/fork-rebase.last/ledger" \
   && [ "$(git log --format=%s v2.0.0..HEAD | wc -l | tr -d ' ')" = 3 ]; then
  ok "conflict + continue picks all 3"
else bad "conflict + continue"; cat "$tmp/out2"; fi

# 3. Resolution committed by hand during the stop: recorded as picked (not empty), message kept.
make_repo manual conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
printf 'resolved\n' > a.txt; git add a.txt; git commit -qm "fork: a, re-anchored on v2"
bash "$(copy)" continue >"$tmp/out3" 2>&1 || true
ledger3="$(git rev-parse --absolute-git-dir)/fork-rebase.last/ledger"
if [ -f "$ledger3" ] && ! grep -q ' empty' "$ledger3" && [ "$(grep -c ' picked ' "$ledger3")" = 3 ] \
   && git log --format=%s v2.0.0..HEAD | grep -q 're-anchored'; then
  ok "hand-committed resolution counts as picked"
else bad "hand-committed resolution"; cat "$tmp/out3" "$ledger3" 2>/dev/null; fi

# 3b. Two commits made during one stop (a fix to an earlier fork commit, then the resolution): one pick.
make_repo twocommits conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
printf 'resolved\n' > a.txt; git add a.txt; git commit -qm "fork: a, re-anchored on v2"
printf 'extra\n' > e.txt; git add e.txt; git commit -qm "fork: fix an earlier commit for v2"
bash "$(copy)" continue >"$tmp/out3b" 2>&1 || true
ledger3b="$(git rev-parse --absolute-git-dir)/fork-rebase.last/ledger"
if [ -f "$ledger3b" ] && grep -q ' picked [0-9a-f]* (+1 more commits)$' "$ledger3b" \
   && [ "$(git log --format=%s v2.0.0..HEAD | wc -l | tr -d ' ')" = 4 ]; then
  ok "two commits at one stop are one pick"
else bad "two commits at one stop"; cat "$tmp/out3b" "$ledger3b" 2>/dev/null; fi

# 3c. HEAD moved back during the stop (abort, then reset): must stop, not record lost commits as picked.
make_repo moveback conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
git cherry-pick --abort; git reset -q --hard HEAD~1
status=0; bash "$(copy)" continue >"$tmp/out3c" 2>&1 || status=$?
if [ "$status" -ne 0 ] && grep -q 'no longer contains the stop point' "$tmp/out3c" \
   && [ ! -e "$(git rev-parse --absolute-git-dir)/fork-rebase.last" ]; then
  ok "HEAD moved back during a stop is refused"
else bad "HEAD moved back (status $status)"; cat "$tmp/out3c"; fi

# 4. Drop: recorded with its reason, run completes with the other two picked.
make_repo drop conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
if bash "$(copy)" drop "upstream has it" >"$tmp/out4" 2>&1 \
   && grep -q 'dropped upstream has it' "$(git rev-parse --absolute-git-dir)/fork-rebase.last/ledger" \
   && [ "$(git log --format=%s v2.0.0..HEAD | wc -l | tr -d ' ')" = 2 ]; then
  ok "drop records the reason"
else bad "drop"; cat "$tmp/out4"; fi

# 5. A pick refused by local changes must stop with the error, never be recorded as empty.
make_repo refused conflict
bash "$script" start v2.0.0 fork-ref >/dev/null 2>&1 || true
printf 'resolved\n' > a.txt; git add a.txt
printf 'stray edit\n' > b.txt            # unstaged change to the file the next fork commit touches
status=0; bash "$(copy)" continue >"$tmp/out5" 2>&1 || status=$?
if [ "$status" -ne 0 ] && ! grep -q "empty" "$(git rev-parse --absolute-git-dir)/fork-rebase/ledger" \
   && grep -q 'did not start' "$tmp/out5"; then
  ok "refused pick stops instead of counting as empty"
else bad "refused pick (status $status)"; cat "$tmp/out5"; fi

# 6. Checkout of the new tag refused (a lock): start must leave no run behind to "continue" on the old base.
make_repo locked noconflict
git checkout -q fork-ref
touch "$(git rev-parse --absolute-git-dir)/index.lock"
status=0; bash "$script" start v2.0.0 fork-ref >"$tmp/out6" 2>&1 || status=$?
rm -f "$(git rev-parse --absolute-git-dir)/index.lock"
if [ "$status" -ne 0 ] && [ ! -e "$(git rev-parse --absolute-git-dir)/fork-rebase" ]; then
  ok "refused checkout leaves no run in progress"
else bad "refused checkout (status $status)"; cat "$tmp/out6"; fi

[ "$failures" -eq 0 ] && echo "all passed" || { echo "$failures failed"; exit 1; }

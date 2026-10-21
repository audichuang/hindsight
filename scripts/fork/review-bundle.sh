#!/bin/bash
# Reviewer inputs for a rebase of the fork onto a new upstream tag (FORK.md "Before tagging", step 3).
# Usage: scripts/fork/review-bundle.sh <old-tag> <old-fork-head> <new-tag> <new-head> <out-dir>
# Writes:
#   range-diff.txt          fork commits whose patch changed in the rebase, and the dropped and added ones
#   extra-commits.diff      commits that exist only on the new side (the rebase's own fixes)
#   upstream-touching.diff  upstream <old-tag>..<new-tag> changes to the files the changed fork commits touch
#   fork-tests.txt          every fork test file: deleted, changed (with the test functions that disappeared) or same
#   README.txt
set -euo pipefail
old=$1 oldhead=$2 new=$3 head=$4 out=$5
mkdir -p "$out"

git range-diff --no-color "$old..$oldhead" "$new..$head" > "$out/range-diff.full.txt"
# Keep a "!" entry only when its patch changed: `cherry-pick -x` alone changes every commit message.
awk '
  function flush() { if (head != "" && (kind != "!" || code)) printf "%s", buf; buf = ""; head = ""; code = 0 }
  /^ *([0-9]+|-): +[0-9a-f-]+ [=!<>] / { flush(); head = $0; kind = $3; if (kind == "=") head = ""; else buf = $0 "\n"; next }
  head != "" { buf = buf $0 "\n"; if ($0 ~ /^    @@ / && $0 !~ /@@ (Metadata|Commit message)/) code = 1 }
  END { flush() }
' "$out/range-diff.full.txt" > "$out/range-diff.txt"

changed=$(awk '/^ *([0-9]+|-): +[0-9a-f-]+ [!>] / { print $5 }' "$out/range-diff.txt" | grep -E '^[0-9a-f]{7,}$' || true)
extra=$(awk '/^ *-: +-+ > / { print $5 }' "$out/range-diff.full.txt" || true)
: > "$out/extra-commits.diff"
for c in $extra; do git show --stat --patch "$c" >> "$out/extra-commits.diff"; done

files=$(for c in $changed; do git show --name-only --format= "$c"; done | sort -u)
if [ -n "$files" ]; then
  # shellcheck disable=SC2086
  git diff "$old" "$new" -- $files > "$out/upstream-touching.diff"
else
  : > "$out/upstream-touching.diff"
fi

# A test the rebase deleted or relaxed may have guarded fork behaviour that upstream's replacement lacks.
{
  echo "# fork test files (added or changed by the fork on $old) and what the rebase did to them"
  git diff --name-only --diff-filter=AM "$old" "$oldhead" -- '*/tests/*' '*/tests/**' | sort -u | while read -r f; do
    if ! git cat-file -e "$head:$f" 2>/dev/null; then
      echo "DELETED  $f"
    elif [ "$(git rev-parse "$oldhead:$f")" = "$(git rev-parse "$head:$f")" ]; then
      echo "same     $f"
    else
      gone=$(comm -23 <(git show "$oldhead:$f" | grep -oE '^(async )?def test_[A-Za-z0-9_]+' | sed 's/^async //' | sort -u) \
                      <(git show "$head:$f" | grep -oE '^(async )?def test_[A-Za-z0-9_]+' | sed 's/^async //' | sort -u))
      echo "changed  $f"
      [ -z "$gone" ] || echo "$gone" | sed 's/^def /    removed: /'
    fi
  done
} > "$out/fork-tests.txt"

cat > "$out/README.txt" <<EOF
old=$old old-head=$oldhead new=$new new-head=$head
changed-or-added fork commits: $(echo "$changed" | grep -c . || true)
rebase-only extra commits: $(echo "$extra" | grep -c . || true)
fork test files deleted: $(grep -c '^DELETED' "$out/fork-tests.txt" || true), changed: $(grep -c '^changed' "$out/fork-tests.txt" || true)
range-diff.txt          fork commits whose content changed in the rebase (old vs new patch)
extra-commits.diff      commits the rebase added (re-anchors, fixes)
upstream-touching.diff  upstream changes to files those fork commits touch
fork-tests.txt          for each DELETED file and each removed test: did it guard fork-owned behaviour?
EOF
wc -l "$out"/*.txt "$out"/*.diff

#!/bin/bash
# Reviewer inputs for a rebase of the fork onto a new upstream tag (FORK.md "Before tagging", step 3).
# Usage: scripts/fork/review-bundle.sh <old-tag> <old-fork-head> <new-tag> <new-head> <out-dir>
# Writes:
#   range-diff.txt          fork commits whose patch (not just message) changed, and the dropped and added ones
#   extra-commits.diff      commits that exist only on the new side (the rebase's own fixes)
#   upstream-touching.diff  upstream <old-tag>..<new-tag> changes to the files any changed, dropped or added commit touches (old and new side)
#   upstream/NN-<subject>.diff  the same, split per fork commit: what upstream changed under that one patch
#   fork-tests.txt          every fork test file: deleted, changed (with the test functions that disappeared) or same
#   README.txt
set -euo pipefail
old=$1 oldhead=$2 new=$3 head=$4 out=$5
mkdir -p "$out"

git range-diff --no-color "$old..$oldhead" "$new..$head" > "$out/range-diff.full.txt"

# Pair old and new commits with range-diff, then judge each pair by patch-id: `cherry-pick -x` changes every
# message, and range-diff can file a removed hunk under "@@ Metadata", so its own hunk headings cannot tell.
# --verbatim: whitespace counts (a dedent in Python changes behaviour); --binary: binary changes count too.
pid() { git show --binary --format= "$1" | git patch-id --verbatim | cut -d' ' -f1; }
paths() { git show --name-only --format= "$1"; }
# per <commit>...: upstream's changes to the files these commits touch, one file per fork commit.
per() {
  local f
  f="$out/upstream/$(printf '%02d' "$n_per")-$(git log -1 --format=%s "${!#}" | tr -c 'A-Za-z0-9' '-' | cut -c1-60).diff"
  # shellcheck disable=SC2046
  git diff "$old" "$new" -- $(for c in "$@"; do paths "$c"; done | sort -u) > "$f"
  [ -s "$f" ] || rm -f "$f"
  n_per=$((n_per + 1))
}
rm -rf "$out/upstream"; mkdir -p "$out/upstream"
: > "$out/range-diff.txt"; : > "$out/extra-commits.diff"; : > "$out/paths"; : > "$out/extra-list"
n_changed=0 n_extra=0 n_per=1
while read -r line; do
  set -- $line
  case "$3" in
    '!')
      [ "$(pid "$2")" = "$(pid "$5")" ] && continue   # message-only change
      git range-diff --no-color "$2^!" "$5^!" >> "$out/range-diff.txt"
      { paths "$2"; paths "$5"; } >> "$out/paths"; per "$2" "$5"; n_changed=$((n_changed + 1)) ;;
    '<')   # dropped: upstream's replacement is what to check
      echo "$line" >> "$out/range-diff.txt"; paths "$2" >> "$out/paths"; per "$2" ;;
    '>')
      echo "$line" >> "$out/range-diff.txt"; git show --stat --patch "$5" >> "$out/extra-commits.diff"
      paths "$5" >> "$out/paths"; per "$5"; git log -1 --format='  %h %s' "$5" >> "$out/extra-list"
      n_extra=$((n_extra + 1)) ;;
  esac
done < <(git range-diff --no-color -s "$old..$oldhead" "$new..$head")
files=$(sort -u "$out/paths"); rm -f "$out/paths"
if [ -n "$files" ]; then
  # shellcheck disable=SC2086
  git diff "$old" "$new" -- $files > "$out/upstream-touching.diff"
else
  : > "$out/upstream-touching.diff"
fi

# Test names in a file: Python test functions and methods, and the titles of it()/test() cases (TS/JS),
# parameterised it.each(table)("title") included; a call may span lines.
test_names() {
  perl -0777 -ne '
    print "$1\n" while /^[ \t]*(?:async[ \t]+)?def[ \t]+(test_\w+)/mg;
    print "$2\n" while /\b(?:it|test)(?:\.(?:only|skip|todo|concurrent|fails))*\s*\(\s*([\x27"`])((?:\\.|(?!\1).)*)\1/sg;
    print "each: $3\n" while /\b(?:it|test)(?:\.\w+)*\.each\s*(\((?:[^()]++|(?1))*\)|`[^`]*`)\s*\(\s*([\x27"`])((?:\\.|(?!\2).)*)\2/sg;
  ' | sort -u
}

# A test the rebase deleted or relaxed may have guarded fork behaviour that upstream's replacement lacks.
{
  echo "# fork test files (added or changed by the fork on $old) and what the rebase did to them"
  git diff --name-only --diff-filter=AM "$old" "$oldhead" \
    -- '*/tests/*' '*.test.*' '*.spec.*' '*/test_*.py' | sort -u | while read -r f; do
    if ! git cat-file -e "$head:$f" 2>/dev/null; then
      echo "DELETED  $f"
    elif [ "$(git rev-parse "$oldhead:$f")" = "$(git rev-parse "$head:$f")" ]; then
      echo "same     $f"
    else
      gone=$(comm -23 <(git show "$oldhead:$f" | test_names) <(git show "$head:$f" | test_names))
      echo "changed  $f"
      [ -z "$gone" ] || echo "$gone" | sed 's/^/    removed: /'
    fi
  done
} > "$out/fork-tests.txt"

cat > "$out/README.txt" <<EOF
old=$old old-head=$oldhead new=$new new-head=$head
fork commits whose patch changed: $n_changed
rebase-only extra commits: $n_extra
$(cat "$out/extra-list")
fork test files deleted: $(grep -c '^DELETED' "$out/fork-tests.txt" || true), changed: $(grep -c '^changed' "$out/fork-tests.txt" || true)
range-diff.txt          fork commits whose content changed in the rebase (old vs new patch)
extra-commits.diff      commits the rebase added (re-anchors, fixes)
upstream-touching.diff  upstream changes to files those fork commits touch
upstream/               the same split per fork commit (NN-<subject>.diff), for reading one patch at a time
Commit messages travel with range-diff.txt and extra-commits.diff, so a rebased commit's message
(which says how its conflict was resolved) is part of what reviewers see.
fork-tests.txt          for each DELETED file and each removed test: did it guard fork-owned behaviour?
EOF
rm -f "$out/extra-list"
wc -l "$out"/*.txt "$out"/*.diff | tail -1; ls "$out/upstream" | wc -l | sed 's/^ */upstream\/ files: /'

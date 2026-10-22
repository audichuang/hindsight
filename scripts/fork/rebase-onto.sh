#!/bin/bash
# Re-apply the fork commits onto a new upstream tag, one cherry-pick at a time (FORK.md "Rebasing onto a
# new upstream tag"), keeping the queue in a state file so a conflict never leaves you counting commits.
#
#   scripts/fork/rebase-onto.sh start <new-tag> [<fork-ref>]   detach HEAD at <new-tag>, queue every non-merge
#                                                              commit of <old-tag>..<fork-ref> (default fork/patched;
#                                                              <old-tag> = the release tag it is based on), pick
#   scripts/fork/rebase-onto.sh continue                       after resolving a conflict and `git add`: commit it, pick on
#   scripts/fork/rebase-onto.sh drop "<reason>"                the stopped commit is already in upstream: discard it, pick on
#   scripts/fork/rebase-onto.sh status                         queue position and the ledger so far
#
# `start` copies this script into the state directory and prints that path: the upstream tag it checks out
# has no scripts/fork/, so use the copy for continue / drop / status until the run ends.
#
# Every queued commit ends in the ledger as picked, empty (applied but changed nothing: upstream has it) or
# dropped with a reason; the run ends with a check that none is missing. To say in the message what a
# resolution changed (so the history does not claim the old fix), commit it yourself with `git commit`,
# then `continue`: the commits made since the stop are recorded as the pick. Make more than one there
# when the resolution also needs a fix to an earlier fork commit's code or tests.
set -euo pipefail
state="$(cd "$(git rev-parse --git-dir)" && pwd)/fork-rebase"
self="$state/rebase-onto.sh"
queue="$state/queue" ledger="$state/ledger" pos="$state/pos"

die() { echo "rebase-onto: $*" >&2; exit 2; }
stopped() { # $1 = the commit that stopped; leaves the conflict for the user
  git rev-parse HEAD > "$state/stop-head"
  echo "STOPPED at $1 $(git log -1 --format=%s "$1")"
  git diff --name-only --diff-filter=U | sed 's/^/  conflict: /'
  echo "Resolve and \`git add\`, then: bash $self continue   (or: bash $self drop \"<why upstream has it>\")"
  exit 1
}

pick_rest() {
  local n total c
  n=$(cat "$pos"); total=$(wc -l < "$queue" | tr -d ' ')
  while [ "$n" -lt "$total" ]; do
    c=$(sed -n "$((n + 1))p" "$queue")
    if git cherry-pick -x "$c" >/dev/null 2>"$state/err"; then
      echo "$c picked $(git rev-parse --short HEAD)" >> "$ledger"
    elif [ "$(git rev-parse -q --verify CHERRY_PICK_HEAD || true)" != "$c" ]; then
      # Refused before applying anything (local changes in the way, a bad ref, a lock): not "empty".
      cat "$state/err" >&2
      die "cherry-pick of $c did not start; fix the cause above and run continue"
    elif [ -n "$(git diff --name-only --diff-filter=U)" ]; then
      stopped "$c"
    elif git diff --cached --quiet; then
      git cherry-pick --skip >/dev/null || die "could not skip the empty pick of $c"
      echo "$c empty" >> "$ledger"
    else
      stopped "$c"
    fi
    n=$((n + 1)); echo "$n" > "$pos"
  done
  verify
}

verify() {
  local missing=0 c
  while read -r c; do
    grep -q "^$c " "$ledger" || { echo "missing from ledger: $c"; missing=1; }
  done < "$queue"
  echo
  awk '{print $2}' "$ledger" | sort | uniq -c
  { grep -E ' (empty|dropped)' "$ledger" || true; } | while read -r c how rest; do
    echo "  $how $c $(git log -1 --format=%s "$c") ${rest:+- $rest}"
  done
  [ "$missing" -eq 0 ] || die "ledger is incomplete"
  git merge-base --is-ancestor "$(cat "$state/base")" HEAD || die "HEAD is not on the new tag $(cat "$state/base")"
  rm -rf "$state.last" && mv "$state" "$state.last"
  echo "done: every queued commit is accounted for. HEAD=$(git rev-parse --short HEAD); ledger: $state.last/ledger"
}

cmd=${1:-}; shift || true
case "$cmd" in
  start)
    new=${1:?usage: start <new-tag> [<fork-ref>]}; from=${2:-fork/patched}
    [ -e "$state" ] && die "a run is in progress ($state); finish it or remove the directory"
    [ -z "$(git status --porcelain)" ] || die "tree is dirty"
    old=$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' --exclude '*-patch*' "$from")
    base=$(git rev-parse --verify "$new^{commit}")
    mkdir -p "$state"; echo "$base" > "$state/base"
    cp "$0" "$self"
    git log --reverse --no-merges --format=%H "$old..$from" > "$queue"
    : > "$ledger"; echo 0 > "$pos"
    printf 'from=%s old=%s new=%s commits=%s\n' "$from" "$old" "$new" "$(wc -l < "$queue" | tr -d ' ')" | tee "$state/info"
    git checkout -q --detach "$new" || { rm -rf "$state"; die "could not check out $new; nothing was started"; }
    pick_rest
    ;;
  continue)
    [ -e "$state" ] || die "no run in progress"
    n=$(cat "$pos"); c=$(sed -n "$((n + 1))p" "$queue")
    [ -z "$(git diff --name-only --diff-filter=U)" ] || die "unresolved conflicts remain"
    if [ -f "$state/stop-head" ] && [ "$(git rev-parse HEAD)" != "$(cat "$state/stop-head")" ]; then
      # Committed by hand since the stop: only on top of it, never by moving HEAD back or sideways
      # (a reset or an amend there would silently lose picked commits), and with nothing left over.
      git merge-base --is-ancestor "$(cat "$state/stop-head")" HEAD \
        || die "HEAD no longer contains the stop point $(cat "$state/stop-head"); restore it (git reflog) and commit on top"
      [ -z "$(git status --porcelain --untracked-files=no)" ] || die "uncommitted changes remain; commit or discard them first"
      git cherry-pick --quit >/dev/null 2>&1 || true
      extra=$(($(git rev-list --count "$(cat "$state/stop-head")..HEAD") - 1))
      note=""; [ "$extra" -eq 0 ] || note=" (+$extra more commits)"
      echo "$c picked $(git rev-parse --short HEAD)$note" >> "$ledger"
    elif [ "$(git rev-parse -q --verify CHERRY_PICK_HEAD || true)" != "$c" ]; then
      rm -f "$state/stop-head"; pick_rest   # the pick never started, or is not this commit's: retry it
      exit 0
    elif git diff --cached --quiet; then
      git cherry-pick --skip >/dev/null || die "could not skip the empty pick of $c"
      echo "$c empty" >> "$ledger"
    else
      GIT_EDITOR=true git cherry-pick --continue >/dev/null
      echo "$c picked $(git rev-parse --short HEAD)" >> "$ledger"
    fi
    rm -f "$state/stop-head"
    echo $((n + 1)) > "$pos"
    pick_rest
    ;;
  drop)
    reason=${1:?usage: drop "<reason>"}
    [ -e "$state" ] || die "no run in progress"
    n=$(cat "$pos"); c=$(sed -n "$((n + 1))p" "$queue")
    git cherry-pick --abort >/dev/null 2>&1 || git reset -q --hard HEAD
    echo "$c dropped $reason" >> "$ledger"
    rm -f "$state/stop-head"
    echo $((n + 1)) > "$pos"
    pick_rest
    ;;
  status)
    [ -e "$state" ] || die "no run in progress"
    cat "$state/info"; echo "position $(cat "$pos") of $(wc -l < "$queue" | tr -d ' ')"; cat "$ledger"
    ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac

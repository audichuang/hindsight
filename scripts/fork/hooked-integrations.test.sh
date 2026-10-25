#!/bin/bash
# Tests for hooked-integrations.sh in a throwaway repo: a hook in a source file selects its package once;
# a marker only in a test file, a plain mention of "fork:" and a hook outside src/ do not.
# Usage: bash scripts/fork/hooked-integrations.test.sh
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/hooked-integrations.sh"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cd "$tmp"; git init -q
f() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; }
f hindsight-integrations/a/src/core/x.ts      'const e = strip(env); // fork: demo'
f hindsight-integrations/a/src/y.ts           'import { strip } from "./s"; // fork: demo'
f hindsight-integrations/b/src/x.test.ts      '// fork: demo — only a test mentions it'
f hindsight-integrations/c/src/x.ts           '/** Kilo is an opencode fork: same plugin array */'
f hindsight-integrations/d/scripts/x.ts       'run(); // fork: demo'
f hindsight-integrations/e/src/x.ts           'plain();'
git add -A
got=$(bash "$script" | tr '\n' ' ')
want="hindsight-integrations/a "
if [ "$got" = "$want" ]; then echo "all passed"; else echo "FAIL"; echo " got: $got"; echo "want: $want"; exit 1; fi
# Nothing hooked: prints nothing and still succeeds.
git rm -qf hindsight-integrations/a/src/core/x.ts hindsight-integrations/a/src/y.ts
[ -z "$(bash "$script")" ] || { echo "FAIL: expected no packages"; exit 1; }

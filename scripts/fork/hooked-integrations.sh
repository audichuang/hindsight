#!/bin/bash
# Print each hindsight-integrations/<pkg> whose non-test sources carry a `// fork: <name>` hook, one per
# line. A hook there is exercised only by that package's own test suite, which neither the api-slim tests
# nor Fork CI's other jobs run: check.sh and Fork CI's integration job run the full suite of every package
# listed here (coding-agents' takes about ten seconds).
# Usage: bash scripts/fork/hooked-integrations.sh   (from the repo root)
set -euo pipefail
git grep -l -E '// fork: [a-z0-9-]+' -- 'hindsight-integrations/*/src/*.ts' ':(exclude)*.test.ts' \
  | cut -d/ -f1-2 | sort -u || true

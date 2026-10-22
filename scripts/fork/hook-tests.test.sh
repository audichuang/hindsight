#!/bin/bash
# Tests for hook-tests.py in a throwaway api dir: direct imports of a hooked module, a mock.patch string,
# and dependents through root-relative, nested-sibling, parent-relative and `from . import` imports and
# through a helper module or a package __init__.py; a hub module and an unrelated test stay out.
# Usage: bash scripts/fork/hook-tests.test.sh
set -euo pipefail
script="$(cd "$(dirname "$0")" && pwd)/hook-tests.py"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
api="$tmp/api"; mkdir -p "$api/hindsight_api/engine" "$api/tests/sub"
cd "$api"; git init -q
printf 'x = 1  # fork: demo\n' > hindsight_api/engine/hooked.py
printf 'y = 1  # fork: demo\n' > hindsight_api/config.py                     # a hub: never selects
printf 'z = 1\n' > hindsight_api/engine/plain.py
t() { printf '%s\n' "$2" > "tests/$1"; }
t test_direct.py   'from hindsight_api.engine.hooked import x'
t test_patch.py    'TARGET = "hindsight_api.engine.hooked.x"'
t test_hub.py      'from hindsight_api import config'
t test_plain.py    'from hindsight_api.engine import plain'
t test_rel.py      'from .test_direct import x'
t test_dotimport.py 'from . import test_direct'
t helper.py        'from hindsight_api.engine import hooked'
t test_viahelper.py 'from tests.helper import hooked'
t sub/test_base.py 'import hindsight_api.engine.hooked'
t sub/test_sibling.py 'from .test_base import hindsight_api'
t sub/test_parent.py 'from ..test_patch import TARGET'
mkdir -p tests/pkg
t pkg/__init__.py   'from .helper2 import hooked'
t pkg/helper2.py    'from hindsight_api.engine import hooked'
t pkg/test_init.py  'import sys'                                         # affected only through pkg/__init__.py
git add -A
got=$(python3 -I "$script" "$api" | sort | tr '\n' ' ')
want="tests/pkg/test_init.py tests/sub/test_base.py tests/sub/test_parent.py tests/sub/test_sibling.py tests/test_direct.py tests/test_dotimport.py tests/test_patch.py tests/test_rel.py tests/test_viahelper.py "
if [ "$got" = "$want" ]; then echo "all passed"; else echo "FAIL"; echo " got: $got"; echo "want: $want"; exit 1; fi

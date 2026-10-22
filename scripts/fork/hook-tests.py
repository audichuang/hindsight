"""Print the api-slim test files that exercise a module carrying a `# fork:` hook.

A hook can break an upstream test's fakes without touching any fork test (v1.1.0: the claude-code
error-usage hook read `message.usage`, which the fakes in test_content_policy_refusal_permanent.py lack,
and only CI caught it). A test file counts when it imports a hooked module in any form (`import a.b`,
`from a import b`, multi-line, aliased) or names one in a string, such as a `mock.patch` target.
Modules nearly every test imports are left to CI, or this would be the whole suite.

Usage: python3 scripts/fork/hook-tests.py   (from the repo root; prints paths relative to hindsight-api-slim)
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path("hindsight-api-slim")
HUBS = {
    "hindsight_api.config",
    "hindsight_api.main",
    "hindsight_api.engine.memory_engine",
    "hindsight_api.api.http",
    "hindsight_api.worker.main",
}


def hooked_modules() -> set[str]:
    out = subprocess.run(
        ["git", "grep", "-l", "# fork:", "--", "hindsight_api/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode not in (0, 1):  # 1: no match
        sys.exit(f"git grep failed: {out.stderr.strip()}")
    mods = set()
    for path in out.stdout.split():
        mod = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        if mod not in HUBS:
            mods.add(mod)
    return mods


def names_in(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("hindsight_api."):
            names.add(node.value)
    return names


def main() -> None:
    mods = hooked_modules()
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            names = names_in(ast.parse(test.read_text(encoding="utf-8")))
        except SyntaxError as exc:
            sys.exit(f"{test}: {exc}")
        # A name hits when it is the module or anything inside it (a function, a patch target).
        if any(n == m or n.startswith(m + ".") for n in names for m in mods):
            print(test.relative_to(ROOT))


if __name__ == "__main__":
    main()

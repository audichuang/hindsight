"""Print the api-slim test files that exercise a module carrying a `# fork:` hook.

A hook can break an upstream test's fakes without touching any fork test (v1.1.0: the claude-code
error-usage hook read `message.usage`, which the fakes in test_content_policy_refusal_permanent.py lack,
and only CI caught it). A test file counts when it imports a hooked module in any form (`import a.b`,
`from a import b`, multi-line, aliased) or names one in a string, such as a `mock.patch` target; so does
a test file that imports, directly or through helper modules, a selected one (shared fakes and stubs).
Modules nearly every test imports are left to CI, or this would be the whole suite.

Usage: python3 scripts/fork/hook-tests.py [<api-dir>]   (from the repo root; default hindsight-api-slim;
prints paths relative to it)
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "hindsight-api-slim")
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


def names_in(tree: ast.AST, package: str) -> set[str]:
    """Dotted names a file imports or quotes; `package` resolves its relative imports (`tests.sub`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:  # `from ..x import y`: drop level-1 trailing parts of the importing file's package
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            if node.module:
                names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("hindsight_api."):
            names.add(node.value)
    return names


def main() -> None:
    mods = hooked_modules()
    # Every module under tests/ is a node, helpers included (fakes, stubs): a test that reaches a hooked
    # module only through one is selected through it. conftest.py is left out: every test loads it.
    files: dict[str, Path] = {}
    imports: dict[str, set[str]] = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if path.name == "conftest.py":
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        mod = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
        package = mod if rel.name == "__init__" else ".".join(rel.parts[:-1])
        try:
            imports[mod] = names_in(ast.parse(path.read_text(encoding="utf-8")), package)
        except SyntaxError as exc:
            sys.exit(f"{path}: {exc}")
        files[mod] = path

    def hits(names: set[str], targets: set[str]) -> bool:
        # A name hits when it is the module or anything inside it (a function, a patch target).
        return any(n == m or n.startswith(m + ".") for n in names for m in targets)

    selected = {m for m, names in imports.items() if hits(names, mods)}
    while True:  # then every module importing a selected one, until nothing changes
        more = {m for m, names in imports.items() if m not in selected and hits(names, selected)}
        if not more:
            break
        selected |= more
    for m in sorted(selected):
        if files[m].name.startswith("test_"):
            print(files[m].relative_to(ROOT))


if __name__ == "__main__":
    main()

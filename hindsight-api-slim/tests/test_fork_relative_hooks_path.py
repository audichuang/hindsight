# fork: relative-hooks-path
"""setup-hooks.sh must keep writing a relative core.hooksPath (commit message of the fix).

core.hooksPath lives in the shared .git/config, and npm ci runs setup-hooks.sh as `prepare`
from whichever worktree it is in. An absolute path pointed every checkout's hooks at that
worktree, and removing it silently disabled them. These checks fail if a rebase restores
upstream's absolute form.
"""

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_setup_hooks_writes_a_relative_path():
    source = (_SCRIPTS / "setup-hooks.sh").read_text()
    assert "git config core.hooksPath .githooks  # fork: relative-hooks-path" in source
    assert "$REPO_ROOT/.githooks" not in source


def test_dev_setup_checks_for_the_relative_path():
    source = (_SCRIPTS / "dev" / "setup.sh").read_text()
    assert '= ".githooks" ]; then  # fork: relative-hooks-path' in source

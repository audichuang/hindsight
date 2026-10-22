#!/bin/bash
# Setup git hooks for the repository

set -e

echo "Setting up git hooks..."
# Relative, so each worktree runs its own .githooks. core.hooksPath lives in the shared .git/config;
# an absolute path written from a worktree (npm ci runs this as `prepare`) hijacked every checkout's
# hooks, and removing that worktree silently disabled them.
git config core.hooksPath .githooks  # fork: relative-hooks-path
echo "Git hooks configured to use .githooks directory"
echo "Done!"

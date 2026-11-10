#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$root"
[[ -f .githooks/pre-push ]] || { echo 'Missing versioned pre-push hook.' >&2; exit 1; }
count=$(git worktree list --porcelain | awk '/^worktree / {n++} END {print n+0}')
[[ "$count" == 1 ]] || { echo 'Multiple worktrees share Git configuration. Install in a dedicated clone.' >&2; exit 1; }
existing=$(git config --get core.hooksPath || true)
[[ -z "$existing" || "$existing" == .githooks ]] || { echo "Existing hooksPath ($existing) preserved. Integrate manually." >&2; exit 1; }
old=$(git rev-parse --git-path hooks/pre-push)
[[ -n "$existing" || ! -f "$old" ]] || { echo 'Existing pre-push hook preserved. Integrate manually.' >&2; exit 1; }
git config --local core.hooksPath .githooks
echo 'Pre-push hook installed for this clone.'

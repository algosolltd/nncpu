#!/usr/bin/env bash
set -euo pipefail
mode=${1:-quick}
install=false
[[ ${2:-} != --install ]] || install=true
export GIT_TERMINAL_PROMPT=0
export NEXT_TELEMETRY_DISABLED=1 CHECKPOINT_DISABLE=1 PRISMA_HIDE_UPDATE_MESSAGE=1
export PUPPETEER_SKIP_DOWNLOAD=true PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
need_node() {
  command -v node >/dev/null && command -v npm >/dev/null || { echo 'Install Node/npm from .node-version.' >&2; exit 1; }
  local wanted actual
  wanted=$(cat .node-version)
  actual=$(node -p process.versions.node)
  [[ "$actual" == "$wanted" ]] || { echo "Expected Node $wanted; found $actual." >&2; exit 1; }
}
drop_auth() { unset GH_TOKEN GITHUB_TOKEN; }
rust_checks() {
  cargo test --locked "$@"
  if [[ "$mode" == full ]]; then cargo clippy --locked --all-targets "$@"; fi
}

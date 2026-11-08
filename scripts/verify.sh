#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$root"
mode=${1:-quick}
[[ $# == 0 ]] || shift
case "$mode" in quick|full) ;; *) echo 'Usage: bash scripts/verify.sh [quick|full] [--install]' >&2; exit 2;; esac
[[ $# == 0 || ( $# == 1 && $1 == --install ) ]] || { echo 'Unknown verification option.' >&2; exit 2; }
# Frameworks load dotenv files themselves. Templates are safe; operational
# profiles belong in the development checkout, never in a verification checkout.
for directory in . frontend services/dashboard web; do
  for profile in "$directory"/.env*; do
    [[ -f "$profile" ]] || continue
    case "$profile" in *.example|*.sample|*.template) continue;; esac
    echo "Verification refuses $profile. Use a clean clone or the isolated pre-push hook." >&2
    exit 1
  done
done
# Keep tool locations/caches and process-scoped Git authentication. Application,
# database, cloud credentials and injected runtime options are never inherited.
clean=()
for name in PATH HOME USER LOGNAME LANG LC_ALL TERM TZ TMPDIR TMP TEMP CI \
  XDG_CACHE_HOME XDG_CONFIG_HOME GH_CONFIG_DIR CARGO_HOME CARGO_TARGET_DIR CARGO_BUILD_JOBS RUSTUP_HOME \
  CARGO_PROFILE_DEV_DEBUG CARGO_PROFILE_TEST_DEBUG CARGO_INCREMENTAL \
  UV_CACHE_DIR UV_PROJECT_ENVIRONMENT NPM_CONFIG_CACHE npm_config_cache \
  SSH_AUTH_SOCK SSH_AGENT_PID GIT_TERMINAL_PROMPT GIT_ASKPASS SSH_ASKPASS \
  GIT_CONFIG_COUNT GH_TOKEN GITHUB_TOKEN PLAYWRIGHT_BROWSERS_PATH FLIPPA_JSDOM; do
  [[ ! -v "$name" ]] || clean+=("$name=${!name}")
done
for name in ${!GIT_CONFIG_KEY_@} ${!GIT_CONFIG_VALUE_@}; do
  clean+=("$name=${!name}")
done
exec env -i "${clean[@]}" bash scripts/check-project.sh "$mode" "$@"

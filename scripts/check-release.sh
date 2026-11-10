#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd -- "$root"
sha=${1:-}
branch=${2:-main}
[[ $# -ge 1 && $# -le 2 && "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Usage: bash scripts/check-release.sh FULL_SHA [deploy-branch]' >&2; exit 2; }
git check-ref-format --branch "$branch" >/dev/null
command -v gh >/dev/null || { echo 'Release verification requires the operator’s authenticated GitHub CLI.' >&2; exit 1; }
origin=$(git remote get-url origin)
case "$origin" in
  https://github.com/*) repo=${origin#https://github.com/} ;;
  git@github.com*:*) repo=${origin#*:} ;;
  *) echo 'Release verification requires a canonical GitHub origin.' >&2; exit 1 ;;
esac
repo=${repo%.git}
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid GitHub repository identity.' >&2; exit 1; }
check_tree() {
  [[ "$(git rev-parse HEAD)" == "$sha" ]] || { echo 'Checkout HEAD differs from the selected release SHA.' >&2; exit 1; }
  [[ -z "$(git status --porcelain)" ]] || { echo 'Release verification requires a clean checkout, including untracked files.' >&2; exit 1; }
}
check_remote() {
  local remote_sha row checked_sha status conclusion url
  remote_sha=$(gh api "repos/$repo/git/ref/heads/$branch" --jq .object.sha)
  [[ "$remote_sha" == "$sha" ]] || { echo 'The deploy branch points to a different SHA. Select and verify its current commit.' >&2; exit 1; }
  row=$(gh run list --repo "$repo" --workflow ci.yml --commit "$sha" --branch "$branch" --event push --limit 1 --json headSha,status,conclusion,url --jq '.[] | [.headSha,.status,.conclusion,.url] | @tsv')
  IFS=$'\t' read -r checked_sha status conclusion url <<< "$row"
  [[ "$checked_sha" == "$sha" && "$status" == completed && "$conclusion" == success && -n "$url" ]] || {
    echo 'The latest push CI for this exact branch/SHA is missing, unfinished or unsuccessful.' >&2; exit 1;
  }
  echo "CI verified: $url"
}
check_tree
check_remote
if [[ -f scripts/verify.sh ]]; then
  bash scripts/verify.sh full --install
elif [[ -f scripts/verify.mjs ]]; then
  node scripts/verify.mjs full --install
else
  echo 'This checkout has no full verification entrypoint.' >&2; exit 1
fi
check_tree
check_remote
echo "Release checks passed for $repo $branch $sha. No push or deployment was performed."

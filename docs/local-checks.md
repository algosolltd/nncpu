# Local checks and staged publication

Quick covers simulator/statistics/prefetcher/static dashboard tests; full runs the entire pytest suite. Scientific benchmarks and real trace capture remain explicit research operations, not automatic CI jobs.

## Setup

Development and production target Linux. Occasional Windows testing uses WSL2 or Docker. Use the declared tool versions; no global Git configuration or credentials are installed by this repository. An existing authorized Git HTTPS credential helper is sufficient for private source dependencies. CI uses the organization PRIVATE_DEPS_TOKEN secret, enabled for this consumer only when its publication is approved. Never commit tokens.

From a dedicated clean clone, with the project's toolchain installed:

```bash
bash scripts/verify.sh quick --install
bash scripts/install-hooks.sh
bash scripts/verify.sh full --install
```

The installer is explicit because Git never enables versioned hooks automatically after clone. It preserves existing hooks and refuses shared multi-worktree configuration. The hook checks each pushed commit in a disposable detached worktree, including a branch other than the checked-out branch. It ignores ref deletions and does not cache successful results. Your working files and local profiles stay untouched. It can be bypassed by Git users, so essential checks also run on GitHub.

The runner drops inherited application/cloud/database variables and refuses dotenv files other than example/sample/template profiles. Use the normal native development commands separately; external services belong in isolated local containers or existing synthetic fixtures. Do not copy production data or credentials into verification. Dependencies are installed only when --install is given. Existing source dependency pins and lockfiles remain authoritative; a revision that is still local must be published before another computer can fetch it.

## GitHub and release

One bounded Ubuntu job runs essential checks on PRs to main, pushes to main, and explicit dispatch. Heavy browser/database/container workflows have local equivalents or documented prerequisites; previous workflow recipes, where present, are archived in docs/ci-local. A timeout is a budget limit, not proof of a fixed billed cost. Full checks and connected integrations are not claimed by a green quick job.

This branch is prepared locally. Pushes, PR creation, merges and production deployment wait for the user's instruction for each small group. Repository names, permissions, organization secrets, branch rules and Railway/autodeploy settings are unchanged. Before a manual Railway release: verify the exact selected SHA, run its required full local gate, confirm green remote CI for that SHA, take/refresh the relevant backup, then deploy that source explicitly and check health/rollback. Libraries and static repositories need only the applicable publication steps.

## Updating dependencies

Change the native manifest's selected version/revision and regenerate its lock using the native package manager. Review the lock diff, run quick/full checks, commit manifest and lock together, and publish upstream revisions before consumers. Changing a library's source does not automatically move an existing consumer pin.

## Python environment updates

Install uv 0.12.7 and Python 3.12. The initial requirements.lock preserves versions observed in the existing research environment and includes hashes. To select new versions, update requirements.txt if needed, run `uv pip compile requirements.txt --python-version 3.12 --generate-hashes -o requirements.lock`, then run both gates.

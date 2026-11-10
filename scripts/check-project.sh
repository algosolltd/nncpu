#!/usr/bin/env bash
set -euo pipefail
source scripts/check-lib.sh "$@"
if $install; then
  uv venv --python 3.12 --allow-existing .venv
  uv pip sync --python .venv/bin/python --require-hashes requirements.lock
fi
drop_auth
if [[ "$mode" == full ]]; then
  .venv/bin/python -m pytest -q
else
  .venv/bin/python -m pytest -q tests/test_cpu.py tests/test_traces.py tests/test_baselines.py tests/test_statistics.py tests/test_prefetchers.py tests/test_webapp.py
fi

"""Run the matched-control confidence-filter study.

The output is intentionally separate from the legacy paper artifact.  It asks
whether the confidence gate generalizes across classical predictors and
whether any NN advantage remains after matching admission policy, latency,
associativity, bandwidth, MSHRs and lookahead distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from nncpu.experiment import source_digest
from nncpu.research import (
    REGULARITY_MIN_SUPPORT,
    REGULARITY_WINDOW,
    RESEARCH_CONFIGS,
    default_profiles,
    run_gate_study,
)
from nncpu.traces import build_trace_workloads, read_trace
from nncpu.workloads import build_workloads


def build_study_workloads(length: int, seed: int, include_real: bool = True) -> dict:
    workloads = {
        name: stream
        for name, stream in build_workloads(length, seed=seed).items()
        if name != "arithmetic_mix"
    }
    workloads.update(build_trace_workloads(length, seed=seed))
    if include_real:
        trace_dir = "results/traces"
        if os.path.isdir(trace_dir):
            for filename in sorted(os.listdir(trace_dir)):
                if filename.endswith(".trace") and "champ" not in filename:
                    workloads[filename[:-6]] = read_trace(
                        os.path.join(trace_dir, filename), limit=length
                    )
    return workloads


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _tracked_tree_dirty() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet"], check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    return unstaged.returncode != 0 or staged.returncode != 0


def run(
    output: str,
    seeds: int,
    length: int,
    include_real: bool = True,
    seed_start: int = 0,
    configs: tuple[str, ...] = RESEARCH_CONFIGS,
    role: str = "exploratory",
) -> None:
    # Capture provenance before generated files alter repository status.
    revision = _git_revision()
    tracked_dirty = _tracked_tree_dirty()
    workloads = {
        seed: build_study_workloads(length, seed, include_real=include_real)
        for seed in range(seed_start, seed_start + seeds)
    }
    profiles = default_profiles()
    runs, summary, contrasts = run_gate_study(
        workloads, profiles=profiles, configs=configs
    )
    os.makedirs(output, exist_ok=True)
    runs.to_csv(os.path.join(output, "runs.csv"), index=False)
    summary.to_csv(os.path.join(output, "summary.csv"), index=False)
    contrasts.to_csv(os.path.join(output, "contrasts.csv"), index=False)
    config = {
        "seeds": seeds,
        "seed_start": seed_start,
        "length": length,
        "configs": configs,
        "role": role,
        "development_seed_range": [0, 9],
        "regularity_gate": {
            "window": REGULARITY_WINDOW,
            "min_support": REGULARITY_MIN_SUPPORT,
            "selection": "selected on development seeds 0-9",
        },
        "include_real": include_real,
        "profiles": [
            {
                "name": profile.name,
                "machine": profile.machine.as_dict(),
                "distances": profile.distances,
            }
            for profile in profiles
        ],
        "workloads": sorted(next(iter(workloads.values()))),
    }
    config_payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    with open(os.path.join(output, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_revision": revision,
                "git_tracked_dirty": tracked_dirty,
                "source_sha256": source_digest(),
                "config_sha256": hashlib.sha256(config_payload.encode()).hexdigest(),
            },
            f,
            indent=2,
        )
    print(f"saved {len(runs)} paired raw rows to {output}")
    decisive = contrasts[
        contrasts.workload.isin(("random_read", "phased_switch"))
    ]
    print(decisive.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/research_gate")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--configs", default=",".join(RESEARCH_CONFIGS),
        help="comma-separated matched-control subset",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--role", choices=("exploratory", "holdout"), default="exploratory"
    )
    parser.add_argument("--no-real", action="store_true")
    args = parser.parse_args()
    configs = tuple(part.strip() for part in args.configs.split(",") if part.strip())
    run(
        args.output,
        seeds=3 if args.quick else args.seeds,
        length=300 if args.quick else args.length,
        include_real=not args.no_real,
        seed_start=args.seed_start,
        configs=configs,
        role=args.role,
    )


if __name__ == "__main__":
    main()

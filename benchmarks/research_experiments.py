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
from nncpu.experiment import source_digest
from nncpu.research import default_profiles, run_gate_study
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


def run(output: str, seeds: int, length: int, include_real: bool = True) -> None:
    workloads = {
        seed: build_study_workloads(length, seed, include_real=include_real)
        for seed in range(seeds)
    }
    profiles = default_profiles()
    runs, summary, contrasts = run_gate_study(workloads, profiles=profiles)
    os.makedirs(output, exist_ok=True)
    runs.to_csv(os.path.join(output, "runs.csv"), index=False)
    summary.to_csv(os.path.join(output, "summary.csv"), index=False)
    contrasts.to_csv(os.path.join(output, "contrasts.csv"), index=False)
    config = {
        "seeds": seeds,
        "length": length,
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
                "git_revision": _git_revision(),
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-real", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        seeds=3 if args.quick else args.seeds,
        length=300 if args.quick else args.length,
        include_real=not args.no_real,
    )


if __name__ == "__main__":
    main()

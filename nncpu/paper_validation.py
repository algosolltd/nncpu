"""Independent checks for the evidence committed with the paper.

The normal unit tests answer "does this implementation behave as designed?".
This module answers the separate and stricter question "were the published
artifacts produced by *this* implementation, and do their stated claims follow
from the raw rows?".  It never rewrites results.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .benchmark import run_workload
from .cpu import MachineConfig
from .experiment import ExperimentConfig, aggregate, source_digest
from .workloads import build_workloads


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass
class PaperValidationReport:
    checks: list[ValidationCheck] = field(default_factory=list)
    sensitivity: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        self.checks.append(ValidationCheck(name, bool(passed), detail, severity))


def _sha256_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_object_exists(revision: str) -> bool:
    if not revision:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def _compare_summary(raw: pd.DataFrame, saved: pd.DataFrame) -> tuple[bool, str]:
    calculated = aggregate(raw)
    keys = ["workload", "config"]
    common = [
        c for c in calculated.columns
        if c in saved.columns and c not in keys
        and pd.api.types.is_numeric_dtype(calculated[c])
    ]
    merged = calculated.merge(saved, on=keys, suffixes=("_calc", "_saved"), how="outer")
    if len(merged) != len(calculated) or len(merged) != len(saved):
        return False, "summary keys differ from aggregation of runs.csv"
    mismatches = []
    for column in common:
        a = merged[f"{column}_calc"].to_numpy(dtype=float)
        b = merged[f"{column}_saved"].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True):
            mismatches.append(column)
    return not mismatches, (
        "summary.csv is an exact aggregation of runs.csv"
        if not mismatches else f"mismatched columns: {', '.join(mismatches)}"
    )


def _compare_recomputed_seeds(
    config: ExperimentConfig, runs: pd.DataFrame, count: int
) -> tuple[bool, str]:
    seeds = sorted(int(s) for s in runs.seed.unique())[:count]
    mismatches = []
    for seed in seeds:
        workloads = build_workloads(
            config.length, seed=seed if config.vary_workload_seed else None
        )
        kwargs = dict(config.nn_kwargs)
        kwargs.setdefault("random_state", seed)
        for workload in config.workloads:
            for prefetch_config in config.configs:
                report = run_workload(
                    workloads[workload],
                    prefetch_config,
                    machine=config.machine,
                    nn_kwargs=kwargs,
                )
                expected = runs[
                    (runs.seed == seed)
                    & (runs.workload == workload)
                    & (runs.config == prefetch_config)
                ]
                if len(expected) != 1:
                    mismatches.append(f"{seed}/{workload}/{prefetch_config}: missing or duplicate")
                    continue
                row = expected.iloc[0]
                observed = {
                    "cycles": report.cycles,
                    "hit_rate": report.hit_rate,
                    "prefetch_issued": report.prefetch_issued,
                    "prefetch_used": report.prefetch_used,
                }
                for metric, value in observed.items():
                    if not np.isclose(float(row[metric]), float(value), rtol=0, atol=1e-12):
                        mismatches.append(
                            f"{seed}/{workload}/{prefetch_config}/{metric}: "
                            f"saved={row[metric]} current={value}"
                        )
    if mismatches:
        preview = "; ".join(mismatches[:6])
        return False, f"{len(mismatches)} mismatches; {preview}"
    return True, f"recomputed {len(seeds)} seed(s) exactly from current source"


def prefetch_latency_sensitivity(config: ExperimentConfig, seed: int = 0) -> dict:
    """Compare ideal zero-latency results with DRAM-latency prefetches."""
    workloads = build_workloads(
        config.length, seed=seed if config.vary_workload_seed else None
    )
    result = {}
    for workload in ("sequential_read", "strided_read"):
        if workload not in config.workloads:
            continue
        result[workload] = {}
        cycles_by_profile = {}
        for profile, latency in (
            ("ideal", 0),
            ("timed", config.machine.mem_latency),
        ):
            machine_data = config.machine.as_dict()
            machine_data["prefetch_latency"] = latency
            machine = MachineConfig(**machine_data)
            profile_cycles = {}
            for prefetch_config in ("baseline", "stride", "nn"):
                if prefetch_config not in config.configs:
                    continue
                kwargs = dict(config.nn_kwargs)
                kwargs.setdefault("random_state", seed)
                report = run_workload(
                    workloads[workload], prefetch_config,
                    machine=machine, nn_kwargs=kwargs,
                )
                profile_cycles[prefetch_config] = report.cycles
            cycles_by_profile[profile] = profile_cycles
        base = cycles_by_profile["ideal"].get("baseline")
        for prefetch_config in ("stride", "nn"):
            if base is None or prefetch_config not in cycles_by_profile["ideal"]:
                continue
            result[workload][prefetch_config] = {
                "ideal_speedup": base / cycles_by_profile["ideal"][prefetch_config],
                "timed_speedup": base / cycles_by_profile["timed"][prefetch_config],
            }
    return result


def validate_paper_artifact(
    artifact_dir: str = "results/final_30",
    claims_path: str = "paper/claims.json",
    recompute_seeds: int = 1,
) -> PaperValidationReport:
    report = PaperValidationReport()
    required = ("config.json", "manifest.json", "runs.csv", "summary.csv")
    missing = [name for name in required if not os.path.exists(os.path.join(artifact_dir, name))]
    report.add("artifact_files", not missing, "missing: " + ", ".join(missing) if missing else "all required files exist")
    if missing:
        return report

    with open(os.path.join(artifact_dir, "config.json"), encoding="utf-8") as f:
        raw_config = json.load(f)
    with open(os.path.join(artifact_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    config = ExperimentConfig.from_dict(raw_config)
    resolved_config = config.as_dict()

    revision = manifest.get("git_revision", "")
    report.add("git_revision_available", _git_object_exists(revision), f"revision={revision or '(missing)'}")
    report.add("generation_tree_clean", manifest.get("git_dirty") is False, f"git_dirty={manifest.get('git_dirty')!r}")
    report.add(
        "source_digest",
        manifest.get("source_sha256") == source_digest(),
        f"saved={manifest.get('source_sha256', '(missing)')} current={source_digest()}",
    )
    report.add(
        "config_digest",
        manifest.get("config_sha256") == _sha256_json(resolved_config),
        "persisted configuration includes and hashes every effective setting",
    )

    runs = pd.read_csv(os.path.join(artifact_dir, "runs.csv"))
    summary = pd.read_csv(os.path.join(artifact_dir, "summary.csv"))
    key_columns = ["seed", "workload", "config"]
    expected_rows = config.runs * len(config.workloads) * len(config.configs)
    unique = not runs.duplicated(key_columns).any()
    report.add(
        "raw_run_matrix",
        len(runs) == expected_rows and unique,
        f"rows={len(runs)} expected={expected_rows} unique_keys={unique}",
    )
    passed, detail = _compare_summary(runs, summary)
    report.add("summary_derived_from_raw", passed, detail)

    baseline_speedups = runs[runs.config == "baseline"]["speedup"]
    report.add(
        "paired_speedups",
        bool(np.allclose(baseline_speedups, 1.0, rtol=0, atol=0)),
        "baseline speedup is exactly 1 for every seed",
    )

    random_rows = runs[runs.workload == "random_read"]
    base = random_rows[random_rows.config == "baseline"][["seed", "cycles", "hit_rate"]]
    nn = random_rows[random_rows.config == "nn"][["seed", "cycles", "hit_rate"]]
    paired = base.merge(nn, on="seed", suffixes=("_base", "_nn"))
    exact_matches = int(
        ((paired.cycles_base == paired.cycles_nn)
         & (paired.hit_rate_base == paired.hit_rate_nn)).sum()
    )
    regressions = int((paired.cycles_nn > paired.cycles_base).sum())
    report.add(
        "random_non_regression_observation",
        regressions == 0,
        f"exact matches={exact_matches}/{config.runs}; regressions={regressions}/{config.runs}",
        severity="warning",
    )

    if claims_path and os.path.exists(claims_path):
        with open(claims_path, encoding="utf-8") as f:
            claims = json.load(f)
        claim_errors = []
        for claim in claims.get("summary_values", []):
            row = summary[
                (summary.workload == claim["workload"])
                & (summary.config == claim["config"])
            ]
            if len(row) != 1 or not np.isclose(
                float(row.iloc[0][claim["metric"]]), float(claim["value"]),
                rtol=0, atol=float(claim.get("atol", 1e-9)),
            ):
                claim_errors.append(claim["id"])
        observed_runs = {
            "random_exact_matches": exact_matches,
            "random_regressions": regressions,
        }
        for claim in claims.get("run_values", []):
            if observed_runs.get(claim["metric"]) != claim["value"]:
                claim_errors.append(claim["id"])
        report.add(
            "machine_readable_paper_claims",
            not claim_errors,
            "all declared values match summary.csv" if not claim_errors else f"mismatches: {', '.join(claim_errors)}",
        )
    else:
        report.add("machine_readable_paper_claims", False, f"missing claims file: {claims_path}")

    if recompute_seeds:
        passed, detail = _compare_recomputed_seeds(config, runs, recompute_seeds)
        report.add("current_source_reproduction", passed, detail)

    report.sensitivity = prefetch_latency_sensitivity(config)
    report.add(
        "external_validity_scope",
        True,
        "speedups are internally valid only for prefetch_latency=0; timed sensitivity is reported separately",
        severity="warning",
    )
    return report

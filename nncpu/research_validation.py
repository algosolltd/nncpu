"""Integrity and claim checks for the matched-control research artifact."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .experiment import source_digest
from .research import aggregate_research, matched_contrasts


@dataclass(frozen=True)
class ResearchCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ResearchValidationReport:
    checks: list[ResearchCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(ResearchCheck(name, bool(passed), detail))


def _json_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit_exists(revision: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _frames_match(calculated: pd.DataFrame, saved: pd.DataFrame, keys: list[str]):
    calculated = calculated.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    if list(calculated[keys].itertuples(index=False, name=None)) != list(
        saved[keys].itertuples(index=False, name=None)
    ):
        return False, "row keys differ"
    common = [
        column for column in calculated.columns
        if column in saved.columns and column not in keys
    ]
    mismatches = []
    for column in common:
        if pd.api.types.is_numeric_dtype(calculated[column]):
            if not np.allclose(
                calculated[column].to_numpy(dtype=float),
                saved[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            ):
                mismatches.append(column)
        elif not calculated[column].equals(saved[column]):
            mismatches.append(column)
    return not mismatches, (
        "exactly recomputed from runs.csv"
        if not mismatches else f"mismatched columns: {', '.join(mismatches)}"
    )


def validate_research_artifact(
    artifact_dir: str = "results/research_gate_validation",
    claims_path: str = "paper/research_claims.json",
) -> ResearchValidationReport:
    report = ResearchValidationReport()
    required = ("config.json", "manifest.json", "runs.csv", "summary.csv", "contrasts.csv")
    missing = [name for name in required if not os.path.exists(os.path.join(artifact_dir, name))]
    report.add("artifact_files", not missing, "all files exist" if not missing else f"missing: {missing}")
    if missing:
        return report

    with open(os.path.join(artifact_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    with open(os.path.join(artifact_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = pd.read_csv(os.path.join(artifact_dir, "runs.csv"))
    summary = pd.read_csv(os.path.join(artifact_dir, "summary.csv"))
    contrasts = pd.read_csv(os.path.join(artifact_dir, "contrasts.csv"))

    revision = manifest.get("git_revision", "")
    report.add("git_revision", _git_commit_exists(revision), f"revision={revision}")
    report.add(
        "generation_tree_clean",
        manifest.get("git_tracked_dirty") is False,
        f"git_tracked_dirty={manifest.get('git_tracked_dirty')!r}",
    )
    report.add(
        "source_digest",
        manifest.get("source_sha256") == source_digest(revision),
        f"saved={manifest.get('source_sha256')} pinned={source_digest(revision)}",
    )
    report.add(
        "config_digest",
        manifest.get("config_sha256") == _json_digest(config),
        "configuration hash matches" if manifest.get("config_sha256") == _json_digest(config) else "configuration hash mismatch",
    )
    expected = 0
    for profile in config["profiles"]:
        expected += 1 + (len(config["configs"]) - 1) * len(profile["distances"])
    expected *= config["seeds"] * len(config["workloads"])
    keys = ["seed", "profile", "workload", "config", "distance"]
    unique = not runs.duplicated(keys).any()
    report.add("raw_matrix", len(runs) == expected and unique, f"rows={len(runs)} expected={expected} unique={unique}")

    streams_per_pair = runs.groupby(["seed", "workload"]).stream_sha256.nunique()
    report.add(
        "paired_streams",
        bool((streams_per_pair == 1).all()),
        "every config/profile receives the same stream within seed/workload",
    )
    baseline = runs[runs.config == "baseline"].speedup
    report.add("paired_baseline", bool(np.allclose(baseline, 1.0, rtol=0, atol=0)), "baseline speedups are exactly one")
    report.add(
        "holdout_split",
        config.get("role") == "holdout"
        and config.get("seed_start") == 10
        and config.get("seeds") == 30
        and config.get("development_seed_range") == [0, 9],
        f"role={config.get('role')} validation seeds={config.get('seed_start')}..{config.get('seed_start', 0) + config.get('seeds', 0) - 1}",
    )

    calculated_summary = aggregate_research(runs)
    passed, detail = _frames_match(
        calculated_summary, summary, ["profile", "workload", "config", "distance"]
    )
    report.add("summary_recomputed", passed, detail)
    calculated_contrasts = matched_contrasts(runs)
    passed, detail = _frames_match(
        calculated_contrasts,
        contrasts,
        ["candidate", "control", "profile", "workload", "distance"],
    )
    report.add("contrasts_recomputed", passed, detail)

    claim_errors = []
    if os.path.exists(claims_path):
        with open(claims_path, encoding="utf-8") as handle:
            claims = json.load(handle)
        for claim in claims.get("values", []):
            frame = contrasts if claim["source"] == "contrasts" else summary
            row = frame
            for column, value in claim["selector"].items():
                row = row[row[column] == value]
            if len(row) != 1 or not np.isclose(
                float(row.iloc[0][claim["metric"]]),
                float(claim["value"]),
                rtol=0,
                atol=float(claim.get("atol", 1e-10)),
            ):
                claim_errors.append(claim["id"])
    else:
        claim_errors.append("missing claims file")
    report.add(
        "machine_readable_claims",
        not claim_errors,
        "all declared claims match" if not claim_errors else f"mismatches: {claim_errors}",
    )
    return report

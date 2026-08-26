#!/usr/bin/env python3
"""Independently verify a completed ChampSim campaign artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import campaign


def _csv_form(row: dict) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in row.items()}


def _prediction_checks(summary: dict) -> dict[str, bool]:
    matched = summary["regularity_stride_vs_raw_stride"]
    return {
        "accepted_l1d_prefetches_reduced": matched["aggregate_prefetch_issue_reduction"] > 0,
        "l2c_prefetch_misses_reduced": matched["aggregate_l2c_prefetch_miss_reduction"] > 0,
        "llc_prefetch_misses_reduced": matched["aggregate_llc_prefetch_miss_reduction"] > 0,
        "accepted_prefetch_accuracy_increased": matched["candidate_prefetch_accuracy"] > matched["control_prefetch_accuracy"],
        "most_useful_prefetches_retained": matched["useful_prefetch_retention"] > 0.5,
        "dram_reads_within_frozen_margin": abs(matched["aggregate_dram_read_request_reduction"]) <= 0.01,
        "geometric_mean_ipc_not_improved": matched["geometric_mean_ipc_speedup"] <= 1.0,
    }


def _verify_machine_claims(summary: dict, manifest: dict, contrasts: list[dict]) -> None:
    claims_path = campaign.REPOSITORY / "paper" / "champsim_claims.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    protocol = claims["protocol"]
    if protocol["independent_programs"] != len(manifest["trace_order"]):
        raise AssertionError("claim independent-unit count mismatch")
    if protocol["warmup_instructions"] != manifest["warmup_instructions"]:
        raise AssertionError("claim warm-up mismatch")
    if protocol["simulation_instructions"] != manifest["simulation_instructions"]:
        raise AssertionError("claim ROI mismatch")
    if protocol["selection"] != manifest["selection_rule"]:
        raise AssertionError("claim trace-selection rule mismatch")
    official = summary["raw_stride_vs_official_ip_stride"]
    official_claim = claims["raw_control_vs_official_ip_stride"]
    if not official_claim["exact_output_equivalence"]:
        raise AssertionError("machine claim does not require official equivalence")
    if official_claim["geometric_mean_ipc_ratio"] != official["geometric_mean_ipc_speedup"]:
        raise AssertionError("official-equivalence IPC claim mismatch")
    if official_claim["ties"] != official["ties"]:
        raise AssertionError("official-equivalence tie claim mismatch")

    matched = summary["regularity_stride_vs_raw_stride"]
    matched_claim = claims["regularity_stride_vs_raw_stride"]
    mapping = {
        "geometric_mean_ipc_ratio": "geometric_mean_ipc_speedup",
        "bootstrap_95pct_ci": "bootstrap_95pct_ci",
        "exact_two_sided_sign_p": "exact_two_sided_sign_p",
        "wins": "wins",
        "losses": "losses",
        "aggregate_prefetch_issue_reduction": "aggregate_prefetch_issue_reduction",
        "aggregate_l2c_prefetch_miss_reduction": "aggregate_l2c_prefetch_miss_reduction",
        "aggregate_llc_prefetch_miss_reduction": "aggregate_llc_prefetch_miss_reduction",
        "aggregate_dram_read_request_reduction": "aggregate_dram_read_request_reduction",
        "raw_prefetch_accuracy": "control_prefetch_accuracy",
        "gated_prefetch_accuracy": "candidate_prefetch_accuracy",
        "useful_prefetch_retention": "useful_prefetch_retention",
        "maximum_relative_callback_count_difference": "maximum_relative_callback_count_difference",
    }
    for claim_key, summary_key in mapping.items():
        if matched_claim[claim_key] != matched[summary_key]:
            raise AssertionError(f"machine claim mismatch: {claim_key}")

    observed_per_program = [
        {
            "trace": row["trace"],
            "ipc_ratio": row["ipc_speedup"],
            "dram_read_request_reduction": row["dram_read_request_reduction"],
            "prefetch_issue_reduction": row["prefetch_issue_reduction"],
        }
        for row in contrasts
        if row["candidate"] == "regularity_stride" and row["control"] == "raw_stride"
    ]
    if claims["per_program_regularity_stride_vs_raw_stride"] != observed_per_program:
        raise AssertionError("machine per-program claims mismatch")


def _assert_csv_exact(path: Path, expected: list[dict]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    expected_text = [_csv_form(row) for row in expected]
    if observed != expected_text:
        raise AssertionError(f"table does not reproduce exactly: {path}")


def verify(results: Path, champsim_checkout: Path | None, trace_dir: Path | None) -> None:
    results = results.resolve()
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if manifest["records"] != len(manifest["trace_order"]) * len(campaign.CONFIGS):
        raise AssertionError("manifest record count is not a complete matrix")
    if manifest["warmup_instructions"] != 50_000_000:
        raise AssertionError("primary warm-up is not 50 million instructions")
    if manifest["simulation_instructions"] != 200_000_000:
        raise AssertionError("primary ROI is not 200 million instructions")

    if campaign._sha256(Path(identity["trace_list"]["path"])) != identity["trace_list"]["sha256"]:
        raise AssertionError("trace-list hash mismatch")
    for config, path in campaign.CONFIGS.items():
        if campaign._sha256(path) != identity["configs"][config]:
            raise AssertionError(f"configuration hash mismatch: {config}")
    for relative, expected_hash in identity["module"].items():
        if campaign._sha256(campaign.REPOSITORY / relative) != expected_hash:
            raise AssertionError(f"module hash mismatch: {relative}")

    records = []
    metadata_files = sorted((results / "raw").glob("*/*.metadata.json"))
    expected_count = manifest["records"]
    if len(metadata_files) != expected_count:
        raise AssertionError(f"expected {expected_count} metadata files, got {len(metadata_files)}")
    for trace_name in manifest["trace_order"]:
        trace = Path(trace_name)
        run_dir = results / "raw" / campaign._slug(trace)
        for config in campaign.CONFIGS:
            records.append(
                campaign._parse_run(
                    run_dir / f"{config}.json",
                    run_dir / f"{config}.log",
                    trace,
                    config,
                )
            )
    records.sort(key=lambda record: (record.trace, record.config))
    campaign._validate_records(records, manifest["simulation_instructions"])
    _assert_csv_exact(results / "runs.csv", [asdict(record) for record in records])
    contrasts, summary = campaign._contrasts(records)
    _assert_csv_exact(results / "contrasts.csv", contrasts)
    if json.loads((results / "summary.json").read_text(encoding="utf-8")) != summary:
        raise AssertionError("summary does not reproduce exactly from raw outputs")

    checks = _prediction_checks(summary)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"frozen predictions failed: {failed}")
    _verify_machine_claims(summary, manifest, contrasts)

    if trace_dir is not None:
        for trace_name, expected in identity["traces"].items():
            trace_path = trace_dir.resolve() / trace_name
            if trace_path.stat().st_size != expected["bytes"] or campaign._sha256(trace_path) != expected["sha256"]:
                raise AssertionError(f"trace identity mismatch: {trace_name}")
    if champsim_checkout is not None:
        checkout = champsim_checkout.resolve()
        binary = checkout / "bin" / "champsim"
        if campaign._sha256(binary) != identity["champsim_binary_sha256"]:
            raise AssertionError("ChampSim binary hash mismatch")
        if campaign._git(checkout, "rev-parse", "HEAD") != identity["champsim_revision"]:
            raise AssertionError("ChampSim revision mismatch")

    for name in sorted(checks):
        print(f"PASS {name}")
    print(f"PASS exact raw reproduction ({len(records)} runs)")
    print("PASS raw control equals official ip_stride on every trace")
    print("PASS source, configuration, and trace-list hashes")
    print("PASS machine-readable paper claims")
    if trace_dir is not None:
        print("PASS external trace hashes")
    if champsim_checkout is not None:
        print("PASS ChampSim binary and revision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--champsim", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    try:
        verify(args.results, args.champsim, args.trace_dir)
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

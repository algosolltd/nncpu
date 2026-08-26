"""Tests that connect paper claims to raw evidence and current source."""

from nncpu.experiment import ExperimentConfig
from nncpu.paper_validation import (
    prefetch_latency_sensitivity,
    validate_paper_artifact,
)


def test_committed_paper_artifact_matches_current_source_and_claims():
    report = validate_paper_artifact(recompute_seeds=1)
    failures = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
    assert report.ok, "\n".join(failures)


def test_paper_speedups_are_sensitive_to_prefetch_timeliness():
    config = ExperimentConfig(
        name="sensitivity",
        length=400,
        runs=1,
        workloads=("sequential_read", "strided_read"),
        configs=("baseline", "stride", "nn"),
    )
    result = prefetch_latency_sensitivity(config)
    for workload in result.values():
        assert workload["stride"]["ideal_speedup"] > workload["stride"]["timed_speedup"]
    # Preserve the distinction in the test itself: this does not disprove the
    # paper's explicitly idealized model, but it prevents presenting that model
    # as a latency-aware hardware result.
    assert result["strided_read"]["stride"]["timed_speedup"] < 1.1

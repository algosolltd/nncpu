from benchmarks.specdec_endpoint_audit import compute_summary, exact_sign_p


def _row(repeat, policy, prompt, tps, drafted, accepted):
    return {
        "phase": "holdout",
        "repeat": repeat,
        "policy": policy,
        "prompt_id": prompt,
        "content_sha256": f"hash-{prompt}",
        "predicted_n": 96,
        "predicted_ms": 96_000 / tps,
        "predicted_per_second": tps,
        "request_wall_ms": 96_000 / tps + 10,
        "draft_n": drafted,
        "draft_n_accepted": accepted,
    }


def test_exact_sign_test_all_eight_pairs_same_direction():
    assert exact_sign_p(0, 8) == 0.0078125


def test_summary_detects_acceptance_throughput_reversal():
    rows = []
    for repeat in range(2):
        for prompt in ("a", "b"):
            rows.append(_row(repeat, "target_only", prompt, 10, 0, 0))
            rows.append(_row(repeat, "lower_acceptance", prompt, 12, 100, 80))
            rows.append(_row(repeat, "higher_acceptance", prompt, 9, 100, 95))

    summary = compute_summary(rows, expected_repeats=2, expected_n_predict=96)

    assert summary["complete"]
    assert summary["output_mismatches"] == []
    assert summary["token_count_mismatches"] == []
    assert summary["proxy_reversals"] == [
        {
            "higher_acceptance_policy": "higher_acceptance",
            "lower_acceptance_policy": "lower_acceptance",
            "acceptance_difference": 0.1499999999999999,
            "geomean_tps_ratio": 0.75,
            "wins": 0,
            "losses": 2,
            "ties": 0,
            "exact_two_sided_sign_p": 0.5,
        }
    ]

"""CLI verifier for the holdout matched-control research artifact."""

import argparse

from nncpu.research_validation import validate_research_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="results/research_gate_validation")
    parser.add_argument("--claims", default="paper/research_claims.json")
    args = parser.parse_args()
    report = validate_research_artifact(args.artifact, args.claims)
    for check in report.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

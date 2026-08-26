"""Verify that paper artifacts and current source describe the same evidence."""

import argparse
import json
import sys

from nncpu.paper_validation import validate_paper_artifact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="results/final_30")
    parser.add_argument("--claims", default="paper/claims.json")
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of saved seeds to recompute; use 30 for full verification")
    args = parser.parse_args(argv)

    report = validate_paper_artifact(args.artifact, args.claims, args.seeds)
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print("\nPrefetch-latency sensitivity (speedup vs no-prefetch):")
    print(json.dumps(report.sensitivity, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())


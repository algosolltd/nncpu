"""Run an arbitrary trace file through the simulator under all configs.

    python benchmarks/run_trace_file.py my.trace
    python benchmarks/run_trace_file.py my.trace --limit 5000 --seeds 3

Prints cycles / hit-rate / speedup for baseline, stride and the learned
prefetcher, plus the per-instruction cycle histogram summary.
"""

import argparse
import sys

import numpy as np

from nncpu.benchmark import CONFIGS, run_workload
from nncpu.traces import read_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to a .trace file (see read_trace format)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    insts = read_trace(args.trace, limit=args.limit)
    if not insts:
        print("Trace is empty.")
        return 1
    print(f"{len(insts)} instructions loaded from {args.trace}")

    counts = {}
    n = len(insts) * args.seeds + 1
    base = None
    for cfg in CONFIGS:
        cycles = []
        for seed in range(args.seeds):
            rep = run_workload(insts, cfg, nn_kwargs={"random_state": seed})
            cycles.append(rep.cycles)
        cycles = np.asarray(cycles)
        mean = cycles.mean()
        if cfg == "baseline":
            base = mean
        sp = base / mean if base and cfg != "baseline" else 1.0
        counts[cfg] = rep.hit_rate
        print(f"  {cfg:9s} cycles={mean:,.0f} +/- {cycles.std():,.0f}  "
              f"hit={rep.hit_rate:.3f}  speedup={sp:.2f}x")
    print("\nUse results/exp_traces + the dashboard to plot these.")


if __name__ == "__main__":
    sys.exit(main())
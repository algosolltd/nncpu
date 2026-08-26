"""Compare all prefetch configs: core (baseline/stride/nn) + published
mechanisms modeled in-simulator (nextline, berti).

Covers synthetic workloads, ML-pattern traces, and REAL algorithm traces
(merge sort / matmul / binary search captured by capture_real_trace.py).

    python benchmarks/compare_baselines.py [--quick]

Writes results/exp_baselines (patterns + real traces) and
results/exp_baselines_synth (synthetic workloads) as experiment dirs.
"""

import argparse
import os
import time

import pandas as pd

from nncpu.baselines import SIM_CONFIGS
from nncpu.benchmark import run_workload, summarize
from nncpu.experiment import aggregate
from nncpu.traces import build_trace_workloads, read_trace
from nncpu.workloads import build_workloads
from nncpu.traces import run_trace_battery

TRACE_DIR = "results/traces"


def _real_traces():
    workloads = {}
    if os.path.isdir(TRACE_DIR):
        for f in sorted(os.listdir(TRACE_DIR)):
            if f.endswith(".trace") and "champ" not in f:
                workloads[f[:-6]] = read_trace(os.path.join(TRACE_DIR, f))
    return workloads


def _synth_battery(root, seeds, length):
    from nncpu.traces import run_trace_battery as _btw  # reuse for patterns

    workloads = build_workloads(length)
    return run_trace_battery(root=root, name="exp_baselines_synth", seeds=seeds,
                             length=length, workloads=workloads,
                             configs=SIM_CONFIGS)


def _print(summary_df, header):
    print(f"\n{header}")
    for wl in summary_df.workload.unique():
        g = summary_df[summary_df.workload == wl]
        seg = "  ".join(
            f"{cfg} {g[g.config == cfg].cycles_mean.iloc[0]:,.0f} "
            f"{g[g.config == cfg].speedup_mean.iloc[0]:.2f}x "
            f"hit={g[g.config == cfg].hit_rate_mean.iloc[0]:.3f}"
            for cfg in g.config.unique())
        print(f"  {wl:18s} | {seg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--length", type=int, default=2000)
    args = ap.parse_args()
    seeds = 2 if args.quick else args.seeds
    length = 600 if args.quick else args.length

    t0 = time.perf_counter()
    print(f"configs: {', '.join(SIM_CONFIGS)}  |  seeds={seeds} length={length}")

    # patterns + real traces into exp_baselines
    workloads = {}
    workloads.update(build_trace_workloads(length))
    workloads.update(_real_traces())
    res = run_trace_battery(seeds=seeds, name="exp_baselines", length=length,
                            workloads=workloads, configs=SIM_CONFIGS)
    _print(res["summary"], "=== exp_baselines (patterns + real traces) ===")

    # synthetic
    res2 = _synth_battery("results", seeds, length)
    _print(res2["summary"], "=== exp_baselines_synth (synthetic) ===")

    print(f"\nwall {time.perf_counter() - t0:.1f}s")
    print("Rebuild the dashboard with: python webapp/make_dashboard.py")


if __name__ == "__main__":
    main()
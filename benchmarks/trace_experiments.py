"""Run the ML/agent-style trace workloads under all prefetch configs.

    python benchmarks/trace_experiments.py            # full battery
    python benchmarks/trace_experiments.py --quick    # tiny smoke run

Writes results/exp_traces/ (summary.csv + runs.csv + config.json) so the
dashboard shows it as an experiment.
"""

import argparse
import os
import time

from nncpu.traces import build_trace_workloads, run_trace_battery

EMBED_TABLE = 8192  # words; default table bigger than the L1


def scan_aggregators(root="results", seeds=5, length=2000):
    """Ablation: gate aggregator (mean/median) x delta-cap on the traces."""
    import paper_experiments as PE  # type: ignore

    PE.RESULTS_ROOT = root
    from nncpu.benchmark import summarize as _summarize
    from nncpu.benchmark import CONFIGS as _CFG, run_workload as _run
    from nncpu.traces import build_trace_workloads as _btw
    from paper_experiments import save_scan  # type: ignore

    variants = [
        ("mean_cap64", dict(gate_aggregator="mean", delta_cap=64)),
        ("median_cap64", dict(gate_aggregator="median", delta_cap=64)),
        ("median_cap16", dict(gate_aggregator="median", delta_cap=16)),
    ]
    workloads = _btw(length)
    rows = []
    for label, over in variants:
        for seed in range(seeds):
            for wname, insts in workloads.items():
                for cfg in _CFG:
                    rep = _run(insts, cfg, nn_kwargs={"random_state": seed, **over})
                    r = _summarize(rep)
                    rows.append({"label": label, "seed": seed, "workload": wname,
                                 "config": cfg, "cycles": r["cycles"],
                                 "hit_rate": r["hit_rate"]})
    from pandas import DataFrame
    df = DataFrame(rows)
    if "baseline" in df.config.values:
        base = df[df.config == "baseline"].groupby("workload")["cycles"].mean().rename("base_cycles")
        df = df.merge(base, on="workload")
        df["speedup"] = df["base_cycles"] / df["cycles"]
    scan = df.groupby(["label", "workload", "config"])["speedup"] \
             .mean().reset_index()
    df, outdir = save_scan("exp_gate_aggregators", rows,
                           f"gate aggregator x delta_cap on traces x {seeds} seeds")
    scan.to_csv(f"{outdir}/scan_mean.csv", index=False)
    print("\n### exp_gate_aggregators (traces): NN speedup x seeds")
    piv = scan[scan.config == "nn"].pivot(index="label", columns="workload",
                                          values="speedup").round(2)
    print(piv.to_string())
    return scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--root", default="results")
    ap.add_argument("--name", default="exp_traces")
    ap.add_argument("--aggscan", action="store_true",
                    help="also run the gate aggregator x cap ablation")
    args = ap.parse_args()

    length = 600 if args.quick else args.length
    seeds = 2 if args.quick else args.seeds

    workloads = build_trace_workloads(length)
    print("Trace battery:")
    for k, v in workloads.items():
        print(f"  {k:18s} {len(v):6d} instructions")

    t0 = time.perf_counter()
    res = run_trace_battery(root=args.root, name=args.name, seeds=seeds,
                            length=length, workloads=workloads)
    s = res["summary"]
    print("\n=== Trace workload x config (mean over seeds) ===")
    for wl in s.workload.unique():
        g = s[s.workload == wl]
        seg = "  ".join(
            f"{cfg} {g[g.config == cfg].cycles_mean.iloc[0]:,.0f}cyc "
            f"sp={g[g.config == cfg].speedup_mean.iloc[0]:.2f}x "
            f"hit={g[g.config == cfg].hit_rate_mean.iloc[0]:.3f}"
            for cfg in ("baseline", "stride", "nn"))
        print(f"  {wl:18s} | {seg}")
    print(f"\nStored under: {res['outdir']}  (wall {time.perf_counter() - t0:.1f}s)")
    if args.aggscan:
        scan_aggregators(root=args.root, seeds=seeds, length=length)
    print("\nNow rebuild the dashboard to see it:")
    print("  python webapp/make_dashboard.py")


if __name__ == "__main__":
    main()
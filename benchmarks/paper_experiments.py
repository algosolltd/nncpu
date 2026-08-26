"""Run the battery of experiments required for the paper.

Everything is stored under ``results/<name>/`` (raw rows, aggregated means,
and the settings that produced them) so every number below is reproducible::

    python benchmarks/paper_experiments.py            # full battery (~4 min)
    python benchmarks/paper_experiments.py --quick    # tiny subset for CI

Battery:
  1. phased_switch     -- gate must prefetch in easy phases and stop in noisy
                         ones (rolling hit-rate per phase, per config)
  2. gate_ablation     -- NN with vs without the confidence gate
  3. threshold_sweep   -- gate threshold in {4,8,16,32,64} words
  4. mlp_sweep         -- batch/hidden/learning-rate grid
  5. machine_sweep     -- L1 size / line size / DRAM latency grid
  6. final_30          -- 30-seed canonical numbers on all workloads
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from nncpu.cpu import MachineConfig
from nncpu.benchmark import CONFIGS, run_workload, summarize
from nncpu.prefetchers import NNPrefetcher
from nncpu.workloads import WORKLOADS, build_workloads, phased_switch

RESULTS_ROOT = "results"

CORE_WORKLOADS = (
    "sequential_read",
    "strided_read",
    "mixed_read_write",
    "random_read",
    "phased_switch",
)

BASE_NN = dict(batch_size=16, hidden_layers=(32,), learning_rate=1e-2,
               mlp_backend="numpy")


def _loads(length):
    return build_workloads(length)


def _row(seed, label, workload, config, report, extra=None):
    r = summarize(report)
    row = {
        "label": label, "seed": seed, "workload": workload, "config": config,
        "cycles": r["cycles"], "hit_rate": r["hit_rate"], "ipc": r["ipc"],
        "mem_cycles": r["mem_cycles"], "prefetch_issued": r["prefetch_issued"],
        "prefetch_used": r["prefetch_used"],
    }
    if extra:
        row.update(extra)
    return row


def save_scan(name, rows, description):
    outdir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "scan_rows.csv"), index=False)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"description": description,
                   "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return df, outdir


def pivot_mean(df):
    """Aggregate cycles & hit-rate -> (label, workload) x config."""
    out = []
    for (label, workload), g in df.groupby(["label", "workload"], sort=False):
        base = g[g.config == "baseline"]["cycles"].mean()
        rec = {"label": label, "workload": workload}
        for cfg in g.config.unique():
            s = g[g.config == cfg]
            rec[f"{cfg}_cycles"] = s["cycles"].mean()
            rec[f"{cfg}_hit"] = s["hit_rate"].mean()
            if cfg != "baseline" and base:
                rec[f"{cfg}_speedup"] = base / s["cycles"].mean()
        out.append(rec)
    return pd.DataFrame(out)


def print_scan(df):
    pivot = pivot_mean(df)
    for (label, workload), g in pivot.groupby(["label", "workload"], sort=False):
        r = g.iloc[0]
        seg = "  ".join(
            f"{cfg} {r.get(cfg + '_cycles', float('nan')):.0f}cyc "
            f"hit={r.get(cfg + '_hit', float('nan')):.3f} "
            f"sp={r.get(cfg + '_speedup', 1.0):.2f}x"
            for cfg in CONFIGS
        )
        print(f"  [{label}] {workload:18s} | {seg}")
    print()


def scan_machine(name, variants, workloads, configs, seeds, length,
                 extra_nn=None, description=""):
    """variants: list of (label, dict-of-machine-field-overrides)."""
    workloads_cache = _loads(length)
    rows = []
    for label, over in variants:
        machine = MachineConfig(**over)
        for seed in range(seeds):
            nn_kwargs = dict(BASE_NN, random_state=seed)
            if extra_nn:
                nn_kwargs.update(extra_nn)
            for wname in workloads:
                insts = workloads_cache[wname]
                for cfg in configs:
                    rep = run_workload(insts, cfg, machine=machine, nn_kwargs=nn_kwargs)
                    rows.append(_row(seed, label, wname, cfg, rep))
    df, outdir = save_scan(name, rows, description + f" [machine overrides: {variants}]")
    pivot_mean(df).to_csv(os.path.join(outdir, "scan_mean.csv"), index=False)
    print(f"\n### {name}")
    print_scan(df)
    return df


def scan_nn(name, variants, workloads, configs, seeds, length, description=""):
    """variants: list of (label, dict-of-NN-kwargs overrides)."""
    workloads_cache = _loads(length)
    rows = []
    for label, over in variants:
        for seed in range(seeds):
            nn_kwargs = dict(BASE_NN, random_state=seed)
            nn_kwargs.update(over)
            for wname in workloads:
                insts = workloads_cache[wname]
                for cfg in configs:
                    rep = run_workload(insts, cfg, nn_kwargs=nn_kwargs)
                    rows.append(_row(seed, label, wname, cfg, rep))
    df, outdir = save_scan(name, rows, description + f" [NN variants: {variants}]")
    pivot_mean(df).to_csv(os.path.join(outdir, "scan_mean.csv"), index=False)
    print(f"\n### {name}")
    print_scan(df)
    return df


def analyze_phases(name, seeds, length, phase_len):
    """Per-phase hit rate for the phased_switch workload.  For a pure LOAD
    stream a miss costs FETCH(1)+MEM(40)=41 cycles, a hit 2: recoverable
    from per-instruction cycle counts."""
    outdir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    insts = list(phased_switch(length, phase_len=phase_len))
    n_phases = len(insts) // phase_len

    rows = []
    for config in ("baseline", "stride", "nn"):
        for seed in range(seeds):
            nn_kwargs = dict(BASE_NN, random_state=seed)
            rep = run_workload(insts, config, nn_kwargs=nn_kwargs)
            cycles = np.asarray(rep.inst_cycles)
            assert len(cycles) == len(insts)
            hitmask = cycles < 40  # pure LOAD workload: 2 = hit, 41 = miss
            for phase in range(n_phases):
                seg = hitmask[phase * phase_len:(phase + 1) * phase_len]
                kind = "predictable" if phase % 2 == 0 else "noisy"
                rows.append({
                    "config": config, "seed": seed, "phase": phase,
                    "phase_kind": kind,
                    "hit_rate": float(seg.mean()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "phases.csv"), index=False)
    agg = df.groupby(["config", "phase_kind"])["hit_rate"].agg(["mean", "std", "count"])
    print(f"\n### {name}  (phased workload, {n_phases} phases x {seeds} seeds)")
    print(agg.round(3).to_string())
    print()

    fig = _plot_phases(df, n_phases)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "phases.png"), bbox_inches="tight", dpi=200)
    return df


def _plot_phases(df, n_phases):
    """Two clear panels: (a) hit rate by phase kind (bars + std),
    (b) hit rate per phase as LINES with the noisy segments shaded grey."""
    import matplotlib.pyplot as plt

    colors = {"baseline": "#64748b", "stride": "#2563eb", "nn": "#d97706"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.7),
                                   gridspec_kw={"width_ratios": [1.15, 2.2]})

    # (a) mean hit rate per phase kind
    agg = df.groupby(["config", "phase_kind"])["hit_rate"].agg(["mean", "std"])
    kinds = ["predictable", "noisy"]
    width = 0.27
    for i, cfg in enumerate(("baseline", "stride", "nn")):
        means = [agg.loc[(cfg, k), "mean"] for k in kinds]
        stds = [agg.loc[(cfg, k), "std"] for k in kinds]
        ax1.bar([k + (i - 1) * width for k in range(2)], means, width,
                yerr=stds, capsize=3, color=colors[cfg], label=cfg)
    # reference lines: the no-prefetch hit rate in each phase kind
    b_pred = agg.loc[("baseline", "predictable"), "mean"]
    b_noisy = agg.loc[("baseline", "noisy"), "mean"]
    ax1.axhline(b_noisy, color="#0f172a", ls="--", lw=1.2,
                label="no-prefetch (noisy)")
    ax1.axhline(b_pred, color="#94a3b8", ls="--", lw=1.2,
                label="no-prefetch (predictable)")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["predictable", "noisy"])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("mean hit rate")
    ax1.set_title("(a) by phase kind", fontsize=10)
    ax1.legend(loc="lower left", ncols=2, fontsize=7)
    ax1.grid(alpha=0.25)

    # (b) per-phase lines; noisy phases shaded
    agg_line = df.groupby(["config", "phase"], sort=True)["hit_rate"] \
                 .mean().reset_index()
    kind_of = df.drop_duplicates("phase").set_index("phase")["phase_kind"]
    xs = sorted(df.phase.unique())
    for i, cfg in enumerate(("baseline", "stride", "nn")):
        d = agg_line[agg_line.config == cfg]
        ax2.plot(d.phase, d.hit_rate, marker="o", lw=2.2, ms=4,
                 color=colors[cfg], label=cfg)
    for x in xs:
        if kind_of[x] == "noisy":
            ax2.axvspan(x - 0.5, x + 0.5, color="grey", alpha=0.2, lw=0)
    ax2.set_xlabel("phase index  (grey = noisy segment)")
    ax2.set_ylabel("hit rate")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) per phase", fontsize=10)
    ax2.legend(loc="lower left", ncols=3, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phased workload: L1 hit rate", fontsize=11)
    fig.tight_layout()
    return fig


def final_experiment(seeds, length):
    from nncpu.experiment import ExperimentConfig, run_experiment
    cfg = ExperimentConfig(
        name="final_30", description=f"canonical {seeds} seeds, length {length}",
        length=length, runs=seeds, seed=0,
        workloads=CORE_WORKLOADS, configs=CONFIGS,
        machine=MachineConfig(), nn_kwargs=dict(BASE_NN),
        detail=False,
    )
    # NOTE: do NOT pin random_state here: run_experiment seeds each run
    # (seed = base_seed + run_index) so the 95% CI is a real statistic.
    result = run_experiment(cfg, root=RESULTS_ROOT)
    print(f"\n### final_30 stored in {result.outdir}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--length", type=int, default=2000)
    args = ap.parse_args()

    if args.quick:
        seeds_ph, seeds_abl, seeds_thr, seeds_mlp, seeds_mac = 2, 2, 2, 2, 2
        thr = (4, 16)
        mlp_variants = [("b16h32", dict(BASE_NN)), ("b32h64_lr1e-3", dict(
            batch_size=32, hidden_layers=(64,), learning_rate=1e-3))]
        mac_variants = [("l1_32_l40", {}), ("l1_64_l80", dict(l1_lines=64, mem_latency=80))]
        length = 600
    else:
        seeds_ph, seeds_abl, seeds_thr, seeds_mlp, seeds_mac = 10, 10, 5, 3, 2
        thr = (4, 8, 16, 32, 64)
        mlp_variants = [
            ("b8_h16", dict(batch_size=8, hidden_layers=(16,))),
            ("b8_h32", dict(batch_size=8)),
            ("b16_h16", dict(hidden_layers=(16,))),
            ("b16_h32_lr1e-3", dict(learning_rate=1e-3)),
            ("b32_h32", dict(batch_size=32)),
            ("b32_h64", dict(batch_size=32, hidden_layers=(64,))),
        ]
        mac_variants = [
            ("l1_16", dict(l1_lines=16)),
            ("l1_64", dict(l1_lines=64)),
            ("l1_32_lat20", dict(mem_latency=20)),
            ("l1_32_lat80", dict(mem_latency=80)),
            ("line_4", dict(line_size=4)),
            ("line_16", dict(line_size=16)),
        ]
        length = args.length

    t0 = time.perf_counter()
    print(f"nncpu experiment battery | {length} instrs | numpy MLP")

    # 1. phased workload: gate on/off over time
    analyze_phases("exp_phased", seeds_ph, length=min(length, 2400), phase_len=min(200, length // 4))

    # 2. gate ablation on the decisive workloads
    scan_nn(
        "exp_gate_ablation",
        [("gate_off", dict(confidence_gate=False)),
         ("gate_on", dict(confidence_gate=True))],
        ("sequential_read", "strided_read", "random_read"),
        CONFIGS, seeds_abl, length,
        description="confidence gate on/off",
    )

    # 3. gate threshold sweep
    scan_nn(
        "exp_threshold_sweep",
        [(f"thr_{t}", dict(gate_threshold=float(t))) for t in thr],
        ("sequential_read", "strided_read", "random_read", "mixed_read_write"),
        CONFIGS, seeds_thr, length,
        description="gate threshold in words",
    )

    # 4. MLP hyperparameter grid
    scan_nn(
        "exp_mlp_sweep",
        mlp_variants,
        ("sequential_read", "strided_read", "random_read"),
        CONFIGS, seeds_mlp, length,
        description="batch/hidden/lr grid",
    )

    # 5. machine sensitivity
    scan_machine(
        "exp_machine_sweep",
        mac_variants,
        ("sequential_read", "strided_read", "random_read", "mixed_read_write"),
        CONFIGS, seeds_mac, length,
        description="L1 size / line size / DRAM latency",
    )

    # 6. canonical 30-seed numbers
    total_seeds = 4 if args.quick else 30
    final_experiment(total_seeds, length)

    print(f"\nTotal battery wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
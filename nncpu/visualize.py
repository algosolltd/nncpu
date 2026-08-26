"""Generate the benchmark visualizations (3 PNG files)."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .benchmark import CONFIGS, cycles_of

CONFIG_LABELS = {
    "baseline": "Baseline (no prefetch)",
    "stride": "Stride prefetcher",
    "nn": "NN prefetcher",
}


def _collect_inst_cycles(workloads: dict, reports: dict) -> pd.DataFrame:
    """Tidy per-instruction cycle samples for the boxplot."""
    rows = []
    for workload in workloads:
        for config in CONFIGS:
            report = reports[(workload, config)]
            for i, cycles in enumerate(report.inst_cycles):
                rows.append(
                    {
                        "workload": workload,
                        "config": config,
                        "config_label": CONFIG_LABELS[config],
                        "instruction": i,
                        "cycles": cycles,
                    }
                )
    return pd.DataFrame(rows)


def save_all(workloads: dict, reports: dict, df: pd.DataFrame, workdir: str = ".") -> list:
    """Render the three figures next to ``df`` and return the file paths."""
    paths = [
        _plot_operation_comparison(workloads, reports, workdir),
        _plot_memory_vs_arithmetic(df, workdir),
        _plot_performance_metrics(df, workdir),
    ]
    return paths


def _plot_operation_comparison(workloads: dict, reports: dict, workdir: str) -> str:
    """1) Per-instruction cycle cost by workload and config (log boxplot)."""
    sample = _collect_inst_cycles(workloads, reports)
    sample = sample[sample["cycles"] > 0]

    plt.figure(figsize=(14, 7))
    order = list(workloads)
    sns.boxplot(
        x="workload", y="cycles", hue="config_label",
        data=sample, order=order,
        palette="Set2", fliersize=2,
    )
    plt.yscale("log")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Workload")
    plt.ylabel("Cycles per instruction (log)")
    plt.title("Per-instruction cost across workloads and prefetch configs")
    plt.tight_layout()
    path = f"{workdir}/operation_comparison_log.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def _plot_memory_vs_arithmetic(df: pd.DataFrame, workdir: str) -> str:
    """2) Total memory-stall cycles vs arithmetic cycles per config."""
    labels = ["Memory-stall cycles", "Arithmetic cycles"]
    x = np.arange(len(labels))
    width = 0.25

    def total(col: str, config: str) -> float:
        subset = df[df.config == config]
        return float(subset[col].sum()) if not subset.empty else 0.0

    plt.figure(figsize=(12, 7))
    for i, config in enumerate(CONFIGS):
        means = [
            total("mem_cycles", config) / len(df[df.config == config]) if len(df[df.config == config]) else 0.0,
            total("arith_cycles", config) / len(df[df.config == config]) if len(df[df.config == config]) else 0.0,
        ]
        plt.bar(x + (i - 1) * width, means, width, label=CONFIG_LABELS[config])

    plt.yscale("log")
    plt.xticks(x, labels)
    plt.ylabel("Average total cycles (log)")
    plt.title("Memory-stall vs arithmetic cycles, per config (averaged over workloads)")
    plt.legend()
    plt.tight_layout()
    path = f"{workdir}/memory_vs_arithmetic_log.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def _plot_performance_metrics(df: pd.DataFrame, workdir: str) -> str:
    """3) Hit rate, prefetch accuracy and speedup per workload/config."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    workloads = sorted(df.workload.unique())
    metric_cols = [
        ("hit_rate", "Cache hit rate", axes[0]),
        ("prefetch_accuracy", "Prefetch accuracy (used / issued)", axes[1]),
        ("", "Speedup vs baseline", axes[2]),
    ]

    for i, (metric, title, ax) in enumerate(metric_cols):
        width = 0.25
        for j, config in enumerate(CONFIGS):
            values = [
                df[(df.workload == w) & (df.config == config)][metric].iloc[0]
                if metric else df[(df.workload == w) & (df.config == config)]["cycles"].iloc[0]
                for w in workloads
            ]
            if not metric:  # speedup panel: convert cycles -> speedup
                base_cycles = [
                    cycles_of(df, w, "baseline") for w in workloads
                ]
                values = [
                    (b / v if v else 1.0) for b, v in zip(base_cycles, values)
                ]
            offset = (j - 1) * width
            ax.bar(np.arange(len(workloads)) + offset, values, width,
                   label=CONFIG_LABELS[config])
        ax.set_xticks(np.arange(len(workloads)))
        ax.set_xticklabels(workloads, rotation=20, ha="right")
        ax.set_title(title)
        if i == 2:
            ax.axhline(1.0, color="grey", linestyle="--", lw=1)
    axes[0].legend()
    plt.tight_layout()
    path = f"{workdir}/performance_metrics_log.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


# -- publication figures for multi-run experiments -------------------------


def plot_experiment_figures(config, summary: pd.DataFrame, outdir: str) -> list:
    """Render error-bar figures from an aggregated experiment summary.

    Returns the list of generated ``.png`` paths.
    """
    os.makedirs(outdir, exist_ok=True)
    del config  # future-proof API; figures are currently generic
    paths = [
        figure_speedup(summary, outdir),
        figure_hit_rate(summary, outdir),
        figure_cycles(summary, outdir),
    ]
    return paths


def _grouped_bars(
    ax,
    summary: pd.DataFrame,
    workload_order: list,
    metric: str,
    yerr_metric: str,
):
    """One group per workload, one bar per config, with error bars."""
    width = 0.8 / max(len(summary.config.unique()), 1)
    x = np.arange(len(workload_order))
    for i, config in enumerate(summary.config.unique()):
        values = []
        errs = []
        for w in workload_order:
            s = summary[(summary.workload == w) & (summary.config == config)]
            values.append(float(s[metric].iloc[0]) if not s.empty else float("nan"))
            errs.append(float(s[yerr_metric].iloc[0]) if not s.empty else 0.0)
        offset = (i - (len(summary.config.unique()) - 1) / 2) * width
        ax.bar(x + offset, values, width, yerr=errs, capsize=3,
               label=CONFIG_LABELS.get(config, config))
    ax.set_xticks(x)
    ax.set_xticklabels(workload_order, rotation=20, ha="right")


def figure_speedup(summary: pd.DataFrame, outdir: str) -> str:
    """Mean speedup vs baseline per workload/config, with std bars."""
    solos = [c for c in summary.config.unique() if c != "baseline"]
    plot_df = summary[summary.config.isin(solos)]
    if plot_df.empty or "speedup_mean" not in plot_df.columns:
        return ""
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    _grouped_bars(ax, plot_df, list(plot_df.workload.unique()),
                  "speedup_mean", "speedup_std")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_ylabel("Speedup vs baseline (no prefetch)")
    ax.set_title("Mean speedup over seeds (error bars = 1 std)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "speedup.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def figure_hit_rate(summary: pd.DataFrame, outdir: str) -> str:
    """Mean L1 hit rate per workload/config, with 95% CI bars."""
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    _grouped_bars(ax, summary, list(summary.workload.unique()),
                  "hit_rate_mean", "hit_rate_ci95")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("L1 hit rate")
    ax.set_title("Mean L1 hit rate (error bars = 95% CI)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "hit_rate.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def figure_cycles(summary: pd.DataFrame, outdir: str) -> str:
    """Mean total cycles per workload/config, with 95% CI bars (log)."""
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    _grouped_bars(ax, summary, list(summary.workload.unique()),
                  "cycles_mean", "cycles_ci95")
    ax.set_yscale("log")
    ax.set_ylabel("Total cycles (log)")
    ax.set_title("Mean total cycles (error bars = 95% CI)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "cycles.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path
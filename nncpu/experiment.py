"""Reproducible, multi-run experiments with on-disk result storage.

Every experiment is a fully-specified configuration (machine, prefetch
configs, workload set, stream length, NN hyperparameters, seed policy)
that is written verbatim next to its measured results, together with a
provenance manifest (git revision, library versions, platform).  Results
are stored as:

* ``config.json``     -- the complete experiment specification
* ``manifest.json``   -- provenance: git rev, versions, host, timestamp
* ``runs.csv``        -- one row per (seed, workload, config)
* ``summary.csv``     -- per (workload, config): mean, std, 95% CI
* ``speedups.csv``    -- per-run speedup vs baseline, aggregated
* ``table.tex``       -- a booktabs LaTeX table for the paper
* ``figures/*.png``   -- publication figures built from the summary
* ``README.txt``      -- one-glance reproducibility notes
"""

import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .benchmark import CONFIGS, run_workload, summarize
from .cpu import MachineConfig
from .prefetchers import make_prefetcher
from .visualize import plot_experiment_figures
from .workloads import WORKLOADS, build_workloads
from ._version import __version__ as nncpu_version


@dataclass
class ExperimentConfig:
    """Everything needed to re-run an experiment, byte-for-byte."""

    name: str
    description: str = ""
    length: int = 2000
    workloads: tuple = tuple(WORKLOADS)
    configs: tuple = CONFIGS
    runs: int = 10         # seeds; run r uses random_state = seed + r
    seed: int = 0
    machine: MachineConfig = field(default_factory=MachineConfig)
    nn_kwargs: dict = field(default_factory=dict)
    detail: bool = False   # also store per-instruction cycle series (large)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["machine"] = self.machine.as_dict()
        return d

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.workloads)} workloads x {len(self.configs)} "
            f"configs x {self.runs} seeds, length={self.length}, "
            f"{self.machine.describe()}"
        )


@dataclass
class ExperimentResult:
    """The measured outcome of one experiment."""

    config: ExperimentConfig
    runs_df: pd.DataFrame
    summary_df: pd.DataFrame
    outdir: str

    @property
    def figures(self) -> list:
        figdir = os.path.join(self.outdir, "figures")
        return sorted(
            os.path.join(figdir, f) for f in os.listdir(figdir) if f.endswith(".png")
        )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _dirty_working_tree() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(out.stdout.strip())
    except Exception:
        return True


def make_manifest(config: ExperimentConfig) -> dict:
    """Provenance record for one experiment run."""
    import sklearn
    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version
    from matplotlib import __version__ as mpl_version
    import seaborn as sns

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nncpu_version": nncpu_version,
        "git_revision": _git_revision(),
        "git_dirty": _dirty_working_tree(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {
            "numpy": np_version,
            "pandas": pd_version,
            "scikit_learn": sklearn.__version__,
            "matplotlib": mpl_version,
            "seaborn": sns.__version__,
        },
    }


def _stats(values: pd.Series) -> dict:
    """Mean, sample std, and 95% CI (normal approximation) of ``values``."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def run_experiment(config: ExperimentConfig, root: str = "results") -> ExperimentResult:
    """Run ``config.runs`` seeds of every (workload, config) and persist.

    Returns the :class:`ExperimentResult` with ``runs_df`` (tidy, per run)
    and ``summary_df`` (aggregated means + error bars).
    """
    workloads = build_workloads(config.length)
    workloads = {
        name: insts for name, insts in workloads.items() if name in set(config.workloads)
    }

    rows = []
    for run in range(config.runs):
        seed = config.seed + run
        nn_kwargs = dict(config.nn_kwargs)
        nn_kwargs.setdefault("random_state", seed)
        for workload_name, instructions in workloads.items():
            for cfg in config.configs:
                report = run_workload(
                    instructions, cfg, machine=config.machine, nn_kwargs=nn_kwargs
                )
                row = summarize(report)
                row["run"] = run
                row["seed"] = seed
                row["workload"] = workload_name
                row["config"] = cfg
                rows.append(row)
    runs_df = pd.DataFrame(rows)

    # speedup is measured against the *mean* baseline cycles per workload
    if "baseline" in runs_df.config.values:
        base_mean = (
            runs_df[runs_df.config == "baseline"]
            .groupby("workload")["cycles"]
            .mean()
            .rename("base_cycles")
        )
        runs_df = runs_df.merge(base_mean, on="workload", how="left")
        runs_df["speedup"] = runs_df["base_cycles"] / runs_df["cycles"]

    summary_df = aggregate(runs_df)
    outdir = _write_experiment(config, runs_df, summary_df, root)
    return ExperimentResult(config=config, runs_df=runs_df, summary_df=summary_df, outdir=outdir)


def aggregate(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-run frame -> per (workload, config) mean/std/CI summary."""
    groups = runs_df.groupby(["workload", "config"])
    records = []
    for (workload, cfg), g in groups:
        rec = {"workload": workload, "config": cfg, "runs": len(g)}
        cyc = _stats(g["cycles"])
        for k, v in cyc.items():
            rec[f"cycles_{k}"] = v
        rec["mem_cycles_mean"] = g["mem_cycles"].mean()
        rec["arith_cycles_mean"] = g["arith_cycles"].mean()
        for metric in ("hit_rate", "ipc", "prefetch_accuracy"):
            s = _stats(g[metric])
            rec[f"{metric}_mean"] = s["mean"]
            rec[f"{metric}_std"] = s["std"]
            rec[f"{metric}_ci95"] = s["ci95"]
        if "speedup" in g.columns:
            s = _stats(g["speedup"])
            rec["speedup_mean"] = s["mean"]
            rec["speedup_std"] = s["std"]
            rec["speedup_ci95"] = s["ci95"]
            rec["speedup_min"] = float(g["speedup"].min())
            rec["speedup_max"] = float(g["speedup"].max())
        records.append(rec)
    return pd.DataFrame(records)


def _write_experiment(
    config: ExperimentConfig,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    root: str,
) -> str:
    outdir = os.path.join(root, config.name)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config.as_dict(), f, indent=2)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(make_manifest(config), f, indent=2)

    runs_df.to_csv(os.path.join(outdir, "runs.csv"), index=False)

    if "speedup" in runs_df.columns:
        speedup_df = runs_df[["run", "seed", "workload", "config", "cycles", "speedup"]]
    else:
        speedup_df = runs_df[["run", "seed", "workload", "config", "cycles"]]
    speedup_df.to_csv(os.path.join(outdir, "speedups.csv"), index=False)

    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    with open(os.path.join(outdir, "table.tex"), "w") as f:
        f.write(experiment_to_latex(summary_df))

    figures = plot_experiment_figures(config, summary_df, figdir)
    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write(_readme_text(config, [os.path.relpath(p, outdir) for p in figures]))

    return outdir


# -- LaTeX ------------------------------------------------------------------


def experiment_to_latex(summary_df: pd.DataFrame) -> str:
    """One booktabs table, one row per workload, grouped by config."""
    blocks = summary_df.config.unique()
    specs = "ccc" * len(blocks)
    header = ["\\textbf{Workload}"]
    for cfg in blocks:
        header += [f"\\textbf{{{cfg.capitalize()} cycles}}",
                   "\\textbf{Hit}", "\\textbf{Speedup}"]
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{\\texttt{nncpu} experiment summary.  Cycles are the",
        "  mean over seeds with 95\\% CI; hit is the mean L1 hit rate;",
        "  speedup is relative to the baseline (no-prefetch) mean.",
        "  \\label{tab:results}}",
        f"  \\begin{{tabular}}{{l{specs}}}",
    ]
    lines.append("    " + " & ".join(header) + " \\\\")
    lines.append("    \\midrule")

    for workload, g in summary_df.groupby("workload", sort=False):
        row = [f"\\texttt{{{workload}}}"]
        for cfg in blocks:
            s = g[g.config == cfg]
            if s.empty:
                row += ["---", "---", "---"]
                continue
            s = s.iloc[0]
            cyc = f"{s['cycles_mean']:,.0f} $\\pm$ {s['cycles_ci95']:.0f}"
            hit = f"{s['hit_rate_mean']:.3f}"
            sp = f"{s['speedup_mean']:.2f}$\\times$" if "speedup_mean" in s else "--"
            row += [cyc, hit, sp]
        lines.append("    " + " & ".join(row) + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def summary_table(summary: pd.DataFrame) -> str:
    """Human-readable console table of the aggregated results."""
    lines = []
    for workload, g in summary.groupby("workload", sort=False):
        lines.append(f"\n{workload}")
        for _, s in g.iterrows():
            cyc = f"{s['cycles_mean']:,.0f} +/- {s['cycles_ci95']:.0f}"
            hr = f"hit={s['hit_rate_mean']:.3f}"
            extra = ""
            if s["config"] != "baseline" and "speedup_mean" in s.index:
                extra = f"  speedup {s['speedup_mean']:.2f}x +/- {s['speedup_std']:.2f}"
            lines.append(f"  {s['config']:9s} cycles={cyc}  {hr}{extra}")
    return "\n".join(lines)


def _readme_text(config: ExperimentConfig, figures: list) -> str:
    m = make_manifest(config)
    body = [
        f"Experiment: {config.name}",
        f"Description: {config.description or '(none)'}",
        "",
        config.describe(),
        "",
        f"Generated: {m['generated_at']}",
        f"Git revision: {m['git_revision']} (dirty: {m['git_dirty']})",
        f"Python: {m['python']}  platform: {m['platform']}",
        "Library versions: " + ", ".join(f"{k}={v}" for k, v in m["versions"].items()),
        "",
        "Files:",
        "  config.json   - exact experiment specification",
        "  manifest.json - provenance (git rev, versions, host)",
        "  runs.csv      - one row per (seed, workload, config)",
        "  summary.csv   - aggregated means, stds, 95% CI",
        "  speedups.csv  - per-run speedups vs baseline",
        "  table.tex     - LaTeX booktabs table",
        "Figures:",
    ]
    body += [f"  {f}" for f in figures]
    body.append("")
    body.append("Reproduce:  python main.py --name " + config.name)
    return "\n".join(body) + "\n"
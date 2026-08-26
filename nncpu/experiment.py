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
import hashlib
import importlib.metadata
import math
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import CONFIGS, run_workload, summarize
from .cpu import MachineConfig
from .prefetchers import resolved_nn_kwargs
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
    vary_workload_seed: bool = True
    machine: MachineConfig = field(default_factory=MachineConfig)
    nn_kwargs: dict = field(default_factory=dict)
    detail: bool = False   # also store per-instruction cycle series (large)

    def __post_init__(self) -> None:
        self.workloads = tuple(self.workloads)
        self.configs = tuple(self.configs)
        if not self.name or os.path.isabs(self.name) or os.path.basename(self.name) != self.name:
            raise ValueError("experiment name must be a non-empty directory name")
        if self.length <= 0:
            raise ValueError("length must be > 0")
        if self.runs <= 0:
            raise ValueError("runs must be > 0")
        if not self.workloads:
            raise ValueError("at least one workload is required")
        unknown_workloads = set(self.workloads) - set(WORKLOADS)
        if unknown_workloads:
            raise ValueError(f"unknown workloads: {sorted(unknown_workloads)}")
        if not self.configs:
            raise ValueError("at least one prefetch config is required")
        from .baselines import SIM_CONFIGS
        unknown_configs = set(self.configs) - set(SIM_CONFIGS)
        if unknown_configs:
            raise ValueError(f"unknown prefetch configs: {sorted(unknown_configs)}")
        # Validate names and normalize all implicit defaults now, before a run.
        resolved_nn_kwargs(self.nn_kwargs)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["machine"] = self.machine.as_dict()
        nn_settings = resolved_nn_kwargs(self.nn_kwargs)
        # When not explicitly pinned, random_state is derived from
        # ExperimentConfig.seed + run and must not be frozen to constructor
        # default 42 in the persisted settings.
        if "random_state" not in self.nn_kwargs:
            nn_settings.pop("random_state", None)
        d["nn_kwargs"] = json.loads(json.dumps(nn_settings))
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Load the exact configuration persisted in ``config.json``."""
        values = dict(data)
        values["machine"] = MachineConfig(**values.get("machine", {}))
        return cls(**values)

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


def source_digest() -> str:
    """SHA-256 of files that can affect simulation or experiment outputs."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "main.py", root / "requirements.txt", root / "pytest.ini"]
    for directory in (root / "nncpu", root / "benchmarks", root / "webapp"):
        paths.extend(directory.rglob("*.py"))
        paths.extend(directory.rglob("*.html"))
    digest = hashlib.sha256()
    for path in sorted({p for p in paths if p.exists()}):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def make_manifest(config: ExperimentConfig) -> dict:
    """Provenance record for one experiment run."""
    config_json = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nncpu_version": nncpu_version,
        "git_revision": _git_revision(),
        "git_dirty": _dirty_working_tree(),
        "source_sha256": source_digest(),
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {
            "numpy": _installed_version("numpy"),
            "pandas": _installed_version("pandas"),
            "scikit_learn": _installed_version("scikit-learn"),
            "matplotlib": _installed_version("matplotlib"),
            "seaborn": _installed_version("seaborn"),
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
    # Snapshot provenance before writing into a tracked results directory.
    manifest = make_manifest(config)
    rows = []
    for run in range(config.runs):
        seed = config.seed + run
        workloads = build_workloads(
            config.length, seed=seed if config.vary_workload_seed else None
        )
        workloads = {
            name: insts
            for name, insts in workloads.items()
            if name in set(config.workloads)
        }
        nn_kwargs = dict(config.nn_kwargs)
        nn_kwargs.setdefault("random_state", seed)
        nn_kwargs = resolved_nn_kwargs(nn_kwargs)
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
                if config.detail:
                    row["inst_cycles"] = json.dumps(report.inst_cycles)
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
    outdir = _write_experiment(config, runs_df, summary_df, root, manifest)
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
    manifest: dict,
) -> str:
    outdir = os.path.join(root, config.name)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config.as_dict(), f, indent=2)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

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
        f.write(_readme_text(
            config,
            [os.path.relpath(p, outdir) for p in figures],
            manifest,
            os.path.relpath(os.path.join(outdir, "config.json")),
        ))

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


def _readme_text(
    config: ExperimentConfig, figures: list, manifest: dict, config_path: str
) -> str:
    m = manifest
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
    body.append(f"Reproduce:  python main.py --config {config_path}")
    return "\n".join(body) + "\n"

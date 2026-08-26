"""Benchmark harness: runs every workload under every prefetch config."""

from typing import Optional, Sequence

import pandas as pd

from .cpu import CPU, CPUReport, MachineConfig
from .prefetchers import Prefetcher, make_prefetcher
from .workloads import build_workloads

CONFIGS = ("baseline", "stride", "nn")


def run_workload(
    instructions: list,
    config: str,
    prefetcher: Optional[Prefetcher] = None,
    machine: Optional["MachineConfig"] = None,
    nn_kwargs: Optional[dict] = None,
) -> CPUReport:
    """Execute one instruction stream under one prefetch config."""
    if prefetcher is None:
        from .baselines import make_sim_prefetcher
        prefetcher = make_sim_prefetcher(config, machine=machine, **nn_kwargs or {})
    cpu = CPU(prefetcher=prefetcher, machine=machine)
    for instruction in instructions:
        cpu.execute(instruction)
    return cpu.report()


def summarize(report: CPUReport) -> dict:
    """Flatten a run into one row of the results table."""
    return {
        "instructions": report.instructions,
        "cycles": report.cycles,
        "ipc": report.ipc,
        "cycles_per_instruction": report.cycles_per_instruction,
        "hit_rate": report.hit_rate,
        "prefetch_issued": report.prefetch_issued,
        "prefetch_used": report.prefetch_used,
        "prefetch_accuracy": report.prefetch_accuracy,
        "mem_cycles": report.mem_cycles,
        "arith_cycles": report.arith_cycles,
        "wall_seconds": report.wall_seconds,
    }


def run_all(
    workloads: Optional[dict] = None,
    length: int = 2000,
    machine: Optional["MachineConfig"] = None,
    nn_kwargs: Optional[dict] = None,
    configs: Sequence[str] = CONFIGS,
) -> tuple:
    """Run all configs against all workloads.

    Returns ``(DataFrame, reports)`` where the frame has one row per
    (workload, config) and ``reports`` maps ``(workload, config)`` to the
    raw :class:`~nncpu.cpu.CPUReport`.
    """
    if workloads is None:
        workloads = build_workloads(length)

    rows = []
    reports = {}
    for workload_name, instructions in workloads.items():
        for config in configs:
            report = run_workload(
                instructions, config, machine=machine, nn_kwargs=nn_kwargs
            )
            row = summarize(report)
            row["workload"] = workload_name
            row["config"] = config
            rows.append(row)
            reports[(workload_name, config)] = report
    return pd.DataFrame(rows), reports


def cycles_of(df: pd.DataFrame, workload: str, config: str) -> float:
    """Look up total cycles for a (workload, config) pair."""
    row = df[(df.workload == workload) & (df.config == config)]
    return float(row["cycles"].iloc[0]) if not row.empty else float("nan")


def print_table(df: pd.DataFrame) -> None:
    """Print a compact, sorted summary of the results."""
    view = df.pivot(index="workload", columns="config",
                    values=["cycles", "hit_rate", "ipc"])

    print("\n=== Workload x Config results (cycles | hit-rate | ipc) ===")
    for workload in view.index:
        print(f"\n{workload}")
        for config in CONFIGS:
            cyc = int(view.loc[workload, ("cycles", config)])
            hr = view.loc[workload, ("hit_rate", config)]
            ipc = view.loc[workload, ("ipc", config)]
            base = int(view.loc[workload, ("cycles", "baseline")])
            speedup = base / cyc if cyc else 1.0
            tag = "" if config == "baseline" else f"  (speedup vs baseline: {speedup:.2f}x)"
            print(f"  {config:10s} cycles={cyc:7d}  hit_rate={hr:.3f}  ipc={ipc:.2f}{tag}")

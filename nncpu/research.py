"""Matched-control experiments for confidence-filtered prefetching.

This module keeps the scientific comparison behind one small interface:
``run_gate_study`` receives fully materialized workloads and profiles, then
returns raw paired rows, aggregates and direct matched-control contrasts.
It does not decide which outcome is favourable or paper-worthy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .baselines import make_sim_prefetcher
from .benchmark import run_workload, summarize
from .cpu import MachineConfig
from .prefetchers import (
    LookaheadPrefetcher,
    RegularityFilteredPrefetcher,
    StridePrefetcher,
)
from .baselines import BertiPrefetcher


RESEARCH_CONFIGS = (
    "baseline", "stride", "gated_stride", "regularity_stride", "berti",
    "gated_berti", "regularity_berti", "nn"
)
MATCHED_COMPARISONS = (
    ("gated_stride", "stride"),
    ("regularity_stride", "stride"),
    ("gated_berti", "berti"),
    ("regularity_berti", "berti"),
    ("nn", "gated_stride"),
    ("nn", "regularity_stride"),
)
REGULARITY_WINDOW = 16
REGULARITY_MIN_SUPPORT = 0.35


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    machine: MachineConfig
    distances: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty")
        if not self.distances or any(distance <= 0 for distance in self.distances):
            raise ValueError("profile distances must be positive")


def default_profiles() -> tuple[ResearchProfile, ...]:
    """Ideal and two latency/bandwidth profiles used by the gate study."""
    return (
        ResearchProfile("ideal_full", MachineConfig(prefetch_latency=0), (1,)),
        ResearchProfile(
            "timed_8way",
            MachineConfig(
                prefetch_latency=40,
                l1_associativity=8,
                memory_issue_interval=1,
                prefetch_mshr=16,
            ),
            (1, 8, 16, 32),
        ),
        ResearchProfile(
            "constrained_8way",
            MachineConfig(
                prefetch_latency=40,
                l1_associativity=8,
                memory_issue_interval=4,
                prefetch_mshr=8,
            ),
            (1, 8, 16),
        ),
    )


def _prefetcher(config: str, machine: MachineConfig, seed: int, distance: int):
    if config == "regularity_stride":
        predictor = RegularityFilteredPrefetcher(
            StridePrefetcher(),
            window=REGULARITY_WINDOW,
            min_support=REGULARITY_MIN_SUPPORT,
        )
    elif config == "regularity_berti":
        predictor = RegularityFilteredPrefetcher(
            BertiPrefetcher(),
            window=REGULARITY_WINDOW,
            min_support=REGULARITY_MIN_SUPPORT,
        )
    else:
        predictor = make_sim_prefetcher(config, machine=machine, random_state=seed)
    if predictor is not None and distance > 1:
        predictor = LookaheadPrefetcher(predictor, distance=distance)
    return predictor


def run_gate_study(
    workloads_by_seed: Mapping[int, Mapping[str, list]],
    profiles: Sequence[ResearchProfile] | None = None,
    configs: Sequence[str] = RESEARCH_CONFIGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a paired study and return ``(runs, summary, contrasts)``.

    Every configuration within a seed/profile/distance receives the same
    materialized stream. Baseline is evaluated once per profile and reused
    across lookahead distances because it emits no prediction.
    """
    profiles = tuple(profiles or default_profiles())
    unknown = set(configs) - set(RESEARCH_CONFIGS)
    if unknown:
        raise ValueError(f"unknown research configs: {sorted(unknown)}")
    rows = []
    for seed, workloads in sorted(workloads_by_seed.items()):
        for profile in profiles:
            for workload, instructions in workloads.items():
                for config in configs:
                    distances = (1,) if config == "baseline" else profile.distances
                    for distance in distances:
                        predictor = _prefetcher(
                            config, profile.machine, int(seed), distance
                        )
                        report = run_workload(
                            instructions,
                            config,
                            prefetcher=predictor,
                            machine=profile.machine,
                            nn_kwargs={"random_state": int(seed)},
                        )
                        row = summarize(report)
                        row.update(
                            seed=int(seed),
                            workload=workload,
                            profile=profile.name,
                            config=config,
                            distance=distance,
                        )
                        rows.append(row)
    runs = pd.DataFrame(rows)
    baseline = runs[runs.config == "baseline"][
        ["seed", "workload", "profile", "cycles"]
    ].rename(columns={"cycles": "base_cycles"})
    runs = runs.merge(
        baseline,
        on=["seed", "workload", "profile"],
        how="left",
        validate="many_to_one",
    )
    runs["speedup"] = runs.base_cycles / runs.cycles
    return runs, aggregate_research(runs), matched_contrasts(runs)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = values.to_numpy(dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, float(ci95)


def _exact_sign_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test, ignoring paired ties."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def _holm_adjust(values: Sequence[float]) -> list[float]:
    adjusted = [1.0] * len(values)
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    running = 0.0
    count = len(ordered)
    for rank, (value, original) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[original] = running
    return adjusted


def aggregate_research(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "cycles", "speedup", "hit_rate", "prefetch_accuracy",
        "prefetch_coverage", "prefetch_timeliness",
        "prefetch_pollution_rate", "prefetch_issued", "prefetch_dropped",
    )
    rows = []
    groups = runs.groupby(
        ["profile", "workload", "config", "distance"], sort=False
    )
    for keys, group in groups:
        row = dict(zip(("profile", "workload", "config", "distance"), keys))
        row["runs"] = len(group)
        for metric in metrics:
            mean, std, ci95 = _mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def matched_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    """Direct paired ratios for filtered-vs-raw and NN-vs-gated-stride."""
    rows = []
    keys = ["seed", "profile", "workload", "distance"]
    for candidate, control in MATCHED_COMPARISONS:
        left = runs[runs.config == candidate][
            keys + ["cycles", "prefetch_pollution_rate", "prefetch_issued"]
        ]
        right = runs[runs.config == control][
            keys + ["cycles", "prefetch_pollution_rate", "prefetch_issued"]
        ]
        paired = left.merge(right, on=keys, suffixes=("_candidate", "_control"))
        if paired.empty:
            continue
        paired["cycle_benefit"] = (
            paired.cycles_control / paired.cycles_candidate
        )
        paired["pollution_reduction"] = (
            paired.prefetch_pollution_rate_control
            - paired.prefetch_pollution_rate_candidate
        )
        paired["traffic_reduction"] = 1.0 - (
            paired.prefetch_issued_candidate
            / paired.prefetch_issued_control.replace(0, np.nan)
        )
        for group_keys, group in paired.groupby(
            ["profile", "workload", "distance"], sort=False
        ):
            benefit, benefit_std, benefit_ci = _mean_ci(group.cycle_benefit)
            pollution, _, pollution_ci = _mean_ci(group.pollution_reduction)
            traffic, _, traffic_ci = _mean_ci(group.traffic_reduction.fillna(0))
            wins = int((group.cycles_candidate < group.cycles_control).sum())
            losses = int((group.cycles_candidate > group.cycles_control).sum())
            ties = int((group.cycles_candidate == group.cycles_control).sum())
            rows.append({
                "candidate": candidate,
                "control": control,
                "profile": group_keys[0],
                "workload": group_keys[1],
                "distance": group_keys[2],
                "pairs": len(group),
                "cycle_benefit_mean": benefit,
                "cycle_benefit_std": benefit_std,
                "cycle_benefit_ci95": benefit_ci,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "p_value_sign": _exact_sign_p(wins, losses),
                "pollution_reduction_mean": pollution,
                "pollution_reduction_ci95": pollution_ci,
                "traffic_reduction_mean": traffic,
                "traffic_reduction_ci95": traffic_ci,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = _holm_adjust(result.p_value_sign.to_list())
        result["significant_0.05_holm"] = result.p_value_holm < 0.05
    return result

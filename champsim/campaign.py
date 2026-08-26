#!/usr/bin/env python3
"""Run and summarize a reproducible matched-control ChampSim campaign.

The trace, configuration, binary, and module hashes in ``manifest.json`` are
the authoritative experiment identity.  A completed run is reused only when
that identity matches its per-run metadata, so interrupted campaigns can be
resumed without silently mixing experiments.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
CONFIGS = {
    "no_l1d_prefetch": HERE / "config" / "no_l1d_prefetch.json",
    "raw_stride": HERE / "config" / "raw_stride_control.json",
    "regularity_stride": HERE / "config" / "regularity_stride.json",
}
COMPARISONS = (
    ("raw_stride", "no_l1d_prefetch"),
    ("regularity_stride", "no_l1d_prefetch"),
    ("regularity_stride", "raw_stride"),
)
GATE_STATS = re.compile(
    r"NNCPU_REGULARITY roi_decisions=(?P<decisions>\d+) "
    r"roi_suppressed=(?P<suppressed>\d+) roi_issued=(?P<issued>\d+) "
    r"gate_enabled=(?P<enabled>true|false) window=(?P<window>\d+) "
    r"support_percent=(?P<support>\d+) degree=(?P<degree>\d+)"
)


@dataclass(frozen=True)
class RunRecord:
    trace: str
    config: str
    instructions: int
    cycles: int
    ipc: float
    l1d_load_misses: int
    prefetch_requested: int
    prefetch_issued: int
    useful_prefetch: int
    useless_prefetch: int
    gate_decisions: int | None
    gate_suppressed: int | None
    module_issued: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _slug(trace: Path) -> str:
    name = trace.name
    for suffix in (".champsimtrace.xz", ".champsimtrace.gz", ".trace.xz", ".trace.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.replace(".", "_")


def _sum_counter(cache: dict, access: str, counter: str) -> int:
    value = cache.get(access, {}).get(counter, [])
    return int(sum(value)) if isinstance(value, list) else int(value or 0)


def _parse_run(json_path: Path, log_path: Path, trace: Path, config: str) -> RunRecord:
    phases = json.loads(json_path.read_text(encoding="utf-8"))
    phase = next((item for item in phases if item.get("name") == "Simulation"), None)
    if phase is None:
        raise ValueError(f"missing Simulation phase in {json_path}")
    roi = phase["roi"]
    core = roi["cores"][0]
    l1d_name = next((key for key in roi if key.endswith("_L1D")), None)
    if l1d_name is None:
        raise ValueError(f"missing L1D statistics in {json_path}")
    l1d = roi[l1d_name]
    instructions = int(core["instructions"])
    cycles = int(core["cycles"])
    match = GATE_STATS.search(log_path.read_text(encoding="utf-8"))
    return RunRecord(
        trace=trace.name,
        config=config,
        instructions=instructions,
        cycles=cycles,
        ipc=instructions / cycles,
        l1d_load_misses=_sum_counter(l1d, "LOAD", "miss"),
        prefetch_requested=int(l1d["prefetch requested"]),
        prefetch_issued=int(l1d["prefetch issued"]),
        useful_prefetch=int(l1d["useful prefetch"]),
        useless_prefetch=int(l1d["useless prefetch"]),
        gate_decisions=int(match.group("decisions")) if match else None,
        gate_suppressed=int(match.group("suppressed")) if match else None,
        module_issued=int(match.group("issued")) if match else None,
    )


def _run_one(
    champsim: Path,
    output: Path,
    trace: Path,
    config_name: str,
    warmup: int,
    simulation: int,
    identity: dict,
) -> RunRecord:
    run_dir = output / "raw" / _slug(trace)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"{config_name}.json"
    log_path = run_dir / f"{config_name}.log"
    metadata_path = run_dir / f"{config_name}.metadata.json"
    fingerprint = {
        "identity": identity,
        "trace": trace.name,
        "trace_sha256": identity["traces"][trace.name]["sha256"],
        "config": config_name,
        "config_sha256": identity["configs"][config_name],
        "warmup_instructions": warmup,
        "simulation_instructions": simulation,
    }
    if json_path.is_file() and log_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return _parse_run(json_path, log_path, trace, config_name)

    temporary_json = json_path.with_suffix(".json.partial")
    command = [
        str(champsim / "bin" / "champsim"),
        "--config",
        str(CONFIGS[config_name]),
        "--warmup-instructions",
        str(warmup),
        "--simulation-instructions",
        str(simulation),
        "--json",
        str(temporary_json),
        str(trace),
    ]
    result = subprocess.run(
        command,
        cwd=champsim,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        temporary_json.unlink(missing_ok=True)
        raise RuntimeError(
            f"ChampSim failed for {trace.name}/{config_name} "
            f"with status {result.returncode}; see {log_path}"
        )
    temporary_json.replace(json_path)
    metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "command": command,
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _parse_run(json_path, log_path, trace, config_name)


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _exact_sign_p(values: list[float], null: float = 1.0) -> float:
    non_ties = [value for value in values if not math.isclose(value, null, rel_tol=1e-12)]
    if not non_ties:
        return 1.0
    positives = sum(value > null for value in non_ties)
    tail = min(positives, len(non_ties) - positives)
    probability = sum(math.comb(len(non_ties), i) for i in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** len(non_ties)))


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive observations")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _bootstrap_geomean_ci(values: list[float], samples: int = 10000) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(20260826)
    logs = [math.log(value) for value in values]
    boot = sorted(
        math.exp(statistics.fmean(rng.choice(logs) for _ in logs))
        for _ in range(samples)
    )
    return boot[int(0.025 * samples)], boot[int(0.975 * samples)]


def _contrasts(records: list[RunRecord]) -> tuple[list[dict], dict]:
    indexed = {(record.trace, record.config): record for record in records}
    traces = sorted({record.trace for record in records})
    rows: list[dict] = []
    aggregate: dict[str, dict] = {}
    for candidate_name, control_name in COMPARISONS:
        comparison_rows = []
        for trace in traces:
            candidate = indexed[(trace, candidate_name)]
            control = indexed[(trace, control_name)]
            speedup = candidate.ipc / control.ipc
            request_reduction = (
                1.0 - candidate.prefetch_requested / control.prefetch_requested
                if control.prefetch_requested
                else None
            )
            issue_reduction = (
                1.0 - candidate.prefetch_issued / control.prefetch_issued
                if control.prefetch_issued
                else None
            )
            row = {
                "trace": trace,
                "candidate": candidate_name,
                "control": control_name,
                "candidate_ipc": candidate.ipc,
                "control_ipc": control.ipc,
                "ipc_speedup": speedup,
                "candidate_cycles": candidate.cycles,
                "control_cycles": control.cycles,
                "candidate_prefetch_requested": candidate.prefetch_requested,
                "control_prefetch_requested": control.prefetch_requested,
                "prefetch_request_reduction": request_reduction,
                "candidate_prefetch_issued": candidate.prefetch_issued,
                "control_prefetch_issued": control.prefetch_issued,
                "prefetch_issue_reduction": issue_reduction,
                "useful_prefetch_delta": candidate.useful_prefetch - control.useful_prefetch,
            }
            rows.append(row)
            comparison_rows.append(row)
        speedups = [row["ipc_speedup"] for row in comparison_rows]
        lower, upper = _bootstrap_geomean_ci(speedups)
        reductions = [
            row["prefetch_issue_reduction"]
            for row in comparison_rows
            if row["prefetch_issue_reduction"] is not None
        ]
        request_reductions = [
            row["prefetch_request_reduction"]
            for row in comparison_rows
            if row["prefetch_request_reduction"] is not None
        ]
        aggregate[f"{candidate_name}_vs_{control_name}"] = {
            "independent_trace_units": len(speedups),
            "geometric_mean_ipc_speedup": _geomean(speedups),
            "bootstrap_95pct_ci": [lower, upper],
            "exact_two_sided_sign_p": _exact_sign_p(speedups),
            "wins": sum(value > 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in speedups),
            "losses": sum(value < 1.0 and not math.isclose(value, 1.0, rel_tol=1e-12) for value in speedups),
            "ties": sum(math.isclose(value, 1.0, rel_tol=1e-12) for value in speedups),
            "median_prefetch_issue_reduction": statistics.median(reductions) if reductions else None,
            "median_prefetch_request_reduction": statistics.median(request_reductions) if request_reductions else None,
        }
    return rows, aggregate


def _validate_records(records: list[RunRecord], simulation: int) -> None:
    indexed: dict[str, dict[str, RunRecord]] = {}
    for record in records:
        indexed.setdefault(record.trace, {})[record.config] = record
    expected_configs = set(CONFIGS)
    for trace, configs in indexed.items():
        if set(configs) != expected_configs:
            raise ValueError(f"incomplete configuration matrix for {trace}: {sorted(configs)}")
        instruction_counts = [record.instructions for record in configs.values()]
        if min(instruction_counts) < simulation or max(instruction_counts) - min(instruction_counts) > 100:
            raise ValueError(f"inconsistent ROI instruction counts for {trace}: {instruction_counts}")
        baseline = configs["no_l1d_prefetch"]
        raw = configs["raw_stride"]
        gated = configs["regularity_stride"]
        if baseline.prefetch_requested or baseline.prefetch_issued:
            raise ValueError(f"no-prefetch baseline emitted a prefetch for {trace}")
        if raw.gate_suppressed != 0:
            raise ValueError(f"raw matched control suppressed a prediction for {trace}")
        for record in (raw, gated):
            if record.module_issued != record.prefetch_issued:
                raise ValueError(
                    f"module/ChampSim issued counters disagree for {trace}/{record.config}"
                )
            if record.gate_decisions is None or record.gate_suppressed is None:
                raise ValueError(f"missing module counters for {trace}/{record.config}")
        decision_scale = max(raw.gate_decisions, gated.gate_decisions, 1)
        if abs(raw.gate_decisions - gated.gate_decisions) / decision_scale > 0.001:
            raise ValueError(f"raw/gated access streams diverged for {trace}")


def _resolve_traces(trace_dir: Path, names: list[str]) -> list[Path]:
    if not names:
        names = [
            line.split("#", 1)[0].strip()
            for line in (HERE / "dpc3_trace_set.txt").read_text(encoding="utf-8").splitlines()
        ]
        names = [name for name in names if name]
    traces = [Path(name) if Path(name).is_absolute() else trace_dir / name for name in names]
    missing = [str(trace) for trace in traces if not trace.is_file()]
    if missing:
        raise FileNotFoundError("missing trace files:\n  " + "\n  ".join(missing))
    if len({trace.name for trace in traces}) != len(traces):
        raise ValueError("trace basenames must be unique")
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", help="trace filenames; defaults to dpc3_trace_set.txt")
    parser.add_argument("--champsim", type=Path, required=True, help="official ChampSim checkout")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50_000_000)
    parser.add_argument("--simulation", type=int, default=200_000_000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.simulation <= 0 or args.jobs <= 0:
        parser.error("warmup must be non-negative; simulation and jobs must be positive")
    binary = args.champsim / "bin" / "champsim"
    if not binary.is_file():
        parser.error(f"ChampSim binary not found: {binary}")
    traces = _resolve_traces(args.trace_dir, args.traces)
    champsim_tree_dirty = bool(_git(args.champsim, "status", "--porcelain"))
    nncpu_tree_dirty = bool(_git(REPOSITORY, "status", "--porcelain"))
    args.output.mkdir(parents=True, exist_ok=True)

    trace_identity = {
        trace.name: {"bytes": trace.stat().st_size, "sha256": _sha256(trace)}
        for trace in traces
    }
    module_files = sorted((HERE / "prefetcher" / "nncpu_regularity_stride").glob("*"))
    identity = {
        "champsim_revision": _git(args.champsim, "rev-parse", "HEAD"),
        "champsim_binary_sha256": _sha256(binary),
        "nncpu_revision": _git(REPOSITORY, "rev-parse", "HEAD"),
        "configs": {name: _sha256(path) for name, path in CONFIGS.items()},
        "module": {str(path.relative_to(REPOSITORY)): _sha256(path) for path in module_files if path.is_file()},
        "campaign_sha256": _sha256(Path(__file__)),
        "traces": trace_identity,
    }
    work = [
        (trace, config_name)
        for trace in traces
        for config_name in CONFIGS
    ]
    records: list[RunRecord] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _run_one,
                args.champsim,
                args.output,
                trace,
                config_name,
                args.warmup,
                args.simulation,
                identity,
            ): (trace, config_name)
            for trace, config_name in work
        }
        for future in as_completed(futures):
            trace, config_name = futures[future]
            records.append(future.result())
            print(f"complete {trace.name} {config_name}", flush=True)

    records.sort(key=lambda record: (record.trace, record.config))
    if len(records) != len(work):
        raise RuntimeError(f"expected {len(work)} records, got {len(records)}")
    _validate_records(records, args.simulation)
    _write_csv(args.output / "runs.csv", (asdict(record) for record in records))
    contrast_rows, summary = _contrasts(records)
    _write_csv(args.output / "contrasts.csv", contrast_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": identity,
        "champsim_tree_dirty": champsim_tree_dirty,
        "nncpu_tree_dirty": nncpu_tree_dirty,
        "warmup_instructions": args.warmup,
        "simulation_instructions": args.simulation,
        "configs": list(CONFIGS),
        "trace_order": [trace.name for trace in traces],
        "selection_rule": "highest-weight official DPC-3 SimPoint per benchmark program",
        "independent_unit": "one benchmark program",
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "records": len(records),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

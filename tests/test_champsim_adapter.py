import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHAMPSIM = ROOT / "champsim"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


campaign = _load("nncpu_champsim_campaign", CHAMPSIM / "campaign.py")
cost = _load("nncpu_champsim_cost", CHAMPSIM / "hardware_cost.py")


def test_raw_and_regularized_configs_are_matched_except_gate_and_name():
    raw = json.loads((CHAMPSIM / "config" / "raw_stride_control.json").read_text())
    gated = json.loads((CHAMPSIM / "config" / "regularity_stride.json").read_text())
    raw.pop("executable_name")
    gated.pop("executable_name")
    raw_gate = raw["L1D"]["prefetcher"].pop("gate_enabled")
    gated_gate = gated["L1D"]["prefetcher"].pop("gate_enabled")

    assert raw_gate is False
    assert gated_gate is True
    assert raw == gated


def test_official_stride_config_matches_raw_predictor_geometry():
    official = json.loads((CHAMPSIM / "config" / "official_ip_stride.json").read_text())
    raw = json.loads((CHAMPSIM / "config" / "raw_stride_control.json").read_text())
    official_prefetcher = official["L1D"]["prefetcher"]
    raw_prefetcher = raw["L1D"]["prefetcher"]

    assert official["L1D"]["prefetch_activate"] == raw["L1D"]["prefetch_activate"]
    assert official_prefetcher["degree"] == raw_prefetcher["degree"]
    assert official_prefetcher["tracker_sets"] == raw_prefetcher.get("tracker_sets", 256)
    assert official_prefetcher["tracker_ways"] == raw_prefetcher.get("tracker_ways", 4)


def test_fixed_trace_set_has_one_simpoint_per_distinct_program():
    traces = [
        line.split("#", 1)[0].strip()
        for line in (CHAMPSIM / "dpc3_trace_set.txt").read_text().splitlines()
    ]
    traces = [trace for trace in traces if trace]
    programs = [trace.split("-", 1)[0] for trace in traces]

    assert len(traces) == 11
    assert len(programs) == len(set(programs))
    assert all(trace.endswith(".champsimtrace.xz") for trace in traces)


def test_fixed_trace_set_uses_documented_highest_weight_simpoints():
    expected = {
        "600.perlbench_s": 210,
        "603.bwaves_s": 3699,
        "605.mcf_s": 665,
        "607.cactuBSSN_s": 2421,
        "623.xalancbmk_s": 700,
        "638.imagick_s": 10316,
        "644.nab_s": 5853,
        "648.exchange2_s": 1699,
        "649.fotonik3d_s": 1176,
        "654.roms_s": 842,
        "657.xz_s": 3167,
    }
    traces = [
        line.split("#", 1)[0].strip()
        for line in (CHAMPSIM / "dpc3_trace_set.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    actual = {
        trace.split("-", 1)[0]: int(trace.rsplit("-", 1)[1].split("B", 1)[0])
        for trace in traces
    }

    assert actual == expected


def test_holdout_is_disjoint_and_covers_remaining_spec17_programs():
    development = set(campaign._read_trace_list(CHAMPSIM / "dpc3_trace_set.txt"))
    holdout = set(campaign._read_trace_list(CHAMPSIM / "dpc3_holdout_trace_set.txt"))
    development_programs = {trace.split("-", 1)[0] for trace in development}
    holdout_programs = {trace.split("-", 1)[0] for trace in holdout}

    assert len(development) == 11
    assert len(holdout) == 9
    assert development_programs.isdisjoint(holdout_programs)
    assert len(development_programs | holdout_programs) == 20


def test_fetcher_parses_commented_trace_names_without_trailing_spaces():
    script = (CHAMPSIM / "fetch_dpc3.sh").read_text()
    assert "awk 'NF {$1=$1; print}'" in script
    result = subprocess.run(
        [
            "bash",
            "-c",
            "sed 's/#.*//' champsim/dpc3_trace_set.txt | awk 'NF {$1=$1; print}'",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    names = result.stdout.splitlines()
    assert len(names) == 11
    assert all(name == name.strip() and " " not in name for name in names)


def test_exact_gate_storage_account_is_explicit_and_bounded():
    estimate = cost.storage_cost(window=16, delta_bits=64)

    assert estimate["fifo_bits"] == 1024
    assert estimate["histogram_bits"] == 1120
    assert estimate["control_bits"] == 9
    assert estimate["total_bits"] == 2153
    assert estimate["fraction_of_dpc3_64kib_budget"] < 0.005
    assert "not synthesized" in estimate["caveat"]


def test_campaign_parser_uses_roi_and_custom_gate_counters(tmp_path):
    stats = [
        {
            "name": "Simulation",
            "roi": {
                "cores": [{"instructions": 200, "cycles": 100}],
                "DRAM": [{
                    "RQ ROW_BUFFER_HIT": 3,
                    "RQ ROW_BUFFER_MISS": 4,
                    "WQ ROW_BUFFER_HIT": 1,
                    "WQ ROW_BUFFER_MISS": 2,
                }],
                "cpu0_L1D": {
                    "LOAD": {"miss": [7]},
                    "prefetch requested": 11,
                    "prefetch issued": 10,
                    "useful prefetch": 6,
                    "useless prefetch": 2,
                },
                "cpu0_L2C": {"PREFETCH": {"miss": [5]}},
                "LLC": {"PREFETCH": {"miss": [4]}},
            },
        }
    ]
    stats_path = tmp_path / "run.json"
    log_path = tmp_path / "run.log"
    trace_path = tmp_path / "example.champsimtrace.xz"
    stats_path.write_text(json.dumps(stats))
    log_path.write_text(
        "NNCPU_REGULARITY roi_decisions=80 roi_suppressed=9 roi_issued=10 "
        "gate_enabled=true window=16 support_percent=35 degree=3\n"
    )

    record = campaign._parse_run(
        stats_path, log_path, trace_path, "regularity_stride"
    )

    assert record.ipc == 2.0
    assert record.l1d_load_misses == 7
    assert record.prefetch_requested == 11
    assert record.l2c_prefetch_misses == 5
    assert record.llc_prefetch_misses == 4
    assert record.dram_read_requests == 7
    assert record.dram_write_requests == 3
    assert record.gate_decisions == 80
    assert record.gate_suppressed == 9
    assert record.module_issued == 10


def test_runner_passes_absolute_output_and_trace_paths(tmp_path, monkeypatch):
    checkout = tmp_path / "ChampSim"
    binary = checkout / "bin" / "champsim"
    binary.parent.mkdir(parents=True)
    binary.write_text("fake")
    trace = tmp_path / "trace.champsimtrace.xz"
    trace.write_bytes(b"trace")
    output = tmp_path / "relative-output"
    relative_output = Path(os.path.relpath(output, ROOT))
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        json_path = Path(command[command.index("--json") + 1])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps([{
            "name": "Simulation",
            "roi": {
                "cores": [{"instructions": 100, "cycles": 100}],
                "cpu0_L1D": {
                    "LOAD": {"miss": [0]}, "prefetch requested": 0,
                    "prefetch issued": 0, "useful prefetch": 0,
                    "useless prefetch": 0,
                },
            },
        }]))
        return subprocess.CompletedProcess(command, 0, stdout=(
            "NNCPU_REGULARITY roi_decisions=10 roi_suppressed=0 roi_issued=0 "
            "gate_enabled=false window=16 support_percent=35 degree=3\n"
        ))

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    identity = {
        "traces": {trace.name: {"sha256": "trace-hash"}},
        "configs": {"raw_stride": "config-hash"},
    }
    campaign._run_one(
        checkout, relative_output, trace, "raw_stride", 0, 100, identity
    )

    json_argument = Path(observed["command"][observed["command"].index("--json") + 1])
    assert json_argument.is_absolute()
    assert Path(observed["command"][-1]).is_absolute()
    assert Path(observed["cwd"]).is_absolute()


def test_campaign_statistics_treat_traces_as_units():
    assert campaign._exact_sign_p([1.1] * 11) == pytest.approx(2 / 2**11)
    assert campaign._exact_sign_p([1.1, 0.9]) == 1.0
    assert campaign._geomean([2.0, 0.5]) == pytest.approx(1.0)


def test_campaign_rejects_requested_as_accepted_traffic():
    def record(config, requested, issued, useful, decisions, suppressed, dram):
        return campaign.RunRecord(
            trace="t", config=config, instructions=100, cycles=100, ipc=1.0,
            l1d_load_misses=1, prefetch_requested=requested,
            prefetch_issued=issued, useful_prefetch=useful, useless_prefetch=0,
            l2c_prefetch_misses=0, llc_prefetch_misses=0,
            dram_read_requests=dram, dram_write_requests=0,
            gate_decisions=decisions, gate_suppressed=suppressed,
            module_issued=issued if decisions is not None else None,
        )

    baseline = record("no_l1d_prefetch", 0, 0, 0, None, None, 20)
    official = record("official_ip_stride", 40, 10, 2, None, None, 30)
    raw = record("raw_stride", 40, 10, 2, 30, 0, 30)
    gated = record("regularity_stride", 10, 5, 2, 30, 3, 24)

    campaign._validate_records([baseline, official, raw, gated], simulation=100)
    rows, summary = campaign._contrasts([baseline, official, raw, gated])
    matched = next(row for row in rows if row["control"] == "raw_stride")

    assert matched["prefetch_request_reduction"] == pytest.approx(0.75)
    assert matched["prefetch_issue_reduction"] == pytest.approx(0.5)
    assert matched["dram_read_request_reduction"] == pytest.approx(0.2)
    assert summary["regularity_stride_vs_raw_stride"]["median_prefetch_issue_reduction"] == pytest.approx(0.5)
    assert summary["regularity_stride_vs_raw_stride"]["aggregate_dram_read_request_reduction"] == pytest.approx(0.2)
    assert summary["regularity_stride_vs_raw_stride"]["useful_prefetch_retention"] == 1.0

import importlib.util
import json
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
                "cpu0_L1D": {
                    "LOAD": {"miss": [7]},
                    "prefetch requested": 11,
                    "prefetch issued": 10,
                    "useful prefetch": 6,
                    "useless prefetch": 2,
                },
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
    assert record.gate_decisions == 80
    assert record.gate_suppressed == 9
    assert record.module_issued == 10


def test_campaign_statistics_treat_traces_as_units():
    assert campaign._exact_sign_p([1.1] * 11) == pytest.approx(2 / 2**11)
    assert campaign._exact_sign_p([1.1, 0.9]) == 1.0
    assert campaign._geomean([2.0, 0.5]) == pytest.approx(1.0)


def test_campaign_rejects_requested_as_accepted_traffic():
    baseline = campaign.RunRecord("t", "no_l1d_prefetch", 100, 100, 1.0, 1, 0, 0, 0, 0, None, None, None)
    raw = campaign.RunRecord("t", "raw_stride", 100, 100, 1.0, 1, 40, 10, 2, 1, 30, 0, 10)
    gated = campaign.RunRecord("t", "regularity_stride", 100, 100, 1.0, 1, 10, 5, 2, 0, 30, 3, 5)

    campaign._validate_records([baseline, raw, gated], simulation=100)
    rows, summary = campaign._contrasts([baseline, raw, gated])
    matched = next(row for row in rows if row["control"] == "raw_stride")

    assert matched["prefetch_request_reduction"] == pytest.approx(0.75)
    assert matched["prefetch_issue_reduction"] == pytest.approx(0.5)
    assert summary["regularity_stride_vs_raw_stride"]["median_prefetch_issue_reduction"] == pytest.approx(0.5)

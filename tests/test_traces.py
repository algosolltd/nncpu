"""Tests for the trace ingestion module (nncpu/traces.py)."""

import os

import pytest

from nncpu.traces import (
    TRACE_PATTERNS,
    build_trace_workloads,
    kv_cache_append,
    read_trace,
    run_trace_battery,
    trace_recorder,
    write_trace,
)


def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "t.trace"
    insts = [
        {"type": "LOAD", "address": 0x1000},
        {"type": "STORE", "address": 0x1A08, "value": 42},
        {"type": "ADD", "operands": [3, 4]},
    ]
    write_trace(str(path), insts)
    parsed = read_trace(str(path))
    assert parsed == insts


def test_reader_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "c.trace"
    path.write_text("# header\n\nLOAD 0x10   # load\nSTORE 0x20 9\n", encoding="utf-8")
    parsed = read_trace(str(path))
    assert parsed == [{"type": "LOAD", "address": 0x10},
                      {"type": "STORE", "address": 0x20, "value": 9}]


def test_reader_limit(tmp_path):
    path = tmp_path / "l.trace"
    write_trace(str(path), [{"type": "LOAD", "address": i} for i in range(100)])
    assert len(read_trace(str(path), limit=7)) == 7


def test_recorder_captures_real_accesses(tmp_path):
    path = tmp_path / "rec.trace"
    with trace_recorder(str(path)) as rec:
        for j in range(32):
            rec.store(0x1000 + (j % 16), j)
            rec.load(0x1000 + (j % 16))
    parsed = read_trace(str(path))
    assert len(parsed) == 64
    for inst in parsed:
        assert inst["type"] in ("LOAD", "STORE")


def test_patterns_are_deterministic_and_valid():
    for name, gen in TRACE_PATTERNS.items():
        a = list(gen(200))
        b = list(gen(200))
        assert a == b, f"{name} not deterministic"
        for inst in a:
            assert inst["type"] in ("LOAD", "STORE")
            assert isinstance(inst["address"], int)


def test_kv_pattern_has_stores_and_loads():
    insts = list(kv_cache_append(300))
    types = {i["type"] for i in insts}
    assert types == {"LOAD", "STORE"}


def test_trace_battery_writes_experiment(tmp_path):
    workloads = build_trace_workloads(60)
    res = run_trace_battery(root=str(tmp_path), name="t", seeds=2,
                            length=60, workloads=workloads)
    outdir = res["outdir"]
    for f in ("summary.csv", "runs.csv", "config.json"):
        assert os.path.exists(os.path.join(outdir, f)), f
    s = res["summary"]
    assert "speedup_mean" in s.columns
    # stride must beat baseline on the sequential token stream
    row = s[(s.workload == "token_stream") & (s.config == "stride")]
    assert float(row["speedup_mean"].iloc[0]) > 1.0


def test_demo_self_test_runs():
    demo = os.path.join(os.path.dirname(__file__), "..", "demo_trace.txt")
    if os.path.exists(demo):
        inst = read_trace(demo)[:4]
        assert len(inst) <= 4
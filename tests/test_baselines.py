"""Tests for the modeled published prefetchers (baselines.py)."""

import pytest

from nncpu.baselines import (
    BertiPrefetcher,
    NextLinePrefetcher,
    SIM_CONFIGS,
    make_sim_prefetcher,
)
from nncpu.cpu import LINE_SIZE
from nncpu.cpu import MachineConfig
from nncpu.benchmark import run_workload
from nncpu.prefetchers import Prefetcher


def test_nextline_predicts_next_line():
    p = NextLinePrefetcher()
    for addr in (0x1000, 0x1008, 0x1010):
        assert p.predict_next(addr, 0, "LOAD") == addr + LINE_SIZE


def test_nextline_uses_the_selected_machine_geometry():
    machine = MachineConfig(line_size=16)
    stream = [{"type": "LOAD", "address": 0x1000 + i * 16} for i in range(12)]
    base = run_workload(stream, "baseline", machine=machine)
    nxt = run_workload(stream, "nextline", machine=machine)
    assert nxt.prefetch_issued > 0
    assert nxt.cycles < base.cycles


def test_make_sim_prefetcher_resolves_all():
    assert make_sim_prefetcher("baseline") is None
    assert isinstance(make_sim_prefetcher("nextline"), NextLinePrefetcher)
    assert isinstance(make_sim_prefetcher("berti"), BertiPrefetcher)
    assert isinstance(make_sim_prefetcher("nn"), Prefetcher)
    assert isinstance(make_sim_prefetcher("stride"), Prefetcher)


def test_berti_learns_constant_stride():
    """After warm-up Berti behaves like stride on a constant-delta stream."""
    p = BertiPrefetcher()
    addrs = [0x1000 + i * 8 for i in range(40)]
    preds = []
    for addr in addrs:
        preds.append(p.predict_next(addr, 0, "LOAD"))
    # Once it has seen delta=8 enough, it predicts addr + 8.
    assert preds[-1] == addrs[-1] + 8


def test_berti_handles_random_without_crashing():
    p = BertiPrefetcher(capacity=64)
    state = 0x12345
    for _ in range(200):
        state = (1103515245 * state + 12345) % (2 ** 31)
        p.predict_next(0x1000 + (state % 4096), 0, "LOAD")
    assert len(p._table) <= 64


def test_berti_table_bounded():
    p = BertiPrefetcher(capacity=8)
    state = 0x1
    for _ in range(500):
        state = (1103515245 * state + 12345) % (2 ** 31)
        p.predict_next(0x1000 + (state % 8192), 0, "LOAD")
    assert len(p._table) <= 8


def test_sim_configs_includes_core():
    assert set(SIM_CONFIGS) == {"baseline", "stride", "nn", "nextline", "berti"}

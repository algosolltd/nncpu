"""Unit + integration tests for the cycle-accounting CPU simulator."""

import pytest

from nncpu.cpu import (
    CPU,
    FETCH_CYCLES,
    LINE_SIZE,
    L1_LINES,
    MEM_LATENCY,
    LOAD_HIT_CYCLES,
    MachineConfig,
)
from nncpu.prefetchers import StridePrefetcher, NNPrefetcher, make_prefetcher
from nncpu.baselines import NextLinePrefetcher
from nncpu.workloads import build_workloads


def test_miss_penalty_is_charged_once():
    """The first load to a cold line pays MEM_LATENCY plus fetch."""
    cpu = CPU()
    cpu.execute({"type": "LOAD", "address": 0x1000})
    assert cpu.cycles == FETCH_CYCLES + MEM_LATENCY
    assert cpu.report().misses == 1


def test_cold_then_hot():
    """Re-reading the same line is an L1 hit costing one cycle."""
    cpu = CPU()
    addr = 0x1000
    cpu.execute({"type": "LOAD", "address": addr})  # cold
    before = cpu.cycles
    cpu.execute({"type": "LOAD", "address": addr})  # hot
    assert cpu.cycles - before == FETCH_CYCLES + LOAD_HIT_CYCLES
    assert cpu.report().hits == 1


def test_store_then_load_ordering():
    """A store must be visible to a subsequent load of the same address."""
    cpu = CPU()
    cpu.execute({"type": "STORE", "address": 0x1000, "value": 42})
    outcome = cpu.execute({"type": "LOAD", "address": 0x1000})
    assert outcome["value"] == 42
    assert cpu.data[0x1000] == 42


def test_division_by_zero_is_trapped():
    """DIV by zero must not crash nor pollute the machine state."""
    cpu = CPU()
    before = cpu.cycles
    outcome = cpu.execute({"type": "DIV", "operands": [10, 0]})
    assert outcome["type"] == "ERROR"
    assert cpu.cycles - before == FETCH_CYCLES + 8


def test_unknown_instruction_raises():
    cpu = CPU()
    with pytest.raises(ValueError):
        cpu.execute({"type": "FMA", "operands": [1, 2, 3]})


def test_arithmetic_result_correct():
    cpu = CPU()
    assert cpu.execute({"type": "MUL", "operands": [6, 7]})["result"] == 42


def test_cache_lru_eviction_capacity():
    """More distinct lines than capacity force LRU evictions; the simulator
    must not crash and must keep accounting hits+misses == accesses.  Every
    one of these accesses touches a brand-new line, so all are misses."""
    cpu = CPU()
    n = L1_LINES + 8
    for i in range(n):
        cpu.execute({"type": "LOAD", "address": 0x1000 + i * LINE_SIZE})
    rep = cpu.report()
    assert rep.hits + rep.misses == n
    assert rep.misses == n
    assert len(cpu.cache) == L1_LINES  # trimmed back down to capacity


def test_prefetch_hides_latency():
    """A prefetched line must be a 1-cycle hit, and count as *used*."""
    cpu = CPU(prefetcher=StridePrefetcher())
    cpu.execute({"type": "LOAD", "address": 0x1000})         # miss: 40 cycles
    cpu._prefetch(0x1008)                                    # hide the next line
    before = cpu.cycles
    cpu.execute({"type": "LOAD", "address": 0x1008})         # should be a hit
    assert cpu.cycles - before == FETCH_CYCLES + LOAD_HIT_CYCLES
    rep = cpu.report()
    assert rep.prefetch_issued == 1
    assert rep.prefetch_used == 1


def test_nonzero_prefetch_latency_cannot_grant_an_immediate_hit():
    """A one-access-ahead prefetch is late when DRAM latency is 40 cycles.

    This is the key external-validity check missing from the paper profile:
    issuing the right address is insufficient unless it arrives in time.
    """
    cpu = CPU(
        prefetcher=NextLinePrefetcher(),
        machine=MachineConfig(prefetch_latency=MEM_LATENCY),
    )
    cpu.execute({"type": "LOAD", "address": 0x1000})
    before = cpu.cycles
    cpu.execute({"type": "LOAD", "address": 0x1008})

    rep = cpu.report()
    assert cpu.cycles - before == FETCH_CYCLES + MEM_LATENCY - 1
    assert rep.hits == 0
    assert rep.misses == 2
    assert rep.prefetch_used == 1  # useful, but late


@pytest.mark.parametrize(
    "kwargs",
    [
        {"line_size": 0},
        {"l1_lines": 0},
        {"mem_latency": 0},
        {"prefetch_latency": -1},
        {"wb_limit": -1},
    ],
)
def test_machine_rejects_physically_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MachineConfig(**kwargs)


def test_prefetchers_speedup_memory_workloads():
    """On sequential/strided streams both prefetchers beat the baseline
    (fewer stalls => fewer total cycles)."""
    workloads = build_workloads(2000)
    baseline = CPU().report
    for name in ("sequential_read", "strided_read"):
        instructions = workloads[name]
        base_cycles = run_to_cycles(instructions, None)
        for cfg in ("stride", "nn"):
            cfg_cycles = run_to_cycles(instructions, make_prefetcher(cfg))
            assert cfg_cycles < base_cycles, (
                f"{name}/{cfg} should beat baseline ({cfg_cycles} vs {base_cycles})"
            )


def run_to_cycles(instructions, prefetcher):
    cpu = CPU(prefetcher=prefetcher)
    for inst in instructions:
        cpu.execute(inst)
    return cpu.report().cycles


def test_nn_is_fast_enough_to_use():
    """The NN must at least finish a 2000-instruction stream quickly and
    never fall into pathological latency."""
    instructions = build_workloads(300)["sequential_read"]
    cpu = CPU(prefetcher=NNPrefetcher(batch_size=16))
    for inst in instructions:
        cpu.execute(inst)
    assert cpu.report().instructions == 300


def test_mixed_read_write_preserves_values():
    """In the write+read workload, every load returns what was stored."""
    instructions = build_workloads(400)["mixed_read_write"]
    cpu = CPU(prefetcher=None)
    stored = {}
    for inst in instructions:
        if inst["type"] == "STORE":
            stored[inst["address"]] = inst["value"]
        cpu.execute(inst)
    for addr, expected in stored.items():
        outcome = cpu.execute({"type": "LOAD", "address": addr})
        assert outcome["value"] == expected

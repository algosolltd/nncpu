"""Cycle-accounting in-order CPU simulator.

The core executes a simple instruction set (ADD, MUL, DIV, LOAD, STORE)
and advances a clock by the *latency* of each operation:

* every instruction costs ``FETCH_CYCLES``,
* an L1 hit costs 1 cycle, an L1 miss fetches a whole line from memory
  and stalls the core for ``MEM_LATENCY`` cycles,
* arithmetic costs are table-driven (``ARITH_CYCLES``),
* stores are write-through: they update the cache immediately and park a
  write-back in the store buffer; the core only stalls when the buffer
  overflows.

A pluggable ``Prefetcher`` (see :mod:`nncpu.prefetchers`) is consulted
after every memory instruction.  Its predicted address is fetched into
the cache for free (prefetch uses otherwise-idle DRAM bandwidth), which
is how the NN hides memory latency.  All metrics are counted honestly:
hits vs. misses, prefetches issued vs. actually used, and per-category
cycle totals.
"""

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .prefetchers import Prefetcher

# -- machine parameters ----------------------------------------------------
FETCH_CYCLES = 1        # cycles to fetch an instruction
LOAD_HIT_CYCLES = 1     # L1 hit costs
STORE_CYCLES = 1
MEM_LATENCY = 40        # DRAM latency: cycles to pull a line on a miss
L1_LINES = 32           # cache capacity (full-associative, LRU), in lines
LINE_SIZE = 8           # words per cache line
WB_LIMIT = 16           # store-buffer occupancy before the core stalls

ARITH_CYCLES = {
    "ADD": 1,
    "MUL": 3,
    "DIV": 8,
}

_DEFAULT_MEMORY_VALUE = 0

_MEMORY_INSTRUCTIONS = ("LOAD", "STORE")
_ARITHMETIC_INSTRUCTIONS = tuple(ARITH_CYCLES)


@dataclass
class MachineConfig:
    """One fully-specified machine: cache geometry, latencies, buffer size.

    Every field lands in the experiment manifest so each recorded run is
    reproducible down to the machine the instructions ran on.
    """

    fetch_cycles: int = FETCH_CYCLES
    load_hit_cycles: int = LOAD_HIT_CYCLES
    store_cycles: int = STORE_CYCLES
    mem_latency: int = MEM_LATENCY
    # Zero preserves the paper's explicitly idealized "free L1 fill" model.
    # Set this to ``mem_latency`` (or another measured value) to test whether a
    # prefetch is early enough to be useful rather than granting an instant hit.
    prefetch_latency: int = 0
    l1_lines: int = L1_LINES
    line_size: int = LINE_SIZE
    # None means fully associative; publication profiles can select a
    # conventional set-associative organization without changing capacity.
    l1_associativity: Optional[int] = None
    # One memory request can start per interval and this many prefetches may
    # remain outstanding.  Demand misses share the issue schedule.
    memory_issue_interval: int = 1
    prefetch_mshr: int = 16
    wb_limit: int = WB_LIMIT

    def __post_init__(self) -> None:
        positive = {
            "mem_latency": self.mem_latency,
            "l1_lines": self.l1_lines,
            "line_size": self.line_size,
            "memory_issue_interval": self.memory_issue_interval,
        }
        non_negative = {
            "fetch_cycles": self.fetch_cycles,
            "load_hit_cycles": self.load_hit_cycles,
            "store_cycles": self.store_cycles,
            "prefetch_latency": self.prefetch_latency,
            "prefetch_mshr": self.prefetch_mshr,
            "wb_limit": self.wb_limit,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.l1_associativity is not None:
            if self.l1_associativity <= 0:
                raise ValueError("l1_associativity must be > 0")
            if self.l1_associativity > self.l1_lines:
                raise ValueError("l1_associativity cannot exceed l1_lines")
            if self.l1_lines % self.l1_associativity:
                raise ValueError("l1_lines must be divisible by l1_associativity")

    def as_dict(self) -> dict:
        """Plain, JSON-serializable view for manifests."""
        return {
            "fetch_cycles": self.fetch_cycles,
            "load_hit_cycles": self.load_hit_cycles,
            "store_cycles": self.store_cycles,
            "mem_latency": self.mem_latency,
            "prefetch_latency": self.prefetch_latency,
            "l1_lines": self.l1_lines,
            "line_size": self.line_size,
            "l1_associativity": self.l1_associativity,
            "memory_issue_interval": self.memory_issue_interval,
            "prefetch_mshr": self.prefetch_mshr,
            "wb_limit": self.wb_limit,
        }

    def describe(self) -> str:
        return (
            f"{self.l1_lines} L1 lines x {self.line_size} words, "
            f"{self.l1_associativity or self.l1_lines}-way, "
            f"MEM_LATENCY={self.mem_latency} cycles, "
            f"store-buffer limit={self.wb_limit}"
        )


class L1Cache:
    """Set-associative LRU cache keyed by cache-line index."""

    def __init__(
        self,
        capacity: int = L1_LINES,
        line_size: int = LINE_SIZE,
        associativity: Optional[int] = None,
    ):
        self.capacity = capacity
        self.line_size = line_size
        self.associativity = associativity or capacity
        if self.associativity <= 0 or capacity % self.associativity:
            raise ValueError("capacity must be divisible by positive associativity")
        self.num_sets = capacity // self.associativity
        self._sets: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(self.num_sets)
        ]

    def line_of(self, addr: int) -> int:
        return addr // self.line_size

    def contains(self, addr: int) -> bool:
        return self.contains_line(self.line_of(addr))

    def contains_line(self, line: int) -> bool:
        return line in self._sets[line % self.num_sets]

    def touch(self, addr: int) -> None:
        """Refresh LRU recency for a line that is already resident."""
        line = self.line_of(addr)
        self._sets[line % self.num_sets].move_to_end(line)

    def insert(self, addr: int) -> Optional[int]:
        """Allocate ``addr``'s line, evicting the LRU line if full.

        Returns the id of the evicted line, or ``None``.
        """
        line = self.line_of(addr)
        cache_set = self._sets[line % self.num_sets]
        if line in cache_set:
            cache_set.move_to_end(line)
            return None
        evicted: Optional[int] = None
        if len(cache_set) >= self.associativity:
            evicted = next(iter(cache_set))
            del cache_set[evicted]
        cache_set[line] = None
        return evicted

    @property
    def lines(self) -> "OrderedDict[int, None]":
        # Compatibility/debug view; mutation must go through cache methods.
        return OrderedDict((line, None) for cache_set in self._sets for line in cache_set)

    def __len__(self) -> int:
        return sum(len(cache_set) for cache_set in self._sets)


class WriteBuffer:
    """Pending write-backs.  Draining is free; overflow costs stalls."""

    def __init__(self, limit: int = WB_LIMIT):
        self.limit = limit
        self._pending: "deque[int]" = deque()

    def push(self, addr: int) -> None:
        self._pending.append(addr)

    def excess(self) -> int:
        """Number of writes that must be drained immediately."""
        return max(0, len(self._pending) - self.limit)

    def drain(self, n: int) -> None:
        for _ in range(min(n, len(self._pending))):
            self._pending.popleft()

    def __len__(self) -> int:
        return len(self._pending)


@dataclass
class CPUReport:
    """Honest accounting of one simulation run."""

    instructions: int = 0
    cycles: int = 0
    hits: int = 0
    misses: int = 0
    mem_cycles: int = 0
    arith_cycles: int = 0
    prefetch_issued: int = 0
    prefetch_requested: int = 0
    prefetch_redundant: int = 0
    prefetch_used: int = 0
    prefetch_late: int = 0
    prefetch_dropped: int = 0
    prefetch_useful_hits: int = 0
    prefetch_pollution_misses: int = 0
    demand_only_hits: int = 0
    demand_only_misses: int = 0
    wall_seconds: float = 0.0
    inst_cycles: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.hits + self.misses == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)

    @property
    def ipc(self) -> float:
        return self.instructions / self.cycles if self.cycles else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        if self.prefetch_issued == 0:
            return 0.0
        return self.prefetch_used / self.prefetch_issued

    @property
    def prefetch_coverage(self) -> float:
        """Fraction of demand-only misses converted into actual L1 hits."""
        if self.demand_only_misses == 0:
            return 0.0
        return self.prefetch_useful_hits / self.demand_only_misses

    @property
    def prefetch_pollution_rate(self) -> float:
        """Fraction of accesses that miss only because prefetch changed L1."""
        accesses = self.demand_only_hits + self.demand_only_misses
        if accesses == 0:
            return 0.0
        return self.prefetch_pollution_misses / accesses

    @property
    def prefetch_timeliness(self) -> float:
        """Share of useful or late prefetches that arrived before demand."""
        resolved = self.prefetch_useful_hits + self.prefetch_late
        if resolved == 0:
            return 0.0
        return self.prefetch_useful_hits / resolved

    @property
    def cycles_per_instruction(self) -> float:
        return self.cycles / self.instructions if self.instructions else 0.0


class CPU:
    """An in-order scalar core.

    ``prefetcher`` may be ``None`` (every miss pays full DRAM latency) or
    any object exposing ``predict_next(addr, pc, opcode) -> int|None``.
    """

    def __init__(
        self,
        prefetcher: Optional[Prefetcher] = None,
        machine: Optional[MachineConfig] = None,
    ):
        self.machine = machine or MachineConfig()
        self.registers = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
        self.data: dict = {}
        self.pc = 0
        self.cycles = 0
        self.cache = L1Cache(
            capacity=self.machine.l1_lines,
            line_size=self.machine.line_size,
            associativity=self.machine.l1_associativity,
        )
        # Demand-only shadow L1: same geometry, but it never sees prefetches.
        # Comparing it at each demand separates useful hits from pollution
        # misses without requiring a second simulation with potentially
        # divergent timing.
        self._demand_only_cache = L1Cache(
            capacity=self.machine.l1_lines,
            line_size=self.machine.line_size,
            associativity=self.machine.l1_associativity,
        )
        self.write_buffer = WriteBuffer(limit=self.machine.wb_limit)
        self.prefetcher = prefetcher

        self._report = CPUReport()
        self._prefetched_lines: set = set()
        self._pending_prefetches: dict[int, int] = {}
        self._next_memory_issue_cycle = 0

    # -- public API --------------------------------------------------------

    def execute(self, instruction: dict) -> Optional[dict]:
        """Execute one instruction dict, advancing the cycle counter."""
        t0 = time.perf_counter()
        itype = instruction.get("type")
        if itype not in ("LOAD", "STORE") and itype not in _ARITHMETIC_INSTRUCTIONS:
            raise ValueError(f"Unknown instruction type: {itype!r}")

        start_cycles = self.cycles
        self.cycles += self.machine.fetch_cycles
        self._report.instructions += 1
        predictor_pc = instruction.get("pc", self.pc)

        if itype in _ARITHMETIC_INSTRUCTIONS:
            self.cycles += ARITH_CYCLES[itype]
            self._report.arith_cycles += ARITH_CYCLES[itype]
            outcome = self._do_arithmetic(itype, instruction["operands"])
        elif itype == "LOAD":
            outcome = self._load(instruction["address"], predictor_pc)
        else:  # STORE
            outcome = self._store(
                instruction["address"], instruction.get("value", 0), predictor_pc
            )

        self._report.inst_cycles.append(self.cycles - start_cycles)
        self._report.cycles = self.cycles
        self.pc += 1
        self._report.wall_seconds += time.perf_counter() - t0
        return outcome

    # -- implementation ----------------------------------------------------

    def _do_arithmetic(self, itype: str, operands: Sequence[int]) -> dict:
        a, b = operands[0], operands[1] if len(operands) > 1 else 0
        if itype == "ADD":
            return {"type": itype, "result": a + b}
        if itype == "MUL":
            return {"type": itype, "result": a * b}
        if b == 0:
            return {"type": "ERROR", "message": "Division by zero"}
        return {"type": itype, "result": a / b}

    def _load(self, addr: int, predictor_pc: int) -> dict:
        hit, miss_latency = self._touch(addr)
        latency = self.machine.load_hit_cycles if hit else miss_latency
        self.cycles += latency
        self._report.mem_cycles += latency
        value = self.data.get(addr, _DEFAULT_MEMORY_VALUE)
        self._after_mem_access(addr, "LOAD", predictor_pc)
        return {"type": "LOAD", "address": addr, "value": value}

    def _store(self, addr: int, value: int, predictor_pc: int) -> dict:
        hit, miss_latency = self._touch(addr)
        latency = self.machine.store_cycles if hit else miss_latency
        self.cycles += latency
        self._report.mem_cycles += latency
        self.data[addr] = value
        self.write_buffer.push(addr)
        stall = self.write_buffer.excess()
        if stall:
            self.cycles += stall
            self._report.mem_cycles += stall
            self.write_buffer.drain(stall)
        self._after_mem_access(addr, "STORE", predictor_pc)
        return {"type": "STORE", "address": addr, "value": value}

    def _touch(self, addr: int) -> tuple[bool, int]:
        """Access one line and return ``(hit, miss_latency)``.

        A demand that catches an in-flight prefetch is still a cache miss, but
        it waits only for the request's remaining latency.  This makes
        prefetch timeliness observable while retaining the paper's legacy
        behavior when ``prefetch_latency == 0``.
        """
        self._retire_prefetches(self.cycles)
        demand_only_hit = self._touch_demand_only(addr)
        if self.cache.contains(addr):
            self.cache.touch(addr)
            self._report.hits += 1
            if not demand_only_hit:
                self._report.prefetch_useful_hits += 1
            line = self.cache.line_of(addr)
            if line in self._prefetched_lines:
                self._report.prefetch_used += 1
                self._prefetched_lines.discard(line)
            return True, 0
        self._report.misses += 1
        if demand_only_hit:
            self._report.prefetch_pollution_misses += 1

        line = self.cache.line_of(addr)
        ready = self._pending_prefetches.get(line)
        if ready is not None:
            remaining = max(0, ready - self.cycles)
            self._pending_prefetches.pop(line, None)
            self._retire_prefetches(ready)
            # The demand consumes this request at completion; insert it after
            # other requests that completed during the wait so it cannot be
            # spuriously evicted before the waiting demand observes it.
            evicted = self.cache.insert(addr)
            if evicted is not None:
                self._prefetched_lines.discard(evicted)
            self._report.prefetch_used += 1
            if remaining:
                self._report.prefetch_late += 1
            self._prefetched_lines.discard(line)
            return False, remaining

        # A new demand shares the memory issue schedule with prefetches.
        ready = self._schedule_memory_request(self.machine.mem_latency)
        self._retire_prefetches(ready)
        evicted = self.cache.insert(addr)
        if evicted is not None:
            self._prefetched_lines.discard(evicted)
        return False, ready - self.cycles

    def _touch_demand_only(self, addr: int) -> bool:
        """Update the shadow L1 and return whether demand alone would hit."""
        if self._demand_only_cache.contains(addr):
            self._demand_only_cache.touch(addr)
            self._report.demand_only_hits += 1
            return True
        self._demand_only_cache.insert(addr)
        self._report.demand_only_misses += 1
        return False

    def _prefetch(self, addr: int) -> None:
        """Issue a prefetch, immediately or with configured latency."""
        self._report.prefetch_requested += 1
        line = self.cache.line_of(addr)
        if self.cache.contains_line(line):
            self.cache.touch(addr)
            self._report.prefetch_redundant += 1
            return
        if line in self._prefetched_lines or line in self._pending_prefetches:
            self._report.prefetch_redundant += 1
            return
        if self.machine.prefetch_latency:
            if len(self._pending_prefetches) >= self.machine.prefetch_mshr:
                self._report.prefetch_dropped += 1
                return
            self._report.prefetch_issued += 1
            self._pending_prefetches[line] = self._schedule_memory_request(
                self.machine.prefetch_latency
            )
            return
        self._report.prefetch_issued += 1
        self._insert_prefetched_line(line)

    def _schedule_memory_request(self, latency: int) -> int:
        """Reserve issue bandwidth and return the request completion cycle."""
        start = max(self.cycles, self._next_memory_issue_cycle)
        self._next_memory_issue_cycle = start + self.machine.memory_issue_interval
        return start + latency

    def _insert_prefetched_line(self, line: int) -> None:
        """Complete a prefetch and allocate its line in L1."""
        addr = line * self.machine.line_size
        evicted = self.cache.insert(addr)
        if evicted is not None:
            self._prefetched_lines.discard(evicted)
        self._prefetched_lines.add(line)

    def _retire_prefetches(self, up_to_cycle: int) -> None:
        """Complete outstanding requests whose ready cycle has elapsed."""
        ready = sorted(
            (cycle, line)
            for line, cycle in self._pending_prefetches.items()
            if cycle <= up_to_cycle
        )
        for _, line in ready:
            self._pending_prefetches.pop(line, None)
            if not self.cache.contains_line(line):
                self._insert_prefetched_line(line)

    def _after_mem_access(self, addr: int, opcode: str, predictor_pc: int) -> None:
        if self.prefetcher is None:
            return
        predicted = self.prefetcher.predict_next(addr, predictor_pc, opcode)
        if predicted is not None:
            self._prefetch(predicted)

    def report(self) -> CPUReport:
        return self._report

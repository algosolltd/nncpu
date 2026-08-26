"""nncpu: a cycle-accounting CPU simulator with a streaming NN prefetcher.

The simulator models an in-order scalar core with an L1 cache, a
write-through store buffer, and pluggable cache prefetchers.  The
prefetcher learns address streams (a stride heuristic or a learned MLP)
and issues prefetches that hide main-memory latency.
"""

from ._version import __version__

from .cpu import (
    CPU,
    LINE_SIZE,
    L1_LINES,
    MEM_LATENCY,
    ARITH_CYCLES,
    WB_LIMIT,
    MachineConfig,
    CPUReport,
)
from .mlp import DenseMLP
from .prefetchers import Prefetcher, StridePrefetcher, NNPrefetcher, make_prefetcher
from .workloads import WORKLOADS, build_workloads
from .benchmark import run_workload, run_all, summarize
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
    aggregate,
    make_manifest,
)

__all__ = [
    "CPU",
    "LINE_SIZE",
    "L1_LINES",
    "MEM_LATENCY",
    "ARITH_CYCLES",
    "WB_LIMIT",
    "MachineConfig",
    "CPUReport",
    "DenseMLP",
    "Prefetcher",
    "StridePrefetcher",
    "NNPrefetcher",
    "make_prefetcher",
    "WORKLOADS",
    "build_workloads",
    "run_workload",
    "run_all",
    "summarize",
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
    "aggregate",
    "make_manifest",
]
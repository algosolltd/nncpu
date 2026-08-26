"""Modeled implementations of published cache prefetchers.

These are faithful software models of the *mechanism* of each prefetcher,
sharing our Predict-next-address interface, so the paper can compare them
inside our simulator on equal footing:

* :class:`NextLinePrefetcher` -- the classic next-64B/-line prefetcher.
* :class:`BertiPrefetcher`    -- best-offset prefetcher (Berti et al. 2020):
  a table indexed by the observed delta tracks which next-delta followed
  that delta most often, and prefetches the mode (a 1st-order Markov
  delta predictor with an LRU table).
* :class:`StridePrefetcher` (in prefetchers.py) -- PC-stride-style.

None of them has a confidence gate: that is exactly the difference the
paper is about.
"""

from collections import OrderedDict
from typing import Optional

from .cpu import LINE_SIZE
from .prefetchers import Prefetcher, _DEFAULT_DELTA

# extra configs selectable in experiments (CONFIGS keeps the 3 core ones)
SIM_CONFIGS = ("baseline", "stride", "nextline", "berti", "nn")


class NextLinePrefetcher(Prefetcher):
    """Prefetch the very next cache line, always."""

    name = "nextline"

    def __init__(self, line_size: int = LINE_SIZE):
        if line_size <= 0:
            raise ValueError("line_size must be > 0")
        self.line_size = line_size

    def predict_next(self, addr: int, pc: int, opcode: str) -> Optional[int]:
        del pc, opcode
        return addr + self.line_size  # next line down the address space


class BertiPrefetcher(Prefetcher):
    """Best-offset / first-order Markov delta predictor.

    ``table[delta]`` stores how often each *following* delta was seen; the
    prediction for an access with delta ``d`` is ``addr + argmax(table[d])``
    (falls back to the current delta, i.e. stride-like).  Table entries are
    evicted LRU when capacity is reached.  No gate: on unpredictable traffic
    it keeps prefetching (and can pollute), like real best-offset units.
    """

    name = "berti"

    def __init__(self, capacity: int = 128):
        self.capacity = capacity          # number of delta keys remembered
        self._table: "OrderedDict[int, dict]" = OrderedDict()
        self._last_addr: Optional[int] = None
        self._last_delta: Optional[int] = None

    def predict_next(self, addr: int, pc: int, opcode: str) -> Optional[int]:
        del pc, opcode
        if self._last_addr is None:
            self._last_addr = addr
            return addr + _DEFAULT_DELTA

        delta = addr - self._last_addr
        predicted_next = None

        # training: we predicted a follow-up for (last delta) — now we know
        # this access actually followed it, so reward that next-delta.
        if self._last_delta is not None:
            bucket = self._table.setdefault(self._last_delta, {})
            bucket[delta] = bucket.get(delta, 0) + 1
            self._table.move_to_end(self._last_delta)
        self._evict()

        # prediction for the current delta (mode of observed follow-ups)
        bucket = self._table.get(delta)
        if bucket:
            predicted_next = max(bucket, key=bucket.get)
        else:
            predicted_next = delta or _DEFAULT_DELTA

        self._last_delta = delta
        self._last_addr = addr
        nxt = addr + predicted_next
        return nxt if nxt >= 0 else addr + _DEFAULT_DELTA

    def _evict(self) -> None:
        while len(self._table) > self.capacity:
            self._table.popitem(last=False)  # LRU: evict the least-recent key


def prefetcher_configs() -> tuple:
    """All configs that the simulator can run (core 3 + modeled baselines)."""
    return SIM_CONFIGS


def make_sim_prefetcher(name: str, machine=None, **kwargs) -> Optional[Prefetcher]:
    """Resolve a prefetcher for any of the five experiment configs."""
    from .prefetchers import make_prefetcher

    if name in ("baseline", "none"):
        return None
    if name == "nextline":
        return NextLinePrefetcher(
            line_size=machine.line_size if machine is not None else LINE_SIZE
        )
    if name == "berti":
        return BertiPrefetcher()
    return make_prefetcher(name, **kwargs)

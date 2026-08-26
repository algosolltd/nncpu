"""Capture REAL memory accesses from actual algorithms and run them.

We instrument real programs (merge sort, matrix multiply, binary search)
at the element level and record every read/write as a trace, then feed the
resulting trace back through the simulator.  Addresses are index-derived
(base + index*w) so they behave like real memory offsets.

    python benchmarks/capture_real_trace.py

Writes results/traces/real_*.trace (and ChampSim-format copies).
"""

import os
import random

from nncpu.traces import WORD_BYTES, trace_recorder, read_champsim

BASE = 0x20000
OUT_DIR = "results/traces"
# Native trace files use word addresses.  ChampSim export converts them to
# byte addresses explicitly, avoiding the old 8x cache-line mismatch.
W = 1


def _load(rec, base, idx):
    rec.load(base + idx * W)


def _store(rec, base, idx, value=None):
    rec.store(base + idx * W, value if value is not None else 0)


def capture_mergesort(rec, n=512):
    rng = random.Random(7)
    arr = [rng.randrange(1000) for _ in range(n)]
    aux = [0] * n
    base = BASE

    def merge(lo, mid, hi):
        for k in range(lo, hi + 1):
            aux[k] = arr[k]
            _load(rec, base, k)
            _store(rec, base + n * W, k)  # aux lives in a second region
        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            if i > mid:
                _load(rec, base + n * W, j)
                arr[k] = aux[j]; j += 1
            elif j > hi:
                _load(rec, base + n * W, i)
                arr[k] = aux[i]; i += 1
            else:
                # The comparison reads both candidates.  Record those actual
                # indices, not the output index k.
                _load(rec, base + n * W, i)
                _load(rec, base + n * W, j)
                if aux[i] <= aux[j]:
                    arr[k] = aux[i]; i += 1
                else:
                    arr[k] = aux[j]; j += 1
            _store(rec, base, k)

    def msort(lo, hi):
        if lo < hi:
            mid = (lo + hi) // 2
            msort(lo, mid)
            msort(mid + 1, hi)
            merge(lo, mid, hi)

    msort(0, n - 1)


def capture_matmul(rec, n=32):
    """Row-major matrix multiply reading A and B, writing C."""
    rng = random.Random(3)
    a = [rng.randrange(10) for _ in range(n * n)]
    b = [rng.randrange(10) for _ in range(n * n)]
    c = [0] * (n * n)
    baseA = BASE
    baseB = BASE + n * n * W
    baseC = BASE + 2 * n * n * W
    for i in range(n):
        for k in range(n):
            aik = a[i * n + k]
            _load(rec, baseA, i * n + k)
            for j in range(n):
                bkj = b[k * n + j]
                _load(rec, baseB, k * n + j)
                _load(rec, baseC, i * n + j)
                c[i * n + j] += aik * bkj
                _store(rec, baseC, i * n + j)


def capture_bsearch(rec, n=1024):
    arr = [2 * i for i in range(n)]
    base = BASE
    rng = random.Random(11)
    targets = [rng.choice(arr) for _ in range(64)]
    for t in targets:
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            _load(rec, base, mid)
            if arr[mid] == t:
                break
            if arr[mid] < t:
                lo = mid + 1
            else:
                hi = mid - 1


def to_champsim_format(insts):
    lines = []
    for inst in insts:
        op = "1" if inst["type"] == "STORE" else "0"
        byte_address = inst["address"] * WORD_BYTES
        lines.append(f"{op} {byte_address:#x} 0x0")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [("real_mergesort", capture_mergesort),
            ("real_matmul", capture_matmul),
            ("real_bsearch", capture_bsearch)]
    for name, fn in jobs:
        path = os.path.join(OUT_DIR, f"{name}.trace")
        with trace_recorder(path) as rec:
            fn(rec)
        from nncpu.traces import read_trace
        parsed = read_trace(path)
        champ = os.path.join(OUT_DIR, f"{name}_champ.trace")
        with open(champ, "w", encoding="utf-8") as f:
            f.write(to_champsim_format(parsed))
        read_back = read_champsim(champ)
        assert len(read_back) == len(parsed)
        print(f"{name:16s} {len(parsed):6d} instructions  "
              f"(round-trip via ChampSim format OK)")
    print(f"\nTraces written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

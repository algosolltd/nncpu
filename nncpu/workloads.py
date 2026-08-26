"""Benchmark workloads as generated instruction streams.

Every generator yields instruction dicts of the form
``{"type": ..., "operands": [...]}`` or
``{"type": ..., "address": ..., "value": ...}``.

The memory workloads are *unwrapped monotonic streams* -- the classic
scenario cache prefetching targets -- so misses genuinely dominate and
there are no modulo wrap-around outliers to poison learning.
"""

BASE = 0x1000
REGION = 2048  # words; >> the ~32-line L1 cache, so reuse is minimal

_ARITH_OPS = ("ADD", "MUL", "DIV")


def sequential_read(n=2000, base=BASE):
    """Straight-line stride-1 read stream."""
    for i in range(n):
        yield {"type": "LOAD", "address": base + i}


def strided_read(n=2000, base=BASE, step=8):
    """Read stream with a constant stride (gather-like pattern)."""
    for i in range(n):
        yield {"type": "LOAD", "address": base + i * step}


def mixed_read_write(n=2000, base=BASE):
    """Interleaved store/load touching each address twice back-to-back."""
    for i in range(n):
        addr = base + (i // 2) * 8
        if i % 2 == 0:
            yield {"type": "STORE", "address": addr, "value": i}
        else:
            yield {"type": "LOAD", "address": addr}


def random_read(n=2000, base=BASE, region=REGION, seed=0x12345678):
    """Deterministic pseudo-random reads (LCG) -- nothing to prefetch."""
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        yield {"type": "LOAD", "address": base + (state % region)}


def phased_switch(n=2000, base=BASE, phase_len=250, region=REGION, seed=0x12345678):
    """Alternating predictable and unpredictable phases.

    Even phases are stride-1 sequential reads; odd phases are pseudo-random
    (LCG) reads.  This is the workload that exercises the confidence gate:
    it must prefetch during easy phases and stop during the noisy ones.
    """
    state = seed
    i = 0
    counter = 0
    while i < n:
        phase = i // phase_len
        if phase % 2 == 0:
            # predictable: stride-1 stream
            for _ in range(phase_len):
                if i >= n:
                    return
                yield {"type": "LOAD", "address": base + counter}
                counter += 1
                i += 1
        else:
            # unpredictable: LCG jitter
            for _ in range(phase_len):
                if i >= n:
                    return
                state = (1103515245 * state + 12345) % (2 ** 31)
                yield {"type": "LOAD", "address": base + (state % region)}
                i += 1


def arithmetic_mix(n=2000):
    """Pure arithmetic workload (no memory traffic)."""
    for i in range(n):
        yield {"type": _ARITH_OPS[i % 3], "operands": [i + 2, (i % 9) + 1]}


WORKLOADS = {
    "sequential_read": sequential_read,
    "strided_read": strided_read,
    "mixed_read_write": mixed_read_write,
    "random_read": random_read,
    "phased_switch": phased_switch,
    "arithmetic_mix": arithmetic_mix,
}


def build_workloads(length=2000, seed=None) -> dict:
    """Materialize workloads, optionally varying stochastic streams by seed.

    Within one run every prefetch configuration receives the exact same
    materialized instructions.  Across experiment runs, supplying ``seed``
    produces independent deterministic random/phased streams rather than
    merely changing neural-network initialization.
    """
    workloads = {
        "sequential_read": list(sequential_read(length)),
        "strided_read": list(strided_read(length)),
        "mixed_read_write": list(mixed_read_write(length)),
        "random_read": list(random_read(length, seed=seed if seed is not None else 0x12345678)),
        "phased_switch": list(phased_switch(length, seed=seed if seed is not None else 0x12345678)),
        "arithmetic_mix": list(arithmetic_mix(length)),
    }
    return workloads

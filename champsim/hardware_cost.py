#!/usr/bin/env python3
"""Storage accounting for an exact incremental regularity gate.

This is a transparent bit-count model, not an area/timing synthesis.  The
incremental design keeps the exact delta FIFO plus a fully associative table
of at most ``window`` distinct deltas and their occurrence counts.
"""

from __future__ import annotations

import argparse
import json
import math


def storage_cost(window: int, delta_bits: int) -> dict:
    if window <= 0 or delta_bits <= 0:
        raise ValueError("window and delta_bits must be positive")
    count_bits = math.ceil(math.log2(window + 1))
    pointer_bits = max(1, math.ceil(math.log2(window)))
    fill_bits = count_bits
    fifo_bits = window * delta_bits
    histogram_bits = window * (delta_bits + count_bits + 1)
    control_bits = pointer_bits + fill_bits
    total_bits = fifo_bits + histogram_bits + control_bits
    return {
        "implementation": "exact incremental associative histogram",
        "window_entries": window,
        "signed_delta_bits": delta_bits,
        "count_bits": count_bits,
        "fifo_bits": fifo_bits,
        "histogram_bits": histogram_bits,
        "control_bits": control_bits,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8,
        "fraction_of_dpc3_64kib_budget": total_bits / (64 * 1024 * 8),
        "scope": "regularity gate only; predictor storage excluded",
        "caveat": "storage model only; comparator/update logic is not synthesized",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--delta-bits", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(storage_cost(args.window, args.delta_bits), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

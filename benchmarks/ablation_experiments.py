"""Additional ablations for the paper, reusing the scan harness.

* feature ablation  -- which MLP inputs matter (opcode / pc / abs addr / delta)
* mlp backends      -- numpy vs sklearn: hit rate AND per-run wall time
* warm-up epochs    -- epochs on the very first fit (affects initial spike)

Run:  python benchmarks/ablation_experiments.py [--quick]
"""

import argparse
import time

from nncpu.prefetchers import FEATURE_MODES
from paper_experiments import BASE_NN, CONFIGS  # type: ignore

MEMORY_WORKLOADS = ("sequential_read", "strided_read", "mixed_read_write", "random_read")


def feature_ablation(length, seeds):
    from paper_experiments import scan_nn  # type: ignore

    variants = [(f"feat_{mode}", dict(feature_mode=mode)) for mode in FEATURE_MODES]
    scan_nn("exp_feature_ablation", variants, MEMORY_WORKLOADS, CONFIGS,
            seeds, length, description="MLP input feature ablation")


def backend_ablation(length, seeds):
    """numpy vs sklearn on the same streams, plus per-run wall clock."""
    from nncpu.benchmark import run_workload
    from paper_experiments import _loads, _row, save_scan  # type: ignore

    workloads_cache = _loads(length)
    rows = []
    for backend in ("numpy", "sklearn"):
        for seed in range(seeds):
            nn_kwargs = dict(BASE_NN, random_state=seed, mlp_backend=backend)
            for wname in MEMORY_WORKLOADS:
                insts = workloads_cache[wname]
                for cfg in CONFIGS:
                    if cfg == "baseline":
                        t0 = time.perf_counter()
                        rep = run_workload(insts, cfg)
                        dt = time.perf_counter() - t0
                    else:
                        t0 = time.perf_counter()
                        rep = run_workload(insts, cfg, nn_kwargs=nn_kwargs)
                        dt = time.perf_counter() - t0
                    rows.append(_row(seed, backend, wname, cfg, rep,
                                     extra={"wall_ms": round(dt * 1000, 2)}))
    df, outdir = save_scan("exp_mlp_backends", rows,
                           f"numpy vs sklearn backend x {seeds} seeds")

    # wall-time comparison: backend labels only for the NN config
    wall = df[df.config == "nn"].groupby(["label", "workload"])["wall_ms"].mean()
    pivot = wall.unstack()
    print("\n### exp_mlp_backends : NN wall time (ms) per workload")
    print(pivot.round(1).to_string())
    ratio = (pivot.loc["sklearn"] / pivot.loc["numpy"]).mean()
    print(f"sklearn/numpy wall-time ratio (mean): {ratio:.1f}x")


def warmup_ablation(length, seeds):
    from paper_experiments import scan_nn  # type: ignore

    variants = [(f"warm_{e}", dict(warmup_epochs=e)) for e in (1, 2, 4, 8)]
    scan_nn("exp_warmup_epochs", variants, ("strided_read",), CONFIGS,
            seeds, length, description="first-fit warm-up epochs sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--length", type=int, default=2000)
    args = ap.parse_args()

    seeds = 2 if args.quick else 5
    length = 600 if args.quick else args.length
    t0 = time.perf_counter()
    print(f"ablations | {length} instrs | numpy MLP")
    feature_ablation(length, seeds)
    backend_ablation(length, seeds)
    warmup_ablation(length, seeds)
    print(f"\nTotal ablation wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
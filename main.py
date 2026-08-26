"""CLI entry point: run reproducible, multi-seed experiments.

Every invocation records a fully-specified experiment under
``results/<name>/`` (config, provenance manifest, raw runs, aggregated
statistics, LaTeX table, figures) so the paper can cite exact numbers.

Usage:
    python main.py                        # quick defaults, 1 seed
    python main.py --runs 10 --name paper --length 4000
    python main.py --name sweep --l1 64 --latency 100 --runs 5
    python main.py --quick                # small smoke-test run
"""

import argparse
import sys
import time

from nncpu import __version__
from nncpu.cpu import MachineConfig
from nncpu.experiment import ExperimentConfig, run_experiment, summary_table
from nncpu.workloads import WORKLOADS
from nncpu.benchmark import CONFIGS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nncpu reproducible experiments")
    parser.add_argument("--name", type=str, default=None,
                        help="experiment directory name under results/ "
                             "(default: timestamped)")
    parser.add_argument("--description", type=str, default="",
                        help="one-line note recorded in config.json")
    parser.add_argument("--runs", type=int, default=1,
                        help="number of seeds to run (baseline/stride are "
                             "deterministic; the NN varies per seed)")
    parser.add_argument("--length", type=int, default=2000,
                        help="instructions per workload stream")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run (300 instructions, 3 seeds)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; seed r uses random_state=seed+r")
    parser.add_argument("--workloads", type=str, default=None,
                        help="comma-separated subset of: "
                             + ", ".join(WORKLOADS))
    parser.add_argument("--configs", type=str, default=None,
                        help="comma-separated subset of: "
                             + ", ".join(CONFIGS))
    machine = parser.add_argument_group("machine")
    machine.add_argument("--l1", type=int, default=None, help="L1 capacity in lines")
    machine.add_argument("--line", type=int, default=None, help="cache line size in words")
    machine.add_argument("--latency", type=int, default=None,
                         help="DRAM latency on a miss (cycles)")
    machine.add_argument("--wb", type=int, default=None,
                         help="store-buffer limit (writes before a stall)")
    nn = parser.add_argument_group("neural network prefetcher")
    nn.add_argument("--nn-batch", type=int, default=None,
                    help="MLP training batch / replay threshold")
    nn.add_argument("--nn-hidden", type=int, default=None,
                    help="MLP hidden-layer width (single layer)")
    nn.add_argument("--nn-lr", type=float, default=None,
                    help="MLP learning rate")
    nn.add_argument("--mlp", choices=("numpy", "sklearn"), default=None,
                    help="MLP backend for the NN prefetcher (default: numpy)")
    return parser.parse_args(argv)


def configure(args: argparse.Namespace) -> ExperimentConfig:
    """Translate CLI flags into an ExperimentConfig."""
    machine = MachineConfig()
    if args.l1 is not None:
        machine.l1_lines = args.l1
    if args.line is not None:
        machine.line_size = args.line
    if args.latency is not None:
        machine.mem_latency = args.latency
    if args.wb is not None:
        machine.wb_limit = args.wb

    nn_kwargs = {}
    if args.nn_batch is not None:
        nn_kwargs["batch_size"] = args.nn_batch
    if args.nn_hidden is not None:
        nn_kwargs["hidden_layers"] = (args.nn_hidden,)
    if args.nn_lr is not None:
        nn_kwargs["learning_rate"] = args.nn_lr
    if args.mlp is not None:
        nn_kwargs["mlp_backend"] = args.mlp

    name = args.name or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    length = 300 if args.quick else args.length
    runs = 3 if args.quick else args.runs

    return ExperimentConfig(
        name=name,
        description=args.description,
        length=length,
        runs=runs,
        seed=args.seed,
        workloads=_split(args.workloads, WORKLOADS),
        configs=_split(args.configs, CONFIGS) or CONFIGS,
        machine=machine,
        nn_kwargs=nn_kwargs,
    )


def _split(value: str | None, options: tuple) -> tuple:
    if value is None:
        return tuple(options)
    parts = [v.strip() for v in value.split(",") if v.strip()]
    unknown = set(parts) - set(options)
    if unknown:
        raise ValueError(f"Unknown options {sorted(unknown)}; expected one of {options}")
    return tuple(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = configure(args)

    print(f"nncpu v{__version__}")
    print(config.describe())
    t0 = time.perf_counter()
    result = run_experiment(config, root="results")
    elapsed = time.perf_counter() - t0

    print("\n=== Summary (mean over seeds, 95% CI) ===")
    print(summary_table(result.summary_df))
    print(f"\nStored under: {result.outdir}")
    for f in result.figures:
        print(f"  figure: {f}")
    print(f"Total wall time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
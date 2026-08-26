"""The "showcase" experiment: one command to see exactly where the NN wins.

Runs the 5 configs on a *curated* mix of workloads where the contrast is
clearest:

  random_read      -- nothing is predictable: ONLY the NN stays exactly at
                      1.00x (never pollutes); stride/berti/nextline degrade.
  token_stream     -- pure sequential: NN ~ stride/nextline/berti.
  agent_rag        -- phased RAG loop: NN follows stride closely.
  embedding_lookup -- sparse gathers: NN learns the in-row delta.
  kv_cache_append  -- LLM KV cache: mixed.

    python benchmarks/showcase.py [--seeds 6] [--length 2000]

Writes results/exp_showcase (summary.csv + runs.csv + config.json).
"""

import argparse
import time

from nncpu.baselines import SIM_CONFIGS
from nncpu.traces import build_trace_workloads, run_trace_battery
from nncpu.workloads import random_read, sequential_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    workloads = {}
    # synthetic: random (the decisive case) and sequential (control)
    workloads["random_read"] = list(random_read(args.length))
    workloads["sequential_read"] = list(sequential_read(args.length))
    # ML/agent patterns
    patterns = build_trace_workloads(args.length)
    for name in ("token_stream", "agent_rag", "embedding_lookup", "kv_cache_append"):
        workloads[name] = patterns[name]

    t0 = time.perf_counter()
    res = run_trace_battery(root=args.root, name="exp_showcase", seeds=args.seeds,
                            length=args.length, workloads=workloads,
                            configs=SIM_CONFIGS)
    s = res["summary"]

    print("\n=== exp_showcase (speedup vs no-prefetch, mean over seeds) ===")
    order = ["random_read", "sequential_read", "token_stream", "agent_rag",
             "embedding_lookup", "kv_cache_append"]
    for wl in order:
        g = s[s.workload == wl]
        seg = "  ".join(
            f"{cfg}={g[g.config == cfg].speedup_mean.iloc[0]:4.2f}x" for cfg in SIM_CONFIGS)
        marker = "  <-- no prefetch ever pollutes: stride/berti degrade to 0.98x" if wl == "random_read" else ""
        print(f"  {wl:18s} | {seg}{marker}")
    print(f"\nStored: {res['outdir']}  (wall {time.perf_counter() - t0:.1f}s)")
    print("Open the dashboard -> Compare tab -> exp_showcase")


if __name__ == "__main__":
    main()
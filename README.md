# nncpu — Latency-Accounting CPU Simulator with a Streaming NN Prefetcher

A small, honest research sandbox for studying how machine-learning-driven
cache prefetching hides main-memory latency.  It models an in-order scalar
core with an L1 cache and a write-through store buffer, then compares four
prefetch policies on a set of synthetic workloads.

The project replaces the earlier `cpuAI_test.py` proof-of-concept, which
measured raw Python call overhead instead of simulated CPU behavior and
trained its neural network on data unrelated to what it predicted.  This
version counts *cycles*, has real hit/miss accounting, and the NN learns
what it actually uses: the address stream.

## Architecture

```
nncpu/
├── cpu.py          # in-order core, L1 cache (LRU), write-through buffer
├── prefetchers.py  # stride heuristic + streaming MLPRegressor prefetcher
├── workloads.py    # 5 synthetic instruction-stream generators
├── benchmark.py    # harness + result table
├── visualize.py    # the 3 PNG figures
main.py             # CLI entry point
tests/              # pytest suite
```

### The machine (`nncpu/cpu.py`)

Every instruction advances a clock directly by its latency:

* fetch = 1 cycle; arithmetic `ADD`/`MUL`/`DIV` = 1 / 3 / 8 cycles
* a `LOAD`/`STORE` L1 hit costs 1 cycle; a miss fetches a whole 8-word
  line and stalls the core for `MEM_LATENCY = 40` cycles
* stores are write-through: the cache is updated immediately and a
  write-back is parked in a store buffer that only stalls the core when
  it overflows

The cache is fully-associative LRU (32 lines). The paper profile uses
`prefetch_latency=0`: `prefetch()` brings a line in immediately, modeling an
ideal upper bound with idle DRAM bandwidth. This is not a timeliness-aware
hardware result. Set `--prefetch-latency 40` to expose late predictions.

All numbers are counted honestly (`CPUReport`): hits, misses, hit rate,
prefetches issued vs. actually used, per-category cycle totals, IPC.

### The prefetchers (`nncpu/prefetchers.py`)

| Config   | Policy                                                                   |
|----------|--------------------------------------------------------------------------|
| baseline | no prefetching — every miss pays 40 cycles                               |
| stride   | classic heuristic: next address = last address + previous delta          |
| nextline | the classic next-cache-line prefetcher (`nncpu/baselines.py`)            |
| berti    | best-offset / 1st-order Markov delta (`nncpu/baselines.py`)              |
| nn       | streaming learned MLP delta-predictor + confidence gate                  |

The NN learns the *next-address delta* (predicting small deltas is far
better conditioned for a small regressor than absolute addresses).  Each
memory access closes one training sample — features of the previous
access, target = the delta that actually followed — and issues one
prediction.  A per-access confidence gate tracks recent prediction error:
when the stream becomes unpredictable (e.g. random access) it answers
`None` instead of prefetching. Warm-up predictions can still pollute the
cache, so safety is measured empirically rather than claimed universally.

Two MLP backends exist (`--mlp`): the default `numpy` (`nncpu/mlp.py`) is
a hand-rolled 6→32→1 ReLU net with Adam — fully deterministic per seed
and ~5–8× faster than `sklearn` while matching its hit rates
(scikit-learn then becomes optional).

### Workloads (`nncpu/workloads.py`)

Unwrapped monotonic memory streams whose footprint far exceeds the L1, so
misses genuinely dominate, plus a random stream and a pure-arithmetic mix:

* `sequential_read` — stride-1 read stream
* `strided_read` — constant-stride (gather-like) reads
* `mixed_read_write` — interleaved stores and loads
* `random_read` — deterministic LCG, nothing to prefetch
* `phased_switch` — predictable/noisy phases (exercises the gate)
* `arithmetic_mix` — no memory traffic at all

### Real / ML / agent traces (`nncpu/traces.py`)

Text trace I/O (`LOAD 0x1000`, `STORE 0x1a08 42`), a `trace_recorder`
context manager to capture real accesses from your own code, and built-in
patterns mapped to real workload classes — `kv_cache_append` (LLM KV
cache), `embedding_lookup` (sparse embeddings), `agent_rag` (RAG/tool
loop), `token_stream`.  Run them with:

```bash
python benchmarks/trace_experiments.py     # -> results/exp_traces/
python benchmarks/run_trace_file.py my.trace
```

## Paper and current result

`paper/main.tex` is now a matched-control audit, not a claim that the neural
prefetcher wins generally. The audit separates prediction from admission and
adds asynchronous latency, 8-way profiles, memory issue bandwidth, finite
prefetch MSHRs, lookahead, a demand-only shadow cache, and correct experimental
units.

The strongest holdout result is a predictor-independent regularity filter:
over the last 16 deltas it issues only when the dominant delta has at least
35% support. Parameters were selected on seeds 0–9 and evaluated on 10–39.

| Constrained profile, lookahead 8 | filtered stride vs stride | traffic reduction |
|----------------------------------|--------------------------:|------------------:|
| random read                      | 1.0884x                   | 99.55%            |
| phased switch                    | 1.0759x                   | 83.07%            |
| agent RAG                        | 1.0159x                   | 20.86%            |
| sequential / strided / mixed / KV| 1.0000x                   | 0%                |

The matched NN comparison is mostly negative: it is indistinguishable from
the classical filtered control on random, about 1.10x faster on one timed KV
condition, but substantially slower on strided and mixed traffic. This is
evidence for admission control and careful attribution—not proof of hardware
speedup. Native ChampSim/SPEC/GAP evaluation remains the next validity gate.

Compile from `paper/` with `pdflatex main.tex && pdflatex main.tex`.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q            # unit, integration and evidence tests
python main.py                       # default experiment (stored under results/)
python -m benchmarks.verify_paper --seeds 1   # source/artifact/claim parity
python -m benchmarks.verify_research          # matched-control holdout artifact

# static dashboard (no server needed to view)
python webapp/make_dashboard.py      # builds dashboard.html from results/
python -m http.server 8501           # then open http://localhost:8501/dashboard.html
```

The dashboard inlines all result data, while Plotly itself is loaded from a
pinned CDN URL, so charts require network access. Rebuild it after every
experiment battery with `make_dashboard.py`.

## Running a paper-ready experiment

Every experiment is reproducible end-to-end and stored as:

```
results/<name>/
├── config.json    # fully resolved machine + NN + workload + seed specification
├── manifest.json  # git revision, source/config hashes, versions, host, time
├── runs.csv       # one row per (seed, workload, config): raw cycles, hits...
├── speedups.csv   # per-run speedup vs baseline
├── summary.csv    # per (workload, config): mean, std, 95% CI
├── table.tex      # booktabs LaTeX table for the paper
├── figures/       # speedup / hit-rate / cycles (with error bars)
└── README.txt     # reproduce command + provenance at a glance
```

Example that produces a 10-seed dataset for the paper:

```bash
python main.py --name main --description "synthetic streams, 10 seeds" \
    --runs 10 --length 4000
```

Every run uses `seed + run` both for NN initialization and for stochastic
workloads; all configs within that run receive the same materialized stream.
Deterministic workloads still vary only through NN initialization. The
dashboard can load earlier experiments for inspection or export.

For publication, run the complete evidence check after generating artifacts:

```bash
python -m benchmarks.verify_paper --seeds 30
```

It checks the generating commit, clean-tree status, source/config hashes,
raw-to-summary aggregation, machine-readable claims, exact reruns, and reports
zero-latency versus timed-prefetch sensitivity separately.

## Visualizations

`operation_comparison_log.png` — per-instruction cycle cost by workload and
config (log boxplot).  `memory_vs_arithmetic_log.png` — memory-stall vs
arithmetic cycle totals.  `performance_metrics_log.png` — hit rate,
prefetch accuracy, and speedup per config.

## Ideas for extension

* multi-level cache / set-associative L1, TLB-style page tracking,
* a derived/value-prediction MLP for the arithmetic units,
* turning the confidence gate into an explicit forget/re-learn policy,
* a real cycle-by-cycle pipelined front-end (fetch/decode/exec/commit),
* plugging real traces (SPEC, ChampSim) into the same prefetcher
  interface, and comparing against published prefetchers (see below).

## License

MIT; see `LICENSE`.

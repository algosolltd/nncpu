# nncpu — Cycle-Accurate CPU Simulator with a Streaming NN Prefetcher

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

The cache is fully-associative LRU (32 lines).  Because
`prefetch()` brings a line in at zero cost (it models otherwise-idle DRAM
bandwidth), the whole game is *predicting which line comes next*.

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
`None` instead of prefetching, so the NN **never pollutes the cache** —
the thing a blind stride prefetcher cannot do.

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

## Paper

`paper/main.tex` is a draft skeleton with the real numbers and figures
already embedded (30-seed table, trace table, phasing/money plot).
Compile from `paper/`: `pdflatex main.tex && pdflatex main.tex`.

## Results

Full run (2000 instructions per workload):

| Workload          | Speedup vs no-prefetch (cycles) | Hit rate (base→stride→nn) |
|-------------------|-------------------------------:|----------------------------|
| sequential_read   | stride 3.4x / nn 3.3x          | 0.88 → 1.00 → 1.00         |
| strided_read      | stride 19.9x / nn 9.8x         | 0.00 → 1.00 → 0.94         |
| mixed_read_write  | stride 8.7x / nn 7.0x          | 0.50 → 1.00 → 0.98         |
| random_read       | stride 0.98x / nn 1.00x        | 0.10 → 0.08 → 0.10         |
| arithmetic_mix    | 1.00x (no memory)              | —                           |

Takeaways:

* on smooth streams the NN closes most of the gap to the (near-optimal)
  stride heuristic while adapting automatically;
* on unpredictable traffic stride *hurts* (pollutes the cache, hit rate
  drops below baseline) whereas the NN's confidence gate backs off and
  matches baseline exactly;
* NN speedups are real and reproducible because they come from the
  simulated cycle model, not from comparing Python call overhead.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q            # 31 unit + integration tests
python main.py                       # default experiment (stored under results/)

# static dashboard (no server needed to view)
python webapp/make_dashboard.py      # builds dashboard.html from results/
python -m http.server 8501           # then open http://localhost:8501/dashboard.html
```

The dashboard is a self-contained HTML file (tabs + Plotly charts) with the
result data inlined — open it directly or serve the folder with any static
server. Rebuild it after every experiment battery with `make_dashboard.py`.

## Running a paper-ready experiment

Every experiment is reproducible end-to-end and stored as:

```
results/<name>/
├── config.json    # exact machine + NN + workload + seed specification
├── manifest.json  # provenance: git revision, library versions, host, time
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

baseline/stride are deterministic (their std is 0); the **NN is seeded per
run** (`random_state = seed + run`) so its mean, std, and 95% CI are real
statistics. The dashboard runs the same pipeline and can load any earlier
experiment from disk for inspection or export.

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

MIT License.
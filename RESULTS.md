# Results (paper dataset)

Canonical run: **30 seeds**, 5 workloads × 3 configs, 2000 instrs, numpy MLP.
Everything here is regenerable with:

```bash
python benchmarks/paper_experiments.py      # full battery
python benchmarks/ablation_experiments.py   # feature / backend / warm-up ablations
python benchmarks/stat_tests.py             # paired significance tests
```

## Main results (mean over 30 seeds)

| Workload | Baseline | Stride | NN (95% CI) | Hit base→stride→nn |
|---|---|---|---|---|
| sequential_read | 13 750 | 3.40x (1.000) | **3.33x ± 0.03** (0.998) | 0.88 → 1.00 → 1.00 |
| strided_read | 82 000 | 19.92x (0.999) | **9.12x ± 1.53** (0.933) | 0.00 → 1.00 → 0.93 |
| mixed_read_write | 43 984 | 8.69x (0.999) | **6.12x ± 0.65** (0.971) | 0.50 → 1.00 → 0.97 |
| random_read | 74 083 | 0.98x (0.082) | **1.00x (0.102)** | 0.10 → 0.08 → 0.10 |
| phased_switch | 43 819 | 1.10x (0.541) | **1.03x (0.508)** | 0.49 → 0.54 → 0.51 |

**Take-away**: the NN closes most of stride's gap on regular streams and —
unlike stride — is **never worse than "no prefetch"** (random_read: stride
0.98x and hit rate *below* baseline; NN exactly at baseline on all 30 seeds).

### Figures

![Speedup vs baseline](./results/final_30/figures/speedup.png)

![L1 hit rate](./results/final_30/figures/hit_rate.png)

![Total cycles (log)](./results/final_30/figures/cycles.png)

**The money plot** — phased workload, hit rate per phase (grey = noisy):

![Phased hit rate](./results/exp_phased/figures/phases.png)

The NN follows baseline in the grey (noisy) phases where stride collapses
below baseline, and re-learns the predictable phases.

## Ablations

**Features** (speedup on strided_read / mixed_read_write): dropping the
absolute address feature **helps a lot** — `delta_only` and `no_abs_addr`
≈ 16.5x vs 8.3x for the full encoding; removing `pc` hurts.

| variant | strided | mixed | 
|---|---|---|
| full | 8.3x | 6.2x |
| delta_only | **16.5x** | **8.1x** |
| no_abs_addr | **17.0x** | 8.0x |
| no_pc | 6.9x | 6.5x |

**MLP backend**: numpy is **~9x faster** than sklearn end-to-end with
statistically identical hit rates (0.93–0.99).

**Gate threshold**: too strict (4 words) disables prefetching on
mixed (1.01x); 8–64 keeps NN≈baseline on random and NN≥threshold-fold on
smooth streams.

**Significance (Wilcoxon signed-rank, paired by seed, α=0.05):** NN beats
baseline on every workload (p<10⁻⁵ except random, distributions identical);
stride beats NN on smooth streams (p<10⁻⁵) but **loses to baseline on
random_read** (p<10⁻⁷). Full table in `results/final_30/statistics.csv`.

## ML / agent traces (`results/exp_traces`, 8 seeds)

Access patterns mapped to real workload classes (LLM KV cache, sparse
embeddings, RAG agent loop, token stream):

| trace | baseline | stride | NN (median gate) |
|---|---|---|---|
| token_stream (sequential) | 1.00x | 3.40x | **3.26x** |
| agent_rag (phased) | 1.00x | 2.14x | **1.95x** |
| embedding_lookup (sparse) | 1.00x | 1.88x | **1.64x** |
| kv_cache_append (seq+gather) | 1.00x | 1.97x | **1.33x** |

**Robustness fix (issue #2 partial):** switching the confidence gate from
mean to **median** error and **clamping training deltas to ±16 words** lets
the NN handle sparse *gather* streams without being tripped ("everything
looks unpredictable"). Ablation (`results/exp_gate_aggregators`): embedding
1.00x→1.64x, kv_cache 1.00x→1.33x, agent_rag 1.84x→1.95x, with **no
regression** on random_read (NN still exactly = baseline) or the smooth
streams.

## Published baselines, modeled in-simulator (`results/exp_baselines*`)

Next-Line and Berti (best-offset) implemented faithfully in
`nncpu/baselines.py`, compared on synthetic + patterns + **real algorithm
traces** (merge sort, matmul, binary search from `capture_real_trace.py`):

| workload | baseline | stride | nextline | berti | NN |
|---|---|---|---|---|---|
| token_stream | 1.00x | 3.40x | 3.40x | 3.40x | 3.26x |
| random_read | 1.00x | 0.98x | 1.00x | 0.98x | **1.00x** |
| real_matmul | 1.00x | 9.63x | **11.24x** | 8.46x | 1.00x |
| real_mergesort | 1.00x | 6.86x | 8.68x | **9.04x** | 1.00x |
| kv_cache_append | 1.00x | 1.97x | 1.00x | **2.76x** | 1.33x |

Honest interpretation: on dense large-stride computation (matmul,
mergesort) the classic mechanisms are near-optimal and the NN's gate is
too conservative (threshold 8 words ≪ the 32-word row stride) — a clear
limitation, not hidden.  The NN's clear win stays the unpredictable case:
on `random_read` its hit rate stays exactly at baseline (no pollution)
while stride and berti drop below (0.98x, hit 0.081/0.082).

Full raw data lives under `results/` (see `README.md` for the layout).
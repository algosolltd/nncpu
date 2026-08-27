# Alternative application: regularity-guided GPU Unified Memory

Research scouting performed on 26--27 August 2026. This document records the
alternative applications considered for the mechanism in the nncpu paper, the
evidence used to select one, and a falsifiable validation plan. It is a research
proposal, not evidence that the proposed UVM policy improves performance.

## Executive conclusion

The strongest alternative application is a lightweight online dispatcher for
CPU--GPU page migration in oversubscribed Unified Memory (UVM).

The reusable contribution is not the MLP address predictor. It is the
predictor-independent regularity statistic that determines whether a costly
speculative action is justified. Applied at the GPU memory-management unit,
the statistic can select among:

1. demand paging, when recent page faults are irregular;
2. a classical page-stride or sequential-local prefetcher, when a dominant
   page delta is stable; and
3. a learned page predictor, only when the classical structure is insufficient
   and the expected benefit can repay model inference.

The resulting research question is:

> Can a roughly 269-byte base regularity detector avoid unnecessary learned
> UVM predictions and speculative page migrations while preserving application
> runtime and reducing PCIe traffic, page thrashing, and predictor cost?

This is a better fit than simply moving the existing L1 gate unchanged. The
gate becomes a **model and speculation dispatcher**: it decides when learning
is warranted, rather than claiming that learning is itself the source of the
benefit.

## Mechanism transferred from nncpu

For the most recent window of address or page deltas, define

\[
R_t = \frac{\max_d \left|\{i : d_i=d\}\right|}{W}.
\]

The current nncpu design uses `W=16`, a warm-up of eight deltas, and opens when
the dominant delta has at least 35% support. Its exact incremental storage
model uses a 16-entry signed 64-bit FIFO plus a 16-entry associative histogram:
2,153 bits, or 269.125 bytes. Comparator/update logic, timing, area, and energy
have not been synthesized.

For UVM, the observed address becomes a page number. A first implementation
should keep separate windows per source that preserves locality, such as SM,
PC, or allocation, because a single global stream can interleave unrelated GPU
warps. The cited learned-UVM work found SM identifier especially useful for
clustering. Per-context windows multiply the storage cost, so the number and
replacement policy of contexts must be reported rather than hidden in the
base 269-byte figure.

The inherited `16/8/35%` parameters are a fixed starting point, not presumed
optimal for UVM. If tuning is performed, it must use a development split and
freeze the policy before holdout. Warm-up behavior also needs an explicit
choice: nncpu admits during warm-up, while costly page migration may justify a
conservative demand-only warm-up. Both can be evaluated, but one must be
predeclared as primary.

## Why GPU page migration is the best target

### Direct signal compatibility

UVM exposes a scalar sequence of faulting page numbers. Page-number deltas are
the same mathematical object as the block deltas already processed by the
gate. No semantic embedding or learned representation is required.

### Speculation has an expensive, observable endpoint

A wrong cache-line prefetch may consume internal hierarchy traffic without
changing DRAM requests, as the ChampSim audit demonstrated. A wrong UVM page
migration consumes a PCIe/NVLink transfer, can evict a useful GPU-resident
page, and can later require the page to make a round trip. The relevant costs
are therefore measurable at the actual boundary:

- host-to-device and device-to-host bytes;
- PCIe/NVLink busy time and queueing;
- far-fault stall cycles and tail latency;
- evictions, refetches, and page round trips;
- kernel runtime, IPC, or application throughput;
- predictor invocations, wall-clock inference latency, and energy.

Accuracy, coverage, and page-hit rate remain diagnostics. They must not replace
these endpoints.

### Primary literature already exposes the opportunity

Long et al., *Deep Learning based Data Prefetching in CPU-GPU Unified Virtual
Memory*, report that page-address delta, address, and PC carry most of their
predictive information. In ATAX, one delta accounts for 99.26% of their
training outputs; removing sequence information then has almost no effect.
They explicitly connect dominant-delta convergence with reduced reliance on
self-attention. This is precisely the condition the nncpu regularity statistic
detects online.

The same paper reports that its modeled UVM predictor improves geometric-mean
IPC over UVMSmart when predictor latency is 1 microsecond, loses the advantage
at 5 microseconds, and becomes a slowdown at 10 microseconds. This makes model
dispatch an endpoint question, not merely a model-compression exercise. The
paper also reports IPC, page-hit rate, and PCIe usage, so its experimental
framework already contains several required endpoints.

### A public implementation exists

The public `gpgpu-sim_UVMSmart` artifact provides functional and timing UVM
simulation, managed-memory benchmarks, page migration and eviction, PCIe
transfer timing, and configurable prefetch policies. The revision inspected
for this scouting was:

```text
bafab5d728a911d671833a5f843eeda783dab9ac
```

The relevant integration seam is `gmmu_t::do_hardware_prefetch()` in
`src/gpgpu-sim/gpu-sim.cc`. Policy enumerations and per-GMMU state live in
`src/gpgpu-sim/gpu-sim.h`. Existing choices include demand-only, tree-based
neighborhood, sequential-local 64 KiB, and random prefetching, with separate
oversubscription configuration. This makes a matched gate/raw comparison
possible without building a new simulator.

## Proposed policies

The study should distinguish address prediction, admission, and model
dispatch. At minimum it should implement these policies:

| Policy | Address/policy source | Admission/model use |
|---|---|---|
| Demand | faulting page only | no speculation |
| TBN | UVMSmart tree-based neighborhood | always active |
| Sequential-local | UVMSmart 64 KiB policy | always active |
| Raw dominant stride | recent dominant page delta | always active |
| Regularity stride | identical dominant-stride predictor | admitted only by `R_t` |
| Learned | published/reproduced learned predictor | invoked on every decision |
| Regularity dispatcher | classical predictor at high `R_t`; learned or demand fallback otherwise | model invoked conditionally |

There are two distinct dispatcher hypotheses and they should not be merged
after seeing results:

- **Suppressor:** high regularity uses classical stride; low regularity uses
  demand paging. This maximizes simplicity and avoids speculative traffic.
- **Model selector:** high regularity uses classical stride; low regularity
  invokes the learned predictor. This asks whether the model is valuable only
  on the residual cases that classical regularity cannot explain.

Every gated policy needs an otherwise identical gate-disabled control. The
learned predictor also needs a matched admission control if it contains its own
confidence rule.

## Falsifiable experimental protocol

### Experimental units and split

- Use the UVM-managed Rodinia, Lonestar, Parboil, and related workloads
  distributed with UVMSmart.
- Treat one benchmark/input pair as one independent unit. Re-running a
  deterministic simulation is a reproducibility check, not a replicate.
- Split by whole program or input before tuning. Do not place two inputs of the
  same program on opposite sides unless program-level leakage is explicitly
  analyzed.
- Tune context count, window, support threshold, and fallback only on the
  development set. Commit the protocol and predictions before holdout.
- Evaluate both non-oversubscribed memory and at least 125% and 150%
  oversubscription, because page eviction changes the cost of early/wrong
  migration.

### Matched matrix

Run every benchmark/input with the same simulator revision, GPU configuration,
instruction budget, allocation sizes, and random state across the complete
policy matrix. A gate-disabled port must reproduce its upstream classical
control exactly before the gated result is interpreted.

### Primary endpoints

1. paired kernel-runtime or IPC ratio;
2. total host-to-device plus device-to-host bytes;
3. page round trips and evictions attributable to speculative migration;
4. accumulated and p95/p99 far-fault stall time;
5. learned-predictor calls, measured predictor time, and measured/modelled
   energy with assumptions reported separately.

Secondary diagnostics are prefetch accuracy, coverage, page-hit rate, gate-open
fraction, transitions between policies, and useful-prefetch timeliness.

### Pre-specifiable predictions

A defensible primary prediction for the model-selector policy is:

1. it reduces learned-predictor invocations substantially relative to
   always-learned prediction;
2. its runtime is equivalent within a frozen practical margin rather than
   merely “not significant”;
3. it does not increase aggregate PCIe traffic outside a frozen margin; and
4. on high-regularity units, the classical branch matches the learned branch
   without paying inference cost.

The precise margins and minimum useful reduction must be frozen after a
development pilot. Runtime/traffic equivalence needs an interval-based test
such as TOST when assumptions and sample size permit it; a descriptive
`+/-1%` rule must not be presented as a formal equivalence result.

Report a program-level bootstrap for effect magnitude and an exact sign test
for direction, with multiplicity correction across the declared comparison
family. If one application dominates the geometric mean, report both the
magnitude-sensitive aggregate and the direction-sensitive test, as in the
ChampSim audit.

### Falsification conditions

The idea fails as a performance/efficiency mechanism if any of the following
survives a properly powered holdout:

- fewer predictor invocations do not reduce predictor time or energy;
- higher prefetch accuracy does not reduce PCIe bytes or page round trips;
- traffic falls but far-fault stalls or runtime increase;
- per-SM/context state and lookup cost erase the model-invocation savings;
- the learned predictor adds value primarily in high-regularity cases, making
  `R_t` a poor dispatcher; or
- results depend on one program, one oversubscription point, or post-hoc
  threshold selection.

A negative result can still be publishable if it demonstrates, with matched
controls, that a learned UVM predictor's apparent advantage is attributable to
admission or a proxy metric rather than model inference.

## Alternative candidates considered

### MoE expert prefetching

This is attractive because expert weights are large, transfers can lie on the
critical path, and routing exhibits temporal/cross-layer structure. A gate
could suppress uncertain expert transfers or select the extra-prefetch budget.

It was not selected as the first target because the 2026 APEX work already
implements confidence-driven dynamic expert prefetch budgets using a learned
ordinal-logistic CDF, evaluates latency and energy-delay product, and targets
the same “how much should be prefetched?” question. A simple regularity gate is
still a valuable matched baseline for APEX-like systems, but categorical
expert sets require a per-layer transition/set statistic rather than the
current scalar global address delta. Public code was not located during this
scouting.

### Storage-backed sparse LLM weight reads

NeuroPrefetcher is a particularly relevant modern system. It treats NVMe as an
active weight tier, reports 82--85% token-to-token persistence of active
neurons, and fetches only incoming sparse rows. Its public code uses
`io_uring`/`O_DIRECT`, a neuron-major file, and `find_runs_gap1()` to coalesce
exactly consecutive neuron IDs. The inspected repository revision was:

```text
4c47f08b37e5b498ac0056c874b7aea45fc616fa
```

A future use of the gate is **regularity-guided read amplification**: bridge
small, stable gaps between requested neuron rows when the reduction in I/O
operations is expected to repay the extra bytes. Controls would be exact-only
coalescing, fixed-gap coalescing, and regularity-gated coalescing under the same
maximum byte budget. Endpoints would be tokens/s, per-token and tail latency,
NVMe bytes, IOPS, read amplification, and GPU idle time.

This was ranked second because it is topical and has real-hardware artifacts,
but it changes the observed object from a temporal address stream to ordered
sets of neuron IDs and requires a Jetson/model/NVMe environment for decisive
testing. It is best treated as a second validation domain after UVM.

### KV-cache prefetching

InfiniGen, asynchronous KV-cache prefetching, and PRESERVE establish that
selective or overlapped KV movement can improve LLM inference. A regularity
gate could control speculative KV-block movement between host memory, HBM, and
on-chip cache. However, important KV blocks depend on attention semantics, not
only numeric address deltas. The current gate would therefore be an admission
layer around an existing semantic predictor, not an address predictor by
itself. This is less direct and has more confounding model-quality effects than
UVM page migration.

### OS and storage read-ahead

Block/file read-ahead is mathematically the closest transfer: observe logical
block offsets and admit asynchronous reads only under stable deltas. It is easy
to prototype with real I/O endpoints, but mature sequential and strided
read-ahead policies make the novelty lower. A stronger systems question would
need cloud/object-store egress cost, multi-tenant interference, or a data-lake
workload rather than a generic local-file demonstration.

## Recommended implementation sequence

1. Add a standalone incremental page-delta regularity component with exhaustive
   unit tests and explicit state-bit accounting.
2. Insert raw dominant-stride and gate-enabled matched policies at
   `gmmu_t::do_hardware_prefetch()`; verify that gate-disabled output is exactly
   equal to the raw policy.
3. Add endpoint counters before running experiments: requested and transferred
   pages, PCIe bytes by direction, useful/late pages, evictions, refetches,
   round trips, far-fault stall distributions, and policy/model invocations.
4. Reproduce upstream demand, TBN, and sequential-local results before adding
   the learned baseline.
5. Run a small development pilot, choose context granularity and frozen
   parameters, and commit the holdout protocol.
6. Execute the full oversubscription matrix and reconstruct all summaries from
   raw logs with a machine-readable verifier.
7. Only after UVM validation, port the same endpoint-first methodology to
   NeuroPrefetcher read coalescing or an APEX-like expert prefetcher.

## Potential paper framing

Working title:

> **When Not to Learn: Regularity-Guided Predictor Dispatch for GPU Unified
> Memory**

The strongest possible positive contribution would be a tiny transparent
dispatcher that preserves the useful endpoint behavior of a learned page
predictor while avoiding most model invocations and speculative migrations.
The equally useful negative contribution would be evidence that dominant-delta
or prediction-accuracy proxies fail to predict PCIe traffic, thrashing, or
runtime. Either framing must retain matched admission and endpoint metrics as
the methodological core learned from the nncpu audit.

## Primary sources and artifacts

- Y. Long et al., [*Deep Learning based Data Prefetching in CPU-GPU Unified
  Virtual Memory*](https://arxiv.org/abs/2203.12672), 2022.
- D. Ganguly et al., [*A Framework for Memory Oversubscription Management in
  Graphics Processing Units*](https://doi.org/10.1145/3297858.3304044), ASPLOS
  2019.
- D. Ganguly et al., [*Adaptive Page Migration for Irregular Data-intensive
  Applications under GPU Memory Oversubscription*](https://doi.org/10.1109/IPDPS47924.2020.00054),
  IPDPS 2020.
- [UVMSmart public GPGPU-Sim artifact](https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart).
- H. Chien et al., [*Performance Evaluation of Advanced Features in CUDA
  Unified Memory*](https://arxiv.org/abs/1910.09598), 2019.
- [*APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE
  Inference*](https://arxiv.org/abs/2608.11688), 2026.
- N. Dhar et al., [*NeuroPrefetcher: Storage-Aware Sparse LLM Inference via
  Delta Prefetching*](https://arxiv.org/abs/2608.22643), ICPP 2026, and its
  [public artifact](https://github.com/nobeldhar/NeuroPrefetcher).
- [*InfiniGen: Efficient Generative Inference of Large Language Models with
  Dynamic KV Cache Management*](https://arxiv.org/abs/2406.19707), 2024.
- [*Accelerating LLM Inference Throughput via Asynchronous KV Cache
  Prefetching*](https://arxiv.org/abs/2504.06319), 2025.
- [*PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM
  Serving*](https://arxiv.org/abs/2501.08192), 2025.

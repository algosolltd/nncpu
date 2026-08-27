# Speculative decoding: endpoint-metric audit

## Result

A confidence gate can improve speculative-decoding acceptance while making
decoding slower. On the frozen eight-prompt holdout, raising llama.cpp's draft
confidence threshold from 0.8 to 0.95 changed aggregate acceptance from
90.47% to 95.95%, but reduced geometric-mean throughput from 27.24 to 24.41
tokens/s. The direct paired throughput ratio was 0.8963: threshold 0.95 lost
on all eight prompts (exact two-sided sign test, `p=0.0078125`).

The endpoint comparison with target-only decoding makes the practical effect
clear:

| policy | accepted draft fraction | geomean tokens/s | vs target-only |
|---|---:|---:|---:|
| target only | n/a | 25.63 | 1.0000x |
| gate 0.8, max 16 | 90.47% | 27.24 | **1.0627x** |
| gate 0.95, max 16 | **95.95%** | 24.41 | **0.9525x** |
| static length 2 | 79.73% | 24.19 | 0.9438x |
| static length 4 | 70.25% | 24.03 | 0.9375x |

The stricter gate proposed 25.75% fewer draft tokens and accepted 21.26% fewer
draft tokens in absolute terms. Its fraction improved because the denominator
fell faster than the numerator. This is the exact analogue of a prefetch gate
that improves prefetch accuracy by issuing less work without improving the
machine endpoint.

## Per-prompt matched contrast

Each entry is the median of three runs. One prompt, rather than one timing
repeat, is the independent unit.

| prompt | gate 0.8 t/s | gate 0.95 t/s | 0.95 / 0.8 | acceptance 0.8 → 0.95 |
|---|---:|---:|---:|---:|
| GSM8K 10 | 27.06 | 25.75 | 0.9518 | 87.34% → 100.00% |
| GSM8K 25 | 26.73 | 23.65 | 0.8848 | 96.88% → 100.00% |
| GSM8K 50 | 22.80 | 21.06 | 0.9235 | 68.89% → 72.58% |
| GSM8K 100 | 27.99 | 25.56 | 0.9130 | 83.87% → 98.33% |
| HumanEval 10 | 24.47 | 21.67 | 0.8857 | 98.21% → 100.00% |
| HumanEval 25 | 25.84 | 23.40 | 0.9057 | 98.33% → 100.00% |
| HumanEval 50 | 32.39 | 26.77 | 0.8265 | 100.00% → 100.00% |
| HumanEval 100 | 32.04 | 28.34 | 0.8845 | 100.00% → 100.00% |

The primary contrast was frozen after development and before any of these
holdout prompts were executed. It is the only confirmatory contrast, so no
multiple-comparison correction is applied. The full policy sweep is reported
descriptively.

## Experimental design

- Target: official Qwen2.5-3B-Instruct Q4_K_M GGUF.
- Draft: official Qwen2.5-0.5B-Instruct Q4_K_M GGUF.
- Runtime: CPU-only llama.cpp commit `5854625`, four threads pinned to logical
  CPUs 0--3, one request at a time, process priority `nice -n 10`.
- Workloads: four untouched HumanEval and four untouched GSM8K prompts, with
  source repositories and revisions pinned in the prompt artifact.
- Output: exactly 96 tokens per request, greedy sampling, draft branching off,
  no prompt cache.
- Replication: three timings per prompt/policy; deterministic policy order
  shuffle; medians used for endpoint comparisons.
- Artifact: 120 raw runs; target, draft, runtime, configuration, harness, and
  prompts are SHA-256 pinned. The verifier reconstructs every aggregate.

The result is directly visible in
[`summary.json`](../results/specdec_holdout_qwen_cpu/summary.json) and can be
reconstructed with:

```bash
python3 benchmarks/specdec_endpoint_audit.py verify \
  --output results/specdec_holdout_qwen_cpu
```

The runtime and model binaries are deliberately excluded from Git. Their
required revisions and hashes are in
[`manifest.json`](../results/specdec_holdout_qwen_cpu/manifest.json) and
[`specdec_holdout_config.json`](../benchmarks/specdec_holdout_config.json).

## Secondary numerical diagnostic

All policies generated exactly 96 tokens, but output hashes were not identical
for six of eight prompts. Each policy's hash was stable across all three
repetitions, and divergence changed systematically with draft policy. Static
length 2 matched target-only on all prompts, while longer/dynamic verification
shapes sometimes selected a different greedy continuation.

This does **not** show that speculative decoding changes the intended target
distribution. It shows that byte-identical greedy output is not a safe systems
invariant for this quantized CPU configuration. A plausible explanation is
finite-precision sensitivity to verification batch shape; proving the precise
kernel-level cause requires logits and kernel tracing not present here.
Consequently, the holdout fixes output-token count and makes no quality or
losslessness claim. Recent work independently documents that greedy output can
change with numerical precision, hardware count, and evaluation batch size.

## Relationship to prior work

Speculative decoding was introduced as distribution-preserving acceleration by
[Leviathan et al. (ICML 2023)](https://proceedings.mlr.press/v202/leviathan23a.html)
and [Chen et al.](https://arxiv.org/abs/2302.01318). Confidence-aware draft
stopping is not new: it appears in
[SpecDec++](https://arxiv.org/abs/2405.19715), Hugging Face's
[dynamic speculation implementation](https://huggingface.co/blog/dynamic_speculation_lookahead),
and [SVIP](https://aclanthology.org/2025.emnlp-main.844/).

Nor is the broad warning that acceptance is not throughput entirely new.
[Learning to Draft (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)
directly optimizes cycle throughput because acceptance-length objectives omit
draft and verification time. A 2026
[characterization of self-speculative decoding](https://openreview.net/pdf/1b1a683b2c8c838b082c772172999fb1011053cf.pdf)
also concludes that draft cost, not acceptance rate alone, dominates
throughput.

Therefore this experiment should not be sold as discovering the acceptance /
throughput distinction. Its value is:

1. an independent, machine-verifiable replication on a consumer CPU with a
   3B/0.5B quantized model pair;
2. a particularly clean matched-gate counterexample: same models, maximum
   draft length, prompts, output count, and runtime, with only the confidence
   threshold changed; and
3. a cross-domain bridge to the cache-prefetching result, where an admission
   gate likewise improves a shallow ratio while the endpoint worsens.

## How this fits the paper

The strongest paper is no longer a standalone neural-prefetcher paper or a new
speculative-decoding algorithm. It is a systems methodology paper about
**speculation attribution**:

> A predictor proposes work, a gate admits it, and a downstream machine pays
> for it. Improving the fraction of admitted work that succeeds does not prove
> an endpoint gain.

A suitable title is:

> **The Gate, Not the Predictor? Matched-Control and Endpoint-Metric Audits of
> Cache Prefetching and LLM Speculative Decoding**

The corresponding three-page IEEE-style manuscript is available as
[`paper/cross_domain.tex`](../paper/cross_domain.tex) and the compiled
[`paper/cross_domain.pdf`](../paper/cross_domain.pdf).

The two domains then provide independent demonstrations:

- Cache prefetching: accuracy rises 10.64% → 15.62%, but DRAM reads barely
  change and IPC falls 0.566%.
- LLM decoding: acceptance rises 90.47% → 95.95%, but throughput falls 10.37%
  relative to the less restrictive matched gate.

This cross-domain structure is more defensible than claiming a new gate. It
also fits performance-analysis and negative-results venues better than a pure
LLM algorithms venue, where the core proxy warning is already represented by
recent work.

## Limits and next evidence

The holdout has one consumer CPU, one quantized model pair, one runtime commit,
two datasets, eight prompt units, fixed 96-token outputs, and batch size one.
It establishes the stated counterexample on this system, not generality.

Before a full conference submission, add at least:

- a second model family/pair and a non-quantized or Q8 target;
- GPU measurements and a second runtime (for example Transformers or vLLM);
- longer contexts and natural-language summarization/dialogue prompts;
- energy or package-power measurements if efficiency is claimed;
- per-round draft/verify timing to explain the causal mechanism; and
- a controlled numerical study of the policy-specific output divergence.

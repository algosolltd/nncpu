# Speculative-decoding endpoint audit (frozen pilot protocol)

Status: development protocol frozen before the first model inference run on
2026-08-27; confirmatory revision frozen before any holdout prompt was run.

## Development finding and frozen holdout revision

The four-prompt development run falsified the byte-identity invariant: all
policies used greedy decoding, but different verification batch shapes changed
some generated token choices under Q4_K_M CPU arithmetic. This is consistent
with speculative decoding's finite-numerics caveat, but it means variable EOS
positions can confound a timing comparison. The development artifact is kept
as exploratory evidence and is not a confirmatory result.

Before inspecting any of the eight holdout prompts, protocol v2 therefore:

- forces exactly 96 generated tokens with `ignore_eos=true`;
- explicitly sets `top_k=1`, `top_p=1`, `min_p=0`, and temperature zero;
- disables draft branching with `draft_p_split=0`;
- retains output hashes as a numerical-divergence diagnostic rather than
  asserting byte identity;
- compares target-only, static lengths 2 and 4, and maximum-length-16 gates at
  thresholds 0.8 and 0.95; and
- uses three timing repetitions, collapsed to one median per prompt.

Threshold 0.95 was selected on development prompts because it produced a
higher accepted-token fraction but lower throughput than threshold 0.8. The
confirmatory prediction is that this proxy reversal persists on the untouched
holdout prompts: gate 0.95 has at least one percentage point higher aggregate
acceptance, while its geometric-mean paired throughput ratio to gate 0.8 is
below one. Output-token count equality is a hard invariant; output text
identity is not used to support a quality or distributional claim.

## Research question

Does a confidence gate look better because it rejects low-confidence draft
tokens, or does it improve the endpoint that a user actually experiences?

This is the speculative-decoding analogue of the prefetching attribution
problem in the main paper:

| Prefetching | Speculative decoding |
|---|---|
| address predictor | draft model |
| admission gate | draft stopping/confidence gate |
| accepted prefetch accuracy | accepted-token fraction |
| memory requests / IPC | decode latency and tokens/s |

The audit keeps the target, draft, quantization, runtime, prompt, sampling,
thread count, and maximum output fixed. It changes only maximum draft length
and the draft-confidence stopping threshold.

## Predeclared hypotheses

1. Raising the confidence threshold can increase accepted-token fraction by
   suppressing low-confidence proposals.
2. Accepted-token fraction alone will not rank policies by decode throughput.
3. A proxy reversal is established only if a policy with an aggregate
   accepted-token fraction at least one percentage point higher has lower
   geometric-mean throughput on the same holdout prompts.
4. In development, greedy target-only and speculative outputs are tested for
   byte identity. Protocol v2 instead fixes output-token count and reports
   byte divergence because development falsified the stronger invariant.

The result is hardware- and model-pair-specific. This pilot cannot establish a
general theorem or a quality improvement.

## Fixed system

- Runtime: `llama.cpp` commit `58546250cfaa1dc4a33ae6171dbe4344a96206e8`
- Target: official `Qwen/Qwen2.5-3B-Instruct-GGUF`, Q4_K_M, repository commit
  `7dabda4d13d513e3e842b20f0d435c732f172cbe`
- Draft: official `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, Q4_K_M, repository commit
  `9217f5db79a29953eb74d5343926648285ec7e67`
- Backend: CPU-only build; four generation and batch threads; one server slot;
  process priority `nice -n 10`; CPU affinity limited to logical CPUs 0--3.
- Maximum generated tokens: 96; greedy decoding; fixed seed; no prompt cache.
- Each server configuration receives one unmeasured warm-up request.

No package, driver, kernel, or system configuration is modified. Runtime and
model files live under the ignored `.research/` directory.

## Policies

- target-only autoregressive baseline;
- static speculation: maximum draft length 2, 4, 8, or 16 and threshold 0;
- confidence-gated speculation: maximum length 16 and threshold 0.2, 0.5, or
  0.8.

Static policies are required matched controls: they reveal whether a gated
policy benefits from confidence or merely from issuing a different amount of
draft work.

## Prompts and experimental units

Prompts come from the public HumanEval and GSM8K test sets. Development uses
indices 0 and 5 from each dataset. Holdout uses indices 10, 25, 50, and 100
from each dataset. The index lists are fixed before downloading or observing
model output.

One prompt is the independent unit. The three executions per prompt and policy
measure runtime noise and are collapsed to their median; they are not treated
as independent samples. Configuration order is deterministically shuffled in
each repeat block.

## Metrics and gates

Primary endpoint: server-reported generation time (`predicted_ms`) and its
equivalent throughput (`predicted_per_second`). Request wall time is retained
as a diagnostic. Proxy metrics are total draft tokens, accepted draft tokens,
accepted-token fraction, and mean accepted length where available.

For each policy, report the geometric mean of per-prompt throughput ratios to
target-only. For direct policy comparisons, report per-prompt paired ratios,
wins/losses, and an exact two-sided sign test. With only eight holdout prompts,
inferential statistics are descriptive and the scope remains a pilot.

## Resource envelope

- at most four inference threads and one concurrent request;
- model downloads capped at 6 MiB/s;
- no GPU execution in the primary experiment;
- no sudo, kernel modules, driver changes, or writes outside this repository;
- abort a run on server failure, output mismatch, or memory exhaustion.

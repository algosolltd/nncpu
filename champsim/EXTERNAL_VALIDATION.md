# Frozen external validation protocol

Recorded: 2026-08-26, after inspecting the eleven-program development pilot
and before simulating any holdout trace with the nncpu module.

## Fixed mechanism

The primary gate remains the mechanism selected on internal development seeds
0--9: 16 global block deltas, warm-up after 8 deltas, dominant-delta support
of at least 35%, and stride degree 3. No parameter is selected on the external
holdout. The gate-disabled module must match official ChampSim `ip_stride`
exactly on every recorded output field.

## Data and units

The external-development set is `dpc3_trace_set.txt`. The holdout is
`dpc3_holdout_trace_set.txt`: the highest-weight official DPC-3 SimPoint for
each of the nine remaining SPEC CPU2017 programs. One program is one
independent unit. SimPoint weights choose a representative trace; they are not
treated as replicate counts.

## Primary run

- ChampSim develop commit `e6530c7293a4d93857f634447554281a3e582516`.
- 50 million retired warm-up instructions.
- 200 million retired measured instructions.
- No L1D prefetch, official `ip_stride`, matched raw stride, and regularity
  stride use the same binary and trace.
- Primary performance statistic: geometric mean of paired IPC ratios.
- Uncertainty: deterministic 10,000-resample program bootstrap plus exact
  two-sided sign test. The simulator is deterministic, so repeated executions
  of one trace are not independent samples.

## Predictions fixed before holdout

Relative to matched raw stride, the gate is predicted to:

1. reduce aggregate accepted L1D prefetches and aggregate L2C/LLC prefetch
   misses;
2. increase accepted-prefetch accuracy while retaining most useful prefetches;
3. leave aggregate DRAM read requests practically equivalent, with a frozen
   descriptive margin of +/-1%; and
4. not improve geometric-mean IPC.

All four outcomes will be reported. Failure of any prediction is a result, not
a reason to change the gate or trace set. The +/-1% DRAM rule is a practical
equivalence descriptor, not a formal TOST, because there are only nine
program-level holdout units.

Custom callback counts are diagnostic rather than a stream-equivalence gate:
ChampSim defines the ROI at core retirement while cache callbacks for in-flight
requests can cross the phase boundary. Exact equivalence is required between
the custom gate-disabled module and official `ip_stride` on all architectural,
cache, and memory output fields. Raw/gated callback-count differences are
reported, not thresholded.

## Scope limitation

This protocol evaluates one highest-weight SimPoint per program rather than
all SimPoints and has no RTL synthesis or measured energy. It can support a
reproducible negative/methodological result about metric validity; it cannot
establish silicon energy savings or universal workload behavior.

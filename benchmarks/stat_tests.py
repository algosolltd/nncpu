"""Statistical inference over the canonical 30-seed runs (issue #3).

Because every config is run on the *same* seed schedule (run index r uses
random_state = r), performance is paired by construction: the honest test
is a Wilcoxon signed-rank test on per-run cycle differences.

Outputs:
* results/final_30/statistics.csv   -- one row per (workload, comparison)
* printed table (mean/median diff + signed-rank W, z, p)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

RESULTS_ROOT = "results"
COMPARISONS = (("nn", "baseline"), ("nn", "stride"), ("stride", "baseline"))


def paired_cycles(group, a, b):
    """Align two configurations by seed instead of trusting CSV row order."""
    left = group[group.config == a][["seed", "cycles"]].rename(columns={"cycles": "a"})
    right = group[group.config == b][["seed", "cycles"]].rename(columns={"cycles": "b"})
    paired = left.merge(right, on="seed", how="inner", validate="one_to_one").sort_values("seed")
    return paired["a"].to_numpy(), paired["b"].to_numpy()


def holm_adjust(p_values):
    """Family-wise Holm correction, preserving NaN for identical samples."""
    adjusted = [float("nan")] * len(p_values)
    valid = sorted((float(p), i) for i, p in enumerate(p_values) if not np.isnan(p))
    running = 0.0
    m = len(valid)
    for rank, (p, original) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[original] = running
    return adjusted


def wilcoxon_signed_rank(a, b):
    """Two-sided Wilcoxon signed-rank test on paired samples (normal approx).

    Returns ``(w_plus, z, p)``; ``(nan, nan, nan)`` means the paired
    differences are all zero (distributions identical).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    r = pd.Series(np.abs(d)).rank(method="average").values
    w_plus = float(r[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    counts = pd.Series(np.abs(d)).value_counts()
    tie_adj = sum(c ** 3 - c for c in counts.values if c > 1) / 48.0
    var -= tie_adj
    if var <= 0:
        return w_plus, 0.0, 1.0
    z = (w_plus - mean) / np.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w_plus, z, p


def _erf(x):
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592
    sign = 1 if x >= 0 else -1
    return sign * (1.0 - poly * np.exp(-x * x))


def _normal_cdf(x):
    return 0.5 * (1 + _erf(x / np.sqrt(2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/final_30/runs.csv")
    args = ap.parse_args()

    runs = pd.read_csv(args.runs)
    outdir = os.path.dirname(args.runs)
    rows = []

    print("\nWilcoxon signed-rank test (paired by seed) on total cycles.\n"
          "comparison                 median nn   median other   mean diff    W+        z         p (two-sided)")
    for workload, g in runs.groupby("workload", sort=False):
        for (a, b) in COMPARISONS:
            va, vb = paired_cycles(g, a, b)
            n = len(va)
            w, z, p = wilcoxon_signed_rank(va, vb)
            med_diff = np.median(va - vb)
            p_label = "identical" if np.isnan(p) else f"{p:.3g}"
            rows.append({
                "workload": workload, "comparison": f"{a}_vs_{b}",
                "n_pairs": n,
                "median_cycles_a": float(np.median(va)),
                "median_cycles_b": float(np.median(vb)),
                "median_diff_cycles": float(med_diff),
                "w_plus": w if not np.isnan(w) else None,
                "z": z if not np.isnan(z) else None,
                "p_value": p,
                "note": "identical (no paired difference)" if np.isnan(p) else "",
            })
            print(f"{workload:16s} {a+'_vs_'+b:20s} {np.median(va):9.0f} {np.median(vb):11.0f} "
                  f"{med_diff:+10.0f} {w:9.1f} {z:7.2f} {p_label:>10}")

    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_adjust(out["p_value"].to_list())
    out["significant_0.05_holm"] = out["p_value_holm"] < 0.05
    out.to_csv(os.path.join(outdir, "statistics.csv"), index=False)
    with open(os.path.join(outdir, "statistics.json"), "w") as f:
        json.dump(out.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved {len(out)} rows to {os.path.join(outdir, 'statistics.csv')}")


if __name__ == "__main__":
    main()

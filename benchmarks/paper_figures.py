"""Render the full publication figure set from the stored results.

Reads results/... (final_30, exp_phased, exp_feature_ablation,
exp_gate_aggregators, exp_machine_sweep, exp_mlp_backends, final_30 stats)
and writes results/paper_figures/*.png, ready to include in the paper.

    python benchmarks/paper_figures.py [--out results/paper_figures]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "baseline": "#94a3b8", "stride": "#2563eb", "nn": "#d97706",
    "numpy": "#0ea5e9", "sklearn": "#a855f7",
}
PALETTE = ["#2563eb", "#d97706", "#16a34a", "#0ea5e9", "#a855f7",
           "#dc2626", "#f59e0b", "#0f766e"]
CONFIGS = ["baseline", "stride", "nn"]


def _load(root, name, file_):
    p = os.path.join(root, name, file_)
    return pd.read_csv(p) if os.path.exists(p) else None


def _grouped_bars(ax, df, xcol, metric, group_key="config", err=None,
                  order=None):
    cats = order or [c for c in CONFIGS if (df[group_key] == c).any()]
    width = 0.8 / max(len(cats), 1)
    xs = list(range(len(df[xcol].unique())))
    for i, c in enumerate(cats):
        vals, errs = [], []
        for x in xs:
            d = df[(df[xcol] == df[xcol].unique()[x]) & (df[group_key] == c)]
            vals.append(float(d[metric].iloc[0]) if not d.empty else 0)
            errs.append(float(d[err].iloc[0]) if err and not d.empty else 0)
        ax.bar([x + (i - (len(cats) - 1) / 2) * width for x in xs], vals, width,
               yerr=errs or None, capsize=3, color=COLORS.get(c), label=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(df[xcol].unique(), rotation=20, ha="right")


def fig_speedup(out, root):
    s = _load(root, "final_30", "summary.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = s[s.config != "baseline"].copy()
    _grouped_bars(ax, d, "workload", "speedup_mean", err="speedup_std")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_ylabel("speedup vs no-prefetch")
    ax.set_title("Canonical 30-seed: speedup (error bars = 1 std)")
    ax.legend(ncols=2)
    fig.tight_layout(); fig.savefig(f"{out}/t_speedup.png", dpi=220); plt.close(fig)


def fig_hit_cycles(out, root):
    s = _load(root, "final_30", "summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    _grouped_bars(ax1, s, "workload", "hit_rate_mean", err="hit_rate_ci95")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("L1 hit rate")
    ax1.set_title("Hit rate (95% CI)")
    d = s.copy()
    _grouped_bars(ax2, d, "workload", "cycles_mean", err="cycles_ci95")
    ax2.set_yscale("log"); ax2.set_ylabel("total cycles (log)")
    ax2.set_title("Total cycles (95% CI)")
    for ax in (ax1, ax2): ax.legend(ncols=3, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/t_hit_cycles.png", dpi=220); plt.close(fig)


def fig_feature(out, root):
    f = _load(root, "exp_feature_ablation", "scan_mean.csv")
    if f is None or "nn_speedup" not in f.columns:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    wls = sorted(f.workload.unique())
    width = 0.8 / max(len(wls), 1)
    xs = list(range(len(f.label.unique())))
    for i, w in enumerate(wls):
        vals = [f[(f.label == lab) & (f.workload == w)].nn_speedup.iloc[0]
                for lab in f.label.unique()]
        ax.bar([x + (i - (len(wls) - 1) / 2) * width for x in xs], vals, width,
               color=PALETTE[i % len(PALETTE)], label=w)
    ax.set_xticks(xs); ax.set_xticklabels(f.label.unique(), rotation=20, ha="right")
    ax.set_ylabel("NN speedup")
    ax.set_title("Feature ablation (NN speedup by encoding)")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{out}/t_feature.png", dpi=220); plt.close(fig)


def fig_gate(out, root):
    f = _load(root, "exp_gate_aggregators", "scan_mean.csv")
    if f is None:
        return
    if "config" in f.columns:  # long form: label, workload, config, speedup
        nn = f[f.config == "nn"]
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for k, lab in enumerate(sorted(nn.label.unique())):
            d = nn[nn.label == lab]
            xs = list(range(len(d)))
            ax.plot(xs, d.speedup, marker="o", linewidth=2,
                    color=PALETTE[k % len(PALETTE)], label=lab)
            ax.set_xticks(xs); ax.set_xticklabels(d.workload, rotation=20, ha="right")
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.set_ylabel("NN speedup")
        ax.set_title("Gate design: mean/median x delta-cap (ML traces)")
        ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(f"{out}/t_gate.png", dpi=220); plt.close(fig)


def fig_machine(out, root):
    f = _load(root, "exp_machine_sweep", "scan_mean.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    panels = [("cache capacity", ["l1_16", "l1_64"], "stride_speedup"),
              ("DRAM latency", ["l1_32_lat20", "l1_32_lat80"], "stride_speedup"),
              ("line size", ["line_4", "line_16"], "stride_speedup")]
    for ax, (title, labs, m) in zip(axes, panels):
        rows = f[f.label.isin(labs)]
        width = 0.35
        xs = range(len(rows.workload.unique()))
        for i, lab in enumerate(labs):
            for cc, off, c in (("nn", 1, COLORS["nn"]), ("stride", 0, COLORS["stride"])):
                vals = [rows[(rows.label == lab) & (rows.workload == w)][f"{cc}_speedup"].iloc[0]
                        for w in rows.workload.unique()]
                ax.bar([x + (i - 0.5) * width + off * width / 2.4 for x in xs], vals,
                       width / 2.2, color=c, label=f"{cc} {lab}")
        ax.set_xticks(list(xs)); ax.set_xticklabels(rows.workload.unique(),
                                                    rotation=25, ha="right", fontsize=7)
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.set_title(title)
        ax.legend(fontsize=6, ncols=2)
    fig.tight_layout(); fig.savefig(f"{out}/t_machine.png", dpi=220); plt.close(fig)


def fig_backend(out, root):
    f = _load(root, "exp_mlp_backends", "scan_rows.csv")
    if f is None:
        return
    d = f[f.config == "nn"].groupby(["label", "workload"]).wall_ms.mean().reset_index()
    figs2 = {"numpy": "#0ea5e9", "sklearn": "#a855f7"}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    width = 0.35
    xs = list(range(len(d.workload.unique())))
    for i, lab in enumerate(d.label.unique()):
        vals = [d[(d.label == lab) & (d.workload == w)].wall_ms.iloc[0]
                for w in d.workload.unique()]
        ax.bar([x + (i - 0.5) * width for x in xs], vals, width,
               color=figs2.get(lab, COLORS.get(lab, "#16a34a")), label=lab)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(d.workload.unique(), rotation=20, ha="right")
    ax.set_ylabel("wall time (ms, log)")
    ax.set_title("MLP backend cost: numpy vs sklearn")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/t_backend.png", dpi=220); plt.close(fig)


def fig_stats(out, root):
    f = _load(root, "final_30", "statistics.csv")
    if f is None:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    df = f[f.comparison.isin(["nn_vs_baseline", "stride_vs_baseline",
                              "nn_vs_stride"])].copy()
    df["diff_cyc"] = df["median_diff_cycles"]
    width = 0.22
    xs = list(range(len(df.workload.unique())))
    pal = {"nn_vs_baseline": "#d97706", "stride_vs_baseline": "#2563eb",
           "nn_vs_stride": "#16a34a"}
    for i, cmp in enumerate(df.comparison.unique()):
        vals = [df[(df.comparison == cmp) & (df.workload == w)].diff_cyc.iloc[0]
                for w in df.workload.unique()]
        ax.bar([x + (i - 1) * width for x in xs], vals, width,
               color=pal.get(cmp), label=cmp.replace("_", " vs "))
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(df.workload.unique(), rotation=20, ha="right")
    ax.set_ylabel("median cycles difference (paired)")
    ax.set_title("Wilcoxon signed-rank on 30 seeds (positive = worse)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/t_stats.png", dpi=220); plt.close(fig)


def fig_pipeline(out, root):
    """Corrected architecture: every access feeds BOTH the online
    learning loop AND the decide path; the gate is fed by the error
    history; prefetch = free L1 fill."""
    del root
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p1, p2, text="", tpos=None, lcol="#1e3a5f"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                     mutation_scale=18, lw=2.0, color=lcol))
        if text:
            tx, ty = tpos or ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.22)
            ax.text(tx, ty, text, ha="center", fontsize=8.5, color=lcol)

    # ---- decision path (top) ----
    access = (0.2, 6.7, 1.9, 0.95)
    feat = (2.7, 6.7, 1.7, 0.95)
    mlp = (4.9, 6.7, 1.7, 0.95)
    gate = (7.1, 6.7, 1.6, 0.95)
    pref = (9.3, 6.7, 1.9, 0.95)
    noop = (9.3, 4.6, 1.9, 0.9)

    box(*access, "memory\naccess a_t", "#f1f5f9", "#475569")
    box(*feat, "features x_t\n(6 dims)", "#dbeafe", "#2563eb")
    box(*mlp, "MLP\nDX = f_theta(x_t)", "#fef3c7", "#b45309", 10)
    box(*gate, "gate:\nconfident?", "#fce7f3", "#a21caf", 10)
    box(*pref, "prefetch\na_t + DX", "#dcfce7", "#15803d", 10)
    box(*noop, "no-op\n(do nothing)", "#f1f5f9", "#475569")

    arrow((access[0] + access[2], 7.17), (feat[0], 7.17))
    arrow((feat[0] + feat[2], 7.17), (mlp[0], 7.17))
    arrow((mlp[0] + mlp[2], 7.17), (gate[0], 7.17))
    arrow((gate[0] + gate[2], 7.17), (pref[0], 7.17), "YES", (8.9, 7.55))
    arrow((gate[0] + 0.2, gate[1]), (noop[0] + noop[2] - 0.2, noop[1] + noop[3]),
          "if not confident", (9.2, 5.9), )

    # ---- learning loop (bottom, always on) ----
    replay = (1.0, 3.4, 2.4, 0.95)
    refit = (4.4, 3.4, 2.2, 0.95)
    errhist = (7.1, 3.4, 1.7, 1.05)
    box(*replay, "replay (<=64)\nclamp d_t to +/-16", "#fde68a", "#b45309", 9.5)
    box(*refit, "refit f_theta\nevery 16 samples", "#fde68a", "#b45309", 9.5)
    box(*errhist, "error history\nmedian |pred-actual|, W=48", "#ede9fe",
        "#6d28d9", 9.5)

    arrow((access[0] + access[2] / 2, access[1]),
          (replay[0] + replay[2] / 2, replay[1] + replay[3]),
          "every access: close sample $(x_{t-1},\\mathrm{clamp}(d_t))$",
          (2.35, 5.1))
    arrow((replay[0] + replay[2], replay[1] + replay[3] / 2),
          (refit[0], refit[1] + replay[3] / 2))
    arrow((refit[0] + refit[2] / 2, refit[1] + refit[3]),
          (mlp[0] + mlp[2] / 2, mlp[1]), "update $\\theta$", (5.9, 5.15))
    arrow((mlp[0] + mlp[2] / 2, mlp[1]),
          (errhist[0] + errhist[2] / 2, errhist[1] + errhist[3]),
          "prediction error feeds the gate", (6.6, 5.3))
    arrow((errhist[0] + errhist[2] / 2, errhist[1] + errhist[3]),
          (gate[0] + gate[2] / 2, gate[1]),
          "serr vs 0.5*median |d|", (8.15, 5.3))

    # ---- context notes ----
    ax.text(9.4, 3.4, "prefetch = free L1 line fill\n(0 cycles, idle DRAM);\n"
            "hits pay 1 cycle, misses 40",
            fontsize=8.5, va="center", color="#134e4a")
    ax.text(0.3, 7.9, "training runs on every access, gate or not", fontsize=8,
            color="#1e3a5f")

    # ---- legend ----
    from matplotlib.patches import Rectangle
    items = [("features", "#dbeafe", "#2563eb"),
             ("MLP", "#fef3c7", "#b45309"),
             ("gate", "#fce7f3", "#a21caf"),
             ("prefetch", "#dcfce7", "#15803d"),
             ("learning loop", "#fde68a", "#b45309"),
             ("error history", "#ede9fe", "#6d28d9")]
    lx = 0.7
    ly = 0.15
    ax.text(lx, ly + 0.5, "color key:", fontsize=9, va="center",
            fontweight="bold")
    for label, fc, ec in items:
        ax.add_patch(Rectangle((lx, ly), 0.42, 0.42, fc=fc, ec=ec, lw=1.2))
        ax.text(lx + 0.55, ly + 0.22, label, fontsize=9, va="center")
        lx += 1.95
        if lx > 11.8:
            lx = 0.7
            ly -= 1.05

    fig.suptitle("Gated delta predictor: decision path (top), online "
                 "learning loop (middle), color key (bottom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out}/t_pipeline.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/paper_figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in (fig_speedup, fig_hit_cycles, fig_feature, fig_gate,
               fig_machine, fig_backend, fig_stats):
        try:
            fn(args.out, args.root)
            print(f"  ok {fn.__name__}")
        except Exception as e:  # pragma: no cover
            print(f"  !! {fn.__name__}: {e}")
    print(f"\nFigures in {args.out}/")


if __name__ == "__main__":
    main()
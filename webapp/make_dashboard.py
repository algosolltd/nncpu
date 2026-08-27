"""Build a self-contained `dashboard.html` from the results on disk.

No server, no framework: the data from `results/` is inlined as JSON and
rendered by the static template (tabs + Plotly charts).  Just open the
file, or serve the repo with:  python -m http.server 8501

    python webapp/make_dashboard.py [--results dir] [--out dashboard.html]
"""

import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template.html")


def _num(x, none=None):
    try:
        return None if pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return none


def gather(results: str) -> dict:
    experiments, scans, phases, overview = {}, {}, {}, []

    if not os.path.isdir(results):
        return {"experiments": experiments, "scans": scans,
                "phases": phases, "statistics": [], "overview": []}

    for name in sorted(os.listdir(results)):
        base = os.path.join(results, name)
        if not os.path.isdir(base):
            continue
        summary_p = os.path.join(base, "summary.csv")
        scan_p = os.path.join(base, "scan_mean.csv")
        phases_p = os.path.join(base, "phases.csv")

        if os.path.exists(summary_p):
            df = pd.read_csv(summary_p)
            # Other result families may also publish a summary.csv with a
            # different schema (for example speculative-decoding throughput).
            # The cache dashboard must ignore those artifacts, not reinterpret
            # their columns as cache experiment data.
            if not {"workload", "config"}.issubset(df.columns):
                continue
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "workload": r["workload"], "config": r["config"],
                    "cycles_mean": _num(r.get("cycles_mean")),
                    "cycles_ci95": _num(r.get("cycles_ci95"), None),
                    "hit_rate_mean": _num(r.get("hit_rate_mean")),
                    "speedup_mean": _num(r.get("speedup_mean"), None),
                    "speedup_std": _num(r.get("speedup_std"), None),
                })
            experiments[name] = {
                "summary": rows, "runs": int(df["runs"].iloc[0]) if "runs" in df else None,
            }
            overview.append({"name": name, "kind": "experiment",
                             "runs": experiments[name]["runs"],
                             "workloads": ",".join(sorted(df.workload.unique()))})
        elif os.path.exists(scan_p):
            df = pd.read_csv(scan_p)
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "label": r["label"], "workload": r["workload"],
                    "stride_speedup": _num(r.get("stride_speedup")),
                    "nn_speedup": _num(r.get("nn_speedup")),
                })
            scans[name] = {"scan_mean": rows}
            overview.append({"name": name, "kind": "scan",
                             "workloads": ",".join(sorted(df.workload.unique()))})
        elif os.path.exists(phases_p):
            df = pd.read_csv(phases_p)
            rows = [{"config": r["config"], "phase": int(r["phase"]),
                     "phase_kind": r["phase_kind"],
                     "hit_rate": _num(r["hit_rate"])}
                    for _, r in df.iterrows()]
            phases[name] = {"phases": rows}
            overview.append({"name": name, "kind": "phases"})

    statistics = []
    stats_p = os.path.join(results, "final_30", "statistics.csv")
    if os.path.exists(stats_p):
        df = pd.read_csv(stats_p)
        statistics = df.to_dict(orient="records")
        for r in statistics:
            for key in ("median_diff_cycles", "p_value", "n_pairs"):
                if r.get(key) is not None and isinstance(r[key], float) and pd.isna(r[key]):
                    r[key] = None
    return {"experiments": experiments, "scans": scans, "phases": phases,
            "statistics": statistics, "overview": overview}


def build(results: str, out: str) -> str:
    data = gather(results)
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # the template contains:  const DATA = /*__DATA__*/ {};
    html = html.replace("/*__DATA__*/ {}", payload)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def main():
    ap = argparse.ArgumentParser(description="build dashboard.html from results/")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()
    out = build(args.results, args.out)
    print(f"Built {out}")
    print("Open it, or serve with:  python -m http.server 8501")


if __name__ == "__main__":
    main()

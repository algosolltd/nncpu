"""Tests for the static dashboard builder (webapp/make_dashboard.py)."""

import json
import os
import re

import pytest

from webapp.make_dashboard import build, gather


def test_build_produces_complete_html(tmp_path):
    out = tmp_path / "dashboard.html"
    build("results", str(out))
    html = out.read_text(encoding="utf-8")

    assert "Compare" in html
    assert "Plotly" in html
    assert "/*__DATA__*/" not in html  # placeholder was replaced
    assert "final_30" in html           # a known experiment is embedded


def test_gather_loads_all_kinds():
    data = gather("results")
    assert "final_30" in data["experiments"]
    assert "exp_feature_ablation" in data["scans"]
    assert "exp_phased" in data["phases"]
    assert data["statistics"]            # final_30/statistics.csv
    assert data["overview"]

    # summary keeps only the paper-relevant columns, rounded in JS
    row = data["experiments"]["final_30"]["summary"][0]
    assert set(row) == {"workload", "config", "cycles_mean", "cycles_ci95",
                        "hit_rate_mean", "speedup_mean", "speedup_std"}


def test_embedded_json_is_valid():
    if not os.path.exists("dashboard.html"):
        build("results", "dashboard.html")
    html = open("dashboard.html", encoding="utf-8").read()
    match = re.search(r"const DATA = (.*?);\n", html, re.S)
    assert match, "DATA payload not found"
    payload = match.group(1)
    assert payload.strip().endswith("}")
    data = json.loads(payload)
    assert "experiments" in data and "phases" in data and "scans" in data
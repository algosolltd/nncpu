"""Integration tests for the reproducible experiment pipeline."""

import json
import os

from nncpu.cpu import MachineConfig
from nncpu.experiment import ExperimentConfig, _stats, run_experiment, source_digest

EXPECTED_ARTIFACTS = (
    "config.json",
    "manifest.json",
    "runs.csv",
    "summary.csv",
    "speedups.csv",
    "table.tex",
    "README.txt",
)


def _tiny_cfg(tmp_path=".", **overrides) -> ExperimentConfig:
    params = dict(
        name="t",
        length=60,
        runs=3,
        seed=1,
        workloads=("sequential_read",),
        configs=("baseline", "stride"),
        machine=MachineConfig(mem_latency=40),
    )
    params.update(overrides)
    return ExperimentConfig(**params)


def test_experiment_writes_all_artifacts(tmp_path):
    cfg = _tiny_cfg()
    res = run_experiment(cfg, root=str(tmp_path))

    for artifact in EXPECTED_ARTIFACTS:
        assert os.path.exists(os.path.join(res.outdir, artifact)), artifact
    assert len(res.figures) == 3

    # runs = 3 seeds * 1 workload * 2 configs
    assert len(res.runs_df) == 6
    assert {"run", "seed", "workload", "config", "cycles", "speedup"} <= set(res.runs_df.columns)

    stride = res.summary_df[res.summary_df.config == "stride"].iloc[0]
    assert stride["speedup_mean"] > 1.0      # stride must beat baseline
    assert stride["runs"] == 3
    assert stride["cycles_ci95"] >= 0.0
    # baseline is deterministic: zero spread across seeds
    baseline = res.summary_df[res.summary_df.config == "baseline"].iloc[0]
    assert baseline["cycles_std"] == 0.0


def test_experiment_with_nn_seeds(tmp_path):
    cfg = _tiny_cfg(configs=("baseline", "nn"), runs=3)
    res = run_experiment(cfg, root=str(tmp_path))
    nn = res.summary_df[res.summary_df.config == "nn"].iloc[0]
    assert nn["runs"] == 3
    assert nn["speedup_mean"] > 1.0
    assert len(res.runs_df) == 6


def test_manifest_has_provenance(tmp_path):
    cfg = _tiny_cfg(name="prov")
    res = run_experiment(cfg, root=str(tmp_path))
    with open(os.path.join(res.outdir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["git_revision"] not in ("", "unknown")
    assert {"numpy", "pandas", "scikit_learn", "matplotlib", "seaborn"} <= set(manifest["versions"])
    assert manifest["generated_at"]
    assert manifest["source_sha256"] == source_digest()
    assert len(manifest["config_sha256"]) == 64


def test_config_json_is_round_trippable(tmp_path):
    cfg = _tiny_cfg(name="cfg", nn_kwargs={"batch_size": 8, "hidden_layers": (16,)})
    res = run_experiment(cfg, root=str(tmp_path))
    with open(os.path.join(res.outdir, "config.json")) as f:
        saved = json.load(f)
    assert saved["machine"]["mem_latency"] == 40
    assert saved["nn_kwargs"]["batch_size"] == 8
    assert saved["nn_kwargs"]["hidden_layers"] == [16]
    # Implicit behavior-changing defaults must be frozen into the artifact.
    assert {"gate_scale", "gate_aggregator", "delta_cap", "feature_mode"} <= set(
        saved["nn_kwargs"]
    )
    assert saved["runs"] == 3
    assert saved["vary_workload_seed"] is True
    restored = ExperimentConfig.from_dict(saved)
    assert restored.as_dict() == cfg.as_dict()


def test_detail_flag_persists_per_instruction_cycles(tmp_path):
    cfg = _tiny_cfg(name="detail", runs=1, detail=True)
    res = run_experiment(cfg, root=str(tmp_path))
    samples = json.loads(res.runs_df.iloc[0]["inst_cycles"])
    assert len(samples) == cfg.length


def test_invalid_experiment_configuration_fails_early():
    import pytest

    for overrides in (
        {"runs": 0},
        {"length": 0},
        {"workloads": ()},
        {"workloads": ("not_a_workload",)},
        {"configs": ()},
        {"name": "../escape"},
    ):
        with pytest.raises(ValueError):
            _tiny_cfg(**overrides)


def test_cli_reloads_exact_config_and_validates_machine(tmp_path):
    from main import configure, parse_args

    cfg = _tiny_cfg(name="reload", nn_kwargs={"gate_scale": None})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.as_dict()), encoding="utf-8")
    restored = configure(parse_args(["--config", str(path)]))
    assert restored.as_dict() == cfg.as_dict()

    import pytest
    with pytest.raises(ValueError):
        configure(parse_args(["--line", "0"]))


def test_random_workload_varies_across_runs_but_is_paired_by_config(tmp_path):
    cfg = _tiny_cfg(
        name="randomized",
        runs=3,
        workloads=("random_read",),
        configs=("baseline", "stride"),
    )
    res = run_experiment(cfg, root=str(tmp_path))
    baseline = res.runs_df[res.runs_df.config == "baseline"].sort_values("seed")
    assert baseline.cycles.nunique() > 1
    assert list(baseline.seed) == [1, 2, 3]
    assert baseline.speedup.tolist() == [1.0, 1.0, 1.0]


def test_latex_table_contains_rows(tmp_path):
    cfg = _tiny_cfg()
    res = run_experiment(cfg, root=str(tmp_path))
    with open(os.path.join(res.outdir, "table.tex")) as f:
        tex = f.read()
    assert "\\begin{tabular}" in tex
    assert "\\midrule" in tex
    assert "sequential_read" in tex
    assert "speedup" in tex.lower() or "Speedup" in tex


def test_stats_helper():
    import pandas as pd
    s = _stats(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["std"] > 0
    assert s["ci95"] > 0
    s1 = _stats(pd.Series([42.0]))
    assert s1["std"] == 0.0 and s1["ci95"] == 0.0

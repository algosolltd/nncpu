import numpy as np

from nncpu.cpu import MachineConfig
from nncpu.research import ResearchProfile, run_gate_study


def test_gate_study_is_paired_and_reports_matched_controls():
    workloads = {
        3: {"tiny": [{"type": "LOAD", "address": i} for i in range(40)]},
        4: {"tiny": [{"type": "LOAD", "address": i} for i in range(40)]},
    }
    profile = ResearchProfile("test", MachineConfig(), (1, 2))
    runs, summary, contrasts = run_gate_study(workloads, profiles=(profile,))

    assert len(runs[runs.config == "baseline"]) == 2
    assert np.allclose(runs[runs.config == "baseline"].speedup, 1.0)
    assert set(contrasts.candidate) == {"gated_stride", "gated_berti", "nn"}
    assert set(summary.distance) == {1, 2}
    assert {"wins", "losses", "ties", "p_value_holm"} <= set(contrasts.columns)


def test_research_profile_rejects_invalid_distance():
    try:
        ResearchProfile("bad", MachineConfig(), (0,))
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid distance accepted")

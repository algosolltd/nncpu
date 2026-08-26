import numpy as np
import pandas as pd

from benchmarks.stat_tests import holm_adjust, paired_cycles


def test_pairing_uses_seed_not_csv_row_order():
    frame = pd.DataFrame([
        {"seed": 2, "config": "nn", "cycles": 20},
        {"seed": 1, "config": "baseline", "cycles": 11},
        {"seed": 1, "config": "nn", "cycles": 10},
        {"seed": 2, "config": "baseline", "cycles": 21},
    ])
    a, b = paired_cycles(frame, "nn", "baseline")
    assert a.tolist() == [10, 20]
    assert b.tolist() == [11, 21]


def test_holm_adjustment_controls_the_family_and_keeps_identical_nan():
    adjusted = holm_adjust([0.01, 0.03, 0.2, float("nan")])
    assert np.allclose(adjusted[:3], [0.03, 0.06, 0.2])
    assert np.isnan(adjusted[3])

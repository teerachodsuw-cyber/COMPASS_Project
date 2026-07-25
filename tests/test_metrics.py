"""Tests for compass.evaluation.metrics: hand-computed sanity checks for time_dependent_
cindex and integrated_brier_score_metric, plus a structural check for ipcw_calibration_curve.
"""

import numpy as np

from compass.evaluation.metrics import (
    integrated_brier_score_metric,
    ipcw_calibration_curve,
    time_dependent_cindex,
)


def test_time_dependent_cindex_perfect_concordance():
    # Higher predicted risk for patients whose event happens earlier -- perfect ranking.
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 20, 20, 20, 20, 20], dtype=float)
    cif_at_horizon = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

    cindex = time_dependent_cindex(cif_at_horizon, event_indicator, time_years, horizon=10)
    assert cindex > 0.95


def test_time_dependent_cindex_scrambled_near_random():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 20, 20, 20, 20, 20], dtype=float)
    # Deliberately scrambled: predicted risk has no relationship to actual event timing.
    cif_scrambled = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])

    cindex = time_dependent_cindex(cif_scrambled, event_indicator, time_years, horizon=10)
    assert 0.3 < cindex < 0.7  # near 0.5, not necessarily exact given n=10


def test_integrated_brier_score_well_calibrated_near_zero():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 20, 20, 20, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    # Perfect knowledge: event patients' CIF is ~1 from their event time onward, censored
    # patients' CIF stays ~0 throughout the evaluated window.
    well_calibrated = np.zeros((10, 10))
    well_calibrated[:5, :] = 0.999
    well_calibrated[5:, :] = 0.001

    ibs = integrated_brier_score_metric(well_calibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)
    assert ibs < 0.01


def test_integrated_brier_score_miscalibrated_scores_worse():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 20, 20, 20, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    well_calibrated = np.zeros((10, 10))
    well_calibrated[:5, :] = 0.999
    well_calibrated[5:, :] = 0.001
    ibs_well = integrated_brier_score_metric(well_calibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)

    # Badly miscalibrated: predicts near-certain event for everyone, including the 5
    # patients who never had one.
    miscalibrated = np.full((10, 10), 0.99)
    ibs_bad = integrated_brier_score_metric(miscalibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)

    assert ibs_bad > ibs_well
    assert ibs_bad > 0.3


def test_ipcw_calibration_curve_structure():
    rng = np.random.default_rng(0)
    n = 100
    cif_at_horizon = rng.uniform(0, 1, n)
    event_indicator = rng.integers(0, 2, n)
    time_years = rng.uniform(1, 20, n)

    curve = ipcw_calibration_curve(cif_at_horizon, event_indicator, time_years, horizon=10, n_bins=10)

    assert set(curve.keys()) == {"predicted", "observed", "bin_counts", "n_total"}
    assert len(curve["predicted"]) == len(curve["observed"]) == len(curve["bin_counts"])
    assert curve["bin_counts"].sum() == n
    assert curve["n_total"] == n
    assert (curve["observed"] >= 0).all() and (curve["observed"] <= 1).all()

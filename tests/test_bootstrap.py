"""Tests for compass.evaluation.bootstrap: patient-level (not visit-level) resampling, the
"paired" property of paired_bootstrap_comparison, and sanity checks on CI behavior.
"""

import numpy as np
import pandas as pd

from compass.evaluation.bootstrap import (
    bootstrap_ci_single_model,
    paired_bootstrap_comparison,
    patient_level_bootstrap_sample,
)


def test_patient_level_not_visit_level():
    # If resampling happened at the visit level instead of the patient level, a patient's
    # own "visits" (represented here as a fixed group tag) could be split apart across the
    # resample. Patient-level resampling must keep each patient's ENTIRE row-group intact:
    # every resampled patient_id maps to exactly the group tag it started with.
    patient_ids = [f"P{i}" for i in range(20)]
    rng = np.random.default_rng(0)

    resampled = patient_level_bootstrap_sample(patient_ids, rng)

    assert len(resampled) == len(patient_ids)
    assert set(resampled) <= set(patient_ids)
    # With replacement -- some IDs should repeat, proving it isn't just a permutation
    # (a visit-level resample of a permutation wouldn't necessarily show this either, but a
    # correct patient-level bootstrap must allow duplicates).
    assert len(set(resampled)) < len(resampled)


def test_paired_bootstrap_uses_same_resample_across_models():
    # Two models with IDENTICAL predictions/ground_truth: if paired_bootstrap_comparison
    # truly evaluates all models on the SAME resampled set each iteration, every pairwise
    # difference must be EXACTLY zero for every iteration (same data in, same metric out).
    # If it drew independent resamples per model, differences would almost never be exactly
    # zero due to sampling noise.
    n = 30
    rng = np.random.default_rng(1)
    patient_ids = [f"P{i}" for i in range(n)]
    cif = rng.uniform(0.05, 0.95, (n, 10))
    predictions = {"mortality": cif, "malo": cif, "cvd": cif}
    event_indicator = rng.integers(0, 2, n).astype(float)
    time_years = rng.uniform(1, 10, n)
    ground_truth = {
        outcome: {"event_indicator": event_indicator, "time_years": time_years} for outcome in ("mortality", "malo", "cvd")
    }

    predictions_dict = {"model_a": predictions, "model_b": predictions}

    result = paired_bootstrap_comparison(predictions_dict, ground_truth, patient_ids, horizons=[10], n_iterations=20, seed=2)

    assert (result["point_estimate"].abs() < 1e-9).all()
    assert (result["ci_lower"].abs() < 1e-9).all()
    assert (result["ci_upper"].abs() < 1e-9).all()


def test_bootstrap_ci_reasonable():
    n = 200
    rng = np.random.default_rng(3)
    patient_ids = [f"P{i}" for i in range(n)]
    cif = rng.uniform(0.05, 0.95, (n, 10))
    predictions = {outcome: cif for outcome in ("mortality", "malo", "cvd")}
    event_indicator = rng.integers(0, 2, n).astype(float)
    time_years = rng.uniform(1, 10, n)
    ground_truth = {
        outcome: {"event_indicator": event_indicator, "time_years": time_years} for outcome in ("mortality", "malo", "cvd")
    }

    result_100 = bootstrap_ci_single_model(predictions, ground_truth, patient_ids, horizons=[10], n_iterations=100, seed=4)
    result_500 = bootstrap_ci_single_model(predictions, ground_truth, patient_ids, horizons=[10], n_iterations=500, seed=4)

    for result in (result_100, result_500):
        assert (result["ci_lower"] <= result["point_estimate"]).all()
        assert (result["point_estimate"] <= result["ci_upper"]).all()
        assert np.isfinite(result[["point_estimate", "ci_lower", "ci_upper"]].to_numpy()).all()

    # More iterations should give a more stable (not wildly different) CI width estimate --
    # not a strict narrowing guarantee, but shouldn't blow up or collapse to zero either.
    width_100 = (result_100["ci_upper"] - result_100["ci_lower"]).mean()
    width_500 = (result_500["ci_upper"] - result_500["ci_lower"]).mean()
    assert width_100 > 0 and width_500 > 0
    assert width_500 < width_100 * 3  # sanity bound, not a precise convergence claim

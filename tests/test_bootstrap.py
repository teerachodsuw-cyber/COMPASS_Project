"""Tests for compass.evaluation.bootstrap: patient-level resampling, the "paired" property
(one resample shared by all models), point estimate = full-data value, and n_valid."""

import numpy as np
import pandas as pd

from compass.evaluation.bootstrap import (
    bootstrap_ci_single_model,
    paired_bootstrap,
    paired_bootstrap_comparison,
    patient_level_bootstrap_sample,
)
from compass.evaluation.metrics import compute_all_metrics

OUTCOMES = ("mortality", "malo", "cvd")


def _synthetic(n, seed, event_rate=0.5):
    """Predictions + ground truth in the pipeline format, plus an IPCW reference sample."""
    rng = np.random.default_rng(seed)
    cif = np.sort(rng.uniform(0.01, 0.95, (n, 10)), axis=1)
    predictions = {o: cif for o in OUTCOMES}
    time_years = rng.uniform(0.5, 12, n)
    event = rng.random(n) < event_rate
    ground_truth = {}
    for o in OUTCOMES:
        competing = (~event) & (rng.random(n) < 0.2) if o != "mortality" else np.zeros(n, bool)
        ground_truth[o] = {
            "time_years": time_years,
            "event_type": np.where(event, 1, np.where(competing, 2, 0)),
            "at_risk": np.ones(n, bool),
        }
    ref_t = rng.uniform(0.5, 13, 3 * n)
    reference = {o: {"time_years": ref_t, "event_type": (rng.random(3 * n) < 0.4).astype(int)} for o in OUTCOMES}
    return predictions, ground_truth, reference


def test_patient_level_not_visit_level():
    patient_ids = [f"P{i}" for i in range(20)]
    resampled = patient_level_bootstrap_sample(patient_ids, np.random.default_rng(0))
    assert len(resampled) == len(patient_ids)
    assert set(resampled) <= set(patient_ids)
    assert len(set(resampled)) < len(resampled)  # with replacement


def test_paired_bootstrap_uses_same_resample_across_models():
    # identical predictions: every paired difference must be exactly zero in every draw
    predictions, ground_truth, reference = _synthetic(40, 1)
    result = paired_bootstrap_comparison({"model_a": predictions, "model_b": predictions}, ground_truth, reference,
                                         horizon=10, n_iterations=20, seed=2)
    assert (result["point_estimate"].abs() < 1e-12).all()
    assert (result["ci_lower"].abs() < 1e-12).all() and (result["ci_upper"].abs() < 1e-12).all()


def test_point_estimate_is_full_data_value_and_n_valid_reported():
    predictions, ground_truth, reference = _synthetic(120, 3)
    other = {o: np.clip(p + np.random.default_rng(9).normal(0, 0.1, p.shape), 0, 1) for o, p in predictions.items()}
    per_model, pairwise = paired_bootstrap({"a": predictions, "b": other}, ground_truth, reference, horizon=10,
                                           n_iterations=50, seed=4)
    full_a = compute_all_metrics(predictions, ground_truth, reference, 10).set_index(["outcome", "metric_name"])["value"]
    full_b = compute_all_metrics(other, ground_truth, reference, 10).set_index(["outcome", "metric_name"])["value"]
    for _, row in per_model.iterrows():
        full = (full_a if row["model"] == "a" else full_b)[(row["outcome"], row["metric_name"])]
        assert row["point_estimate"] == full  # exactly the full-data value, not a bootstrap mean
        assert row["n_draws"] == 50 and 0 < row["n_valid"] <= 50
    for _, row in pairwise.iterrows():
        key = (row["outcome"], row["metric_name"])
        assert np.isclose(row["point_estimate"], full_b[key] - full_a[key])
        assert "n_valid" in row


def test_zero_event_draws_skipped_and_counted():
    # rare outcome (3 events / 60): some resamples contain no event -> NaN, skipped, n_valid < n_draws
    predictions, ground_truth, reference = _synthetic(60, 5, event_rate=0.0)
    for o in OUTCOMES:
        et = ground_truth[o]["event_type"].copy()
        et[:3] = 1
        ground_truth[o]["event_type"] = et
        ground_truth[o]["time_years"] = np.where(np.arange(60) < 3, 2.0, ground_truth[o]["time_years"])
    result = bootstrap_ci_single_model(predictions, ground_truth, reference, horizon=10, n_iterations=200, seed=6)
    assert (result["n_valid"] < 200).all()
    assert (result["n_valid"] > 0).all()
    assert np.isfinite(result["point_estimate"]).all()


def test_bootstrap_ci_reasonable():
    predictions, ground_truth, reference = _synthetic(200, 3)
    result_100 = bootstrap_ci_single_model(predictions, ground_truth, reference, horizon=10, n_iterations=100, seed=4)
    result_500 = bootstrap_ci_single_model(predictions, ground_truth, reference, horizon=10, n_iterations=500, seed=4)
    for result in (result_100, result_500):
        assert np.isfinite(result[["point_estimate", "ci_lower", "ci_upper"]].to_numpy()).all()
        assert (result["ci_lower"] <= result["ci_upper"]).all()
    width_100 = (result_100["ci_upper"] - result_100["ci_lower"]).mean()
    width_500 = (result_500["ci_upper"] - result_500["ci_lower"]).mean()
    assert width_100 > 0 and width_500 > 0
    assert width_500 < width_100 * 3

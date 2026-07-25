"""Tests for compass.visualization.discrimination_calibration."""

import matplotlib

matplotlib.use("Agg")  # headless test environment, no display backend needed

import numpy as np
import pandas as pd

from compass.evaluation.bootstrap import bootstrap_ci_single_model
from compass.visualization.discrimination_calibration import plot_discrimination_calibration_scatter


def test_plot_discrimination_calibration_scatter_runs_and_shapes_correctly():
    n = 50
    rng = np.random.default_rng(0)
    patient_ids = [f"P{i}" for i in range(n)]
    outcomes = ("mortality", "malo", "cvd")

    event_indicator = rng.integers(0, 2, n).astype(float)
    time_years = rng.uniform(1, 10, n)
    ground_truth = {outcome: {"event_indicator": event_indicator, "time_years": time_years} for outcome in outcomes}

    bootstrap_results = {}
    for model_name in ("dynamic_deephit", "tracer", "survlatent_ode"):
        cif = rng.uniform(0.05, 0.95, (n, 10))
        predictions = {outcome: cif for outcome in outcomes}
        bootstrap_results[model_name] = bootstrap_ci_single_model(
            predictions, ground_truth, patient_ids, horizons=[10], n_iterations=20, seed=1
        )

    fig = plot_discrimination_calibration_scatter(bootstrap_results, outcomes=list(outcomes), horizon=10)

    assert fig is not None
    assert len(fig.axes) == len(outcomes)

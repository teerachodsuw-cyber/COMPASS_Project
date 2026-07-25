"""Evaluation metrics for COMPASS candidate models: time-dependent (Uno's) C-index and
integrated Brier score per outcome per horizon, via scikit-survival -- hand-implementing
IPCW-adjusted metrics is exactly the kind of thing that's easy to get subtly wrong, so this
wraps sksurv rather than reimplementing, per the locked methodology's choice of Uno's C
(horizon-specific, IPCW-adjusted) over a whole-follow-up Harrell's C. Also implements the
IPCW-weighted calibration curve (via lifelines' Kaplan-Meier estimator, already a project
dependency), locked as part of the methodology as a supplementary diagnostic separate from
the discrimination-calibration scatter.

Units: all times (event_or_censoring_time, horizon, time_grid) are in YEARS from landmark,
matching the project's horizons convention (configs/training.yaml: 10/20/30yr) -- distinct
from compass.training.batching's internal days-based convention; ground truth is converted
at the boundary when built from a collate_examples-style batch (see compute_all_metrics).

Prediction interface: predictions[outcome] is a (n_patients, num_time_bins) array of
predicted cumulative incidence at YEARLY bins 1..num_time_bins (bin i, 0-indexed, is CIF by
year i+1) -- i.e. exactly what CompassSurvivalModel.predict_risk(..., horizons=range(1,
num_time_bins+1)) produces for one outcome, already using base.py's shared hazards_to_cif().
This one array serves both metrics: time_dependent_cindex uses a single sparse column (CIF
at the reporting horizon), integrated_brier_score_metric uses a dense slice (CIF at every
year up to the horizon) -- no separate prediction pass needed for the two metrics.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

from compass.models.base import OUTCOME_NAMES


def _structured_survival_array(event_indicator: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.array(list(zip(event_indicator.astype(bool), time.astype(float))), dtype=[("event", bool), ("time", float)])


def time_dependent_cindex(
    cif_at_horizon: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
) -> float:
    """Uno's C-index (concordance_index_ipcw) for one outcome at one horizon.

    cif_at_horizon: (n,) each patient's predicted cumulative incidence by `horizon` --
    higher = predicted higher risk, matching what concordance_index_ipcw's `estimate` param
    expects.

    No separate held-out split is threaded through this function for estimating the
    censoring distribution, so the same evaluation set's own censoring distribution is used
    as both `survival_train` and `survival_test` -- a common practical simplification when a
    dedicated censoring-distribution reference isn't available; it does mean the IPCW
    weights aren't validated against an independent estimate.

    Returns NaN if the sample has zero events for this outcome. This is EXPECTED, not just
    a defensive guard: with a rare outcome (e.g. malo, ~1% base rate) and patient-level
    bootstrap resampling (compass.evaluation.bootstrap), a resample with zero events is a
    real, foreseeable draw, not an edge case that "can't happen" -- sksurv itself raises
    ValueError("all samples are censored") in this situation. Returning NaN here lets
    bootstrap_ci_single_model / paired_bootstrap_comparison skip that one iteration's value
    (pandas quantile/mean skip NaN by default) instead of crashing the entire run over a
    single degenerate resample.
    """
    if event_indicator.astype(bool).sum() == 0:
        return float("nan")
    # TODO before real-data use: currently reusing the same evaluation set as both
    # survival_train (censoring-distribution reference) and survival_test (actual
    # evaluation cohort) for sksurv's IPCW methods. sksurv's convention expects these to
    # ideally be separate samples -- reusing one set is a known source of mild optimistic
    # bias, worse with small/rare-event samples (directly relevant given MALO's low
    # incidence). Fine for synthetic proof-of-concept; must be revisited once real Stage 1
    # data is available, likely using the full training split as survival_train regardless
    # of which split is being evaluated.
    y = _structured_survival_array(event_indicator, event_or_censoring_time)
    tau = min(float(horizon), float(event_or_censoring_time.max()))  # sksurv requires tau within the observed range
    result = concordance_index_ipcw(y, y, cif_at_horizon, tau=tau)
    return float(result[0])


def integrated_brier_score_metric(
    cif_over_time: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
    time_grid: np.ndarray,
) -> float:
    """IBS up to `horizon`, via sksurv.metrics.integrated_brier_score.

    cif_over_time: (n, len(time_grid)) predicted CIF at each point in time_grid (years).
    Converted to survival probabilities (1 - CIF) since sksurv's IBS expects survival
    probabilities, not risk/CIF.

    time_grid: this project's yearly-bin convention (bin_width_years=1.0, shared by all
    three models) is used as the default resolution by compute_all_metrics -- a reasonable
    balance of integral accuracy vs. compute, and it's the finest grid all three models
    already natively support without extra interpolation.

    Returns NaN if the sample has zero events for this outcome -- same reasoning as
    time_dependent_cindex: a real, foreseeable draw under patient-level bootstrap
    resampling of a rare outcome, not an edge case that "can't happen".
    """
    if event_indicator.astype(bool).sum() == 0:
        return float("nan")
    # TODO before real-data use: currently reusing the same evaluation set as both
    # survival_train (censoring-distribution reference) and survival_test (actual
    # evaluation cohort) for sksurv's IPCW methods. sksurv's convention expects these to
    # ideally be separate samples -- reusing one set is a known source of mild optimistic
    # bias, worse with small/rare-event samples (directly relevant given MALO's low
    # incidence). Fine for synthetic proof-of-concept; must be revisited once real Stage 1
    # data is available, likely using the full training split as survival_train regardless
    # of which split is being evaluated.
    y = _structured_survival_array(event_indicator, event_or_censoring_time)
    survival_over_time = 1.0 - cif_over_time

    # sksurv requires every value in `times` to lie strictly within the observed follow-up
    # range of survival_test; clip defensively so a boundary horizon (e.g. exactly at the
    # max observed time) doesn't crash the whole evaluation run.
    max_time = float(event_or_censoring_time.max())
    min_time = float(event_or_censoring_time.min())
    safe_grid = np.clip(time_grid, min_time + 1e-6, max_time - 1e-6)
    safe_grid = np.unique(safe_grid)
    if len(safe_grid) < len(time_grid):
        # Clipping collapsed some grid points to the same value (e.g. horizon >= max_time
        # for several points) -- keep the corresponding columns via nearest-match indices.
        col_idx = np.searchsorted(time_grid, safe_grid)
    else:
        col_idx = np.arange(len(time_grid))

    return float(integrated_brier_score(y, y, survival_over_time[:, col_idx], safe_grid))


def ipcw_calibration_curve(
    cif_at_horizon: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> dict:
    """Bin patients by predicted cif_at_horizon into n_bins groups; for each non-empty bin,
    compute the IPCW-corrected true event rate (1 - Kaplan-Meier survival at `horizon` for
    that bin's patients, which naturally accounts for censoring within the bin) vs. the mean
    predicted CIF in that bin.

    Returns {"predicted": array, "observed": array, "bin_counts": array, "n_total": int}.
    """
    bin_edges = np.quantile(cif_at_horizon, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[0] -= 1e-9  # ensure the minimum value falls inside bin 0 with right-inclusive digitize
    bin_edges = np.unique(bin_edges)  # guard against degenerate/duplicate edges (heavy ties in CIF)

    bin_idx = np.digitize(cif_at_horizon, bin_edges[1:-1], right=True)

    predicted, observed, bin_counts = [], [], []
    for b in range(len(bin_edges) - 1):
        in_bin = bin_idx == b
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue
        predicted.append(float(cif_at_horizon[in_bin].mean()))

        kmf = KaplanMeierFitter()
        kmf.fit(durations=event_or_censoring_time[in_bin], event_observed=event_indicator[in_bin])
        survival_at_horizon = float(kmf.predict(horizon))
        observed.append(1.0 - survival_at_horizon)
        bin_counts.append(n_in_bin)

    return {
        "predicted": np.array(predicted),
        "observed": np.array(observed),
        "bin_counts": np.array(bin_counts),
        "n_total": len(cif_at_horizon),
    }


def compute_all_metrics(predictions: dict, ground_truth: dict, horizons: list) -> pd.DataFrame:
    """Orchestrates time_dependent_cindex + integrated_brier_score_metric across all 3
    outcomes x all horizons for ONE model's predictions.

    predictions: {"mortality": (n, T) CIF array, "malo": ..., "cvd": ...} at yearly bins
    1..T, T >= max(horizons) -- see module docstring for the exact convention.
    ground_truth: {"mortality": {"event_indicator": (n,), "time_years": (n,)}, "malo": ...,
    "cvd": ...} -- time_years is that outcome's own event time if it occurred, else its
    censoring time, using the SAME per-outcome censoring rule already implemented in
    compass.training.survival_losses / batching.py (mortality censors malo/cvd; malo/cvd
    don't censor each other), just consumed here for evaluation instead of training.

    Returns a tidy DataFrame: columns [outcome, horizon, metric_name, value].
    """
    rows = []
    for outcome in OUTCOME_NAMES:
        cif_all = predictions[outcome]
        event_indicator = ground_truth[outcome]["event_indicator"]
        time_years = ground_truth[outcome]["time_years"]

        for horizon in horizons:
            bin_idx = int(horizon) - 1
            cif_at_horizon = cif_all[:, bin_idx]

            cindex = time_dependent_cindex(cif_at_horizon, event_indicator, time_years, horizon)

            time_grid = np.arange(1, horizon + 1, dtype=float)
            cif_over_time = cif_all[:, : int(horizon)]
            ibs = integrated_brier_score_metric(cif_over_time, event_indicator, time_years, horizon, time_grid)

            rows.append({"outcome": outcome, "horizon": horizon, "metric_name": "cindex", "value": cindex})
            rows.append({"outcome": outcome, "horizon": horizon, "metric_name": "ibs", "value": ibs})

    return pd.DataFrame(rows)

"""Patient-level bootstrap resampling for confidence intervals on COMPASS evaluation
metrics. Resamples whole patient IDs with replacement -- never individual visit rows --
since a patient's repeated measurements aren't independent; every one of that patient's
outcome data (predictions + ground truth) moves together.
"""

import numpy as np
import pandas as pd

from compass.evaluation.metrics import compute_all_metrics


def patient_level_bootstrap_sample(patient_ids: list, rng: np.random.Generator) -> list:
    """Resample patient_ids WITH replacement, same length as input. Returns the resampled
    list of IDs (with repeats) -- caller uses this to index into that patient's full data.
    """
    n = len(patient_ids)
    idx = rng.integers(0, n, size=n)
    return [patient_ids[i] for i in idx]


def _resample_by_ids(predictions: dict, ground_truth: dict, patient_ids: list, resampled_ids: list):
    """Gather rows (with repeats) from predictions/ground_truth arrays according to
    resampled_ids, via a patient_id -> row-index lookup built from the original patient_ids
    order. predictions/ground_truth arrays are assumed aligned with patient_ids (same row
    order) on input.
    """
    id_to_index = {pid: i for i, pid in enumerate(patient_ids)}
    idx = np.array([id_to_index[pid] for pid in resampled_ids])

    resampled_predictions = {outcome: arr[idx] for outcome, arr in predictions.items()}
    resampled_ground_truth = {
        outcome: {key: values[idx] for key, values in outcome_gt.items()} for outcome, outcome_gt in ground_truth.items()
    }
    return resampled_predictions, resampled_ground_truth


def bootstrap_ci_single_model(
    predictions: dict,
    ground_truth: dict,
    patient_ids: list,
    horizons: list,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """For one model: repeat patient_level_bootstrap_sample + compute_all_metrics
    n_iterations times. Returns a DataFrame with
    [outcome, horizon, metric_name, point_estimate, ci_lower, ci_upper] (2.5/97.5
    percentiles of the bootstrap distribution; point_estimate is computed once on the full,
    unresampled data -- not the mean of the bootstrap draws).

    patient_ids: the patient ordering that predictions/ground_truth arrays are aligned to.
    Not part of the original sketch signature but structurally required for patient-level
    resampling to know which array row belongs to which patient.
    """
    rng = np.random.default_rng(seed)
    point_estimate_df = compute_all_metrics(predictions, ground_truth, horizons)

    bootstrap_draws = []
    for _ in range(n_iterations):
        resampled_ids = patient_level_bootstrap_sample(patient_ids, rng)
        rp, rgt = _resample_by_ids(predictions, ground_truth, patient_ids, resampled_ids)
        bootstrap_draws.append(compute_all_metrics(rp, rgt, horizons))

    all_draws = pd.concat(bootstrap_draws, ignore_index=True)
    ci = (
        all_draws.groupby(["outcome", "horizon", "metric_name"])["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "ci_lower", 0.975: "ci_upper"})
        .reset_index()
    )

    result = point_estimate_df.rename(columns={"value": "point_estimate"}).merge(ci, on=["outcome", "horizon", "metric_name"])
    return result


def paired_bootstrap_comparison(
    predictions_dict: dict,
    ground_truth: dict,
    patient_ids: list,
    horizons: list,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """predictions_dict: {"dynamic_deephit": predictions, "tracer": predictions,
    "survlatent_ode": predictions} (each in the same format compute_all_metrics expects).

    Each bootstrap iteration draws ONE patient resample and evaluates ALL THREE models on
    that IDENTICAL resampled set -- the defining property of "paired" (not independent
    per-model resamples), which is what makes the resulting CI a valid statement about
    whether one model actually beats another on the same patients, not just an artifact of
    each model happening to be evaluated on a different random subset.

    Returns a DataFrame with [comparison, outcome, horizon, metric_name, point_estimate,
    ci_lower, ci_upper] for every pairwise difference (later_model - earlier_model, in the
    order given by predictions_dict) of every (outcome, horizon, metric) combination.
    """
    rng = np.random.default_rng(seed)
    model_names = list(predictions_dict.keys())

    diff_draws = []
    for _ in range(n_iterations):
        resampled_ids = patient_level_bootstrap_sample(patient_ids, rng)  # SAME resample, all models, this iteration

        metrics_by_model = {}
        for model_name, predictions in predictions_dict.items():
            rp, rgt = _resample_by_ids(predictions, ground_truth, patient_ids, resampled_ids)
            metrics_by_model[model_name] = compute_all_metrics(rp, rgt, horizons).set_index(["outcome", "horizon", "metric_name"])["value"]

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                earlier, later = model_names[i], model_names[j]
                diff = (metrics_by_model[later] - metrics_by_model[earlier]).reset_index()
                diff["comparison"] = f"{later}_minus_{earlier}"
                diff_draws.append(diff)

    all_diffs = pd.concat(diff_draws, ignore_index=True)
    summary = (
        all_diffs.groupby(["comparison", "outcome", "horizon", "metric_name"])["value"]
        .agg(point_estimate="mean", ci_lower=lambda s: s.quantile(0.025), ci_upper=lambda s: s.quantile(0.975))
        .reset_index()
    )
    return summary

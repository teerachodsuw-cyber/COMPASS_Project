"""Patient-level bootstrap resampling for confidence intervals on COMPASS evaluation
metrics. Each evaluated row is ONE patient (one fixed landmark per patient), so resampling
rows with replacement resamples whole patients -- never individual visits; a patient's
predictions and ground truth move together.

paired_bootstrap() is the core: every draw uses ONE patient resample for ALL models (the
"paired" property -- differences between models are then measured on identical patients).
- point_estimate = the metric on the FULL, unresampled data (never the bootstrap mean)
- ci_lower / ci_upper = 2.5 / 97.5 percentiles of the valid draws
- n_valid = draws where the metric was defined; a draw with zero events for an outcome
  gives NaN and is skipped (a real, expected draw for a rare outcome like MALO), so CIs rest
  on n_valid draws, reported alongside
The IPCW censoring reference (training split) is fixed -- it is not resampled.
"""

import numpy as np
import pandas as pd

from compass.evaluation.metrics import compute_all_metrics

KEYS = ["outcome", "horizon", "metric_name"]


def patient_level_bootstrap_sample(patient_ids: list, rng: np.random.Generator) -> list:
    """Resample patient_ids WITH replacement, same length as input."""
    n = len(patient_ids)
    idx = rng.integers(0, n, size=n)
    return [patient_ids[i] for i in idx]


def _take(predictions: dict, ground_truth: dict, idx: np.ndarray):
    rp = {outcome: np.asarray(arr)[idx] for outcome, arr in predictions.items()}
    rgt = {outcome: {k: np.asarray(v)[idx] for k, v in gt.items()} for outcome, gt in ground_truth.items()}
    return rp, rgt


def _summarise(point: pd.Series, draws: pd.DataFrame, n_iterations: int) -> pd.DataFrame:
    """point: metric values indexed by KEYS; draws: one column per draw, same index."""
    return pd.DataFrame(
        {
            "point_estimate": point,
            "ci_lower": draws.quantile(0.025, axis=1),
            "ci_upper": draws.quantile(0.975, axis=1),
            "n_valid": draws.notna().sum(axis=1).astype(int),
            "n_draws": n_iterations,
        }
    ).reset_index()


def paired_bootstrap(
    predictions_by_model: dict,
    ground_truth: dict,
    censoring_reference: dict,
    horizon: int,
    n_iterations: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """predictions_by_model: {model_name: {outcome: (n, T) CIF}}, rows aligned with
    ground_truth (see compute_all_metrics for the ground_truth / reference format).

    Returns (per_model, pairwise):
    - per_model: [model, outcome, horizon, metric_name, point_estimate, ci_lower, ci_upper,
      n_valid, n_draws]
    - pairwise: [comparison, ...same...] for every ordered pair (later - earlier, in the
      order of predictions_by_model), from the SAME draws; n_valid counts draws where both
      models' metric was defined.
    """
    rng = np.random.default_rng(seed)
    models = list(predictions_by_model)
    n = len(next(iter(ground_truth.values()))["time_years"])

    point = {
        m: compute_all_metrics(p, ground_truth, censoring_reference, horizon).set_index(KEYS)["value"]
        for m, p in predictions_by_model.items()
    }
    draws = {m: [] for m in models}
    for _ in range(n_iterations):
        idx = np.asarray(patient_level_bootstrap_sample(list(range(n)), rng))  # ONE resample, all models
        for m in models:
            rp, rgt = _take(predictions_by_model[m], ground_truth, idx)
            draws[m].append(compute_all_metrics(rp, rgt, censoring_reference, horizon).set_index(KEYS)["value"])
    draw_frames = {m: pd.concat(d, axis=1, ignore_index=True) for m, d in draws.items()}

    per_model = pd.concat(
        [_summarise(point[m], draw_frames[m], n_iterations).assign(model=m) for m in models], ignore_index=True
    )
    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            earlier, later = models[i], models[j]
            diff_point = point[later] - point[earlier]
            diff_draws = draw_frames[later] - draw_frames[earlier]  # NaN if either side is NaN
            pairwise.append(_summarise(diff_point, diff_draws, n_iterations).assign(comparison=f"{later}_minus_{earlier}"))
    pairwise = pd.concat(pairwise, ignore_index=True) if pairwise else pd.DataFrame()

    cols = KEYS + ["point_estimate", "ci_lower", "ci_upper", "n_valid", "n_draws"]
    per_model = per_model[["model"] + cols]
    if not pairwise.empty:
        pairwise = pairwise[["comparison"] + cols]
    return per_model, pairwise


def bootstrap_ci_single_model(
    predictions: dict,
    ground_truth: dict,
    censoring_reference: dict,
    horizon: int,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """One model: [outcome, horizon, metric_name, point_estimate, ci_lower, ci_upper,
    n_valid, n_draws] (point_estimate = full-data value)."""
    per_model, _ = paired_bootstrap({"model": predictions}, ground_truth, censoring_reference, horizon, n_iterations, seed)
    return per_model.drop(columns="model")


def paired_bootstrap_comparison(
    predictions_dict: dict,
    ground_truth: dict,
    censoring_reference: dict,
    horizon: int,
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise differences only (see paired_bootstrap)."""
    _, pairwise = paired_bootstrap(predictions_dict, ground_truth, censoring_reference, horizon, n_iterations, seed)
    return pairwise

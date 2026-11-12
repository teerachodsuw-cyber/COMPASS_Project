"""Classical baseline at cohort entry (pre-specified secondary analysis 1): Cox proportional
hazards for mortality, Fine-Gray (death competing) for MALO and CVD, on each patient's first
valid landmark -- the same examples the primary evaluation uses. Fitted on the TRAINING split
only (scripts/fit_baseline.py), no tuning; evaluated as model "cox_baseline" by
scripts/evaluate.py --baseline.

Features (CoxBaselineFeatures): the deep models' input variables (build_feature_names order),
each patient's most recent OBSERVED value on/before the landmark (last visit with mask == 1 in
the landmark example -- input_visits only ever holds visits dated <= the landmark). The deep
models' inputs are z-scored with training statistics (compass.data.scaling, a linear map), so
the most recent scaled value is a linear function of the most recent raw value and the
standardised baseline features below are identical either way.
- missing (never observed up to the landmark): filled with the TRAINING-split mean of the
  observed values, plus a missing indicator for every feature with any missingness in the
  training split;
- every column then standardised with training-split mean / SD (ddof = 1);
- a column with zero variance in the training split is dropped (reported).

Models (fixed ridge penalty 0.01, pre-set, not tuned):
- mortality: lifelines CoxPHFitter(penalizer=0.01), Efron ties. lifelines' penalty is
  n * penalizer / 2 * ||beta||^2 on the summed log partial likelihood (n = rows).
- MALO / CVD: Fine-Gray via R survival::finegray() + weighted coxph() with ridge(theta,
  scale = FALSE), Efron ties, theta = penalizer * n -- the same penalty on the same scale
  (the mapping is checked against lifelines in tests/test_cox_baseline.py). At-risk patients
  only (CVD: cvd_at_risk == 1), death as the competing event.
Follow-up is used as recorded by build_targets (not truncated), as for the deep models.

Prediction: risk(t | z) = 1 - exp(-H0(t) * exp(z . beta)), with H0 the baseline cumulative
(subdistribution) hazard at z = 0 -- 1 - S(t) for mortality, the CIF for MALO / CVD. H0 is stored
only at yearly times 1..num_time_bins (aggregates; no individual event times are kept).
"""

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compass.models.base import DEATH_OUTCOME, OUTCOME_NAMES

MODEL_NAME = "cox_baseline"
R_SCRIPT = Path(__file__).with_name("fine_gray_fit.R")


# ---- features -------------------------------------------------------------------------------


def last_observed(values, mask, seq_lengths) -> np.ndarray:
    """(n, F): per patient and feature, the value at the most recent visit (within the patient's
    sequence) where the feature was observed; NaN if never observed."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask) > 0
    n, t_max, _ = values.shape
    valid = np.arange(t_max)[None, :, None] < np.asarray(seq_lengths)[:, None, None]
    observed = mask & valid
    idx = np.where(observed, np.arange(t_max)[None, :, None], -1).max(axis=1)  # (n, F) last observed visit
    out = np.take_along_axis(values, np.maximum(idx, 0)[:, None, :], axis=1)[:, 0, :]
    out[idx < 0] = np.nan
    return out


def fit_feature_transform(x_train: np.ndarray, feature_names: list[str]) -> dict:
    """Training-split statistics: observed means (fill), which features get a missing
    indicator, standardisation mean / SD, and zero-variance columns to drop."""
    x_train = np.asarray(x_train, dtype=float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> NaN, raised below
        fill = np.nanmean(x_train, axis=0)
    never = np.isnan(fill)
    if never.any():
        raise ValueError(f"feature(s) never observed in the training split: {list(np.array(feature_names)[never])}")
    indicators = [f for f, any_missing in zip(feature_names, np.isnan(x_train).any(axis=0)) if any_missing]
    transform = {"feature_names": list(feature_names), "fill_means": fill.tolist(), "indicators": indicators,
                 "columns": None, "center": None, "scale": None, "dropped_zero_variance": []}
    raw = _expand(x_train, transform)
    columns = list(feature_names) + [f"{f}_missing" for f in indicators]
    center = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=1)
    keep = scale > 0
    transform.update({
        "columns": [c for c, k in zip(columns, keep) if k],
        "center": center[keep].tolist(),
        "scale": scale[keep].tolist(),
        "dropped_zero_variance": [c for c, k in zip(columns, keep) if not k],
        "_keep": keep.tolist(),
    })
    return transform


def _expand(x: np.ndarray, transform: dict) -> np.ndarray:
    """Mean-fill with the TRAINING means and append the missing indicators (0 / 1)."""
    x = np.asarray(x, dtype=float)
    names = transform["feature_names"]
    missing = np.isnan(x)
    filled = np.where(missing, np.asarray(transform["fill_means"])[None, :], x)
    ind_idx = [names.index(f) for f in transform["indicators"]]
    return np.hstack([filled, missing[:, ind_idx].astype(float)])


def apply_feature_transform(x: np.ndarray, transform: dict) -> np.ndarray:
    """(n, len(columns)) standardised design matrix, using training statistics only."""
    raw = _expand(x, transform)[:, np.asarray(transform["_keep"], dtype=bool)]
    return (raw - np.asarray(transform["center"])) / np.asarray(transform["scale"])


# ---- fitting (training split only) -------------------------------------------------------------


def fit_cox(z: np.ndarray, time, event, penalizer: float, grid_years) -> dict:
    """Penalised Cox (lifelines, Efron ties). Returns coefficients and the baseline cumulative
    hazard at z = 0 on the yearly grid, plus convergence warnings."""
    from lifelines import CoxPHFitter

    cols = [f"z{i}" for i in range(z.shape[1])]
    df = pd.DataFrame(z, columns=cols)
    df["duration"], df["event"] = np.asarray(time, dtype=float), np.asarray(event).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=penalizer).fit(df, duration_col="duration", event_col="event")
    beta = cph.params_[cols].to_numpy()
    h_mean = cph.baseline_cumulative_hazard_.iloc[:, 0]  # lifelines: at the covariate means
    h_at = [float(h_mean[h_mean.index <= t].iloc[-1]) if (h_mean.index <= t).any() else 0.0 for t in grid_years]
    shift = float(np.exp(-np.dot(cph._norm_mean[cols].to_numpy(), beta)))  # re-reference to z = 0
    return {
        "method": "Cox proportional hazards (lifelines CoxPHFitter, Efron ties)",
        "coefficients": beta.tolist(),
        "baseline_cumhaz": [h * shift for h in h_at],
        "penalizer": penalizer,
        "n": int(len(df)),
        "n_events": int(df["event"].sum()),
        "converged": True,
        "warnings": sorted({str(w.message).splitlines()[0] for w in caught}),
    }


def fit_fine_gray(z: np.ndarray, time, event_type, penalizer: float, grid_years, ties: str = "efron",
                  rscript: str = "Rscript") -> dict:
    """Fine-Gray (death competing) in R: survival::finegray + coxph with ridge(theta = penalizer
    * n). The patient-level input exists only in a temporary directory removed on return."""
    n = len(time)
    theta = float(penalizer) * n
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = os.path.join(tmp, "train.csv"), os.path.join(tmp, "fit.json")
        df = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
        df.insert(0, "status", np.asarray(event_type).astype(int))
        df.insert(0, "time", np.asarray(time, dtype=float))
        df.to_csv(infile, index=False, float_format="%.17g")
        proc = subprocess.run([rscript, str(R_SCRIPT), infile, outfile, repr(theta), ties,
                               ",".join(repr(float(t)) for t in grid_years)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"R Fine-Gray fit failed:\n{proc.stderr[-2000:]}")
        res = json.loads(Path(outfile).read_text())
    return {
        "method": f"Fine-Gray (R survival {res['survival_version']}: finegray + coxph, {ties} ties)",
        "coefficients": np.atleast_1d(res["coefficients"]).astype(float).tolist(),
        "baseline_cumhaz": np.atleast_1d(res["baseline_cumhaz"]).astype(float).tolist(),
        "penalizer": penalizer,
        "theta": theta,
        "n": int(res["n"]),
        "n_events": int(res["n_events"]),
        "n_competing": int(res["n_competing"]),
        "iterations": res["iterations"],
        "warnings": list(np.atleast_1d(res["warnings"])) if res["warnings"] else [],
        "r_version": res["r_version"],
    }


def fit_baseline(x_train: np.ndarray, feature_names: list[str], ground_truth: dict, penalizer: float,
                 num_time_bins: int, fit_fine_gray_fn=fit_fine_gray) -> dict:
    """Fit the feature transform and the three outcome models on TRAINING data only.
    x_train: last_observed(...) of the training split's fixed examples; ground_truth: the same
    examples' build_targets ground truth ({outcome: time_years, event_type, at_risk})."""
    transform = fit_feature_transform(x_train, feature_names)
    z = apply_feature_transform(x_train, transform)
    grid = [float(t) for t in range(1, int(num_time_bins) + 1)]
    outcomes = {}
    for name in OUTCOME_NAMES:
        g = ground_truth[name]
        keep = np.asarray(g["at_risk"]).astype(bool)
        time, et = np.asarray(g["time_years"], float)[keep], np.asarray(g["event_type"]).astype(int)[keep]
        if name == DEATH_OUTCOME:
            outcomes[name] = fit_cox(z[keep], time, et == 1, penalizer, grid)
        else:
            outcomes[name] = fit_fine_gray_fn(z[keep], time, et, penalizer, grid)
    return {"model": MODEL_NAME, "transform": transform, "grid_years": grid, "outcomes": outcomes}


# ---- prediction --------------------------------------------------------------------------------


def predict_risk_from_features(z: np.ndarray, fit: dict, horizons) -> np.ndarray:
    """(n, 3, len(horizons)) predicted risk 1 - exp(-H0(h) exp(z . beta)) at each horizon (years,
    must be on the stored yearly grid)."""
    grid = list(fit["grid_years"])
    idx = [grid.index(float(h)) for h in horizons]
    out = np.empty((z.shape[0], len(OUTCOME_NAMES), len(horizons)))
    for k, name in enumerate(OUTCOME_NAMES):
        o = fit["outcomes"][name]
        lp = z @ np.asarray(o["coefficients"])
        h0 = np.asarray(o["baseline_cumhaz"])[idx]
        out[:, k, :] = 1.0 - np.exp(-h0[None, :] * np.exp(lp)[:, None])
    return out


class CoxBaselineModel:
    """Evaluation adapter: the same predict_risk interface as the deep models, computed from the
    same collated landmark batches."""

    def __init__(self, fit: dict):
        self.fit = fit

    def eval(self):
        return self

    def predict_risk(self, values, mask, seq_lengths, horizons, elapsed_days=None) -> torch.Tensor:
        x = last_observed(values.numpy(), mask.numpy(), seq_lengths.numpy())
        z = apply_feature_transform(x, self.fit["transform"])
        return torch.tensor(predict_risk_from_features(z, self.fit, horizons), dtype=torch.float32)


def save_fit(path, fit: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, **fit}, indent=2))
    return path


def load_fit(path) -> dict:
    return json.loads(Path(path).read_text())

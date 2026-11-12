"""Pre-specified secondary analysis 2: serial evaluation at prediction points k = 1, 2, 3 years
after index_date, 5-year horizon (scripts/serial_evaluate.py). Aggregates only.

Prediction point k (serial_landmark_date):
- window (index_date + (k - 1) yr, index_date + k yr], 1 yr = 365.25 days (the project's
  DAYS_PER_YEAR, as in build_targets); the landmark visit is the MOST RECENT visit in it;
- eligible only if the patient is alive at that visit (visit_date < death_date, the same
  rule as compass.data.landmarking.landmark_candidate_mask); no visit in the window, or the
  most recent one is on/after death -> not evaluated at that k;
- per-outcome at-risk flags and targets come from build_targets (prior event / zero
  follow-up -> not at risk), as everywhere else.

Inputs for the deep models (same final checkpoints, never refitted):
- FULL history: construct_landmark_example (all visits on/before the landmark visit) scaled
  and masked with the Trainer's training-fit scalers -- exactly what training builds for a
  landmark at that date;
- SNAPSHOT (ablation): ONE visit row dated at the landmark visit, holding each variable's most
  recent observed value on/before it (LOCF over the raw visit columns; missing if never
  observed), every earlier visit removed; then the same scaling / masking.

Landmark Cox / Fine-Gray (2a), fitted separately at each k on the TRAINING split's landmark
examples (compass.models.cox_baseline, unchanged): features = the snapshot's values (the most
recent observed value of each variable = last_observed of the snapshot batch) + missing
indicators, training-mean fill, training standardisation, fixed ridge 0.01; Cox (lifelines) for
mortality, Fine-Gray (R) for MALO / CVD (at-risk only; CVD only where cvd_at_risk == 1).
Follow-up is truncated at the 5-yr horizon for the fit (events / competing deaths after 5 yr
-> censored at 5 yr), the usual landmark-model convention; predictions are 5-yr risks.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from compass.data.landmarking import construct_landmark_example, death_date_of
from compass.data.preprocessing import scale_and_mask_examples
from compass.evaluation.metrics import _censoring_km_curve, _censoring_survival_at
from compass.evaluation.pipeline import _collate, _concat_gt, ground_truth_from_targets
from compass.models.base import OUTCOME_NAMES
from compass.models.cox_baseline import CoxBaselineModel, fit_baseline, last_observed
from compass.training.targets import DAYS_PER_YEAR

SERIAL_POINTS = (1, 2, 3)
SERIAL_HORIZON = 5
LANDMARK_COX = "landmark_cox"
MIN_EVENTS = 30


# ---- prediction points --------------------------------------------------------------------------


def serial_window(index_date, k: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of prediction point k's window: (index + (k-1) yr, index + k yr]."""
    index_date = pd.Timestamp(index_date)
    year = pd.Timedelta(days=DAYS_PER_YEAR)
    return index_date + (k - 1) * year, index_date + k * year


def serial_landmark_date(visit_dates, index_date, death_date, k: int):
    """The landmark visit date at prediction point k, or None if the patient is not evaluated
    there (no visit in the window, or not alive at the most recent one)."""
    dates = pd.to_datetime(pd.Series(visit_dates)).to_numpy()
    start, end = serial_window(index_date, k)
    in_window = dates[(dates > np.datetime64(start)) & (dates <= np.datetime64(end))]
    if in_window.size == 0:
        return None
    landmark = pd.Timestamp(in_window.max())
    if pd.notna(death_date) and landmark >= pd.Timestamp(death_date):
        return None
    return landmark


def raw_variable_columns(trainer) -> list[str]:
    """The raw visit columns the model inputs are built from (continuous, medication flags,
    sex, smoking status)."""
    return list(trainer.continuous_columns) + list(trainer.medication_columns) + ["sex", "smoking_status"]


def locf_snapshot(input_visits: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row dated at the last visit of `input_visits` (the landmark visit), holding each of
    `columns`' most recent observed value on/before it (NaN if never observed)."""
    visits = input_visits.sort_values("visit_date").reset_index(drop=True)
    row = visits.iloc[[-1]].copy().reset_index(drop=True)
    for col in columns:
        row[col] = visits[col].ffill().iloc[-1:].to_numpy()  # last observed value (keeps the column dtype)
    return row


def serial_raw_examples(trainer, patient_ids: list, k: int) -> list[dict]:
    """Full-history raw landmark examples at prediction point k for the eligible patients
    among `patient_ids` (in that order); ineligible patients are skipped."""
    examples = []
    for pid in patient_ids:
        visits = trainer._visits_by_patient[pid]
        outcomes = trainer._outcomes_indexed.loc[pid]
        landmark = serial_landmark_date(visits["visit_date"], outcomes["index_date"], death_date_of(outcomes), k)
        if landmark is not None:
            examples.append(construct_landmark_example(visits, outcomes, landmark))
    return examples


def snapshot_raw_examples(raw_examples: list[dict], columns: list[str]) -> list[dict]:
    return [{**ex, "input_visits": locf_snapshot(ex["input_visits"], columns)} for ex in raw_examples]


def scale_examples(trainer, raw_examples: list[dict]) -> list[dict]:
    """Scale / mask with the Trainer's training-fit scalers (never refitted)."""
    return scale_and_mask_examples(
        raw_examples, trainer.scalers, trainer.continuous_columns, trainer.medication_columns,
        trainer.sex_categories, trainer.smoking_status_categories, trainer.feature_names,
    )


def serial_inputs(trainer, patient_ids: list, k: int) -> dict:
    """{"ids", "full", "snapshot"}: scaled full-history and snapshot examples at k."""
    raw = serial_raw_examples(trainer, patient_ids, k)
    return {
        "ids": [ex["patient_id"] for ex in raw],
        "full": scale_examples(trainer, raw),
        "snapshot": scale_examples(trainer, snapshot_raw_examples(raw, raw_variable_columns(trainer))),
    }


# ---- batches, ground truth, predictions ---------------------------------------------------------


def _chunks(examples: list, size: int):
    for i in range(0, len(examples), size):
        yield examples[i : i + size]


def serial_ground_truth(trainer, examples: list[dict]) -> dict:
    """build_targets ground truth ({outcome: time_years, event_type, at_risk}) of `examples`."""
    return _concat_gt([ground_truth_from_targets(_collate(trainer, c)) for c in _chunks(examples, trainer.batch_size)])


def snapshot_features(trainer, snapshot_examples: list[dict]) -> np.ndarray:
    """(n, F) most recent observed (scaled) value of each feature, NaN if never observed."""
    parts = []
    for c in _chunks(snapshot_examples, trainer.batch_size):
        b = _collate(trainer, c)
        parts.append(last_observed(b["values"].numpy(), b["mask"].numpy(), b["seq_lengths"].numpy()))
    return np.vstack(parts)


def predict_examples(model, trainer, examples: list[dict], horizon: int, batch_size: int) -> dict:
    """{outcome: (n, horizon) CIF at years 1..horizon}. model: one model, or {outcome: model}
    (per-outcome checkpoints: outcome k's CIF from model k only)."""
    import torch

    horizons = list(range(1, int(horizon) + 1))

    def run(m) -> np.ndarray:
        m.eval()
        out = []
        with torch.no_grad():
            for c in _chunks(examples, batch_size):
                b = _collate(trainer, c)
                out.append(m.predict_risk(b["values"], b["mask"], b["seq_lengths"], horizons, b["elapsed_days"]).numpy())
        return np.concatenate(out, axis=0)

    if isinstance(model, dict):
        return {name: run(model[name])[:, k, :] for k, name in enumerate(OUTCOME_NAMES)}
    risk = run(model)
    return {name: risk[:, k, :] for k, name in enumerate(OUTCOME_NAMES)}


def truncate_ground_truth(ground_truth: dict, horizon: float) -> dict:
    """Events / competing deaths after `horizon` -> censored at `horizon`."""
    out = {}
    for name, g in ground_truth.items():
        beyond = np.asarray(g["time_years"]) > horizon
        out[name] = {
            "time_years": np.minimum(np.asarray(g["time_years"], float), float(horizon)),
            "event_type": np.where(beyond, 0, g["event_type"]).astype(int),
            "at_risk": np.asarray(g["at_risk"]).astype(bool),
        }
    return out


# ---- landmark Cox / Fine-Gray (training split only) ---------------------------------------------


def fit_landmark_cox(trainer, train_snapshot_examples: list[dict], penalizer: float, horizon: int = SERIAL_HORIZON,
                     fit_fn=fit_baseline) -> dict:
    """Fit at one prediction point from the TRAINING split's snapshot examples (follow-up
    truncated at the horizon). Returns the cox_baseline fit dict (aggregates)."""
    x = snapshot_features(trainer, train_snapshot_examples)
    gt = truncate_ground_truth(serial_ground_truth(trainer, train_snapshot_examples), horizon)
    return fit_fn(x, list(trainer.feature_names), gt, float(penalizer), int(horizon))


def predict_landmark_cox(fit: dict, trainer, snapshot_examples: list[dict], horizon: int) -> dict:
    return predict_examples(CoxBaselineModel(fit), trainer, snapshot_examples, horizon, trainer.batch_size)


# ---- reporting helpers --------------------------------------------------------------------------


def censoring_reference(ground_truth: dict) -> dict:
    """IPCW reference {outcome: time_years, event_type} from at-risk rows."""
    return {n: {"time_years": g["time_years"][g["at_risk"]], "event_type": g["event_type"][g["at_risk"]]}
            for n, g in ground_truth.items()}


def ipcw_diagnostics(reference: dict, ground_truth: dict, horizon: float) -> dict:
    """Per outcome: G(horizon) of the censoring distribution (fitted on the reference, as the
    metrics do) and the largest IPCW weight 1/G(t-) among the evaluated at-risk patients
    (times truncated at the horizon; the C-index uses this weight squared)."""
    out = {}
    for name in OUTCOME_NAMES:
        t_unique, g_knot = _censoring_km_curve(reference[name]["time_years"], reference[name]["event_type"], 1)
        g = ground_truth[name]
        times = np.minimum(np.asarray(g["time_years"])[g["at_risk"]], float(horizon))
        g_left = _censoring_survival_at(t_unique, g_knot, times, side="left") if times.size else np.array([1.0])
        out[name] = {
            "G_at_horizon": float(_censoring_survival_at(t_unique, g_knot, np.array([float(horizon)]), side="right")[0]),
            "max_ipcw_weight": float(1.0 / max(float(g_left.min()), 1e-12)),
        }
    return out


def history_lengths(full_examples: list[dict]) -> dict:
    """Median (IQR) number of visits in the full history (including the landmark visit)."""
    n = np.array([len(ex["input_visits"]) for ex in full_examples])
    if n.size == 0:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None}
    q1, med, q3 = np.percentile(n, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "min": int(n.min()), "max": int(n.max())}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_hashes(model_dirs: dict) -> dict:
    """{model: {checkpoint file: sha256}} for every checkpoint the models are evaluated from."""
    from compass.training.train import outcome_checkpoints

    return {m: {Path(p).name: file_sha256(p) for p in sorted(set(outcome_checkpoints(d).values()))}
            for m, d in model_dirs.items()}

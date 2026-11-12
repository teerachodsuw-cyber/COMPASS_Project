"""Evaluation pipeline: best-checkpoint models -> predict_risk on a split's FIXED landmarks
-> ground truth from build_targets -> metrics -> paired bootstrap. Used by
scripts/evaluate.py; everything here returns aggregates only.

- Per-outcome checkpoints (training_loop.checkpoint_selection: per_outcome): a model is given
  as {outcome: model}, and outcome k's predictions come from model k only
  (predict_selected_cifs). A single model (total selection, or older runs) predicts all three.

- Ground truth comes only from compass.training.targets.build_targets (via
  collate_examples): time, event, competing death, at_risk. Event codes for the metrics:
  1 = event, 2 = competing death (MALO / CVD), 0 = censored. at_risk == 0 rows are excluded
  per outcome (inside compute_all_metrics).
- IPCW censoring reference = the TRAINING split's survival data (its fixed, seed-sampled
  landmarks, at-risk rows only) -- never the split being evaluated.
- All models are evaluated on the same patients and landmarks (same split seed); this is
  checked, not assumed.
"""

import numpy as np
import torch

from compass.evaluation.bootstrap import paired_bootstrap
from compass.evaluation.metrics import compute_all_metrics
from compass.models.base import DEATH_OUTCOME, OUTCOME_NAMES
from compass.training.batching import collate_examples
from compass.training.targets import DAYS_PER_YEAR


def ground_truth_from_targets(batch: dict) -> dict:
    """{outcome: {"time_years", "event_type", "at_risk"}} from a collate_examples batch."""
    gt = {}
    for name in OUTCOME_NAMES:
        event = batch[f"{name}_event_indicator"].numpy().astype(bool)
        competing = batch[f"{name}_competing_indicator"].numpy().astype(bool)
        gt[name] = {
            "time_years": batch[f"{name}_target_days"].numpy().astype(float) / DAYS_PER_YEAR,
            "event_type": np.where(event, 1, np.where(competing, 2, 0)).astype(int),
            "at_risk": batch[f"{name}_at_risk"].numpy().astype(bool),
        }
    return gt


def _concat_gt(parts: list[dict]) -> dict:
    return {name: {k: np.concatenate([p[name][k] for p in parts]) for k in parts[0][name]} for name in OUTCOME_NAMES}


def _collate(trainer, examples: list[dict]) -> dict:
    return collate_examples(
        examples, trainer.outcomes_df, trainer.administrative_cutoff, trainer.bin_width_years, trainer.num_time_bins,
        last_visit_dates=trainer.last_visit_dates,
    )


def split_ground_truth(trainer, split: str) -> tuple[list, dict]:
    """(patient_ids, ground truth) for a split's fixed landmark examples."""
    examples = trainer.fixed_examples(split)
    parts = [ground_truth_from_targets(_collate(trainer, examples[i : i + trainer.batch_size]))
             for i in range(0, len(examples), trainer.batch_size)]
    return [ex["patient_id"] for ex in examples], _concat_gt(parts)


def censoring_reference_from_train(trainer) -> dict:
    """{outcome: {"time_years", "event_type"}} from the TRAINING split, at-risk rows only."""
    _, gt = split_ground_truth(trainer, "train")
    return {
        name: {"time_years": g["time_years"][g["at_risk"]], "event_type": g["event_type"][g["at_risk"]]}
        for name, g in gt.items()
    }


@torch.no_grad()
def predict_cifs(model, trainer, split: str, horizon: int) -> dict:
    """{outcome: (n, horizon) CIF at yearly bins 1..horizon} via model.predict_risk."""
    model.eval()
    examples = trainer.fixed_examples(split)
    horizons = list(range(1, int(horizon) + 1))
    chunks = []
    for i in range(0, len(examples), trainer.batch_size):
        b = _collate(trainer, examples[i : i + trainer.batch_size])
        chunks.append(model.predict_risk(b["values"], b["mask"], b["seq_lengths"], horizons, b["elapsed_days"]).numpy())
    risk = np.concatenate(chunks, axis=0)  # (n, 3, horizon)
    return {name: risk[:, k, :] for k, name in enumerate(OUTCOME_NAMES)}


def predict_selected_cifs(model, trainer, split: str, horizon: int) -> dict:
    """model: one model (predicts every outcome) or {outcome: model} (per-outcome checkpoints:
    outcome k's CIF comes from model k only, its other outputs are discarded)."""
    if isinstance(model, dict):
        return {name: predict_cifs(model[name], trainer, split, horizon)[name] for name in OUTCOME_NAMES}
    return predict_cifs(model, trainer, split, horizon)


def outcome_counts(ground_truth: dict, horizon: int) -> dict:
    out = {}
    for name, g in ground_truth.items():
        r = g["at_risk"]
        within = g["time_years"] <= horizon
        out[name] = {
            "n_at_risk": int(r.sum()),
            "n_events": int((r & within & (g["event_type"] == 1)).sum()),
            "n_competing": int((r & within & (g["event_type"] == 2)).sum()) if name != DEATH_OUTCOME else 0,
        }
    return out


def collect_predictions(trainers: dict, models: dict, split: str, horizon: int) -> tuple[list, dict, dict]:
    """(patient_ids, ground truth, {model_name: {outcome: (n, horizon) CIF}}) on a split's
    fixed landmarks, checking every model sees the same patients in the same order. Shared by
    the metrics, calibration and risk banding so all three use the same predictions."""
    names = list(models)
    patient_ids, ground_truth = split_ground_truth(trainers[names[0]], split)
    for name in names[1:]:
        other_ids, _ = split_ground_truth(trainers[name], split)
        if other_ids != patient_ids:
            raise ValueError(f"model '{name}' was evaluated on a different {split} split / patient order")
    predictions = {name: predict_selected_cifs(models[name], trainers[name], split, horizon) for name in names}
    return patient_ids, ground_truth, predictions


def evaluate_models(trainers: dict, models: dict, split: str, horizon: int, n_bootstrap: int, seed: int,
                    collected: tuple | None = None) -> dict:
    """trainers / models: {model_name: Trainer / best-checkpoint weights} -- each model entry
    is one model, or {outcome: model} for per-outcome checkpoints (predict_selected_cifs).
    collected: collect_predictions(...)'s result, if already computed. Returns a
    JSON-serialisable dict of aggregates (C-index, IBS, bootstrap CIs, pairwise differences)."""
    names = list(models)
    patient_ids, ground_truth, predictions = collected or collect_predictions(trainers, models, split, horizon)
    reference = censoring_reference_from_train(trainers[names[0]])

    per_model, pairwise = paired_bootstrap(predictions, ground_truth, reference, horizon, n_iterations=n_bootstrap, seed=seed)
    return {
        "split": split,
        "horizon_years": int(horizon),
        "n_patients": len(patient_ids),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "ipcw_reference": {"split": "train", "n_at_risk": {k: int(len(v["time_years"])) for k, v in reference.items()}},
        "counts": outcome_counts(ground_truth, horizon),
        "per_model": per_model.to_dict(orient="records"),
        "pairwise": pairwise.to_dict(orient="records"),
    }


def _fmt(row) -> str:
    if np.isnan(row["point_estimate"]):
        return f"n/a (no events) [n_valid {row['n_valid']}]"
    return f"{row['point_estimate']:.3f} ({row['ci_lower']:.3f} to {row['ci_upper']:.3f}) [n_valid {row['n_valid']}]"


def format_tables(results: dict) -> str:
    counts = results["counts"]
    lines = [
        f"Split: {results['split']} | horizon {results['horizon_years']} yr | patients {results['n_patients']} | "
        f"bootstrap {results['n_bootstrap']} draws | IPCW reference: train split",
        "",
        "| Model | Outcome | C-index (95% CI) | IBS (95% CI) | n at risk | n events |",
        "|---|---|---|---|---|---|",
    ]
    rows = {(r["model"], r["outcome"], r["metric_name"]): r for r in results["per_model"]}
    models = list(dict.fromkeys(r["model"] for r in results["per_model"]))
    for m in models:
        for o in OUTCOME_NAMES:
            lines.append(f"| {m} | {o} | {_fmt(rows[(m, o, 'cindex')])} | {_fmt(rows[(m, o, 'ibs')])} | "
                         f"{counts[o]['n_at_risk']} | {counts[o]['n_events']} |")
    if results["pairwise"]:
        lines += ["", "| Comparison | Outcome | Δ C-index (95% CI) | Δ IBS (95% CI) |", "|---|---|---|---|"]
        prow = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in results["pairwise"]}
        for c in dict.fromkeys(r["comparison"] for r in results["pairwise"]):
            for o in OUTCOME_NAMES:
                lines.append(f"| {c} | {o} | {_fmt(prow[(c, o, 'cindex')])} | {_fmt(prow[(c, o, 'ibs')])} |")
    return "\n".join(lines)

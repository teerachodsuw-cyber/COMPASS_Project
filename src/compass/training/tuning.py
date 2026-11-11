"""Hyperparameter grid tuning (learning rate x weight decay) on the VALIDATION split only.

Used by scripts/tune.py. For every model and grid config: same pinned split, same seed, same
max_epochs / early_stopping_patience / checkpoint_selection; train with early stopping into
<run_dir>/<model>/<config_id>/, then record epochs run, time, each outcome's best epoch and
best val NLL (and their sum), the best-total epoch and loss, and the val C-index per outcome
(point estimate, no bootstrap; from each outcome's own checkpoint under per-outcome
selection, exactly as scripts/evaluate.py). Each finished config writes result.json;
re-running skips finished configs.

Winner per model = the lowest SUM of the three per-outcome best val NLLs (best_val_nll_sum)
-- never the total loss, which for Dynamic-DeepHit also contains ranking / auxiliary terms.

The test split is never loaded: Trainers are built with the pinned split's test list EMPTY, so
no test patient is landmarked, scaled, predicted or scored. The full pinned split (incl. test
IDs) is kept in <run_dir>/split.json.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.splitting import check_fingerprint, data_fingerprint, load_split, patient_level_split, save_split
from compass.evaluation import pipeline
from compass.evaluation.metrics import compute_all_metrics
from compass.models.base import OUTCOME_NAMES
from compass.models.registry import MODEL_REGISTRY, build_model
from compass.training.train import Trainer, load_selected_models

RESULT_FILE = "result.json"
SUMMARY_COLUMNS = (
    ["model", "config_id", "learning_rate", "weight_decay", "checkpoint_selection", "epochs_run", "seconds"]
    + [f"best_epoch_{o}" for o in OUTCOME_NAMES]
    + [f"best_val_nll_{o}" for o in OUTCOME_NAMES]
    + ["best_val_nll_sum", "best_epoch", "best_val_total"]
    + [f"val_cindex_{o}" for o in OUTCOME_NAMES]
    + ["winner"]
)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_id(learning_rate: float, weight_decay: float) -> str:
    return f"lr{learning_rate:.3g}_wd{weight_decay:.3g}"


def expand_grid(tuning_cfg: dict, models: list[str], base_learning_rates: dict[str, float]) -> list[dict]:
    """[{model, config_id, learning_rate, weight_decay}] -- len(multipliers) x len(weight_decay)
    configs per model, in grid order."""
    grid = tuning_cfg["grid"]
    entries = []
    for model in models:
        for mult in grid["learning_rate_multipliers"]:
            for wd in grid["weight_decay"]:
                lr = float(base_learning_rates[model]) * float(mult)
                entries.append({"model": model, "config_id": config_id(lr, float(wd)), "learning_rate": lr,
                                "weight_decay": float(wd)})
    return entries


def resolve_split(run_dir, visits_df, outcomes_df, training_cfg: dict, split_from: str | None = None) -> dict:
    """The one split every config uses. Re-uses <run_dir>/split.json if present (resume), else
    copies the split saved in `split_from` (a training run dir), else derives it once with the
    training seed. The data must match the saved fingerprint."""
    run_dir = Path(run_dir)
    fingerprint = data_fingerprint(visits_df, outcomes_df)
    if (run_dir / "split.json").exists():
        split = load_split(run_dir)
    elif split_from:
        split = load_split(split_from)
    else:
        sp = training_cfg["splitting"]
        train, val, test = patient_level_split(outcomes_df, sp["train_frac"], sp["val_frac"], sp["test_frac"], training_cfg["seed"])
        split = {"train": train, "val": val, "test": test, "seed": int(training_cfg["seed"]), "fingerprint": fingerprint}
    check_fingerprint(split["fingerprint"], fingerprint)
    save_split(run_dir, split["train"], split["val"], split["test"], fingerprint, split["seed"])
    return load_split(run_dir)


def without_test(split: dict) -> dict:
    """The split handed to Trainers: train / val only, test EMPTY (never loaded)."""
    return {"train": split["train"], "val": split["val"], "test": [], "seed": split["seed"]}


def make_trainer(entry: dict, visits_df, outcomes_df, schema: dict, training_cfg: dict, split: dict,
                 model_config_overrides: dict | None = None):
    """(trainer, model) for one grid config: the model config with this config's learning_rate
    and weight_decay, built and seeded exactly the same way for every config."""
    name = entry["model"]
    model_cfg = {**load_yaml(MODEL_REGISTRY[name][1]), **(model_config_overrides or {}),
                 "learning_rate": entry["learning_rate"], "weight_decay": entry["weight_decay"]}
    continuous = [v["name"] for v in schema["continuous_variables"]]
    meds = [m["name"] for m in schema["medication_flags"]]
    smoking = next(d["values"] for d in schema["demographics"] if d["name"] == "smoking_status")
    horizon = int(training_cfg["horizons"]["primary"])
    cfg = {**training_cfg, "seed": int(split["seed"])}
    torch.manual_seed(cfg["seed"])
    model = build_model(name, model_cfg, len(build_feature_names(continuous, meds, ("male", "female"), smoking)), horizon)
    trainer = Trainer(
        model=model, model_config=model_cfg, training_config=cfg, visits_df=visits_df, outcomes_df=outcomes_df,
        continuous_columns=continuous, medication_columns=meds, sex_categories=("male", "female"),
        smoking_status_categories=smoking, horizons=[horizon],
        administrative_cutoff=pd.Timestamp(schema["censoring"]["administrative_cutoff"]),
        split=without_test(split),
    )
    return trainer, model


def val_cindex(trainer, model, horizon: int) -> dict[str, float]:
    """Val C-index per outcome at the horizon, point estimate (no bootstrap); IPCW reference =
    the training split, same pipeline as scripts/evaluate.py. model: one model, or
    {outcome: model} for per-outcome checkpoints."""
    _, gt = pipeline.split_ground_truth(trainer, "val")
    reference = pipeline.censoring_reference_from_train(trainer)
    preds = pipeline.predict_selected_cifs(model, trainer, "val", horizon)
    df = compute_all_metrics(preds, gt, reference, horizon)
    return {r.outcome: float(r.value) for r in df.itertuples() if r.metric_name == "cindex"}


def train_and_score(entry: dict, visits_df, outcomes_df, schema: dict, training_cfg: dict, split: dict,
                    run_dir, max_epochs: int, patience: int | None, model_config_overrides: dict | None = None) -> dict:
    out = Path(run_dir) / entry["model"] / entry["config_id"]
    out.mkdir(parents=True, exist_ok=True)
    stale_log = out / "training_log.jsonl"
    if stale_log.exists():  # unfinished earlier attempt (no result.json): start this config over
        stale_log.unlink()
    trainer, model = make_trainer(entry, visits_df, outcomes_df, schema, training_cfg, split, model_config_overrides)
    t0 = time.monotonic()
    history = trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=max_epochs, checkpoint_dir=str(out),
                          early_stopping_patience=patience)
    seconds = time.monotonic() - t0
    best_nll = {o: float(history["best_val_nll_per_outcome"][o]) for o in OUTCOME_NAMES}
    selected = load_selected_models(model, out)  # what evaluation would use
    result = {
        **entry,
        "checkpoint_selection": history["checkpoint_selection"],
        "epochs_run": len(history["train"]),
        "seconds": round(seconds, 1),
        **{f"best_epoch_{o}": int(history["best_epoch_per_outcome"][o]) for o in OUTCOME_NAMES},
        **{f"best_val_nll_{o}": best_nll[o] for o in OUTCOME_NAMES},
        "best_val_nll_sum": sum(best_nll.values()),
        "best_epoch": int(history["best_epoch"]),
        "best_val_total": float(history["best_val_loss"]),
        **{f"val_cindex_{o}": c for o, c in val_cindex(trainer, selected, int(training_cfg["horizons"]["primary"])).items()},
    }
    (out / RESULT_FILE).write_text(json.dumps(result, indent=2))
    return result


def finished_result(run_dir, entry: dict) -> dict | None:
    path = Path(run_dir) / entry["model"] / entry["config_id"] / RESULT_FILE
    return json.loads(path.read_text()) if path.exists() else None


def mark_winners(rows: list[dict]) -> list[dict]:
    """winner = True for each model's config with the lowest best_val_nll_sum (the sum of the
    three per-outcome best val NLLs)."""
    best = {}
    for r in rows:
        if r["model"] not in best or r["best_val_nll_sum"] < best[r["model"]]["best_val_nll_sum"]:
            best[r["model"]] = r
    return [{**r, "winner": r is best[r["model"]]} for r in rows]


def write_summary(rows: list[dict], out_dir) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| Model | Config | lr | wd | Epochs | Time (s) | Best epoch mort / MALO / CVD | Best val NLL mort / MALO / CVD | "
             "Sum | val C mort / MALO / CVD | Winner |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['config_id']} | {r['learning_rate']:.3g} | {r['weight_decay']:.3g} | {r['epochs_run']} | "
            f"{r['seconds']:.0f} | " + " / ".join(str(r[f"best_epoch_{o}"]) for o in OUTCOME_NAMES) + " | "
            + " / ".join(f"{r[f'best_val_nll_{o}']:.4f}" for o in OUTCOME_NAMES) + f" | {r['best_val_nll_sum']:.4f} | "
            + " / ".join(f"{r[f'val_cindex_{o}']:.3f}" for o in OUTCOME_NAMES) + f" | {'**WINNER**' if r['winner'] else ''} |"
        )
    table = "\n".join(lines)
    (out / "summary.md").write_text(table + "\n")
    return table


def estimate(entries: list[dict], tuning_cfg: dict, max_epochs: int, patience: int | None) -> list[dict]:
    """Per config: seconds/epoch (v1) and estimated seconds for typical (v1 epochs run), minimum
    (patience + 1 epochs) and maximum (max_epochs) run lengths."""
    ref = tuning_cfg["v1_reference"]
    out = []
    for e in entries:
        spe = float(ref["seconds_per_epoch"][e["model"]])
        typical = min(int(ref["epochs_run"][e["model"]]), max_epochs)
        low = min((patience or max_epochs) + 1, max_epochs)
        out.append({**e, "s_per_epoch": spe, "typical_s": spe * typical, "min_s": spe * low, "max_s": spe * max_epochs})
    return out


def run_grid(entries: list[dict], visits_df, outcomes_df, schema: dict, training_cfg: dict, split: dict, run_dir,
             out_dir, max_epochs: int, patience: int | None, model_config_overrides: dict | None = None,
             log=print) -> list[dict]:
    """Train + score every config (skipping finished ones), writing the summary after each."""
    rows = []
    for i, e in enumerate(entries, 1):
        done = finished_result(run_dir, e)
        if done is not None:
            log(f"[{i}/{len(entries)}] {e['model']} {e['config_id']}: finished earlier -- skipped")
            rows.append(done)
        else:
            log(f"[{i}/{len(entries)}] {e['model']} {e['config_id']}: training (max {max_epochs} epochs, patience {patience})")
            overrides = (model_config_overrides or {}).get(e["model"])
            rows.append(train_and_score(e, visits_df, outcomes_df, schema, training_cfg, split, run_dir,
                                        max_epochs, patience, overrides))
        rows = mark_winners(rows)
        write_summary(rows, out_dir)
    return rows

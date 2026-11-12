#!/usr/bin/env python
"""Evaluate trained models on a split's fixed landmarks (aggregates only): discrimination,
overall accuracy, calibration and 3-band risk banding at the 10-yr horizon.

Usage:
    python scripts/evaluate.py --checkpoint-dir runs/                 # val split, synthetic
    python scripts/evaluate.py --final-models configs/final_models.yaml --data data/processed \\
        --out-dir outputs/final_val                                   # final models, val split
    python scripts/evaluate.py --final-models configs/final_models.yaml --data data/processed \\
        --split test --final --cutpoints outputs/final_val/cutpoints.json \\
        --out-dir outputs/final_test                                  # ONE-TIME final report

Models: --checkpoint-dir <dir> (train.py's dir: <dir>/<model>/) or --final-models <yaml>
(configs/final_models.yaml: one run directory per model plus the split directory). Per-outcome
selection: outcome k's predictions come from checkpoint_best_<k>.pt; total selection (or runs
without checkpoint_selection.json): checkpoint_best.pt for all. With --final-models, every
checkpoint's stored optimiser settings must match configs/model_<model>.yaml (learning rate,
weight decay) -- the locked final settings -- or the run is refused.

Per model and outcome, in one run:
- metrics: mortality Uno C + IBS; MALO / CVD Wolbers C + Schoop competing-risk IBS (IPCW
  reference = the TRAINING split); patient-level paired bootstrap + pairwise differences.
- calibration (compass.evaluation.calibration): deciles of predicted 10-yr risk vs observed
  (Kaplan-Meier for mortality, Aalen-Johansen with death competing for MALO / CVD), overall
  O/E; a plot per outcome (calibration_<outcome>.png, 3 models per plot).
- risk banding (compass.evaluation.risk_banding): on --split val the two cut-points are FITTED
  and saved to <out-dir>/cutpoints.json (in-sample: separation and HRs are optimistic); on
  --split test they are LOADED from --cutpoints and applied unchanged -- never refitted.

Split pinning: the split saved at training (split.json: patient IDs, landmark seed, data
fingerprint) is loaded, never re-derived; the data given must match its fingerprint. On the
validation split the Trainers are given an EMPTY test list, so no test patient is landmarked,
scaled or predicted. --split test requires --final (one-time final reporting) and --cutpoints.

Classical baseline (--baseline runs/baseline/cox_baseline.json, from scripts/fit_baseline.py,
fitted on the training split only): evaluated as model "cox_baseline" on the same examples, with the
same metrics, calibration and paired bootstrap (it is listed first, so every pairwise row reads
"<deep model>_minus_cox_baseline"). Nothing is refitted here. It is never banded: risk banding
uses the deep models only. --skip-banding leaves banding out altogether (no cut-points fitted,
loaded or saved; --cutpoints then not needed on the test split).

Writes to <out-dir> (outputs/ is git-ignored): metrics.json, metrics_table.md,
calibration_table.md, banding_table.md, calibration_<outcome>.png, and cutpoints.json (val).
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.splitting import SplitMismatchError, check_fingerprint, data_fingerprint, load_split
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.calibration import calibration_all, format_calibration_tables
from compass.evaluation.pipeline import collect_predictions, evaluate_models, format_tables
from compass.evaluation.risk_banding import (
    CUTPOINTS_FILE,
    DEFAULT_MIN_BAND_FRACTION,
    DEFAULT_PERCENTILES,
    CutpointsMismatchError,
    fit_all_cutpoints,
    format_banding_tables,
    load_cutpoints,
    save_cutpoints,
    summarise_all_bands,
)
from compass.models.base import OUTCOME_NAMES
from compass.models.cox_baseline import MODEL_NAME as BASELINE_NAME
from compass.models.cox_baseline import CoxBaselineModel, load_fit
from compass.models.registry import MODEL_REGISTRY, build_model
from compass.training.train import Trainer, load_selected_models, outcome_checkpoints
from compass.visualization.calibration_plot import save_calibration_plots

log = logging.getLogger("evaluate")


class FinalModelsError(ValueError):
    """configs/final_models.yaml points at runs that do not match the locked settings / split."""


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    return x


def make_trainer(name: str, visits_df, outcomes_df, schema: dict, training_config: dict, split: dict | None = None):
    """(trainer, model) built exactly as scripts/train.py builds them -- weights untrained.
    split: a pinned split (from split.json) with its "seed"; that seed then drives landmark
    sampling and scaler fitting, so the fixed landmarks match training exactly."""
    if split is not None:
        training_config = {**training_config, "seed": int(split["seed"])}
    with open(MODEL_REGISTRY[name][1]) as f:
        model_config = yaml.safe_load(f)
    continuous = [v["name"] for v in schema["continuous_variables"]]
    medications = [m["name"] for m in schema["medication_flags"]]
    sex_categories = ("male", "female")
    smoking = next(d["values"] for d in schema["demographics"] if d["name"] == "smoking_status")
    horizon = int(training_config["horizons"]["primary"])
    model = build_model(name, model_config, len(build_feature_names(continuous, medications, sex_categories, smoking)), horizon)
    trainer = Trainer(
        model=model, model_config=model_config, training_config=training_config,
        visits_df=visits_df, outcomes_df=outcomes_df, continuous_columns=continuous, medication_columns=medications,
        sex_categories=sex_categories, smoking_status_categories=smoking, horizons=[horizon],
        administrative_cutoff=pd.Timestamp(schema["censoring"]["administrative_cutoff"]),
        split=split,
    )
    return trainer, model


def load_pinned_split(name: str, checkpoint_dir: str, visits_df, outcomes_df) -> dict:
    """The split saved at training for this model; raises SplitMismatchError if the data
    given is not the data it was trained on."""
    split = load_split(Path(checkpoint_dir) / name)
    check_fingerprint(split["fingerprint"], data_fingerprint(visits_df, outcomes_df))
    return split


def load_trained_from(name: str, model_dir, split: dict, visits_df, outcomes_df, schema: dict, training_config: dict):
    """(trainer, model) with `split` pinned and the best checkpoint weights in `model_dir`
    loaded (never the latest): one model, or {outcome: model} for per-outcome checkpoints."""
    trainer, model = make_trainer(name, visits_df, outcomes_df, schema, training_config, split=split)
    return trainer, load_selected_models(model, Path(model_dir))


def load_trained(name: str, checkpoint_dir: str, visits_df, outcomes_df, schema: dict, training_config: dict):
    """(trainer, model) for <checkpoint_dir>/<name>: its saved split re-used and its best
    checkpoint(s) loaded, as recorded in the run's checkpoint_selection.json."""
    split = load_pinned_split(name, checkpoint_dir, visits_df, outcomes_df)
    return load_trained_from(name, Path(checkpoint_dir) / name, split, visits_df, outcomes_df, schema, training_config)


def load_final_models_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "split_dir" not in cfg or not isinstance(cfg.get("models"), dict):
        raise FinalModelsError(f"{path}: expected 'split_dir' and a 'models' mapping")
    unknown = [m for m in cfg["models"] if m not in MODEL_REGISTRY]
    if unknown:
        raise FinalModelsError(f"{path}: unknown model(s) {unknown}")
    return cfg


def check_locked_settings(name: str, model_dir) -> None:
    """Every checkpoint the model is evaluated from must have been trained with the learning
    rate and weight decay in configs/model_<name>.yaml (the locked final settings)."""
    with open(MODEL_REGISTRY[name][1]) as f:
        cfg = yaml.safe_load(f)
    for outcome, path in outcome_checkpoints(model_dir).items():
        if not Path(path).exists():
            raise FinalModelsError(f"{name}: missing checkpoint {path}")
        group = torch.load(path, weights_only=False)["optimizer_state_dict"]["param_groups"][0]
        lr = group.get("initial_lr", group["lr"])
        if not (math.isclose(group["weight_decay"], float(cfg["weight_decay"]))
                and math.isclose(lr, float(cfg["learning_rate"]))):
            raise FinalModelsError(
                f"{name}: {Path(path).name} ({outcome}) was trained with lr {lr:g}, weight decay {group['weight_decay']:g}; "
                f"configs/model_{name}.yaml has lr {cfg['learning_rate']:g}, weight decay {cfg['weight_decay']:g}")


def resolve_models(args, visits_df, outcomes_df) -> tuple[list[str], dict, dict]:
    """(model names, {name: model run dir}, the pinned split). Raises SplitMismatchError /
    FinalModelsError when the runs and the data do not belong together."""
    fingerprint = data_fingerprint(visits_df, outcomes_df)
    if args.final_models:
        cfg = load_final_models_config(args.final_models)
        names = [m for m in (args.models or list(cfg["models"]))]
        missing = [m for m in names if m not in cfg["models"]]
        if missing:
            raise FinalModelsError(f"{args.final_models} lists no run directory for {missing}")
        dirs = {m: Path(cfg["models"][m]) for m in names}
        split = load_split(cfg["split_dir"])
        check_fingerprint(split["fingerprint"], fingerprint)
        for m in names:
            own = load_split(dirs[m])
            if any(own[k] != split[k] for k in ("train", "val", "seed")) or own.get("fingerprint") != split["fingerprint"]:
                raise SplitMismatchError(f"model '{m}' ({dirs[m]}) was trained on a different split than {cfg['split_dir']}")
            check_locked_settings(m, dirs[m])
        return names, dirs, split
    names = args.models or list(MODEL_REGISTRY)
    splits = {m: load_pinned_split(m, args.checkpoint_dir, visits_df, outcomes_df) for m in names}
    first = splits[names[0]]
    for m, sp in splits.items():
        if any(sp[k] != first[k] for k in ("train", "val", "test", "seed")):
            raise SplitMismatchError(f"model '{m}' was trained on a different split than '{names[0]}'")
    return names, {m: Path(args.checkpoint_dir) / m for m in names}, first


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="Required for --split test (one-time final reporting).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint-dir", help="train.py's --checkpoint-dir (holds <model>/checkpoint_best*.pt)")
    src.add_argument("--final-models", help="configs/final_models.yaml: the locked final run directory per model")
    p.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=None)
    p.add_argument("--cutpoints", default=None,
                   help="--split test: the cutpoints.json saved by the validation run (applied unchanged).")
    p.add_argument("--data", default=None, help="Processed real-data dir (scripts/build_cohort.py). Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only; checked against the saved fingerprint.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only; checked against the saved fingerprint.")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=None, help="Defaults to configs/training.yaml's seed.")
    p.add_argument("--baseline", default=None,
                   help="Fitted classical baseline (scripts/fit_baseline.py); evaluated as model 'cox_baseline'.")
    p.add_argument("--skip-banding", action="store_true", help="Leave risk banding out (no cut-points fitted or loaded).")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args(argv)

    if args.split == "test":
        if not args.final:
            print("refusing: --split test requires --final (the test split is for ONE-TIME final reporting)", file=sys.stderr)
            return 2
        if not args.cutpoints and not args.skip_banding:
            print("refusing: --split test requires --cutpoints (the validation run's cutpoints.json; "
                  "cut-points are never refitted on the test split)", file=sys.stderr)
            return 2
        log.warning("Evaluating the TEST split. The test split is for one-time final reporting only -- "
                    "do not use these numbers for model selection or tuning.")

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)
    if args.data:
        visits_df = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes_df = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        seed = args.data_seed if args.data_seed is not None else training_config["seed"]
        visits_df, outcomes_df = generate_synthetic_cohort(n_patients=args.n_patients, seed=seed)

    try:
        names, model_dirs, split = resolve_models(args, visits_df, outcomes_df)
    except (SplitMismatchError, FinalModelsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3

    baseline_fit = None
    if args.baseline:
        baseline_fit = load_fit(args.baseline)
        problem = ("it was not fitted on the training split" if baseline_fit.get("fitted_on_split") != "train"
                   else "different split seed" if baseline_fit.get("split_seed") != int(split["seed"])
                   else "different data" if baseline_fit.get("data_fingerprint") != split["fingerprint"] else None)
        if problem:
            print(f"refusing: baseline {args.baseline}: {problem}", file=sys.stderr)
            return 3

    horizon = int(training_config["horizons"]["primary"])
    meta = {
        "fitted_on_split": "val",
        "horizon_years": horizon,
        "split_seed": int(split["seed"]),
        "data_fingerprint": split["fingerprint"],
        "model_sources": {m: os.path.normpath(str(model_dirs[m])) for m in names},
        "candidate_percentiles": [min(DEFAULT_PERCENTILES), max(DEFAULT_PERCENTILES), 1],
        "min_band_fraction": DEFAULT_MIN_BAND_FRACTION,
    }
    saved_cutpoints = None
    if args.split == "test" and not args.skip_banding:  # load (and check) the validation cut-points BEFORE any test-split work
        try:
            saved_cutpoints = load_cutpoints(args.cutpoints, meta, names)
        except CutpointsMismatchError as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 4

    # validation runs: the Trainers never see the test IDs (nothing in the test split is landmarked)
    trainer_split = split if args.split == "test" else {**split, "test": []}
    trainers, models = {}, {}
    for m in names:
        trainers[m], models[m] = load_trained_from(m, model_dirs[m], trainer_split, visits_df, outcomes_df, schema,
                                                   training_config)
    if baseline_fit is not None:  # listed first; predicts from the same collated examples as the deep models
        trainers = {BASELINE_NAME: trainers[names[0]], **trainers}
        models = {BASELINE_NAME: CoxBaselineModel(baseline_fit), **models}

    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(training_config["seed"])
    collected = collect_predictions(trainers, models, args.split, horizon)
    _, ground_truth, predictions = collected
    results = evaluate_models(trainers, models, args.split, horizon, args.n_bootstrap, bootstrap_seed, collected=collected)
    results["data"] = "real (processed)" if args.data else "synthetic"
    results["model_sources"] = dict(meta["model_sources"])
    if baseline_fit is not None:
        results["model_sources"][BASELINE_NAME] = os.path.normpath(str(args.baseline))

    calibration = calibration_all(predictions, ground_truth, horizon)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["calibration"] = calibration
    tables = {
        "metrics_table.md": format_tables(results),
        "calibration_table.md": format_calibration_tables(calibration),
    }
    if args.skip_banding:
        results["banding"] = {"skipped": True}
    else:
        deep_predictions = {m: predictions[m] for m in names}  # the baseline is never banded
        if args.split == "val":
            cutpoints = fit_all_cutpoints(deep_predictions, ground_truth, horizon)
            save_cutpoints(out / CUTPOINTS_FILE, _jsonable(cutpoints), meta)
            cutpoints_source = str(out / CUTPOINTS_FILE)
        else:
            cutpoints = saved_cutpoints
            cutpoints_source = args.cutpoints
        bands = summarise_all_bands(deep_predictions, ground_truth, cutpoints, horizon)
        results["banding"] = {"cutpoints_source": cutpoints_source, "refitted_on_this_split": args.split == "val",
                              "cutpoints": cutpoints, "bands": bands}
        tables["banding_table.md"] = format_banding_tables(cutpoints, bands, args.split)
    plots = save_calibration_plots(calibration, OUTCOME_NAMES, out)

    for fname, text in tables.items():
        print(text + "\n")
        (out / fname).write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(_jsonable(results), indent=2))
    written = ["metrics.json", *tables, *(p.name for p in plots)]
    written += [CUTPOINTS_FILE] if args.split == "val" and not args.skip_banding else []
    print(f"wrote to {out}/: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())

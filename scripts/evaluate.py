#!/usr/bin/env python
"""Evaluate trained models on a split's fixed landmarks (aggregates only).

Usage:
    python scripts/evaluate.py --checkpoint-dir runs/                 # val split, synthetic
    python scripts/evaluate.py --checkpoint-dir runs/ --data data/processed
    python scripts/evaluate.py --checkpoint-dir runs/ --split test --final   # ONE-TIME final report

Steps: load each model's best checkpoint(s) -- per-outcome selection: outcome k's predictions
come from checkpoint_best_<k>.pt; total selection (or runs without checkpoint_selection.json):
checkpoint_best.pt for all -> predict_risk at the 10-yr horizon (yearly CIF 1..10) on the
split's fixed landmarks -> ground truth from build_targets (time, event,
competing, at_risk; at_risk == 0 excluded per outcome) -> metrics (mortality: Uno C + IBS;
MALO / CVD: Wolbers C + Schoop competing-risk IBS; IPCW reference = the TRAINING split) ->
patient-level paired bootstrap (same resample for all models).

Split pinning: training saved <checkpoint-dir>/<model>/split.json (patient IDs per split, the
landmark seed, and a fingerprint of the input data). This script loads that split and seed --
it never re-derives them -- and refuses if the data given (--data, or --n-patients /
--data-seed for synthetic) does not match the saved fingerprint, or if the models were
trained on different splits. train.py writes checkpoints to <checkpoint-dir>/<model>/.

Writes <out-dir>/metrics.json and <out-dir>/metrics_table.md (outputs/ is git-ignored).
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.splitting import SplitMismatchError, check_fingerprint, data_fingerprint, load_split
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.pipeline import evaluate_models, format_tables
from compass.models.registry import MODEL_REGISTRY, build_model
from compass.training.train import Trainer, load_selected_models

log = logging.getLogger("evaluate")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
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


def load_trained(name: str, checkpoint_dir: str, visits_df, outcomes_df, schema: dict, training_config: dict):
    """(trainer, model): the saved split re-used and the best checkpoint weights loaded (never
    the latest). model is one model (total selection / older runs: checkpoint_best.pt) or
    {outcome: model} (per-outcome selection: checkpoint_best_<outcome>.pt), as recorded in the
    run's checkpoint_selection.json."""
    split = load_pinned_split(name, checkpoint_dir, visits_df, outcomes_df)
    trainer, model = make_trainer(name, visits_df, outcomes_df, schema, training_config, split=split)
    return trainer, load_selected_models(model, Path(checkpoint_dir) / name)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="Required for --split test (one-time final reporting).")
    p.add_argument("--checkpoint-dir", required=True, help="train.py's --checkpoint-dir (holds <model>/checkpoint_best*.pt)")
    p.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    p.add_argument("--data", default=None, help="Processed real-data dir (scripts/build_cohort.py). Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only; checked against the saved fingerprint.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only; checked against the saved fingerprint.")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=None, help="Defaults to configs/training.yaml's seed.")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args(argv)

    if args.split == "test":
        if not args.final:
            print("refusing: --split test requires --final (the test split is for ONE-TIME final reporting)", file=sys.stderr)
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

    trainers, models = {}, {}
    try:
        splits = {name: load_pinned_split(name, args.checkpoint_dir, visits_df, outcomes_df) for name in args.models}
        first = splits[args.models[0]]
        for name, sp in splits.items():
            if any(sp[k] != first[k] for k in ("train", "val", "test", "seed")):
                raise SplitMismatchError(f"model '{name}' was trained on a different split than '{args.models[0]}'")
    except SplitMismatchError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3
    for name in args.models:
        trainers[name], models[name] = load_trained(name, args.checkpoint_dir, visits_df, outcomes_df, schema, training_config)

    horizon = int(training_config["horizons"]["primary"])
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(training_config["seed"])
    results = evaluate_models(trainers, models, args.split, horizon, args.n_bootstrap, bootstrap_seed)
    results["data"] = "real (processed)" if args.data else "synthetic"

    table = format_tables(results)
    print(table)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_jsonable(results), indent=2))
    (out / "metrics_table.md").write_text(table + "\n")
    print(f"\nwrote {out}/metrics.json and {out}/metrics_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

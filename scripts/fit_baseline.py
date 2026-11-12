#!/usr/bin/env python
"""Fit the classical baseline (pre-specified secondary analysis 1) on the TRAINING split only:
Cox (mortality) and Fine-Gray with death competing (MALO, CVD) on each patient's first valid
landmark, fixed ridge penalty (configs/baseline.yaml), no tuning. See compass.models.cox_baseline.

Usage:
    python scripts/fit_baseline.py --final-models configs/final_models.yaml --data data/processed
    # writes runs/baseline/cox_baseline.json (aggregates: feature statistics, coefficients,
    # baseline cumulative hazards at yearly times, counts, warnings)

The pinned split comes from configs/final_models.yaml's split directory (checked against the
data fingerprint). The Trainer used to build the examples is given the training IDs only
(validation and test lists EMPTY), so no validation or test patient is landmarked, scaled or
seen by the fit. Evaluate with scripts/evaluate.py --baseline runs/baseline/cox_baseline.json.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.pipeline import _collate, split_ground_truth
from compass.models.cox_baseline import fit_baseline, last_observed, save_fit

_spec = importlib.util.spec_from_file_location("evaluate_script", Path(__file__).with_name("evaluate.py"))
evaluate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_script)


def training_features(trainer) -> tuple[list, "object"]:
    """(patient_ids, last-observed feature matrix) of the training split's fixed examples (the
    first valid landmark), batch by batch through the same collation as evaluation."""
    import numpy as np

    examples = trainer.fixed_examples("train")
    parts = []
    for i in range(0, len(examples), trainer.batch_size):
        b = _collate(trainer, examples[i : i + trainer.batch_size])
        parts.append(last_observed(b["values"].numpy(), b["mask"].numpy(), b["seq_lengths"].numpy()))
    return [ex["patient_id"] for ex in examples], np.vstack(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final-models", default="configs/final_models.yaml", help="Gives the pinned split directory.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data", default=None, help="Processed real-data dir. Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only.")
    p.add_argument("--out", default=None, help="Defaults to configs/baseline.yaml's fit_path.")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
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

    ns = argparse.Namespace(final_models=args.final_models, checkpoint_dir=None, models=None)
    try:
        names, model_dirs, split = evaluate_script.resolve_models(ns, visits_df, outcomes_df)
    except (evaluate_script.SplitMismatchError, evaluate_script.FinalModelsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3

    train_only = {"train": split["train"], "val": [], "test": [], "seed": split["seed"]}
    trainer, _ = evaluate_script.make_trainer(names[0], visits_df, outcomes_df, schema, training_config, split=train_only)
    ids, x_train = training_features(trainer)
    gt_ids, ground_truth = split_ground_truth(trainer, "train")
    if ids != gt_ids or set(ids) - set(split["train"]):
        raise RuntimeError("training examples do not match the pinned training split")

    continuous = [v["name"] for v in schema["continuous_variables"]]
    medications = [m["name"] for m in schema["medication_flags"]]
    smoking = next(d["values"] for d in schema["demographics"] if d["name"] == "smoking_status")
    feature_names = build_feature_names(continuous, medications, ("male", "female"), smoking)
    fit = fit_baseline(x_train, feature_names, ground_truth, float(cfg["penalizer"]), int(trainer.num_time_bins))

    meta = {
        "fitted_on_split": "train",
        "n_train_patients": len(ids),
        "split_seed": int(split["seed"]),
        "data_fingerprint": split["fingerprint"],
        "example_source": f"first valid landmark (training split), built as for {names[0]}",
        "config": cfg,
    }
    out = Path(args.out or cfg["fit_path"])
    save_fit(out, fit, meta)

    t = fit["transform"]
    print(f"fitted on {len(ids)} training patients (validation / test not loaded)")
    print(f"features: {len(t['feature_names'])} variables + {len(t['indicators'])} missing indicators "
          f"-> {len(t['columns'])} columns; dropped (zero variance in training): {t['dropped_zero_variance'] or 'none'}")
    for name, o in fit["outcomes"].items():
        extra = f", competing {o['n_competing']}" if "n_competing" in o else ""
        print(f"{name}: {o['method']}; n {o['n']}, events {o['n_events']}{extra}; "
              f"warnings: {o['warnings'] or 'none'}")
    print(f"wrote {os.path.normpath(str(out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

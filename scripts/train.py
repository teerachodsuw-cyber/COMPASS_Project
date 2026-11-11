#!/usr/bin/env python
"""Thin CLI entry point for compass.training.train.Trainer.

Usage:
    python scripts/train.py --model all --checkpoint-dir runs/       # epochs / patience from configs/training.yaml
    python scripts/train.py --model dynamic_deephit --epochs 15 --checkpoint-dir runs/ddh   # CLI overrides config

--model all trains all three architectures SEQUENTIALLY (not parallelized in this pass) on
the SAME train/val/test patient split. This is not something this script has to work hard to
guarantee: compass.data.splitting.patient_level_split is a pure deterministic function of
(outcomes_df, train_frac, val_frac, test_frac, seed), and Trainer.__init__ calls it (via
run_preprocessing) with the SAME training_config seed and the SAME outcomes_df regardless of
which model is being trained -- so training all three separately (three separate `python
scripts/train.py --model X` invocations, on different days even) produces the identical split
as training all three via one `--model all` invocation, as long as --n-patients/--data-seed
(which determine the synthetic cohort itself) and configs/training.yaml's seed are unchanged.
This matters because the paired bootstrap comparison downstream needs all three models
evaluated on comparable data.

Data source: synthetic by default (--n-patients/--data-seed control the synthetic cohort).
--data <dir> instead reads the real-data cohort built by scripts/build_cohort.py
(<dir>/cohort_visits.parquet + cohort_outcomes.parquet). configs/training.yaml's own `seed`
controls the train/val/test split and all landmark sampling, independent of --data-seed.
"""

import argparse
import json

import pandas as pd
import torch
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.registry import MODEL_REGISTRY, build_model
from compass.training.train import Trainer

def train_one_model(name: str, args, visits_df, outcomes_df, schema, training_config) -> dict:
    with open(MODEL_REGISTRY[name][1]) as f:
        model_config = yaml.safe_load(f)

    continuous_columns = [v["name"] for v in schema["continuous_variables"]]
    medication_columns = [m["name"] for m in schema["medication_flags"]]
    sex_categories = ("male", "female")
    smoking_status_categories = next(d["values"] for d in schema["demographics"] if d["name"] == "smoking_status")
    num_features = len(build_feature_names(continuous_columns, medication_columns, sex_categories, smoking_status_categories))
    administrative_cutoff = pd.Timestamp(schema["censoring"]["administrative_cutoff"])
    horizons = [int(training_config["horizons"]["primary"])]

    torch.manual_seed(training_config["seed"])
    model = build_model(name, model_config, num_features, horizon_years=horizons[0])

    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        visits_df=visits_df,
        outcomes_df=outcomes_df,
        continuous_columns=continuous_columns,
        medication_columns=medication_columns,
        sex_categories=sex_categories,
        smoking_status_categories=smoking_status_categories,
        horizons=horizons,
        administrative_cutoff=administrative_cutoff,
    )

    checkpoint_dir = f"{args.checkpoint_dir}/{name}"
    print(f"=== training {name}: {len(trainer.train_ids)} train / {len(trainer.val_ids)} val / {len(trainer.test_ids)} test patients ===")
    history = trainer.fit(
        train_patients=trainer.train_ids,
        val_patients=trainer.val_ids,
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
    )

    # fit() has reloaded checkpoint_best.pt (best total); evaluate.py loads per-outcome
    # checkpoints when checkpoint_selection is per_outcome. No test-split loss is computed here:
    # the test split is touched only by scripts/evaluate.py --split test --final (one-time reporting).
    print(f"{name}: final train total={history['train'][-1]['total']:.4f}  val total={history['val'][-1]['total']:.4f}  "
          f"best total epoch={history.get('best_epoch')} (val total={history.get('best_val_loss', float('nan')):.4f})")
    per_outcome = "  ".join(f"{o} epoch {history['best_epoch_per_outcome'][o]} ({history['best_val_nll_per_outcome'][o]:.4f})"
                            for o in history["best_epoch_per_outcome"])
    print(f"{name}: checkpoint selection {history['checkpoint_selection']}; best val NLL per outcome: {per_outcome}")
    return {"history": history, "split": {"train": trainer.train_ids, "val": trainer.val_ids, "test": trainer.test_ids}}


def resolve_epochs_and_patience(args, training_config: dict) -> tuple[int, int | None]:
    """CLI flags override; otherwise configs/training.yaml training_loop.{max_epochs,
    early_stopping_patience}."""
    loop = training_config.get("training_loop", {})
    epochs = args.epochs if args.epochs is not None else loop.get("max_epochs")
    if epochs is None:
        raise SystemExit("no max epochs: pass --epochs or set training_loop.max_epochs in configs/training.yaml")
    patience = args.early_stopping_patience if args.early_stopping_patience is not None else loop.get("early_stopping_patience")
    return int(epochs), (int(patience) if patience is not None else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODEL_REGISTRY.keys(), "all"], required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs; default configs/training.yaml training_loop.max_epochs.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=None, help="Defaults to configs/training.yaml's seed if unset.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Default configs/training.yaml training_loop.early_stopping_patience.")
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
    parser.add_argument("--data", type=str, default=None,
                        help="Processed real-data dir (from scripts/build_cohort.py). Omit for synthetic data.")
    args = parser.parse_args()

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)
    args.epochs, args.early_stopping_patience = resolve_epochs_and_patience(args, training_config)
    print(f"max epochs {args.epochs}, early-stopping patience {args.early_stopping_patience}")

    if args.data:
        visits_df = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes_df = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        data_seed = args.data_seed if args.data_seed is not None else training_config["seed"]
        visits_df, outcomes_df = generate_synthetic_cohort(n_patients=args.n_patients, seed=data_seed)

    names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    results = {}
    for name in names:
        results[name] = train_one_model(name, args, visits_df, outcomes_df, schema, training_config)

    if len(names) > 1:
        splits = {name: results[name]["split"] for name in names}
        reference = splits[names[0]]
        all_match = all(splits[name] == reference for name in names[1:])
        print(f"=== same train/val/test split across all models: {all_match} ===")

    summary_path = f"{args.checkpoint_dir}/run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {name: {"checkpoint_selection": results[name]["history"]["checkpoint_selection"],
                    "best_epoch": results[name]["history"].get("best_epoch"),
                    "best_val_loss": results[name]["history"].get("best_val_loss"),
                    "best_epoch_per_outcome": results[name]["history"]["best_epoch_per_outcome"],
                    "best_val_nll_per_outcome": results[name]["history"]["best_val_nll_per_outcome"]}
             for name in names},
            f, indent=2,
        )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

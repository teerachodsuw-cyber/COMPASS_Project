#!/usr/bin/env python
"""Thin CLI entry point for compass.training.train.Trainer.

Usage:
    python scripts/train.py --model dynamic_deephit --epochs 15 --checkpoint-dir runs/ddh
    python scripts/train.py --model all --epochs 15 --checkpoint-dir runs/

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

Data source: synthetic only, for now (no real-data loader exists yet -- see README's Status
section). --n-patients/--data-seed control the synthetic cohort; configs/training.yaml's own
`seed` controls the train/val/test split and all landmark sampling, independent of
--data-seed.
"""

import argparse
import json

import pandas as pd
import torch
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.models.tracer.model import TracerModel
from compass.training.train import Trainer

MODEL_REGISTRY = {
    "dynamic_deephit": (DynamicDeepHitModel, "configs/model_dynamic_deephit.yaml"),
    "tracer": (TracerModel, "configs/model_tracer.yaml"),
    "survlatent_ode": (SurvLatentODEModel, "configs/model_survlatent_ode.yaml"),
}


def _build_model(name: str, model_config: dict, num_features: int, horizon_years: float):
    model_cls, _ = MODEL_REGISTRY[name]
    if name == "dynamic_deephit":
        return model_cls(
            num_features=num_features,
            hidden_dim=model_config["hidden_dim"],
            num_gru_layers=model_config["num_layers"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
            ranking_weight=model_config["ranking_weight"],
            auxiliary_weight=model_config["auxiliary_weight"],
            horizon_years=horizon_years,
        )
    if name == "tracer":
        return model_cls(
            num_features=num_features,
            feature_embed_dim=model_config["feature_embed_dim"],
            covariate_attention_heads=model_config["covariate_attention_heads"],
            hidden_dim=model_config["hidden_dim"],
            temporal_attention_heads=model_config["temporal_attention_heads"],
            num_transformer_layers=model_config["num_transformer_layers"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
        )
    if name == "survlatent_ode":
        return model_cls(
            num_features=num_features,
            latent_dim=model_config["latent_dim"],
            ode_hidden_dim=model_config["ode_hidden_dim"],
            cause_head_hidden_dim=model_config["cause_head_hidden_dim"],
            num_time_bins=model_config["num_time_bins"],
            bin_width_years=model_config["bin_width_years"],
            solver_method=model_config["solver_method"],
        )
    raise ValueError(f"unknown model name: {name}")


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
    model = _build_model(name, model_config, num_features, horizon_years=horizons[0])

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

    test_metrics = trainer.evaluate(trainer.test_ids, resample_landmarks=False)
    print(f"{name}: final train total={history['train'][-1]['total']:.4f}  val total={history['val'][-1]['total']:.4f}  test total={test_metrics['total']:.4f}")
    return {"history": history, "test_metrics": test_metrics, "split": {"train": trainer.train_ids, "val": trainer.val_ids, "test": trainer.test_ids}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODEL_REGISTRY.keys(), "all"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=None, help="Defaults to configs/training.yaml's seed if unset.")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
    args = parser.parse_args()

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

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
        json.dump({name: {"test_metrics": results[name]["test_metrics"]} for name in names}, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

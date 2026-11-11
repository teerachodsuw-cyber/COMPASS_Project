"""Tests for the tuning tool (compass.training.tuning + scripts/tune.py). Synthetic data only."""

import argparse
import importlib.util
import json

import pytest
import torch

import compass.data.preprocessing as preprocessing
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation import pipeline
from compass.training import tuning
from compass.training.train import Trainer

N_PATIENTS, DATA_SEED = 300, 42
TUNING_CFG = tuning.load_yaml("configs/tuning.yaml")
TRAINING_CFG = tuning.load_yaml("configs/training.yaml")
SCHEMA = tuning.load_yaml("configs/data_schema.yaml")
BASE_LR = {"dynamic_deephit": 0.001, "tracer": 0.001, "survlatent_ode": 0.001}
SMALL = {  # tiny models so the synthetic tests stay fast
    "dynamic_deephit": {"hidden_dim": 8, "num_layers": 1, "cause_head_hidden_dim": 4, "batch_size": 32},
}

_spec = importlib.util.spec_from_file_location("tune_script", "scripts/tune.py")
tune_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune_script)


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)


def _small_grid():
    return {**TUNING_CFG, "grid": {"learning_rate_multipliers": [1.0, 0.1], "weight_decay": [0.0, 1e-3]}}


def test_grid_expands_to_nine_configs_per_model():
    entries = tuning.expand_grid(TUNING_CFG, list(BASE_LR), BASE_LR)
    assert len(entries) == 27
    for model in BASE_LR:
        mine = [e for e in entries if e["model"] == model]
        assert len(mine) == 9 and len({e["config_id"] for e in mine}) == 9
        assert sorted({round(e["learning_rate"], 12) for e in mine}) == sorted({1e-3, 3e-4, 1e-4})
        assert sorted({e["weight_decay"] for e in mine}) == [0.0, 1e-4, 1e-3]


def test_weight_decay_and_learning_rate_reach_adamw(data, tmp_path):
    visits, outcomes = data
    split = tuning.resolve_split(tmp_path, visits, outcomes, TRAINING_CFG)
    entry = {"model": "dynamic_deephit", "config_id": "x", "learning_rate": 3e-4, "weight_decay": 1e-3}
    trainer, _ = tuning.make_trainer(entry, visits, outcomes, SCHEMA, TRAINING_CFG, split, SMALL["dynamic_deephit"])
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    group = trainer.optimizer.param_groups[0]
    assert group["weight_decay"] == 1e-3 and group["lr"] == 3e-4


def test_default_model_configs_use_adamw_with_zero_weight_decay():
    for name in BASE_LR:
        cfg = tuning.load_yaml(f"configs/model_{name}.yaml")
        assert cfg["weight_decay"] == 0.0 and cfg["learning_rate"] == 0.001


def test_train_py_reads_epochs_and_patience_from_config():
    spec = importlib.util.spec_from_file_location("train_script", "scripts/train.py")
    train_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_script)
    assert TRAINING_CFG["training_loop"] == {"max_epochs": 100, "early_stopping_patience": 10}
    none = argparse.Namespace(epochs=None, early_stopping_patience=None)
    assert train_script.resolve_epochs_and_patience(none, TRAINING_CFG) == (100, 10)
    cli = argparse.Namespace(epochs=7, early_stopping_patience=3)
    assert train_script.resolve_epochs_and_patience(cli, TRAINING_CFG) == (7, 3)


@pytest.fixture(scope="module")
def tuned(data, tmp_path_factory):
    """A 2x2 grid for dynamic_deephit, 2 epochs each, on synthetic data."""
    visits, outcomes = data
    run_dir, out_dir = tmp_path_factory.mktemp("tune_run"), tmp_path_factory.mktemp("tune_out")
    split = tuning.resolve_split(run_dir, visits, outcomes, TRAINING_CFG)
    entries = tuning.expand_grid(_small_grid(), ["dynamic_deephit"], BASE_LR)
    rows = tuning.run_grid(entries, visits, outcomes, SCHEMA, TRAINING_CFG, split, run_dir, out_dir,
                           max_epochs=2, patience=1, model_config_overrides=SMALL, log=lambda *_: None)
    return run_dir, out_dir, split, entries, rows


def test_every_config_uses_the_same_pinned_split(tuned):
    run_dir, _, split, entries, _ = tuned
    for e in entries:
        saved = json.loads((run_dir / e["model"] / e["config_id"] / "split.json").read_text())
        assert saved["train"] == split["train"] and saved["val"] == split["val"] and saved["seed"] == split["seed"]
        assert saved["test"] == []  # test split never handed to the Trainer
    assert len(split["test"]) > 0  # ... but kept in the run-level split.json


def test_results_and_summary_written(tuned):
    _, out_dir, _, entries, rows = tuned
    assert len(rows) == len(entries) == 4
    for r in rows:
        assert r["epochs_run"] <= 2 and 0 <= r["best_epoch"] < r["epochs_run"]
        assert all(f"val_nll_{o}" in r and f"val_cindex_{o}" in r for o in ("mortality", "malo", "cvd"))
    header = (out_dir / "summary.csv").read_text().splitlines()[0].split(",")
    assert header == tuning.SUMMARY_COLUMNS
    assert "**WINNER**" in (out_dir / "summary.md").read_text()


def test_resume_skips_finished_configs(tuned, data, monkeypatch):
    run_dir, out_dir, split, entries, rows = tuned
    visits, outcomes = data

    def must_not_train(*a, **k):
        raise AssertionError("a finished config was retrained")

    monkeypatch.setattr(tuning, "train_and_score", must_not_train)
    again = tuning.run_grid(entries, visits, outcomes, SCHEMA, TRAINING_CFG, split, run_dir, out_dir,
                            max_epochs=2, patience=1, model_config_overrides=SMALL, log=lambda *_: None)
    assert [r["best_val_total"] for r in again] == [r["best_val_total"] for r in rows]


def test_winner_is_lowest_best_val_loss():
    rows = [
        {"model": "a", "config_id": "1", "best_val_total": 1.5},
        {"model": "a", "config_id": "2", "best_val_total": 1.2},
        {"model": "a", "config_id": "3", "best_val_total": 1.9},
        {"model": "b", "config_id": "1", "best_val_total": 0.8},
        {"model": "b", "config_id": "2", "best_val_total": 0.9},
    ]
    marked = tuning.mark_winners(rows)
    assert [(r["model"], r["config_id"]) for r in marked if r["winner"]] == [("a", "2"), ("b", "1")]


def test_tuned_winner_matches_minimum(tuned):
    _, _, _, _, rows = tuned
    winners = [r for r in rows if r["winner"]]
    assert len(winners) == 1 and winners[0]["best_val_total"] == min(r["best_val_total"] for r in rows)


def test_test_split_is_never_loaded(data, tmp_path, monkeypatch):
    visits, outcomes = data
    split = tuning.resolve_split(tmp_path, visits, outcomes, TRAINING_CFG)
    test_ids = set(split["test"])
    landmarked = []
    real_construct = preprocessing.construct_landmark_example

    def spy_construct(patient_visits, patient_outcomes, landmark_date):
        landmarked.append(patient_visits["patient_id"].iloc[0])
        return real_construct(patient_visits, patient_outcomes, landmark_date)

    real_fixed = Trainer.fixed_examples

    def guarded_fixed(self, which):
        assert which != "test", "tuning asked for test-split examples"
        return real_fixed(self, which)

    seen_splits = []
    real_gt = pipeline.split_ground_truth

    def spy_gt(trainer, which):
        seen_splits.append(which)
        return real_gt(trainer, which)

    monkeypatch.setattr(preprocessing, "construct_landmark_example", spy_construct)
    monkeypatch.setattr(Trainer, "fixed_examples", guarded_fixed)
    monkeypatch.setattr(pipeline, "split_ground_truth", spy_gt)
    entry = tuning.expand_grid(_small_grid(), ["dynamic_deephit"], BASE_LR)[0]
    result = tuning.train_and_score(entry, visits, outcomes, SCHEMA, TRAINING_CFG, split, tmp_path, 1, None,
                                    SMALL["dynamic_deephit"])
    assert landmarked and not (set(landmarked) & test_ids)  # no test patient ever landmarked / scaled
    assert "test" not in seen_splits and "val" in seen_splits
    assert result["epochs_run"] == 1


def test_dry_run_lists_27_configs_with_estimate(capsys, tmp_path):
    assert tune_script.main(["--dry-run", "--run-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("27 configs (9 per model)")
    assert sum(1 for line in out.splitlines() if line.startswith("| ") and line[2].isdigit()) == 27
    assert "Estimated total" in out and "ALL: typical" in out

"""Tests for compass.training.train.Trainer -- the shared, architecture-agnostic training
loop. Uses small synthetic cohorts (n=300, the floor MALO's rarity needs for
patient_level_split's stratification to succeed at all, per compass.data.splitting) and
small hidden dims throughout, purely for test speed -- these are not meant to prove anything
about final model quality, just about the Trainer's own mechanics (epoch loop, landmark
resampling policy, checkpoint/resume, reproducibility, shared split).
"""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.preprocessing import build_feature_names
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.models.tracer.model import TracerModel
from compass.training import train as train_module
from compass.training.train import Trainer


@pytest.fixture(scope="module")
def schema():
    with open("configs/data_schema.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def training_config():
    with open("configs/training.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def synthetic_data():
    return generate_synthetic_cohort(n_patients=300, seed=42)


@pytest.fixture(scope="module")
def data_columns(schema):
    continuous_columns = [v["name"] for v in schema["continuous_variables"]]
    medication_columns = [m["name"] for m in schema["medication_flags"]]
    num_features = len(build_feature_names(continuous_columns, medication_columns, ("male", "female"), [0, 1, 2]))
    return continuous_columns, medication_columns, num_features


def _make_trainer(name, model_config_overrides, schema, training_config, synthetic_data, data_columns):
    continuous_columns, medication_columns, num_features = data_columns
    visits_df, outcomes_df = synthetic_data
    administrative_cutoff = pd.Timestamp(schema["censoring"]["administrative_cutoff"])

    model_config = {
        "num_time_bins": 30,
        "bin_width_years": 1.0,
        "batch_size": 16,
        "learning_rate": 0.01,
        **model_config_overrides,
    }

    torch.manual_seed(0)
    if name == "dynamic_deephit":
        model = DynamicDeepHitModel(
            num_features=num_features, hidden_dim=16, num_gru_layers=1, cause_head_hidden_dim=8,
            num_time_bins=model_config["num_time_bins"], bin_width_years=model_config["bin_width_years"],
        )
    elif name == "tracer":
        model = TracerModel(
            num_features=num_features, feature_embed_dim=8, covariate_attention_heads=2, hidden_dim=16,
            temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=8,
            num_time_bins=model_config["num_time_bins"], bin_width_years=model_config["bin_width_years"],
        )
    elif name == "survlatent_ode":
        model = SurvLatentODEModel(
            num_features=num_features, latent_dim=16, ode_hidden_dim=32, cause_head_hidden_dim=16,
            num_time_bins=model_config["num_time_bins"], bin_width_years=model_config["bin_width_years"],
            solver_method="rk4",
        )
    else:
        raise ValueError(name)

    return Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        visits_df=visits_df,
        outcomes_df=outcomes_df,
        continuous_columns=continuous_columns,
        medication_columns=medication_columns,
        sex_categories=("male", "female"),
        smoking_status_categories=[0, 1, 2],
        horizons=[10],
        administrative_cutoff=administrative_cutoff,
    )


@pytest.mark.parametrize("name", ["dynamic_deephit", "tracer", "survlatent_ode"])
def test_trainer_single_epoch_runs(name, schema, training_config, synthetic_data, data_columns):
    trainer = _make_trainer(name, {}, schema, training_config, synthetic_data, data_columns)
    rng = np.random.default_rng(0)

    metrics = trainer.train_one_epoch(trainer.train_ids, rng)

    assert "total" in metrics
    assert np.isfinite(metrics["total"])


def test_landmarks_resampled_each_epoch_in_training(schema, training_config, synthetic_data, data_columns, monkeypatch):
    trainer = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    rng = np.random.default_rng(0)

    seen_landmarks = []
    real_sample_landmark = train_module.sample_landmark

    def spy(patient_visits, rng_arg, **kwargs):
        landmark = real_sample_landmark(patient_visits, rng_arg, **kwargs)
        seen_landmarks.append(landmark)
        return landmark

    monkeypatch.setattr(train_module, "sample_landmark", spy)

    trainer.train_one_epoch(trainer.train_ids, rng)
    epoch_1_landmarks = list(seen_landmarks)
    seen_landmarks.clear()

    trainer.train_one_epoch(trainer.train_ids, rng)
    epoch_2_landmarks = list(seen_landmarks)

    assert len(epoch_1_landmarks) == len(epoch_2_landmarks) == len(trainer.train_ids)
    # Same continuing rng, same patient order -> if resampling is real (not accidentally
    # fixed), at least SOME patients must have gotten a different landmark epoch to epoch.
    assert epoch_1_landmarks != epoch_2_landmarks


def test_landmarks_fixed_in_evaluation(schema, training_config, synthetic_data, data_columns):
    trainer = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)

    before = {pid: ex["landmark_date"] for pid, ex in trainer._fixed_eval_examples.items()}
    trainer.evaluate(trainer.val_ids, resample_landmarks=False)
    trainer.evaluate(trainer.val_ids, resample_landmarks=False)
    after = {pid: ex["landmark_date"] for pid, ex in trainer._fixed_eval_examples.items()}

    assert before == after  # untouched by repeated evaluate() calls -- never rebuilt


def test_checkpoint_and_resume(schema, training_config, synthetic_data, data_columns, tmp_path):
    checkpoint_dir = str(tmp_path / "ckpt")

    trainer1 = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    trainer1.fit(trainer1.train_ids, trainer1.val_ids, num_epochs=2, checkpoint_dir=checkpoint_dir)
    state_after_2_epochs = copy.deepcopy(trainer1.model.state_dict())

    log_lines_after_first_fit = (tmp_path / "ckpt" / "training_log.jsonl").read_text().strip().splitlines()
    assert len(log_lines_after_first_fit) == 2

    trainer2 = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    trainer2.resume(f"{checkpoint_dir}/checkpoint_latest.pt")

    # Resume must restore the EXACT state fit() left off at, not a fresh restart.
    for key in state_after_2_epochs:
        assert torch.allclose(state_after_2_epochs[key], trainer2.model.state_dict()[key])
    assert trainer2._resume_epoch == 1  # 0-indexed, 2 epochs already ran (0, 1)

    trainer2.fit(trainer2.train_ids, trainer2.val_ids, num_epochs=4, checkpoint_dir=checkpoint_dir)

    log_lines_after_resume = (tmp_path / "ckpt" / "training_log.jsonl").read_text().strip().splitlines()
    assert len(log_lines_after_resume) == 4  # appended, not overwritten
    import json

    epochs_logged = [json.loads(line)["epoch"] for line in log_lines_after_resume]
    assert epochs_logged == [0, 1, 2, 3]  # continuous, no restart back to 0


def test_reproducibility_with_fixed_seed(schema, training_config, synthetic_data, data_columns, tmp_path):
    trainer_a = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    history_a = trainer_a.fit(trainer_a.train_ids, trainer_a.val_ids, num_epochs=3, checkpoint_dir=str(tmp_path / "a"))

    trainer_b = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    history_b = trainer_b.fit(trainer_b.train_ids, trainer_b.val_ids, num_epochs=3, checkpoint_dir=str(tmp_path / "b"))

    assert len(history_a["train"]) == len(history_b["train"]) == 3
    for epoch_a, epoch_b in zip(history_a["train"], history_b["train"]):
        assert epoch_a == epoch_b  # exact equality -- same seed, same init, same landmarks
    for epoch_a, epoch_b in zip(history_a["val"], history_b["val"]):
        assert epoch_a == epoch_b


def test_all_three_models_use_same_split(schema, training_config, synthetic_data, data_columns):
    trainer_ddh = _make_trainer("dynamic_deephit", {}, schema, training_config, synthetic_data, data_columns)
    trainer_tracer = _make_trainer("tracer", {}, schema, training_config, synthetic_data, data_columns)
    trainer_survlatent = _make_trainer("survlatent_ode", {}, schema, training_config, synthetic_data, data_columns)

    assert trainer_ddh.train_ids == trainer_tracer.train_ids == trainer_survlatent.train_ids
    assert trainer_ddh.val_ids == trainer_tracer.val_ids == trainer_survlatent.val_ids
    assert trainer_ddh.test_ids == trainer_tracer.test_ids == trainer_survlatent.test_ids


# ---- evaluation landmark rule (configs/training.yaml: evaluation.landmark) -----------------


def _first_candidate(patient_visits, index_date, death_date):
    d = patient_visits["visit_date"]
    ok = d >= index_date
    if pd.notna(death_date):
        ok &= d < death_date
    return d[ok].min()


@pytest.fixture(scope="module")
def shifted_index_data(synthetic_data):
    """Synthetic cohort with index_date moved to each patient's 2nd visit (when there is one),
    so the first VALID landmark differs from the first visit."""
    visits_df, outcomes_df = synthetic_data
    second = visits_df.groupby("patient_id")["visit_date"].apply(lambda s: s.sort_values().iloc[min(1, len(s) - 1)])
    shifted = outcomes_df.copy()
    shifted["index_date"] = shifted["patient_id"].map(second)
    return visits_df, shifted


def _expected_first_landmarks(trainer, visits_df, outcomes_df, ids):
    o = outcomes_df.set_index("patient_id")
    out = {}
    for pid in ids:
        death = o.loc[pid, "mortality_date"] if o.loc[pid, "mortality_event"] == 1 else pd.NaT
        out[pid] = _first_candidate(visits_df[visits_df.patient_id == pid], o.loc[pid, "index_date"], death)
    return out


def test_eval_landmark_is_first_post_index_for_val_test_and_ipcw_reference(schema, training_config, shifted_index_data, data_columns):
    visits_df, outcomes_df = shifted_index_data
    assert training_config["evaluation"]["landmark"] == "first_post_index"
    trainer = _make_trainer("dynamic_deephit", {}, schema, training_config, shifted_index_data, data_columns)
    first_visit = visits_df.groupby("patient_id")["visit_date"].min()
    for split in ("val", "test", "train"):  # "train" = the IPCW reference examples
        examples = trainer.fixed_examples(split)
        expected = _expected_first_landmarks(trainer, visits_df, outcomes_df, [ex["patient_id"] for ex in examples])
        assert all(ex["landmark_date"] == expected[ex["patient_id"]] for ex in examples), split
        assert any(ex["landmark_date"] != first_visit[ex["patient_id"]] for ex in examples), split  # non-trivial


def test_training_landmarks_stay_random(schema, training_config, shifted_index_data, data_columns):
    visits_df, outcomes_df = shifted_index_data
    trainer = _make_trainer("dynamic_deephit", {}, schema, training_config, shifted_index_data, data_columns)
    ids = trainer.train_ids[:80]
    expected = _expected_first_landmarks(trainer, visits_df, outcomes_df, ids)
    rng = np.random.default_rng(0)
    epoch_a = {ex["patient_id"]: ex["landmark_date"] for ex in trainer._build_examples_for(ids, rng)}
    epoch_b = {ex["patient_id"]: ex["landmark_date"] for ex in trainer._build_examples_for(ids, rng)}
    assert any(epoch_a[p] != epoch_b[p] for p in ids)            # vary across epochs
    assert any(epoch_a[p] != expected[p] for p in ids)           # not the first-landmark rule
    o = outcomes_df.set_index("patient_id")
    assert all(epoch_a[p] >= o.loc[p, "index_date"] for p in ids)  # still valid (on/after index)


def test_eval_landmark_random_setting(schema, training_config, shifted_index_data, data_columns):
    visits_df, outcomes_df = shifted_index_data
    cfg = {**training_config, "evaluation": {"landmark": "random"}}
    trainer = _make_trainer("dynamic_deephit", {}, schema, cfg, shifted_index_data, data_columns)
    examples = trainer.fixed_examples("val")
    expected = _expected_first_landmarks(trainer, visits_df, outcomes_df, [ex["patient_id"] for ex in examples])
    assert trainer.eval_landmark == "random"
    assert any(ex["landmark_date"] != expected[ex["patient_id"]] for ex in examples)


# ---- per-outcome loss logging --------------------------------------------------------------


@pytest.mark.parametrize("name", ["dynamic_deephit", "tracer", "survlatent_ode"])
def test_training_log_has_per_outcome_losses(name, schema, training_config, synthetic_data, data_columns, tmp_path, capsys):
    import json

    trainer = _make_trainer(name, {}, schema, training_config, synthetic_data, data_columns)
    trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=1, checkpoint_dir=str(tmp_path))
    row = json.loads((tmp_path / "training_log.jsonl").read_text().splitlines()[0])
    for part in ("train", "val"):
        for outcome in ("mortality", "malo", "cvd"):
            assert np.isfinite(row[part][f"nll_{outcome}"]), (part, outcome)
    console = capsys.readouterr().out
    assert "epoch 0: train total=" in console
    assert all(f"{o}=" in console for o in ("mortality", "malo", "cvd"))

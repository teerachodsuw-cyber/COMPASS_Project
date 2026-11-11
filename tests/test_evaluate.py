"""Tests for the evaluation pipeline (scripts/evaluate.py + compass.evaluation.pipeline).
Synthetic data only."""

import importlib.util
import json

import numpy as np
import pytest
import torch
import yaml

import compass.evaluation.metrics as metrics_module
import compass.evaluation.pipeline as pipeline
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.metrics import compute_all_metrics

N_PATIENTS, DATA_SEED = 300, 42
MODELS = ["dynamic_deephit", "tracer", "survlatent_ode"]

_spec = importlib.util.spec_from_file_location("evaluate_script", "scripts/evaluate.py")
evaluate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_script)


def _configs():
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training = yaml.safe_load(f)
    return schema, training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train each model for 1 epoch exactly as train.py would; checkpoints in a temp dir."""
    ckpt_dir = tmp_path_factory.mktemp("runs")
    schema, training = _configs()
    visits, outcomes = generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    trainers = {}
    for name in MODELS:
        torch.manual_seed(training["seed"])
        trainer, _ = evaluate_script.make_trainer(name, visits, outcomes, schema, training)
        trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=1, checkpoint_dir=str(ckpt_dir / name))
        trainers[name] = trainer
    return ckpt_dir, trainers


def _argv(ckpt_dir, out_dir, *extra):
    return ["--checkpoint-dir", str(ckpt_dir), "--n-patients", str(N_PATIENTS), "--data-seed", str(DATA_SEED),
            "--n-bootstrap", "20", "--out-dir", str(out_dir), *extra]


def test_evaluate_runs_end_to_end_on_val(trained, tmp_path):
    ckpt_dir, _ = trained
    assert evaluate_script.main(_argv(ckpt_dir, tmp_path)) == 0
    results = json.loads((tmp_path / "metrics.json").read_text())
    assert results["split"] == "val" and results["horizon_years"] == 10
    assert results["ipcw_reference"]["split"] == "train"
    assert {r["model"] for r in results["per_model"]} == set(MODELS)
    assert {r["outcome"] for r in results["per_model"]} == {"mortality", "malo", "cvd"}
    assert {r["metric_name"] for r in results["per_model"]} == {"cindex", "ibs"}
    assert all("n_valid" in r and r["n_draws"] == 20 for r in results["per_model"])
    assert len({r["comparison"] for r in results["pairwise"]}) == 3
    table = (tmp_path / "metrics_table.md").read_text()
    assert "| Model | Outcome | C-index (95% CI) | IBS (95% CI) | n at risk | n events |" in table


def test_test_split_without_final_refuses(trained, tmp_path, capsys):
    ckpt_dir, _ = trained
    assert evaluate_script.main(_argv(ckpt_dir, tmp_path, "--split", "test")) == 2
    assert "requires --final" in capsys.readouterr().err
    assert not (tmp_path / "metrics.json").exists()


def test_test_split_with_final_warns(trained, tmp_path, caplog):
    ckpt_dir, _ = trained
    with caplog.at_level("WARNING"):
        assert evaluate_script.main(_argv(ckpt_dir, tmp_path, "--split", "test", "--final", "--models", "dynamic_deephit")) == 0
    assert any("one-time final reporting" in r.message for r in caplog.records)


def test_ipcw_reference_is_the_train_split(trained, monkeypatch):
    _, trainers = trained
    seen = {"bootstrap": None, "metric_refs": []}
    real_bootstrap, real_wolbers = pipeline.paired_bootstrap, metrics_module.wolbers_cindex

    def spy_bootstrap(predictions, ground_truth, censoring_reference, *a, **kw):
        seen["bootstrap"] = censoring_reference
        return real_bootstrap(predictions, ground_truth, censoring_reference, *a, **kw)

    def spy_wolbers(*a, reference=None, **kw):
        seen["metric_refs"].append(reference)
        return real_wolbers(*a, reference=reference, **kw)

    monkeypatch.setattr(pipeline, "paired_bootstrap", spy_bootstrap)
    monkeypatch.setattr(metrics_module, "wolbers_cindex", spy_wolbers)
    trainer = trainers["dynamic_deephit"]
    model = trainer.model
    pipeline.evaluate_models({"dynamic_deephit": trainer}, {"dynamic_deephit": model}, "val", 10, n_bootstrap=2, seed=0)

    ref = seen["bootstrap"]
    expected = pipeline.censoring_reference_from_train(trainer)
    _, train_gt = pipeline.split_ground_truth(trainer, "train")
    _, val_gt = pipeline.split_ground_truth(trainer, "val")
    for o in ("mortality", "malo", "cvd"):
        np.testing.assert_array_equal(ref[o]["time_years"], expected[o]["time_years"])
        np.testing.assert_array_equal(ref[o]["event_type"], expected[o]["event_type"])
        assert len(ref[o]["time_years"]) == int(train_gt[o]["at_risk"].sum())     # train split, at-risk rows
        assert len(ref[o]["time_years"]) != int(val_gt[o]["at_risk"].sum())       # not the evaluated split
    # ... and that exact object is what the metric functions receive
    assert seen["metric_refs"] and all(r is ref["malo"] or r is ref["cvd"] for r in seen["metric_refs"])


def test_at_risk_zero_examples_are_excluded_from_metrics():
    rng = np.random.default_rng(0)
    n = 150
    cif = np.sort(rng.uniform(0.01, 0.9, (n, 10)), axis=1)
    time_years = rng.uniform(0.5, 12, n)
    event_type = (rng.random(n) < 0.4).astype(int)
    gt = {o: {"time_years": time_years, "event_type": event_type, "at_risk": np.ones(n, bool)} for o in ("mortality", "malo", "cvd")}
    ref = {o: {"time_years": rng.uniform(0.5, 13, 400), "event_type": (rng.random(400) < 0.4).astype(int)} for o in gt}
    base = compute_all_metrics({o: cif for o in gt}, gt, ref, 10)

    # append 40 not-at-risk rows with adversarial values (early events, lowest predicted risk)
    m = 40
    junk_cif = np.zeros((m, 10))
    preds = {o: np.vstack([cif, junk_cif]) for o in gt}
    gt2 = {o: {"time_years": np.concatenate([time_years, np.full(m, 0.6)]),
               "event_type": np.concatenate([event_type, np.ones(m, int)]),
               "at_risk": np.concatenate([np.ones(n, bool), np.zeros(m, bool)])} for o in gt}
    masked = compute_all_metrics(preds, gt2, ref, 10)
    np.testing.assert_allclose(masked["value"].to_numpy(), base["value"].to_numpy())
    # sanity: if those rows WERE at risk, the metrics would move
    for o in gt2:
        gt2[o]["at_risk"] = np.ones(n + m, bool)
    assert not np.allclose(compute_all_metrics(preds, gt2, ref, 10)["value"].to_numpy(), base["value"].to_numpy())


def test_counts_use_at_risk_only(trained):
    _, trainers = trained
    _, gt = pipeline.split_ground_truth(trainers["dynamic_deephit"], "val")
    counts = pipeline.outcome_counts(gt, 10)
    for o in gt:
        assert counts[o]["n_at_risk"] == int(gt[o]["at_risk"].sum())
    assert counts["cvd"]["n_at_risk"] < len(gt["cvd"]["at_risk"])  # some cvd_at_risk == 0 patients excluded


def test_best_checkpoint_is_reloaded_not_last_epoch(tmp_path, monkeypatch):
    """checkpoint_selection: total (previous behaviour): one checkpoint_best.pt on val total."""
    schema, training = _configs()
    training = {**training, "training_loop": {**training["training_loop"], "checkpoint_selection": "total"}}
    visits, outcomes = generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    torch.manual_seed(0)
    trainer, model = evaluate_script.make_trainer("dynamic_deephit", visits, outcomes, schema, training)
    val_losses = iter([3.0, 1.0, 2.0, 5.0])  # best at epoch 1; later epochs are worse
    monkeypatch.setattr(trainer, "evaluate", lambda patients, resample_landmarks=False: {"total": next(val_losses)})

    history = trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=4, checkpoint_dir=str(tmp_path / "dynamic_deephit"))
    best = torch.load(tmp_path / "dynamic_deephit" / "checkpoint_best.pt", weights_only=False)
    latest = torch.load(tmp_path / "dynamic_deephit" / "checkpoint_latest.pt", weights_only=False)
    assert history["best_epoch"] == 1 and best["epoch"] == 1 and latest["epoch"] == 3
    state = model.state_dict()
    assert all(torch.equal(state[k], best["model_state_dict"][k]) for k in state)        # best weights in the model
    assert not all(torch.equal(state[k], latest["model_state_dict"][k]) for k in state)  # not the last epoch's

    # evaluate.py loads the best checkpoint too
    _, loaded = evaluate_script.load_trained("dynamic_deephit", str(tmp_path), visits, outcomes, schema, training)
    assert all(torch.equal(loaded.state_dict()[k], best["model_state_dict"][k]) for k in state)


# ---- split pinning ------------------------------------------------------------------------


def test_training_saves_split_and_fingerprint(trained):
    ckpt_dir, trainers = trained
    for name in MODELS:
        saved = json.loads((ckpt_dir / name / "split.json").read_text())
        t = trainers[name]
        assert saved["train"] == t.train_ids and saved["val"] == t.val_ids and saved["test"] == t.test_ids
        assert set(saved["fingerprint"]) == {"visit_rows", "patients", "patient_id_sha256"}
        assert saved["fingerprint"]["patients"] == N_PATIENTS
        assert isinstance(saved["seed"], int)


def test_evaluation_uses_the_saved_split(trained, tmp_path, monkeypatch):
    ckpt_dir, _ = trained
    import compass.data.preprocessing as preprocessing

    def _must_not_rederive(*a, **kw):
        raise AssertionError("evaluation re-derived the split instead of loading split.json")

    monkeypatch.setattr(preprocessing, "patient_level_split", _must_not_rederive)
    assert evaluate_script.main(_argv(ckpt_dir, tmp_path, "--models", "dynamic_deephit")) == 0
    saved = json.loads((ckpt_dir / "dynamic_deephit" / "split.json").read_text())
    results = json.loads((tmp_path / "metrics.json").read_text())
    assert results["n_patients"] == len(saved["val"])


def test_mismatched_dataset_refuses(trained, tmp_path, capsys):
    ckpt_dir, _ = trained
    argv = ["--checkpoint-dir", str(ckpt_dir), "--n-patients", str(N_PATIENTS + 1), "--data-seed", str(DATA_SEED),
            "--n-bootstrap", "5", "--out-dir", str(tmp_path)]
    assert evaluate_script.main(argv) == 3
    err = capsys.readouterr().err
    assert "does not match the data this run was trained on" in err and "patients: saved 300 vs given 301" in err
    assert not (tmp_path / "metrics.json").exists()
    # same size, different data (different seed) -> refused on the visit-row count / ID hash
    argv[5] = str(DATA_SEED)
    argv[3] = str(N_PATIENTS)
    argv += ["--data-seed", str(DATA_SEED + 1)]
    assert evaluate_script.main(argv) == 3

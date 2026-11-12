"""Serial evaluation (pre-specified secondary analysis 2): landmark window rule, LOCF snapshot,
full-history inputs, landmark Cox fitted on train only, checkpoints unchanged, test-split
refusal. Synthetic data only; the end-to-end tests need R (Fine-Gray) and are skipped without it."""

import functools
import importlib.util
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation import serial
from compass.models.cox_baseline import last_observed
from compass.training import train as train_module

N_PATIENTS, DATA_SEED, MALO_RATE = 400, 7, 0.10  # MALO raised so every prediction point has MALO events
MODELS = ["dynamic_deephit", "tracer", "survlatent_ode"]
YEAR = pd.Timedelta(days=365.25)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate_script = _load("evaluate_script", "scripts/evaluate.py")
serial_script = _load("serial_script", "scripts/serial_evaluate.py")
cohort = functools.partial(generate_synthetic_cohort, malo_rate=MALO_RATE)


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = "suppressMessages({library(survival); library(jsonlite)})"
    return subprocess.run(["Rscript", "-e", probe], capture_output=True).returncode == 0


needs_r = pytest.mark.skipif(not _r_available(), reason="Rscript with survival / jsonlite not installed")


# ---- unit tests ----------------------------------------------------------------------------------


def test_landmark_window_rule():
    index = pd.Timestamp("2015-01-01")
    start2, end2 = index + YEAR, index + 2 * YEAR
    visits = [index, index + pd.Timedelta(days=200), start2, start2 + pd.Timedelta(days=1),
              end2 - pd.Timedelta(days=30), end2]
    # k = 1: (index, index + 1 yr] -> the index visit itself is excluded, the boundary visit included
    assert serial.serial_landmark_date(visits, index, pd.NaT, 1) == start2
    # k = 2: the most recent visit in (index + 1 yr, index + 2 yr], end point included
    assert serial.serial_landmark_date(visits, index, pd.NaT, 2) == end2
    assert serial.serial_landmark_date(visits[:-1], index, pd.NaT, 2) == end2 - pd.Timedelta(days=30)
    # no visit in the window -> not evaluated
    assert serial.serial_landmark_date(visits, index, pd.NaT, 3) is None
    assert serial.serial_landmark_date([index, start2], index, pd.NaT, 2) is None  # start point excluded
    # alive at the landmark visit: death strictly after it
    assert serial.serial_landmark_date(visits, index, end2 + pd.Timedelta(days=1), 2) == end2
    assert serial.serial_landmark_date(visits, index, end2, 2) is None  # died on the landmark date
    assert serial.serial_landmark_date(visits, index, end2 - pd.Timedelta(days=10), 2) is None  # most recent visit after death


def test_snapshot_is_one_visit_with_locf_values():
    visits = pd.DataFrame({
        "patient_id": ["P1"] * 3,
        "visit_date": pd.to_datetime(["2015-01-01", "2016-01-01", "2017-01-01"]),
        "alt": [30.0, np.nan, np.nan],         # observed early only -> carried forward
        "ast": [np.nan, 25.0, 40.0],           # latest observed = the landmark visit's value
        "hba1c": [np.nan, np.nan, np.nan],     # never observed -> missing
        "metformin": [1.0, 0.0, np.nan],
        "sex": ["female", "female", "female"],
        "smoking_status": [2, np.nan, np.nan],
    })
    cols = ["alt", "ast", "hba1c", "metformin", "sex", "smoking_status"]
    snap = serial.locf_snapshot(visits, cols)
    assert len(snap) == 1 and snap["visit_date"].iloc[0] == pd.Timestamp("2017-01-01")
    assert snap["alt"].iloc[0] == 30.0 and snap["ast"].iloc[0] == 40.0 and np.isnan(snap["hba1c"].iloc[0])
    assert snap["metformin"].iloc[0] == 0.0 and snap["sex"].iloc[0] == "female" and snap["smoking_status"].iloc[0] == 2
    assert snap["alt"].dtype == visits["alt"].dtype


@pytest.fixture(scope="module")
def data_trainer():
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training = yaml.safe_load(f)
    visits, outcomes = cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    trainer, _ = evaluate_script.make_trainer(MODELS[0], visits, outcomes, schema, training)
    return trainer


def test_snapshot_inputs_equal_locf_of_full_history(data_trainer):
    """Every scaled snapshot example has exactly one visit, and its values / mask equal the
    most recent observed scaled value of the full history (last_observed)."""
    inputs = serial.serial_inputs(data_trainer, data_trainer.val_ids, 2)
    assert inputs["ids"] and len(inputs["full"]) == len(inputs["snapshot"]) == len(inputs["ids"])
    assert all(len(ex["input_visits"]) == 1 and len(ex["input_values"]) == 1 for ex in inputs["snapshot"])
    assert max(len(ex["input_visits"]) for ex in inputs["full"]) > 1
    for full, snap in zip(inputs["full"], inputs["snapshot"]):
        assert snap["landmark_date"] == full["landmark_date"] == full["input_visits"]["visit_date"].max()
        v, m = full["input_values"].to_numpy()[None], full["input_mask"].to_numpy()[None]
        locf = last_observed(v, m, np.array([v.shape[1]]))[0]
        observed = ~np.isnan(locf)
        np.testing.assert_array_equal(snap["input_mask"].to_numpy()[0], observed.astype(int))
        np.testing.assert_allclose(snap["input_values"].to_numpy()[0][observed], locf[observed], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(snap["input_values"].to_numpy()[0][~observed], 0.0)


def test_full_history_inputs_equal_what_training_builds(data_trainer, monkeypatch):
    """The full-history example at a prediction point is exactly the example the Trainer's own
    training path builds for a landmark at that date (same scalers, encoding, visits)."""
    k = 1
    inputs = serial.serial_inputs(data_trainer, data_trainer.train_ids, k)
    landmarks = {ex["patient_id"]: ex["landmark_date"] for ex in inputs["full"]}
    pending = list(inputs["ids"])
    monkeypatch.setattr(train_module, "sample_landmark", lambda visits, rng, **kw: landmarks[visits["patient_id"].iloc[0]])
    built = data_trainer._build_examples_for(pending, np.random.default_rng(0))
    for ours, theirs in zip(inputs["full"], built):
        assert ours["patient_id"] == theirs["patient_id"] and ours["landmark_date"] == theirs["landmark_date"]
        pd.testing.assert_frame_equal(ours["input_values"], theirs["input_values"])
        pd.testing.assert_frame_equal(ours["input_mask"], theirs["input_mask"])
        pd.testing.assert_frame_equal(ours["input_visits"], theirs["input_visits"])
    # and nothing dated after the landmark visit
    assert all((ex["input_visits"]["visit_date"] <= ex["landmark_date"]).all() for ex in inputs["full"])


def test_targets_come_from_build_targets_at_the_landmark(data_trainer):
    from compass.training.targets import build_targets

    inputs = serial.serial_inputs(data_trainer, data_trainer.val_ids, 1)
    gt = serial.serial_ground_truth(data_trainer, inputs["full"])
    t = build_targets(inputs["ids"], [ex["landmark_date"] for ex in inputs["full"]], data_trainer.outcomes_df,
                      data_trainer.last_visit_dates, data_trainer.administrative_cutoff, 1.0, 13)
    for o in ("mortality", "malo", "cvd"):
        np.testing.assert_array_equal(gt[o]["at_risk"], t[f"{o}_at_risk"].numpy().astype(bool))


# ---- end to end (synthetic final models) ---------------------------------------------------------


@pytest.fixture(scope="module")
def final_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("serial")
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training = yaml.safe_load(f)
    visits, outcomes = cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    for name in MODELS:
        torch.manual_seed(training["seed"])
        trainer, _ = evaluate_script.make_trainer(name, visits, outcomes, schema, training)
        trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=1, checkpoint_dir=str(root / "runs" / name))
    cfg = root / "final_models.yaml"
    cfg.write_text(yaml.safe_dump({"split_dir": str(root / "runs" / MODELS[0]),
                                   "models": {m: str(root / "runs" / m) for m in MODELS}}))
    return {"root": root, "cfg": cfg, "dirs": {m: root / "runs" / m for m in MODELS},
            "split": json.loads((root / "runs" / MODELS[0] / "split.json").read_text())}


def _argv(fm, out, *extra):
    return ["--final-models", str(fm["cfg"]), "--n-patients", str(N_PATIENTS), "--data-seed", str(DATA_SEED),
            "--n-bootstrap", "5", "--landmark-cox", str(fm["root"] / "landmark_cox.json"), "--out-dir", str(out), *extra]


@pytest.fixture(scope="module")
def val_run(final_models, tmp_path_factory):
    if not _r_available():
        pytest.skip("Rscript with survival / jsonlite not installed")
    mp = pytest.MonkeyPatch()
    mp.setattr(serial_script, "generate_synthetic_cohort", cohort)
    fit_calls = []
    original_fit = serial_script.fit_landmark_cox

    def spy_fit(trainer, examples, *a, **kw):
        fit_calls.append({"ids": [ex["patient_id"] for ex in examples], "trainer_val": list(trainer.val_ids),
                          "trainer_test": list(trainer.test_ids)})
        return original_fit(trainer, examples, *a, **kw)

    mp.setattr(serial_script, "fit_landmark_cox", spy_fit)
    hashes_before = serial.checkpoint_hashes(final_models["dirs"])
    out = tmp_path_factory.mktemp("serial_val")
    assert serial_script.main(_argv(final_models, out)) == 0
    mp.undo()
    return {"out": out, "fit_calls": fit_calls, "hashes_before": hashes_before,
            "results": json.loads((out / "metrics.json").read_text())}


@needs_r
def test_serial_val_run_reports_all_seven_predictors_and_both_comparisons(val_run):
    res = val_run["results"]
    assert res["predictors"][0] == "landmark_cox" and len(res["predictors"]) == 7
    for k in ("1", "2", "3"):
        p = res["points"][k]
        assert p["ipcw_reference"]["split"] == "train" and p["ipcw_reference"]["prediction_point"] == int(k)
        assert {r["model"] for r in p["per_model"]} == set(res["predictors"])
        comparisons = {r["comparison"] for r in p["pairwise"]}
        assert len(comparisons) == 21
        for m in MODELS:
            assert f"{m}_full_minus_landmark_cox" in comparisons and f"{m}_full_minus_{m}_snapshot" in comparisons
        assert all(r["horizon"] == 5 for r in p["per_model"])
        assert set(p["calibration"]) == set(res["predictors"])
        assert 0 < p["ipcw"]["mortality"]["G_at_horizon"] <= 1 and p["ipcw"]["mortality"]["max_ipcw_weight"] >= 1
        assert p["history_visits"]["median"] >= 1
    assert (val_run["out"] / "serial_tables.md").exists()


@needs_r
def test_landmark_cox_is_fitted_on_train_only(val_run, final_models):
    calls = val_run["fit_calls"]
    assert len(calls) == 3  # one fit per prediction point
    train = set(final_models["split"]["train"])
    for c in calls:
        assert c["ids"] and set(c["ids"]) <= train
        assert c["trainer_test"] == []  # the validation run never landmarks a test patient
    fits = json.loads((final_models["root"] / "landmark_cox.json").read_text())
    assert fits["fitted_on_split"] == "train" and set(fits["fits"]) == {"1", "2", "3"}
    assert all(f["grid_years"] == [1.0, 2.0, 3.0, 4.0, 5.0] for f in fits["fits"].values())


@needs_r
def test_deep_checkpoints_unchanged(val_run, final_models):
    assert serial.checkpoint_hashes(final_models["dirs"]) == val_run["hashes_before"]
    assert val_run["results"]["checkpoint_sha256"] == val_run["hashes_before"]
    assert val_run["results"]["checkpoints_unchanged"] is True


@needs_r
def test_test_split_refuses_without_final_and_never_refits(val_run, final_models, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serial_script, "generate_synthetic_cohort", cohort)
    assert serial_script.main(_argv(final_models, tmp_path / "a", "--split", "test")) == 2
    assert "requires --final" in capsys.readouterr().err
    missing = _argv(final_models, tmp_path / "b", "--split", "test", "--final")
    missing[missing.index("--landmark-cox") + 1] = str(tmp_path / "none.json")
    assert serial_script.main(missing) == 3
    sha = serial.file_sha256(final_models["root"] / "landmark_cox.json")

    def no_fit(*a, **kw):
        raise AssertionError("the test run must not fit the landmark Cox")

    monkeypatch.setattr(serial_script, "fit_landmark_cox", no_fit)
    assert serial_script.main(_argv(final_models, tmp_path / "c", "--split", "test", "--final")) == 0
    res = json.loads((tmp_path / "c" / "metrics.json").read_text())
    assert res["split"] == "test" and res["landmark_cox"]["fitted_in_this_run"] is False
    assert res["landmark_cox"]["sha256"] == sha == val_run["results"]["landmark_cox"]["sha256"]
    assert serial.file_sha256(final_models["root"] / "landmark_cox.json") == sha


def test_fit_landmark_cox_uses_only_given_examples_and_truncates_at_horizon(data_trainer):
    """The fit sees the given (training) examples only, with follow-up truncated at 5 yr."""
    inputs = serial.serial_inputs(data_trainer, data_trainer.train_ids, 1)
    seen = {}

    def fake_fit(x, names, gt, penalizer, num_bins):
        seen.update(n=len(x), max_time=max(float(g["time_years"].max()) for g in gt.values()), bins=num_bins,
                    penalizer=penalizer)
        return {}

    serial.fit_landmark_cox(data_trainer, inputs["snapshot"], 0.01, fit_fn=fake_fit)
    assert seen == {"n": len(inputs["ids"]), "max_time": pytest.approx(5.0), "bins": 5, "penalizer": 0.01}

"""Tests for the classical baseline (compass.models.cox_baseline, scripts/fit_baseline.py) and its
evaluation as model "cox_baseline" (scripts/evaluate.py --baseline). Synthetic data only; the R
tests are skipped when Rscript / survival / cmprsk are not installed."""

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
from compass.models.cox_baseline import (
    CoxBaselineModel,
    apply_feature_transform,
    fit_cox,
    fit_feature_transform,
    fit_fine_gray,
    last_observed,
    predict_risk_from_features,
)

N_PATIENTS, DATA_SEED = 600, 42


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = 'q(status = !all(sapply(c("survival", "cmprsk", "jsonlite"), requireNamespace, quietly = TRUE)))'
    return subprocess.run(["Rscript", "-e", probe], capture_output=True).returncode == 0


needs_r = pytest.mark.skipif(not _r_available(), reason="Rscript with survival / cmprsk / jsonlite not installed")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate_script = _load("evaluate_script", "scripts/evaluate.py")
fit_script = _load("fit_baseline_script", "scripts/fit_baseline.py")


# ---- features -----------------------------------------------------------------------------------


def test_last_observed_takes_the_most_recent_observed_visit_within_the_sequence():
    values = np.array([[[1.0, 10.0], [2.0, 0.0], [3.0, 0.0], [9.0, 9.0]]])  # visit 4 is padding
    mask = np.array([[[1, 1], [1, 0], [0, 0], [1, 1]]])
    out = last_observed(values, mask, np.array([3]))
    np.testing.assert_array_equal(out, [[2.0, 10.0]])  # feature 0: visit 2; feature 1: visit 1; padding ignored
    assert np.isnan(last_observed(values, np.zeros_like(mask), np.array([3]))).all()


def test_missing_fill_uses_training_means_with_indicators():
    x_train = np.array([[1.0, 5.0, 7.0, 2.0], [3.0, np.nan, 7.0, 4.0], [5.0, 9.0, 7.0, np.nan], [7.0, 7.0, 7.0, 6.0]])
    names = ["a", "b", "const", "d"]
    t = fit_feature_transform(x_train, names)
    assert t["indicators"] == ["b", "d"]                    # only features with missingness in TRAINING
    assert t["dropped_zero_variance"] == ["const"]          # zero variance in training -> dropped
    assert t["fill_means"] == pytest.approx([4.0, 7.0, 7.0, 4.0])
    x_other = np.array([[np.nan, np.nan, 7.0, 1.0], [100.0, 100.0, 7.0, np.nan]])
    z = apply_feature_transform(x_other, t)
    cols = t["columns"]
    assert cols == ["a", "b", "d", "b_missing", "d_missing"]
    center, scale = np.array(t["center"]), np.array(t["scale"])
    raw = z * scale + center
    assert raw[0, cols.index("a")] == pytest.approx(4.0)    # train mean of a, not the other split's
    assert raw[0, cols.index("b")] == pytest.approx(7.0) and raw[0, cols.index("b_missing")] == 1.0
    assert raw[1, cols.index("d")] == pytest.approx(4.0) and raw[1, cols.index("d_missing")] == 1.0
    # standardisation uses training statistics (ddof = 1), never the transformed split's own
    z_train = apply_feature_transform(x_train, t)
    np.testing.assert_allclose(z_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z_train.std(axis=0, ddof=1), 1)
    a_train = np.array([1.0, 3.0, 5.0, 7.0])
    assert center[0] == pytest.approx(a_train.mean()) and scale[0] == pytest.approx(a_train.std(ddof=1))


def test_feature_never_observed_in_training_is_refused():
    with pytest.raises(ValueError, match="never observed"):
        fit_feature_transform(np.array([[1.0, np.nan], [2.0, np.nan]]), ["a", "b"])


# ---- model fits ---------------------------------------------------------------------------------


def test_cox_prediction_matches_lifelines():
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 3)) + np.array([0.5, -0.2, 0.1])
    time = rng.exponential(1 / (0.1 * np.exp(z @ np.array([0.5, -0.3, 0.0]))))
    event = rng.random(400) < 0.7
    fit = fit_cox(z, time, event, penalizer=0.01, grid_years=[1.0, 2.0, 3.0])
    df = pd.DataFrame(z, columns=["z0", "z1", "z2"]).assign(duration=time, event=event.astype(int))
    cph = CoxPHFitter(penalizer=0.01).fit(df, "duration", "event")
    z_new = rng.normal(size=(5, 3))
    # lifelines' step function (predict_survival_function(times=...) interpolates between event times)
    surv = cph.predict_survival_function(pd.DataFrame(z_new, columns=["z0", "z1", "z2"]))
    expected = np.stack([1 - surv[surv.index <= t].iloc[-1].to_numpy() for t in (1, 2, 3)], axis=1)
    full = {"grid_years": [1.0, 2.0, 3.0], "outcomes": {o: fit for o in ("mortality", "malo", "cvd")}}
    got = predict_risk_from_features(z_new, full, [1, 2, 3])[:, 0, :]
    np.testing.assert_allclose(got, expected, atol=1e-10)


@needs_r
def test_fine_gray_cif_matches_crr_reference():
    data = pd.read_csv("tests/fixtures/finegray_reference_data.csv")
    ref = pd.read_csv("tests/fixtures/finegray_crr_reference.csv")
    z = data[["x1", "x2"]].to_numpy()
    fit = fit_fine_gray(z, data["time"].to_numpy(), data["status"].to_numpy(), penalizer=0.0,
                        grid_years=[1.0, 2.0, 3.0, 4.0, 5.0])
    full = {"grid_years": [1.0, 2.0, 3.0, 4.0, 5.0], "outcomes": {o: fit for o in ("mortality", "malo", "cvd")}}
    for _, r in ref.iterrows():
        got = predict_risk_from_features(np.array([[r["x1"], r["x2"]]]), full, [r["t"]])[0, 1, 0]
        assert got == pytest.approx(r["cif"], abs=1e-5), (r["t"], r["x1"], r["x2"])
    assert fit["n_events"] == int((data["status"] == 1).sum()) and fit["warnings"] == []


@needs_r
def test_fine_gray_without_covariate_effect_equals_aalen_johansen():
    from compass.evaluation.calibration import aalen_johansen_cif

    data = pd.read_csv("tests/fixtures/finegray_reference_data.csv")
    z = np.zeros((len(data), 1))
    z[0, 0] = 1e-9  # a covariate with no information: beta -> 0
    # the Fine-Gray CIF 1 - exp(-Lambda) then approximates the Aalen-Johansen product-limit CIF
    # 1 - prod(1 - dLambda) (equal only to first order in dLambda), so close but not identical
    fit = fit_fine_gray(z, data["time"].to_numpy(), data["status"].to_numpy(), penalizer=1.0, grid_years=[1.0, 3.0, 5.0])
    full = {"grid_years": [1.0, 3.0, 5.0], "outcomes": {o: fit for o in ("mortality", "malo", "cvd")}}
    for t in (1.0, 3.0, 5.0):
        got = predict_risk_from_features(np.zeros((1, 1)), full, [t])[0, 1, 0]
        assert got == pytest.approx(aalen_johansen_cif(data["time"], data["status"], t), abs=2e-3)


@needs_r
def test_r_ridge_theta_equals_the_lifelines_penalty():
    """R coxph ridge(theta = 0.01 * n, scale = FALSE) and lifelines penalizer = 0.01 maximise the
    same penalised partial likelihood: the same coefficients."""
    data = pd.read_csv("tests/fixtures/finegray_reference_data.csv")
    z = data[["x1", "x2"]].to_numpy()
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    event = (data["status"] == 1).to_numpy()
    py = fit_cox(z, data["time"].to_numpy(), event, penalizer=0.01, grid_years=[1.0])["coefficients"]
    script = (
        "suppressMessages(library(survival)); d <- read.csv('{f}');"
        "f <- coxph(Surv(time, event) ~ ridge(z0, z1, theta = {theta}, scale = FALSE), data = d, ties = 'efron');"
        "cat(sprintf('%.12f', coef(f)), sep = ',')"
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = f"{tmp}/d.csv"
        pd.DataFrame({"time": data["time"], "event": event.astype(int), "z0": z[:, 0], "z1": z[:, 1]}).to_csv(f, index=False)
        out = subprocess.run(["Rscript", "-e", script.format(f=f, theta=0.01 * len(data))], capture_output=True, text=True)
    r_beta = [float(v) for v in out.stdout.strip().split(",")]
    np.testing.assert_allclose(py, r_beta, atol=1e-5)


# ---- fit on train only + evaluation -------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """One 1-epoch deep model (gives the pinned split + locked settings), then the baseline fit."""
    root = tmp_path_factory.mktemp("baseline")
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training = yaml.safe_load(f)
    visits, outcomes = generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    torch.manual_seed(training["seed"])
    trainer, _ = evaluate_script.make_trainer("dynamic_deephit", visits, outcomes, schema, training)
    trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=1, checkpoint_dir=str(root / "runs" / "dynamic_deephit"))
    final_yaml = root / "final_models.yaml"
    final_yaml.write_text(yaml.safe_dump({"split_dir": str(root / "runs" / "dynamic_deephit"),
                                          "models": {"dynamic_deephit": str(root / "runs" / "dynamic_deephit")}}))
    return {"root": root, "final": final_yaml, "split": json.loads((root / "runs" / "dynamic_deephit" / "split.json").read_text())}


def _fit_argv(f, out):
    return ["--final-models", str(f["final"]), "--n-patients", str(N_PATIENTS), "--data-seed", str(DATA_SEED),
            "--out", str(out)]


@needs_r
def test_baseline_is_fitted_on_the_training_split_only(fitted, tmp_path, monkeypatch):
    from compass.training.train import Trainer

    seen_splits, trainer_splits = [], []
    real_fixed, real_init = Trainer.fixed_examples, Trainer.__init__

    def spy_fixed(self, which):
        seen_splits.append(which)
        return real_fixed(self, which)

    def spy_init(self, *a, split=None, **kw):
        trainer_splits.append(split)
        return real_init(self, *a, split=split, **kw)

    monkeypatch.setattr(Trainer, "fixed_examples", spy_fixed)
    monkeypatch.setattr(Trainer, "__init__", spy_init)
    out = tmp_path / "cox_baseline.json"
    assert fit_script.main(_fit_argv(fitted, out)) == 0
    assert seen_splits and set(seen_splits) == {"train"}                    # only training examples requested
    assert len(trainer_splits) == 1 and trainer_splits[0]["val"] == [] and trainer_splits[0]["test"] == []
    assert trainer_splits[0]["train"] == fitted["split"]["train"]
    fit = json.loads(out.read_text())
    assert fit["fitted_on_split"] == "train" and fit["n_train_patients"] == len(fitted["split"]["train"])
    assert fit["outcomes"]["mortality"]["n"] == len(fitted["split"]["train"])  # all training patients at risk
    assert fit["outcomes"]["cvd"]["n"] < fit["outcomes"]["mortality"]["n"]     # CVD: cvd_at_risk == 1 only
    assert "Fine-Gray" in fit["outcomes"]["malo"]["method"] and "Cox" in fit["outcomes"]["mortality"]["method"]


@pytest.fixture(scope="module")
def baseline_fit(fitted):
    if not _r_available():
        pytest.skip("Rscript with survival / cmprsk / jsonlite not installed")
    out = fitted["root"] / "cox_baseline.json"
    assert fit_script.main(_fit_argv(fitted, out)) == 0
    return out


def _eval_argv(f, fit_path, out, *extra):
    return ["--final-models", str(f["final"]), "--baseline", str(fit_path), "--n-patients", str(N_PATIENTS),
            "--data-seed", str(DATA_SEED), "--n-bootstrap", "10", "--out-dir", str(out), *extra]


def test_pairwise_bootstrap_includes_the_baseline(fitted, baseline_fit, tmp_path):
    assert evaluate_script.main(_eval_argv(fitted, baseline_fit, tmp_path, "--skip-banding")) == 0
    res = json.loads((tmp_path / "metrics.json").read_text())
    assert {r["model"] for r in res["per_model"]} == {"cox_baseline", "dynamic_deephit"}
    assert {r["comparison"] for r in res["pairwise"]} == {"dynamic_deephit_minus_cox_baseline"}
    assert {(r["outcome"], r["metric_name"]) for r in res["pairwise"]} == {
        (o, m) for o in ("mortality", "malo", "cvd") for m in ("cindex", "ibs")}
    assert "cox_baseline" in res["calibration"] and res["banding"] == {"skipped": True}
    assert res["model_sources"]["cox_baseline"].endswith("cox_baseline.json")
    assert not (tmp_path / "cutpoints.json").exists()


def test_deep_model_results_unchanged_by_adding_the_baseline(fitted, baseline_fit, tmp_path):
    common = ["--final-models", str(fitted["final"]), "--n-patients", str(N_PATIENTS), "--data-seed", str(DATA_SEED),
              "--n-bootstrap", "10", "--skip-banding"]
    assert evaluate_script.main(common + ["--out-dir", str(tmp_path / "deep")]) == 0
    assert evaluate_script.main(common + ["--baseline", str(baseline_fit), "--out-dir", str(tmp_path / "both")]) == 0
    deep = json.loads((tmp_path / "deep" / "metrics.json").read_text())
    both = json.loads((tmp_path / "both" / "metrics.json").read_text())
    rows = lambda r: [x for x in r["per_model"] if x["model"] == "dynamic_deephit"]
    assert rows(deep) == rows(both)  # the shared resample makes the deep model's numbers identical


def test_baseline_is_never_banded(fitted, baseline_fit, tmp_path):
    assert evaluate_script.main(_eval_argv(fitted, baseline_fit, tmp_path)) == 0
    saved = json.loads((tmp_path / "cutpoints.json").read_text())
    assert set(saved["models"]) == {"dynamic_deephit"}


def test_baseline_test_split_needs_final_but_not_cutpoints_when_banding_is_skipped(fitted, baseline_fit, tmp_path):
    assert evaluate_script.main(_eval_argv(fitted, baseline_fit, tmp_path / "a", "--split", "test", "--skip-banding")) == 2
    assert evaluate_script.main(_eval_argv(fitted, baseline_fit, tmp_path / "b", "--split", "test", "--final",
                                           "--skip-banding")) == 0
    res = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert res["split"] == "test" and "cox_baseline" in {r["model"] for r in res["per_model"]}


def test_baseline_from_another_split_is_refused(fitted, baseline_fit, tmp_path, capsys):
    fit = json.loads(baseline_fit.read_text())
    fit["split_seed"] += 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(fit))
    assert evaluate_script.main(_eval_argv(fitted, bad, tmp_path / "out", "--skip-banding")) == 3
    assert "different split seed" in capsys.readouterr().err


def test_baseline_adapter_predicts_from_batches(fitted, baseline_fit):
    fit = json.loads(baseline_fit.read_text())
    model = CoxBaselineModel(fit)
    values = torch.zeros(2, 3, len(fit["transform"]["feature_names"]))
    mask = torch.ones_like(values)
    risk = model.predict_risk(values, mask, torch.tensor([3, 2]), list(range(1, 11)))
    assert risk.shape == (2, 3, 10) and torch.all(risk[:, :, 1:] >= risk[:, :, :-1]) and torch.all((risk >= 0) & (risk <= 1))

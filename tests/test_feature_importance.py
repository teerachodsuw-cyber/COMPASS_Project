"""compass.evaluation.importance on synthetic data: a signal variable gives a clear C-index drop,
an irrelevant one ~0; values and masks move together; a group uses one shared permutation; the
unpermuted C-index equals the stored value. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from compass.data.preprocessing import build_feature_names
from compass.evaluation import importance as imp
from compass.evaluation.metrics import compute_all_metrics
from compass.models.base import OUTCOME_NAMES

COLS = ["signal", "noise", "a", "b"]


def _examples(n, rng, lengths=None):
    """Landmark-example stand-ins: input_values / input_mask DataFrames over COLS."""
    out = []
    for i in range(n):
        L = 1 if lengths is None else lengths[i]
        vals = pd.DataFrame(rng.normal(size=(L, len(COLS))), columns=COLS)
        mask = pd.DataFrame((rng.random((L, len(COLS))) > 0.2).astype(float), columns=COLS)
        vals[mask == 0] = 0.0  # the pipeline stores missing entries as 0 with mask 0
        out.append({"patient_id": i, "input_values": vals, "input_mask": mask})
    return out


def _last(ex, col):
    m = ex["input_mask"][col].to_numpy() > 0
    return float(ex["input_values"][col].to_numpy()[m][-1]) if m.any() else np.nan


def _ground_truth(signal, rng):
    """Outcomes driven by `signal` only (MALO / CVD with a competing death)."""
    n = len(signal)
    s = np.nan_to_num(signal)
    gt = {}
    for name in OUTCOME_NAMES:
        t_event = rng.exponential(8 * np.exp(-1.2 * s))
        t_cens = rng.uniform(2, 14, n)
        t_comp = rng.exponential(25, n) if name != "mortality" else np.full(n, np.inf)
        t = np.minimum.reduce([t_event, t_cens, t_comp])
        et = np.where(t == t_event, 1, np.where(t == t_comp, 2, 0))
        gt[name] = {"time_years": t, "event_type": et.astype(int), "at_risk": np.ones(n, bool)}
    return gt


def _reference(gt):
    return {k: {"time_years": g["time_years"], "event_type": g["event_type"]} for k, g in gt.items()}


def _risk(score, horizon):
    p = 1 / (1 + np.exp(-score))
    return {name: np.repeat(p[:, None], horizon, axis=1) * np.linspace(0.2, 1, horizon)[None, :] for name in OUTCOME_NAMES}


def _predictors(examples, horizon):
    """A 'deep-style' predictor on the examples and a 'Cox-style' one on the last-observed matrix;
    both use signal (relevant) and noise (irrelevant to the outcome, but used by the model)."""
    x = np.array([[_last(ex, c) for c in COLS] for ex in examples])

    def deep(columns, perm):
        exs = examples if columns is None else imp.permute_examples(examples, columns, perm)
        sig = np.nan_to_num([_last(e, "signal") for e in exs])
        noi = np.nan_to_num([_last(e, "noise") for e in exs])
        return _risk(1.5 * sig + 0.3 * noi, horizon)

    def cox(columns, perm):
        xx = x if columns is None else imp.permute_features(x, COLS, columns, perm)
        return _risk(1.5 * np.nan_to_num(xx[:, 0]) + 0.3 * np.nan_to_num(xx[:, 1]), horizon)

    return x, {"deep": deep, "cox": cox}


def test_signal_variable_drops_c_index_and_irrelevant_variable_is_near_zero():
    rng = np.random.default_rng(0)
    examples = _examples(800, rng)
    x, predictors = _predictors(examples, 10)
    gt = _ground_truth(x[:, 0], rng)
    units = [{"kind": "variable", "name": "signal", "label": "signal", "columns": ["signal"]},
             {"kind": "variable", "name": "noise", "label": "noise", "columns": ["noise"]}]
    base, rows = imp.run_importance(predictors, len(examples), units, gt, _reference(gt), 10, n_repeats=8, seed=1)
    by = {(r["unit"], r["predictor"], r["outcome"]): r for r in rows}
    for pred in predictors:
        for o in OUTCOME_NAMES:
            assert by[("signal", pred, o)]["mean_drop"] > 0.08, (pred, o)
            assert by[("signal", pred, o)]["p2_5"] > 0.03
            assert abs(by[("noise", pred, o)]["mean_drop"]) < 0.02, (pred, o)
            r = by[("signal", pred, o)]
            assert r["p2_5"] <= r["mean_drop"] <= r["p97_5"] and len(r["drops"]) == 8
            assert r["c_unpermuted"] == base[(pred, o)]


def test_value_and_mask_move_together_aligned_from_the_landmark_visit():
    rng = np.random.default_rng(1)
    lengths = [1, 2, 3, 1, 2]
    examples = _examples(5, rng, lengths)
    perm = np.array([4, 2, 3, 1, 0])
    out = imp.permute_examples(examples, ["a"], perm)
    for i, (ex, new) in enumerate(zip(examples, out)):
        donor = examples[perm[i]]
        n_own, n_don = len(ex["input_values"]), len(donor["input_values"])
        k = min(n_own, n_don)
        for col_kind in ("input_values", "input_mask"):
            got = new[col_kind]["a"].to_numpy()
            assert np.array_equal(got[n_own - k:], donor[col_kind]["a"].to_numpy()[n_don - k:])  # donor's most recent rows
            assert np.all(got[: n_own - k] == 0)  # rows the donor lacks -> missing (0, mask 0)
            for c in ("signal", "noise", "b"):  # nothing else changes
                assert new[col_kind][c].equals(ex[col_kind][c])
        assert len(new["input_values"]) == n_own
    assert all(e["input_values"].equals(f["input_values"]) for e, f in zip(examples, _examples(5, np.random.default_rng(1), lengths)))

    x = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 6.0]])  # Cox: NaN (-> missing indicator) moves with its column
    moved = imp.permute_features(x, ["p", "q"], ["q"], np.array([1, 0, 2]))
    assert np.array_equal(moved[:, 0], x[:, 0], equal_nan=True)
    assert np.array_equal(moved[:, 1], np.array([5.0, np.nan, 6.0]), equal_nan=True)


def test_group_uses_one_shared_permutation_and_every_predictor_gets_the_same_one():
    rng = np.random.default_rng(2)
    examples = _examples(50, rng)
    for i, ex in enumerate(examples):  # a == b == patient code, always observed
        ex["input_values"].loc[:, ["a", "b"]] = float(i)
        ex["input_mask"].loc[:, ["a", "b"]] = 1.0
    perm = imp.permutation_for(50, seed=3, unit_index=0, repeat=0)
    out = imp.permute_examples(examples, ["a", "b"], perm)
    for i, ex in enumerate(out):
        assert ex["input_values"]["a"].iloc[-1] == ex["input_values"]["b"].iloc[-1] == float(perm[i])

    seen = []
    x, predictors = _predictors(examples, 10)
    gt = _ground_truth(x[:, 0], rng)

    def spy(name):
        def fn(columns, p):
            if columns is not None:
                seen.append((name, tuple(columns), tuple(p)))
            return predictors[name](columns, p)
        return fn

    units = [{"kind": "group", "name": "ab", "label": "ab", "columns": ["a", "b"]}]
    imp.run_importance({n: spy(n) for n in predictors}, 50, units, gt, _reference(gt), 10, n_repeats=3, seed=3)
    assert len(seen) == 2 * 3 and all(cols == ("a", "b") for _, cols, _ in seen)
    for r in range(3):  # per repeat: one permutation, shared by both predictors, = permutation_for(seed, unit, repeat)
        perms = {p for _, _, p in seen[2 * r: 2 * r + 2]}
        assert perms == {tuple(imp.permutation_for(50, 3, 0, r))}


def test_unpermuted_c_index_equals_the_stored_value():
    rng = np.random.default_rng(4)
    examples = _examples(300, rng)
    x, predictors = _predictors(examples, 10)
    gt = _ground_truth(x[:, 0], rng)
    ref = _reference(gt)
    stored = {}
    for name, fn in predictors.items():
        df = compute_all_metrics(fn(None, None), gt, ref, 10)  # what the primary evaluation stores
        stored.update({(name, r.outcome): r.value for r in df[df.metric_name == "cindex"].itertuples()})
    observed = {(name, o): c for name, fn in predictors.items() for o, c in imp.c_indices(fn(None, None), gt, ref, 10).items()}
    assert imp.check_unpermuted(observed, stored) == []
    bumped = {**stored, ("deep", "malo"): stored[("deep", "malo")] + 1e-15}
    assert imp.check_unpermuted(observed, bumped) == [f"deep / malo: {observed[('deep', 'malo')]!r} vs locked {bumped[('deep', 'malo')]!r}"]


def test_real_feature_layout_gives_34_variables_and_9_groups():
    import yaml
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    names = build_feature_names([v["name"] for v in schema["continuous_variables"]], [m["name"] for m in schema["medication_flags"]],
                                ("male", "female"), [0, 1, 2])
    cols = imp.variable_columns(names)
    assert len(names) == 36 and len(cols) == 34
    assert cols["smoking"] == ["smoking_status_0", "smoking_status_1", "smoking_status_2"] and cols["sex"] == ["sex_female"]
    units = imp.importance_units(names)
    groups = [u for u in units if u["kind"] == "group"]
    assert len(groups) == 9 and sorted(c for g in groups for c in g["columns"]) == sorted(names)  # groups cover every column once


def test_hazard_ratio_rescaling_and_ci():
    """HR per 1 SD of the observed values (continuous) / present vs absent (binary) from standardised
    design coefficients; CI from the given variance."""
    rng = np.random.default_rng(5)
    x = np.column_stack([rng.normal(0, 2, 400), rng.integers(0, 2, 400).astype(float)])
    x[:20, 0] = np.nan
    fit = {"transform": {"feature_names": ["cont", "bin"], "fill_means": [float(np.nanmean(x[:, 0])), float(x[:, 1].mean())],
                         "indicators": ["cont"], "_keep": [True, True, True], "columns": ["cont", "bin", "cont_missing"],
                         "center": [0.0, 0.0, 0.0], "scale": [1.8, 0.5, 0.2]},
           "outcomes": {o: {"coefficients": [0.36, 0.1, -0.04], "penalizer": 0.01} for o in OUTCOME_NAMES}}
    gt = {o: {"time_years": rng.uniform(1, 5, 400), "event_type": rng.integers(0, 2, 400), "at_risk": np.ones(400, bool)}
          for o in OUTCOME_NAMES}
    rows = imp.cox_hazard_ratios(fit, x, gt, {"cont": 9.0}, variance_fn=lambda *a, **k: np.diag([0.01, 0.04, 0.09]))
    r = {(q["outcome"], q["column"]): q for q in rows}
    sd = float(np.nanstd(x[:, 0], ddof=1))
    assert r[("mortality", "cont")]["hr"] == pytest.approx(np.exp(0.36 * sd / 1.8))
    assert r[("mortality", "cont")]["hr_ci_upper"] == pytest.approx(np.exp((0.36 + 1.959963984540054 * 0.1) * sd / 1.8))
    assert r[("mortality", "bin")]["hr"] == pytest.approx(np.exp(0.1 / 0.5)) and r[("mortality", "bin")]["kind"] == "binary"
    assert r[("malo", "cont_missing")]["contrast"] == "missing vs observed"
    assert r[("malo", "cont")]["type"].startswith("subdistribution")


def test_variance_at_stored_coefficients_matches_lifelines_and_keeps_them():
    """hr_variance.R at a lifelines-fitted beta (same ridge penalty): coefficients returned unchanged,
    standard errors agree with lifelines' own (requires R, like the Fine-Gray tests)."""
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=(n, 3))
    t, c = rng.exponential(np.exp(-z @ np.array([0.5, -0.3, 0.0]))), rng.uniform(0, 2, n)
    time, event = np.minimum(t, c), (t <= c).astype(int)
    df = pd.DataFrame(z, columns=["a", "b", "c"]).assign(T=time, E=event)
    cph = CoxPHFitter(penalizer=0.01).fit(df, "T", "E")
    beta = cph.params_[["a", "b", "c"]].to_numpy()
    var = imp.coefficient_variance(z, time, event, beta, 0.01, "cox")
    assert np.sqrt(np.diag(var)) == pytest.approx(cph.standard_errors_.to_numpy(), rel=2e-3)
    with pytest.raises(RuntimeError, match="different coefficients"):  # the guard: R must not move the estimates
        imp.coefficient_variance(z, time, event, beta, 0.01, "cox", rscript=str(_moving_rscript()))


def _moving_rscript():
    """A stand-in Rscript that runs the real script and then perturbs the returned coefficients."""
    import os
    import stat
    import sys
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "Rscript")
    with open(path, "w") as f:
        f.write(f"""#!{sys.executable}
import json, subprocess, sys
r = subprocess.run(["Rscript", *sys.argv[1:]])
out = sys.argv[4]
d = json.load(open(out)); d["coefficients"] = [b + 1e-6 for b in d["coefficients"]]
json.dump(d, open(out, "w"))
sys.exit(r.returncode)
""")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def _feature_importance_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "feature_importance.py"
    spec = importlib.util.spec_from_file_location("feature_importance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_table_s1_drops_smoking_adds_exposure_and_flags_sparse_medications(tmp_path):
    """Table S1: no smoking-indicator or missing-indicator rows; medication rows carry n (%) exposed
    in training; < 1% exposure is marked and named in the footnote."""
    fi = _feature_importance_script()
    names = ["index_age", "smoking_status_0", "smoking_status_1", "metformin", "arni", "sex_female"]
    x = np.zeros((1000, len(names)))
    x[:400, names.index("metformin")] = 1
    x[:5, names.index("arni")] = 1
    full = names + [m for m in fi.MEDICATIONS if m not in names]
    x = np.column_stack([x, np.ones((1000, len(full) - len(names)))])
    exposure = fi.medication_exposure(x, full)
    assert exposure["metformin"] == (400, 1000) and exposure["arni"] == (5, 1000)
    assert fi.sparse_medications(exposure) == ["arni"]

    def hr(col, kind, contrast):
        return [{"outcome": o, "column": col, "kind": kind, "contrast": contrast, "hr": 1.5,
                 "hr_ci_lower": 1.1, "hr_ci_upper": 2.0} for o in OUTCOME_NAMES]
    hr_rows = [*hr("index_age", "continuous", "per 1 SD (13.2)"), *hr("smoking_status_0", "binary", "present vs absent"),
               *hr("smoking_status_1", "binary", "present vs absent"), *hr("metformin", "binary", "present vs absent"),
               *hr("arni", "binary", "present vs absent"), *hr("sex_female", "binary", "present vs absent"),
               *hr("index_age_missing", "missing indicator", "missing vs observed")]
    rows = fi.table_s1_rows(hr_rows, {"index_age": "years"}, exposure)
    labels = [r["variable"] for r in rows]
    assert labels == ["Age, per 1 SD (13.2 years)", "Metformin (yes vs no)", "ARNI (yes vs no)†", "Female vs male"]
    by = {r["variable"]: r for r in rows}
    assert by["Metformin (yes vs no)"]["exposed"] == "400 (40.0)"
    assert by["ARNI (yes vs no)†"]["exposed"] == "5 (0.5)"
    assert by["Female vs male"]["exposed"] == "" and by["Age, per 1 SD (13.2 years)"]["exposed"] == ""
    notes = fi.s1_footnotes(exposure)
    assert notes[0] == fi.S1_MODEL_NOTE and notes[1] == fi.S1_SMOKING_NOTE
    assert notes[2] == "† Sparse exposure (< 1% of the training split: ARNI); estimates are unstable."
    assert "n = 1,000" in notes[3]
    fi.write_table_s1(rows, exposure, tmp_path)
    csv = pd.read_csv(tmp_path / "table_s1.csv", keep_default_na=False)
    assert list(csv.columns) == list(fi.S1_COLUMNS.values()) and len(csv) == 4
    assert (tmp_path / "table_s1.docx").exists()

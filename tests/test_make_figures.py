"""End-to-end test of scripts/make_figures.py (and scripts/export_band_assignments.py) on
synthetic evaluation outputs produced by the real pipeline. Synthetic data only."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.calibration import aalen_johansen_cif

N_PATIENTS, DATA_SEED = 300, 42
MODELS = ["dynamic_deephit", "tracer", "survlatent_ode"]
FIGURES = ["fig1_cohort_flow", "fig2_cindex", "fig3_calibration", "fig4_risk_band_curves",
           "figS1_pairwise_cindex", "figS2_band_hazard_ratios", "figS3_ibs"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate_script = _load("evaluate_script", "scripts/evaluate.py")
export_script = _load("export_script", "scripts/export_band_assignments.py")
figures = _load("make_figures", "scripts/make_figures.py")


@pytest.fixture(scope="module")
def synthetic_outputs(tmp_path_factory):
    """1-epoch models -> val evaluation (cut-points) -> test evaluation (--final, saved cut-points)
    -> exported band assignments, plus a synthetic flow log consistent with the split."""
    root = tmp_path_factory.mktemp("figs")
    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training = yaml.safe_load(f)
    visits, outcomes = generate_synthetic_cohort(n_patients=N_PATIENTS, seed=DATA_SEED)
    for name in MODELS:
        torch.manual_seed(training["seed"])
        trainer, _ = evaluate_script.make_trainer(name, visits, outcomes, schema, training)
        trainer.fit(trainer.train_ids, trainer.val_ids, num_epochs=1, checkpoint_dir=str(root / "runs" / name))
    final_yaml = root / "final_models.yaml"
    final_yaml.write_text(yaml.safe_dump({"split_dir": str(root / "runs" / MODELS[0]),
                                          "models": {m: str(root / "runs" / m) for m in MODELS}}))
    common = ["--final-models", str(final_yaml), "--n-patients", str(N_PATIENTS), "--data-seed", str(DATA_SEED),
              "--n-bootstrap", "10"]
    assert evaluate_script.main(common + ["--out-dir", str(root / "val")]) == 0
    assert evaluate_script.main(common + ["--split", "test", "--final", "--cutpoints", str(root / "val" / "cutpoints.json"),
                                          "--out-dir", str(root / "test")]) == 0
    assignments = root / "band_assignments.csv"
    assert export_script.main(["--final", "--final-models", str(final_yaml), "--n-patients", str(N_PATIENTS),
                               "--data-seed", str(DATA_SEED), "--cutpoints", str(root / "val" / "cutpoints.json"),
                               "--check-against", str(root / "test" / "metrics.json"), "--out", str(assignments)]) == 0
    split_file = root / "runs" / MODELS[0] / "split.json"
    n_final = sum(len(json.loads(split_file.read_text())[k]) for k in ("train", "val", "test"))
    flow = root / "flow_log.json"
    flow.write_text(json.dumps({
        "source": {"rows": 5000, "patients": n_final + 12},
        "exclusions": [{"id": "E1", "label": "x", "removed": 10, "remaining": n_final + 2},
                       {"id": "E5", "label": "y", "removed": 2, "remaining": n_final}],
        "final": {"patients": n_final}}))
    return {"root": root, "test": root / "test", "assignments": assignments, "split": split_file, "flow": flow}


def _argv(o, out_dir, *extra):
    return ["--eval-dir", str(o["test"]), "--out-dir", str(out_dir), "--flow", str(o["flow"]),
            "--split-file", str(o["split"]), "--band-assignments", str(o["assignments"]), *extra]


def test_make_figures_end_to_end_writes_every_file(synthetic_outputs, tmp_path):
    out = tmp_path / "figures"
    assert figures.main(_argv(synthetic_outputs, out)) == 0
    for name in FIGURES:
        pdf, png = out / f"{name}.pdf", out / f"{name}.png"
        assert pdf.exists() and pdf.stat().st_size > 5000, name
        assert png.exists() and png.stat().st_size > 5000, name
        data = pdf.read_bytes()
        assert b"/FontFile2" in data and b"/Type3" not in data, f"{name}: fonts not embedded as TrueType text"
    from PIL import Image
    with Image.open(out / "fig4_risk_band_curves.png") as im:  # 174 mm wide at 300 dpi, portrait
        assert abs(im.size[0] - round(174 / 25.4 * 300)) <= 2 and im.size[1] > im.size[0]


def test_exported_assignments_have_no_patient_ids_and_match_saved_bands(synthetic_outputs):
    rows = pd.read_csv(synthetic_outputs["assignments"])
    assert list(rows.columns) == ["model", "outcome", "band", "time_years", "event_type"]
    saved = json.loads((synthetic_outputs["test"] / "metrics.json").read_text())["banding"]["bands"]
    for (m, o), sub in rows.groupby(["model", "outcome"]):
        assert [int((sub["band"] == b).sum()) for b in range(3)] == [r["n"] for r in saved[m][o]]


def test_band_curves_end_at_the_saved_observed_risk(synthetic_outputs):
    rows = pd.read_csv(synthetic_outputs["assignments"])
    results = json.loads((synthetic_outputs["test"] / "metrics.json").read_text())
    curves = figures.band_curves(rows, results)
    fitted = [k for k, v in curves.items() if "curves" in v]
    assert fitted
    for m, o in fitted:
        for b, (t, cif) in enumerate(curves[(m, o)]["curves"]):
            assert cif[-1] == pytest.approx(results["banding"]["bands"][m][o][b]["observed_risk"])
            assert np.all(np.diff(cif) >= 0) and t[0] == 0 and t[-1] == 10
        assert [r[0] for r in curves[(m, o)]["at_risk"]] == curves[(m, o)]["n"]
        sub = rows[(rows["model"] == m) & (rows["outcome"] == o)]
        for b, (t, cif) in enumerate(curves[(m, o)]["drawn"]):  # drawn curves stop at >= 10 at risk
            times = sub.loc[sub["band"] == b, "time_years"].to_numpy()
            t_end = curves[(m, o)]["t_end"][b]
            assert t[-1] == t_end <= 10
            if len(times) >= figures.MIN_AT_RISK:
                assert (times >= t_end).sum() >= figures.MIN_AT_RISK
                assert t_end == 10 or (times > t_end).sum() < figures.MIN_AT_RISK


def test_curve_matches_aalen_johansen_and_km_without_competing():
    rng = np.random.default_rng(0)
    time, et = rng.exponential(6, 300), rng.choice([0, 1, 2], 300, p=[0.5, 0.3, 0.2])
    t, cif = figures.cumulative_incidence_curve(time, et, 10)
    for x in (1.0, 3.5, 7.0, 10.0):
        assert cif[np.searchsorted(t, x, side="right") - 1] == pytest.approx(aalen_johansen_cif(time, et, x))


def test_tampered_assignments_are_refused(synthetic_outputs, tmp_path, capsys):
    rows = pd.read_csv(synthetic_outputs["assignments"])
    rows = rows.iloc[1:]  # one patient missing -> band n no longer matches the saved table
    bad = tmp_path / "bad.csv"
    rows.to_csv(bad, index=False)
    argv = _argv(synthetic_outputs, tmp_path / "out")
    argv[argv.index("--band-assignments") + 1] = str(bad)
    assert figures.main(argv) == 3
    assert "saved banding table" in capsys.readouterr().err


def test_include_baseline_refuses_without_baseline_results(synthetic_outputs, tmp_path, capsys):
    assert figures.main(_argv(synthetic_outputs, tmp_path / "out", "--include-baseline")) == 3
    assert "no 'cox_baseline' model results" in capsys.readouterr().err


def test_flow_counts_must_add_up(synthetic_outputs):
    flow = json.loads(synthetic_outputs["flow"].read_text())
    split = json.loads(synthetic_outputs["split"].read_text())
    flow["exclusions"][0]["removed"] = 11
    with pytest.raises(figures.FigureDataError, match="do not add up"):
        figures.cohort_flow_counts(flow, split)


def test_overlap_check_catches_overlapping_text():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "overlapping label")
    ax.text(0.5, 0.5, "another label")
    assert any(p.startswith("overlap") for p in figures.text_problems(fig))
    plt.close(fig)


def _synthetic_serial_results(split: str) -> dict:
    """A serial_evaluate.py-shaped metrics.json with random values; MALO flagged at k = 2, 3."""
    rng = np.random.default_rng(3)
    predictors = ["landmark_cox"] + [f"{m}_snapshot" for m in MODELS] + [f"{m}_full" for m in MODELS]
    points = {}
    for k in (1, 2, 3):
        per_model, pairwise = [], []
        for o in ("mortality", "malo", "cvd"):
            for p in predictors:
                pe = float(rng.uniform(0.6, 0.85))
                per_model.append({"model": p, "outcome": o, "metric_name": "cindex", "point_estimate": pe,
                                  "ci_lower": pe - float(rng.uniform(0.02, 0.08)), "ci_upper": pe + float(rng.uniform(0.02, 0.08))})
            for m in MODELS:
                pe = float(rng.uniform(-0.05, 0.08))
                pairwise.append({"comparison": f"{m}_full_minus_{m}_snapshot", "outcome": o, "metric_name": "cindex",
                                 "point_estimate": pe, "ci_lower": pe - 0.03, "ci_upper": pe + 0.03})
                pe2 = 0.05 if (m == "tracer" and k == 2) else 0.01  # 2a: only TraCeR at k = 2 excludes 0
                pairwise.append({"comparison": f"{m}_full_minus_landmark_cox", "outcome": o, "metric_name": "cindex",
                                 "point_estimate": pe2, "ci_lower": pe2 - 0.03, "ci_upper": pe2 + 0.03})
        points[str(k)] = {"per_model": per_model, "pairwise": pairwise, "flags": ["malo"] if k > 1 else [],
                          "n_train_evaluated": 1000 - 100 * k,
                          "counts": {o: {"n_events": 25 if (o == "malo" and k > 1) else 60} for o in ("mortality", "malo", "cvd")}}
    return {"analysis": "serial", "split": split, "data": "synthetic", "horizon_years": 5, "prediction_points": [1, 2, 3],
            "n_bootstrap": 1000, "bootstrap_seed": 42, "predictors": predictors, "model_sources": {}, "checkpoint_sha256": {"x": "a"},
            "checkpoints_unchanged": True, "points": points,
            "landmark_cox": {"sha256": "f", "fitted_in_this_run": split == "val"}}


def test_fig5_plots_the_serial_json_exactly_and_daggers_malo(tmp_path):
    import matplotlib.pyplot as plt
    plt.rcParams.update(figures.STYLE)
    test, val = _synthetic_serial_results("test"), _synthetic_serial_results("val")
    figures.check_serial_runs(test, val)

    fig = figures.fig_serial(test)
    assert figures.check_serial_plotted(fig, test) == 3 * (4 + 3) * 3  # outcomes x (row 1 + row 2 series) x k
    assert figures.text_problems(fig) == []
    ticks = [[t.get_text() for t in ax.get_xticklabels()] for ax in fig.axes[:6]]
    assert ticks[1] == ticks[4] == ["1", "2†", "3†"]  # MALO panels B and E
    assert all(t == ["1", "2", "3"] for i, t in enumerate(ticks) if i not in (1, 4))
    assert any("† Fewer than 30 events" in t.get_text() for t in fig.texts)
    stars = {tuple(t.get_gid().split("|")[1:]) for ax in fig.axes for t in ax.texts if (t.get_gid() or "").startswith("star|")}
    assert stars == {(o, "tracer_full", "2") for o in ("mortality", "malo", "cvd")}
    assert all(t.get_text() == "*" for ax in fig.axes for t in ax.texts if (t.get_gid() or "").startswith("star|"))
    assert any("* Paired difference vs landmark Cox / Fine-Gray, 95% CI excludes 0." in t.get_text() for t in fig.texts)
    no_star = json.loads(json.dumps(test))  # the JSON says TraCeR k = 2 no longer excludes 0 -> drawn star refused
    for r in no_star["points"]["2"]["pairwise"]:
        if r["comparison"] == "tracer_full_minus_landmark_cox":
            r["ci_lower"] = -0.01
    with pytest.raises(figures.FigureDataError, match="asterisks"):
        figures.check_serial_plotted(fig, no_star)

    tampered = json.loads(json.dumps(test))
    tampered["points"]["2"]["per_model"][0]["point_estimate"] += 1e-12
    with pytest.raises(figures.FigureDataError, match="differs from the JSON"):
        figures.check_serial_plotted(fig, tampered)
    plt.close(fig)

    refit = json.loads(json.dumps(test))
    refit["landmark_cox"]["fitted_in_this_run"] = True
    with pytest.raises(figures.FigureDataError, match="landmark Cox"):
        figures.check_serial_runs(refit, val)


def _stand_in_baseline(results: dict) -> dict:
    """A baseline evaluation of the same split: the deep rows unchanged, TraCeR's rows relabelled
    cox_baseline, and '<model>_minus_cox_baseline' C-index / IBS differences (TraCeR's exclude 0)."""
    base = json.loads(json.dumps(results))
    base["per_model"] = results["per_model"] + [{**r, "model": "cox_baseline"} for r in results["per_model"] if r["model"] == "tracer"]
    for m in MODELS:
        for o in ("mortality", "malo", "cvd"):
            for metric, pe in (("cindex", 0.03 if m == "tracer" else 0.004), ("ibs", -0.006 if m == "tracer" else 0.001)):
                base["pairwise"].append({"comparison": f"{m}_minus_cox_baseline", "outcome": o, "metric_name": metric,
                                         "point_estimate": pe, "ci_lower": pe - abs(pe) / 2 if m == "tracer" else pe - 0.01,
                                         "ci_upper": pe + abs(pe) / 2 if m == "tracer" else pe + 0.01})
    return base


def test_figS4_plots_the_baseline_differences_exactly(synthetic_outputs):
    import matplotlib.pyplot as plt
    plt.rcParams.update(figures.STYLE)
    results = json.loads((synthetic_outputs["test"] / "metrics.json").read_text())
    base = _stand_in_baseline(results)
    figures.check_baseline_comparison(results, base)
    fig = figures.fig_baseline_differences(base)
    assert figures.check_baseline_diff_plotted(fig, base) == 2 * 3 * 3  # metrics x outcomes x deep models
    assert figures.text_problems(fig) == []
    for ax in fig.axes[:6]:
        for line in ax.lines:
            gid = line.get_gid() or ""
            if gid.startswith("pe|"):  # filled only where the CI excludes 0 (TraCeR in this stand-in)
                assert (line.get_markerfacecolor() != "white") == (gid.split("|")[3] == "tracer"), gid
    assert any("Positive ΔC-index: deep model higher; positive ΔIBS: deep model worse." in t.get_text() for t in fig.texts)
    tampered = json.loads(json.dumps(base))
    next(r for r in tampered["pairwise"] if r["comparison"] == "survlatent_ode_minus_cox_baseline")["ci_upper"] += 1e-12
    with pytest.raises(figures.FigureDataError, match="differs from the JSON"):
        figures.check_baseline_diff_plotted(fig, tampered)
    plt.close(fig)
    other = json.loads(json.dumps(base))
    next(r for r in other["per_model"] if r["model"] == "dynamic_deephit")["point_estimate"] += 0.01
    with pytest.raises(figures.FigureDataError, match="deep-model results differ"):
        figures.check_baseline_comparison(results, other)


def test_include_baseline_draws_the_grey_baseline_row(synthetic_outputs, tmp_path):
    results = json.loads((synthetic_outputs["test"] / "metrics.json").read_text())
    base = _stand_in_baseline(results)
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "metrics.json").write_text(json.dumps(base))
    merged = figures.merge_baseline_rows(results, base)
    import matplotlib.pyplot as plt
    plt.rcParams.update(figures.STYLE)
    for metric in ("cindex", "ibs"):
        fig = figures.fig_metric_forest(merged, metric, include_baseline=True)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels[-1] == "Cox/Fine-Gray baseline" and len(labels) == 4
        grey = [l for ax in fig.axes for l in ax.lines if l.get_color() == "#6E6E6E" and l.get_marker() == "D"]
        assert len(grey) == 3  # one grey diamond per panel
        plt.close(fig)
    out = tmp_path / "figs"
    assert figures.main(_argv(synthetic_outputs, out, "--include-baseline", "--baseline-eval-dir", str(tmp_path / "base"))) == 0
    assert (out / "fig2_cindex.pdf").exists() and (out / "figS3_ibs.png").exists()
    assert (out / "figS4_baseline_differences.pdf").exists() and (out / "figS4_baseline_differences.png").exists()
    base["n_patients"] += 1
    with pytest.raises(figures.FigureDataError, match="n_patients"):
        figures.merge_baseline_rows(results, base)

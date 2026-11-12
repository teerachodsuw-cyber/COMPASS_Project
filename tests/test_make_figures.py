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
    assert "no 'baseline' model" in capsys.readouterr().err


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

"""Tests for compass.evaluation.calibration (Aalen-Johansen / Kaplan-Meier observed risk,
decile calibration tables, O/E) and the calibration plot. Synthetic data only."""

import numpy as np
import pytest
from lifelines import AalenJohansenFitter, KaplanMeierFitter

from compass.evaluation.calibration import (
    aalen_johansen_cif,
    calibration_all,
    calibration_table,
    kaplan_meier_risk,
    observed_risk,
)
from compass.visualization.calibration_plot import save_calibration_plots

# 6 patients: 1 = event of interest, 2 = competing death, 0 = censored.
HAND_TIME = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
HAND_TYPE = np.array([1, 2, 0, 1, 0, 2])


def test_aalen_johansen_matches_hand_computed_case():
    # t=1: n=6, cause      -> CIF = 1/6,                 S = 5/6
    # t=2: n=5, death (the censoring at 2 stays at risk) -> S = 5/6 * 4/5 = 2/3
    # t=3: n=3, cause      -> CIF = 1/6 + 2/3 * 1/3 = 7/18, S = 4/9
    # t=5: n=1, death      -> CIF unchanged
    assert aalen_johansen_cif(HAND_TIME, HAND_TYPE, horizon=10) == pytest.approx(7 / 18, abs=1e-12)
    assert aalen_johansen_cif(HAND_TIME, HAND_TYPE, horizon=2.5) == pytest.approx(1 / 6, abs=1e-12)
    assert aalen_johansen_cif(HAND_TIME, HAND_TYPE, horizon=0.5) == 0.0
    # the competing cause's CIF: deaths at t=2 (S(2-)=5/6, n=5) and t=5 (S(5-)=4/9, n=1) -> 11/18;
    # everyone has had an event or been censored by t=5 with S(5)=0, so the two CIFs sum to 1
    assert aalen_johansen_cif(HAND_TIME, HAND_TYPE, horizon=10, cause=2) == pytest.approx(5 / 6 / 5 + 4 / 9, abs=1e-12)
    assert aalen_johansen_cif(HAND_TIME, HAND_TYPE, 10, 1) + aalen_johansen_cif(HAND_TIME, HAND_TYPE, 10, 2) == pytest.approx(1.0)


def test_aalen_johansen_differs_from_km_when_deaths_compete():
    # KM censoring the deaths: t=1 S=5/6, t=3 (n=3) S=5/9 -> risk 4/9 > Aalen-Johansen 7/18
    km = kaplan_meier_risk(HAND_TIME, HAND_TYPE == 1, horizon=10)
    assert km == pytest.approx(4 / 9, abs=1e-12)
    assert km > aalen_johansen_cif(HAND_TIME, HAND_TYPE, horizon=10) + 0.05
    # MALO / CVD use Aalen-Johansen, mortality uses KM
    assert observed_risk(HAND_TIME, HAND_TYPE, 10, "malo") == pytest.approx(7 / 18)
    assert observed_risk(HAND_TIME, np.where(HAND_TYPE == 1, 1, 0), 10, "mortality") == pytest.approx(
        kaplan_meier_risk(HAND_TIME, HAND_TYPE == 1, 10))


def test_aalen_johansen_equals_km_without_competing_events():
    rng = np.random.default_rng(0)
    time = rng.exponential(8, 400)
    event = (rng.random(400) < 0.6).astype(int)
    kmf = KaplanMeierFitter().fit(time, event)
    assert aalen_johansen_cif(time, event, horizon=10) == pytest.approx(1 - float(kmf.predict(10)), abs=1e-12)


def test_aalen_johansen_matches_lifelines_on_tie_free_data():
    rng = np.random.default_rng(1)
    n = 500
    time = rng.exponential(8, n)  # continuous: no ties, so lifelines does not jitter
    event_type = rng.choice([0, 1, 2], n, p=[0.5, 0.2, 0.3])
    ajf = AalenJohansenFitter(calculate_variance=False).fit(time, event_type, event_of_interest=1)
    cif = ajf.cumulative_density_.iloc[:, 0]
    expected = float(cif[cif.index <= 10].iloc[-1])
    assert aalen_johansen_cif(time, event_type, horizon=10) == pytest.approx(expected, abs=1e-10)


def _sample(n=1000, seed=2):
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(0.02, 0.6, n)
    latent = rng.exponential(10 / (predicted * 5), n)
    death = rng.exponential(25, n)
    censor = rng.uniform(2, 14, n)
    time = np.minimum.reduce([latent, death, censor])
    event_type = np.where(time == latent, 1, np.where(time == death, 2, 0))
    return predicted, time, event_type


def test_calibration_table_deciles_and_oe():
    predicted, time, event_type = _sample()
    table = calibration_table(predicted, time, event_type, horizon=10, outcome="malo")
    groups = table["groups"]
    assert len(groups) == 10 and sum(g["n"] for g in groups) == predicted.size
    assert max(g["n"] for g in groups) - min(g["n"] for g in groups) <= 1
    means = [g["mean_predicted"] for g in groups]
    assert means == sorted(means)  # lowest predicted risk first
    ov = table["overall"]
    assert ov["expected"] == pytest.approx(predicted.mean())
    assert ov["observed"] == pytest.approx(aalen_johansen_cif(time, event_type, 10))
    assert ov["oe_ratio"] == pytest.approx(ov["observed"] / ov["expected"])
    assert table["estimator"].startswith("Aalen-Johansen")
    assert calibration_table(predicted, time, np.where(event_type == 1, 1, 0), 10, "mortality")["estimator"] == "Kaplan-Meier"


def test_calibration_deciles_survive_heavy_ties():
    predicted, time, event_type = _sample(n=95)
    predicted = np.round(predicted, 1)  # few distinct values
    groups = calibration_table(predicted, time, event_type, 10, "cvd")["groups"]
    assert len(groups) == 10 and all(g["n"] > 0 for g in groups)


def test_calibration_all_uses_at_risk_rows_and_saves_plots(tmp_path):
    predicted, time, event_type = _sample(n=300)
    cif = np.repeat(predicted[:, None], 10, axis=1)
    at_risk = np.ones(300, bool)
    at_risk[:50] = False
    gt = {o: {"time_years": time, "event_type": event_type if o != "mortality" else np.where(event_type == 1, 1, 0),
              "at_risk": at_risk} for o in ("mortality", "malo", "cvd")}
    preds = {m: {o: cif for o in gt} for m in ("dynamic_deephit", "tracer", "survlatent_ode")}
    cal = calibration_all(preds, gt, horizon=10)
    assert set(cal) == set(preds) and all(cal[m][o]["overall"]["n"] == 250 for m in cal for o in gt)
    paths = save_calibration_plots(cal, ("mortality", "malo", "cvd"), tmp_path)
    assert [p.name for p in paths] == ["calibration_mortality.png", "calibration_malo.png", "calibration_cvd.png"]
    assert all(p.stat().st_size > 1000 for p in paths)

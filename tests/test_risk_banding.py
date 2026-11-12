"""Tests for compass.evaluation.risk_banding: two cut-points / three bands (each >= 10%),
log-rank (mortality) vs Gray's test (MALO / CVD), fit-on-validation / apply-unchanged,
Cox / cause-specific Cox hazard ratios, cutpoints.json save / load checks. Synthetic only."""

import math

import numpy as np
import pytest
from comprisk.gray_test import gray_test
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

from compass.evaluation.calibration import aalen_johansen_cif
from compass.evaluation.risk_banding import (
    CutpointsMismatchError,
    NoValidCutpointsError,
    assign_bands,
    band_hazard_ratios,
    band_summary,
    fit_all_cutpoints,
    fit_band_cutpoints,
    load_cutpoints,
    save_cutpoints,
    summarise_all_bands,
)

COARSE = tuple(range(10, 91, 5))  # fewer candidates keeps the tests fast


def _cohort(n=600, seed=0, competing=True):
    """Risk score -> event hazard; optional competing death. Times in years."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 1, n)
    latent = rng.exponential(1 / (0.02 + 0.25 * score ** 2))
    death = rng.exponential(30, n) if competing else np.full(n, np.inf)
    censor = rng.uniform(3, 15, n)
    time = np.minimum.reduce([latent, death, censor])
    event_type = np.where(time == latent, 1, np.where(time == death, 2, 0))
    return score, time, event_type


def test_cutpoints_give_three_bands_each_at_least_10_percent():
    score, time, event_type = _cohort()
    for outcome, et in (("mortality", np.where(event_type == 1, 1, 0)), ("malo", event_type)):
        fit = fit_band_cutpoints(score, time, et, outcome, horizon=10, percentiles=COARSE)
        c1, c2 = fit["cutpoints"]
        assert fit["status"] == "fitted" and c1 < c2
        bands = assign_bands(score, fit["cutpoints"])
        counts = np.bincount(bands, minlength=3)
        assert len(counts) == 3 and counts.min() >= math.ceil(0.10 * score.size)
        assert all(f >= 0.10 for f in fit["band_fractions_fit"])
        assert fit["band_fractions_fit"] == pytest.approx(counts / score.size)


def test_search_uses_logrank_for_mortality_and_gray_for_malo():
    score, time, event_type = _cohort(seed=1)
    t10, e10 = np.minimum(time, 10), np.where(time > 10, 0, event_type)
    mort = fit_band_cutpoints(score, time, np.where(event_type == 1, 1, 0), "mortality", 10, percentiles=COARSE)
    bands = assign_bands(score, mort["cutpoints"])
    assert mort["test"] == "log-rank"
    assert mort["statistic"] == pytest.approx(multivariate_logrank_test(t10, bands, e10 == 1).test_statistic)
    malo = fit_band_cutpoints(score, time, event_type, "malo", 10, percentiles=COARSE)
    bands = assign_bands(score, malo["cutpoints"])
    assert malo["test"] == "Gray"
    assert malo["statistic"] == pytest.approx(gray_test(t10, e10, bands, cause=1).stat)


def test_chosen_pair_maximises_the_statistic_among_eligible_pairs():
    score, time, event_type = _cohort(n=300, seed=2)
    fit = fit_band_cutpoints(score, time, event_type, "cvd", 10, percentiles=COARSE)
    t10, e10 = np.minimum(time, 10), np.where(time > 10, 0, event_type)
    cands = np.unique(np.quantile(score, np.array(COARSE) / 100))
    best = max(gray_test(t10, e10, assign_bands(score, (a, b)), cause=1).stat
               for i, a in enumerate(cands) for b in cands[i + 1:]
               if np.bincount(assign_bands(score, (a, b)), minlength=3).min() >= math.ceil(0.1 * score.size))
    assert fit["statistic"] == pytest.approx(best)


def test_no_eligible_pair_raises():
    score, time, event_type = _cohort(n=200, seed=3)
    with pytest.raises(NoValidCutpointsError):
        fit_band_cutpoints(score, time, event_type, "malo", 10, percentiles=COARSE, min_band_fraction=0.4)
    with pytest.raises(NoValidCutpointsError):  # all predictions tied -> a single candidate, no pair
        fit_band_cutpoints(np.full(200, 0.3), time, event_type, "malo", 10, percentiles=COARSE)


def test_cutpoints_fitted_on_val_are_applied_unchanged_to_another_split():
    score, time, event_type = _cohort(seed=4)
    fit = fit_band_cutpoints(score, time, event_type, "malo", 10, percentiles=COARSE)
    c1, c2 = fit["cutpoints"]
    # another split with a completely different score distribution: refitting would put ~1/3
    # in each band; applying the saved values puts everyone above c2 in the high band.
    other = np.random.default_rng(5).uniform(c2 + 0.01, c2 + 1.0, 250)
    assert (assign_bands(other, fit["cutpoints"]) == 2).all()
    assert (assign_bands(np.array([c1, c1 + 1e-9, c2, c2 + 1e-9]), (c1, c2)) == [0, 1, 1, 2]).all()


def test_summarise_applies_saved_cutpoints_without_refitting(monkeypatch):
    import compass.evaluation.risk_banding as rb

    score, time, event_type = _cohort(seed=6)
    gt = {o: {"time_years": time, "event_type": event_type if o != "mortality" else np.where(event_type == 1, 1, 0),
              "at_risk": np.ones(score.size, bool)} for o in ("mortality", "malo", "cvd")}
    preds = {"tracer": {o: np.repeat(score[:, None], 10, axis=1) for o in gt}}
    fitted = fit_all_cutpoints(preds, gt, 10, percentiles=COARSE)

    def must_not_fit(*a, **k):
        raise AssertionError("cut-points were refitted at application time")

    monkeypatch.setattr(rb, "fit_band_cutpoints", must_not_fit)
    shifted = {"tracer": {o: v + 0.5 for o, v in preds["tracer"].items()}}  # another split's scores
    bands = summarise_all_bands(shifted, gt, fitted, 10)
    for o in gt:
        expected = np.bincount(assign_bands(score + 0.5, fitted["tracer"][o]["cutpoints"]), minlength=3)
        assert [r["n"] for r in bands["tracer"][o]] == list(expected)


def test_outcome_without_events_is_reported_not_banded():
    score, time, _ = _cohort(n=200, seed=7)
    gt = {o: {"time_years": time, "event_type": np.zeros(200, int), "at_risk": np.ones(200, bool)}
          for o in ("mortality", "malo", "cvd")}
    fitted = fit_all_cutpoints({"tracer": {o: np.repeat(score[:, None], 10, axis=1) for o in gt}}, gt, 10, COARSE)
    assert all(fitted["tracer"][o]["status"].startswith("not fitted") for o in gt)


def test_hazard_ratios_vs_low_band_known_case():
    rng = np.random.default_rng(8)
    n = 700
    bands = np.repeat([0, 1, 2], n)
    time = rng.exponential(1 / (0.1 * np.array([1.0, 2.0, 4.0])[bands]))
    event_type = np.ones(3 * n, int)
    rows = {r["band"]: r for r in band_hazard_ratios(bands, np.minimum(time, 30), event_type, "mortality")}
    assert rows["intermediate"]["ci_lower"] < 2.0 < rows["intermediate"]["ci_upper"]
    assert rows["high"]["ci_lower"] < 4.0 < rows["high"]["ci_upper"]
    assert rows["high"]["model"] == "Cox"


def test_malo_hazard_ratios_are_cause_specific():
    score, time, event_type = _cohort(seed=9)
    cutpoints = fit_band_cutpoints(score, time, event_type, "malo", 10, percentiles=COARSE)["cutpoints"]
    rows = band_summary(score, cutpoints, time, event_type, "malo", 10)
    assert [r["band"] for r in rows] == ["low", "intermediate", "high"]
    assert rows[0]["hazard_ratio"] == 1.0 and all(r["hr_model"] == "cause-specific Cox" for r in rows)
    # cause-specific: a Cox fit with competing deaths CENSORED, follow-up truncated at 10 yr
    t10, e10 = np.minimum(time, 10), np.where(time > 10, 0, event_type)
    bands = assign_bands(score, cutpoints)
    import pandas as pd
    df = pd.DataFrame({"d": t10, "e": (e10 == 1).astype(int), "i": (bands == 1).astype(int), "h": (bands == 2).astype(int)})
    hr = CoxPHFitter().fit(df, "d", "e").summary["exp(coef)"]
    assert rows[1]["hazard_ratio"] == pytest.approx(hr["i"]) and rows[2]["hazard_ratio"] == pytest.approx(hr["h"])
    # per band: n, events within 10 yr, Aalen-Johansen observed risk
    for b, r in enumerate(rows):
        sel = bands == b
        assert r["n"] == sel.sum() and r["n_events"] == int((e10[sel] == 1).sum())
        assert r["observed_risk"] == pytest.approx(aalen_johansen_cif(t10[sel], e10[sel], 10))


def test_band_without_events_gives_nan_hr_with_reason():
    bands = np.repeat([0, 1, 2], 20)
    event_type = np.concatenate([np.zeros(20, int), np.ones(20, int), np.ones(20, int)])
    rows = band_hazard_ratios(bands, np.linspace(1, 9, 60), event_type, "cvd")
    assert all(math.isnan(r["hazard_ratio"]) and "no events" in r["note"] for r in rows)


def test_cutpoints_file_round_trip_and_refusals(tmp_path):
    meta = {"fitted_on_split": "val", "horizon_years": 10, "split_seed": 42,
            "data_fingerprint": {"patients": 10}, "model_sources": {"tracer": "runs/x/tracer"}}
    fitted = {"tracer": {"mortality": {"status": "fitted", "cutpoints": [0.1, 0.3]}}}
    path = save_cutpoints(tmp_path / "cutpoints.json", fitted, meta)
    assert load_cutpoints(path, meta, ["tracer"]) == fitted
    with pytest.raises(CutpointsMismatchError, match="split_seed"):
        load_cutpoints(path, {**meta, "split_seed": 1}, ["tracer"])
    with pytest.raises(CutpointsMismatchError, match="different checkpoint"):
        load_cutpoints(path, {**meta, "model_sources": {"tracer": "runs/y/tracer"}}, ["tracer"])
    with pytest.raises(CutpointsMismatchError, match="no saved cut-points for model"):
        load_cutpoints(path, meta, ["tracer", "survlatent_ode"])
    save_cutpoints(tmp_path / "bad.json", fitted, {**meta, "fitted_on_split": "test"})
    with pytest.raises(CutpointsMismatchError, match="not the validation split"):
        load_cutpoints(tmp_path / "bad.json", meta, ["tracer"])
    with pytest.raises(CutpointsMismatchError, match="no saved cut-points at"):
        load_cutpoints(tmp_path / "missing.json", meta, ["tracer"])

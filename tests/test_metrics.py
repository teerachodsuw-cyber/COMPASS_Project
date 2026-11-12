"""Tests for compass.evaluation.metrics: hand-computed sanity checks for time_dependent_
cindex and integrated_brier_score_metric (calibration: tests/test_calibration.py).

wolbers_cindex tests: hand-computed sanity checks (perfect/near-random concordance) plus a
first-pass cross-check against comprisk's independent concordance_index_uno_cr on synthetic
data. This cross-check is NOT the final verification of wolbers_cindex -- that's the R
pec::cindex ground-truth comparison in tests/test_r_reference_fixtures.py. Both
wolbers_cindex and comprisk's implementation apply IPCW weighting for censoring, but via
different censoring-KM conventions (comprisk folds competing events into its censoring
hazard; wolbers_cindex does not -- see _censoring_ipcw_weights's docstring for why that
matters and diverges from R), so exact agreement isn't expected; this test only checks the
two are in the same ballpark, catching gross implementation errors.
"""

import numpy as np

from comprisk.metrics import compute_uno_weights, concordance_index_uno_cr

from compass.evaluation.metrics import (
    integrated_brier_score_metric,
    schoop_competing_risks_ibs,
    time_dependent_cindex,
    wolbers_cindex,
)


def test_time_dependent_cindex_perfect_concordance():
    # Higher predicted risk for patients whose event happens earlier -- perfect ranking.
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 20, 20, 20, 20, 20], dtype=float)
    cif_at_horizon = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

    cindex = time_dependent_cindex(cif_at_horizon, event_indicator, time_years, horizon=10)
    assert cindex > 0.95


def test_time_dependent_cindex_scrambled_near_random():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 20, 20, 20, 20, 20], dtype=float)
    # Deliberately scrambled: predicted risk has no relationship to actual event timing.
    cif_scrambled = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])

    cindex = time_dependent_cindex(cif_scrambled, event_indicator, time_years, horizon=10)
    assert 0.3 < cindex < 0.7  # near 0.5, not necessarily exact given n=10


def test_integrated_brier_score_well_calibrated_near_zero():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 20, 20, 20, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    # Perfect knowledge: event patients' CIF is ~1 from their event time onward, censored
    # patients' CIF stays ~0 throughout the evaluated window.
    well_calibrated = np.zeros((10, 10))
    well_calibrated[:5, :] = 0.999
    well_calibrated[5:, :] = 0.001

    ibs = integrated_brier_score_metric(well_calibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)
    assert ibs < 0.01


def test_integrated_brier_score_miscalibrated_scores_worse():
    event_indicator = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 20, 20, 20, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    well_calibrated = np.zeros((10, 10))
    well_calibrated[:5, :] = 0.999
    well_calibrated[5:, :] = 0.001
    ibs_well = integrated_brier_score_metric(well_calibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)

    # Badly miscalibrated: predicts near-certain event for everyone, including the 5
    # patients who never had one.
    miscalibrated = np.full((10, 10), 0.99)
    ibs_bad = integrated_brier_score_metric(miscalibrated, event_indicator, time_years, horizon=10, time_grid=time_grid)

    assert ibs_bad > ibs_well
    assert ibs_bad > 0.3


def test_wolbers_cindex_perfect_concordance():
    # cause=1 patients ranked by score in the same order as their event time; a competing
    # event (code 2) and a pure censoring (code 0) thrown in as comparable/excluded partners.
    event_type = np.array([1, 1, 1, 1, 1, 2, 2, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 2.5, 6, 20, 20], dtype=float)
    cif = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.3, 0.1, 0.1])

    cindex = wolbers_cindex(cif, event_type, time_years, horizon=10, cause_of_interest=1)
    assert cindex > 0.95


def test_wolbers_cindex_scrambled_near_random():
    event_type = np.array([1, 1, 1, 1, 1, 2, 2, 0, 0])
    time_years = np.array([1, 2, 3, 4, 5, 2.5, 6, 20, 20], dtype=float)
    cif_scrambled = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])

    cindex = wolbers_cindex(cif_scrambled, event_type, time_years, horizon=10, cause_of_interest=1)
    assert 0.2 < cindex < 0.8  # near 0.5, not necessarily exact given n=9


def test_wolbers_cindex_competing_event_is_not_neutral_censoring():
    # Two patients, identical predicted CIF. Patient A gets the outcome of interest at t=5.
    # Patient B is CENSORED at t=3 (before A's event) in one version, and has a COMPETING
    # event at t=3 in the other. Per Wolbers, a competing event before i's event time IS a
    # comparable, contributing pair (B's fate is permanently decided); a pure censoring
    # before i's event time is NOT comparable (B's fate is unknown). This is exactly the
    # distinction Uno's C (time_dependent_cindex) gets wrong for malo/cvd.
    cif = np.array([0.9, 0.9])
    time_years = np.array([5.0, 3.0])

    censored_case = wolbers_cindex(cif, np.array([1, 0]), time_years, horizon=10, cause_of_interest=1)
    competing_case = wolbers_cindex(cif, np.array([1, 2]), time_years, horizon=10, cause_of_interest=1)

    assert np.isnan(censored_case)  # no comparable pairs -- B's fate was unknown at t=5
    assert competing_case == 0.5  # tied CIF -> half credit, but the pair DOES count


def _synthetic_case(n_patients, seed, outcome_name="malo", horizon_years=10.0):
    from compass.data.synthetic_generator import generate_synthetic_cohort

    visits_df, outcomes_df = generate_synthetic_cohort(n_patients=n_patients, seed=seed)
    first_visit = visits_df.sort_values("visit_date").groupby("patient_id").first()
    outcomes_indexed = outcomes_df.set_index("patient_id")
    cohort_start = np.datetime64("2013-06-01")

    times, event_codes, scores = [], [], []
    for pid in outcomes_indexed.index:
        if pid not in first_visit.index:
            continue
        outcome_event = int(outcomes_indexed.loc[pid, f"{outcome_name}_event"])
        mortality_event = int(outcomes_indexed.loc[pid, "mortality_event"])
        if outcome_event == 1:
            code = 1
            t = outcomes_indexed.loc[pid, f"{outcome_name}_date"]
        elif mortality_event == 1:
            code = 2
            t = outcomes_indexed.loc[pid, "mortality_date"]
        else:
            code = 0
            t = np.datetime64("2025-06-30")
        years = (np.datetime64(t) - cohort_start) / np.timedelta64(1, "D") / 365.25
        times.append(max(years, 0.01))
        event_codes.append(code)
        age = first_visit.loc[pid, "index_age"]
        bmi = first_visit.loc[pid, "bmi"]
        scores.append(age + bmi)

    time_years = np.array(times)
    event_type = np.array(event_codes)
    score = np.array(scores)
    score = (score - score.mean()) / score.std()
    return time_years, event_type, score


def test_wolbers_cindex_vs_comprisk_uno_cr_same_ballpark():
    for seed in (100, 200):
        time_years, event_type, score = _synthetic_case(n_patients=300, seed=seed)
        horizon = 10.0

        ours = wolbers_cindex(score, event_type, time_years, horizon=horizon, cause_of_interest=1)

        weights = compute_uno_weights(time_years, event_type)
        comprisk_val = concordance_index_uno_cr(event_type, time_years, score, cause=1, weights=weights)

        assert not np.isnan(ours)
        # First-pass sanity check only (not final verification -- see
        # tests/test_r_reference_fixtures.py for the R pec::cindex ground truth). comprisk
        # additionally applies IPCW weighting that wolbers_cindex does not, so allow a
        # generous tolerance -- this just needs to catch gross errors, not drift.
        assert abs(ours - comprisk_val) < 0.05


def test_schoop_ibs_well_calibrated_near_zero():
    event_type = np.array([1, 1, 1, 1, 1, 2, 2, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 3, 3, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    # Perfect knowledge: cause-1 patients' CIF is ~1 from their event time onward; everyone
    # else (competing OR still at risk) stays ~0 throughout.
    well_calibrated = np.zeros((9, 10))
    well_calibrated[:5, :] = 0.999

    ibs = schoop_competing_risks_ibs(well_calibrated, event_type, time_years, horizon=10, time_grid=time_grid, cause_of_interest=1)
    assert ibs < 0.01


def test_schoop_ibs_miscalibrated_worse():
    event_type = np.array([1, 1, 1, 1, 1, 2, 2, 0, 0])
    time_years = np.array([1, 1, 1, 1, 1, 3, 3, 20, 20], dtype=float)
    time_grid = np.arange(1, 11, dtype=float)

    well_calibrated = np.zeros((9, 10))
    well_calibrated[:5, :] = 0.999
    ibs_well = schoop_competing_risks_ibs(well_calibrated, event_type, time_years, horizon=10, time_grid=time_grid, cause_of_interest=1)

    miscalibrated = np.full((9, 10), 0.99)
    ibs_bad = schoop_competing_risks_ibs(miscalibrated, event_type, time_years, horizon=10, time_grid=time_grid, cause_of_interest=1)

    assert ibs_bad > ibs_well
    assert ibs_bad > 0.3


def test_schoop_ibs_zero_cause_events_returns_nan():
    event_type = np.array([2, 2, 0, 0])  # no cause-1 events at all
    time_years = np.array([1.0, 2.0, 20.0, 20.0])
    time_grid = np.arange(1, 11, dtype=float)
    cif = np.full((4, 10), 0.3)

    assert np.isnan(schoop_competing_risks_ibs(cif, event_type, time_years, horizon=10, time_grid=time_grid, cause_of_interest=1))


def test_schoop_ibs_diverges_from_naive_ibs_with_uneven_competing_risk():
    # A model that predicts a MODEST, roughly-flat CIF for everyone (e.g. 0.15). Group A
    # (n=150) has almost no competing events -- naive IBS (treating competing events as
    # censoring) and Schoop's should roughly agree there. Group B (n=150) has HEAVY early
    # competing-event burden (death before cause-1 could occur) -- naive IBS scores this
    # as "still at risk, uncertain," penalizing the flat 0.15 prediction as if those patients
    # might still get cause-1 later; Schoop's knows their cause-1 fate is ALREADY decided
    # (permanently 0) and scores the same flat prediction as a modest, known overprediction
    # instead -- the same qualitative distinction as Gray's-vs-log-rank and Wolbers-vs-Uno.
    rng = np.random.default_rng(7)
    n_per_group = 150
    horizon = 10.0
    time_grid = np.arange(1, 11, dtype=float)

    # Group A: mostly cause-1 or censored, few competing events.
    a_type = rng.choice([1, 0], size=n_per_group, p=[0.3, 0.7])
    a_time = np.where(a_type == 1, rng.uniform(1, 9, n_per_group), 20.0)

    # Group B: heavy EARLY competing-event burden, few cause-1 events.
    b_type = rng.choice([2, 1], size=n_per_group, p=[0.8, 0.2])
    b_time = np.where(b_type == 2, rng.uniform(0.5, 2.0, n_per_group), rng.uniform(1, 9, n_per_group))

    event_type = np.concatenate([a_type, b_type])
    event_time = np.concatenate([a_time, b_time])
    flat_cif = np.full((event_type.size, 10), 0.15)

    schoop_ibs = schoop_competing_risks_ibs(flat_cif, event_type, event_time, horizon, time_grid, cause_of_interest=1)

    # Naive: treats non-cause-1 events (competing OR censoring) identically as censoring,
    # exactly the bias the Schoop correction exists to fix.
    naive_event_indicator = (event_type == 1).astype(int)
    naive_ibs = integrated_brier_score_metric(flat_cif, naive_event_indicator, event_time, horizon, time_grid)

    assert not np.isnan(schoop_ibs) and not np.isnan(naive_ibs)
    assert abs(schoop_ibs - naive_ibs) > 0.02  # must disagree substantially, not just be present but inert

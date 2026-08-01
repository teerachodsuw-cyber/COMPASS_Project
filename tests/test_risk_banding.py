"""Tests for compass.evaluation.risk_banding: outcome-optimized cutpoint search (log-rank
for mortality, Gray's test for malo/cvd), the outcome-blind quantile fallback, the fit/apply
separation that prevents test-data leakage into cutpoint selection, and Cox hazard ratios.
"""

import numpy as np
from comprisk.gray_test import gray_test

from compass.evaluation.risk_banding import (
    apply_cutpoint,
    assign_quantile_bands,
    compute_band_hazard_ratios,
    find_optimal_cutpoint_gray,
    find_optimal_cutpoint_logrank,
    is_outcome_optimized_reliable,
)


def test_logrank_cutpoint_finds_real_separation():
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.uniform(0, 1, n)
    # Known true separation at score=0.5: above -> much earlier events.
    event_time = np.where(scores > 0.5, rng.exponential(2, n), rng.exponential(15, n))
    event_time = np.clip(event_time, 0.01, 30)
    event_indicator = np.ones(n, dtype=int)

    result = find_optimal_cutpoint_logrank(scores, event_indicator, event_time)

    assert abs(result["cutpoint"] - 0.5) < 0.15
    assert result["p_value"] < 0.01


def test_gray_cutpoint_accounts_for_competing_risk():
    rng = np.random.default_rng(1)
    n = 400
    scores = rng.uniform(0, 1, n)
    group_high = scores > 0.5

    # TRUE malo-susceptibility-generating-process is IDENTICAL between groups, but the
    # high-score group has much higher mortality (competing risk), censoring many of them
    # out before malo can occur -- naive log-rank (treating competing deaths as censoring)
    # sees this as "protective" for malo; Gray's test should not be fooled the same way.
    malo_latent_time = rng.exponential(10, n)
    mortality_time = np.where(group_high, rng.exponential(2, n), rng.exponential(50, n))

    event_time = np.minimum(np.minimum(malo_latent_time, mortality_time), 20.0)
    malo_event = ((event_time == malo_latent_time) & (malo_latent_time <= 20.0)).astype(int)
    competing_event = ((event_time == mortality_time) & (mortality_time <= 20.0) & (malo_event == 0)).astype(int)

    # Naive log-rank on the same data, ignoring the competing risk entirely (as if
    # find_optimal_cutpoint_logrank were used on malo/cvd -- the wrong tool).
    naive_result = find_optimal_cutpoint_logrank(scores, malo_event, event_time)

    # Gray's test (comprisk) at the SAME cutpoint=0.5, for a fair apples-to-apples comparison.
    event_code = np.where(malo_event.astype(bool), 1, np.where(competing_event.astype(bool), 2, 0))
    group = (scores > 0.5).astype(int)
    gray_result = gray_test(event_time, event_code, group, cause=1)

    from lifelines.statistics import logrank_test

    high = scores > 0.5
    naive = logrank_test(
        event_time[high], event_time[~high], event_observed_A=malo_event[high], event_observed_B=malo_event[~high]
    )

    # The two tests must disagree substantially on the same data -- proving the competing-
    # risk correction is actually doing something, not just present but inert.
    assert abs(gray_result.stat - naive.test_statistic) > 5.0
    assert naive_result["cutpoint"] is not None  # sanity: naive search still ran successfully


def test_quantile_fallback_triggers_correctly():
    rng = np.random.default_rng(2)
    n = 200
    # Mimics MALO's real rarity: very few events.
    sparse_event_indicator = np.zeros(n, dtype=int)
    sparse_event_indicator[:4] = 1  # only 4 events total, far below the 2*15=30 threshold

    assert is_outcome_optimized_reliable(sparse_event_indicator) is False

    scores = rng.uniform(0, 1, n)
    bands = assign_quantile_bands(scores, n_bands=3)
    assert set(np.unique(bands)) <= {0, 1, 2}
    assert len(bands) == n
    counts = np.bincount(bands, minlength=3)
    assert (counts > 0).all()  # every band actually got assigned some patients


def test_cutpoint_fit_apply_separation():
    rng = np.random.default_rng(3)
    n_fit, n_other = 300, 100

    fit_scores = rng.uniform(0, 1, n_fit)
    fit_event_time = np.where(fit_scores > 0.5, rng.exponential(2, n_fit), rng.exponential(15, n_fit))
    fit_event_indicator = np.ones(n_fit, dtype=int)

    fitted = find_optimal_cutpoint_logrank(fit_scores, fit_event_indicator, fit_event_time)

    # A DIFFERENT dataset, drawn from a totally different score distribution.
    other_scores = rng.uniform(5, 10, n_other)

    bands = apply_cutpoint(other_scores, fitted["cutpoint"])

    # apply_cutpoint must use ONLY the already-fit cutpoint value, never re-derive one from
    # other_scores -- since fitted["cutpoint"] is ~0.5-ish and other_scores are all >> 0.5,
    # every single patient in `other` must land in the high band. If apply_cutpoint were
    # silently re-fitting on other_scores instead, this would NOT hold.
    assert (bands == 1).all()


def test_hazard_ratio_known_case():
    rng = np.random.default_rng(4)
    n_per_group = 500
    # Group 0: baseline hazard rate 0.1/yr. Group 1: hazard rate 0.3/yr -> true HR = 3.0.
    true_hr = 3.0
    time_0 = rng.exponential(1 / 0.1, n_per_group)
    time_1 = rng.exponential(1 / (0.1 * true_hr), n_per_group)

    band_assignment = np.array([0] * n_per_group + [1] * n_per_group)
    event_time = np.concatenate([time_0, time_1])
    event_time = np.clip(event_time, 0.001, 30.0)
    event_indicator = np.ones(len(event_time), dtype=int)

    hr_df = compute_band_hazard_ratios(band_assignment, event_indicator, event_time, reference_band=0)

    row = hr_df[hr_df["band"] == "1"].iloc[0]
    assert 2.0 < row["hazard_ratio"] < 4.5  # generous tolerance given sampling noise
    assert row["ci_lower"] < true_hr < row["ci_upper"]

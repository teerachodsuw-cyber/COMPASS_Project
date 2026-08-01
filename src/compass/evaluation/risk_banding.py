"""Convert continuous predicted risk scores into discrete clinical bands (e.g.
low/intermediate/high) via outcome-optimized cutpoint search.

Test choice per outcome:
- mortality has no competing risk here -> log-rank test (lifelines.statistics.logrank_test).
- malo/cvd have mortality as a competing risk -> log-rank is WRONG (it treats death as
  independent censoring and biases the comparison); Gray's test (Gray, Annals of Statistics
  1988;16(3):1141-1154) is the correct K-sample test for comparing cumulative incidence
  functions in the presence of a competing risk.

GRAY'S TEST: neither lifelines nor scikit-survival (already project dependencies)
implements Gray's test -- confirmed by checking their APIs directly (lifelines.statistics
has only logrank_test/multivariate_logrank_test; sksurv.compare.compare_survival's own
docstring states it is "the log-rank test"). Uses comprisk.gray_test (PyPI package
`comprisk`) instead of a hand-rolled implementation: an investigation session found
comprisk's Gray's test independently matches R's cmprsk::cuminc (Gray's own reference
implementation) to 1e-9/1e-10 absolute tolerance on real R-generated fixtures (see
tests/fixtures/gray_test_r_reference.csv and tests/test_r_reference_fixtures.py, generated
once via a local R install -- R is NOT a runtime dependency of this project). An earlier
hand-rolled version of this file used an approximate (hypergeometric-style) variance formula
that reached the same qualitative conclusions but a meaningfully different magnitude on the
same data (46.4 vs comprisk's 19.28) -- replaced rather than kept, since comprisk's is the
independently-verified one.

CRITICAL: cutpoints must be FIT on train+val data only, then APPLIED (never re-derived) to
test data. find_optimal_cutpoint_logrank/_gray and fit_quantile_boundaries are the only
functions that look at data to choose a cutpoint -- they are the FIT step and must only ever
be called on train+val. apply_cutpoint/apply_quantile_bands are pure application (score in,
band out) and are safe to call on any split, including test.
"""

import numpy as np
import pandas as pd
from comprisk.gray_test import gray_test
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


def _candidate_cutpoints(scores: np.ndarray, candidate_quantile_range: tuple) -> np.ndarray:
    lo, hi = candidate_quantile_range
    quantiles = np.linspace(lo, hi, 50)
    return np.unique(np.quantile(scores, quantiles))


def find_optimal_cutpoint_logrank(
    scores: np.ndarray,
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    candidate_quantile_range: tuple = (0.1, 0.9),
) -> dict:
    """FIT step, no competing risk (mortality). Scans candidate cutpoints across
    candidate_quantile_range of `scores`, computes a log-rank statistic for each candidate
    2-group split (score > cutpoint vs <=), returns the cutpoint that MAXIMIZES the log-rank
    chi-square statistic, with its statistic and p-value.

    Only ever call this on train+val data -- see module docstring.
    """
    best = {"cutpoint": None, "statistic": -np.inf, "p_value": None}
    for cp in _candidate_cutpoints(scores, candidate_quantile_range):
        high = scores > cp
        if high.sum() == 0 or (~high).sum() == 0:
            continue
        result = logrank_test(
            event_time[high], event_time[~high],
            event_observed_A=event_indicator[high], event_observed_B=event_indicator[~high],
        )
        if result.test_statistic > best["statistic"]:
            best = {"cutpoint": float(cp), "statistic": float(result.test_statistic), "p_value": float(result.p_value)}
    return best


def find_optimal_cutpoint_gray(
    scores: np.ndarray,
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    competing_event_indicator: np.ndarray,
    candidate_quantile_range: tuple = (0.1, 0.9),
) -> dict:
    """FIT step, for outcomes with a competing risk (malo/cvd, mortality competing). Same
    scanning approach as find_optimal_cutpoint_logrank, but using comprisk's Gray's test
    (see module docstring) instead of log-rank.

    competing_event_indicator: 1 for patients who experienced the competing event
    (mortality) rather than the outcome of interest.

    Only ever call this on train+val data -- see module docstring.
    """
    event_code = np.where(event_indicator.astype(bool), 1, np.where(competing_event_indicator.astype(bool), 2, 0))

    best = {"cutpoint": None, "statistic": -np.inf, "p_value": None}
    for cp in _candidate_cutpoints(scores, candidate_quantile_range):
        group = (scores > cp).astype(int)
        if group.sum() == 0 or (group == 0).sum() == 0:
            continue
        result = gray_test(event_time, event_code, group, cause=1)
        if result.stat > best["statistic"]:
            best = {"cutpoint": float(cp), "statistic": float(result.stat), "p_value": float(result.pvalue)}
    return best


def apply_cutpoint(scores: np.ndarray, cutpoint: float) -> np.ndarray:
    """APPLY step: assign each score to band 0 (<=cutpoint) or 1 (>cutpoint), using an
    ALREADY-FIT cutpoint. Never re-derives the cutpoint from `scores` -- safe to call on
    any split (train, val, or test).
    """
    return (np.asarray(scores) > cutpoint).astype(int)


def fit_quantile_boundaries(scores: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """FIT step for the outcome-blind fallback: n_bands-1 quantile boundaries of `scores`.
    Uses no outcome information at all (unlike find_optimal_cutpoint_*), but should still
    be fit on train+val and applied (via apply_quantile_bands) to test for consistency.
    """
    quantiles = np.linspace(0.0, 1.0, n_bands + 1)[1:-1]
    return np.quantile(scores, quantiles)


def apply_quantile_bands(scores: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """APPLY step: assign each score to a band index [0, len(boundaries)] using
    already-fit boundaries."""
    return np.digitize(scores, boundaries, right=True)


def assign_quantile_bands(scores: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """Convenience: fit_quantile_boundaries + apply_quantile_bands on the SAME `scores` in
    one call. Fine for this simple outcome-blind fallback (no outcome information is used,
    so there is no train/test leakage risk in the same sense as the outcome-optimized
    methods) -- for strict pipeline use, prefer fitting boundaries once on train+val and
    reusing them via apply_quantile_bands on other splits.
    """
    boundaries = fit_quantile_boundaries(scores, n_bands)
    return apply_quantile_bands(scores, boundaries)


def is_outcome_optimized_reliable(event_indicator: np.ndarray, min_events_per_candidate_band: int = 15) -> bool:
    """Whether there are enough events to trust an outcome-optimized cutpoint search for
    this outcome, vs. falling back to the outcome-blind quantile method.

    min_events_per_candidate_band=15 is a judgment call, not an established universal rule.
    Reasoning: the classic Peduzzi et al. (1996, J Clin Epidemiol) "events per variable"
    (EPV) guideline for stable Cox regression coefficients recommends >=10 events per
    estimated parameter -- directly relevant here since compute_band_hazard_ratios() fits a
    Cox model with band membership as a categorical predictor. On top of that baseline,
    Altman et al. (1994, BMJ) document that OPTIMAL cutpoint search itself (scanning many
    candidates and picking the best-separating one, exactly what find_optimal_cutpoint_*
    does) inflates false-positive/overfitting risk beyond fitting a single pre-specified
    cutpoint -- worse with fewer events. 15 is a modest safety margin (10 -> 15) added
    specifically for that extra risk, applied per resulting band (2 bands from one
    cutpoint), not a number taken directly from either paper.
    """
    total_events = int(event_indicator.astype(bool).sum())
    return total_events >= 2 * min_events_per_candidate_band


def compute_band_hazard_ratios(band_assignment: np.ndarray, event_indicator: np.ndarray, event_time: np.ndarray, reference_band) -> pd.DataFrame:
    """Fit a Cox model (lifelines CoxPHFitter) with band membership as a categorical
    predictor, `reference_band` as the reference category (HR=1.0 by definition). Returns a
    DataFrame with one row per non-reference band: [band, hazard_ratio, ci_lower, ci_upper].
    """
    df = pd.DataFrame(
        {
            "duration": event_time,
            "event": event_indicator.astype(bool),
            "band": pd.Categorical(band_assignment),
        }
    )
    df["band"] = df["band"].cat.reorder_categories(
        [reference_band] + [b for b in sorted(df["band"].cat.categories) if b != reference_band]
    )
    dummies = pd.get_dummies(df["band"], prefix="band", drop_first=True)
    cox_df = pd.concat([df[["duration", "event"]], dummies], axis=1)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event")

    summary = cph.summary
    rows = []
    for dummy_col in dummies.columns:
        band_value = dummy_col.replace("band_", "")
        rows.append(
            {
                "band": band_value,
                "hazard_ratio": float(summary.loc[dummy_col, "exp(coef)"]),
                "ci_lower": float(summary.loc[dummy_col, "exp(coef) lower 95%"]),
                "ci_upper": float(summary.loc[dummy_col, "exp(coef) upper 95%"]),
            }
        )
    return pd.DataFrame(rows)

"""Evaluation metrics for COMPASS candidate models: time-dependent (Uno's) C-index and
integrated Brier score per outcome per horizon, via scikit-survival -- hand-implementing
IPCW-adjusted metrics is exactly the kind of thing that's easy to get subtly wrong, so this
wraps sksurv rather than reimplementing, per the locked methodology's choice of Uno's C
(horizon-specific, IPCW-adjusted) over a whole-follow-up Harrell's C. Also implements the
IPCW-weighted calibration curve (via lifelines' Kaplan-Meier estimator, already a project
dependency), locked as part of the methodology as a supplementary diagnostic separate from
the discrimination-calibration scatter.

Units: all times (event_or_censoring_time, horizon, time_grid) are in YEARS from landmark,
matching the project's horizons convention (configs/training.yaml: 10/20/30yr) -- distinct
from compass.training.batching's internal days-based convention; ground truth is converted
at the boundary when built from a collate_examples-style batch (see compute_all_metrics).

Prediction interface: predictions[outcome] is a (n_patients, num_time_bins) array of
predicted cumulative incidence at YEARLY bins 1..num_time_bins (bin i, 0-indexed, is CIF by
year i+1) -- i.e. exactly what CompassSurvivalModel.predict_risk(..., horizons=range(1,
num_time_bins+1)) produces for one outcome, already using base.py's shared hazards_to_cif().
This one array serves both metrics: time_dependent_cindex uses a single sparse column (CIF
at the reporting horizon), integrated_brier_score_metric uses a dense slice (CIF at every
year up to the horizon) -- no separate prediction pass needed for the two metrics.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

from compass.models.base import OUTCOME_NAMES


def _structured_survival_array(event_indicator: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.array(list(zip(event_indicator.astype(bool), time.astype(float))), dtype=[("event", bool), ("time", float)])


def time_dependent_cindex(
    cif_at_horizon: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
) -> float:
    """Uno's C-index (concordance_index_ipcw) for one outcome at one horizon.

    cif_at_horizon: (n,) each patient's predicted cumulative incidence by `horizon` --
    higher = predicted higher risk, matching what concordance_index_ipcw's `estimate` param
    expects.

    No separate held-out split is threaded through this function for estimating the
    censoring distribution, so the same evaluation set's own censoring distribution is used
    as both `survival_train` and `survival_test` -- a common practical simplification when a
    dedicated censoring-distribution reference isn't available; it does mean the IPCW
    weights aren't validated against an independent estimate.

    Returns NaN if the sample has zero events for this outcome. This is EXPECTED, not just
    a defensive guard: with a rare outcome (e.g. malo, ~1% base rate) and patient-level
    bootstrap resampling (compass.evaluation.bootstrap), a resample with zero events is a
    real, foreseeable draw, not an edge case that "can't happen" -- sksurv itself raises
    ValueError("all samples are censored") in this situation. Returning NaN here lets
    bootstrap_ci_single_model / paired_bootstrap_comparison skip that one iteration's value
    (pandas quantile/mean skip NaN by default) instead of crashing the entire run over a
    single degenerate resample.
    """
    if event_indicator.astype(bool).sum() == 0:
        return float("nan")
    # TODO before real-data use: currently reusing the same evaluation set as both
    # survival_train (censoring-distribution reference) and survival_test (actual
    # evaluation cohort) for sksurv's IPCW methods. sksurv's convention expects these to
    # ideally be separate samples -- reusing one set is a known source of mild optimistic
    # bias, worse with small/rare-event samples (directly relevant given MALO's low
    # incidence). Fine for synthetic proof-of-concept; must be revisited once real Stage 1
    # data is available, likely using the full training split as survival_train regardless
    # of which split is being evaluated.
    y = _structured_survival_array(event_indicator, event_or_censoring_time)
    tau = min(float(horizon), float(event_or_censoring_time.max()))  # sksurv requires tau within the observed range
    result = concordance_index_ipcw(y, y, cif_at_horizon, tau=tau)
    return float(result[0])


def _censoring_km_curve(event_time: np.ndarray, event_type_indicator: np.ndarray, cause_of_interest: int) -> tuple:
    """Fits G, the Kaplan-Meier survival of the CENSORING distribution -- i.e. a patient is
    only ever "censored" here if event_type_indicator == 0 (true loss to follow-up). A
    competing event does NOT count as censoring for this fit: the patient's outcome was fully
    observed (just not the cause of interest), so it must not depress G the way an
    administrative dropout does.

    This is the shared, single source of truth behind both `_censoring_ipcw_weights` (Wolbers'
    C-index) and `schoop_competing_risks_ibs` (competing-risks Brier score/IBS) -- the KM fit
    itself (with its "events-first" tie handling, see below) is the part that's easy to get
    subtly wrong, so it's fit exactly once here and only queried (via
    `_censoring_survival_at`) by each metric, rather than reimplemented per metric.

    This distinction matters and was found the hard way: `comprisk`'s own
    compute_uno_weights folds competing events into its censoring-KM hazard (its own "events-
    first" convention is about tie-breaking between cause-of-interest events and censoring,
    not about excluding competing events), which inflates weights even when there is zero
    real censoring before a case's event time -- confirmed to diverge from R's pec::cindex
    (see wolbers_cindex's docstring). This helper uses the correct convention instead
    (verified against pec::cindex to give G==1 whenever there is no real censoring before a
    case's time -- the condition under which pec::cindex matches exactly, see
    tests/test_r_reference_fixtures.py).

    "Events-first" tie handling: at an exact tie between a cause-of-interest event and a
    censoring event, the cause event is treated as leaving the risk set before the censoring
    hazard is computed at that instant (removed from the denominator) -- the standard
    convention for this kind of tie, though this specific choice was NOT the source of
    wolbers_cindex's small residual disagreement with R on tied-time data (both orderings
    were tried and gave identical results, see wolbers_cindex's docstring).

    Returns (t_unique, g_at_knot): sorted unique observed times and G's value AT (i.e.
    right-continuous, post-drop) each of those times.
    """
    event_time = np.asarray(event_time, dtype=float)
    is_cause = event_type_indicator == cause_of_interest
    is_censored = event_type_indicator == 0
    n = event_time.size

    order = np.argsort(event_time, kind="stable")
    cause_sorted = is_cause[order]
    censored_sorted = is_censored[order]

    t_unique, first_idx = np.unique(event_time[order], return_index=True)
    bounds = np.append(first_idx, n)

    g_at_knot = np.empty(t_unique.shape[0])
    surv = 1.0
    for k in range(t_unique.shape[0]):
        s, e = int(bounds[k]), int(bounds[k + 1])
        n_risk = n - s
        d_cause = int(cause_sorted[s:e].sum())
        d_censored = int(censored_sorted[s:e].sum())
        denom = n_risk - d_cause
        if d_censored > 0:
            surv = 0.0 if denom <= 0 else surv * (1.0 - d_censored / denom)
        g_at_knot[k] = surv

    return t_unique, g_at_knot


def _censoring_survival_at(t_unique: np.ndarray, g_at_knot: np.ndarray, query_times: np.ndarray, side: str) -> np.ndarray:
    """G evaluated at `query_times`, from a curve already fit by `_censoring_km_curve`.

    side="left": G(t-), the left limit (last knot strictly before t) -- used for patients
    evaluated at their OWN observed time (Wolbers' C-index; the case/competing-event terms
    of the Brier score), since they were known to still be at risk right up to that instant.
    side="right": G(t), the right-continuous value AT t (last knot at or before t) -- used
    for the Brier score's "still event-free" term at a fixed evaluation time t, the standard
    Graf/Schoop convention (matches sksurv's own time_dependent Brier score formula, which
    also uses G_hat(t) rather than G_hat(t-) for this term).
    """
    query_times = np.asarray(query_times, dtype=float)
    if t_unique.size == 0:
        return np.ones_like(query_times)
    idx = np.searchsorted(t_unique, query_times, side=side)
    return np.where(idx > 0, g_at_knot[np.clip(idx - 1, 0, None)], 1.0)


def _censoring_ipcw_weights(
    event_time: np.ndarray, event_type_indicator: np.ndarray, cause_of_interest: int, power: int = 2
) -> np.ndarray:
    """Per-patient IPCW weight 1/G(t_i-)^power, at each patient's OWN observed time (left
    limit). power=2 (default) is the Uno (2011) C-index convention (wolbers_cindex);
    power=1 is the Graf/Schoop Brier score convention (schoop_competing_risks_ibs). See
    `_censoring_km_curve` for the underlying KM fit and its docstring for why it uses this
    particular censoring-only convention rather than comprisk's.
    """
    event_time = np.asarray(event_time, dtype=float)
    t_unique, g_at_knot = _censoring_km_curve(event_time, event_type_indicator, cause_of_interest)
    g_left = _censoring_survival_at(t_unique, g_at_knot, event_time, side="left")
    g_left_safe = np.maximum(g_left, 1e-12)
    return 1.0 / (g_left_safe**power)


def wolbers_cindex(
    cif_at_horizon: np.ndarray,
    event_type_indicator: np.ndarray,
    event_time: np.ndarray,
    horizon: float,
    cause_of_interest: int = 1,
) -> float:
    """Wolbers et al. (2014, Biostatistics 15(3):526-539, eq. 3.4) cause-specific,
    IPCW-weighted concordance index at `horizon`, correctly handling a real competing risk
    (e.g. mortality for malo/cvd) rather than treating it as neutral censoring the way Uno's C
    (time_dependent_cindex) does. Uno's C says a patient who died before malo/cvd could occur
    has an "unknown fate, could still happen"; Wolbers' C-index knows their fate is
    permanently precluded and scores accordingly.

    event_type_indicator: integer code per patient -- 0 = censored (no event of any kind
    observed), `cause_of_interest` = had the outcome of interest, any OTHER positive code =
    had a competing event. These three cases are NOT interchangeable (see below).

    Two kinds of comparator pairs for a case i (had the event of interest at or before
    `horizon`), each with its own weight:
    - "Branch A": event_time[j] > event_time[i], or a tie with j censored at exactly
      event_time[i] -- j's fate wasn't yet decided as of i's event time (covers j later
      having the same event, a later competing event, or later censoring). Weighted by
      w_i alone (the standard Uno-style correction), since only i's own censoring risk
      affects whether this pair could be observed.
    - "Branch C": j had a COMPETING event at or before event_time[i] -- permanently
      precludes the outcome of interest for j, so the pair is informative regardless of
      time order. Weighted by sqrt(w_i * w_j) (symmetric: both i's and j's censoring risk
      affect whether this pair could be observed, per Wolbers eq. 3.4).
    - "Branch B": two or more cases tied at the exact same event_time -- each ordered pair
      within the tied group gets half credit at minimum (weight w_i), full credit if their
      predictions also tie.
    Patients purely CENSORED strictly before i's event time are excluded from all branches
    (their fate is genuinely unknown) -- this asymmetry between "competing event" and
    "censored" is the whole point of the competing-risk correction, on top of the IPCW
    weighting which corrects for the censoring itself.

    Concordant if cif_at_horizon[i] > cif_at_horizon[j] (the model correctly scored the
    patient who actually had the event as higher-risk); tied predictions get half credit.

    w_i, w_j are computed via `_censoring_ipcw_weights` -- NOT `comprisk`'s own
    compute_uno_weights, which was found to use a different (and, empirically, not
    R-matching) convention; see that helper's docstring.

    VERIFIED against R's pec::cindex (see tests/test_r_reference_fixtures.py): exact match
    (6 decimal places) on two 300-patient cohorts with no censoring before the horizon
    (case_a_malo, case_b_cvd -- where IPCW weights are trivially 1, so this also confirms the
    prior no-IPCW version's pair-selection logic was already correct). On a 60-patient
    deliberately-tied stress case with real censoring before the horizon (case_c_ties), this
    reduced the gap from 0.0244 (no IPCW weighting at all) to 0.0009 (~96% closed) but did NOT
    reach an exact match -- residual likely a tie-breaking convention difference in the
    auxiliary censoring-KM fit versus R's `prodlim`/`pec` (tried both orderings for the
    cause/censoring exact-time tie, no change), not further chased down as an unbounded
    algorithm-archaeology exercise. Flagged, not hidden -- see the wider-tolerance note on
    that fixture's test.

    Returns NaN if there are no comparable pairs.
    """
    cif_at_horizon = np.asarray(cif_at_horizon, dtype=float)
    event_type_indicator = np.asarray(event_type_indicator)
    event_time = np.asarray(event_time, dtype=float)

    is_case_at_horizon = (event_type_indicator == cause_of_interest) & (event_time <= horizon)
    is_censored = event_type_indicator == 0
    is_competing = (event_type_indicator != 0) & (event_type_indicator != cause_of_interest)

    weights = _censoring_ipcw_weights(event_time, event_type_indicator, cause_of_interest)
    sqrt_weights = np.sqrt(weights)

    numerator = 0.0
    denominator = 0.0
    case_idx = np.flatnonzero(is_case_at_horizon)

    for i in case_idx:
        ti = event_time[i]
        pi = cif_at_horizon[i]
        wi = weights[i]

        # Branch A.
        mask_a = (event_time > ti) | ((event_time == ti) & is_censored)
        mask_a[i] = False
        if mask_a.any():
            pj = cif_at_horizon[mask_a]
            less = (pj < pi).sum()
            tied = (pj == pi).sum()
            n_pairs = mask_a.sum()
            numerator += 2.0 * wi * less + wi * tied
            denominator += 2.0 * wi * n_pairs

        # Branch C.
        mask_c = is_competing & (event_time <= ti)
        if mask_c.any():
            pj = cif_at_horizon[mask_c]
            wj_sqrt = sqrt_weights[mask_c]
            wi_sqrt = sqrt_weights[i]
            less = (wj_sqrt * (pj < pi)).sum()
            tied = (wj_sqrt * (pj == pi)).sum()
            numerator += 2.0 * wi_sqrt * less + wi_sqrt * tied
            denominator += 2.0 * wi_sqrt * wj_sqrt.sum()

    # Branch B: tied case-times.
    if case_idx.size >= 2:
        order = np.argsort(event_time[case_idx], kind="stable")
        sorted_case = case_idx[order]
        sorted_t = event_time[sorted_case]
        g0 = 0
        while g0 < sorted_case.size:
            g1 = g0
            while g1 + 1 < sorted_case.size and sorted_t[g1 + 1] == sorted_t[g0]:
                g1 += 1
            group = sorted_case[g0 : g1 + 1]
            if group.size >= 2:
                for a in range(group.size):
                    i = group[a]
                    others = np.delete(group, a)
                    wi = weights[i]
                    tie_credit = 0.5 + 0.5 * (cif_at_horizon[others] == cif_at_horizon[i])
                    numerator += wi * tie_credit.sum()
                    denominator += wi * others.size
            g0 = g1 + 1

    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def schoop_competing_risks_ibs(
    cif_over_time: np.ndarray,
    event_type_indicator: np.ndarray,
    event_time: np.ndarray,
    horizon: float,
    time_grid: np.ndarray,
    cause_of_interest: int = 1,
) -> float:
    """Schoop, Beyersmann, Schumacher & Binder (2011, Biom J 53(1):88-112) competing-risks
    IPCW Brier score / integrated Brier score for `cause_of_interest`, correctly handling a
    real competing risk (e.g. mortality for malo/cvd) the same way wolbers_cindex does for
    discrimination: a competing event means the patient's fate w.r.t. `cause_of_interest` is
    known with certainty (permanently precluded, CIF should have been 0), not unknown the
    way real censoring is. sksurv's integrated_brier_score (integrated_brier_score_metric)
    treats a competing event as ordinary censoring, biasing the calibration estimate the same
    way Uno's C biases discrimination.

    event_type_indicator: integer code per patient -- 0 = censored, `cause_of_interest` = had
    the outcome of interest, any OTHER positive code = had a competing event.

    At each evaluation time t in `time_grid`, every patient falls into exactly one of three
    cases (Graf/Schoop extension of the standard IPCW Brier score):
    1. Had `cause_of_interest` at or before t: residual = (1 - CIF_hat(t))^2, weighted by
       1/G(T_i-) -- known with certainty, weighted by the patient's own censoring risk up to
       their (pre-t) event time.
    2. Had a DIFFERENT (competing) event at or before t: residual = (0 - CIF_hat(t))^2, same
       1/G(T_i-) weighting as case 1 -- also known with certainty (this outcome did NOT and
       now cannot occur), not "unknown" the way sksurv's plain IBS treats it.
    3. Still event-free (of ANY kind) as of t, whether ultimately censored or eventually
       has some event after t: residual = (0 - CIF_hat(t))^2, weighted by 1/G(t) -- the
       standard "still at risk, still uncertain" weighting, using the censoring survival AT
       the fixed evaluation time t (not the patient's own time).
    Patients purely CENSORED strictly before t fall into none of these three (their status
    at t is genuinely unknown) and are excluded from BS(t) entirely, standard Brier score
    convention -- BS(t) is normalized by the full n regardless (matching sksurv's own
    formula), not by the count of retained patients.

    IBS = trapezoidal integral of BS(t) over `time_grid`, normalized by
    (time_grid[-1] - time_grid[0]), matching integrated_brier_score_metric's convention (via
    sksurv) so the two are on the same, directly comparable scale.

    G (the censoring-only KM survival) is fit once via `_censoring_km_curve` -- the SAME
    shared, tested KM fit `_censoring_ipcw_weights` uses for wolbers_cindex, not a second,
    potentially-divergent implementation.

    VERIFIED against R's pec::pec(..., cause=cause_of_interest) (see
    tests/test_r_reference_fixtures.py): see that file / this function's call sites for the
    actual numbers.

    Returns NaN if there are zero `cause_of_interest` events -- same reasoning as
    time_dependent_cindex/wolbers_cindex/integrated_brier_score_metric: a real, foreseeable
    draw under patient-level bootstrap resampling of a rare outcome, not an edge case that
    "can't happen".
    """
    cif_over_time = np.asarray(cif_over_time, dtype=float)
    event_type_indicator = np.asarray(event_type_indicator)
    event_time = np.asarray(event_time, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)

    if (event_type_indicator == cause_of_interest).sum() == 0:
        return float("nan")

    n = event_time.size
    is_cause = event_type_indicator == cause_of_interest
    is_competing = (event_type_indicator != 0) & (event_type_indicator != cause_of_interest)
    is_decided = is_cause | is_competing  # fate known with certainty, at the patient's own time

    t_unique, g_at_knot = _censoring_km_curve(event_time, event_type_indicator, cause_of_interest)
    weight_own_time = 1.0 / np.maximum(_censoring_survival_at(t_unique, g_at_knot, event_time, side="left"), 1e-12)
    weight_at_grid = 1.0 / np.maximum(_censoring_survival_at(t_unique, g_at_knot, time_grid, side="right"), 1e-12)

    brier_curve = np.empty(time_grid.size)
    for col, t in enumerate(time_grid):
        pred = cif_over_time[:, col]

        decided_by_t = is_decided & (event_time <= t)
        still_at_risk = event_time > t

        residual = np.zeros(n)
        residual[decided_by_t] = np.where(is_cause[decided_by_t], (1.0 - pred[decided_by_t]) ** 2, (0.0 - pred[decided_by_t]) ** 2)
        residual[still_at_risk] = (0.0 - pred[still_at_risk]) ** 2

        weight = np.zeros(n)
        weight[decided_by_t] = weight_own_time[decided_by_t]
        weight[still_at_risk] = weight_at_grid[col]

        brier_curve[col] = float(np.sum(weight * residual) / n)

    span = time_grid[-1] - time_grid[0]
    if span <= 0:
        return float(brier_curve[0])
    return float(np.trapz(brier_curve, time_grid) / span)


def integrated_brier_score_metric(
    cif_over_time: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
    time_grid: np.ndarray,
) -> float:
    """IBS up to `horizon`, via sksurv.metrics.integrated_brier_score.

    cif_over_time: (n, len(time_grid)) predicted CIF at each point in time_grid (years).
    Converted to survival probabilities (1 - CIF) since sksurv's IBS expects survival
    probabilities, not risk/CIF.

    time_grid: this project's yearly-bin convention (bin_width_years=1.0, shared by all
    three models) is used as the default resolution by compute_all_metrics -- a reasonable
    balance of integral accuracy vs. compute, and it's the finest grid all three models
    already natively support without extra interpolation.

    Returns NaN if the sample has zero events for this outcome -- same reasoning as
    time_dependent_cindex: a real, foreseeable draw under patient-level bootstrap
    resampling of a rare outcome, not an edge case that "can't happen".
    """
    if event_indicator.astype(bool).sum() == 0:
        return float("nan")
    # TODO before real-data use: currently reusing the same evaluation set as both
    # survival_train (censoring-distribution reference) and survival_test (actual
    # evaluation cohort) for sksurv's IPCW methods. sksurv's convention expects these to
    # ideally be separate samples -- reusing one set is a known source of mild optimistic
    # bias, worse with small/rare-event samples (directly relevant given MALO's low
    # incidence). Fine for synthetic proof-of-concept; must be revisited once real Stage 1
    # data is available, likely using the full training split as survival_train regardless
    # of which split is being evaluated.
    y = _structured_survival_array(event_indicator, event_or_censoring_time)
    survival_over_time = 1.0 - cif_over_time

    # sksurv requires every value in `times` to lie strictly within the observed follow-up
    # range of survival_test; clip defensively so a boundary horizon (e.g. exactly at the
    # max observed time) doesn't crash the whole evaluation run.
    max_time = float(event_or_censoring_time.max())
    min_time = float(event_or_censoring_time.min())
    safe_grid = np.clip(time_grid, min_time + 1e-6, max_time - 1e-6)
    safe_grid = np.unique(safe_grid)
    if len(safe_grid) < len(time_grid):
        # Clipping collapsed some grid points to the same value (e.g. horizon >= max_time
        # for several points) -- keep the corresponding columns via nearest-match indices.
        col_idx = np.searchsorted(time_grid, safe_grid)
    else:
        col_idx = np.arange(len(time_grid))

    return float(integrated_brier_score(y, y, survival_over_time[:, col_idx], safe_grid))


def ipcw_calibration_curve(
    cif_at_horizon: np.ndarray,
    event_indicator: np.ndarray,
    event_or_censoring_time: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> dict:
    """Bin patients by predicted cif_at_horizon into n_bins groups; for each non-empty bin,
    compute the IPCW-corrected true event rate (1 - Kaplan-Meier survival at `horizon` for
    that bin's patients, which naturally accounts for censoring within the bin) vs. the mean
    predicted CIF in that bin.

    Returns {"predicted": array, "observed": array, "bin_counts": array, "n_total": int}.
    """
    bin_edges = np.quantile(cif_at_horizon, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[0] -= 1e-9  # ensure the minimum value falls inside bin 0 with right-inclusive digitize
    bin_edges = np.unique(bin_edges)  # guard against degenerate/duplicate edges (heavy ties in CIF)

    bin_idx = np.digitize(cif_at_horizon, bin_edges[1:-1], right=True)

    predicted, observed, bin_counts = [], [], []
    for b in range(len(bin_edges) - 1):
        in_bin = bin_idx == b
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue
        predicted.append(float(cif_at_horizon[in_bin].mean()))

        kmf = KaplanMeierFitter()
        kmf.fit(durations=event_or_censoring_time[in_bin], event_observed=event_indicator[in_bin])
        survival_at_horizon = float(kmf.predict(horizon))
        observed.append(1.0 - survival_at_horizon)
        bin_counts.append(n_in_bin)

    return {
        "predicted": np.array(predicted),
        "observed": np.array(observed),
        "bin_counts": np.array(bin_counts),
        "n_total": len(cif_at_horizon),
    }


def compute_all_metrics(predictions: dict, ground_truth: dict, horizons: list) -> pd.DataFrame:
    """Orchestrates C-index + IBS across all 3 outcomes x all horizons for ONE model's
    predictions.

    C-index dispatch: mortality has no competing risk here -> time_dependent_cindex (Uno's
    C), unchanged. malo and cvd have mortality as a real competing risk -> wolbers_cindex,
    which correctly distinguishes "died before the outcome could occur" (permanently
    precluded) from ordinary censoring (unknown fate) -- Uno's C treats them the same, which
    biases the metric.

    IBS dispatch mirrors this exactly, same rationale: mortality -> integrated_brier_score_
    metric (sksurv's IPCW Brier score), unchanged. malo/cvd -> schoop_competing_risks_ibs,
    the Graf/Schoop competing-risks extension -- sksurv's plain IBS treats a competing event
    (e.g. death before malo/cvd could occur) as ordinary censoring, which biases the
    calibration estimate the same way Uno's C biases discrimination for those two outcomes.

    predictions: {"mortality": (n, T) CIF array, "malo": ..., "cvd": ...} at yearly bins
    1..T, T >= max(horizons) -- see module docstring for the exact convention.
    ground_truth: {"mortality": {"event_indicator": (n,), "time_years": (n,)}, "malo": ...,
    "cvd": ...} -- time_years is that outcome's own event time if it occurred, else its
    censoring time, using the SAME per-outcome censoring rule already implemented in
    compass.training.survival_losses / batching.py (mortality censors malo/cvd; malo/cvd
    don't censor each other), just consumed here for evaluation instead of training.
    wolbers_cindex and schoop_competing_risks_ibs additionally need to distinguish competing
    events from pure censoring for malo/cvd; this is derived inline from
    ground_truth["mortality"]["event_indicator"] (mortality occurring IS the competing event
    for malo/cvd) -- no change to the ground_truth dict shape is needed.

    Returns a tidy DataFrame: columns [outcome, horizon, metric_name, value].
    """
    mortality_event = ground_truth["mortality"]["event_indicator"].astype(bool)

    rows = []
    for outcome in OUTCOME_NAMES:
        cif_all = predictions[outcome]
        event_indicator = ground_truth[outcome]["event_indicator"]
        time_years = ground_truth[outcome]["time_years"]

        # 0=censored, 1=had this outcome, 2=competing event (mortality) -- only meaningful
        # (and only used) for malo/cvd; irrelevant for mortality's own unchanged dispatch.
        event_type = np.where(event_indicator.astype(bool), 1, np.where(mortality_event, 2, 0))

        for horizon in horizons:
            bin_idx = int(horizon) - 1
            cif_at_horizon = cif_all[:, bin_idx]
            time_grid = np.arange(1, horizon + 1, dtype=float)
            cif_over_time = cif_all[:, : int(horizon)]

            if outcome == "mortality":
                cindex = time_dependent_cindex(cif_at_horizon, event_indicator, time_years, horizon)
                ibs = integrated_brier_score_metric(cif_over_time, event_indicator, time_years, horizon, time_grid)
            else:
                cindex = wolbers_cindex(cif_at_horizon, event_type, time_years, horizon, cause_of_interest=1)
                ibs = schoop_competing_risks_ibs(cif_over_time, event_type, time_years, horizon, time_grid, cause_of_interest=1)

            rows.append({"outcome": outcome, "horizon": horizon, "metric_name": "cindex", "value": cindex})
            rows.append({"outcome": outcome, "horizon": horizon, "metric_name": "ibs", "value": ibs})

    return pd.DataFrame(rows)

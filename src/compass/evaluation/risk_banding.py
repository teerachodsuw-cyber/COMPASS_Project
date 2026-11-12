"""Three risk bands (low / intermediate / high) per model and outcome from the predicted 10-yr
risk, with two cut-points chosen on the VALIDATION split and then applied unchanged to any
later split. Aggregates only.

FIT (validation split only -- fit_band_cutpoints / fit_all_cutpoints):
- candidates: the predicted-risk percentiles 10, 11, ..., 90 of the fitting sample (unique
  values); every pair c1 < c2 is tried.
- bands: low = risk <= c1, intermediate = c1 < risk <= c2, high = risk > c2 (assign_bands).
- a pair is eligible only if each band holds >= min_band_fraction (default 10%) of patients.
- the eligible pair with the largest 3-group test statistic wins:
    mortality (no competing risk): log-rank (lifelines.statistics.multivariate_logrank_test);
    MALO / CVD (death competes): Gray's K-sample test (comprisk.gray_test), which compares
    cumulative incidence functions with death as a competing event. Log-rank would treat
    competing deaths as independent censoring -- wrong for these outcomes. comprisk's Gray's
    test matches R's cmprsk::cuminc to 1e-9 (tests/test_r_reference_fixtures.py).
  Ties in the statistic: the first pair in (c1, c2) order wins (deterministic).
- no eligible pair -> NoValidCutpointsError (never a silent fallback to another rule).
- an outcome with zero events in the fitting sample has no separation to maximise: it is
  recorded as not fitted (status in cutpoints.json, "n/a" in the tables), not banded by a
  substitute rule.
- the reported p-value is that of the WINNING pair, i.e. after searching many pairs: it is
  optimistic and not a valid test of association.

APPLY (any split): assign_bands(scores, saved cut-points) never looks at outcomes or at the
distribution of the scores it is given; evaluate.py loads cutpoints.json for the test split
and never calls a fit function there.

Follow-up is truncated at the horizon for both the search and the band summaries (later
times -> horizon, later events -> censored), so bands are compared on the same 10-yr window
the predicted risk refers to.

Per band (band_summary): n, events within the horizon, observed 10-yr risk (Kaplan-Meier for
mortality, Aalen-Johansen with death competing for MALO / CVD), and hazard ratios vs the low
band: Cox for mortality, CAUSE-SPECIFIC Cox for MALO / CVD (competing deaths censored), so
the MALO / CVD HRs are cause-specific hazard ratios, not subdistribution HRs.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from comprisk.gray_test import gray_test
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import multivariate_logrank_test

from compass.evaluation.calibration import at_risk_horizon_data, estimator_name, observed_risk
from compass.models.base import DEATH_OUTCOME, OUTCOME_NAMES

BAND_NAMES = ("low", "intermediate", "high")
DEFAULT_PERCENTILES = tuple(range(10, 91))
DEFAULT_MIN_BAND_FRACTION = 0.10
CUTPOINTS_FILE = "cutpoints.json"


class NoValidCutpointsError(ValueError):
    """No pair of candidate cut-points gives three bands that each meet the minimum size."""


class CutpointsMismatchError(ValueError):
    """Saved cut-points do not belong to this evaluation (split, data, horizon or models)."""


def separation_test_name(outcome: str) -> str:
    return "log-rank" if outcome == DEATH_OUTCOME else "Gray"


def hazard_ratio_label(outcome: str) -> str:
    return "Cox" if outcome == DEATH_OUTCOME else "cause-specific Cox"


def truncate_at_horizon(time, event_type, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type).astype(int)
    return np.minimum(time, float(horizon)), np.where(time > horizon, 0, event_type)


def assign_bands(scores, cutpoints) -> np.ndarray:
    """APPLY step: band index per score (0 low, 1 intermediate, 2 high) from ALREADY-FITTED
    cut-points (c1, c2). Uses nothing but the scores and the two values -- safe on any split."""
    c1, c2 = (float(c) for c in cutpoints)
    if not c1 < c2:
        raise ValueError(f"cut-points must be increasing, got {c1}, {c2}")
    return np.digitize(np.asarray(scores, dtype=float), [c1, c2], right=True)


def _statistic(time, event_type, bands, outcome: str) -> tuple[float, float]:
    if outcome == DEATH_OUTCOME:
        r = multivariate_logrank_test(time, bands, event_type == 1)
        return float(r.test_statistic), float(r.p_value)
    r = gray_test(time, event_type, bands, cause=1)
    return float(r.stat), float(r.pvalue)


def fit_band_cutpoints(scores, time, event_type, outcome: str, horizon: float,
                       percentiles=DEFAULT_PERCENTILES, min_band_fraction: float = DEFAULT_MIN_BAND_FRACTION) -> dict:
    """FIT step (validation split only): the two cut-points maximising the 3-group separation
    (log-rank for mortality, Gray's test for MALO / CVD), each band >= min_band_fraction of
    patients. scores: predicted risk at the horizon; time (years), event_type (1 / 2 / 0):
    at-risk rows. Raises NoValidCutpointsError if no pair is eligible."""
    scores = np.asarray(scores, dtype=float)
    time, event_type = truncate_at_horizon(time, event_type, horizon)
    n = scores.size
    min_n = math.ceil(min_band_fraction * n)
    candidates = np.unique(np.quantile(scores, np.asarray(percentiles, dtype=float) / 100.0))

    best, n_eligible = None, 0
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1:]:
            bands = assign_bands(scores, (c1, c2))
            counts = np.bincount(bands, minlength=3)
            if counts.min() < min_n:
                continue
            n_eligible += 1
            stat, p = _statistic(time, event_type, bands, outcome)
            if np.isfinite(stat) and (best is None or stat > best["statistic"]):
                best = {"cutpoints": [float(c1), float(c2)], "statistic": stat, "p_value": p,
                        "band_counts": [int(c) for c in counts]}
    if best is None:
        raise NoValidCutpointsError(
            f"{outcome}: no pair of the {candidates.size} candidate cut-points (percentiles "
            f"{min(percentiles)}-{max(percentiles)}) gives three bands of >= {min_band_fraction:.0%} of {n} patients")
    return {
        "status": "fitted",
        "cutpoints": best["cutpoints"],
        "test": separation_test_name(outcome),
        "statistic": best["statistic"],
        "p_value_after_search": best["p_value"],
        "n_fit": int(n),
        "n_events_fit": int((event_type == 1).sum()),
        "band_fractions_fit": [c / n for c in best["band_counts"]],
        "n_candidates": int(candidates.size),
        "n_pairs_eligible": int(n_eligible),
    }


def band_hazard_ratios(bands, time, event_type, outcome: str) -> list[dict]:
    """HR (95% CI) of the intermediate and high bands vs the low band: Cox (mortality) or
    cause-specific Cox (MALO / CVD: competing deaths censored). time / event_type already
    truncated at the horizon. A band with no events, or a fit that does not converge, gives
    NaN with the reason -- never a silently substituted estimate."""
    bands = np.asarray(bands)
    event = (np.asarray(event_type) == 1).astype(int)
    label = hazard_ratio_label(outcome)
    nan_rows = lambda reason: [{"band": BAND_NAMES[b], "hazard_ratio": float("nan"), "ci_lower": float("nan"),
                                "ci_upper": float("nan"), "model": label, "note": reason} for b in (1, 2)]
    events_per_band = [int(event[bands == b].sum()) for b in range(3)]
    if min(events_per_band) == 0:
        return nan_rows("not estimable: a band has no events")
    df = pd.DataFrame({"duration": np.asarray(time, dtype=float), "event": event,
                       "intermediate": (bands == 1).astype(int), "high": (bands == 2).astype(int)})
    try:
        cph = CoxPHFitter().fit(df, duration_col="duration", event_col="event")
    except ConvergenceError:
        return nan_rows("Cox fit did not converge")
    s = cph.summary
    return [{"band": BAND_NAMES[b], "hazard_ratio": float(s.loc[col, "exp(coef)"]),
             "ci_lower": float(s.loc[col, "exp(coef) lower 95%"]), "ci_upper": float(s.loc[col, "exp(coef) upper 95%"]),
             "model": label, "note": ""} for b, col in ((1, "intermediate"), (2, "high"))]


def band_summary(scores, cutpoints, time, event_type, outcome: str, horizon: float) -> list[dict]:
    """Per band: n, fraction, events within the horizon, competing deaths within the horizon
    (MALO / CVD), observed horizon risk (KM / Aalen-Johansen), and HR vs low (low = 1)."""
    time, event_type = truncate_at_horizon(time, event_type, horizon)
    bands = assign_bands(scores, cutpoints)
    hrs = {r["band"]: r for r in band_hazard_ratios(bands, time, event_type, outcome)}
    n = bands.size
    rows = []
    for b, name in enumerate(BAND_NAMES):
        sel = bands == b
        hr = hrs.get(name, {"hazard_ratio": 1.0, "ci_lower": float("nan"), "ci_upper": float("nan"),
                            "model": hazard_ratio_label(outcome), "note": "reference"})
        rows.append({
            "band": name,
            "n": int(sel.sum()),
            "fraction": float(sel.sum() / n) if n else float("nan"),
            "n_events": int((event_type[sel] == 1).sum()),
            "n_competing": int((event_type[sel] == 2).sum()),
            "observed_risk": observed_risk(time[sel], event_type[sel], horizon, outcome) if sel.any() else float("nan"),
            "observed_estimator": estimator_name(outcome),
            "hazard_ratio": hr["hazard_ratio"],
            "hr_ci_lower": hr["ci_lower"],
            "hr_ci_upper": hr["ci_upper"],
            "hr_model": hr["model"],
            "hr_note": hr["note"],
        })
    return rows


# ---- orchestration over models x outcomes ------------------------------------------------


def fit_all_cutpoints(predictions: dict, ground_truth: dict, horizon: int, percentiles=DEFAULT_PERCENTILES,
                      min_band_fraction: float = DEFAULT_MIN_BAND_FRACTION) -> dict:
    """FIT on the evaluated (validation) split: {model: {outcome: fit result}}. An outcome with
    zero events in the fitting sample -> {"status": "not fitted: no events ..."}."""
    out = {}
    for model, preds in predictions.items():
        out[model] = {}
        for outcome in OUTCOME_NAMES:
            scores, time, event_type = at_risk_horizon_data(preds[outcome], ground_truth, outcome, horizon)
            n_events = int(((event_type == 1) & (time <= horizon)).sum())
            if n_events == 0:
                out[model][outcome] = {"status": "not fitted: no events in the fitting split", "n_fit": int(scores.size)}
                continue
            out[model][outcome] = fit_band_cutpoints(scores, time, event_type, outcome, horizon, percentiles, min_band_fraction)
    return out


def summarise_all_bands(predictions: dict, ground_truth: dict, fitted: dict, horizon: int) -> dict:
    """APPLY saved cut-points: {model: {outcome: band rows or {"status": ...}}}."""
    out = {}
    for model, preds in predictions.items():
        out[model] = {}
        for outcome in OUTCOME_NAMES:
            fit = fitted[model][outcome]
            if fit.get("status") != "fitted":
                out[model][outcome] = {"status": fit.get("status", "not fitted")}
                continue
            scores, time, event_type = at_risk_horizon_data(preds[outcome], ground_truth, outcome, horizon)
            out[model][outcome] = band_summary(scores, fit["cutpoints"], time, event_type, outcome, horizon)
    return out


def save_cutpoints(path, fitted: dict, meta: dict) -> Path:
    """Write the fitted cut-points (aggregates: two predicted-risk values per model / outcome)
    with what they were fitted on (meta: split, horizon, split seed, data fingerprint, model
    sources, search settings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "models": fitted}, indent=2))
    return path


def load_cutpoints(path, expected_meta: dict, models: list[str]) -> dict:
    """{model: {outcome: fit}} from a saved cutpoints.json, refusing (CutpointsMismatchError)
    unless it was fitted on the validation split of the same data / split seed / horizon and
    covers the same models from the same sources."""
    path = Path(path)
    if not path.exists():
        raise CutpointsMismatchError(f"no saved cut-points at {path} -- run the validation evaluation first")
    saved = json.loads(path.read_text())
    if saved.get("fitted_on_split") != "val":
        raise CutpointsMismatchError(f"cut-points were fitted on '{saved.get('fitted_on_split')}', not the validation split")
    for key in ("horizon_years", "split_seed", "data_fingerprint"):
        if saved.get(key) != expected_meta.get(key):
            raise CutpointsMismatchError(f"saved cut-points differ from this evaluation in '{key}'")
    missing = [m for m in models if m not in saved["models"]]
    if missing:
        raise CutpointsMismatchError(f"no saved cut-points for model(s) {missing}")
    for m in models:
        if saved.get("model_sources", {}).get(m) != expected_meta.get("model_sources", {}).get(m):
            raise CutpointsMismatchError(f"cut-points for '{m}' were fitted on a different checkpoint directory")
    return {m: saved["models"][m] for m in models}


def _num(x, fmt=".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def format_banding_tables(fitted: dict, bands: dict, split: str) -> str:
    """Markdown: cut-points per model / outcome, then the per-band table."""
    lines = [
        f"Risk bands on the {split} split (cut-points fitted on the validation split"
        + ("; in-sample here, so separation and HRs are optimistic)" if split == "val" else ", applied unchanged)"),
        "",
        "| Model | Outcome | Cut-points (10-yr risk) | Test | Statistic | p (after search) | Fit band fractions |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, per in fitted.items():
        for o in OUTCOME_NAMES:
            f = per[o]
            if f.get("status") != "fitted":
                lines.append(f"| {m} | {o} | {f.get('status')} | | | | |")
                continue
            lines.append(f"| {m} | {o} | {f['cutpoints'][0]:.4f}, {f['cutpoints'][1]:.4f} | {f['test']} | "
                         f"{f['statistic']:.1f} | {f['p_value_after_search']:.2g} | "
                         + " / ".join(f"{x:.0%}" for x in f["band_fractions_fit"]) + " |")
    lines += ["", "| Model | Outcome | Band | n | Events | Observed 10-yr risk | HR vs low (95% CI) | HR model |",
              "|---|---|---|---|---|---|---|---|"]
    for m, per in bands.items():
        for o in OUTCOME_NAMES:
            rows = per[o]
            if isinstance(rows, dict):
                lines.append(f"| {m} | {o} | n/a ({rows['status']}) | | | | | |")
                continue
            for r in rows:
                if r["hr_note"] == "reference":
                    hr = "1 (reference)"
                elif r["hr_note"]:
                    hr = f"n/a ({r['hr_note']})"
                else:
                    hr = f"{r['hazard_ratio']:.2f} ({r['hr_ci_lower']:.2f} to {r['hr_ci_upper']:.2f})"
                lines.append(f"| {m} | {o} | {r['band']} | {r['n']} | {r['n_events']} | {_num(r['observed_risk'])} | "
                             f"{hr} | {r['hr_model']} |")
    return "\n".join(lines)

"""Calibration at the evaluation horizon (10 yr): predicted vs observed risk per decile of
predicted risk, plus the overall observed / expected (O/E) ratio. Aggregates only.

Observed risk:
- mortality (no competing risk): Kaplan-Meier, 1 - S_KM(horizon).
- MALO / CVD (death competes): Aalen-Johansen cumulative incidence of the cause at the horizon,
  with death as a competing event. A KM estimate that censors competing deaths over-estimates
  the cumulative incidence (it assumes those who died could still have had the event), which
  is why the previous KM-based curve (metrics.ipcw_calibration_curve) was replaced.

Both estimators are computed here by the same product-limit recursion over the distinct event
times t <= horizon (ties: events at t happen before censorings at t, the usual convention):
    S(t)    = S(t-) * (1 - d_all(t) / n(t))           (all-cause event-free survival)
    CIF(t)  = CIF(t-) + S(t-) * d_cause(t) / n(t)     (Aalen-Johansen)
With no competing events (mortality) the Aalen-Johansen CIF equals 1 - KM exactly.

Event codes match the evaluation pipeline: 1 = event of interest, 2 = competing death,
0 = censored. Only rows with at_risk == 1 for the outcome are used (as for the metrics).

Deciles: patients sorted by predicted risk at the horizon and split into 10 groups of
(near-)equal size (np.array_split of the ranks), so heavy ties in predicted risk cannot leave
a group empty.
"""

import numpy as np

from compass.models.base import DEATH_OUTCOME, OUTCOME_NAMES


def aalen_johansen_cif(time, event_type, horizon: float, cause: int = 1) -> float:
    """Aalen-Johansen cumulative incidence of `cause` at `horizon`; every other positive code
    is a competing event, 0 is censored. Returns NaN for an empty sample."""
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type)
    if time.size == 0:
        return float("nan")
    is_event = (event_type != 0) & (time <= horizon)
    if not is_event.any():
        return 0.0
    event_times, inverse = np.unique(time[is_event], return_inverse=True)
    d_all = np.bincount(inverse, minlength=event_times.size)
    d_cause = np.bincount(inverse, weights=(event_type[is_event] == cause).astype(float), minlength=event_times.size)
    n_at_risk = time.size - np.searchsorted(np.sort(time), event_times, side="left")  # time >= t
    survival_after = np.cumprod(1.0 - d_all / n_at_risk)
    survival_before = np.concatenate([[1.0], survival_after[:-1]])
    return float(np.sum(survival_before * d_cause / n_at_risk))


def kaplan_meier_risk(time, event, horizon: float) -> float:
    """1 - Kaplan-Meier survival at `horizon`; `event` is truthy for an event, anything else
    is treated as censored (so a competing death passed as 0 is CENSORED -- the biased
    approach for MALO / CVD; correct for mortality)."""
    return aalen_johansen_cif(time, np.asarray(event).astype(bool).astype(int), horizon, cause=1)


def observed_risk(time, event_type, horizon: float, outcome: str) -> float:
    """Observed risk at `horizon`: Kaplan-Meier for mortality, Aalen-Johansen (death competing)
    for MALO / CVD."""
    if outcome == DEATH_OUTCOME:
        return kaplan_meier_risk(time, np.asarray(event_type) == 1, horizon)
    return aalen_johansen_cif(time, event_type, horizon, cause=1)


def estimator_name(outcome: str) -> str:
    return "Kaplan-Meier" if outcome == DEATH_OUTCOME else "Aalen-Johansen (death competing)"


def decile_groups(predicted, n_groups: int = 10) -> list[np.ndarray]:
    """Index arrays of n_groups (near-)equal-size groups, lowest predicted risk first."""
    order = np.argsort(np.asarray(predicted, dtype=float), kind="stable")
    return [g for g in np.array_split(order, n_groups) if g.size]


def calibration_table(predicted, time, event_type, horizon: float, outcome: str, n_groups: int = 10) -> dict:
    """Predicted vs observed risk at `horizon` per decile of predicted risk, plus overall O/E.

    predicted: (n,) predicted risk (CIF) at the horizon; time (years), event_type (1 / 2 / 0):
    at-risk rows only. Returns aggregates:
      {"outcome", "estimator", "horizon_years",
       "groups": [{"group", "n", "n_events", "mean_predicted", "observed"}, ...],
       "overall": {"n", "n_events", "expected", "observed", "oe_ratio"}}
    expected = mean predicted risk; observed = KM / Aalen-Johansen risk of the whole sample;
    oe_ratio = observed / expected (> 1: the model under-predicts).
    """
    predicted = np.asarray(predicted, dtype=float)
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type)
    within = time <= horizon
    groups = []
    for i, idx in enumerate(decile_groups(predicted, n_groups), 1):
        groups.append({
            "group": i,
            "n": int(idx.size),
            "n_events": int(((event_type[idx] == 1) & within[idx]).sum()),
            "mean_predicted": float(predicted[idx].mean()),
            "observed": observed_risk(time[idx], event_type[idx], horizon, outcome),
        })
    expected = float(predicted.mean()) if predicted.size else float("nan")
    observed = observed_risk(time, event_type, horizon, outcome)
    return {
        "outcome": outcome,
        "estimator": estimator_name(outcome),
        "horizon_years": float(horizon),
        "groups": groups,
        "overall": {
            "n": int(predicted.size),
            "n_events": int(((event_type == 1) & within).sum()),
            "expected": expected,
            "observed": observed,
            "oe_ratio": float(observed / expected) if expected > 0 else float("nan"),
        },
    }


def at_risk_horizon_data(cif, ground_truth: dict, outcome: str, horizon: int) -> tuple:
    """(predicted risk at the horizon, time_years, event_type) for one outcome's at-risk rows.
    cif: (n, T) predicted CIF at yearly bins 1..T (T >= horizon)."""
    g = ground_truth[outcome]
    keep = np.asarray(g["at_risk"]).astype(bool)
    predicted = np.asarray(cif)[keep][:, int(horizon) - 1]
    return predicted, np.asarray(g["time_years"], dtype=float)[keep], np.asarray(g["event_type"])[keep]


def calibration_all(predictions: dict, ground_truth: dict, horizon: int, n_groups: int = 10) -> dict:
    """{model: {outcome: calibration_table}} for every model's predictions
    ({model: {outcome: (n, T) CIF}}), rows aligned with ground_truth."""
    out = {}
    for model, preds in predictions.items():
        out[model] = {}
        for outcome in OUTCOME_NAMES:
            predicted, time, event_type = at_risk_horizon_data(preds[outcome], ground_truth, outcome, horizon)
            out[model][outcome] = calibration_table(predicted, time, event_type, horizon, outcome, n_groups)
    return out


def format_calibration_tables(calibration: dict) -> str:
    """Markdown: overall O/E per model and outcome, then the per-decile table."""
    models = list(calibration)
    first = calibration[models[0]]
    lines = [
        f"Calibration at {first[OUTCOME_NAMES[0]]['horizon_years']:g} yr | observed: Kaplan-Meier (mortality), "
        "Aalen-Johansen with death competing (MALO, CVD) | expected = mean predicted risk",
        "",
        "| Model | Outcome | n | Events | Expected | Observed | O/E |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for o in OUTCOME_NAMES:
            ov = calibration[m][o]["overall"]
            lines.append(f"| {m} | {o} | {ov['n']} | {ov['n_events']} | {ov['expected']:.4f} | {ov['observed']:.4f} | "
                         f"{ov['oe_ratio']:.2f} |")
    for o in OUTCOME_NAMES:
        lines += ["", f"Deciles of predicted risk: {o} ({estimator_name(o)}), predicted / observed",
                  "", "| Decile | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        n_rows = max(len(calibration[m][o]["groups"]) for m in models)
        for i in range(n_rows):
            cells = []
            for m in models:
                gs = calibration[m][o]["groups"]
                cells.append(f"{gs[i]['mean_predicted']:.3f} / {gs[i]['observed']:.3f}" if i < len(gs) else "")
            n = calibration[models[0]][o]["groups"][i]["n"] if i < len(calibration[models[0]][o]["groups"]) else ""
            lines.append(f"| {i + 1} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

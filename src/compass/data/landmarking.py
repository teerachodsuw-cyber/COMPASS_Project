"""Landmark sampling and per-patient training-example construction.

The single most important correctness property in this file: construct_landmark_example()
must never let information dated after landmark_date leak into input_visits, and outcome
windows must start strictly after landmark_date (not on it).
"""

import numpy as np
import pandas as pd

OUTCOME_NAMES = ("mortality", "malo", "cvd")


def death_date_of(patient_outcomes: pd.Series) -> pd.Timestamp:
    """The patient's death date if mortality_event == 1, else NaT."""
    if patient_outcomes["mortality_event"] == 1 and pd.notna(patient_outcomes["mortality_date"]):
        return pd.Timestamp(patient_outcomes["mortality_date"])
    return pd.NaT


def sample_landmark(
    patient_visits: pd.DataFrame, rng: np.random.Generator, *, death_date: pd.Timestamp
) -> pd.Timestamp:
    """Sample one random landmark time from a single patient's own visit dates.

    Candidate landmarks on or after death_date are dropped (pass NaT/None for a patient
    who did not die). If only one candidate remains, it IS the landmark. Raises if no visit
    precedes death -- the loader/validator guarantee death is strictly after index.
    """
    visit_dates = patient_visits["visit_date"].to_numpy()
    if death_date is not None and pd.notna(death_date):
        visit_dates = visit_dates[visit_dates < np.datetime64(pd.Timestamp(death_date))]
    if len(visit_dates) == 0:
        raise ValueError("sample_landmark: no visit before death_date -- no valid landmark for this patient")
    idx = rng.integers(0, len(visit_dates))
    return pd.Timestamp(visit_dates[idx])


def construct_landmark_example(
    patient_visits: pd.DataFrame,
    patient_outcomes: pd.Series,
    landmark_date: pd.Timestamp,
    horizons: list[int],
) -> dict:
    """Construct one training example anchored at landmark_date.

    - input_visits: all of patient_visits with visit_date <= landmark_date, chronologically
      ordered. Nothing dated after landmark_date is included.
    - outcome_labels: {outcome_name: {horizon: 0/1}} -- 1 iff that outcome's own event_date
      falls strictly after landmark_date and on/before landmark_date + horizon years. mortality
      /malo/cvd are evaluated independently from their own event_date/event_flag (not
      mutually exclusive, not composited) -- the semi-competing-risks ordering (mortality
      can't precede a non-fatal outcome) is already enforced upstream in how event dates are
      generated/validated, so no additional adjustment is needed here.
    """
    landmark_date = pd.Timestamp(landmark_date)

    input_visits = (
        patient_visits.loc[patient_visits["visit_date"] <= landmark_date]
        .sort_values("visit_date")
        .reset_index(drop=True)
    )

    outcome_labels = {}
    for name in OUTCOME_NAMES:
        event_flag = patient_outcomes[f"{name}_event"]
        event_date = patient_outcomes[f"{name}_date"]
        event_date = pd.Timestamp(event_date) if pd.notna(event_date) else pd.NaT

        horizon_labels = {}
        for h in horizons:
            window_end = landmark_date + pd.DateOffset(years=h)
            occurred_in_window = (
                bool(event_flag) and pd.notna(event_date) and landmark_date < event_date <= window_end
            )
            horizon_labels[h] = int(occurred_in_window)
        outcome_labels[name] = horizon_labels

    patient_id = patient_visits["patient_id"].iloc[0]

    return {
        "patient_id": patient_id,
        "landmark_date": landmark_date,
        "input_visits": input_visits,
        "outcome_labels": outcome_labels,
    }

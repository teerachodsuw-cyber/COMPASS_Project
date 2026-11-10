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


def _as_datetime64(x):
    """Scalar (Timestamp / NaT / None) or array-like of dates -> numpy datetime64[ns]."""
    if x is None or (np.ndim(x) == 0 and pd.isna(x)):
        return np.datetime64("NaT", "ns")
    if np.ndim(x) == 0:
        return np.datetime64(pd.Timestamp(x), "ns")
    return pd.to_datetime(pd.Series(x)).to_numpy()


def landmark_candidate_mask(visit_dates, index_date, death_date) -> np.ndarray:
    """Which visits may serve as landmarks: visit_date >= index_date AND < death_date.
    Visits before index_date are history only. NaT/None death_date = did not die.
    index_date / death_date may be scalars (one patient) or arrays aligned with visit_dates.
    The single definition of a valid landmark -- used by sample_landmark() and by the
    real-data loader (compass.data.loader) to drop patients with no valid candidate."""
    dates = pd.to_datetime(pd.Series(visit_dates)).to_numpy()
    index_date, death_date = _as_datetime64(index_date), _as_datetime64(death_date)
    return (dates >= index_date) & (np.isnat(death_date) | (dates < death_date))


def sample_landmark(
    patient_visits: pd.DataFrame, rng: np.random.Generator, *, index_date: pd.Timestamp, death_date: pd.Timestamp
) -> pd.Timestamp:
    """Sample one random landmark time from a single patient's own visit dates.

    Candidates are visits on/after index_date and before death_date (see
    landmark_candidate_mask; pass NaT/None death_date for a patient who did not die). If
    only one candidate remains, it IS the landmark. Raises if there is none -- the loader
    drops such patients before training.
    """
    visit_dates = patient_visits["visit_date"].to_numpy()
    visit_dates = visit_dates[landmark_candidate_mask(visit_dates, index_date, death_date)]
    if len(visit_dates) == 0:
        raise ValueError("sample_landmark: no visit on/after index_date and before death_date -- no valid landmark")
    idx = rng.integers(0, len(visit_dates))
    return pd.Timestamp(visit_dates[idx])


def construct_landmark_example(
    patient_visits: pd.DataFrame,
    patient_outcomes: pd.Series,
    landmark_date: pd.Timestamp,
) -> dict:
    """Construct one example anchored at landmark_date.

    - input_visits: all of patient_visits with visit_date <= landmark_date, chronologically
      ordered. Nothing dated after landmark_date is included.

    No outcome labels are attached here. Outcome times, event / competing / censoring status
    and at-risk flags come ONLY from compass.training.targets.build_targets (used by training
    and evaluation alike). The censoring-blind per-horizon 0/1 labels this function used to
    produce were removed: they ignored censoring and competing death, so nothing may use them.
    patient_outcomes is accepted for interface stability but not read.
    """
    landmark_date = pd.Timestamp(landmark_date)

    input_visits = (
        patient_visits.loc[patient_visits["visit_date"] <= landmark_date]
        .sort_values("visit_date")
        .reset_index(drop=True)
    )
    patient_id = patient_visits["patient_id"].iloc[0]

    return {
        "patient_id": patient_id,
        "landmark_date": landmark_date,
        "input_visits": input_visits,
    }

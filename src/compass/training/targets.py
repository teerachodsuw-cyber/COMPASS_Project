"""Per-outcome survival targets for landmark examples -- the ONLY place outcome times are
computed. compass.training.batching.collate_examples calls build_targets(); evaluation code
should call it too, so training and evaluation can never disagree on event / censoring times.

Inputs per example: landmark date L, the patient's outcomes row, and the patient's
last_visit_date (the last visit_date in the FULL visit table, computed once by
compute_last_visit_dates() before any landmark truncation). cutoff is the administrative
cutoff (configs/data_schema.yaml: censoring.administrative_cutoff).

Rules (semi-competing risks):

- Mortality: event if L < death_date <= cutoff (time = death_date); otherwise censored at
  the cutoff. Mortality follow-up runs to the cutoff, past the last visit (registry-linked).
- MALO and CVD, each on its own:
    * event if L < own event_date <= cutoff -> time = event_date. Counted even when the event
      is dated after last_visit_date.
    * otherwise C = min(last_visit_date, cutoff):
        - competing (death) if L < death_date <= C -> time = death_date, not an event
        - otherwise censored at C. A death after C is NOT competing (the patient was already
          out of observation for this outcome when they died).
- at_risk = 0 (the outcome is skipped for this example; the losses mask it out) when:
    * MALO: malo_date <= L (a prior event)
    * CVD: cvd_at_risk == 0 (prevalent CVD at index), or cvd_date <= L
    * no event and the censoring time is <= L (zero follow-up): C <= L for MALO / CVD,
      cutoff <= L for mortality
    * mortality: death_date <= L (defensive -- landmarks on/after death are never sampled,
      see compass.data.landmarking.sample_landmark)
- Window cap: bins cover (0, num_time_bins * bin_width_years] years from L. An event or
  competing death after the end of the last bin is censored there (survived all bins), and
  any later censoring time is capped there. Never clipped into the last bin as an event.

Every target is still filled for at_risk == 0 examples (censored, 1 day of follow-up, bin 0)
so tensors stay finite; the losses multiply by at_risk and average over at-risk examples.
"""

import numpy as np
import pandas as pd
import torch

from compass.models.base import OUTCOME_NAMES, years_to_bin_index

DAYS_PER_YEAR = 365.25
DEATH_OUTCOME = "mortality"
CVD_OUTCOME = "cvd"


def compute_last_visit_dates(visits_df: pd.DataFrame) -> pd.Series:
    """patient_id -> last visit_date in the FULL visit table. Call once, on the untruncated
    visits, never on landmark-truncated input_visits."""
    return pd.to_datetime(visits_df.groupby("patient_id")["visit_date"].max())


def _days_from(landmarks: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Days from each landmark to each date (float, NaN where the date is missing)."""
    return (pd.to_datetime(dates).to_numpy() - landmarks) / np.timedelta64(1, "D")


def build_targets(
    patient_ids: list,
    landmark_dates: list,
    outcomes_df: pd.DataFrame,
    last_visit_dates: pd.Series,
    administrative_cutoff: pd.Timestamp,
    bin_width_years: float,
    num_time_bins: int,
) -> dict[str, torch.Tensor]:
    """Per outcome, (batch,) tensors keyed `{name}_<target>`:

    - event_indicator (float 0/1): the outcome's own event, within the window
    - competing_indicator (float 0/1): MALO / CVD only (always 0 for mortality) -- death
      ended follow-up first; the hazard losses treat it as censored at death_date
    - target_days (float): days from L to the event / competing death / censoring time
    - event_bin, censoring_bin (long): years_to_bin_index of target_days (same value; the
      losses read event_bin when event_indicator == 1, censoring_bin otherwise)
    - at_risk (float 0/1): 0 -> this outcome is skipped for this example
    """
    landmarks = pd.to_datetime(pd.Series(landmark_dates)).to_numpy()
    rows = outcomes_df.set_index("patient_id").loc[list(patient_ids)]

    last_visit = pd.to_datetime(last_visit_dates).reindex(list(patient_ids))
    if last_visit.isna().any():
        raise ValueError(f"{int(last_visit.isna().sum())} example(s) have no last_visit_date (patient not in the visit table)")
    cutoff = pd.Timestamp(administrative_cutoff)

    cutoff_days = _days_from(landmarks, pd.Series([cutoff] * len(landmarks)))
    nonfatal_censor_days = _days_from(landmarks, last_visit.clip(upper=cutoff).reset_index(drop=True))  # C = min(last visit, cutoff)
    window_days = num_time_bins * bin_width_years * DAYS_PER_YEAR

    death_flag = rows[f"{DEATH_OUTCOME}_event"].to_numpy() == 1
    death_days = np.where(death_flag, _days_from(landmarks, rows[f"{DEATH_OUTCOME}_date"].reset_index(drop=True)), np.nan)

    targets: dict[str, torch.Tensor] = {}
    for name in OUTCOME_NAMES:
        if name == DEATH_OUTCOME:
            own_days = death_days
            event = (own_days > 0) & (own_days <= cutoff_days)
            competing = np.zeros_like(event)
            censor_days = cutoff_days
            prior = own_days <= 0
        else:
            own_flag = rows[f"{name}_event"].to_numpy() == 1
            own_days = np.where(own_flag, _days_from(landmarks, rows[f"{name}_date"].reset_index(drop=True)), np.nan)
            event = (own_days > 0) & (own_days <= cutoff_days)
            censor_days = nonfatal_censor_days
            competing = ~event & (death_days > 0) & (death_days <= censor_days)
            prior = own_days <= 0
            if name == CVD_OUTCOME:
                prior = prior | (rows["cvd_at_risk"].to_numpy() == 0)

        time_days = np.where(event, own_days, np.where(competing, death_days, censor_days))
        zero_followup = ~event & (censor_days <= 0)
        at_risk = ~prior & ~zero_followup

        # Window cap: anything after the end of the last bin is censored there.
        beyond_window = time_days > window_days
        event = event & ~beyond_window
        competing = competing & ~beyond_window
        time_days = np.minimum(time_days, window_days)

        # Not at risk: neutral, finite placeholder targets (masked out of every loss).
        event = event & at_risk
        competing = competing & at_risk
        time_days = np.where(at_risk, time_days, 1.0)

        time_years = torch.tensor(time_days / DAYS_PER_YEAR, dtype=torch.float32)
        time_bin = years_to_bin_index(time_years, bin_width_years, num_time_bins)
        targets[f"{name}_event_indicator"] = torch.tensor(event.astype(np.float32))
        targets[f"{name}_competing_indicator"] = torch.tensor(competing.astype(np.float32))
        targets[f"{name}_target_days"] = torch.tensor(time_days, dtype=torch.float32)
        targets[f"{name}_event_bin"] = time_bin
        targets[f"{name}_censoring_bin"] = time_bin.clone()
        targets[f"{name}_at_risk"] = torch.tensor(at_risk.astype(np.float32))
    return targets

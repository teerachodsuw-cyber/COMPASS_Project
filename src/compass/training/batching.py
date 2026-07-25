"""Batch collation for variable-length patient sequences, plus derivation of per-outcome
discrete-time hazard targets (event_indicator, event_bin, censoring_bin) from raw
landmark/outcome data -- applying this project's locked semi-competing-risks censoring
rule (configs/data_schema.yaml: censoring.rule): mortality censors both malo and cvd if it
occurs first; malo and cvd do not censor each other.
"""

import numpy as np
import pandas as pd
import torch

from compass.models.base import OUTCOME_NAMES, years_to_bin_index


def collate_examples(
    examples: list[dict],
    outcomes_df: pd.DataFrame,
    administrative_cutoff: pd.Timestamp,
    bin_width_years: float,
    num_time_bins: int,
) -> dict[str, torch.Tensor]:
    """Pad a list of PreprocessedData landmark examples into batch tensors, and compute
    each outcome's (event_indicator, event_bin, censoring_bin) hazard-loss targets.

    Per-outcome censoring rule:
    - mortality: censored at the administrative cutoff only.
    - malo / cvd: censored at min(administrative cutoff, mortality event time) if mortality
      occurred -- death stops further risk-set exposure for the non-fatal outcomes,
      regardless of the OTHER non-fatal outcome's own status (they don't censor each other).
    """
    batch_size = len(examples)
    seq_lengths = torch.tensor([len(ex["input_values"]) for ex in examples], dtype=torch.long)
    max_len = int(seq_lengths.max().item())
    num_features = examples[0]["input_values"].shape[1]

    values = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    for i, ex in enumerate(examples):
        n = len(ex["input_values"])
        values[i, :n] = torch.tensor(ex["input_values"].to_numpy(), dtype=torch.float32)
        mask[i, :n] = torch.tensor(ex["input_mask"].to_numpy(), dtype=torch.float32)

    outcomes_indexed = outcomes_df.set_index("patient_id")
    landmark_dates = [ex["landmark_date"] for ex in examples]

    admin_years = np.array([(administrative_cutoff - ld).days / 365.25 for ld in landmark_dates], dtype=np.float32)
    admin_years = np.clip(admin_years, a_min=1e-3, a_max=None)  # guard against landmark == cutoff

    outcome_rows = outcomes_indexed.loc[[ex["patient_id"] for ex in examples]]
    mortality_flag = outcome_rows["mortality_event"].to_numpy().astype(int)
    mortality_date = pd.to_datetime(outcome_rows["mortality_date"]).to_numpy()
    mortality_years = np.array(
        [
            (pd.Timestamp(d) - ld).days / 365.25 if flag else np.inf
            for d, flag, ld in zip(mortality_date, mortality_flag, landmark_dates)
        ],
        dtype=np.float32,
    )

    batch: dict[str, torch.Tensor] = {"values": values, "mask": mask, "seq_lengths": seq_lengths}

    for name in OUTCOME_NAMES:
        event_flag = outcome_rows[f"{name}_event"].to_numpy().astype(int)
        event_date = pd.to_datetime(outcome_rows[f"{name}_date"]).to_numpy()
        event_years = np.array(
            [
                (pd.Timestamp(d) - ld).days / 365.25 if flag else np.nan
                for d, flag, ld in zip(event_date, event_flag, landmark_dates)
            ],
            dtype=np.float32,
        )
        # An event flagged but dated at/before landmark or beyond the admin cutoff isn't a
        # valid in-follow-up event for THIS landmark example -- treat as censored rather
        # than feeding a negative/out-of-range time into the loss (defensive: this data
        # should already respect landmark<event per landmarking.py/validate_schema.py, but
        # this is new derived computation those checks don't directly cover).
        event_in_followup = (event_flag == 1) & (event_years > 0) & (event_years <= admin_years)

        if name == "mortality":
            censor_years = admin_years
        else:
            mortality_censor = np.where(mortality_flag == 1, mortality_years, np.inf)
            censor_years = np.minimum(admin_years, mortality_censor)
        censor_years = np.clip(censor_years, a_min=1e-3, a_max=None)

        event_time_for_bin = np.where(event_in_followup, event_years, censor_years)

        batch[f"{name}_event_indicator"] = torch.tensor(event_in_followup.astype(np.float32))
        batch[f"{name}_event_bin"] = years_to_bin_index(torch.tensor(event_time_for_bin), bin_width_years, num_time_bins)
        batch[f"{name}_censoring_bin"] = years_to_bin_index(torch.tensor(censor_years), bin_width_years, num_time_bins)

    return batch

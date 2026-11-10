"""Batch collation for variable-length patient sequences. Outcome targets come from
compass.training.targets.build_targets -- the only place outcome times are computed (event /
competing / censoring rules, at-risk flags, 13-bin window cap: see that module).

Also computes `elapsed_days` (batch, seq_len): days since the previous visit (0 for each
patient's first visit), from each example's raw `input_visits["visit_date"]`. This is
generic, model-agnostic timing info -- computed here once so any architecture that needs
real elapsed time (currently TRACER, SurvLatentODE; Dynamic-DeepHit ignores it) doesn't have
to re-derive it from raw dates itself.
"""

import numpy as np
import pandas as pd
import torch

from compass.training.targets import build_targets


def collate_examples(
    examples: list[dict],
    outcomes_df: pd.DataFrame,
    administrative_cutoff: pd.Timestamp,
    bin_width_years: float,
    num_time_bins: int,
    *,
    last_visit_dates: pd.Series,
) -> dict[str, torch.Tensor]:
    """Pad a list of landmark examples into batch tensors and attach each outcome's targets
    from build_targets(): {name}_event_indicator, _competing_indicator, _event_bin,
    _censoring_bin, _target_days, _at_risk.

    last_visit_dates: patient_id -> last visit_date in the FULL visit table
    (compass.training.targets.compute_last_visit_dates), computed once by the caller before
    landmark truncation -- never derived from the truncated input_visits here.
    """
    batch_size = len(examples)
    seq_lengths = torch.tensor([len(ex["input_values"]) for ex in examples], dtype=torch.long)
    max_len = int(seq_lengths.max().item())
    num_features = examples[0]["input_values"].shape[1]

    values = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    elapsed_days = torch.zeros(batch_size, max_len, dtype=torch.float32)
    for i, ex in enumerate(examples):
        n = len(ex["input_values"])
        values[i, :n] = torch.tensor(ex["input_values"].to_numpy(), dtype=torch.float32)
        mask[i, :n] = torch.tensor(ex["input_mask"].to_numpy(), dtype=torch.float32)

        visit_dates = pd.to_datetime(ex["input_visits"]["visit_date"]).to_numpy()
        if n > 1:
            gaps_days = np.diff(visit_dates).astype("timedelta64[D]").astype(np.float32)
            elapsed_days[i, 1:n] = torch.tensor(gaps_days)

    batch: dict[str, torch.Tensor] = {
        "values": values,
        "mask": mask,
        "seq_lengths": seq_lengths,
        "elapsed_days": elapsed_days,
    }
    batch.update(
        build_targets(
            [ex["patient_id"] for ex in examples],
            [ex["landmark_date"] for ex in examples],
            outcomes_df,
            last_visit_dates,
            administrative_cutoff,
            bin_width_years,
            num_time_bins,
        )
    )
    return batch

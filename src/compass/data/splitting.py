"""Patient-level train/val/test splitting.

Uses iterative multilabel stratification (iterstrat) to balance each of the three
outcomes' MARGINAL event rate across splits independently. This replaces an earlier
combined-8-way-joint-category approach, which failed even at the real cohort's n=12,667
because the rarest joint combination (all three outcomes positive) is too small -- that
was never actually required: each cause-specific model head learns from patients who have
that specific outcome's status, regardless of the other two outcomes' status.
"""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

OUTCOME_EVENT_COLUMNS = ["mortality_event", "malo_event", "cvd_event"]

# Hard minimum: with fewer than 2 total positive cases for an outcome, it is
# mathematically impossible for that outcome to appear in more than one split. iterstrat
# itself does not raise in this situation -- it silently places the lone positive case in
# whichever split it lands in -- so this is checked explicitly.
_MIN_POSITIVE_CASES = 2


def _check_label_columns_splittable(outcomes_df: pd.DataFrame) -> None:
    for col in OUTCOME_EVENT_COLUMNS:
        n_positive = int(outcomes_df[col].sum())
        n_total = len(outcomes_df)
        if n_positive < _MIN_POSITIVE_CASES:
            raise ValueError(
                f"patient_level_split: outcome '{col}' has only {n_positive} positive case(s) "
                f"out of {n_total} patients -- too few to stratify across train/val/test. "
                "Not falling back to an unstratified split; surfacing this instead."
            )
        n_negative = n_total - n_positive
        if n_negative < _MIN_POSITIVE_CASES:
            raise ValueError(
                f"patient_level_split: outcome '{col}' has only {n_negative} negative case(s) "
                f"out of {n_total} patients -- too few to stratify across train/val/test. "
                "Not falling back to an unstratified split; surfacing this instead."
            )


def patient_level_split(
    outcomes_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Split patient_ids into (train_ids, val_ids, test_ids) using iterative multilabel
    stratification on the (mortality_event, malo_event, cvd_event) label matrix -- balances
    each outcome's marginal rate across splits independently, without requiring every joint
    combination to be populated.

    Two-stage split: first train vs (val+test), then (val+test) into val vs test, both
    stages stratified on the same 3-column label matrix via
    iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {train_frac + val_frac + test_frac}")

    _check_label_columns_splittable(outcomes_df)

    patient_ids = outcomes_df["patient_id"].to_numpy()
    labels = outcomes_df[OUTCOME_EVENT_COLUMNS].to_numpy()
    n = len(patient_ids)

    stage1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(stage1.split(np.zeros((n, 1)), labels))

    rest_labels = labels[rest_idx]
    relative_test_frac = test_frac / (val_frac + test_frac)
    stage2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=relative_test_frac, random_state=seed)
    val_idx_rel, test_idx_rel = next(stage2.split(np.zeros((len(rest_idx), 1)), rest_labels))

    train_ids = patient_ids[train_idx].tolist()
    val_ids = patient_ids[rest_idx][val_idx_rel].tolist()
    test_ids = patient_ids[rest_idx][test_idx_rel].tolist()

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    return train_ids, val_ids, test_ids

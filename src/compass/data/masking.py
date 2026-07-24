"""Missingness masking, kept deliberately separate from scaling (scaling.py) so each is
independently testable. Callers must scale first, then mask -- see scaling.py and
preprocessing.py for the required order.
"""

import pandas as pd


def apply_masking(df: pd.DataFrame, continuous_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (values_df, mask_df) for the given continuous_columns.

    mask_df: same shape as df[continuous_columns], 1 where the value was observed
    (non-null), 0 where missing.
    values_df: a copy of df with continuous_columns' missing entries zero-filled. All other
    columns are passed through unchanged.

    Call this on already-scaled values (see scaling.apply_scalers) -- zero-filling raw,
    unscaled values would corrupt them once scaling happens afterward.
    """
    mask_df = df[continuous_columns].notna().astype(int)

    values_df = df.copy()
    values_df[continuous_columns] = df[continuous_columns].fillna(0.0)

    return values_df, mask_df

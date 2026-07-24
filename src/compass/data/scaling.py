"""Scaler fit/apply with a strict order: fit on observed TRAINING values only, and never
zero-fill before transforming. Getting this order wrong silently distorts the mean/SD for
high-missingness features (e_median/cap_median are ~95-97% missing in this schema) -- this
has happened before, do not repeat it.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scalers(
    train_values_df: pd.DataFrame,
    train_mask_df: pd.DataFrame,
    continuous_columns: list[str],
) -> dict[str, StandardScaler]:
    """Fit one StandardScaler per continuous column, using ONLY entries where
    train_mask_df == 1 for that column (observed, training-split values). Never fit on
    zero-filled or missing entries.
    """
    scalers = {}
    for col in continuous_columns:
        observed = train_values_df.loc[train_mask_df[col] == 1, col].to_numpy().reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(observed)
        scalers[col] = scaler
    return scalers


def apply_scalers(
    values_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    scalers: dict[str, StandardScaler],
    continuous_columns: list[str],
) -> pd.DataFrame:
    """Transform values_df using already-fitted scalers (works on train/val/test alike --
    the scalers themselves only ever came from training data).

    Transforms every row of every continuous column, including entries that are missing.
    Missing entries are NaN and must stay NaN here -- zero-filling is masking.apply_masking's
    job, called separately afterward, never here. Because scikit-learn's transform() rejects
    NaN input outright, the transform is applied manually via the fitted mean_/scale_ so NaN
    propagates through naturally instead of raising.
    """
    transformed = values_df.copy()
    for col in continuous_columns:
        scaler = scalers[col]
        mean, scale = scaler.mean_[0], scaler.scale_[0]
        transformed[col] = (transformed[col] - mean) / scale

        # Defensive check: transforming must not change which entries are missing. If it
        # did, scaling and masking have been called out of order somewhere upstream.
        still_matches_mask = (transformed[col].isna() == (mask_df[col] == 0)).all()
        if not still_matches_mask:
            raise ValueError(
                f"apply_scalers: NaN pattern in '{col}' after transform does not match "
                "mask_df. This usually means values were zero-filled before scaling, or "
                "scaling/masking were called out of order."
            )

    return transformed

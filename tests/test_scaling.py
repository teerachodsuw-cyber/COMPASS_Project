"""Unit tests for compass.data.scaling normal-case behavior."""

import numpy as np
import pandas as pd
import pytest

from compass.data.scaling import apply_scalers, fit_scalers


def test_fit_scalers_matches_manual_mean_std():
    values_df = pd.DataFrame({"bmi": [10.0, 20.0, 30.0]})
    mask_df = pd.DataFrame({"bmi": [1, 1, 1]})
    scalers = fit_scalers(values_df, mask_df, ["bmi"])
    assert scalers["bmi"].mean_[0] == pytest.approx(20.0)
    assert scalers["bmi"].scale_[0] == pytest.approx(np.std([10.0, 20.0, 30.0]))


def test_apply_scalers_transforms_and_preserves_nan():
    values_df = pd.DataFrame({"bmi": [10.0, 20.0, 30.0, np.nan]})
    mask_df = pd.DataFrame({"bmi": [1, 1, 1, 0]})
    scalers = fit_scalers(values_df, mask_df, ["bmi"])

    transformed = apply_scalers(values_df, mask_df, scalers, ["bmi"])

    assert transformed["bmi"].isna().tolist() == [False, False, False, True]
    # Fitted on its own data, so the transformed observed values are standardized (mean 0).
    assert transformed.loc[mask_df["bmi"] == 1, "bmi"].mean() == pytest.approx(0.0, abs=1e-9)

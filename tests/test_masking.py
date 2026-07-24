"""Unit tests for compass.data.masking."""

import numpy as np
import pandas as pd

from compass.data.masking import apply_masking


def test_apply_masking_shapes_and_values():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1"],
            "bmi": [22.0, np.nan],
            "hba1c": [np.nan, 6.5],
        }
    )
    values_df, mask_df = apply_masking(df, ["bmi", "hba1c"])

    assert mask_df.shape == (2, 2)
    assert list(mask_df["bmi"]) == [1, 0]
    assert list(mask_df["hba1c"]) == [0, 1]

    assert values_df.loc[1, "bmi"] == 0.0
    assert values_df.loc[0, "hba1c"] == 0.0
    assert values_df.loc[0, "bmi"] == 22.0
    assert values_df.loc[1, "hba1c"] == 6.5

    # Non-continuous columns pass through untouched.
    assert list(values_df["patient_id"]) == ["P1", "P1"]


def test_apply_masking_no_missing_values():
    df = pd.DataFrame({"bmi": [20.0, 25.0, 30.0]})
    values_df, mask_df = apply_masking(df, ["bmi"])
    assert (mask_df["bmi"] == 1).all()
    assert list(values_df["bmi"]) == [20.0, 25.0, 30.0]

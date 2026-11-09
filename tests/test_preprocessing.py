"""Unit tests for compass.data.preprocessing's combined feature tensor construction."""

import numpy as np
import pandas as pd
import pytest

from compass.data.preprocessing import build_feature_names, run_preprocessing
from compass.data.synthetic_generator import generate_synthetic_cohort

CONTINUOUS_COLUMNS = ["bmi", "hba1c"]
MEDICATION_COLUMNS = ["metformin", "statins"]
SEX_CATEGORIES = ("male", "female")
SMOKING_STATUS_CATEGORIES = [0, 1, 2]


def test_build_feature_names_order_and_count():
    names = build_feature_names(CONTINUOUS_COLUMNS, MEDICATION_COLUMNS, SEX_CATEGORIES, SMOKING_STATUS_CATEGORIES)
    assert names == [
        "bmi",
        "hba1c",
        "metformin",
        "statins",
        "sex_female",
        "smoking_status_0",
        "smoking_status_1",
        "smoking_status_2",
    ]


def test_build_feature_names_rejects_non_binary_sex():
    with pytest.raises(ValueError):
        build_feature_names(CONTINUOUS_COLUMNS, MEDICATION_COLUMNS, ("male", "female", "other"), SMOKING_STATUS_CATEGORIES)


def test_run_preprocessing_combines_all_feature_groups():
    import yaml

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    continuous_columns = [v["name"] for v in schema["continuous_variables"]]
    medication_columns = [m["name"] for m in schema["medication_flags"]]

    visits_df, outcomes_df = generate_synthetic_cohort(n_patients=200, seed=42)
    result = run_preprocessing(
        visits_df,
        outcomes_df,
        continuous_columns,
        medication_columns,
        sex_categories=("male", "female"),
        smoking_status_categories=[0, 1, 2],
        horizons=[10, 20, 30],
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )

    expected_n_features = len(continuous_columns) + len(medication_columns) + 1 + 3  # +1 sex, +3 smoking one-hot (0/1/2)
    assert len(result.feature_names) == expected_n_features

    ex = result.train[0]
    assert ex["input_values"].shape[1] == expected_n_features
    assert ex["input_mask"].shape[1] == expected_n_features
    assert list(ex["input_values"].columns) == result.feature_names
    assert list(ex["input_mask"].columns) == result.feature_names

    # Medication flags are binary and, in our synthetic data, never missing -> mask always 1.
    for med in medication_columns:
        assert set(ex["input_values"][med].unique()) <= {0.0, 1.0}
        assert (ex["input_mask"][med] == 1).all()

    # smoking_status one-hot: exactly one indicator is 1 per row (never missing here).
    smoking_cols = [c for c in result.feature_names if c.startswith("smoking_status_")]
    concat_values = pd.concat([e["input_values"] for e in result.train], ignore_index=True)
    assert (concat_values[smoking_cols].sum(axis=1) == 1).all()


def test_medication_and_demographic_missingness_is_masked_not_assumed_absent():
    from compass.data.preprocessing import _encode_medication_and_demographic_features

    df = pd.DataFrame(
        {
            "metformin": [1.0, np.nan, 0.0],
            "statins": [0.0, 1.0, np.nan],
            "sex": ["male", np.nan, "female"],
            "smoking_status": [0, 2, np.nan],
        }
    )
    values, mask = _encode_medication_and_demographic_features(df, ["metformin", "statins"], ("male", "female"), SMOKING_STATUS_CATEGORIES)

    assert list(mask["metformin"]) == [1, 0, 1]
    assert values.loc[1, "metformin"] == 0.0  # zero-filled where missing

    assert list(mask["sex_female"]) == [1, 0, 1]
    assert values.loc[1, "sex_female"] == 0.0

    smoking_cols = [c for c in mask.columns if c.startswith("smoking_status_")]
    assert list(mask.loc[2, smoking_cols]) == [0, 0, 0]
    assert list(values.loc[2, smoking_cols]) == [0.0, 0.0, 0.0]

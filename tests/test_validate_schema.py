"""Tests for compass.data.validate_schema.

One deliberate-failure test per check in validate_dataset(), plus two for the
combined dtype check (numeric continuous vars / binary medication flags), since
that check verifies two distinct conditions.
"""

import pandas as pd
import pytest

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.data.validate_schema import SchemaValidationError, validate_dataset

CONFIG_PATH = "configs/data_schema.yaml"


@pytest.fixture
def valid_cohort():
    return generate_synthetic_cohort(n_patients=50, seed=42, config_path=CONFIG_PATH)


def test_valid_cohort_passes(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    validate_dataset(visits_df, outcomes_df, config_path=CONFIG_PATH)


def test_missing_expected_column_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.drop(columns=["bmi"])
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_unexpected_extra_column_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken["not_in_schema"] = 1
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_non_numeric_continuous_variable_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken["bmi"] = broken["bmi"].astype(str)
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_non_binary_medication_flag_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken.loc[broken.index[0], "metformin"] = 2
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_visit_date_out_of_range_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken.loc[broken.index[0], "visit_date"] = pd.Timestamp("2030-01-01")
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_duplicate_patient_visit_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = pd.concat([visits_df, visits_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaValidationError):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_outcome_flag_date_inconsistency_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    broken.loc[broken.index[0], "mortality_event"] = 1
    broken.loc[broken.index[0], "mortality_date"] = pd.NaT
    with pytest.raises(SchemaValidationError):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_event_date_before_first_visit_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    pid = outcomes_df.loc[outcomes_df["mortality_event"] == 0, "patient_id"].iloc[0]
    first_visit = visits_df.loc[visits_df["patient_id"] == pid, "visit_date"].min()

    broken = outcomes_df.copy()
    idx = broken.index[broken["patient_id"] == pid][0]
    broken.loc[idx, "mortality_event"] = 1
    broken.loc[idx, "mortality_date"] = first_visit - pd.Timedelta(days=1)
    with pytest.raises(SchemaValidationError):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_missing_outcomes_row_for_patient_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.iloc[1:]
    with pytest.raises(SchemaValidationError):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)

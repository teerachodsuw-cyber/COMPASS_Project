"""Tests for compass.data.validate_schema.

One deliberate-failure test per check in validate_dataset(), plus two for the
combined dtype check (numeric continuous vars / binary medication flags), since
that check verifies two distinct conditions. New rules (categorical value sets,
outcomes-table structure, date order) each get a failing case that checks the
error message, plus passing cases for the valid cohort and for NaN sex/smoking.
"""

import numpy as np
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


def test_missing_outcomes_row_for_patient_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.iloc[1:]
    with pytest.raises(SchemaValidationError):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


# ---- helpers for the new rules ---------------------------------------------------------


def _clean_patients(visits_df, outcomes_df, n=1):
    """Patients with no events, cvd_at_risk==1, >= 2 visits, and a last visit at least
    30 days before the cohort end -- room to place a valid or invalid date either side."""
    n_visits = visits_df.groupby("patient_id").size()
    last_visit = visits_df.groupby("patient_id")["visit_date"].max()
    ok = (
        (outcomes_df["mortality_event"] == 0)
        & (outcomes_df["malo_event"] == 0)
        & (outcomes_df["cvd_event"] == 0)
        & (outcomes_df["cvd_at_risk"] == 1)
        & (outcomes_df["patient_id"].map(n_visits) >= 2)
        & (outcomes_df["patient_id"].map(last_visit) <= pd.Timestamp("2025-05-31"))
    )
    rows = outcomes_df.index[ok][:n]
    assert len(rows) == n
    return [(r, last_visit[outcomes_df.loc[r, "patient_id"]]) for r in rows]


def _set_event(df, row, name, date):
    df.loc[row, f"{name}_event"] = 1
    df.loc[row, f"{name}_date"] = date


# ---- 1. categorical value sets -----------------------------------------------------------


def test_sex_outside_schema_values_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken.loc[broken.index[0], "sex"] = "unknown"
    with pytest.raises(SchemaValidationError, match=r"categorical column 'sex' .*\['unknown'\]"):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_smoking_status_outside_schema_values_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = visits_df.copy()
    broken["smoking_status"] = broken["smoking_status"].astype(float)
    broken.loc[broken.index[0], "smoking_status"] = 3
    with pytest.raises(SchemaValidationError, match=r"categorical column 'smoking_status' .*\[3\.0\]"):
        validate_dataset(broken, outcomes_df, config_path=CONFIG_PATH)


def test_nan_sex_and_smoking_status_are_allowed(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    with_missing = visits_df.copy()
    with_missing["smoking_status"] = with_missing["smoking_status"].astype(float)
    with_missing.loc[with_missing.index[:5], "sex"] = np.nan
    with_missing.loc[with_missing.index[:5], "smoking_status"] = np.nan
    validate_dataset(with_missing, outcomes_df, config_path=CONFIG_PATH)


# ---- 2. outcomes table structure --------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    ["patient_id", "index_date", "cvd_at_risk"]
    + [f"{o}_{k}" for o in ("mortality", "malo", "cvd") for k in ("event", "date")],
)
def test_missing_required_outcome_column_raises(valid_cohort, column):
    visits_df, outcomes_df = valid_cohort
    with pytest.raises(SchemaValidationError, match=f"missing required .*'{column}'"):
        validate_dataset(visits_df, outcomes_df.drop(columns=[column]), config_path=CONFIG_PATH)


def test_non_binary_cvd_at_risk_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    broken.loc[broken.index[0], "cvd_at_risk"] = 2
    with pytest.raises(SchemaValidationError, match=r"cvd_at_risk must be 0/1"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_null_index_date_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    broken.loc[broken.index[0], "index_date"] = pd.NaT
    with pytest.raises(SchemaValidationError, match=r"1 row\(s\) in outcomes_df have a null index_date"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_duplicate_outcomes_row_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = pd.concat([outcomes_df, outcomes_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaValidationError, match=r"more than one row in outcomes_df"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


# ---- 3. date-order rules ------------------------------------------------------------------


def test_death_before_last_visit_raises_with_patient_count(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    for row, last_visit in _clean_patients(visits_df, outcomes_df, n=2):
        _set_event(broken, row, "mortality", last_visit - pd.Timedelta(days=1))
    with pytest.raises(SchemaValidationError, match=r"^2 patient\(s\) have mortality_date before their last visit_date"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_death_on_last_visit_is_allowed(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    ok = outcomes_df.copy()
    [(row, last_visit)] = _clean_patients(visits_df, outcomes_df)
    _set_event(ok, row, "mortality", last_visit)
    validate_dataset(visits_df, ok, config_path=CONFIG_PATH)


@pytest.mark.parametrize("name", ["malo", "cvd"])
def test_nonfatal_event_after_death_raises(valid_cohort, name):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    [(row, last_visit)] = _clean_patients(visits_df, outcomes_df)
    _set_event(broken, row, "mortality", last_visit)
    _set_event(broken, row, name, last_visit + pd.Timedelta(days=1))
    with pytest.raises(SchemaValidationError, match=rf"^1 patient\(s\) have {name}_date after mortality_date"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


@pytest.mark.parametrize("name", ["mortality", "malo", "cvd"])
@pytest.mark.parametrize("days_from_index", [0, -1], ids=["on_index", "before_index"])
def test_event_not_after_index_date_raises(valid_cohort, name, days_from_index):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    [(row, _)] = _clean_patients(visits_df, outcomes_df)
    index_date = broken.loc[row, "index_date"]
    if index_date + pd.Timedelta(days=days_from_index) < pd.Timestamp("2013-06-01"):
        pytest.skip("index_date is the first day of the cohort window")
    _set_event(broken, row, name, index_date + pd.Timedelta(days=days_from_index))
    with pytest.raises(SchemaValidationError, match=rf"^1 patient\(s\) have {name}_date on or before index_date"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def test_cvd_event_when_not_at_risk_raises(valid_cohort):
    visits_df, outcomes_df = valid_cohort
    broken = outcomes_df.copy()
    [(row, last_visit)] = _clean_patients(visits_df, outcomes_df)
    broken.loc[row, "cvd_at_risk"] = 0
    _set_event(broken, row, "cvd", last_visit)
    with pytest.raises(SchemaValidationError, match=r"^1 patient\(s\) have cvd_at_risk==0 but cvd_event==1"):
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)


def _break_death_before_last_visit(visits_df, outcomes_df):
    broken = outcomes_df.copy()
    [(row, last_visit)] = _clean_patients(visits_df, outcomes_df)
    _set_event(broken, row, "mortality", last_visit - pd.Timedelta(days=1))
    return broken, [broken.loc[row, "patient_id"]], r"^1 patient\(s\) have mortality_date before"


def _break_missing_outcomes_row(visits_df, outcomes_df):
    dropped = outcomes_df["patient_id"].iloc[:2].tolist()
    return outcomes_df.iloc[2:], dropped, r"^2 patient\(s\) in visits_df have no row in outcomes_df$"


def _break_duplicate_outcomes_row(visits_df, outcomes_df):
    duplicated = outcomes_df["patient_id"].iloc[:2].tolist()
    broken = pd.concat([outcomes_df, outcomes_df.iloc[:2]], ignore_index=True)
    return broken, duplicated, r"^2 patient\(s\) have more than one row in outcomes_df$"


@pytest.mark.parametrize(
    "break_outcomes",
    [_break_death_before_last_visit, _break_missing_outcomes_row, _break_duplicate_outcomes_row],
    ids=["death_before_last_visit", "missing_outcomes_row", "duplicate_outcomes_row"],
)
def test_error_messages_do_not_list_patient_ids(valid_cohort, break_outcomes):
    visits_df, outcomes_df = valid_cohort
    broken, offending_ids, expected = break_outcomes(visits_df, outcomes_df)
    with pytest.raises(SchemaValidationError, match=expected) as exc:
        validate_dataset(visits_df, broken, config_path=CONFIG_PATH)
    message = str(exc.value)
    for pid in offending_ids:
        assert pid not in message
    assert "SYN" not in message  # no synthetic patient ID of any kind

"""Hard pre-flight validator for COMPASS patient-visit datasets.

Any dataset (synthetic now, real later) must pass validate_dataset() before it
touches any other part of the pipeline. Every check below raises
SchemaValidationError on failure -- never warns and continues.
"""

import pandas as pd
import yaml


class SchemaValidationError(Exception):
    """Raised when a dataset fails a COMPASS schema pre-flight check."""


def _load_schema(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _schema_columns(schema: dict) -> set[str]:
    cols = {v["name"] for v in schema["continuous_variables"]}
    cols |= {m["name"] for m in schema["medication_flags"]}
    cols |= {d["name"] for d in schema["demographics"]}
    return cols


def _check_structural_columns(visits_df: pd.DataFrame, outcomes_df: pd.DataFrame, outcome_names: list[str]) -> None:
    for col in ("patient_id", "visit_date"):
        if col not in visits_df.columns:
            raise SchemaValidationError(f"visits_df is missing required structural column: '{col}'")

    if "patient_id" not in outcomes_df.columns:
        raise SchemaValidationError("outcomes_df is missing required structural column: 'patient_id'")
    for name in outcome_names:
        for col in (f"{name}_event", f"{name}_date"):
            if col not in outcomes_df.columns:
                raise SchemaValidationError(f"outcomes_df is missing required outcome column: '{col}'")


def _check_expected_columns_present(visits_df: pd.DataFrame, schema_cols: set[str]) -> None:
    missing = schema_cols - set(visits_df.columns)
    if missing:
        raise SchemaValidationError(f"visits_df is missing expected columns from data_schema.yaml: {sorted(missing)}")


def _check_no_unexpected_columns(visits_df: pd.DataFrame, schema_cols: set[str]) -> None:
    allowed = schema_cols | {"patient_id", "visit_date"}
    extra = set(visits_df.columns) - allowed
    if extra:
        raise SchemaValidationError(
            f"visits_df has columns not present in data_schema.yaml: {sorted(extra)}. "
            "This likely means the schema file is out of date."
        )


def _check_dtypes(visits_df: pd.DataFrame, schema: dict) -> None:
    for var in schema["continuous_variables"]:
        name = var["name"]
        if not pd.api.types.is_numeric_dtype(visits_df[name]):
            raise SchemaValidationError(f"continuous variable '{name}' is not numeric dtype")

    allowed = {0, 1, True, False}
    for med in schema["medication_flags"]:
        name = med["name"]
        observed = set(visits_df[name].dropna().unique())
        if not observed <= allowed:
            raise SchemaValidationError(
                f"medication flag '{name}' has non-binary values: {sorted(observed - allowed, key=str)}"
            )


def _check_visit_dates(visits_df: pd.DataFrame, range_start: pd.Timestamp, range_end: pd.Timestamp) -> None:
    try:
        dates = pd.to_datetime(visits_df["visit_date"])
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(f"visit_date does not parse as a valid date: {exc}") from exc

    out_of_range = (dates < range_start) | (dates > range_end)
    if out_of_range.any():
        raise SchemaValidationError(
            f"{int(out_of_range.sum())} visit_date value(s) fall outside the cohort "
            f"date_range [{range_start.date()}, {range_end.date()}]"
        )


def _check_unique_patient_visit(visits_df: pd.DataFrame) -> None:
    dup = visits_df.duplicated(subset=["patient_id", "visit_date"])
    if dup.any():
        raise SchemaValidationError(f"{int(dup.sum())} duplicate (patient_id, visit_date) row(s) found in visits_df")


def _check_outcome_flag_date_consistency(outcomes_df: pd.DataFrame, outcome_names: list[str]) -> None:
    for name in outcome_names:
        event_col, date_col = f"{name}_event", f"{name}_date"
        dates = pd.to_datetime(outcomes_df[date_col])

        flagged_missing_date = (outcomes_df[event_col] == 1) & dates.isna()
        if flagged_missing_date.any():
            raise SchemaValidationError(
                f"{int(flagged_missing_date.sum())} row(s) have {event_col}==1 but null {date_col}"
            )

        censored_with_date = (outcomes_df[event_col] == 0) & dates.notna()
        if censored_with_date.any():
            raise SchemaValidationError(
                f"{int(censored_with_date.sum())} row(s) have {event_col}==0 (censored) but non-null {date_col}"
            )


def _check_event_dates_in_range_and_after_first_visit(
    visits_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    outcome_names: list[str],
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> None:
    # Related to, but NOT the full landmarking leakage check -- that comes later
    # in preprocessing, once landmark sampling exists. This only checks that an
    # event date is not before the patient's own earliest observed visit.
    first_visit = pd.to_datetime(visits_df.groupby("patient_id")["visit_date"].min())

    for name in outcome_names:
        event_col, date_col = f"{name}_event", f"{name}_date"
        events = outcomes_df[outcomes_df[event_col] == 1]
        if events.empty:
            continue
        dates = pd.to_datetime(events.set_index("patient_id")[date_col])

        out_of_range = (dates < range_start) | (dates > range_end)
        if out_of_range.any():
            raise SchemaValidationError(f"{int(out_of_range.sum())} {date_col} value(s) fall outside the cohort date_range")

        joined = dates.to_frame(date_col).join(first_visit.rename("first_visit_date"), how="left")
        before_first_visit = joined[date_col] < joined["first_visit_date"]
        if before_first_visit.any():
            bad_ids = joined.index[before_first_visit].tolist()
            raise SchemaValidationError(
                f"{date_col} precedes the patient's earliest visit_date for patient_id(s): {bad_ids}"
            )


def _check_one_outcomes_row_per_patient(visits_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> None:
    visit_patients = set(visits_df["patient_id"].unique())
    outcome_counts = outcomes_df["patient_id"].value_counts()

    missing = visit_patients - set(outcome_counts.index)
    if missing:
        raise SchemaValidationError(f"patient_id(s) in visits_df have no row in outcomes_df: {sorted(missing)}")

    duplicated = outcome_counts[outcome_counts > 1]
    if not duplicated.empty:
        raise SchemaValidationError(f"patient_id(s) have more than one row in outcomes_df: {duplicated.index.tolist()}")


def validate_dataset(
    visits_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    config_path: str = "configs/data_schema.yaml",
) -> None:
    """Raise SchemaValidationError if visits_df/outcomes_df violate data_schema.yaml."""
    schema = _load_schema(config_path)
    schema_cols = _schema_columns(schema)
    outcome_names = [o["name"] for o in schema["outcomes"]]
    range_start = pd.Timestamp(schema["cohort"]["date_range"][0])
    range_end = pd.Timestamp(schema["cohort"]["date_range"][1])

    _check_structural_columns(visits_df, outcomes_df, outcome_names)
    _check_expected_columns_present(visits_df, schema_cols)
    _check_no_unexpected_columns(visits_df, schema_cols)
    _check_dtypes(visits_df, schema)
    _check_visit_dates(visits_df, range_start, range_end)
    _check_unique_patient_visit(visits_df)
    _check_outcome_flag_date_consistency(outcomes_df, outcome_names)
    _check_event_dates_in_range_and_after_first_visit(visits_df, outcomes_df, outcome_names, range_start, range_end)
    _check_one_outcomes_row_per_patient(visits_df, outcomes_df)

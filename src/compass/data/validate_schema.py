"""Hard pre-flight validator for COMPASS patient-visit datasets.

Any dataset (synthetic now, real later) must pass validate_dataset() before it
touches any other part of the pipeline. It runs on the LOADER OUTPUT (visits_df +
outcomes_df), not on the raw source file. Every check below raises
SchemaValidationError on failure -- never warns and continues.

Prior (pre-index) events are the loader's job, not this module's: the loader excludes
patients with a prior liver outcome and sets cvd_at_risk=0 for prevalent CVD. The
validator only checks that what the loader produced is internally consistent (every
event strictly after index_date, cvd_at_risk==0 -> no CVD event).

Error messages report counts and column names / category codes only, never
patient-level values.
"""

import pandas as pd
import yaml


class SchemaValidationError(Exception):
    """Raised when a dataset fails a COMPASS schema pre-flight check."""


# Outcome names (configs/data_schema.yaml: outcomes) that the date-order rules refer to.
DEATH_OUTCOME = "mortality"
CVD_OUTCOME = "cvd"


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

    for col in ("patient_id", "index_date", "cvd_at_risk"):
        if col not in outcomes_df.columns:
            raise SchemaValidationError(f"outcomes_df is missing required structural column: '{col}'")
    for name in outcome_names:
        for col in (f"{name}_event", f"{name}_date"):
            if col not in outcomes_df.columns:
                raise SchemaValidationError(f"outcomes_df is missing required outcome column: '{col}'")


def _check_outcome_structure(outcomes_df: pd.DataFrame) -> None:
    try:
        index_dates = pd.to_datetime(outcomes_df["index_date"])
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(f"index_date does not parse as a valid date: {exc}") from exc
    n_missing_index = int(index_dates.isna().sum())
    if n_missing_index:
        raise SchemaValidationError(f"{n_missing_index} row(s) in outcomes_df have a null index_date")

    at_risk = outcomes_df["cvd_at_risk"]
    n_null = int(at_risk.isna().sum())
    if n_null:
        raise SchemaValidationError(f"{n_null} row(s) in outcomes_df have a null cvd_at_risk (must be 0/1)")
    bad = ~at_risk.isin([0, 1])
    if bad.any():
        codes = sorted(at_risk[bad].unique().tolist(), key=str)
        raise SchemaValidationError(f"cvd_at_risk must be 0/1; found {int(bad.sum())} row(s) with value(s) {codes}")


def _check_categorical_values(visits_df: pd.DataFrame, schema: dict) -> None:
    """Every categorical demographic must take a value from its schema `values` list.
    NaN is allowed (missing); any other value raises."""
    for var in schema["demographics"]:
        if var.get("type") != "categorical":
            continue
        name, allowed = var["name"], var["values"]
        col = visits_df[name]
        bad = col.notna() & ~col.isin(allowed)
        if bad.any():
            codes = sorted(col[bad].unique().tolist(), key=str)
            raise SchemaValidationError(
                f"categorical column '{name}' has {int(bad.sum())} row(s) with code(s) {codes} "
                f"not in data_schema.yaml values {allowed} (NaN is allowed)"
            )


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


def _check_event_dates_not_after_cutoff(
    outcomes_df: pd.DataFrame,
    outcome_names: list[str],
    cutoff: pd.Timestamp,
) -> None:
    """Event dates must be on/before the administrative cutoff. There is NO lower calendar
    bound for events: the lower bound is index_date (rule 3c, strictly after index), and
    index_date can precede the visit window (cohort.date_range), so an incident event can
    fall between a pre-window index_date and the window start. The visit-date window is
    checked separately (_check_visit_dates)."""
    for name in outcome_names:
        event_col, date_col = f"{name}_event", f"{name}_date"
        dates = pd.to_datetime(outcomes_df.loc[outcomes_df[event_col] == 1, date_col])
        after_cutoff = dates > cutoff
        if after_cutoff.any():
            raise SchemaValidationError(
                f"{int(after_cutoff.sum())} {date_col} value(s) fall after the administrative cutoff {cutoff.date()}"
            )


def _check_date_order(visits_df: pd.DataFrame, outcomes_df: pd.DataFrame, outcome_names: list[str]) -> None:
    """Date-order rules on the loader output (one outcomes row per patient, already checked).
    Each raises with the number of offending patients.

    a. death date >= the patient's last visit_date
    b. each non-fatal event date <= death date, when both exist
    c. every event date > index_date (strictly after index; prior events are the loader's job)
    d. cvd_at_risk == 0 -> cvd_event == 0
    """
    index_date = pd.to_datetime(outcomes_df["index_date"])
    death_col = f"{DEATH_OUTCOME}_date"
    death_date = pd.to_datetime(outcomes_df[death_col])
    last_visit = outcomes_df["patient_id"].map(pd.to_datetime(visits_df.groupby("patient_id")["visit_date"].max()))

    # c. every event strictly after index (checked first: a death before index would
    #    otherwise surface as the less specific rule a)
    for name in outcome_names:
        event_col, date_col = f"{name}_event", f"{name}_date"
        event_date = pd.to_datetime(outcomes_df[date_col])
        not_after_index = (outcomes_df[event_col] == 1) & event_date.notna() & (event_date <= index_date)
        if not_after_index.any():
            raise SchemaValidationError(
                f"{int(not_after_index.sum())} patient(s) have {date_col} on or before index_date "
                "(events must be strictly after index; prior events are handled by the loader)"
            )

    # a. death on or after the last visit
    died_before_last_visit = death_date.notna() & last_visit.notna() & (death_date < last_visit)
    if died_before_last_visit.any():
        raise SchemaValidationError(
            f"{int(died_before_last_visit.sum())} patient(s) have {death_col} before their last visit_date"
        )

    # b. non-fatal events on or before death
    for name in outcome_names:
        if name == DEATH_OUTCOME:
            continue
        date_col = f"{name}_date"
        event_date = pd.to_datetime(outcomes_df[date_col])
        after_death = event_date.notna() & death_date.notna() & (event_date > death_date)
        if after_death.any():
            raise SchemaValidationError(f"{int(after_death.sum())} patient(s) have {date_col} after {death_col}")

    # d. not at risk for CVD -> no CVD event
    cvd_event_col = f"{CVD_OUTCOME}_event"
    event_not_at_risk = (outcomes_df["cvd_at_risk"] == 0) & (outcomes_df[cvd_event_col] == 1)
    if event_not_at_risk.any():
        raise SchemaValidationError(
            f"{int(event_not_at_risk.sum())} patient(s) have cvd_at_risk==0 but {cvd_event_col}==1"
        )


def _check_one_outcomes_row_per_patient(visits_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> None:
    visit_patients = set(visits_df["patient_id"].unique())
    outcome_counts = outcomes_df["patient_id"].value_counts()

    # Counts only -- never list patient IDs (patient-level identifiers on real data).
    missing = visit_patients - set(outcome_counts.index)
    if missing:
        raise SchemaValidationError(f"{len(missing)} patient(s) in visits_df have no row in outcomes_df")

    duplicated = outcome_counts[outcome_counts > 1]
    if not duplicated.empty:
        raise SchemaValidationError(f"{len(duplicated)} patient(s) have more than one row in outcomes_df")


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
    for required in (DEATH_OUTCOME, CVD_OUTCOME):
        if required not in outcome_names:
            raise SchemaValidationError(f"data_schema.yaml outcomes must include '{required}'")

    _check_structural_columns(visits_df, outcomes_df, outcome_names)
    _check_expected_columns_present(visits_df, schema_cols)
    _check_no_unexpected_columns(visits_df, schema_cols)
    _check_dtypes(visits_df, schema)
    _check_categorical_values(visits_df, schema)
    _check_visit_dates(visits_df, range_start, range_end)
    _check_unique_patient_visit(visits_df)
    _check_outcome_structure(outcomes_df)
    _check_outcome_flag_date_consistency(outcomes_df, outcome_names)
    _check_event_dates_not_after_cutoff(outcomes_df, outcome_names, pd.Timestamp(schema["censoring"]["administrative_cutoff"]))
    _check_one_outcomes_row_per_patient(visits_df, outcomes_df)  # before date-order rules, which assume one row per patient
    _check_date_order(visits_df, outcomes_df, outcome_names)

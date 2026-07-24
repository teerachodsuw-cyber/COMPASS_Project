"""Tests for compass.data.synthetic_generator."""

import pytest
import yaml

from compass.data.synthetic_generator import CONTINUOUS_VARIABLE_BOUNDS, generate_synthetic_cohort

CONFIG_PATH = "configs/data_schema.yaml"


def _schema():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def cohort():
    return generate_synthetic_cohort(n_patients=300, seed=42, config_path=CONFIG_PATH)


def test_expected_columns_present(cohort):
    visits_df, outcomes_df = cohort
    schema = _schema()

    expected_visit_cols = {"patient_id", "visit_date"}
    expected_visit_cols |= {v["name"] for v in schema["continuous_variables"]}
    expected_visit_cols |= {m["name"] for m in schema["medication_flags"]}
    expected_visit_cols |= {d["name"] for d in schema["demographics"]}
    assert expected_visit_cols <= set(visits_df.columns)

    expected_outcome_cols = {"patient_id"}
    for o in schema["outcomes"]:
        expected_outcome_cols |= {f"{o['name']}_event", f"{o['name']}_date"}
    assert expected_outcome_cols <= set(outcomes_df.columns)


def test_continuous_variables_within_bounds(cohort):
    visits_df, _ = cohort
    for name, (low, high) in CONTINUOUS_VARIABLE_BOUNDS.items():
        values = visits_df[name].dropna()
        assert (values >= low).all() and (values <= high).all()


def test_missingness_roughly_matches_config(cohort):
    visits_df, _ = cohort
    schema = _schema()
    for var in schema["continuous_variables"]:
        name, expected_rate = var["name"], var.get("missing_rate", 0.0)
        observed_rate = visits_df[name].isna().mean()
        # Generous tolerance -- this checks the missingness mechanism is wired up
        # correctly, not that one random draw matches the config exactly.
        assert abs(observed_rate - expected_rate) < 0.08


def test_malo_incidence_is_rare(cohort):
    _, outcomes_df = cohort
    malo_rate = outcomes_df["malo_event"].mean()
    # Rare-event check: also confirms malo was not accidentally balanced to ~50%.
    assert malo_rate < 0.15


def test_visits_vary_within_patient(cohort):
    visits_df, _ = cohort
    multi_visit = visits_df.groupby("patient_id").filter(lambda g: len(g) > 1)
    assert not multi_visit.empty
    n_unique_bmi_per_patient = multi_visit.groupby("patient_id")["bmi"].nunique()
    assert (n_unique_bmi_per_patient > 1).any()


def test_reproducible_with_seed():
    visits_a, outcomes_a = generate_synthetic_cohort(n_patients=50, seed=7, config_path=CONFIG_PATH)
    visits_b, outcomes_b = generate_synthetic_cohort(n_patients=50, seed=7, config_path=CONFIG_PATH)
    assert visits_a.equals(visits_b)
    assert outcomes_a.equals(outcomes_b)

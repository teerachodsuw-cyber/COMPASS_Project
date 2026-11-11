"""Unit tests for compass.data.landmarking normal-case behavior."""

import numpy as np
import pandas as pd

from compass.data.landmarking import construct_landmark_example, sample_landmark


def test_sample_landmark_single_visit():
    patient_visits = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "visit_date": pd.to_datetime(["2020-06-01"]),
        }
    )
    rng = np.random.default_rng(0)
    landmark = sample_landmark(patient_visits, rng, index_date=patient_visits["visit_date"].min(), death_date=pd.NaT)
    assert landmark == pd.Timestamp("2020-06-01")


def test_sample_landmark_picks_from_patient_own_visits():
    dates = pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"])
    patient_visits = pd.DataFrame({"patient_id": ["P1"] * 3, "visit_date": dates})
    rng = np.random.default_rng(1)
    for _ in range(20):
        landmark = sample_landmark(patient_visits, rng, index_date=patient_visits["visit_date"].min(), death_date=pd.NaT)
        assert landmark in set(dates)


def test_construct_landmark_example_structure():
    patient_visits = pd.DataFrame(
        {
            "patient_id": ["P1"] * 2,
            "visit_date": pd.to_datetime(["2015-01-01", "2016-01-01"]),
            "bmi": [22.0, 23.0],
        }
    )
    landmark_date = pd.Timestamp("2016-01-01")
    patient_outcomes = pd.Series(
        {
            "mortality_event": 0,
            "mortality_date": pd.NaT,
            "malo_event": 0,
            "malo_date": pd.NaT,
            "cvd_event": 0,
            "cvd_date": pd.NaT,
        }
    )

    example = construct_landmark_example(patient_visits, patient_outcomes, landmark_date)

    assert example["patient_id"] == "P1"
    assert example["landmark_date"] == landmark_date
    assert list(example["input_visits"]["visit_date"]) == list(pd.to_datetime(["2015-01-01", "2016-01-01"]))
    # censoring-blind outcome labels were removed: targets come only from build_targets
    assert "outcome_labels" not in example


def test_first_landmark_skips_pre_index_and_death_date_visits():
    import pytest

    from compass.data.landmarking import first_landmark, select_landmark

    visits = pd.DataFrame({"patient_id": ["P"] * 4,
                           "visit_date": pd.to_datetime(["2014-12-01", "2015-02-01", "2016-01-01", "2017-01-01"])})
    assert first_landmark(visits, index_date=pd.Timestamp("2015-01-15"), death_date=pd.NaT) == pd.Timestamp("2015-02-01")
    # visits on/after index 2016-01-01 are 2016-01-01 (= the death date, not valid) and 2017 (after death)
    with pytest.raises(ValueError, match="no valid landmark"):
        first_landmark(visits, index_date=pd.Timestamp("2016-01-01"), death_date=pd.Timestamp("2016-01-01"))
    rng = np.random.default_rng(0)
    assert select_landmark(visits, rng, "first_post_index", index_date=pd.Timestamp("2015-01-15"), death_date=pd.NaT) \
        == pd.Timestamp("2015-02-01")
    with pytest.raises(ValueError, match="unknown landmark rule"):
        select_landmark(visits, rng, "last", index_date=pd.Timestamp("2015-01-15"), death_date=pd.NaT)

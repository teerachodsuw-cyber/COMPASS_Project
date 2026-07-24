"""Proves the preprocessing pipeline cannot leak future information into the past, and
that the two historically-buggy areas (landmarking leakage, scaler fit-order) are correct.
Each test is explicit and named for the single property it proves -- not folded into
general pipeline tests.
"""

import numpy as np
import pandas as pd
import pytest

from compass.data.landmarking import construct_landmark_example
from compass.data.scaling import apply_scalers, fit_scalers
from compass.data.splitting import patient_level_split
from compass.data.synthetic_generator import generate_synthetic_cohort


def _no_event_outcomes() -> pd.Series:
    return pd.Series(
        {
            "mortality_event": 0,
            "mortality_date": pd.NaT,
            "malo_event": 0,
            "malo_date": pd.NaT,
            "cvd_event": 0,
            "cvd_date": pd.NaT,
        }
    )


def test_no_post_landmark_input_leakage():
    patient_visits = pd.DataFrame(
        {
            "patient_id": ["P1"] * 4,
            "visit_date": pd.to_datetime(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]),
            "bmi": [22.0, 23.0, 24.0, 25.0],
        }
    )
    landmark_date = pd.Timestamp("2016-01-01")

    example = construct_landmark_example(patient_visits, _no_event_outcomes(), landmark_date, horizons=[10])
    input_visits = example["input_visits"]

    assert (input_visits["visit_date"] <= landmark_date).all()
    post_landmark_dates = {pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")}
    assert not post_landmark_dates & set(input_visits["visit_date"])
    assert len(input_visits) == 2


def test_outcome_window_starts_strictly_after_landmark():
    patient_visits = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "visit_date": pd.to_datetime(["2016-01-01"]),
            "bmi": [22.0],
        }
    )
    landmark_date = pd.Timestamp("2016-01-01")
    patient_outcomes = pd.Series(
        {
            "mortality_event": 1,
            "mortality_date": pd.Timestamp("2016-01-01"),  # event ON the landmark date
            "malo_event": 0,
            "malo_date": pd.NaT,
            "cvd_event": 0,
            "cvd_date": pd.NaT,
        }
    )

    example = construct_landmark_example(patient_visits, patient_outcomes, landmark_date, horizons=[10])

    assert example["outcome_labels"]["mortality"][10] == 0


def test_semi_competing_risks_respected():
    patient_visits = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "visit_date": pd.to_datetime(["2015-01-01"]),
            "bmi": [22.0],
        }
    )
    landmark_date = pd.Timestamp("2015-01-01")
    patient_outcomes = pd.Series(
        {
            "mortality_event": 1,
            "mortality_date": pd.Timestamp("2020-01-01"),
            "malo_event": 1,
            "malo_date": pd.Timestamp("2016-01-01"),
            "cvd_event": 1,
            "cvd_date": pd.Timestamp("2018-01-01"),
        }
    )

    example = construct_landmark_example(patient_visits, patient_outcomes, landmark_date, horizons=[10])

    assert example["outcome_labels"]["malo"][10] == 1
    assert example["outcome_labels"]["cvd"][10] == 1
    assert example["outcome_labels"]["mortality"][10] == 1


def test_no_patient_overlap_across_splits():
    # Realistic default event rates -- marginal stratification (unlike the earlier joint
    # 8-way approach, which failed even at the real cohort's n=12,667) handles these fine.
    _, outcomes_df = generate_synthetic_cohort(n_patients=1000, seed=42)
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=42)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    _, outcomes_df_2 = generate_synthetic_cohort(n_patients=1500, seed=99)
    train_ids2, val_ids2, test_ids2 = patient_level_split(outcomes_df_2, 0.7, 0.15, 0.15, seed=99)
    assert set(train_ids2).isdisjoint(val_ids2)
    assert set(train_ids2).isdisjoint(test_ids2)
    assert set(val_ids2).isdisjoint(test_ids2)


def test_committed_sample_cohort_now_stratifies():
    """The committed data/synthetic/sample_outcomes.csv (n=50) previously failed under the
    old joint-8-way stratification (one combination had only 1 patient). Under marginal
    stratification it succeeds: each outcome individually has enough positive cases
    (mortality=7, malo=2, cvd=5) to clear the 2-per-label minimum. This documents that the
    fix actually raised the usable floor, not just moved where the failure text appears.
    """
    outcomes_df = pd.read_csv("data/synthetic/sample_outcomes.csv")
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=42)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(outcomes_df)


def test_marginal_stratification_still_fails_below_its_own_floor():
    """A genuinely too-small n -- malo (~2% base rate) has only 1 positive case here -- must
    still raise a clear, named error rather than silently succeeding with malo absent from
    one or more splits. This is the NEW, much smaller failure boundary (n=20 here) than the
    old joint-8-way approach's boundary (which failed even at n=12,667).
    """
    _, outcomes_df = generate_synthetic_cohort(n_patients=20, seed=42)
    assert outcomes_df["malo_event"].sum() < 2  # confirms this case actually exercises the floor

    with pytest.raises(ValueError, match="malo_event"):
        patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=42)


def test_scaler_fit_uses_only_training_observed_values():
    train_values_df = pd.DataFrame({"bmi": [10.0, 20.0, 30.0, np.nan, np.nan]})
    train_mask_df = pd.DataFrame({"bmi": [1, 1, 1, 0, 0]})

    scalers = fit_scalers(train_values_df, train_mask_df, ["bmi"])

    true_observed_mean = np.mean([10.0, 20.0, 30.0])  # 20.0
    zero_distorted_mean = np.mean([10.0, 20.0, 30.0, 0.0, 0.0])  # 12.0 -- the historical bug

    assert scalers["bmi"].mean_[0] == pytest.approx(true_observed_mean)
    assert scalers["bmi"].mean_[0] != pytest.approx(zero_distorted_mean)


def test_scalers_fit_on_train_only_not_val_test():
    train_values_df = pd.DataFrame({"bmi": [10.0, 20.0, 30.0]})
    train_mask_df = pd.DataFrame({"bmi": [1, 1, 1]})
    scalers = fit_scalers(train_values_df, train_mask_df, ["bmi"])
    train_mean, train_scale = scalers["bmi"].mean_[0], scalers["bmi"].scale_[0]

    # val has a wildly different distribution -- if apply_scalers refit or leaked val
    # stats in, the transform below would not match manual arithmetic using train's
    # mean/scale.
    val_values_df = pd.DataFrame({"bmi": [1000.0, 2000.0, 3000.0]})
    val_mask_df = pd.DataFrame({"bmi": [1, 1, 1]})

    transformed = apply_scalers(val_values_df, val_mask_df, scalers, ["bmi"])

    expected = (val_values_df["bmi"] - train_mean) / train_scale
    pd.testing.assert_series_equal(transformed["bmi"], expected, check_names=False)

    # Applying scalers to val data must not mutate the fitted scaler itself.
    assert scalers["bmi"].mean_[0] == train_mean
    assert scalers["bmi"].scale_[0] == train_scale

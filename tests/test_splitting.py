"""Unit tests for compass.data.splitting normal-case behavior."""

import pytest

from compass.data.splitting import patient_level_split
from compass.data.synthetic_generator import generate_synthetic_cohort


def test_split_fractions_roughly_match_requested():
    # Realistic default event rates (mortality ~15%, malo ~2%, cvd ~10%) -- marginal
    # stratification (unlike the earlier joint-8-way approach) handles these at n=1000.
    _, outcomes_df = generate_synthetic_cohort(n_patients=1000, seed=42)
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=42)

    n = len(outcomes_df)
    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    assert abs(len(train_ids) / n - 0.7) < 0.03
    assert abs(len(val_ids) / n - 0.15) < 0.03
    assert abs(len(test_ids) / n - 0.15) < 0.03


def test_split_covers_all_patients_exactly_once():
    _, outcomes_df = generate_synthetic_cohort(n_patients=1000, seed=7)
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=7)

    all_ids = set(train_ids) | set(val_ids) | set(test_ids)
    assert all_ids == set(outcomes_df["patient_id"])
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_ids)


def test_invalid_fractions_raise():
    _, outcomes_df = generate_synthetic_cohort(n_patients=100, seed=1)
    with pytest.raises(ValueError):
        patient_level_split(outcomes_df, 0.7, 0.15, 0.2, seed=1)  # sums to 1.05


def test_marginal_rates_preserved_across_splits():
    # The actual point of the marginal-stratification approach: each outcome's own rate
    # stays close to its overall rate in every split, independent of the other two outcomes.
    _, outcomes_df = generate_synthetic_cohort(n_patients=2000, seed=42)
    train_ids, val_ids, test_ids = patient_level_split(outcomes_df, 0.7, 0.15, 0.15, seed=42)

    for col in ("mortality_event", "malo_event", "cvd_event"):
        overall_rate = outcomes_df[col].mean()
        for ids in (train_ids, val_ids, test_ids):
            split_rate = outcomes_df.loc[outcomes_df["patient_id"].isin(ids), col].mean()
            assert abs(split_rate - overall_rate) < 0.05


def test_data_fingerprint_and_check():
    import pandas as pd
    import pytest

    from compass.data.splitting import SplitMismatchError, check_fingerprint, data_fingerprint

    outcomes = pd.DataFrame({"patient_id": ["A", "B", "C"]})
    visits = pd.DataFrame({"patient_id": ["A", "A", "B", "C"]})
    fp = data_fingerprint(visits, outcomes)
    assert fp["visit_rows"] == 4 and fp["patients"] == 3
    assert "A" not in str(fp)  # aggregates + hash only, no IDs
    assert data_fingerprint(visits.iloc[::-1], outcomes.iloc[::-1]) == fp  # order-independent
    check_fingerprint(fp, fp)
    with pytest.raises(SplitMismatchError, match="visit_rows: saved 4 vs given 3"):
        check_fingerprint(fp, data_fingerprint(visits.iloc[:3], outcomes))
    with pytest.raises(SplitMismatchError, match="patient ID hash differs"):
        check_fingerprint(fp, data_fingerprint(visits, pd.DataFrame({"patient_id": ["A", "B", "D"]})))

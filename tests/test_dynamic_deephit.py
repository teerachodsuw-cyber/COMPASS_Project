"""Tests for Dynamic-DeepHit: shape correctness, the semi-competing-risks censoring rule
(the core correctness property this architecture exists to get right), and the ability to
actually learn.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.preprocessing import run_preprocessing
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.base import OUTCOME_NAMES, years_to_bin_index
from compass.models.dynamic_deephit.losses import ranking_loss
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.training.batching import collate_examples


def _make_example(patient_id: str, landmark_date: pd.Timestamp, num_features: int = 3) -> dict:
    input_values = pd.DataFrame(np.zeros((1, num_features)), columns=[f"f{i}" for i in range(num_features)])
    input_mask = pd.DataFrame(np.ones((1, num_features)), columns=[f"f{i}" for i in range(num_features)])
    return {
        "patient_id": patient_id,
        "landmark_date": pd.Timestamp(landmark_date),
        "input_values": input_values,
        "input_mask": input_mask,
    }


LANDMARK = pd.Timestamp("2015-01-01")
ADMIN_CUTOFF = LANDMARK + pd.DateOffset(years=30)


def test_output_shape():
    model = DynamicDeepHitModel(num_features=5, hidden_dim=8, num_gru_layers=1, cause_head_hidden_dim=4, num_time_bins=10)
    batch, seq_len, num_features = 4, 3, 5
    values = torch.randn(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.tensor([3, 2, 1, 3])

    hazards = model.forward(values, mask, seq_lengths)

    assert set(hazards.keys()) == set(OUTCOME_NAMES)
    for name in OUTCOME_NAMES:
        assert hazards[name].shape == (batch, 10)
        assert (hazards[name] >= 0).all() and (hazards[name] <= 1).all()


def test_malo_cvd_can_coexist():
    example = _make_example("P1", LANDMARK)
    outcomes_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "mortality_event": 0,
                "mortality_date": pd.NaT,
                "malo_event": 1,
                "malo_date": LANDMARK + pd.DateOffset(years=3),
                "cvd_event": 1,
                "cvd_date": LANDMARK + pd.DateOffset(years=8),
            }
        ]
    )
    batch = collate_examples([example], outcomes_df, ADMIN_CUTOFF, bin_width_years=1.0, num_time_bins=30)

    # The classic joint-softmax DeepHit head would have no valid way to represent this
    # patient (malo and cvd "competing" for the same probability mass) -- proves the
    # independent-heads architecture actually handles it.
    assert batch["malo_event_indicator"].item() == 1
    assert batch["cvd_event_indicator"].item() == 1

    model = DynamicDeepHitModel(num_features=3, hidden_dim=8, num_gru_layers=1, cause_head_hidden_dim=4, num_time_bins=30)
    loss_dict = model.compute_loss(batch)
    assert torch.isfinite(loss_dict["total"])


def test_mortality_censors_malo_and_cvd():
    example = _make_example("P1", LANDMARK)
    death_date = LANDMARK + pd.DateOffset(years=2)
    outcomes_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "mortality_event": 1,
                "mortality_date": death_date,
                "malo_event": 0,
                "malo_date": pd.NaT,
                "cvd_event": 0,
                "cvd_date": pd.NaT,
            }
        ]
    )
    batch = collate_examples([example], outcomes_df, ADMIN_CUTOFF, bin_width_years=1.0, num_time_bins=30)

    death_years = (death_date - LANDMARK).days / 365.25
    admin_years = (ADMIN_CUTOFF - LANDMARK).days / 365.25
    expected_death_bin = years_to_bin_index(torch.tensor([death_years]), 1.0, 30).item()
    admin_cutoff_bin = years_to_bin_index(torch.tensor([admin_years]), 1.0, 30).item()

    assert batch["malo_censoring_bin"].item() == expected_death_bin
    assert batch["cvd_censoring_bin"].item() == expected_death_bin
    assert batch["malo_censoring_bin"].item() != admin_cutoff_bin


def test_malo_does_not_censor_cvd():
    example_a = _make_example("A", LANDMARK)
    outcomes_a = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "mortality_event": 0,
                "mortality_date": pd.NaT,
                "malo_event": 1,
                "malo_date": LANDMARK + pd.DateOffset(years=3),
                "cvd_event": 1,
                "cvd_date": LANDMARK + pd.DateOffset(years=8),
            }
        ]
    )

    example_b = _make_example("B", LANDMARK)
    outcomes_b = pd.DataFrame(
        [
            {
                "patient_id": "B",
                "mortality_event": 0,
                "mortality_date": pd.NaT,
                "malo_event": 0,  # <- only difference from patient A
                "malo_date": pd.NaT,
                "cvd_event": 1,
                "cvd_date": LANDMARK + pd.DateOffset(years=8),  # same cvd as A
            }
        ]
    )

    batch_a = collate_examples([example_a], outcomes_a, ADMIN_CUTOFF, 1.0, 30)
    batch_b = collate_examples([example_b], outcomes_b, ADMIN_CUTOFF, 1.0, 30)

    assert batch_a["cvd_event_indicator"].item() == batch_b["cvd_event_indicator"].item()
    assert batch_a["cvd_event_bin"].item() == batch_b["cvd_event_bin"].item()
    assert batch_a["cvd_censoring_bin"].item() == batch_b["cvd_censoring_bin"].item()


def test_hazards_to_cif():
    # 3 bins, constant hazard 0.5: survival(0)=0.5, survival(1)=0.25, survival(2)=0.125
    # CIF(0)=0.5, CIF(1)=0.75, CIF(2)=0.875
    hazards = torch.tensor([[0.5, 0.5, 0.5]])
    cif = DynamicDeepHitModel.hazards_to_cif(hazards, [0, 1, 2])
    expected = torch.tensor([[0.5, 0.75, 0.875]])
    assert torch.allclose(cif, expected, atol=1e-6)


def test_ranking_loss_direction():
    event_indicator = torch.tensor([1.0, 0.0])
    observed_bin = torch.tensor([2, 8])  # patient 0's event at bin 2, patient 1 censored at bin 8

    correctly_ordered = torch.tensor([0.8, 0.2])  # earlier-event patient scored higher risk
    incorrectly_ordered = torch.tensor([0.2, 0.8])

    loss_correct = ranking_loss(correctly_ordered, event_indicator, observed_bin)
    loss_incorrect = ranking_loss(incorrectly_ordered, event_indicator, observed_bin)

    assert loss_correct < loss_incorrect


@pytest.fixture(scope="module")
def real_pipeline_batch():
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
        smoking_status_categories=[1, 2, 3, 4],
        horizons=[10, 20, 30],
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    admin_cutoff = pd.Timestamp(schema["cohort"]["date_range"][1])
    batch = collate_examples(result.train[:16], outcomes_df, admin_cutoff, bin_width_years=1.0, num_time_bins=30)
    return batch, len(result.feature_names)


def test_loss_components_and_total(real_pipeline_batch):
    batch, num_features = real_pipeline_batch
    model = DynamicDeepHitModel(num_features=num_features, hidden_dim=16, num_gru_layers=1, cause_head_hidden_dim=8, num_time_bins=30)

    loss_dict = model.compute_loss(batch)

    assert set(loss_dict.keys()) == {"survival_nll", "ranking_loss", "auxiliary_loss", "total"}
    expected_total = (
        loss_dict["survival_nll"] + model.ranking_weight * loss_dict["ranking_loss"] + model.auxiliary_weight * loss_dict["auxiliary_loss"]
    )
    assert torch.allclose(loss_dict["total"], expected_total)


def test_overfits_tiny_batch(real_pipeline_batch):
    """Proves the model can actually learn, not just run without crashing."""
    batch, num_features = real_pipeline_batch
    torch.manual_seed(0)
    model = DynamicDeepHitModel(num_features=num_features, hidden_dim=16, num_gru_layers=1, cause_head_hidden_dim=8, num_time_bins=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss_dict = model.compute_loss(batch)
        loss_dict["total"].backward()
        optimizer.step()
        losses.append(loss_dict["total"].item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease substantially: {losses[0]:.4f} -> {losses[-1]:.4f}"

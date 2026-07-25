"""Tests for compass.models.base's shared interface."""

import numpy as np
import pytest
import torch

from compass.models.base import OUTCOME_NAMES, CompassSurvivalModel, quantile_time_grid, years_to_bin_index


class _DummyModel(CompassSurvivalModel):
    """Minimal subclass proving the interface is implementable and its shapes hold."""

    def __init__(self, num_time_bins=10, bin_width_years=1.0):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.bin_width_years = bin_width_years
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, values, mask, seq_lengths):
        batch = values.shape[0]
        # Deterministic, valid-probability dummy hazard per outcome.
        hazard = torch.full((batch, self.num_time_bins), 0.05)
        return {name: hazard.clone() for name in OUTCOME_NAMES}

    def compute_loss(self, batch):
        total = self.linear(torch.zeros(1, 1)).sum()
        return {"dummy": total, "total": total}


def test_forward_output_has_three_outcome_keys_valid_hazards():
    model = _DummyModel()
    batch, seq_len, num_features = 4, 3, 5
    values = torch.zeros(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.long)

    hazards = model.forward(values, mask, seq_lengths)

    assert set(hazards.keys()) == set(OUTCOME_NAMES)
    for name in OUTCOME_NAMES:
        assert hazards[name].shape == (batch, model.num_time_bins)
        assert (hazards[name] >= 0).all() and (hazards[name] <= 1).all()


def test_predict_risk_shape_and_valid_probabilities():
    model = _DummyModel()
    batch, seq_len, num_features = 4, 3, 5
    values = torch.zeros(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.long)
    horizons = [1, 5, 10]

    risk = model.predict_risk(values, mask, seq_lengths, horizons)

    assert risk.shape == (batch, len(OUTCOME_NAMES), len(horizons))
    assert (risk >= 0).all() and (risk <= 1).all()


def test_compute_loss_has_total_key():
    model = _DummyModel()
    loss_dict = model.compute_loss({})
    assert "total" in loss_dict


def test_hazards_to_cif_hand_computed():
    # 3 bins, constant hazard 0.5: survival(0)=0.5, survival(1)=0.25, survival(2)=0.125
    # CIF(0)=0.5, CIF(1)=0.75, CIF(2)=0.875
    hazards = torch.tensor([[0.5, 0.5, 0.5]])
    cif = CompassSurvivalModel.hazards_to_cif(hazards, [0, 1, 2])
    expected = torch.tensor([[0.5, 0.75, 0.875]])
    assert torch.allclose(cif, expected, atol=1e-6)


def test_years_to_bin_index_yearly_bins():
    # bin i covers (i, i+1] years with bin_width_years=1.0
    years = torch.tensor([0.5, 1.0, 1.5, 10.0])
    bins = years_to_bin_index(years, bin_width_years=1.0, num_time_bins=30)
    assert bins.tolist() == [0, 0, 1, 9]


def test_years_to_bin_index_clamped_to_range():
    years = torch.tensor([100.0])
    bins = years_to_bin_index(years, bin_width_years=1.0, num_time_bins=10)
    assert bins.tolist() == [9]


def test_quantile_time_grid_basic():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    boundaries = quantile_time_grid(times, num_bins=5)
    assert boundaries[0] == 0.0
    assert boundaries[-1] == pytest.approx(10.0)
    assert len(boundaries) <= 6


def test_quantile_time_grid_handles_duplicate_quantiles():
    times = np.array([5.0] * 20)
    boundaries = quantile_time_grid(times, num_bins=20)
    assert len(boundaries) == len(np.unique(boundaries))

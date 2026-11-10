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

    def forward(self, values, mask, seq_lengths, elapsed_days=None):
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


# ---- competing-risk CIF (MALO / CVD; death competing) -----------------------------------


def _old_single_cause_cif(h: torch.Tensor) -> torch.Tensor:
    """The pre-competing-risk formula, kept here as the reference: 1 - prod(1 - h)."""
    return 1.0 - torch.cumprod(1.0 - h, dim=1)


ALL_BINS = list(range(13))


def test_competing_cif_matches_closed_form_for_constant_hazards():
    # constant h_k = a, h_death = b: CIF_k(j) = a * (1 - q^(j+1)) / (1 - q), q = (1-a)(1-b)
    a, b = 0.07, 0.04
    h_k = torch.full((1, 13), a, dtype=torch.float64)
    h_d = torch.full((1, 13), b, dtype=torch.float64)
    q = (1 - a) * (1 - b)
    expected = torch.tensor([[a * (1 - q ** (j + 1)) / (1 - q) for j in range(13)]], dtype=torch.float64)
    cif = CompassSurvivalModel.hazards_to_cif(h_k, ALL_BINS, death_hazards=h_d)
    assert torch.allclose(cif, expected, rtol=0, atol=1e-12)


def test_competing_cif_with_zero_death_hazard_equals_old_formula():
    torch.manual_seed(0)
    h_k = torch.rand(5, 13, dtype=torch.float64) * 0.3
    cif = CompassSurvivalModel.hazards_to_cif(h_k, ALL_BINS, death_hazards=torch.zeros_like(h_k))
    assert torch.allclose(cif, _old_single_cause_cif(h_k), rtol=0, atol=1e-12)


def test_competing_cif_below_old_formula_when_death_hazard_positive():
    torch.manual_seed(1)
    h_k = torch.rand(5, 13, dtype=torch.float64) * 0.3 + 0.01
    h_d = torch.rand(5, 13, dtype=torch.float64) * 0.3 + 0.01
    new = CompassSurvivalModel.hazards_to_cif(h_k, ALL_BINS, death_hazards=h_d)
    old = _old_single_cause_cif(h_k)
    assert torch.allclose(new[:, 0], old[:, 0])  # year 1: nobody has died yet (same-year convention)
    assert (new[:, 1:] < old[:, 1:]).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cif_bounded_and_non_decreasing(seed):
    torch.manual_seed(seed)
    hazards = {name: torch.rand(64, 13) for name in OUTCOME_NAMES}  # full (0, 1) range, incl. extreme hazards
    for name in OUTCOME_NAMES:
        cif = CompassSurvivalModel.cif_for_outcome(hazards, name, ALL_BINS)
        assert (cif >= 0).all() and (cif <= 1 + 1e-6).all(), name
        assert (cif[:, 1:] - cif[:, :-1] >= -1e-6).all(), name


def test_malo_and_cvd_have_no_cross_term_and_each_cif_at_most_one():
    torch.manual_seed(3)
    hazards = {name: torch.rand(16, 13) * 0.5 for name in OUTCOME_NAMES}
    cifs = {name: CompassSurvivalModel.cif_for_outcome(hazards, name, ALL_BINS) for name in OUTCOME_NAMES}
    for name in OUTCOME_NAMES:
        assert (cifs[name] <= 1 + 1e-6).all(), name
    # MALO's CIF ignores CVD's hazard entirely (and vice versa); only death competes.
    extreme = dict(hazards, cvd=torch.full((16, 13), 0.99))
    assert torch.equal(CompassSurvivalModel.cif_for_outcome(extreme, "malo", ALL_BINS), cifs["malo"])
    extreme = dict(hazards, malo=torch.full((16, 13), 0.99))
    assert torch.equal(CompassSurvivalModel.cif_for_outcome(extreme, "cvd", ALL_BINS), cifs["cvd"])


def test_mortality_cif_unchanged_from_old_formula():
    torch.manual_seed(4)
    hazards = {name: torch.rand(8, 13) * 0.3 for name in OUTCOME_NAMES}
    assert torch.equal(CompassSurvivalModel.cif_for_outcome(hazards, "mortality", ALL_BINS), _old_single_cause_cif(hazards["mortality"]))
    assert torch.equal(CompassSurvivalModel.hazards_to_cif(hazards["mortality"], ALL_BINS), _old_single_cause_cif(hazards["mortality"]))


def _real_models():
    from compass.models.dynamic_deephit.model import DynamicDeepHitModel
    from compass.models.survlatent_ode.model import SurvLatentODEModel
    from compass.models.tracer.model import TracerModel

    torch.manual_seed(0)
    return {
        "dynamic_deephit": DynamicDeepHitModel(num_features=3, hidden_dim=8, cause_head_hidden_dim=4),
        "tracer": TracerModel(num_features=3, feature_embed_dim=4, covariate_attention_heads=1, hidden_dim=8,
                              temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=4),
        "survlatent_ode": SurvLatentODEModel(num_features=3, latent_dim=4, ode_hidden_dim=8, cause_head_hidden_dim=4, solver_method="rk4"),
    }


@pytest.mark.parametrize("model_name", ["dynamic_deephit", "tracer", "survlatent_ode"])
def test_every_model_predict_risk_uses_competing_cif(model_name):
    model = _real_models()[model_name].eval()
    batch, seq_len = 3, 2
    values, mask = torch.randn(batch, seq_len, 3), torch.ones(batch, seq_len, 3)
    seq_lengths = torch.full((batch,), seq_len, dtype=torch.long)
    elapsed = torch.tensor([[0.0, 200.0]] * batch)
    horizons = [1, 5, 10]
    with torch.no_grad():
        hazards = model.forward(values, mask, seq_lengths, elapsed)  # ODE: forward() == hazards_at_bins()
        risk = model.predict_risk(values, mask, seq_lengths, horizons, elapsed)
    bins = [model.years_to_bin_index(h) for h in horizons]
    for k, name in enumerate(OUTCOME_NAMES):
        expected = CompassSurvivalModel.cif_for_outcome(hazards, name, bins)
        assert torch.allclose(risk[:, k, :], expected, atol=1e-6), name
    # MALO / CVD differ from the old single-cause formula (death hazard > 0) ...
    for k, name in [(1, "malo"), (2, "cvd")]:
        old = _old_single_cause_cif(hazards[name])[:, bins]
        assert (risk[:, k, 1:] < old[:, 1:]).all(), name
    # ... mortality does not
    assert torch.allclose(risk[:, 0, :], _old_single_cause_cif(hazards["mortality"])[:, bins], atol=1e-6)


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

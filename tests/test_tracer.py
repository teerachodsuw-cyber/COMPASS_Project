"""Tests for TRACER: the shared outcome-level invariants every architecture must satisfy
(mirroring test_dynamic_deephit.py), plus TRACER-specific proofs of its own native
missingness mechanism and factorized attention structure.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.preprocessing import run_preprocessing
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.base import OUTCOME_NAMES, years_to_bin_index
from compass.models.tracer.model import TracerModel
from compass.training.batching import collate_examples
from compass.training.targets import compute_last_visit_dates


def _make_example(patient_id: str, landmark_date: pd.Timestamp, num_features: int = 3) -> dict:
    landmark_date = pd.Timestamp(landmark_date)
    input_values = pd.DataFrame(np.zeros((1, num_features)), columns=[f"f{i}" for i in range(num_features)])
    input_mask = pd.DataFrame(np.ones((1, num_features)), columns=[f"f{i}" for i in range(num_features)])
    input_visits = pd.DataFrame({"patient_id": [patient_id], "visit_date": [landmark_date]})
    return {
        "patient_id": patient_id,
        "landmark_date": landmark_date,
        "input_visits": input_visits,
        "input_values": input_values,
        "input_mask": input_mask,
    }


LANDMARK = pd.Timestamp("2015-01-01")
ADMIN_CUTOFF = LANDMARK + pd.DateOffset(years=30)


def _last_visits(outcomes_df: pd.DataFrame) -> pd.Series:
    """Hand-built cases: last visit on the admin cutoff, so MALO/CVD follow-up runs to the
    cutoff and a death before it is a competing event (what these cases were written for)."""
    return pd.Series(ADMIN_CUTOFF, index=outcomes_df["patient_id"])


def _small_model(num_features=5, num_time_bins=10):
    return TracerModel(
        num_features=num_features,
        feature_embed_dim=8,
        covariate_attention_heads=2,
        hidden_dim=16,
        temporal_attention_heads=2,
        num_transformer_layers=1,
        cause_head_hidden_dim=4,
        num_time_bins=num_time_bins,
    )


def test_output_shape():
    model = _small_model(num_features=5, num_time_bins=10)
    batch, seq_len, num_features = 4, 3, 5
    values = torch.randn(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.tensor([3, 2, 1, 3])
    elapsed_days = torch.zeros(batch, seq_len)
    elapsed_days[:, 1:] = 365.0

    hazards = model.forward(values, mask, seq_lengths, elapsed_days)

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
                "cvd_at_risk": 1,
            }
        ]
    )
    batch = collate_examples([example], outcomes_df, ADMIN_CUTOFF, bin_width_years=1.0, num_time_bins=30, last_visit_dates=_last_visits(outcomes_df))

    assert batch["malo_event_indicator"].item() == 1
    assert batch["cvd_event_indicator"].item() == 1

    model = TracerModel(num_features=3, feature_embed_dim=4, covariate_attention_heads=1, hidden_dim=8, temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=4, num_time_bins=30)
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
                "cvd_at_risk": 1,
            }
        ]
    )
    batch = collate_examples([example], outcomes_df, ADMIN_CUTOFF, bin_width_years=1.0, num_time_bins=30, last_visit_dates=_last_visits(outcomes_df))

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
                "cvd_at_risk": 1,
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
                "malo_event": 0,
                "malo_date": pd.NaT,
                "cvd_event": 1,
                "cvd_date": LANDMARK + pd.DateOffset(years=8),
                "cvd_at_risk": 1,
            }
        ]
    )

    batch_a = collate_examples([example_a], outcomes_a, ADMIN_CUTOFF, 1.0, 30, last_visit_dates=_last_visits(outcomes_a))
    batch_b = collate_examples([example_b], outcomes_b, ADMIN_CUTOFF, 1.0, 30, last_visit_dates=_last_visits(outcomes_b))

    assert batch_a["cvd_event_indicator"].item() == batch_b["cvd_event_indicator"].item()
    assert batch_a["cvd_event_bin"].item() == batch_b["cvd_event_bin"].item()
    assert batch_a["cvd_censoring_bin"].item() == batch_b["cvd_censoring_bin"].item()


def test_hazards_to_cif():
    hazards = torch.tensor([[0.5, 0.5, 0.5]])
    cif = TracerModel.hazards_to_cif(hazards, [0, 1, 2])
    expected = torch.tensor([[0.5, 0.75, 0.875]])
    assert torch.allclose(cif, expected, atol=1e-6)


def test_time_decay_gate_downweights_stale_features():
    model = _small_model(num_features=2, num_time_bins=10)
    # 2 features, 3 visits, 365 days apart. Feature 0 observed at every visit (never
    # stale). Feature 1 observed only at visit 0, then missing -- stale by visit 2.
    mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]])  # (1, 3, 2)
    elapsed_days = torch.tensor([[0.0, 365.0, 365.0]])  # (1, 3)

    gate = model.compute_decay_gate(mask, elapsed_days)  # (1, 3, 2)

    recent_feature_weight = gate[0, 2, 0].item()  # observed at visit 2 itself -> weight 1
    stale_feature_weight = gate[0, 2, 1].item()  # last observed 730 days ago -> heavily decayed

    assert recent_feature_weight == pytest.approx(1.0, abs=1e-4)
    assert stale_feature_weight < recent_feature_weight
    assert stale_feature_weight < 0.5  # meaningfully down-weighted, not just marginally lower


def test_factorized_attention_shapes():
    model = _small_model(num_features=4, num_time_bins=10)
    batch, seq_len, num_features = 2, 3, 4
    values = torch.randn(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.tensor([3, 2])
    elapsed_days = torch.zeros(batch, seq_len)

    gate_weight = model.compute_decay_gate(mask, elapsed_days)
    embed_input = torch.stack([values * gate_weight, gate_weight], dim=-1)
    feature_embed = model.feature_embed(embed_input)
    assert feature_embed.shape == (batch, seq_len, num_features, 8)  # feature_embed_dim=8

    covariate_out = model._covariate_attention(feature_embed)
    assert covariate_out.shape == feature_embed.shape  # covariate attention preserves shape

    pooled = covariate_out.mean(dim=2)
    assert pooled.shape == (batch, seq_len, 8)

    projected = model.covariate_to_hidden(pooled)
    assert projected.shape == (batch, seq_len, 16)  # hidden_dim=16

    temporal_out = model._temporal_attention(projected, seq_lengths)
    assert temporal_out.shape == (batch, seq_len, 16)


def test_no_ranking_loss_term():
    import inspect

    from compass.models.tracer.losses import tracer_survival_loss

    source = inspect.getsource(tracer_survival_loss)
    assert "ranking" not in source.lower()
    assert "auxiliary" not in source.lower()

    hazards = {name: torch.full((2, 10), 0.05) for name in OUTCOME_NAMES}
    batch = {}
    for name in OUTCOME_NAMES:
        batch[f"{name}_event_indicator"] = torch.tensor([1.0, 0.0])
        batch[f"{name}_event_bin"] = torch.tensor([3, 3])
        batch[f"{name}_censoring_bin"] = torch.tensor([5, 5])
        batch[f"{name}_at_risk"] = torch.tensor([1.0, 1.0])
    loss_dict = tracer_survival_loss(hazards, batch)

    assert set(loss_dict.keys()) == {"survival_nll", "total"}
    assert torch.allclose(loss_dict["survival_nll"], loss_dict["total"])


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
        smoking_status_categories=[0, 1, 2],
        horizons=[10],
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    admin_cutoff = pd.Timestamp(schema["cohort"]["date_range"][1])
    batch = collate_examples(result.train[:16], outcomes_df, admin_cutoff, bin_width_years=1.0, num_time_bins=30, last_visit_dates=compute_last_visit_dates(visits_df))
    return batch, len(result.feature_names)


def test_single_forward_backward_all_params_get_gradients(real_pipeline_batch):
    batch, num_features = real_pipeline_batch
    model = TracerModel(num_features=num_features, feature_embed_dim=8, covariate_attention_heads=2, hidden_dim=16, temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=8, num_time_bins=30)

    loss_dict = model.compute_loss(batch)
    assert torch.isfinite(loss_dict["total"])
    loss_dict["total"].backward()

    missing_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
    assert not missing_grad, f"parameters with no/zero gradient: {missing_grad}"


# ---- time-decay gate on the scale of visit gaps (tau in years, init ~0.5 yr) -------------

GAPS_DAYS = (30.0, 90.0, 180.0, 365.0)


def _stale_gate(model, gap_days: float) -> torch.Tensor:
    """Gate for every feature at a visit `gap_days` after the visit it was last observed."""
    f = model.num_features
    mask = torch.tensor([[[1.0] * f, [0.0] * f]])            # observed at visit 0, missing at visit 1
    elapsed = torch.tensor([[0.0, gap_days]])
    return model.compute_decay_gate(mask, elapsed)[0, 1]      # (num_features,)


def test_tau_initialised_at_half_a_year():
    model = _small_model(num_features=4)
    assert torch.allclose(model.decay_tau_years(), torch.full((4,), 0.5), atol=1e-6)


@pytest.mark.parametrize("gap_days", GAPS_DAYS)
def test_gate_at_init_is_not_saturated_for_realistic_gaps(gap_days):
    model = _small_model(num_features=4)
    with torch.no_grad():
        gate = _stale_gate(model, gap_days)
    assert ((gate > 0.1) & (gate < 0.99)).all(), (gap_days, gate)
    expected = np.exp(-(gap_days / 365.25) / 0.5)
    assert torch.allclose(gate, torch.full_like(gate, expected), atol=1e-5)


def test_gate_gradient_wrt_decay_parameter_is_nonzero(real_pipeline_batch):
    batch, num_features = real_pipeline_batch
    torch.manual_seed(0)
    model = TracerModel(num_features=num_features, feature_embed_dim=8, covariate_attention_heads=2, hidden_dim=16,
                        temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=8, num_time_bins=30)
    model.compute_loss(batch)["total"].backward()
    grad = model.decay_rate_raw.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    # A feature's tau gets a gradient exactly where that feature is stale (0 < time since
    # last observed < cap) at a real (non-padding) visit; always-observed features (gate == 1,
    # e.g. medication flags, sex) correctly get none.
    with torch.no_grad():
        time_since = model._compute_time_since_last_observed(batch["mask"], batch["elapsed_days"])
    valid = (torch.arange(time_since.shape[1]).unsqueeze(0) < batch["seq_lengths"].unsqueeze(1)).unsqueeze(-1)
    stale = ((time_since > 0) & (time_since < model.max_staleness_years) & valid).any(dim=(0, 1))
    assert stale.any()
    assert (grad[stale].abs() > 0).all()
    assert (grad[~stale] == 0).all()


def test_tau_changes_after_a_few_optimiser_steps(real_pipeline_batch):
    batch, num_features = real_pipeline_batch
    torch.manual_seed(0)
    model = TracerModel(num_features=num_features, feature_embed_dim=8, covariate_attention_heads=2, hidden_dim=16,
                        temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=8, num_time_bins=30)
    tau_before = model.decay_tau_years().detach().clone()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        optimiser.zero_grad()
        model.compute_loss(batch)["total"].backward()
        optimiser.step()
    tau_after = model.decay_tau_years().detach()
    assert not torch.allclose(tau_after, tau_before)
    assert (tau_after > 0).all()

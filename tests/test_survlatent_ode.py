"""Tests for SurvLatent ODE: the shared outcome-level invariants every architecture must
satisfy (mirroring test_dynamic_deephit.py/test_tracer.py), plus proofs specific to this
model's genuinely continuous-time internal representation.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from compass.data.preprocessing import run_preprocessing
from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.base import OUTCOME_NAMES, years_to_bin_index
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.models.tracer.model import TracerModel
from compass.training.batching import collate_examples


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


def _small_model(num_features=5, num_time_bins=10):
    return SurvLatentODEModel(
        num_features=num_features,
        latent_dim=8,
        ode_hidden_dim=16,
        cause_head_hidden_dim=4,
        num_time_bins=num_time_bins,
        solver_method="rk4",  # faster fixed-step solver for tests
    )


def test_output_shape():
    model = _small_model(num_features=5, num_time_bins=10)
    batch, seq_len, num_features = 4, 3, 5
    values = torch.randn(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.tensor([3, 2, 1, 3])
    elapsed_days = torch.zeros(batch, seq_len)
    elapsed_days[:, 1:] = 200.0

    hazards = model.hazards_at_bins(values, mask, seq_lengths, elapsed_days)

    assert set(hazards.keys()) == set(OUTCOME_NAMES)
    for name in OUTCOME_NAMES:
        assert hazards[name].shape == (batch, 10)
        assert (hazards[name] >= 0).all() and (hazards[name] <= 1).all()
        assert torch.isfinite(hazards[name]).all()


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

    assert batch["malo_event_indicator"].item() == 1
    assert batch["cvd_event_indicator"].item() == 1

    model = SurvLatentODEModel(num_features=3, latent_dim=8, ode_hidden_dim=16, cause_head_hidden_dim=4, num_time_bins=30, solver_method="rk4")
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

    death_days = (death_date - LANDMARK).days
    admin_days = (ADMIN_CUTOFF - LANDMARK).days

    assert batch["malo_target_days"].item() == pytest.approx(death_days, abs=1.0)
    assert batch["cvd_target_days"].item() == pytest.approx(death_days, abs=1.0)
    assert batch["malo_target_days"].item() != pytest.approx(admin_days, abs=1.0)


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
                "malo_event": 0,
                "malo_date": pd.NaT,
                "cvd_event": 1,
                "cvd_date": LANDMARK + pd.DateOffset(years=8),
            }
        ]
    )

    batch_a = collate_examples([example_a], outcomes_a, ADMIN_CUTOFF, 1.0, 30)
    batch_b = collate_examples([example_b], outcomes_b, ADMIN_CUTOFF, 1.0, 30)

    assert batch_a["cvd_event_indicator"].item() == batch_b["cvd_event_indicator"].item()
    assert batch_a["cvd_target_days"].item() == pytest.approx(batch_b["cvd_target_days"].item(), abs=1e-3)


def test_hazards_to_cif():
    hazards = torch.tensor([[0.5, 0.5, 0.5]])
    cif = SurvLatentODEModel.hazards_to_cif(hazards, [0, 1, 2])
    expected = torch.tensor([[0.5, 0.75, 0.875]])
    assert torch.allclose(cif, expected, atol=1e-6)


def test_continuous_query_between_bins():
    model = _small_model(num_features=4, num_time_bins=10)
    batch, seq_len, num_features = 2, 2, 4
    values = torch.randn(batch, seq_len, num_features)
    mask = torch.ones(batch, seq_len, num_features)
    seq_lengths = torch.tensor([2, 2])
    elapsed_days = torch.zeros(batch, seq_len)
    elapsed_days[:, 1] = 180.0

    # Query at 1.5 bin-widths (bin_width_years=1.0 -> 547.875 days), strictly BETWEEN the
    # bin-1 and bin-2 boundaries -- not a point hazards_at_bins() would ever land on.
    query_days = torch.full((batch,), 1.5 * 365.25)
    hazards = model.hazard_at_time(values, mask, seq_lengths, elapsed_days, query_days)

    assert set(hazards.keys()) == set(OUTCOME_NAMES)
    for name in OUTCOME_NAMES:
        assert hazards[name].shape == (batch,)
        assert torch.isfinite(hazards[name]).all()
        assert (hazards[name] >= 0).all()  # a rate, not bounded to [0,1]


def test_irregular_timing_affects_output():
    torch.manual_seed(0)
    model = _small_model(num_features=3, num_time_bins=10)
    values = torch.randn(1, 3, 3)
    mask = torch.ones(1, 3, 3)
    seq_lengths = torch.tensor([3])

    # Identical visit VALUES, same total span from first to last visit (~200 days), but
    # different intermediate timing: regular spacing vs. clustered-then-gapped.
    elapsed_regular = torch.tensor([[0.0, 100.0, 100.0]])
    elapsed_irregular = torch.tensor([[0.0, 5.0, 195.0]])

    with torch.no_grad():
        hazards_regular = model.hazards_at_bins(values, mask, seq_lengths, elapsed_regular)
        hazards_irregular = model.hazards_at_bins(values, mask, seq_lengths, elapsed_irregular)

    # At random (untrained) initialization the ODE dynamics are deliberately near-zero (see
    # model.py's stability-motivated small init), so the timing-sensitivity signal is real
    # but can be tiny (~1e-6/1e-7) -- too small for torch.allclose's default tolerance to
    # reliably register as "different" across random seeds. What actually proves the
    # mechanism is being used (mirroring the exact byte-identical-vs-not proof used for
    # Dynamic-DeepHit/TraCeR) is that the outputs are NOT bit-for-bit identical, not that
    # they differ by some practically-meaningful amount pre-training.
    differs = any(not torch.equal(hazards_regular[name], hazards_irregular[name]) for name in OUTCOME_NAMES)
    assert differs, "output should depend on real elapsed timing between visits, not just visit values"


@pytest.fixture(scope="module")
def real_pipeline_setup():
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


def test_single_forward_backward_all_params_get_gradients(real_pipeline_setup):
    batch, num_features = real_pipeline_setup
    model = SurvLatentODEModel(num_features=num_features, latent_dim=16, ode_hidden_dim=32, cause_head_hidden_dim=16, num_time_bins=30)

    loss_dict = model.compute_loss(batch)
    assert torch.isfinite(loss_dict["total"])
    loss_dict["total"].backward()

    missing_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
    assert not missing_grad, f"parameters with no/zero gradient: {missing_grad}"


def test_diverges_from_other_two_models(real_pipeline_setup):
    batch, num_features = real_pipeline_setup
    torch.manual_seed(0)

    ddh = DynamicDeepHitModel(num_features=num_features, hidden_dim=16, num_gru_layers=1, cause_head_hidden_dim=8, num_time_bins=30)
    tracer = TracerModel(num_features=num_features, feature_embed_dim=8, covariate_attention_heads=2, hidden_dim=16, temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=8, num_time_bins=30)
    survlatent = SurvLatentODEModel(num_features=num_features, latent_dim=16, ode_hidden_dim=32, cause_head_hidden_dim=16, num_time_bins=30, solver_method="rk4")

    with torch.no_grad():
        ddh_out = ddh.forward(batch["values"], batch["mask"], batch["seq_lengths"], batch["elapsed_days"])
        tracer_out = tracer.forward(batch["values"], batch["mask"], batch["seq_lengths"], batch["elapsed_days"])
        survlatent_out = survlatent.forward(batch["values"], batch["mask"], batch["seq_lengths"], batch["elapsed_days"])

    for name in OUTCOME_NAMES:
        assert not torch.allclose(ddh_out[name], tracer_out[name]), f"{name}: DDH and TraCeR produced identical output"
        assert not torch.allclose(ddh_out[name], survlatent_out[name]), f"{name}: DDH and SurvLatentODE produced identical output"
        assert not torch.allclose(tracer_out[name], survlatent_out[name]), f"{name}: TraCeR and SurvLatentODE produced identical output"

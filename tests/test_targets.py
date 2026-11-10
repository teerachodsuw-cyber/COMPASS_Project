"""Tests for compass.training.targets.build_targets (the only place outcome times are
computed): last-visit censoring for MALO/CVD, competing death, mortality to the cutoff,
at-risk flags, and their effect on all three models' losses. Also: landmarks are never
sampled on/after the death date.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from compass.data.landmarking import sample_landmark
from compass.models.dynamic_deephit.losses import ranking_loss
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.losses import _outcome_nll
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.models.tracer.model import TracerModel
from compass.training.batching import collate_examples
from compass.training.survival_losses import discrete_time_survival_nll
from compass.training.targets import build_targets, compute_last_visit_dates

L = pd.Timestamp("2015-01-01")
LAST_VISIT = pd.Timestamp("2018-01-01")
CUTOFF = pd.Timestamp("2025-06-30")
NUM_BINS, BIN_WIDTH = 13, 1.0


def _days(date) -> float:
    return float((pd.Timestamp(date) - L).days)


def _outcomes(pid="P", death=None, malo=None, cvd=None, cvd_at_risk=1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [pid],
            "mortality_event": [int(death is not None)],
            "mortality_date": [pd.Timestamp(death) if death else pd.NaT],
            "malo_event": [int(malo is not None)],
            "malo_date": [pd.Timestamp(malo) if malo else pd.NaT],
            "cvd_event": [int(cvd is not None)],
            "cvd_date": [pd.Timestamp(cvd) if cvd else pd.NaT],
            "cvd_at_risk": [cvd_at_risk],
        }
    )


def _targets(outcomes, landmark=L, last_visit=LAST_VISIT):
    pid = outcomes["patient_id"].iloc[0]
    return build_targets([pid], [landmark], outcomes, pd.Series({pid: pd.Timestamp(last_visit)}), CUTOFF, BIN_WIDTH, NUM_BINS)


def _get(t, key) -> float:
    return float(t[key][0])


# ---- censoring rules --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["malo", "cvd"])
def test_no_event_censored_at_last_visit_not_cutoff(name):
    t = _targets(_outcomes())
    assert _get(t, f"{name}_event_indicator") == 0.0
    assert _get(t, f"{name}_competing_indicator") == 0.0
    assert _get(t, f"{name}_at_risk") == 1.0
    assert _get(t, f"{name}_target_days") == _days(LAST_VISIT)
    assert _get(t, f"{name}_target_days") != _days(CUTOFF)
    # 1096 days = 3.0007 yr -> bin 3 (bins are (i, i+1] yr); the cutoff (10.5 yr) would be bin 10
    assert int(t[f"{name}_censoring_bin"][0]) == 3


@pytest.mark.parametrize("name", ["malo", "cvd"])
def test_event_after_last_visit_counted_at_its_date(name):
    event_date = pd.Timestamp("2020-03-01")  # after LAST_VISIT, before cutoff
    t = _targets(_outcomes(**{name: event_date}))
    assert _get(t, f"{name}_event_indicator") == 1.0
    assert _get(t, f"{name}_target_days") == _days(event_date)
    assert int(t[f"{name}_event_bin"][0]) == 5  # ~5.2 yr


@pytest.mark.parametrize("name", ["malo", "cvd"])
def test_death_after_last_visit_is_censored_at_last_visit_not_competing(name):
    t = _targets(_outcomes(death="2020-03-01"))
    assert _get(t, f"{name}_event_indicator") == 0.0
    assert _get(t, f"{name}_competing_indicator") == 0.0
    assert _get(t, f"{name}_target_days") == _days(LAST_VISIT)


@pytest.mark.parametrize("name", ["malo", "cvd"])
def test_death_before_last_visit_is_competing(name):
    death = pd.Timestamp("2017-06-01")  # L < death <= C = last visit
    t = _targets(_outcomes(death=death), last_visit=pd.Timestamp("2017-06-01"))
    assert _get(t, f"{name}_event_indicator") == 0.0
    assert _get(t, f"{name}_competing_indicator") == 1.0
    assert _get(t, f"{name}_target_days") == _days(death)


def test_mortality_runs_to_cutoff_past_last_visit():
    alive = _targets(_outcomes())
    assert _get(alive, "mortality_event_indicator") == 0.0
    assert _get(alive, "mortality_target_days") == _days(CUTOFF)
    assert _get(alive, "mortality_at_risk") == 1.0
    died = _targets(_outcomes(death="2020-03-01"))
    assert _get(died, "mortality_event_indicator") == 1.0
    assert _get(died, "mortality_target_days") == _days("2020-03-01")
    assert _get(died, "mortality_competing_indicator") == 0.0


def test_event_after_cutoff_is_not_an_event():
    t = _targets(_outcomes(malo="2025-07-01"), last_visit=pd.Timestamp("2025-06-01"))
    assert _get(t, "malo_event_indicator") == 0.0
    assert _get(t, "malo_target_days") == _days("2025-06-01")  # censored at C = last visit


# ---- at-risk flags ----------------------------------------------------------------------


@pytest.mark.parametrize("malo_date", [L, L - pd.Timedelta(days=30)], ids=["on_landmark", "before_landmark"])
def test_malo_on_or_before_landmark_not_at_risk(malo_date):
    t = _targets(_outcomes(malo=malo_date))
    assert _get(t, "malo_at_risk") == 0.0
    assert _get(t, "malo_event_indicator") == 0.0
    assert _get(t, "cvd_at_risk") == 1.0 and _get(t, "mortality_at_risk") == 1.0


def test_cvd_not_at_risk_when_flag_zero_or_prior_event():
    assert _get(_targets(_outcomes(cvd_at_risk=0)), "cvd_at_risk") == 0.0
    assert _get(_targets(_outcomes(cvd=L)), "cvd_at_risk") == 0.0
    assert _get(_targets(_outcomes(cvd="2016-01-01")), "cvd_at_risk") == 1.0


def test_zero_followup_not_at_risk():
    # landmark on the last visit, no event -> C == L -> MALO/CVD not at risk
    t = _targets(_outcomes(), landmark=LAST_VISIT, last_visit=LAST_VISIT)
    assert _get(t, "malo_at_risk") == 0.0
    assert _get(t, "cvd_at_risk") == 0.0
    assert _get(t, "mortality_at_risk") == 1.0  # mortality still followed to the cutoff
    # ... but an event after the last visit keeps the outcome at risk
    t = _targets(_outcomes(malo="2019-01-01"), landmark=LAST_VISIT, last_visit=LAST_VISIT)
    assert _get(t, "malo_at_risk") == 1.0 and _get(t, "malo_event_indicator") == 1.0
    # landmark on the cutoff -> mortality has zero follow-up too
    t = _targets(_outcomes(), landmark=CUTOFF, last_visit=CUTOFF)
    assert _get(t, "mortality_at_risk") == 0.0


# ---- wiring -----------------------------------------------------------------------------


def test_last_visit_from_full_table_and_collate_uses_build_targets():
    visits = pd.DataFrame(
        {"patient_id": ["P"] * 3, "visit_date": pd.to_datetime(["2014-01-01", "2015-01-01", "2018-01-01"])}
    )
    last_visits = compute_last_visit_dates(visits)
    assert last_visits["P"] == LAST_VISIT
    # The example is truncated at L (input_visits end on L); censoring still uses the full-table last visit.
    example = {
        "patient_id": "P",
        "landmark_date": L,
        "input_values": pd.DataFrame([[0.0]], columns=["f0"]),
        "input_mask": pd.DataFrame([[1.0]], columns=["f0"]),
        "input_visits": visits[visits.visit_date <= L],
    }
    batch = collate_examples([example], _outcomes(), CUTOFF, BIN_WIDTH, NUM_BINS, last_visit_dates=last_visits)
    expected = _targets(_outcomes())
    for key, value in expected.items():
        assert torch.equal(batch[key], value), key
    assert float(batch["malo_target_days"][0]) == _days(LAST_VISIT)


# ---- losses: at_risk == 0 contributes nothing; mean over at-risk examples only ----------


def _two_patient_batch(num_features=2):
    """P0: cvd_at_risk == 0 (with a CVD event in the raw data -- must be ignored).
    P1: at risk, CVD event. Both visits at L, last visit 2018."""
    outcomes = pd.concat(
        [
            _outcomes("P0", cvd="2016-06-01", cvd_at_risk=0),
            _outcomes("P1", cvd="2016-06-01", malo="2017-01-01"),
        ],
        ignore_index=True,
    )
    examples = [
        {
            "patient_id": pid,
            "landmark_date": L,
            "input_values": pd.DataFrame(np.full((1, num_features), float(i)), columns=[f"f{j}" for j in range(num_features)]),
            "input_mask": pd.DataFrame(np.ones((1, num_features)), columns=[f"f{j}" for j in range(num_features)]),
            "input_visits": pd.DataFrame({"patient_id": [pid], "visit_date": [L]}),
        }
        for i, pid in enumerate(["P0", "P1"])
    ]
    last_visits = pd.Series({"P0": LAST_VISIT, "P1": LAST_VISIT})
    return collate_examples(examples, outcomes, CUTOFF, BIN_WIDTH, NUM_BINS, last_visit_dates=last_visits)


def _models(num_features=2):
    torch.manual_seed(0)
    return {
        "dynamic_deephit": DynamicDeepHitModel(num_features=num_features, hidden_dim=8, cause_head_hidden_dim=4, num_time_bins=NUM_BINS),
        "tracer": TracerModel(
            num_features=num_features, feature_embed_dim=4, covariate_attention_heads=1, hidden_dim=8,
            temporal_attention_heads=2, num_transformer_layers=1, cause_head_hidden_dim=4, num_time_bins=NUM_BINS,
        ),
        "survlatent_ode": SurvLatentODEModel(
            num_features=num_features, latent_dim=4, ode_hidden_dim=8, cause_head_hidden_dim=4,
            num_time_bins=NUM_BINS, solver_method="rk4",
        ),
    }


@pytest.mark.parametrize("model_name", ["dynamic_deephit", "tracer", "survlatent_ode"])
def test_cvd_not_at_risk_contributes_zero_cvd_loss(model_name):
    batch = _two_patient_batch()
    assert batch["cvd_at_risk"].tolist() == [0.0, 1.0]
    model = _models()[model_name].eval()
    with torch.no_grad():
        base = model.compute_loss(batch)["total"]
        # Scramble every CVD target of the not-at-risk patient: the loss must not move.
        scrambled = dict(batch)
        for key, value in {"event_indicator": 1.0, "event_bin": 7, "censoring_bin": 7, "target_days": 2500.0}.items():
            t = batch[f"cvd_{key}"].clone()
            t[0] = value
            scrambled[f"cvd_{key}"] = t
        assert float(model.compute_loss(scrambled)["total"]) == pytest.approx(float(base), rel=1e-6)
        # ... whereas the same scramble with P0 at risk does move it.
        at_risk = dict(scrambled)
        at_risk["cvd_at_risk"] = torch.tensor([1.0, 1.0])
        assert float(model.compute_loss(at_risk)["total"]) != pytest.approx(float(base), rel=1e-6)


def test_discrete_nll_averages_over_at_risk_examples_only():
    torch.manual_seed(0)
    hazard = torch.rand(3, NUM_BINS) * 0.3
    event = torch.tensor([1.0, 0.0, 0.0])
    ebin = torch.tensor([2, 5, 8])
    at_risk = torch.tensor([1.0, 0.0, 1.0])
    masked = discrete_time_survival_nll(hazard, event, ebin, ebin, at_risk)
    subset = discrete_time_survival_nll(hazard[[0, 2]], event[[0, 2]], ebin[[0, 2]], ebin[[0, 2]])
    assert float(masked) == pytest.approx(float(subset))
    assert float(discrete_time_survival_nll(hazard, event, ebin, ebin, torch.zeros(3))) == 0.0


def test_ode_nll_averages_over_at_risk_examples_only():
    model = _models()["survlatent_ode"]
    state = torch.randn(3, 4)
    event = torch.tensor([1.0, 0.0, 0.0])
    days = torch.tensor([400.0, 900.0, 1500.0])
    at_risk = torch.tensor([1.0, 0.0, 1.0])
    with torch.no_grad():
        masked = _outcome_nll(model, state, event, days, "malo", at_risk)
        subset = _outcome_nll(model, state[[0, 2]], event[[0, 2]], days[[0, 2]], "malo")
    assert float(masked) == pytest.approx(float(subset), rel=1e-5)


def test_ranking_loss_ignores_not_at_risk_pairs():
    scores = torch.tensor([0.9, 0.1, 0.5])
    event = torch.tensor([1.0, 0.0, 0.0])
    observed = torch.tensor([1, 5, 5])
    at_risk = torch.tensor([1.0, 0.0, 1.0])
    masked = ranking_loss(scores, event, observed, at_risk=at_risk)
    subset = ranking_loss(scores[[0, 2]], event[[0, 2]], observed[[0, 2]])
    assert float(masked) == pytest.approx(float(subset))


# ---- landmark sampling --------------------------------------------------------------------


def test_landmark_on_death_date_never_sampled():
    death = pd.Timestamp("2019-01-01")
    visits = pd.DataFrame(
        {"patient_id": ["P"] * 4, "visit_date": pd.to_datetime(["2015-01-01", "2017-01-01", "2019-01-01", "2019-02-01"])}
    )
    rng = np.random.default_rng(0)
    landmarks = {sample_landmark(visits, rng, death_date=death) for _ in range(200)}
    assert landmarks == {pd.Timestamp("2015-01-01"), pd.Timestamp("2017-01-01")}
    assert all(lm < death for lm in landmarks)
    # alive patient: every visit is a candidate
    alive = {sample_landmark(visits, rng, death_date=pd.NaT) for _ in range(200)}
    assert len(alive) == 4


def test_no_visit_before_death_raises():
    visits = pd.DataFrame({"patient_id": ["P"], "visit_date": pd.to_datetime(["2019-01-01"])})
    with pytest.raises(ValueError, match="no visit before death_date"):
        sample_landmark(visits, np.random.default_rng(0), death_date=pd.Timestamp("2019-01-01"))

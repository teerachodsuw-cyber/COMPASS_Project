"""The binned window is (0, num_time_bins * bin_width_years] years from landmark. With the
default 13 yearly bins it covers all observable follow-up (2013-06-01 -> 2025-06-30 ~ 12.1 yr);
the evaluation horizon stays 10 yr (bin 9). An event or censoring time after the end of the
last bin is a safety-net case: it must become "censored at the end of the last bin" (survived
all bins), never be clipped into the last bin as an event. Covers batching (all three models'
targets), the shared discrete-time NLL (Dynamic-DeepHit, TRACER), SurvLatent ODE's
continuous-time loss, and the 10-yr horizon bin used by the DDH ranking loss / CIF readout.

Window-dependent tests are parameterised on num_time_bins (10 and 13).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from compass.models.base import years_to_bin_index
from compass.models.dynamic_deephit.model import DynamicDeepHitModel
from compass.models.survlatent_ode.losses import _outcome_nll
from compass.models.survlatent_ode.model import SurvLatentODEModel
from compass.training.batching import collate_examples
from compass.training.survival_losses import discrete_time_survival_nll

LANDMARK = pd.Timestamp("2010-01-01")
ADMIN_CUTOFF = LANDMARK + pd.DateOffset(years=15)
BIN_WIDTH = 1.0
YEAR_DAYS = 365.25
HORIZON_YEARS = 10
HORIZON_BIN = 9

# Row order of the fixture batch.
IDS = ["late", "inside", "ten", "beyond", "alive", "died"]
LATE, INSIDE, TEN, BEYOND, ALIVE, DIED = range(len(IDS))


def _example(pid: str) -> dict:
    return {
        "patient_id": pid,
        "landmark_date": LANDMARK,
        "input_values": pd.DataFrame([[0.0]], columns=["f0"]),
        "input_mask": pd.DataFrame([[1.0]], columns=["f0"]),
        "input_visits": pd.DataFrame({"patient_id": [pid], "visit_date": [LANDMARK]}),
    }


def _years(y: float) -> pd.Timestamp:
    return LANDMARK + pd.Timedelta(days=int(y * YEAR_DAYS))


def _last_visits(ids) -> pd.Series:
    # Last visit on the 15-yr cutoff: MALO/CVD follow-up runs to the cutoff, so these cases
    # isolate the window cap (last-visit censoring is covered in tests/test_targets.py).
    return pd.Series(ADMIN_CUTOFF, index=list(ids))


def _window_days(num_bins: int) -> float:
    return num_bins * BIN_WIDTH * YEAR_DAYS


@pytest.fixture(params=[10, 13], ids=lambda n: f"{n}bins")
def num_bins(request) -> int:
    return request.param


@pytest.fixture()
def batch(num_bins):
    # late:   malo event at 12 yr        inside: malo event at 9.5 yr
    # ten:    malo event 10 calendar years after landmark (3652 days, just under 10.0 yr)
    # beyond: malo event at 14 yr (after the last bin for both 10 and 13 bins)
    # alive:  no event, censored at the 15-yr admin cutoff
    # died:   death at 11 yr, no malo
    outcomes = pd.DataFrame(
        {
            "patient_id": IDS,
            "mortality_event": [0, 0, 0, 0, 0, 1],
            "mortality_date": [pd.NaT] * 5 + [_years(11)],
            "malo_event": [1, 1, 1, 1, 0, 0],
            "malo_date": [_years(12), _years(9.5), LANDMARK + pd.DateOffset(years=10), _years(14), pd.NaT, pd.NaT],
            "cvd_event": [0] * 6,
            "cvd_date": [pd.NaT] * 6,
            "cvd_at_risk": [1] * 6,
        }
    )
    return collate_examples(
        [_example(p) for p in IDS], outcomes, ADMIN_CUTOFF, BIN_WIDTH, num_bins, last_visit_dates=_last_visits(IDS)
    )


# ---- batching -------------------------------------------------------------------------


def test_event_after_window_is_censored_at_last_bin_not_an_event(batch, num_bins):
    # 14 yr is past the last bin for both settings -> censored at the window end.
    assert batch["malo_event_indicator"].tolist()[BEYOND] == 0.0
    assert batch["malo_censoring_bin"].tolist()[BEYOND] == num_bins - 1
    assert batch["malo_target_days"].tolist()[BEYOND] == pytest.approx(_window_days(num_bins))
    # 12 yr: past the window with 10 bins (censored at bin 9), inside it with 13 bins.
    if num_bins == 10:
        assert batch["malo_event_indicator"].tolist()[LATE] == 0.0
        assert batch["malo_censoring_bin"].tolist()[LATE] == 9
        assert batch["malo_target_days"].tolist()[LATE] == pytest.approx(_window_days(10))


def test_event_at_12yr_with_13_bins_stays_event_in_bin_11():
    outcomes = pd.DataFrame(
        {
            "patient_id": ["late"],
            "mortality_event": [0], "mortality_date": [pd.NaT],
            "malo_event": [1], "malo_date": [_years(12)],
            "cvd_event": [0], "cvd_date": [pd.NaT], "cvd_at_risk": [1],
        }
    )
    b = collate_examples([_example("late")], outcomes, ADMIN_CUTOFF, BIN_WIDTH, 13, last_visit_dates=_last_visits(["late"]))
    assert b["malo_event_indicator"].tolist() == [1.0]
    assert b["malo_event_bin"].tolist() == [11]
    assert b["malo_target_days"].tolist()[0] == pytest.approx(12 * YEAR_DAYS)


def test_event_at_exactly_10yr_is_bin_9(batch, num_bins):
    # Bin i covers (i, i+1] years, so exactly 10.0 yr is the closed right edge of bin 9.
    assert int(years_to_bin_index(torch.tensor(10.0), BIN_WIDTH, num_bins)) == HORIZON_BIN
    assert int(years_to_bin_index(torch.tensor(10.0, dtype=torch.float32), BIN_WIDTH, num_bins)) == HORIZON_BIN
    # Through batching: 10 calendar years after landmark -> event, bin 9.
    assert batch["malo_event_indicator"].tolist()[TEN] == 1.0
    assert batch["malo_event_bin"].tolist()[TEN] == HORIZON_BIN


def test_events_inside_window_stay_events(batch, num_bins):
    assert batch["malo_event_indicator"].tolist()[INSIDE] == 1.0
    assert batch["malo_event_bin"].tolist()[INSIDE] == 9
    assert batch["malo_target_days"].tolist()[INSIDE] == pytest.approx(int(9.5 * YEAR_DAYS), abs=0.5)


def test_censoring_after_window_is_capped_at_window(batch, num_bins):
    window = _window_days(num_bins)
    # alive patient: censored at the 15-yr cutoff -> capped at the window end
    assert batch["malo_event_indicator"].tolist()[ALIVE] == 0.0
    assert batch["malo_censoring_bin"].tolist()[ALIVE] == num_bins - 1
    assert batch["malo_target_days"].tolist()[ALIVE] == pytest.approx(window)
    # death at 11 yr: outside a 10-bin window (censored at 10 yr), inside a 13-bin one (event, bin 10)
    if num_bins == 10:
        assert batch["mortality_event_indicator"].tolist()[DIED] == 0.0
        assert batch["mortality_censoring_bin"].tolist()[DIED] == 9
        assert batch["mortality_target_days"].tolist()[DIED] == pytest.approx(window)
    else:
        assert batch["mortality_event_indicator"].tolist()[DIED] == 1.0
        assert batch["mortality_event_bin"].tolist()[DIED] == 10
        # malo for the same patient: censored at death (11 yr), not at the window end
        assert batch["malo_event_indicator"].tolist()[DIED] == 0.0
        assert batch["malo_censoring_bin"].tolist()[DIED] == 10


def test_no_target_exceeds_window(batch, num_bins):
    for name in ("mortality", "malo", "cvd"):
        assert float(batch[f"{name}_target_days"].max()) <= _window_days(num_bins) + 1e-3
        assert int(batch[f"{name}_event_bin"].max()) <= num_bins - 1
        assert int(batch[f"{name}_censoring_bin"].max()) <= num_bins - 1


# ---- shared discrete-time NLL ---------------------------------------------------------


def test_shared_nll_treats_out_of_range_event_bin_as_censored_at_last_bin(num_bins):
    last = num_bins - 1
    hazard = torch.linspace(0.05, 0.3, num_bins).unsqueeze(0)
    beyond = discrete_time_survival_nll(hazard, torch.tensor([1.0]), torch.tensor([num_bins + 2]), torch.tensor([num_bins + 2]))
    censored_all_bins = discrete_time_survival_nll(hazard, torch.tensor([0.0]), torch.tensor([last]), torch.tensor([last]))
    event_in_last_bin = discrete_time_survival_nll(hazard, torch.tensor([1.0]), torch.tensor([last]), torch.tensor([last]))
    assert float(beyond) == pytest.approx(float(censored_all_bins))
    assert float(beyond) != pytest.approx(float(event_in_last_bin))


# ---- SurvLatent ODE -------------------------------------------------------------------


def _ode(num_bins: int) -> SurvLatentODEModel:
    torch.manual_seed(0)
    return SurvLatentODEModel(
        num_features=1, latent_dim=4, ode_hidden_dim=8, cause_head_hidden_dim=4,
        num_time_bins=num_bins, bin_width_years=BIN_WIDTH, solver_method="rk4",
    )


def test_survlatent_ode_caps_target_at_window_end_and_censors_late_events(num_bins):
    model = _ode(num_bins)
    state = torch.randn(1, 4)
    window = _window_days(num_bins)
    with torch.no_grad():
        late_event = _outcome_nll(model, state, torch.tensor([1.0]), torch.tensor([window + 2 * YEAR_DAYS]), "malo")
        censored_at_window = _outcome_nll(model, state, torch.tensor([0.0]), torch.tensor([window]), "malo")
        late_censor = _outcome_nll(model, state, torch.tensor([0.0]), torch.tensor([window + 5 * YEAR_DAYS]), "malo")
        event_at_window = _outcome_nll(model, state, torch.tensor([1.0]), torch.tensor([window]), "malo")
    assert float(late_event) == pytest.approx(float(censored_at_window), rel=1e-5)
    assert float(late_censor) == pytest.approx(float(censored_at_window), rel=1e-5)
    assert float(late_event) != pytest.approx(float(event_at_window))


def test_survlatent_ode_13_bins_does_not_cap_at_10yr():
    model = _ode(13)
    state = torch.randn(1, 4)
    with torch.no_grad():
        event_12 = _outcome_nll(model, state, torch.tensor([1.0]), torch.tensor([12 * YEAR_DAYS]), "malo")
        censored_12 = _outcome_nll(model, state, torch.tensor([0.0]), torch.tensor([12 * YEAR_DAYS]), "malo")
        censored_10 = _outcome_nll(model, state, torch.tensor([0.0]), torch.tensor([10 * YEAR_DAYS]), "malo")
    assert float(event_12) != pytest.approx(float(censored_12))  # still an event at 12 yr
    assert float(censored_12) != pytest.approx(float(censored_10))  # integrated past 10 yr


# ---- 10-yr horizon bin (DDH ranking loss, CIF readout) --------------------------------


def test_ddh_ranking_uses_horizon_bin(num_bins):
    model = DynamicDeepHitModel(num_features=1, num_time_bins=num_bins, bin_width_years=BIN_WIDTH, horizon_years=HORIZON_YEARS)
    assert model.horizon_bin == HORIZON_BIN  # the horizon's bin, not the last bin
    wider = DynamicDeepHitModel(num_features=1, num_time_bins=30, bin_width_years=BIN_WIDTH, horizon_years=HORIZON_YEARS)
    assert wider.horizon_bin == HORIZON_BIN
    with pytest.raises(ValueError):
        DynamicDeepHitModel(num_features=1, num_time_bins=num_bins, bin_width_years=BIN_WIDTH, horizon_years=num_bins + 1)


def test_10yr_cif_reads_bins_0_to_9_only(num_bins):
    model = DynamicDeepHitModel(num_features=1, num_time_bins=num_bins, bin_width_years=BIN_WIDTH, horizon_years=HORIZON_YEARS)
    assert model.years_to_bin_index(HORIZON_YEARS) == HORIZON_BIN  # predict_risk's lookup
    torch.manual_seed(1)
    hazards = torch.rand(4, num_bins) * 0.2

    cif_10 = model.hazards_to_cif(hazards, [model.horizon_bin])[:, 0]
    expected = 1.0 - torch.prod(1.0 - hazards[:, :10], dim=1)
    assert torch.allclose(cif_10, expected)

    # Changing bins after bin 9 must not move the 10-yr CIF ...
    later_changed = hazards.clone()
    later_changed[:, 10:] = 0.9
    assert torch.allclose(model.hazards_to_cif(later_changed, [model.horizon_bin])[:, 0], cif_10)
    # ... while changing bin 9 must.
    bin9_changed = hazards.clone()
    bin9_changed[:, 9] = 0.9
    assert not torch.allclose(model.hazards_to_cif(bin9_changed, [model.horizon_bin])[:, 0], cif_10)


def test_10yr_competing_cif_reads_bins_0_to_9_only(num_bins):
    # MALO/CVD competing-risk CIF at 10 yr (bin 9) reads the cause hazard in bins 0-9 and the
    # death hazard in bins 0-8 only: by the same-year convention an event in year s needs
    # survival from death through year s-1, so death in bin 9 cannot pre-empt a bin-9 event.
    model = DynamicDeepHitModel(num_features=1, num_time_bins=num_bins, bin_width_years=BIN_WIDTH, horizon_years=HORIZON_YEARS)
    torch.manual_seed(2)
    hazards = {"mortality": torch.rand(4, num_bins) * 0.2, "malo": torch.rand(4, num_bins) * 0.2, "cvd": torch.rand(4, num_bins) * 0.2}
    cif_10 = model.cif_for_outcome(hazards, "malo", [model.horizon_bin])[:, 0]

    def _with(name, bins, value):
        changed = {k: v.clone() for k, v in hazards.items()}
        changed[name][:, bins] = value
        return model.cif_for_outcome(changed, "malo", [model.horizon_bin])[:, 0]

    for name in ("malo", "mortality"):
        assert torch.allclose(_with(name, slice(10, None), 0.9), cif_10), name   # after bin 9: no effect
    assert not torch.allclose(_with("malo", 9, 0.9), cif_10)                    # cause hazard, bin 9: counts
    assert torch.allclose(_with("mortality", 9, 0.9), cif_10)                   # death hazard, bin 9: no effect
    assert not torch.allclose(_with("mortality", 8, 0.9), cif_10)               # death hazard, bin 8: counts

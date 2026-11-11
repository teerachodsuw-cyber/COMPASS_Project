"""SurvLatent ODE continuous-time survival loss: per-outcome negative log-likelihood using
the model's own ODE solver to compute the survival integral (cumulative hazard), using the
SAME per-outcome targets as Dynamic-DeepHit/TraCeR (compass.training.targets.build_targets;
a competing death is censoring at the death date) -- here expressed as WHICH continuous time
each outcome's likelihood integrates up to, rather than a discrete censoring bin. Each
outcome is averaged over its at-risk examples only.

- event occurred at continuous time t: likelihood = hazard_rate(t) * survival(t), where
  survival(t) = exp(-integral of hazard from 0 to t).
- censored at continuous time t: likelihood = survival(t).

The integral (cumulative hazard) is computed by the ODE solver itself via an augmented
state (see model._evolve_with_hazard_integral), not a separate numerical integration pass.

No ranking loss, no auxiliary prediction loss -- not part of SurvLatent ODE's design either,
same simplicity as TraCeR.
"""

import torch

from compass.models.base import OUTCOME_NAMES
from compass.training.survival_losses import at_risk_mean


def _cap_to_window(model, event_indicator: torch.Tensor, target_days: torch.Tensor):
    """Cap the integration target at the end of the last bin (num_time_bins *
    bin_width_years, 13 yr with the default 13 yearly bins -- the binned window, not the
    10-yr evaluation horizon): an event after the cap is treated as censored at the cap, and
    a censoring time after the cap is censored at the cap. collate_examples already applies
    this cap; it is repeated here defensively."""
    max_days = float(model.num_time_bins * model.bin_width_years * 365.25)
    event_beyond_window = target_days > max_days + 1e-3  # tolerance for float32 days<->years round-trip
    event_indicator = event_indicator.float() * (~event_beyond_window).float()
    return event_indicator, torch.clamp(target_days, max=max_days)


def _nll_from_integral(model, outcome_name, event_indicator, h_at_target, cum_hazard_at_target, at_risk):
    hazard_rate_at_t = model._hazard_rate(outcome_name, h_at_target)
    log_lik_event = torch.log(torch.clamp(hazard_rate_at_t, min=1e-8)) - cum_hazard_at_target
    log_lik_censored = -cum_hazard_at_target
    log_lik = event_indicator * log_lik_event + (1.0 - event_indicator) * log_lik_censored
    return at_risk_mean(-log_lik, at_risk)


def _outcome_nll(
    model,
    landmark_state: torch.Tensor,
    event_indicator: torch.Tensor,
    target_days: torch.Tensor,
    outcome_name: str,
    at_risk: torch.Tensor | None = None,
) -> torch.Tensor:
    """One outcome's NLL on its own. target_days: (batch,) days from landmark to this
    outcome's event or censoring time (build_targets), capped at the binned window (see
    _cap_to_window). at_risk: (batch,) 0/1 -- averaged over at-risk examples only.
    survlatent_ode_survival_loss() computes all three outcomes from ONE integration instead.
    """
    event_indicator, target_days = _cap_to_window(model, event_indicator, target_days)
    h_at, cum_at = model._integrate_hazard_to_targets(landmark_state, {outcome_name: target_days})
    return _nll_from_integral(model, outcome_name, event_indicator, h_at[outcome_name], cum_at[outcome_name], at_risk)


def survlatent_ode_survival_loss(
    model,
    landmark_state: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """batch must contain, per outcome, {name}_event_indicator, {name}_target_days and
    {name}_at_risk (see compass.training.batching.collate_examples).

    ONE integration per batch up to the furthest target, each outcome read at its own
    target (model._integrate_hazard_to_targets) -- not one solve per outcome.

    Returns {"survival_nll", "total", "nll_mortality", "nll_malo", "nll_cvd"} -- total IS the
    survival NLL, the unweighted sum of the three outcomes' independent contributions (also
    returned per outcome, for logging). Nothing else is added, same as TraCeR.
    """
    events, targets = {}, {}
    for name in OUTCOME_NAMES:
        events[name], targets[name] = _cap_to_window(model, batch[f"{name}_event_indicator"], batch[f"{name}_target_days"])
    h_at, cum_at = model._integrate_hazard_to_targets(landmark_state, targets)
    nll_terms = {
        name: _nll_from_integral(model, name, events[name], h_at[name], cum_at[name], batch[f"{name}_at_risk"])
        for name in OUTCOME_NAMES
    }
    total = sum(nll_terms[name] for name in OUTCOME_NAMES)  # unweighted sum (no class-imbalance weighting)
    return {"survival_nll": total, "total": total, **{f"nll_{name}": nll_terms[name] for name in OUTCOME_NAMES}}

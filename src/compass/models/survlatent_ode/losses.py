"""SurvLatent ODE continuous-time survival loss: per-outcome negative log-likelihood using
the model's own ODE solver to compute the survival integral (cumulative hazard), applying
the SAME per-outcome censoring rule as Dynamic-DeepHit/TraCeR (mortality censors malo/cvd;
malo and cvd do not censor each other) -- here expressed as WHICH continuous time each
outcome's likelihood integrates up to, rather than a discrete censoring bin.

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


def _outcome_nll(
    model,
    landmark_state: torch.Tensor,
    event_indicator: torch.Tensor,
    target_days: torch.Tensor,
    outcome_name: str,
) -> torch.Tensor:
    """target_days: (batch,) days from landmark to this outcome's event time (if it
    occurred) or its censoring time (under the project's per-outcome censoring rule) --
    same value construct_landmark_example/collate_examples already derive for the discrete-
    time models, exposed here in continuous (days) form rather than a discrete bin index.
    """
    evolved_h, cum_hazard = model._integrate_hazard_to(landmark_state, target_days)
    cum_hazard_outcome = cum_hazard[outcome_name]
    hazard_rate_at_t = model._hazard_rate(outcome_name, evolved_h)

    log_lik_event = torch.log(torch.clamp(hazard_rate_at_t, min=1e-8)) - cum_hazard_outcome
    log_lik_censored = -cum_hazard_outcome

    event_indicator = event_indicator.float()
    log_lik = event_indicator * log_lik_event + (1.0 - event_indicator) * log_lik_censored
    return -log_lik.mean()


def survlatent_ode_survival_loss(
    model,
    landmark_state: torch.Tensor,
    batch: dict[str, torch.Tensor],
    outcome_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """batch must contain, per outcome, {name}_event_indicator and {name}_target_days (see
    compass.training.batching.collate_examples).

    Returns {"survival_nll": total, "total": total} -- total IS the survival NLL, sum of
    the three outcomes' independent, inverse-frequency-weighted contributions. Nothing else
    is added, same as TraCeR.
    """
    nll_terms = {
        name: _outcome_nll(
            model,
            landmark_state,
            batch[f"{name}_event_indicator"],
            batch[f"{name}_target_days"],
            name,
        )
        for name in OUTCOME_NAMES
    }
    total = sum(outcome_weights[name] * nll_terms[name] for name in OUTCOME_NAMES)
    return {"survival_nll": total, "total": total}

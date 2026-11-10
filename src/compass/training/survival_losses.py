"""Discrete-time survival loss primitives shared across candidate architectures.

discrete_time_survival_nll() was originally written in
compass.models.dynamic_deephit.losses, but it is fully outcome-agnostic (operates on one
hazard sequence + targets at a time, no Dynamic-DeepHit-specific assumptions) and TRACER
needs the exact same computation, using the same per-outcome censoring rule. Moved here
rather than duplicated; compass.models.dynamic_deephit.losses re-exports it so existing
imports keep working unchanged.
"""

import torch


def discrete_time_survival_nll(
    hazard: torch.Tensor,
    event_indicator: torch.Tensor,
    event_bin: torch.Tensor,
    censoring_bin: torch.Tensor,
) -> torch.Tensor:
    """Discrete-time survival negative log-likelihood for ONE outcome's independent hazard
    sequence, averaged over the batch.

    - event_indicator=1 (event occurred): likelihood = hazard[event_bin] *
      prod_{k<event_bin}(1-hazard[k]).
    - event_indicator=0 (censored): likelihood = prod_{k<=censoring_bin}(1-hazard[k]).

    Callers must supply event_bin/censoring_bin already adjusted for this project's
    per-outcome censoring rule (see compass.training.batching.collate_examples): mortality
    is censored at the administrative cutoff only; malo/cvd are censored at
    min(administrative cutoff, mortality event time) if mortality occurred, since death
    stops further risk-set exposure for the non-fatal outcomes -- independent of the OTHER
    non-fatal outcome's status. This function itself is outcome-agnostic and
    architecture-agnostic.

    Beyond the last bin: an event_bin past the last bin (>= num_time_bins) is NOT clipped
    into the last bin as an event -- it is treated as censored at the end of the last bin
    (survived all bins). A censoring_bin past the last bin is likewise censored at the last
    bin. compass.training.batching already produces targets in range; this is defensive.

    hazard: (batch, num_time_bins). event_indicator/event_bin/censoring_bin: (batch,).
    """
    eps = 1e-8
    num_time_bins = hazard.shape[1]

    event_beyond_window = event_bin > num_time_bins - 1
    event_indicator = event_indicator.float() * (~event_beyond_window).float()
    censoring_bin = torch.where(event_beyond_window, torch.full_like(censoring_bin, num_time_bins - 1), censoring_bin)

    log_survival = torch.log(torch.clamp(1.0 - hazard, min=eps))
    cumulative_log_survival = torch.cumsum(log_survival, dim=1)  # log P(survive through bin t)

    log_survival_strictly_before = torch.cat(
        [torch.zeros(hazard.shape[0], 1, device=hazard.device), cumulative_log_survival[:, :-1]], dim=1
    )

    event_bin_clamped = event_bin.clamp(0, num_time_bins - 1)
    log_hazard_at_event = torch.log(torch.clamp(hazard.gather(1, event_bin_clamped.unsqueeze(1)).squeeze(1), min=eps))
    log_survival_before_event = log_survival_strictly_before.gather(1, event_bin_clamped.unsqueeze(1)).squeeze(1)
    log_lik_event = log_hazard_at_event + log_survival_before_event

    censoring_bin_clamped = censoring_bin.clamp(0, num_time_bins - 1)
    log_lik_censored = cumulative_log_survival.gather(1, censoring_bin_clamped.unsqueeze(1)).squeeze(1)

    log_lik = event_indicator * log_lik_event + (1.0 - event_indicator) * log_lik_censored

    return -log_lik.mean()

"""Dynamic-DeepHit loss components, corrected for three INDEPENDENT per-outcome hazard
heads (see compass.models.base for why: malo/cvd can co-occur, so a single joint softmax
across causes is not valid here).

Each function operates on ONE outcome's hazard sequence/targets at a time. The model's
compute_loss() calls discrete_time_survival_nll() and ranking_loss() once per outcome and
sums the results; auxiliary_prediction_loss() is shared (not outcome-specific).
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
    non-fatal outcome's status. This function itself is outcome-agnostic.

    hazard: (batch, num_time_bins). event_indicator/event_bin/censoring_bin: (batch,).
    """
    eps = 1e-8
    num_time_bins = hazard.shape[1]

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

    event_indicator = event_indicator.float()
    log_lik = event_indicator * log_lik_event + (1.0 - event_indicator) * log_lik_censored

    return -log_lik.mean()


def ranking_loss(
    risk_scores: torch.Tensor,
    event_indicator: torch.Tensor,
    observed_bin: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Pairwise concordance-style ranking loss for ONE outcome (call once per outcome,
    sum the three results in the caller).

    For every ordered pair (i, j) where patient i had the event (event_indicator=1) at a
    strictly earlier observed_bin than patient j's own observed_bin (event or censoring,
    whichever applies to j), penalizes risk_score(i) <= risk_score(j) via a smooth
    exp((score_j - score_i) / sigma) term -- encourages patients whose event happened
    earlier to be scored as higher risk than patients who were still event-free (or
    censored) at that time.

    risk_scores/event_indicator/observed_bin: (batch,).
    """
    batch_size = risk_scores.shape[0]
    if batch_size < 2:
        return torch.zeros((), device=risk_scores.device)

    time_i = observed_bin.unsqueeze(1).expand(batch_size, batch_size)
    time_j = observed_bin.unsqueeze(0).expand(batch_size, batch_size)
    event_i = event_indicator.unsqueeze(1).expand(batch_size, batch_size)

    comparable = (event_i == 1) & (time_i < time_j)

    if comparable.sum() == 0:
        return torch.zeros((), device=risk_scores.device)

    score_i = risk_scores.unsqueeze(1).expand(batch_size, batch_size)
    score_j = risk_scores.unsqueeze(0).expand(batch_size, batch_size)
    pairwise_loss = torch.exp((score_j - score_i) / sigma)

    return pairwise_loss[comparable].mean()


def auxiliary_prediction_loss(
    predicted_next: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    valid_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Shared (not outcome-specific) auxiliary next-visit prediction loss: MSE between the
    prediction made at each timestep t and the actual values at t+1, masked to timesteps
    where t+1 is a real (non-padding) visit AND the target feature was actually observed
    (not zero-filled missingness).

    predicted_next/values/mask: (batch, seq_len, num_features).
    valid_timesteps: (batch, seq_len) bool, which timesteps are real (not padding).
    """
    pred = predicted_next[:, :-1, :]
    target = values[:, 1:, :]
    target_mask = mask[:, 1:, :].float()
    target_valid = valid_timesteps[:, 1:].unsqueeze(-1).float()

    combined_mask = target_mask * target_valid
    squared_error = (pred - target) ** 2 * combined_mask

    denom = combined_mask.sum()
    if denom == 0:
        return torch.zeros((), device=values.device)
    return squared_error.sum() / denom

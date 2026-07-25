"""TRACER loss: sum of three independent per-outcome discrete-time hazard NLLs, inverse-
frequency weighted. Unlike Dynamic-DeepHit, TRACER has NO ranking loss and NO auxiliary
next-visit prediction loss -- just direct discrete-time hazard negative log-likelihood per
outcome, using the same per-outcome censoring rule (mortality censors malo/cvd; malo and
cvd do not censor each other).
"""

import torch

from compass.models.base import OUTCOME_NAMES
from compass.training.survival_losses import discrete_time_survival_nll


def tracer_survival_loss(
    hazards: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    outcome_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """hazards: {"mortality": ..., "malo": ..., "cvd": ...}, each (batch, num_time_bins).
    batch must contain {name}_event_indicator / {name}_event_bin / {name}_censoring_bin per
    outcome, from compass.training.batching.collate_examples (the same targets Dynamic-
    DeepHit uses, so the two architectures are compared on identical censoring semantics).

    Returns {"survival_nll": total, "total": total} -- total loss IS the survival NLL, sum
    of the three outcomes' independent, inverse-frequency-weighted contributions. Nothing
    else is added.
    """
    nll_terms = {
        name: discrete_time_survival_nll(
            hazards[name],
            batch[f"{name}_event_indicator"],
            batch[f"{name}_event_bin"],
            batch[f"{name}_censoring_bin"],
        )
        for name in OUTCOME_NAMES
    }
    total = sum(outcome_weights[name] * nll_terms[name] for name in OUTCOME_NAMES)
    return {"survival_nll": total, "total": total}

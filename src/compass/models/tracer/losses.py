"""TRACER loss: unweighted sum of three independent per-outcome discrete-time hazard NLLs
(no class-imbalance weighting -- dropped 2026-11-11). Unlike Dynamic-DeepHit, TRACER has NO
ranking loss and NO auxiliary next-visit prediction loss -- just direct discrete-time hazard
negative log-likelihood per outcome, using the same per-outcome targets
(compass.training.targets.build_targets), each averaged over that outcome's at-risk examples
only.
"""

import torch

from compass.models.base import OUTCOME_NAMES
from compass.training.survival_losses import discrete_time_survival_nll


def tracer_survival_loss(
    hazards: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """hazards: {"mortality": ..., "malo": ..., "cvd": ...}, each (batch, num_time_bins).
    batch must contain {name}_event_indicator / {name}_event_bin / {name}_censoring_bin /
    {name}_at_risk per outcome, from compass.training.batching.collate_examples (the same targets Dynamic-
    DeepHit uses, so the two architectures are compared on identical censoring semantics).

    Returns {"survival_nll", "total", "nll_mortality", "nll_malo", "nll_cvd"} -- total loss IS
    the survival NLL, the unweighted sum of the three outcomes' independent contributions (also
    returned per outcome, for logging). Nothing else is added.
    """
    nll_terms = {
        name: discrete_time_survival_nll(
            hazards[name],
            batch[f"{name}_event_indicator"],
            batch[f"{name}_event_bin"],
            batch[f"{name}_censoring_bin"],
            batch[f"{name}_at_risk"],
        )
        for name in OUTCOME_NAMES
    }
    total = sum(nll_terms[name] for name in OUTCOME_NAMES)  # unweighted sum (no class-imbalance weighting)
    return {"survival_nll": total, "total": total, **{f"nll_{name}": nll_terms[name] for name in OUTCOME_NAMES}}

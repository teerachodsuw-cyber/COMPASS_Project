"""Shared interface all COMPASS candidate architectures (Dynamic-DeepHit, TRACER,
SurvLatent ODE) implement, so training/evaluation/comparison code works identically across
all three without duplication. This contract is meant to be stable.

Each outcome (mortality, malo, cvd) is modeled as an INDEPENDENT discrete-time hazard
sequence -- NOT a shared softmax competing across causes. The classic Dynamic-DeepHit joint
softmax over (cause x time-bin) assumes mutually-exclusive competing risks, which does not
hold here: malo and cvd can co-occur in the same patient (only mortality is a true terminal
event that censors both). See compass.models.dynamic_deephit.losses for the corresponding
per-outcome censoring rule used at training time.

Input contract: values/mask are the FULL combined per-visit feature vector already produced
by compass.data.preprocessing.run_preprocessing (continuous + medication flags +
demographics folded into one (seq_len, num_features) tensor per patient, per
build_feature_names()) -- there is no separate medication/demographic input; preprocessing
already merges everything into one per-visit vector, so the model interface doesn't need to.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

OUTCOME_NAMES = ("mortality", "malo", "cvd")


def years_to_bin_index(years: torch.Tensor, bin_width_years: float, num_time_bins: int) -> torch.Tensor:
    """Map time-from-landmark (years) to a 0-indexed discrete-time bin, where bin i covers
    (i*bin_width_years, (i+1)*bin_width_years]. Shared by CompassSurvivalModel.years_to_bin_index
    and the loss-target construction in compass.training.batching, so both use identical
    binning. Clamped to [0, num_time_bins-1].
    """
    bin_index = torch.ceil(years / bin_width_years).long() - 1
    return bin_index.clamp(0, num_time_bins - 1)


class CompassSurvivalModel(nn.Module, ABC):
    """Common interface for all COMPASS candidate architectures.

    Input contract (all models must accept):
    - values: (batch, seq_len, num_features) float tensor, scaled + zero-filled
    - mask: (batch, seq_len, num_features) float tensor, 1=observed 0=missing
    - seq_lengths: (batch,) int tensor, actual (unpadded) sequence length per patient
    - elapsed_days: (batch, seq_len) float tensor, OPTIONAL (default None). Days since the
      previous visit (0 for each patient's first visit), from
      compass.training.batching.collate_examples. Added for TRACER, whose factorized
      temporal attention and time-decay missingness gate need real elapsed time between
      irregularly-spaced visits, not just sequence position. Models that don't need it
      (Dynamic-DeepHit) accept and ignore it -- this keeps the shared signature stable
      across all three architectures rather than giving TRACER a different one.

    Output contract (all models must implement):
    - forward(...) -> dict with exactly three keys "mortality", "malo", "cvd", each
      (batch, num_time_bins): that outcome's independent discrete-time HAZARD (probability
      of the event in that bin, given survival to its start) via sigmoid, not cumulative,
      not softmax-normalized across outcomes. Nothing forces the three tensors to relate to
      each other -- a patient can have high hazard on both malo and cvd simultaneously.
    - compute_loss(batch) -> dict[str, Tensor]: named loss components plus 'total'.

    Concrete (non-abstract, shared) helpers:
    - hazards_to_cif(): discrete-time hazard -> cumulative incidence, identical formula
      regardless of architecture.
    - predict_risk(): the method evaluation/comparison code actually calls -- wraps
      forward() + hazards_to_cif() into (batch, 3, len(horizons)) CIF probabilities.
    - years_to_bin_index(): default yearly-bin mapping from requested horizon (years) to
      this model's internal bin index; subclasses set self.bin_width_years/num_time_bins in
      __init__, or override this method directly for a non-default/non-discrete scheme.
    """

    @abstractmethod
    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def hazards_to_cif(hazards: torch.Tensor, horizon_bins: list[int]) -> torch.Tensor:
        """Convert a discrete-time hazard sequence (batch, num_time_bins) into cumulative
        incidence at each requested (0-indexed, inclusive) bin index, via the standard
        product-of-survival formula:
            survival(t) = prod_{k=0..t} (1 - hazard_k)
            CIF(t) = 1 - survival(t)
        Returns (batch, len(horizon_bins)).
        """
        survival_per_bin = 1.0 - hazards
        cumulative_survival = torch.cumprod(survival_per_bin, dim=1)
        cif_all_bins = 1.0 - cumulative_survival

        horizon_idx = torch.tensor(horizon_bins, dtype=torch.long, device=hazards.device)
        return cif_all_bins.index_select(dim=1, index=horizon_idx)

    def years_to_bin_index(self, horizon_years: int) -> int:
        return int(years_to_bin_index(torch.tensor(float(horizon_years)), self.bin_width_years, self.num_time_bins).item())

    def predict_risk(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        horizons: list[int],
        elapsed_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward() + hazards_to_cif() for each outcome, stacked into
        (batch, 3, len(horizons)) -- the method evaluation/comparison code actually calls.
        """
        hazards_by_outcome = self.forward(values, mask, seq_lengths, elapsed_days)
        horizon_bins = [self.years_to_bin_index(h) for h in horizons]
        cifs = [self.hazards_to_cif(hazards_by_outcome[name], horizon_bins) for name in OUTCOME_NAMES]
        return torch.stack(cifs, dim=1)


def quantile_time_grid(event_times_years: np.ndarray, num_bins: int = 20) -> np.ndarray:
    """Quantile-based discrete-time bin boundaries from observed event/censoring times.

    PLACEHOLDER matching the locked "quantile_grid" discretization method (see
    configs/training.yaml: horizons.discretization_method). Exact cutpoints are deferred to
    Stage 1 real data. NOT currently used by DynamicDeepHitModel, which uses simple yearly
    bins instead (see model.py) -- kept here as a documented, available utility for later
    use, since the locked config still names quantile_grid as the intended method.

    Returns boundaries such that bin i covers (boundaries[i], boundaries[i+1]], boundaries[0]
    is always 0.0. Duplicate quantiles (common with few distinct/rare-event times) are
    collapsed via np.unique, so the returned array may have fewer than num_bins+1 entries --
    callers must use len(boundaries) - 1 as the actual bin count, not assume num_bins.
    """
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    boundaries = np.quantile(event_times_years, quantiles)
    boundaries[0] = 0.0
    return np.unique(boundaries)

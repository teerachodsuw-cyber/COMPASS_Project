"""Dynamic-DeepHit: GRU + attention shared representation, three independent per-outcome
discrete-time hazard heads (sigmoid, not a shared softmax across causes -- see
compass.models.base for why).

Uses simple yearly discrete-time bins by default (bin_width_years=1.0, num_time_bins=30 ->
follow-up up to 30 years, matching the project's longest reporting horizon). This is a
PLACEHOLDER discretization; exact cutpoints are deferred to Stage 1 per the project's locked
"quantile_grid" methodology (see compass.models.base.quantile_time_grid).
"""

import torch
from torch import nn

from compass.models.base import OUTCOME_NAMES, CompassSurvivalModel
from compass.models.dynamic_deephit.losses import (
    auxiliary_prediction_loss,
    discrete_time_survival_nll,
    ranking_loss,
)


class DynamicDeepHitModel(CompassSurvivalModel):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_gru_layers: int = 2,
        cause_head_hidden_dim: int = 32,
        num_time_bins: int = 30,
        bin_width_years: float = 1.0,
        ranking_weight: float = 0.1,
        auxiliary_weight: float = 0.1,
        outcome_weights: dict[str, float] | None = None,
    ):
        """outcome_weights: optional per-outcome multiplier on the survival log-likelihood
        term (inverse-frequency weighting, per configs/training.yaml's locked
        class_imbalance_handling method). Callers should compute these from the TRAINING
        split's actual outcome rates -- this class stays agnostic to how they're derived.
        Defaults to equal weight (1.0) per outcome.
        """
        super().__init__()
        self.num_features = num_features
        self.num_time_bins = num_time_bins
        self.bin_width_years = bin_width_years
        self.ranking_weight = ranking_weight
        self.auxiliary_weight = auxiliary_weight
        self.outcome_weights = outcome_weights or {name: 1.0 for name in OUTCOME_NAMES}

        # values + mask concatenated along the feature dim, so the model sees what's
        # missing, not just zero-filled values.
        self.gru = nn.GRU(
            input_size=num_features * 2,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Bahdanau-style additive attention over GRU hidden states -> one context vector.
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.cause_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, cause_head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cause_head_hidden_dim, num_time_bins),
                )
                for name in OUTCOME_NAMES
            }
        )

        # Auxiliary next-visit feature prediction head (shared across outcomes).
        self.auxiliary_head = nn.Linear(hidden_dim, num_features)

    def _encode(self, values: torch.Tensor, mask: torch.Tensor, seq_lengths: torch.Tensor):
        x = torch.cat([values, mask], dim=-1)  # (batch, seq_len, 2*num_features)

        packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        # gru_out: (batch, seq_len, hidden_dim)

        seq_range = torch.arange(gru_out.size(1), device=x.device).unsqueeze(0)
        valid = seq_range < seq_lengths.unsqueeze(1)  # (batch, seq_len)

        scores = self.attention_score(gru_out).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (weights * gru_out).sum(dim=1)  # (batch, hidden_dim)

        return context, gru_out, valid

    def _hazards_from_context(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: torch.sigmoid(head(context)) for name, head in self.cause_heads.items()}

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # elapsed_days accepted for interface consistency with base.py but unused here --
        # the GRU's sequence order already captures temporal structure for this
        # architecture; see base.py for why TRACER needs it and Dynamic-DeepHit doesn't.
        context, _, _ = self._encode(values, mask, seq_lengths)
        return self._hazards_from_context(context)

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        values, mask, seq_lengths = batch["values"], batch["mask"], batch["seq_lengths"]
        context, gru_out, valid = self._encode(values, mask, seq_lengths)
        hazards = self._hazards_from_context(context)

        nll_terms = {}
        rank_terms = {}
        for name in OUTCOME_NAMES:
            event_indicator = batch[f"{name}_event_indicator"]
            event_bin = batch[f"{name}_event_bin"]
            censoring_bin = batch[f"{name}_censoring_bin"]

            nll_terms[name] = discrete_time_survival_nll(hazards[name], event_indicator, event_bin, censoring_bin)

            observed_bin = torch.where(event_indicator.bool(), event_bin, censoring_bin)
            risk_score = self.hazards_to_cif(hazards[name], [self.num_time_bins - 1])[:, 0]
            rank_terms[name] = ranking_loss(risk_score, event_indicator, observed_bin)

        survival_nll = sum(self.outcome_weights[name] * nll_terms[name] for name in OUTCOME_NAMES)
        total_ranking = sum(rank_terms.values())

        predicted_next = self.auxiliary_head(gru_out)
        aux_loss = auxiliary_prediction_loss(predicted_next, values, mask, valid)

        total = survival_nll + self.ranking_weight * total_ranking + self.auxiliary_weight * aux_loss

        return {
            "survival_nll": survival_nll,
            "ranking_loss": total_ranking,
            "auxiliary_loss": aux_loss,
            "total": total,
        }

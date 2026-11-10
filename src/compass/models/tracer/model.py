"""TRACER: factorized-attention transformer with a native time-decay missingness gate,
three independent per-outcome discrete-time hazard heads (same output contract as
Dynamic-DeepHit -- see compass.models.base).

Two things make this architecturally different from Dynamic-DeepHit, not just "a different
backbone":

1. Missingness handling: per this project's locked decision, each candidate architecture
   keeps its OWN native missingness mechanism, to keep model comparison a fair fight between
   complete systems rather than standardizing away real architectural differences.
   Dynamic-DeepHit uses mask+zero-fill (concat value and binary mask). TRACER instead uses a
   continuous time-decay gate: weight = exp(-time_since_last_observed / tau), tau a
   learnable per-feature decay rate. time_since_last_observed is derived here, per feature,
   per timestep, from the mask sequence + elapsed_days (see compute_decay_gate()) --
   elapsed_days itself comes from compass.training.batching.collate_examples (shared,
   derived from real visit dates), but the per-FEATURE staleness tracking is TRACER-specific
   and lives entirely in this file.

2. Attention structure: factorized, not flattened. Covariate (feature-axis) self-attention
   runs first, independently per timestep, across the num_features "tokens" at that visit.
   The result is pooled across features into one per-timestep vector, then a causal-masked
   temporal transformer attends across visits. This avoids the quadratic blowup of a single
   attention over (seq_len * num_features) flattened tokens and matches TRACER's published
   design, which already uses independent per-outcome hazards natively (no joint-softmax
   correction needed here, unlike Dynamic-DeepHit).
"""

import torch
import torch.nn.functional as F
from torch import nn

from compass.models.base import OUTCOME_NAMES, CompassSurvivalModel
from compass.models.tracer.losses import tracer_survival_loss


class TracerModel(CompassSurvivalModel):
    def __init__(
        self,
        num_features: int,
        feature_embed_dim: int = 16,
        covariate_attention_heads: int = 2,
        hidden_dim: int = 64,
        temporal_attention_heads: int = 4,
        num_transformer_layers: int = 2,
        cause_head_hidden_dim: int = 32,
        dropout: float = 0.1,
        num_time_bins: int = 13,
        bin_width_years: float = 1.0,
        max_staleness_days: float = 3650.0,
        outcome_weights: dict[str, float] | None = None,
    ):
        """outcome_weights: optional per-outcome multiplier on the survival log-likelihood
        (inverse-frequency weighting, same convention as DynamicDeepHitModel -- see
        configs/training.yaml's locked class_imbalance_handling method). Defaults to equal
        weight (1.0) per outcome.
        """
        super().__init__()
        self.num_features = num_features
        self.num_time_bins = num_time_bins
        self.bin_width_years = bin_width_years
        self.max_staleness_days = max_staleness_days
        self.outcome_weights = outcome_weights or {name: 1.0 for name in OUTCOME_NAMES}

        # Learnable per-feature decay rate (softplus-parameterized to stay positive).
        self.decay_rate_raw = nn.Parameter(torch.zeros(num_features))

        # [gated_value, gate_weight] per feature per timestep -> per-feature embedding.
        self.feature_embed = nn.Linear(2, feature_embed_dim)

        self.covariate_attn = nn.MultiheadAttention(
            embed_dim=feature_embed_dim, num_heads=covariate_attention_heads, dropout=dropout, batch_first=True
        )
        self.covariate_norm = nn.LayerNorm(feature_embed_dim)

        self.covariate_to_hidden = nn.Linear(feature_embed_dim, hidden_dim)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_transformer_layers)

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

    def _compute_time_since_last_observed(self, mask: torch.Tensor, elapsed_days: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, num_features) days since each feature was last actually
        observed. mask==1 timesteps get 0. A feature not yet observed by timestep t
        (including t=0 with mask=0) accumulates from a "never observed" sentinel
        (max_staleness_days), capped there for numerical stability -- exp(-large/tau)
        underflows safely to ~0 rather than over/underflowing.
        """
        batch, seq_len, num_features = mask.shape
        time_since = torch.zeros(batch, seq_len, num_features, device=mask.device)
        running = torch.full((batch, num_features), self.max_staleness_days, device=mask.device)
        for t in range(seq_len):
            gap = elapsed_days[:, t].unsqueeze(-1)  # (batch, 1); 0 for each patient's first visit
            observed_now = mask[:, t, :].bool()
            running = torch.where(observed_now, torch.zeros_like(running), running + gap)
            running = running.clamp(max=self.max_staleness_days)
            time_since[:, t, :] = running
        return time_since

    def compute_decay_gate(self, mask: torch.Tensor, elapsed_days: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, num_features) continuous decay weight in (0, 1] -- TRACER's
        native missingness signal, replacing Dynamic-DeepHit's binary mask+zero-fill.
        """
        time_since = self._compute_time_since_last_observed(mask, elapsed_days)
        tau = F.softplus(self.decay_rate_raw) + 1e-3  # (num_features,), always positive
        return torch.exp(-time_since / tau)

    def _covariate_attention(self, feature_embed: torch.Tensor) -> torch.Tensor:
        """Self-attention across features, independently per (batch, timestep).
        feature_embed: (batch, seq_len, num_features, feature_embed_dim).
        """
        batch, seq_len, num_features, dim = feature_embed.shape
        x = feature_embed.reshape(batch * seq_len, num_features, dim)
        attn_out, _ = self.covariate_attn(x, x, x)
        x = self.covariate_norm(x + attn_out)
        return x.reshape(batch, seq_len, num_features, dim)

    def _temporal_attention(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """Causal-masked self-attention across visits. x: (batch, seq_len, hidden_dim)."""
        seq_len = x.shape[1]
        # Boolean masks throughout (True = disallowed/ignore) -- torch warns/deprecates
        # mixing a float additive mask with a bool key_padding_mask.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        seq_range = torch.arange(seq_len, device=x.device).unsqueeze(0)
        key_padding_mask = seq_range >= seq_lengths.unsqueeze(1)  # True = ignore (padding)
        return self.temporal_encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)

    def _encode(self, values: torch.Tensor, mask: torch.Tensor, seq_lengths: torch.Tensor, elapsed_days: torch.Tensor):
        gate_weight = self.compute_decay_gate(mask, elapsed_days)
        gated_value = values * gate_weight

        embed_input = torch.stack([gated_value, gate_weight], dim=-1)  # (batch, seq_len, num_features, 2)
        feature_embed = self.feature_embed(embed_input)

        covariate_out = self._covariate_attention(feature_embed)
        pooled = covariate_out.mean(dim=2)  # pool across features -> (batch, seq_len, feature_embed_dim)
        projected = self.covariate_to_hidden(pooled)  # (batch, seq_len, hidden_dim)

        temporal_out = self._temporal_attention(projected, seq_lengths)

        last_idx = (seq_lengths - 1).clamp(min=0)
        context = temporal_out[torch.arange(temporal_out.shape[0], device=temporal_out.device), last_idx]
        return context, gate_weight

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if elapsed_days is None:
            elapsed_days = torch.zeros(values.shape[0], values.shape[1], device=values.device)
        context, _ = self._encode(values, mask, seq_lengths, elapsed_days)
        return {name: torch.sigmoid(head(context)) for name, head in self.cause_heads.items()}

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        values, mask, seq_lengths = batch["values"], batch["mask"], batch["seq_lengths"]
        elapsed_days = batch["elapsed_days"]
        context, _ = self._encode(values, mask, seq_lengths, elapsed_days)
        hazards = {name: torch.sigmoid(head(context)) for name, head in self.cause_heads.items()}
        return tracer_survival_loss(hazards, batch, self.outcome_weights)

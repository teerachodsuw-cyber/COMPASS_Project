"""SurvLatent ODE: ODE-RNN encoder (latent state evolves continuously via a neural ODE
between visits, updated by a GRU cell at each observation), three independent per-outcome
continuous-time hazard decoders. Genuinely continuous internally -- discrete bins only exist
at the base.py interface boundary (hazards_at_bins()), for comparison with Dynamic-DeepHit
and TraCeR.

Backprop choice: DIRECT autodiff through the solver via `torchdiffeq.odeint`, NOT the
adjoint method (`odeint_adjoint`). Sequences here are short (<=~30 visits, small synthetic
batches), so the adjoint method's memory savings aren't needed, and direct backprop gives
exact gradients through the actual solver operations rather than the adjoint's separately-
solved (and therefore only approximately consistent) backward-time ODE -- more reliable for
validating a research proof-of-concept where correctness matters more than scaling to very
long sequences or huge batches.

Batched heterogeneous time spans: torchdiffeq solves for one SHARED set of time points
across a whole batch, but different patients have different real elapsed_days between
visits. Rather than looping per-patient (slow) or forcing a fixed step, this exploits that
the latent ODE is autonomous (dh/dt = f(h), no explicit t dependence): evolving h over real
time T is equivalent to evolving over a normalized tau in [0,1] with the vector field scaled
by T. So every segment is solved over the SAME tau grid [0,1], with each batch element's own
elapsed_days baked in as a per-element multiplicative scale on the derivative -- one shared
odeint call handles the whole batch's differing real-time gaps at once.

Missingness within a visit: standard mask+zero-fill (concat values and mask), same as
Dynamic-DeepHit -- that's a separate concern from what's native to this architecture. What's
native here is feeding the solver the REAL elapsed time between visits (reused from
compass.training.batching.collate_examples, not rebuilt) so the latent state evolves across
the true calendar gap, not a fixed per-step assumption.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint

from compass.models.base import OUTCOME_NAMES, CompassSurvivalModel
from compass.models.survlatent_ode.losses import survlatent_ode_survival_loss

_TAU_SPAN = torch.tensor([0.0, 1.0])
_YEAR_DAYS = 365.25


class SurvLatentODEModel(CompassSurvivalModel):
    def __init__(
        self,
        num_features: int,
        latent_dim: int = 16,
        ode_hidden_dim: int = 32,
        cause_head_hidden_dim: int = 16,
        num_time_bins: int = 13,
        bin_width_years: float = 1.0,
        solver_method: str = "dopri5",
        outcome_weights: dict[str, float] | None = None,
    ):
        """outcome_weights: optional per-outcome multiplier on the survival log-likelihood
        (inverse-frequency weighting, same convention as the other two models). Defaults to
        equal weight (1.0) per outcome.
        """
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_time_bins = num_time_bins
        self.bin_width_years = bin_width_years
        self.solver_method = solver_method
        self.outcome_weights = outcome_weights or {name: 1.0 for name in OUTCOME_NAMES}

        # dh/dt = f(h) -- autonomous latent dynamics. Final tanh bounds the (unscaled, per-
        # tau) derivative to [-1,1] elementwise: an unconstrained MLP vector field has no
        # inherent stability guarantee and, integrated over real visit gaps of hundreds of
        # days, blows up exponentially (empirically produced latent states in the
        # thousands from a purely random, untrained network). Bounding the derivative
        # caps growth at LINEAR in elapsed time instead, which is what makes integrating
        # over realistic irregular calendar gaps numerically tractable at all.
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        # Small init on the final layer so dynamics start close to dh/dt ~ 0 (near-identity
        # map) rather than immediately saturating the tanh bound -- standard Neural ODE
        # initialization practice for training stability.
        nn.init.uniform_(self.ode_func[-2].weight, -0.01, 0.01)
        nn.init.zeros_(self.ode_func[-2].bias)

        # GRU-style update at each observation: combine the ODE-evolved state with the new
        # visit's (values, mask) -- same mask+zero-fill convention as Dynamic-DeepHit.
        self.gru_cell = nn.GRUCell(input_size=num_features * 2, hidden_size=latent_dim)

        self.initial_latent = nn.Parameter(torch.zeros(latent_dim))

        self.cause_decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim, cause_head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cause_head_hidden_dim, 1),
                )
                for name in OUTCOME_NAMES
            }
        )
        # Hazard rate is per-YEAR internally (see _YEAR_DAYS conversion below -- elapsed_days
        # /target_days stay in days at the public interface, matching the rest of the
        # project's convention, e.g. TraCeR). Softplus(0) ~ 0.69/yr integrated over the 13yr
        # window would still give a cumulative hazard around 9 at random init; initializing
        # each decoder's output bias negative starts the rate near a realistic per-year scale
        # (e.g. bias=-5 -> rate ~0.007/yr -> ~0.09 cumulative hazard over the 13yr window) instead.
        for decoder in self.cause_decoders.values():
            nn.init.constant_(decoder[-1].bias, -5.0)

    def _hazard_rate(self, name: str, h: torch.Tensor) -> torch.Tensor:
        """Instantaneous hazard/intensity for outcome `name` at latent state h: (batch,).
        Positive (softplus), per YEAR. A continuous-time RATE, not bounded to [0,1] like a
        discrete per-bin probability (see hazards_at_bins() for that conversion).
        """
        return F.softplus(self.cause_decoders[name](h)).squeeze(-1)

    def _evolve(self, h: torch.Tensor, real_days: torch.Tensor) -> torch.Tensor:
        """Evolve latent state h by real_days (batch,), one shared odeint call for the
        whole batch even though each element's real elapsed time differs (tau-in-[0,1]
        reparametrization, see module docstring). Internally scales by YEARS, not raw days
        -- with a bounded (tanh) derivative, scaling by raw days (up to ~4750 for the 13yr
        censoring window) still let per-segment drift reach the hundreds for realistic
        multi-hundred-day visit gaps; years shrinks that ~365x to a numerically tractable
        range without changing the public days-based interface.
        """
        real_years = real_days / _YEAR_DAYS

        def scaled_func(tau, y):
            return self.ode_func(y) * real_years.unsqueeze(-1)

        solution = odeint(scaled_func, h, _TAU_SPAN.to(h.device), method=self.solver_method)
        return solution[-1]

    def _evolve_with_hazard_integral(
        self, h: torch.Tensor, real_days: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Same as _evolve(), but also integrates each outcome's hazard rate over the
        interval via an augmented ODE state -- using the solver itself to compute the
        cumulative-hazard integral, rather than a separate numerical integration pass.
        Returns (evolved_latent_state, {outcome: cumulative_hazard_over_interval}).
        """
        real_years = real_days / _YEAR_DAYS
        batch = h.shape[0]
        z0 = torch.cat([h, torch.zeros(batch, len(OUTCOME_NAMES), device=h.device)], dim=-1)

        def augmented_func(tau, z):
            hz = z[:, : self.latent_dim]
            dh = self.ode_func(hz) * real_years.unsqueeze(-1)
            hazard_rates = torch.stack([self._hazard_rate(name, hz) for name in OUTCOME_NAMES], dim=-1)
            dcum = hazard_rates * real_years.unsqueeze(-1)
            return torch.cat([dh, dcum], dim=-1)

        solution = odeint(augmented_func, z0, _TAU_SPAN.to(h.device), method=self.solver_method)
        z_final = solution[-1]
        h_final = z_final[:, : self.latent_dim]
        cum_hazard = {name: z_final[:, self.latent_dim + i] for i, name in enumerate(OUTCOME_NAMES)}
        return h_final, cum_hazard

    def _integrate_hazard_to(
        self, h: torch.Tensor, target_days: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evolve h to each patient's own target_days (batch,) -- which can range from a
        few days to 13 years (capped at the end of the last bin) within one batch (event/censoring times vary enormously per
        patient) -- and return the cumulative hazard integrated over [0, target_days].

        A single _evolve_with_hazard_integral() call scales the vector field by real_days
        directly; with such a wide range of real_days in one batch, dopri5's SHARED
        adaptive step schedule cannot resolve all elements at once, and the smallest-
        magnitude/most extreme elements can blow up numerically (empirically produced
        cumulative hazards in the millions at random initialization, versus the expected
        O(1-10) range). Fixed by walking forward in UNIFORM sub-steps (bounded to
        bin_width_years, the same granularity hazards_at_bins() already uses successfully)
        instead of one giant heterogeneous-scale jump -- each step's real_days range is
        [0, bin_width_days], not [0, ~4750], which is numerically well-behaved. Patients
        who reach their own target_days early get step_days=0 (a true no-op: dh/dtau =
        0 * f(h) = 0) for remaining steps, so this naturally handles per-patient stopping.
        """
        step_days = self.bin_width_years * 365.25
        n_steps = max(1, int(torch.ceil(target_days.max() / step_days).item()))

        batch = h.shape[0]
        device = h.device
        remaining = target_days.clone()
        cum_hazard = {name: torch.zeros(batch, device=device) for name in OUTCOME_NAMES}
        h_at_target = torch.zeros_like(h)
        reached = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(n_steps):
            this_step = torch.clamp(remaining, min=0.0, max=step_days)
            h_new, cum_hazard_step = self._evolve_with_hazard_integral(h, this_step)
            for name in OUTCOME_NAMES:
                cum_hazard[name] = cum_hazard[name] + cum_hazard_step[name]
            h = h_new
            remaining = remaining - this_step
            newly_reached = (~reached) & (remaining <= 1e-6)
            h_at_target = torch.where(newly_reached.unsqueeze(-1), h, h_at_target)
            reached = reached | newly_reached

        return h_at_target, cum_hazard

    def encode_trajectory(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor,
    ) -> torch.Tensor:
        """Run the ODE-RNN across all observed visits, returning the latent state AT the
        last observed visit for each patient (batch, latent_dim) -- the landmark state,
        since landmark_date is itself the last row of input_visits by construction.
        """
        batch, seq_len, _ = values.shape
        h = self.initial_latent.unsqueeze(0).expand(batch, -1).contiguous()
        x = torch.cat([values, mask], dim=-1)  # (batch, seq_len, 2*num_features)

        states = []
        for t in range(seq_len):
            if t > 0:
                h = self._evolve(h, elapsed_days[:, t])
            h = self.gru_cell(x[:, t, :], h)
            states.append(h)

        stacked = torch.stack(states, dim=1)  # (batch, seq_len, latent_dim)
        last_idx = (seq_lengths - 1).clamp(min=0)
        return stacked[torch.arange(batch, device=h.device), last_idx]

    def hazards_at_bins(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sample the continuous hazard function at the shared discrete bin points (same
        bins Dynamic-DeepHit/TraCeR use) -- the ONLY place discrete bins exist in this
        model. Converts each bin's incremental cumulative hazard (from the augmented ODE
        integral) into a discrete per-bin event probability: 1 - exp(-delta_cum_hazard),
        so hazards_to_cif()'s product-of-survival formula exactly reconstructs the true
        continuous survival function at each bin boundary.
        """
        landmark_state = self.encode_trajectory(values, mask, seq_lengths, elapsed_days)
        batch = landmark_state.shape[0]
        bin_width_days = self.bin_width_years * 365.25
        real_days = torch.full((batch,), bin_width_days, device=landmark_state.device)

        h = landmark_state
        discrete_hazards = {name: [] for name in OUTCOME_NAMES}
        for _ in range(self.num_time_bins):
            h, cum_hazard = self._evolve_with_hazard_integral(h, real_days)
            for name in OUTCOME_NAMES:
                discrete_hazards[name].append(1.0 - torch.exp(-cum_hazard[name]))

        return {name: torch.stack(bins, dim=1) for name, bins in discrete_hazards.items()}

    def hazard_at_time(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor,
        query_days: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Query the instantaneous hazard rate for each outcome at an ARBITRARY continuous
        time (query_days, days from landmark) -- not restricted to bin boundaries. Proves
        genuine continuous-time capability rather than a secretly-discrete internal model.
        """
        landmark_state = self.encode_trajectory(values, mask, seq_lengths, elapsed_days)
        evolved_h = self._evolve(landmark_state, query_days)
        return {name: self._hazard_rate(name, evolved_h) for name in OUTCOME_NAMES}

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        elapsed_days: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if elapsed_days is None:
            elapsed_days = torch.zeros(values.shape[0], values.shape[1], device=values.device)
        return self.hazards_at_bins(values, mask, seq_lengths, elapsed_days)

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        values, mask, seq_lengths, elapsed_days = (
            batch["values"],
            batch["mask"],
            batch["seq_lengths"],
            batch["elapsed_days"],
        )
        landmark_state = self.encode_trajectory(values, mask, seq_lengths, elapsed_days)
        return survlatent_ode_survival_loss(self, landmark_state, batch, self.outcome_weights)

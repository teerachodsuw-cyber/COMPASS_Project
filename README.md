# COMPASS

**Competing-risk Multi-outcome Prognosis Assessment for Steatotic-liver and diabetes Sequelae**

COMPASS is a longitudinal deep-learning research project for predicting three competing-risk
outcomes — all-cause mortality, major adverse liver outcomes (MALO), and cardiovascular
disease (CVD) — jointly, at a single 10-year horizon, in patients with steatotic liver
disease and type 2 diabetes, using repeated-measures clinical data. The single-horizon choice
is deliberate: the source database's maximum follow-up (~12 years) doesn't support validating
predictions further out, so 20/30-year horizons were dropped rather than reported without
real validation data behind them. Modeling all three outcomes simultaneously — with mortality
treated as a genuine competing risk for MALO/CVD, not ordinary censoring — is the project's
central methodological bet, benchmarked head-to-head across three candidate architectures.

## Candidate architectures

- **Dynamic-DeepHit** — recurrent (GRU + attention) architecture for discrete-time
  competing-risks survival prediction from longitudinal data (Lee et al., 2019), corrected to
  use three independent per-outcome hazard heads rather than the original paper's shared
  softmax across causes (see `compass.models.base` for why the joint-softmax framing doesn't
  fit this project's outcome structure).
- **TraCeR** — factorized-attention transformer for competing-risks time-to-event prediction,
  using a native continuous time-decay gate (`exp(-time_since_last_observed / tau)`, a
  learnable per-feature decay) for missingness, rather than the simple mask-and-zero-fill the
  other two architectures use — each candidate keeps its own native missingness mechanism, by
  design, to keep the comparison a fair fight between complete systems.
- **SurvLatent ODE** — neural ODE latent-variable model (Moon et al., 2022) whose latent
  state evolves continuously between visits, over each patient's REAL elapsed calendar time
  rather than a fixed per-step assumption — genuinely continuous-time internally, with
  discrete bins only introduced at the shared evaluation interface for comparison with the
  other two architectures.

## Evaluation

Discrimination and calibration are both split by outcome, matching how each outcome's
censoring actually works:

- **Mortality** has no competing risk here — standard IPCW-based metrics (Uno's C-index,
  IPCW Brier score / IBS).
- **MALO and CVD** have mortality as a real competing risk — a patient who dies before MALO
  or CVD could occur has a permanently precluded (not merely unknown) outcome, which
  standard survival metrics get wrong by treating death as ordinary censoring. These two
  outcomes use Wolbers' (2014) competing-risks C-index (matches R's pec::cindex closely;
  small residual on ties documented in code) and Schoop et al.'s (2011)
  competing-risks-adjusted Brier score (near-exact match to pec::pec) instead.

On top of per-model metrics: paired bootstrap comparison across all three architectures (same
resampled patients, same metrics, for a fair head-to-head), risk score banding into clinical
bands via outcome-optimized cutpoint search (log-rank for mortality, Gray's test for MALO/CVD)
with a quantile-based fallback when event counts are too sparse for outcome-optimized
cutpoints (as MALO's rarity already triggers in testing), hazard-ratio reporting between
bands, and discrimination-calibration scatter visualization.

## Setup

```bash
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Status

The full pipeline is built end-to-end: synthetic data generation → schema validation →
preprocessing (landmarking, missingness masking, scaling, patient-level splitting) → all
three candidate architectures → outcome-split evaluation → paired model comparison → risk
banding → visualization. 105 tests passing. Still pending: the real dataset (IRB approval in
progress) — everything above has been built and validated against the synthetic generator in
`data/synthetic/` in the meantime.

## Methodology

`docs/methodology.md` is currently a short stub (project description + candidate
architectures), not a full methodology writeup — full methodology plan maintained separately;
see [link placeholder].

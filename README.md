# COMPASS

**Competing-risk Multi-horizon Prognosis Assessment for Steatotic-liver and diabetes Sequelae**

COMPASS is a longitudinal deep-learning research project for predicting competing-risk,
multi-horizon outcomes (mortality, major adverse liver outcomes, and cardiovascular disease)
in patients with steatotic liver disease and type 2 diabetes, using repeated-measures clinical
data.

## Candidate architectures

- **Dynamic-DeepHit** — recurrent architecture for discrete-time competing-risks survival
  prediction from longitudinal data (Lee et al., 2019).
- **TRACER** — transformer-based competing-risks model for multi-horizon time-to-event
  prediction.
- **SurvLatent ODE** — neural ODE latent-variable model for continuous-time survival analysis
  over irregularly sampled longitudinal data (Moon et al., 2022).

## Setup

```bash
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Status

Under active development against synthetic data (`data/synthetic/`) pending arrival of the
real dataset.

## Methodology

Full methodology plan maintained separately — see [docs/methodology.md](docs/methodology.md)
and [link placeholder].

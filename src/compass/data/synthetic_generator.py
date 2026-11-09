"""Synthetic patient-visit cohort generator for pipeline development and testing.

Generates data that structurally mimics the Siriraj MASLD-T2DM cohort schema
(configs/data_schema.yaml) without using any real patient data. All variable-level
statistics (mean/sd/prevalence/missing_rate) are read from the schema config, not
hardcoded here, so the config stays the single source of truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Physiologically plausible clipping bounds per continuous variable, (low, high).
# These are simple, reviewable sanity bounds -- not clinical reference ranges --
# chosen only to keep Normal-distribution sampling from producing impossible
# values (e.g. negative age or weight) at the tails.
CONTINUOUS_VARIABLE_BOUNDS = {
    "index_age": (18.0, 105.0),
    "height": (100.0, 210.0),
    "weight": (20.0, 250.0),
    "bmi": (10.0, 70.0),
    "dbp": (30.0, 150.0),
    "sbp": (60.0, 250.0),
    "fbs": (30.0, 600.0),
    "hba1c": (3.0, 18.0),
    "cholesterol": (50.0, 500.0),
    "triglyceride": (20.0, 2000.0),
    "hdl_cholesterol": (10.0, 150.0),
    "ldl_cholesterol": (10.0, 400.0),
    "plt": (5.0, 1000.0),
    # Serum creatinine in mg/dL (source column "crcl" is mislabelled CrCl in the source data).
    "serum_creatinine": (0.05, 15.0),
    "egfr": (2.0, 200.0),
    "alt": (1.0, 2000.0),
    "ast": (1.0, 2000.0),
    "e_median": (1.0, 75.0),
    "cap_median": (100.0, 400.0),
}

DEFAULT_MEAN_VISITS_PER_PATIENT = 8.0  # placeholder pending Stage 1 visit-density audit
DEFAULT_MORTALITY_RATE = 0.15
DEFAULT_CVD_RATE = 0.10
# Intentionally rare -- matches the real rare-event problem this project has to
# handle. Do not "fix" this by balancing it toward 50/50.
DEFAULT_MALO_RATE = 0.02


def _load_schema(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_cohort(
    n_patients: int,
    seed: int = 42,
    config_path: str = "configs/data_schema.yaml",
    mean_visits_per_patient: float = DEFAULT_MEAN_VISITS_PER_PATIENT,
    mortality_rate: float = DEFAULT_MORTALITY_RATE,
    cvd_rate: float = DEFAULT_CVD_RATE,
    malo_rate: float = DEFAULT_MALO_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic patient-visit cohort matching configs/data_schema.yaml.

    Returns (visits_df, outcomes_df): one row per patient-visit, and one row per
    patient with static semi-competing-risks outcome fields.
    """
    schema = _load_schema(config_path)
    rng = np.random.default_rng(seed)

    continuous_vars = schema["continuous_variables"]
    medication_flags = schema["medication_flags"]
    range_start = pd.Timestamp(schema["cohort"]["date_range"][0])
    range_end = pd.Timestamp(schema["cohort"]["date_range"][1])
    range_days = (range_end - range_start).days

    patient_ids = [f"SYN{i:06d}" for i in range(1, n_patients + 1)]

    # Demographics are patient-level static attributes -- sampled once per patient,
    # unlike continuous variables below, which are re-sampled per visit so a
    # patient's labs/vitals vary visit to visit rather than repeating identically.
    sex = rng.choice(["male", "female"], size=n_patients, p=[0.4710, 0.5290])
    # NOTE: no confirmed category proportions exist yet for smoking_status in the
    # schema, so it is sampled uniformly here as a placeholder pending Stage 1.
    # Codes come from the schema: 0=never, 1=former, 2=current.
    smoking_categories = next(d["values"] for d in schema["demographics"] if d["name"] == "smoking_status")
    smoking_status = rng.choice(smoking_categories, size=n_patients)

    # Capped at range_days + 1 so sampling without replacement (below) never runs
    # out of distinct days to draw from -- irrelevant in practice at mean=8, but
    # keeps the generator correct even with an unusually large mean_visits_per_patient.
    n_visits = np.clip(rng.poisson(mean_visits_per_patient, size=n_patients), 1, range_days + 1)

    visit_rows = []
    first_visit_date = {}
    for i, pid in enumerate(patient_ids):
        # Sampled without replacement: a patient cannot have two visits on the
        # same day (visit_date must be unique per patient_id).
        offsets = rng.choice(range_days + 1, size=n_visits[i], replace=False)
        visit_dates = sorted(range_start + pd.to_timedelta(offsets, unit="D"))
        first_visit_date[pid] = visit_dates[0]

        for visit_date in visit_dates:
            row = {
                "patient_id": pid,
                "visit_date": visit_date,
                "sex": sex[i],
                "smoking_status": smoking_status[i],
            }

            for var in continuous_vars:
                name = var["name"]
                value = rng.normal(var["mean"], var["sd"])
                low, high = CONTINUOUS_VARIABLE_BOUNDS[name]
                value = float(np.clip(value, low, high))
                # Missingness applied per visit, completely at random (MCAR).
                # Simplifying assumption for pipeline-testing purposes only --
                # not a claim about the real missingness mechanism.
                if rng.random() < var.get("missing_rate", 0.0):
                    value = np.nan
                row[name] = value

            for med in medication_flags:
                row[med["name"]] = int(rng.random() < med["prevalence"])

            visit_rows.append(row)

    visits_df = pd.DataFrame(visit_rows)

    def _sample_event_date(lower: pd.Timestamp) -> pd.Timestamp:
        lower = max(lower, range_start)
        if lower >= range_end:
            return range_end
        offset = rng.integers(0, (range_end - lower).days + 1)
        return lower + pd.Timedelta(days=int(offset))

    outcome_rows = []
    for pid in patient_ids:
        fvd = first_visit_date[pid]

        malo_event = int(rng.random() < malo_rate)
        malo_date = _sample_event_date(fvd) if malo_event else pd.NaT

        cvd_event = int(rng.random() < cvd_rate)
        cvd_date = _sample_event_date(fvd) if cvd_event else pd.NaT

        mortality_event = int(rng.random() < mortality_rate)
        if mortality_event:
            # Mortality date can't logically precede a non-fatal outcome for the
            # same patient (synthetic simplification, not clinically rigorous).
            lower_bound = fvd
            if malo_event:
                lower_bound = max(lower_bound, malo_date)
            if cvd_event:
                lower_bound = max(lower_bound, cvd_date)
            mortality_date = _sample_event_date(lower_bound)
        else:
            mortality_date = pd.NaT

        outcome_rows.append(
            {
                "patient_id": pid,
                "mortality_event": mortality_event,
                "mortality_date": mortality_date,
                "malo_event": malo_event,
                "malo_date": malo_date,
                "cvd_event": cvd_event,
                "cvd_date": cvd_date,
            }
        )

    outcomes_df = pd.DataFrame(outcome_rows)

    return visits_df, outcomes_df


if __name__ == "__main__":
    sample_visits, sample_outcomes = generate_synthetic_cohort(n_patients=50, seed=42)
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_visits.to_csv(out_dir / "sample_cohort.csv", index=False)
    sample_outcomes.to_csv(out_dir / "sample_outcomes.csv", index=False)
    print(f"Wrote {len(sample_visits)} visit rows and {len(sample_outcomes)} patient outcome rows.")

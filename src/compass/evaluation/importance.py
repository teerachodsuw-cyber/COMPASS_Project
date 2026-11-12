"""Permutation feature importance (predictive reliance, not causal effect) and Cox / Fine-Gray
hazard ratios from the stored baseline fit (scripts/feature_importance.py). Aggregates only.

Importance of a variable (or group) for one predictor and outcome = C(unpermuted) - C(permuted):
the C-index (Uno for mortality, Wolbers for MALO / CVD; compass.evaluation.metrics via
compute_all_metrics, IPCW reference = the training split) at the horizon, after the variable is
shuffled across the evaluated patients. The same permutation is used for every predictor
(paired), drawn from np.random.default_rng([seed, unit index, repeat]).

What is moved:
- deep models (permute_examples): in each landmark example, the variable's columns in
  input_values AND input_mask are replaced by the donor patient's, aligned from the most recent
  visit backwards (the landmark visit receives the donor's landmark-visit entries, and so on);
  rows the donor does not have become missing (value 0, mask 0). Everything else (other
  variables, visit dates, sequence length) stays the patient's own.
- Cox / Fine-Gray (permute_features): the variable's columns of the last-observed feature matrix,
  NaN included, so each missing indicator moves with its column.
- a group: every member column moved with ONE shared permutation.

Variables: the 36 input columns of build_feature_names grouped into 34 variables -- sex is one
column (sex_female); smoking status is three one-hot columns that are always moved together
(moving one indicator alone would create impossible inputs).

Hazard ratios (cox_hazard_ratios): exp(beta) on the stored, ridge-penalised (0.01) fit, rescaled
to 1 SD of the variable's observed training values (continuous) or present vs absent (binary
inputs coded 0 / 1); subdistribution HRs for MALO / CVD (Fine-Gray). 95% CIs from the
model-based variance at the STORED coefficients (hr_variance.R: coxph with init = beta,
iter.max = 0, same data / penalty / ties -- nothing re-estimated; the returned coefficients are
checked to equal the stored ones).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from compass.evaluation.metrics import compute_all_metrics
from compass.models.base import DEATH_OUTCOME, OUTCOME_NAMES
from compass.models.cox_baseline import (
    apply_feature_transform,
    predict_risk_from_features,
)

R_VARIANCE_SCRIPT = Path(__file__).with_name("hr_variance.R")
COX_NAME = "cox_baseline"

VARIABLE_LABEL = {
    "index_age": "Age", "sex": "Sex (female)", "smoking": "Smoking status", "height": "Height", "weight": "Weight",
    "bmi": "BMI", "sbp": "SBP", "dbp": "DBP", "fbs": "FBS", "hba1c": "HbA1c", "cholesterol": "Total cholesterol",
    "triglyceride": "Triglycerides", "hdl_cholesterol": "HDL-C", "ldl_cholesterol": "LDL-C", "plt": "Platelets",
    "serum_creatinine": "Serum creatinine", "egfr": "eGFR", "alt": "ALT", "ast": "AST", "e_median": "LSM (FibroScan)",
    "cap_median": "CAP (FibroScan)", "metformin": "Metformin", "sulfonylureas": "Sulfonylureas",
    "pioglitazone": "Pioglitazone", "insulin": "Insulin", "dpp4": "DPP-4 inhibitors", "glp1": "GLP-1 receptor agonists",
    "sglt2": "SGLT2 inhibitors", "acei": "ACE inhibitors", "arb": "ARBs", "arni": "ARNI", "statins": "Statins",
    "fibrate": "Fibrates", "ezetimibe": "Ezetimibe",
}
GROUPS = {
    "Demographics": ["index_age", "sex", "smoking"],
    "Body size & BP": ["height", "weight", "bmi", "sbp", "dbp"],
    "Glycaemic": ["fbs", "hba1c"],
    "Lipids": ["cholesterol", "triglyceride", "hdl_cholesterol", "ldl_cholesterol"],
    "Liver & platelets": ["alt", "ast", "plt"],
    "Kidney": ["serum_creatinine", "egfr"],
    "FibroScan": ["e_median", "cap_median"],
    "Diabetes medications": ["metformin", "sulfonylureas", "pioglitazone", "insulin", "dpp4", "glp1", "sglt2"],
    "CV medications": ["acei", "arb", "arni", "statins", "fibrate", "ezetimibe"],
}


# ---- variables and permutations -------------------------------------------------------------


def variable_columns(feature_names: list[str]) -> dict[str, list[str]]:
    """{variable: [input columns]} in feature order: one column each, except smoking (its
    one-hot columns together) and sex (sex_female)."""
    out = {}
    for f in feature_names:
        if f.startswith("smoking_status_"):
            out.setdefault("smoking", []).append(f)
        elif f.startswith("sex_"):
            out.setdefault("sex", []).append(f)
        else:
            out[f] = [f]
    missing = set(VARIABLE_LABEL) - set(out)
    if missing or set(out) - set(VARIABLE_LABEL):
        raise ValueError(f"feature names do not match the importance variables: {sorted(missing ^ (set(out) - set(VARIABLE_LABEL)))}")
    return out


def importance_units(feature_names: list[str]) -> list[dict]:
    """[{"kind": "variable" | "group", "name", "label", "columns"}]: every variable, then every
    group (its members' columns, shuffled with one shared permutation)."""
    cols = variable_columns(feature_names)
    units = [{"kind": "variable", "name": v, "label": VARIABLE_LABEL[v], "columns": c} for v, c in cols.items()]
    for g, members in GROUPS.items():
        units.append({"kind": "group", "name": g, "label": g, "columns": [c for m in members for c in cols[m]],
                      "members": list(members)})
    return units


def permutation_for(n: int, seed: int, unit_index: int, repeat: int) -> np.ndarray:
    return np.random.default_rng([int(seed), int(unit_index), int(repeat)]).permutation(n)


def permute_examples(examples: list[dict], columns: list[str], perm: np.ndarray) -> list[dict]:
    """Copies of `examples` where patient i's `columns` (values AND mask) come from patient
    perm[i], aligned from the most recent visit backwards; rows the donor lacks -> missing."""
    out = []
    for i, ex in enumerate(examples):
        donor = examples[int(perm[i])]
        vals, mask = ex["input_values"].copy(), ex["input_mask"].copy()
        n_own, n_don = len(vals), len(donor["input_values"])
        k = min(n_own, n_don)
        dv = donor["input_values"][columns].to_numpy()[n_don - k:]
        dm = donor["input_mask"][columns].to_numpy()[n_don - k:]
        new_v = np.zeros((n_own, len(columns)), dtype=vals[columns].to_numpy().dtype)
        new_m = np.zeros((n_own, len(columns)), dtype=mask[columns].to_numpy().dtype)
        new_v[n_own - k:], new_m[n_own - k:] = dv, dm
        vals.loc[:, columns] = new_v
        mask.loc[:, columns] = new_m
        out.append({**ex, "input_values": vals, "input_mask": mask})
    return out


def permute_features(x: np.ndarray, feature_names: list[str], columns: list[str], perm: np.ndarray) -> np.ndarray:
    """Copy of the (n, F) last-observed matrix with `columns` taken from row perm[i] (NaN kept,
    so the missing indicator moves with its column)."""
    x = np.array(x, dtype=float, copy=True)
    idx = [feature_names.index(c) for c in columns]
    x[:, idx] = x[np.asarray(perm)][:, idx]
    return x


# ---- predictors and C-index -----------------------------------------------------------------


def cox_predictions(x: np.ndarray, fit: dict, horizon: int) -> dict:
    """{outcome: (n, horizon) risk} of the stored Cox / Fine-Gray fit from a last-observed matrix."""
    z = apply_feature_transform(x, fit["transform"])
    risk = predict_risk_from_features(z, fit, list(range(1, int(horizon) + 1)))
    return {name: risk[:, k, :] for k, name in enumerate(OUTCOME_NAMES)}


def c_indices(predictions: dict, ground_truth: dict, reference: dict, horizon: int) -> dict:
    """{outcome: C-index at the horizon}, exactly as the primary evaluation computes it."""
    df = compute_all_metrics(predictions, ground_truth, reference, horizon)
    return {r.outcome: float(r.value) for r in df[df.metric_name == "cindex"].itertuples()}


def check_unpermuted(observed: dict, locked: dict, tol: float = 0.0) -> list[str]:
    """Differences between the unpermuted C-index and the locked values ({(predictor, outcome): C});
    empty if all equal (exactly, by default)."""
    diffs = []
    for key, c in observed.items():
        ref = locked.get(key)
        if ref is None or not abs(c - ref) <= tol:
            diffs.append(f"{key[0]} / {key[1]}: {c!r} vs locked {ref!r}")
    return diffs


def run_importance(predictors: dict, n_patients: int, units: list[dict], ground_truth: dict, reference: dict,
                   horizon: int, n_repeats: int, seed: int, progress=None) -> tuple[dict, list[dict]]:
    """predictors: {name: fn(columns | None, perm | None) -> {outcome: (n, horizon) risk}} (None =
    unpermuted). Returns (unpermuted {(name, outcome): C}, rows: one per unit x predictor x outcome
    with the per-repeat drops, mean and 2.5 / 97.5 percentiles)."""
    base = {(name, o): c for name, fn in predictors.items() for o, c in c_indices(fn(None, None), ground_truth, reference, horizon).items()}
    rows = []
    for u_idx, unit in enumerate(units):
        drops = {(name, o): [] for name in predictors for o in OUTCOME_NAMES}
        for r in range(n_repeats):
            perm = permutation_for(n_patients, seed, u_idx, r)
            for name, fn in predictors.items():
                for o, c in c_indices(fn(unit["columns"], perm), ground_truth, reference, horizon).items():
                    drops[(name, o)].append(base[(name, o)] - c)
        for (name, o), d in drops.items():
            d = np.asarray(d, dtype=float)
            rows.append({"kind": unit["kind"], "unit": unit["name"], "label": unit["label"], "predictor": name, "outcome": o,
                         "c_unpermuted": base[(name, o)], "mean_drop": float(d.mean()),
                         "p2_5": float(np.percentile(d, 2.5)), "p97_5": float(np.percentile(d, 97.5)),
                         "drops": d.tolist()})
        if progress:
            progress(u_idx + 1, len(units), unit["name"])
    return base, rows


# ---- hazard ratios (stored fit; variance at the stored coefficients) --------------------------


def coefficient_variance(z: np.ndarray, time, event_type, beta, penalizer: float, mode: str, ties: str = "efron",
                         rscript: str = "Rscript") -> np.ndarray:
    """(p, p) model-based variance of the stored penalised fit at `beta` (no re-estimation)."""
    n = len(time)
    with tempfile.TemporaryDirectory() as tmp:
        infile, bfile, outfile = (os.path.join(tmp, f) for f in ("train.csv", "beta.json", "var.json"))
        df = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
        df.insert(0, "status", np.asarray(event_type).astype(int))
        df.insert(0, "time", np.asarray(time, dtype=float))
        df.to_csv(infile, index=False, float_format="%.17g")
        Path(bfile).write_text(json.dumps([float(b) for b in beta]))
        proc = subprocess.run([rscript, str(R_VARIANCE_SCRIPT), infile, bfile, outfile, repr(float(penalizer) * n), ties, mode],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"R variance evaluation failed:\n{proc.stderr[-2000:]}")
        res = json.loads(Path(outfile).read_text())
    returned = np.atleast_1d(np.asarray(res["coefficients"], dtype=float))
    if not np.allclose(returned, np.asarray(beta, dtype=float), rtol=0, atol=1e-12):
        raise RuntimeError(f"R returned different coefficients (max diff {np.abs(returned - np.asarray(beta)).max():.3g}); "
                           "the variance would not be at the stored estimates")
    return np.asarray(res["var"], dtype=float).reshape(len(beta), len(beta))


def cox_hazard_ratios(fit: dict, x_train: np.ndarray, ground_truth_train: dict, raw_sd: dict,
                      variance_fn=coefficient_variance) -> list[dict]:
    """Per outcome and input variable: HR (95% CI) per 1 SD of the observed training values
    (continuous) or present vs absent (0 / 1 inputs), from the stored coefficients; missing
    indicators too (missing vs observed). raw_sd: {feature: SD in raw units} for the table."""
    t = fit["transform"]
    names, columns = list(t["feature_names"]), list(t["columns"])
    scale = dict(zip(columns, t["scale"]))
    z = apply_feature_transform(x_train, t)
    rows = []
    for outcome in OUTCOME_NAMES:
        o = fit["outcomes"][outcome]
        beta = np.asarray(o["coefficients"], dtype=float)
        g = ground_truth_train[outcome]
        keep = np.asarray(g["at_risk"]).astype(bool)
        mode = "cox" if outcome == DEATH_OUTCOME else "finegray"
        var = variance_fn(z[keep], np.asarray(g["time_years"], float)[keep], np.asarray(g["event_type"])[keep], beta,
                          float(o["penalizer"]), mode)
        se = np.sqrt(np.diag(var))
        for j, col in enumerate(columns):
            if col.endswith("_missing") and col[: -len("_missing")] in names:
                feature, kind, delta, unit = col[: -len("_missing")], "missing indicator", 1.0, "missing vs observed"
            else:
                feature = col
                observed = x_train[:, names.index(col)]
                observed = observed[~np.isnan(observed)]
                binary = set(np.unique(observed)) <= {0.0, 1.0}
                kind = "binary" if binary else "continuous"
                delta = 1.0 if binary else float(np.std(observed, ddof=1))
                sd_txt = (f"{raw_sd[col]:.1f}" if raw_sd[col] >= 1 else f"{raw_sd[col]:.2f}") if col in raw_sd else ""
                unit = "present vs absent" if binary else (f"per 1 SD ({sd_txt})" if sd_txt else "per 1 SD")
            k = delta / scale[col]  # design columns are standardised: one input unit = 1 / scale
            b, s = beta[j] * k, se[j] * k
            rows.append({"outcome": outcome, "column": col, "feature": feature, "kind": kind, "contrast": unit,
                         "log_hr": float(b), "se": float(s), "hr": float(np.exp(b)),
                         "hr_ci_lower": float(np.exp(b - 1.959963984540054 * s)),
                         "hr_ci_upper": float(np.exp(b + 1.959963984540054 * s)),
                         "type": "HR (Cox)" if outcome == DEATH_OUTCOME else "subdistribution HR (Fine-Gray)"})
    return rows

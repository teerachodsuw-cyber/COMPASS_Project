#!/usr/bin/env python
"""Permutation feature importance on the TEST split (predictive reliance, not causal effect) and
Cox / Fine-Gray hazard ratios from the stored baseline fit. Aggregates only; nothing is refitted.

Usage:
    python scripts/feature_importance.py --final --data data/processed            # real data
    python scripts/feature_importance.py --final --data data/processed --estimate-only

Setting: each test patient's first valid landmark (the primary test evaluation's examples),
10-yr horizon; predictors = the three deep models (final per-outcome checkpoints,
configs/final_models.yaml) + cox_baseline (runs/baseline/cox_baseline.json). C-index (Uno
mortality; Wolbers MALO / CVD), IPCW reference = the training split. Importance = C(unpermuted)
- C(permuted); --n-repeats permutations per variable / group (fixed --seed), reported as the mean
drop and the 2.5-97.5 percentile range across repeats. See compass.evaluation.importance.

Order of work:
1. The unpermuted C-index of every predictor / outcome must equal the locked test values
   (--locked-eval: deep models; --locked-baseline: cox_baseline) EXACTLY, else exit 4.
2. The run time is estimated from that pass; above --max-hours: report and exit 5.
   --estimate-only stops here.
3. Permutations; then hazard ratios (stored coefficients; variance evaluated at them in R).

Outputs in --out-dir (git-ignored): importance.json, importance_individual.csv,
importance_grouped.csv, cox_hazard_ratios.csv / .json, table_s1.csv, table_s1.docx.

--table-s1-only rebuilds table_s1.csv / .docx from the stored cox_hazard_ratios.json and the
training split (medication exposure counts) without touching the test split or re-running
the permutations.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compass.evaluation.importance import (
    COX_NAME,
    GROUPS,
    VARIABLE_LABEL,
    c_indices,
    check_unpermuted,
    cox_hazard_ratios,
    cox_predictions,
    importance_units,
    permute_examples,
    permute_features,
    run_importance,
    variable_columns,
)
from compass.evaluation.pipeline import (
    censoring_reference_from_train,
    split_ground_truth,
)
from compass.evaluation.serial import predict_examples, snapshot_features
from compass.models.base import OUTCOME_NAMES
from compass.models.cox_baseline import load_fit

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate_script = _load_script("evaluate")
generate_synthetic_cohort = evaluate_script.generate_synthetic_cohort

OUTCOME_LABEL = {"mortality": "Mortality", "malo": "MALO", "cvd": "CVD"}
FOOTNOTE = ("Permutation importance: decrease in C-index when the variable (or group) is randomly shuffled across "
            "patients; reflects predictive reliance, not causal effect.")


def locked_cindex(eval_dir, baseline_dir) -> dict:
    locked = {}
    for r in json.loads((Path(eval_dir) / "metrics.json").read_text())["per_model"]:
        if r["metric_name"] == "cindex" and r["model"] != COX_NAME:
            locked[(r["model"], r["outcome"])] = r["point_estimate"]
    for r in json.loads((Path(baseline_dir) / "metrics.json").read_text())["per_model"]:
        if r["metric_name"] == "cindex" and r["model"] == COX_NAME:
            locked[(COX_NAME, r["outcome"])] = r["point_estimate"]
    return locked


def raw_sds(trainer, x_train: np.ndarray) -> dict:
    """{continuous feature: SD of the observed training first-landmark values in raw units}."""
    names = list(trainer.feature_names)
    out = {}
    for col, scaler in trainer.scalers.items():
        v = x_train[:, names.index(col)]
        out[col] = float(np.nanstd(v, ddof=1) * scaler.scale_[0])
    return out


# ---- Table S1 -------------------------------------------------------------------------------


MEDICATIONS = [*GROUPS["Diabetes medications"], *GROUPS["CV medications"]]
SPARSE_EXPOSURE = 0.01  # < 1% exposed in the training split: flagged as unstable


def medication_exposure(x_train: np.ndarray, feature_names: list[str]) -> dict:
    """{medication: (n exposed, n)} at each training patient's first valid landmark (the most
    recent recorded value, i.e. the value the baseline model saw)."""
    out = {}
    for c in MEDICATIONS:
        v = x_train[:, feature_names.index(c)]
        out[c] = (int(np.sum(v == 1)), len(v))
    return out


def sparse_medications(exposure: dict) -> list[str]:
    return [c for c, (k, n) in exposure.items() if k / n < SPARSE_EXPOSURE]


def table_s1_rows(hr_rows: list[dict], units: dict, exposure: dict) -> list[dict]:
    """One row per input variable, HR (95% CI) per outcome. Left out: missing indicators (CSV /
    JSON only) and the smoking indicators (not separately interpretable). Medication rows carry
    n (%) exposed in the training split; sparse exposures are marked with a dagger."""
    by = {(r["column"], r["outcome"]): r for r in hr_rows}
    cols = [c for c in dict.fromkeys(r["column"] for r in hr_rows)
            if not c.endswith("_missing") and not c.startswith("smoking_status_")]
    sparse = set(sparse_medications(exposure))
    rows = []
    for c in cols:
        r0 = by[(c, OUTCOME_NAMES[0])]
        if c == "sex_female":
            label = "Female vs male"
        elif r0["kind"] == "binary":
            label = f"{VARIABLE_LABEL[c]} (yes vs no)" + ("\u2020" if c in sparse else "")
        else:
            unit = units.get(c, "")
            sd = r0["contrast"].split("(")[1].rstrip(")") if "(" in r0["contrast"] else ""
            label = f"{VARIABLE_LABEL[c]}, per 1 SD ({sd} {unit})".replace(" )", ")")
        row = {"variable": label, "exposed": ""}
        if c in exposure:
            k, n = exposure[c]
            row["exposed"] = f"{k:,} ({100 * k / n:.1f})"
        for o in OUTCOME_NAMES:
            r = by[(c, o)]
            row[o] = f"{r['hr']:.2f} ({r['hr_ci_lower']:.2f}–{r['hr_ci_upper']:.2f})"
        rows.append(row)
    return rows


S1_COLUMNS = {"variable": "Variable", "exposed": "Exposed in training, n (%)", "mortality": "Mortality, HR (95% CI)",
              "malo": "MALO, sHR (95% CI)", "cvd": "CVD, sHR (95% CI)"}
S1_TITLE = "Table S1. Hazard ratios of the classical Cox / Fine-Gray baseline (training split)"
S1_MODEL_NOTE = (
    "Ridge-penalised (0.01) Cox (mortality) and Fine-Gray (MALO, CVD) models; MALO and CVD values are "
    "subdistribution hazard ratios. Associations are adjusted for all listed variables but are not causal; medication "
    "associations are likely confounded by indication."
)
S1_SMOKING_NOTE = (
    "Smoking was modelled as a categorical variable with a missing category; its indicator coefficients are not "
    "separately interpretable and are not shown."
)
S1_METHODS_NOTE = (
    "Models were fitted on the training split at each patient's first valid landmark (n = {n:,}); 95% CIs use the "
    "model-based variance of the penalised fit at the stored estimates. Continuous variables: per 1 SD of the observed "
    "training values (SD in brackets), the most recent value on or before the landmark; missing values were filled "
    "with the training mean and each such variable has a missing indicator (estimates in the supplementary CSV). "
    "Binary variables: present vs absent; exposed, n (%) of training patients with the medication recorded at the "
    "landmark."
)
S1_ABBREVIATIONS = (
    "ACE, angiotensin-converting enzyme; ARB, angiotensin II receptor blocker; ARNI, angiotensin receptor-neprilysin "
    "inhibitor; ALT, alanine aminotransferase; AST, aspartate aminotransferase; BMI, body mass index; CAP, controlled "
    "attenuation parameter; CVD, cardiovascular disease; DBP, diastolic blood pressure; DPP-4, dipeptidyl peptidase-4; "
    "eGFR, estimated glomerular filtration rate; FBS, fasting blood sugar; GLP-1, glucagon-like peptide-1; HbA1c, "
    "glycated haemoglobin; HDL-C, high-density lipoprotein cholesterol; LDL-C, low-density lipoprotein cholesterol; "
    "LSM, liver stiffness measurement; MALO, major adverse liver outcomes; SBP, systolic blood pressure; SGLT2, "
    "sodium-glucose cotransporter-2."
)


def s1_footnotes(exposure: dict) -> list[str]:
    sparse = sparse_medications(exposure)
    n = next(iter(exposure.values()))[1]
    notes = [S1_MODEL_NOTE, S1_SMOKING_NOTE]
    if sparse:
        names = ", ".join(VARIABLE_LABEL[c] for c in sparse)
        notes.append(f"\u2020 Sparse exposure (< 1% of the training split: {names}); estimates are unstable.")
    return [*notes, S1_METHODS_NOTE.format(n=n), S1_ABBREVIATIONS]


def write_table_s1(rows: list[dict], exposure: dict, out: Path) -> None:
    pd.DataFrame(rows)[list(S1_COLUMNS)].rename(columns=S1_COLUMNS).to_csv(out / "table_s1.csv", index=False)
    write_table_s1_docx(rows, s1_footnotes(exposure), out / "table_s1.docx")


def write_table_s1_docx(rows: list[dict], footnotes: list[str], path: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    def border(cell, *edges):
        tc_pr = cell._tc.get_or_add_tcPr()
        b = tc_pr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tc_pr.append(b)
        for e in edges:
            el = OxmlElement(f"w:{e}")
            for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "0"), ("w:color", "000000")):
                el.set(qn(k), v)
            b.append(el)

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    doc.styles["Normal"].font.name, doc.styles["Normal"].font.size = "Arial", Pt(8)
    doc.styles["Normal"].paragraph_format.space_after = Pt(0)
    t = doc.add_paragraph().add_run(S1_TITLE)
    t.bold, t.font.size = True, Pt(10)
    keys = list(S1_COLUMNS)
    table = doc.add_table(rows=1 + len(rows), cols=len(keys))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Cm(5.6), Cm(2.6), Cm(2.9), Cm(2.9), Cm(2.9)]
    for i, r in enumerate([S1_COLUMNS, *rows]):
        cells = table.rows[i].cells
        for j, key in enumerate(keys):
            cells[j].width = widths[j]
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            run = p.add_run(r[key])
            run.bold = i == 0
            if i == 0:
                border(cells[j], "top", "bottom")
            elif i == len(rows):
                border(cells[j], "bottom")
    for k, text in enumerate(footnotes):
        note = doc.add_paragraph(text)
        note.paragraph_format.space_before = Pt(4 if k == 0 else 1)
        for r in note.runs:
            r.font.size = Pt(7)
    doc.save(path)


# ---- main -----------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", action="store_true", help="Required: this reads the TEST split.")
    p.add_argument("--final-models", default="configs/final_models.yaml")
    p.add_argument("--baseline", default="runs/baseline/cox_baseline.json")
    p.add_argument("--locked-eval", default="outputs/final_test", help="Locked deep-model test results (metrics.json).")
    p.add_argument("--locked-baseline", default="outputs/baseline_test", help="Locked cox_baseline test results.")
    p.add_argument("--data", default=None, help="Processed real-data dir. Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only.")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=None, help="Permutation seed (default: configs/training.yaml seed).")
    p.add_argument("--max-hours", type=float, default=3.0)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--out-dir", default="outputs/importance")
    p.add_argument("--table-s1-only", action="store_true",
                   help="Rebuild table_s1.csv/.docx from <out-dir>/cox_hazard_ratios.json (training split only).")
    args = p.parse_args(argv)
    if not args.final:
        print("refusing: this reads the TEST split; pass --final", file=sys.stderr)
        return 2

    schema = yaml.safe_load(Path("configs/data_schema.yaml").read_text())
    training = yaml.safe_load(Path("configs/training.yaml").read_text())
    horizon = int(training["horizons"]["primary"])
    seed = args.seed if args.seed is not None else int(training["seed"])
    if args.data:
        visits = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        visits, outcomes = generate_synthetic_cohort(n_patients=args.n_patients,
                                                     seed=args.data_seed if args.data_seed is not None else int(training["seed"]))
    ns = argparse.Namespace(final_models=args.final_models, checkpoint_dir=None, models=None)
    try:
        names, dirs, split = evaluate_script.resolve_models(ns, visits, outcomes)
    except (evaluate_script.SplitMismatchError, evaluate_script.FinalModelsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3
    fit = load_fit(args.baseline)
    if (fit.get("fitted_on_split") != "train" or fit.get("split_seed") != int(split["seed"])
            or fit.get("data_fingerprint") != split["fingerprint"]):
        print(f"refusing: {args.baseline} was not fitted on this data's training split", file=sys.stderr)
        return 3

    trainers, models = {}, {}
    for m in names:
        trainers[m], models[m] = evaluate_script.load_trained_from(m, dirs[m], split, visits, outcomes, schema, training)
    t0 = trainers[names[0]]
    pretty = {"percent": "%", "10^3/uL": "×10³/µL", "kg/m2": "kg/m²", "mL/min/1.73m2": "mL/min/1.73 m²"}
    units_by_name = {v["name"]: pretty.get(v["unit"], v["unit"]) for v in schema["continuous_variables"]}
    out = Path(args.out_dir)
    if args.table_s1_only:
        stored = json.loads((out / "cox_hazard_ratios.json").read_text())
        if stored["source"] != args.baseline:
            print(f"refusing: stored HRs come from {stored['source']}, not {args.baseline}", file=sys.stderr)
            return 3
        x_train = snapshot_features(t0, t0.fixed_examples("train"))
        exposure = medication_exposure(x_train, list(t0.feature_names))
        write_table_s1(table_s1_rows(stored["rows"], units_by_name, exposure), exposure, out)
        print(f"wrote {out}/table_s1.csv, table_s1.docx (from stored HRs); sparse (<1%): "
              f"{', '.join(VARIABLE_LABEL[c] for c in sparse_medications(exposure)) or 'none'}")
        return 0
    examples = {m: trainers[m].fixed_examples("test") for m in names}
    ids = [ex["patient_id"] for ex in examples[names[0]]]
    for m in names[1:]:
        if [ex["patient_id"] for ex in examples[m]] != ids:
            raise RuntimeError(f"{m}: different test patients / order")
    feature_names = list(t0.feature_names)
    if feature_names != fit["transform"]["feature_names"]:
        raise RuntimeError("baseline fit and deep models use different input variables")
    gt_ids, ground_truth = split_ground_truth(t0, "test")
    if gt_ids != ids:
        raise RuntimeError("ground truth rows do not match the test examples")
    reference = censoring_reference_from_train(t0)
    x_test = snapshot_features(t0, examples[names[0]])

    def deep(m):
        def fn(columns, perm):
            exs = examples[m] if columns is None else permute_examples(examples[m], columns, perm)
            return predict_examples(models[m], trainers[m], exs, horizon, trainers[m].batch_size)
        return fn

    def cox(columns, perm):
        x = x_test if columns is None else permute_features(x_test, feature_names, columns, perm)
        return cox_predictions(x, fit, horizon)

    predictors = {**{m: deep(m) for m in names}, COX_NAME: cox}
    units = importance_units(feature_names)

    # 1. unpermuted C-index == locked values (exactly); 2. run-time estimate
    start = time.time()
    base = {(name, o): c for name, fn in predictors.items() for o, c in c_indices(fn(None, None), ground_truth, reference, horizon).items()}
    per_pass = time.time() - start
    if args.data:
        diffs = check_unpermuted(base, locked_cindex(args.locked_eval, args.locked_baseline))
        if diffs:
            print("STOP: unpermuted C-index differs from the locked test values:\n  " + "\n  ".join(diffs), file=sys.stderr)
            return 4
        print(f"check: unpermuted C-index equals the locked test values exactly ({len(base)} predictor x outcome)")
    n_passes = len(units) * args.n_repeats
    estimate = per_pass * n_passes * 1.05
    print(f"units: {sum(u['kind'] == 'variable' for u in units)} variables + {len(GROUPS)} groups; {args.n_repeats} repeats "
          f"-> {n_passes} permutation passes x {len(predictors)} predictors; {per_pass:.1f} s per pass "
          f"-> estimated {estimate / 60:.0f} min")
    sys.stdout.flush()
    if estimate > args.max_hours * 3600:
        print(f"STOP: estimated run time {estimate / 3600:.1f} h exceeds {args.max_hours:g} h", file=sys.stderr)
        return 5
    if args.estimate_only:
        return 0

    # 3. permutations
    run_start = time.time()

    def progress(i, n, name):
        el = time.time() - run_start
        print(f"  [{i}/{n}] {name}: {el / 60:.1f} min elapsed, ~{el / i * (n - i) / 60:.0f} min left", flush=True)

    base2, rows = run_importance(predictors, len(ids), units, ground_truth, reference, horizon, args.n_repeats, seed, progress)
    if base2 != base:
        raise RuntimeError("unpermuted C-index changed during the run")
    runtime = time.time() - run_start

    # hazard ratios: stored fit, training split (the examples it was fitted on)
    train_examples = t0.fixed_examples("train")
    x_train = snapshot_features(t0, train_examples)
    _, gt_train = split_ground_truth(t0, "train")
    if len(train_examples) != int(fit.get("n_train_patients", len(train_examples))):
        raise RuntimeError("training examples do not match the stored baseline fit")
    hr_rows = cox_hazard_ratios(fit, x_train, gt_train, raw_sds(t0, x_train))

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "analysis": "permutation feature importance (predictive reliance, not causal effect)",
        "split": "test", "data": "real (processed)" if args.data else "synthetic", "n_patients": len(ids),
        "landmark": "first valid landmark (primary test evaluation examples)", "horizon_years": horizon,
        "metric": {"mortality": "Uno C-index", "malo": "Wolbers C-index", "cvd": "Wolbers C-index"},
        "ipcw_reference": "training split", "predictors": list(predictors), "n_repeats": args.n_repeats, "seed": seed,
        "unpermuted_cindex": [{"predictor": k[0], "outcome": k[1], "cindex": v} for k, v in base.items()],
        "locked_check": "equal" if args.data else "not run (synthetic)",
        "variables": variable_columns(feature_names), "groups": GROUPS,
        "runtime_minutes": round(runtime / 60, 1), "estimated_minutes": round(estimate / 60, 1),
        "footnote": FOOTNOTE,
    }
    individual = [r for r in rows if r["kind"] == "variable"]
    grouped = [r for r in rows if r["kind"] == "group"]
    (out / "importance.json").write_text(json.dumps({**meta, "individual": individual, "grouped": grouped}, indent=2))
    cols = ["unit", "label", "predictor", "outcome", "c_unpermuted", "mean_drop", "p2_5", "p97_5"]
    pd.DataFrame(individual)[cols].to_csv(out / "importance_individual.csv", index=False)
    pd.DataFrame(grouped)[cols].to_csv(out / "importance_grouped.csv", index=False)
    hr_meta = {"source": args.baseline, "penalizer": fit["outcomes"]["mortality"]["penalizer"],
               "note": "ridge-penalised (0.01); MALO / CVD are subdistribution HRs (Fine-Gray); 95% CI from the model-based "
                       "variance at the stored coefficients (no refitting)"}
    (out / "cox_hazard_ratios.json").write_text(json.dumps({**hr_meta, "rows": hr_rows}, indent=2))
    pd.DataFrame(hr_rows).to_csv(out / "cox_hazard_ratios.csv", index=False)
    exposure = medication_exposure(x_train, feature_names)
    write_table_s1(table_s1_rows(hr_rows, units_by_name, exposure), exposure, out)
    print(f"runtime {runtime / 60:.1f} min; wrote {out}/: importance.json, importance_individual.csv, importance_grouped.csv, "
          "cox_hazard_ratios.csv/.json, table_s1.csv/.docx")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Pre-specified secondary analysis 2: serial evaluation (aggregates only).

2a: each deep model with its FULL visit history vs a landmark Cox / Fine-Gray model fitted
    separately at each prediction point on the training split.
2b: history ablation -- the same deep model, checkpoint, prediction point and patients, given
    (i) the full visit history vs (ii) a single-visit LOCF snapshot at the landmark visit.

Usage:
    python scripts/serial_evaluate.py --final-models configs/final_models.yaml --data data/processed \\
        --out-dir outputs/serial_val                                  # validation (fits the landmark Cox)
    python scripts/serial_evaluate.py --final-models configs/final_models.yaml --data data/processed \\
        --split test --final --out-dir outputs/serial_test            # ONE-TIME test run

Prediction points k = 1, 2, 3 yr after index_date; landmark visit = the most recent visit in
(index + k - 1 yr, index + k yr], patient alive at it; horizon 5 yr from the landmark visit
(compass.evaluation.serial). Per k: C-index (Uno mortality, Wolbers MALO / CVD) at 5 yr, IBS
to 5 yr (Schoop for MALO / CVD), O/E at 5 yr, IPCW reference = the training split's landmark
examples at the same k, one 1,000-draw paired patient-level bootstrap over all 7 predictors
(landmark Cox, 3 deep snapshot, 3 deep full; listed in that order, so every pairwise row reads
"later_minus_earlier": <m>_full_minus_landmark_cox (2a), <m>_full_minus_<m>_snapshot (2b)).

The deep models are the final checkpoints (configs/final_models.yaml, per-outcome checkpoints),
never refitted; their files are hashed before and after the run and must be unchanged. The
landmark Cox fits are made by the VALIDATION run only (training split only) and saved to
--landmark-cox (runs/serial/landmark_cox.json); the test run loads them and never fits.
On the validation run the Trainer is given an empty test list (no test patient is landmarked);
--split test requires --final. Writes metrics.json and serial_tables.md to --out-dir.
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.bootstrap import paired_bootstrap
from compass.evaluation.calibration import calibration_all
from compass.evaluation.pipeline import outcome_counts
from compass.evaluation.serial import (
    LANDMARK_COX,
    MIN_EVENTS,
    SERIAL_HORIZON,
    SERIAL_POINTS,
    censoring_reference,
    checkpoint_hashes,
    file_sha256,
    fit_landmark_cox,
    history_lengths,
    ipcw_diagnostics,
    predict_examples,
    predict_landmark_cox,
    serial_ground_truth,
    serial_inputs,
)
from compass.models.base import OUTCOME_NAMES
from compass.models.registry import MODEL_REGISTRY, build_model
from compass.training.train import load_selected_models

_spec = importlib.util.spec_from_file_location("evaluate_script", Path(__file__).with_name("evaluate.py"))
evaluate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_script)

log = logging.getLogger("serial_evaluate")
DEFAULT_FIT_PATH = "runs/serial/landmark_cox.json"


def predictor_order(names: list[str]) -> list[str]:
    return [LANDMARK_COX] + [f"{m}_snapshot" for m in names] + [f"{m}_full" for m in names]


def load_model(name: str, model_dir, num_features: int, horizon: int):
    """Final checkpoint weights (one model, or {outcome: model}) and the model's batch size."""
    with open(MODEL_REGISTRY[name][1]) as f:
        cfg = yaml.safe_load(f)
    return load_selected_models(build_model(name, cfg, num_features, horizon), Path(model_dir)), int(cfg["batch_size"])


def check_fit_file(fits: dict, split: dict, points: list[int]) -> str | None:
    if fits.get("fitted_on_split") != "train":
        return "it was not fitted on the training split"
    if fits.get("split_seed") != int(split["seed"]) or fits.get("data_fingerprint") != split["fingerprint"]:
        return "different split seed or data"
    if fits.get("horizon_years") != SERIAL_HORIZON:
        return "different horizon"
    missing = [k for k in points if str(k) not in fits.get("fits", {})]
    return f"no fit for prediction point(s) {missing}" if missing else None


def _ci(r: dict) -> str:
    if r["point_estimate"] is None or (isinstance(r["point_estimate"], float) and np.isnan(r["point_estimate"])):
        return "n/a"
    return f"{r['point_estimate']:.3f} ({r['ci_lower']:.3f} to {r['ci_upper']:.3f})"


def _excludes_zero(r: dict) -> bool:
    lo, hi = r["ci_lower"], r["ci_upper"]
    return lo is not None and hi is not None and not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0)


def format_point(k: int, res: dict, names: list[str]) -> str:
    c, d = res["counts"], res["ipcw"]
    h = res["history_visits"]
    lines = [f"## Prediction point k = {k} yr (horizon {res['horizon_years']} yr)", "",
             f"Patients evaluated: {res['n_evaluated']} of {res['n_split']} | visits in the full history "
             f"(incl. landmark): median {h['median']:g} (IQR {h['q1']:g}-{h['q3']:g}) | "
             f"IPCW reference: train at k = {k} ({res['n_train_evaluated']} patients)", "",
             "| Outcome | At risk | Events ≤5 yr | Competing ≤5 yr | G(5 yr) | Max IPCW weight 1/G(t-) | Flag |",
             "|---|---|---|---|---|---|---|"]
    for o in OUTCOME_NAMES:
        flag = f"< {MIN_EVENTS} events" if c[o]["n_events"] < MIN_EVENTS else ""
        lines.append(f"| {o} | {c[o]['n_at_risk']} | {c[o]['n_events']} | {c[o]['n_competing']} | "
                     f"{d[o]['G_at_horizon']:.3f} | {d[o]['max_ipcw_weight']:.2f} | {flag} |")
    pm = {(r["model"], r["outcome"], r["metric_name"]): r for r in res["per_model"]}
    lines += ["", "| Predictor | Outcome | C-index (95% CI) | IBS (95% CI) | O/E |", "|---|---|---|---|---|"]
    for m in predictor_order(names):
        for o in OUTCOME_NAMES:
            oe = res["calibration"][m][o]["overall"]["oe_ratio"]
            lines.append(f"| {m} | {o} | {_ci(pm[(m, o, 'cindex')])} | {_ci(pm[(m, o, 'ibs')])} | "
                         f"{'n/a' if oe is None or np.isnan(oe) else f'{oe:.2f}'} |")
    pw = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in res["pairwise"]}
    for title, comp in (("2a: deep FULL minus landmark Cox", lambda m: f"{m}_full_minus_{LANDMARK_COX}"),
                        ("2b: deep FULL minus its own SNAPSHOT", lambda m: f"{m}_full_minus_{m}_snapshot")):
        lines += ["", f"{title} (* = 95% CI excludes 0; ΔIBS > 0 = full history worse)", "",
                  "| Model | Outcome | ΔC-index (95% CI) | ΔIBS (95% CI) |", "|---|---|---|---|"]
        for m in names:
            for o in OUTCOME_NAMES:
                a, b = pw[(comp(m), o, "cindex")], pw[(comp(m), o, "ibs")]
                lines.append(f"| {m} | {o} | {_ci(a)}{' *' if _excludes_zero(a) else ''} | "
                             f"{_ci(b)}{' *' if _excludes_zero(b) else ''} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="Required for --split test (one-time final reporting).")
    p.add_argument("--final-models", default="configs/final_models.yaml")
    p.add_argument("--data", default=None, help="Processed real-data dir. Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only.")
    p.add_argument("--points", type=int, nargs="+", default=list(SERIAL_POINTS))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=None, help="Defaults to configs/training.yaml's seed.")
    p.add_argument("--baseline-config", default="configs/baseline.yaml", help="Gives the fixed ridge penalty.")
    p.add_argument("--landmark-cox", default=DEFAULT_FIT_PATH,
                   help="Landmark Cox fits: written by the validation run, loaded (never refitted) by the test run.")
    p.add_argument("--out-dir", default="outputs/serial_val")
    args = p.parse_args(argv)

    if args.split == "test":
        if not args.final:
            print("refusing: --split test requires --final (the test split is for ONE-TIME final reporting)", file=sys.stderr)
            return 2
        if not Path(args.landmark_cox).exists():
            print(f"refusing: {args.landmark_cox} not found -- run the validation split first "
                  "(the test run never fits the landmark Cox)", file=sys.stderr)
            return 3
        log.warning("Evaluating the TEST split. One-time final reporting only.")

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)
    with open(args.baseline_config) as f:
        penalizer = float(yaml.safe_load(f)["penalizer"])
    if args.data:
        visits_df = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes_df = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        seed = args.data_seed if args.data_seed is not None else training_config["seed"]
        visits_df, outcomes_df = generate_synthetic_cohort(n_patients=args.n_patients, seed=seed)

    ns = argparse.Namespace(final_models=args.final_models, checkpoint_dir=None, models=None)
    try:
        names, model_dirs, split = evaluate_script.resolve_models(ns, visits_df, outcomes_df)
    except (evaluate_script.SplitMismatchError, evaluate_script.FinalModelsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3

    saved_fits = None
    if args.split == "test":
        saved_fits = json.loads(Path(args.landmark_cox).read_text())
        problem = check_fit_file(saved_fits, split, args.points)
        if problem:
            print(f"refusing: landmark Cox {args.landmark_cox}: {problem}", file=sys.stderr)
            return 3

    hashes_before = checkpoint_hashes(model_dirs)
    # only the evaluated split's IDs (plus train) are given to the Trainer
    trainer_split = {"train": split["train"], "val": split["val"] if args.split == "val" else [],
                     "test": split["test"] if args.split == "test" else [], "seed": split["seed"]}
    trainer, _ = evaluate_script.make_trainer(names[0], visits_df, outcomes_df, schema, training_config, split=trainer_split)
    primary_horizon = int(training_config["horizons"]["primary"])
    models, batch_sizes = {}, {}
    for m in names:
        models[m], batch_sizes[m] = load_model(m, model_dirs[m], len(trainer.feature_names), primary_horizon)

    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(training_config["seed"])
    eval_ids = list(split[args.split])
    fits = {} if saved_fits is None else {int(k): v for k, v in saved_fits["fits"].items()}
    points = {}
    for k in args.points:
        log.info("prediction point k = %d: building landmark examples", k)
        train_inputs = serial_inputs(trainer, list(split["train"]), k)
        if saved_fits is None:
            fits[k] = fit_landmark_cox(trainer, train_inputs["snapshot"], penalizer, SERIAL_HORIZON)
            fits[k]["n_train_patients"] = len(train_inputs["ids"])
        reference = censoring_reference(serial_ground_truth(trainer, train_inputs["full"]))
        n_train_evaluated = len(train_inputs["ids"])
        del train_inputs

        inputs = serial_inputs(trainer, eval_ids, k)
        gt = serial_ground_truth(trainer, inputs["full"])
        gt_snap = serial_ground_truth(trainer, inputs["snapshot"])
        if any(not np.array_equal(gt[o][f], gt_snap[o][f]) for o in OUTCOME_NAMES for f in gt[o]):
            raise RuntimeError("snapshot and full-history examples do not share the same targets")

        predictions = {LANDMARK_COX: predict_landmark_cox(fits[k], trainer, inputs["snapshot"], SERIAL_HORIZON)}
        for kind in ("snapshot", "full"):
            for m in names:
                predictions[f"{m}_{kind}"] = predict_examples(models[m], trainer, inputs[kind], SERIAL_HORIZON, batch_sizes[m])
        predictions = {n: predictions[n] for n in predictor_order(names)}

        log.info("k = %d: %d patients evaluated; bootstrap (%d draws, 7 predictors)", k, len(inputs["ids"]), args.n_bootstrap)
        per_model, pairwise = paired_bootstrap(predictions, gt, reference, SERIAL_HORIZON, n_iterations=args.n_bootstrap,
                                               seed=bootstrap_seed)
        counts = outcome_counts(gt, SERIAL_HORIZON)
        points[str(k)] = {
            "horizon_years": SERIAL_HORIZON,
            "n_split": len(eval_ids),
            "n_evaluated": len(inputs["ids"]),
            "n_train_evaluated": n_train_evaluated,
            "ipcw_reference": {"split": "train", "prediction_point": k,
                               "n_at_risk": {o: int(len(v["time_years"])) for o, v in reference.items()}},
            "counts": counts,
            "flags": [o for o in OUTCOME_NAMES if counts[o]["n_events"] < MIN_EVENTS],
            "ipcw": ipcw_diagnostics(reference, gt, SERIAL_HORIZON),
            "history_visits": history_lengths(inputs["full"]),
            "per_model": per_model.to_dict(orient="records"),
            "pairwise": pairwise.to_dict(orient="records"),
            "calibration": calibration_all(predictions, gt, SERIAL_HORIZON),
        }

    if saved_fits is None:
        fit_file = {"fitted_on_split": "train", "split_seed": int(split["seed"]), "data_fingerprint": split["fingerprint"],
                    "horizon_years": SERIAL_HORIZON, "penalizer": penalizer,
                    "follow_up": f"truncated at {SERIAL_HORIZON} yr (later events / competing deaths censored there)",
                    "fits": {str(k): v for k, v in fits.items()}}
        Path(args.landmark_cox).parent.mkdir(parents=True, exist_ok=True)
        Path(args.landmark_cox).write_text(json.dumps(evaluate_script._jsonable(fit_file), indent=2))

    hashes_after = checkpoint_hashes(model_dirs)
    if hashes_after != hashes_before:
        print("error: a deep-model checkpoint changed during the run", file=sys.stderr)
        return 5

    results = {
        "analysis": "secondary analysis 2: serial evaluation (2a landmark Cox, 2b history ablation)",
        "split": args.split,
        "data": "real (processed)" if args.data else "synthetic",
        "horizon_years": SERIAL_HORIZON,
        "prediction_points": list(args.points),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
        "predictors": predictor_order(names),
        "model_sources": {m: os.path.normpath(str(model_dirs[m])) for m in names},
        "checkpoint_sha256": hashes_before,
        "checkpoints_unchanged": True,
        "landmark_cox": {"path": os.path.normpath(args.landmark_cox), "sha256": file_sha256(args.landmark_cox),
                         "fitted_in_this_run": saved_fits is None,
                         "per_point": {str(k): {o: {kk: f["outcomes"][o].get(kk) for kk in
                                                    ("method", "n", "n_events", "n_competing", "warnings")}
                                                for o in OUTCOME_NAMES} | {
                                           "n_train_patients": f.get("n_train_patients"),
                                           "dropped_zero_variance": f["transform"]["dropped_zero_variance"],
                                           "n_columns": len(f["transform"]["columns"])}
                                       for k, f in fits.items()}},
        "points": points,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = (f"# Serial evaluation ({args.split}) -- horizon {SERIAL_HORIZON} yr, {args.n_bootstrap} paired bootstrap draws "
              f"(seed {bootstrap_seed}); deep checkpoints unchanged (sha256 checked)")
    text = "\n\n".join([header] + [format_point(int(k), r, names) for k, r in points.items()])
    (out / "serial_tables.md").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(evaluate_script._jsonable(results), indent=2))
    print(text)
    print(f"\nwrote {out}/metrics.json, {out}/serial_tables.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

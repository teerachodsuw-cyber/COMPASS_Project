#!/usr/bin/env python
"""Export per-patient risk-band assignments on the TEST split for the risk-band curves (Fig 4).
Inference only: no metric, calibration or cut-point is computed or refitted here.

Usage:
    python scripts/export_band_assignments.py --final --final-models configs/final_models.yaml \\
        --data data/processed --cutpoints outputs/final_val/cutpoints.json \\
        --check-against outputs/final_test/metrics.json --out runs/final_test/band_assignments.csv

Steps: the final models (configs/final_models.yaml; per-outcome checkpoints, locked settings
checked) predict the 10-yr risk on the test split's fixed landmarks; the SAVED validation
cut-points (load_cutpoints: same split seed, data, horizon and model sources) assign each
at-risk patient to a band; the ground truth (time, event type) comes from build_targets, the
same as the evaluation.

Before anything is written, every band's n and events within the horizon are checked against
the saved test evaluation (--check-against, the final test run's metrics.json), and the
cut-points against the ones that run used -- any difference refuses the export.

Output (patient-level, so it goes to the git-ignored runs/ directory; NO patient IDs): one row
per model x outcome x at-risk patient: model, outcome, band (0 low / 1 intermediate / 2 high),
time_years, event_type (1 event, 2 competing death, 0 censored).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.evaluation.calibration import at_risk_horizon_data
from compass.evaluation.pipeline import collect_predictions
from compass.evaluation.risk_banding import (
    BAND_NAMES,
    CutpointsMismatchError,
    assign_bands,
    load_cutpoints,
    truncate_at_horizon,
)
from compass.models.base import OUTCOME_NAMES

_spec = importlib.util.spec_from_file_location("evaluate_script", Path(__file__).with_name("evaluate.py"))
evaluate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_script)


class ExportMismatchError(ValueError):
    """The re-derived bands do not reproduce the saved test evaluation."""


def check_against_saved(rows: pd.DataFrame, saved_results: dict, cutpoints: dict, horizon: int) -> None:
    """Per model / outcome / band: n and events within the horizon must equal the saved test
    run's banding table, and the cut-points must equal the ones it used."""
    if saved_results.get("split") != "test":
        raise ExportMismatchError("--check-against is not a test-split evaluation")
    if saved_results["banding"]["cutpoints"] != cutpoints:
        raise ExportMismatchError("cut-points differ from the ones the saved test evaluation used")
    for model, per in saved_results["banding"]["bands"].items():
        for outcome, bands in per.items():
            if isinstance(bands, dict):
                continue
            sub = rows[(rows["model"] == model) & (rows["outcome"] == outcome)]
            _, event_type = truncate_at_horizon(sub["time_years"].to_numpy(), sub["event_type"].to_numpy(), horizon)
            for b, saved in enumerate(bands):
                sel = sub["band"].to_numpy() == b
                n, events = int(sel.sum()), int((event_type[sel] == 1).sum())
                if (n, events) != (saved["n"], saved["n_events"]):
                    raise ExportMismatchError(f"{model} / {outcome} / {BAND_NAMES[b]}: re-derived n={n}, events={events}; "
                                              f"saved n={saved['n']}, events={saved['n_events']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="Required: this reads the test split.")
    p.add_argument("--final-models", required=True)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--cutpoints", required=True, help="The validation run's cutpoints.json (applied unchanged).")
    p.add_argument("--check-against", required=True, help="The saved final test evaluation's metrics.json.")
    p.add_argument("--data", default=None, help="Processed real-data dir. Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only.")
    p.add_argument("--out", default="runs/final_test/band_assignments.csv")
    args = p.parse_args(argv)
    if not args.final:
        print("refusing: this reads the test split; pass --final", file=sys.stderr)
        return 2

    with open("configs/data_schema.yaml") as f:
        schema = yaml.safe_load(f)
    with open("configs/training.yaml") as f:
        training_config = yaml.safe_load(f)
    if args.data:
        visits_df = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes_df = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        seed = args.data_seed if args.data_seed is not None else training_config["seed"]
        visits_df, outcomes_df = generate_synthetic_cohort(n_patients=args.n_patients, seed=seed)

    ns = argparse.Namespace(final_models=args.final_models, checkpoint_dir=None, models=args.models)
    try:
        names, model_dirs, split = evaluate_script.resolve_models(ns, visits_df, outcomes_df)
    except (evaluate_script.SplitMismatchError, evaluate_script.FinalModelsError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3
    horizon = int(training_config["horizons"]["primary"])
    meta = {"horizon_years": horizon, "split_seed": int(split["seed"]), "data_fingerprint": split["fingerprint"],
            "model_sources": {m: evaluate_script.os.path.normpath(str(model_dirs[m])) for m in names}}
    try:
        cutpoints = load_cutpoints(args.cutpoints, meta, names)
    except CutpointsMismatchError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 4

    trainers, models = {}, {}
    for m in names:
        trainers[m], models[m] = evaluate_script.load_trained_from(m, model_dirs[m], split, visits_df, outcomes_df,
                                                                   schema, training_config)
    _, ground_truth, predictions = collect_predictions(trainers, models, "test", horizon)

    parts = []
    for m in names:
        for o in OUTCOME_NAMES:
            fit = cutpoints[m][o]
            if fit.get("status") != "fitted":
                continue
            scores, time, event_type = at_risk_horizon_data(predictions[m][o], ground_truth, o, horizon)
            parts.append(pd.DataFrame({"model": m, "outcome": o, "band": assign_bands(scores, fit["cutpoints"]),
                                       "time_years": time, "event_type": event_type.astype(int)}))
    rows = pd.concat(parts, ignore_index=True)

    saved = json.loads(Path(args.check_against).read_text())
    try:
        check_against_saved(rows, saved, cutpoints, horizon)
    except ExportMismatchError as exc:
        print(f"refusing to write: {exc}", file=sys.stderr)
        return 5

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False)
    counts = rows.groupby(["model", "outcome"]).size()
    print(f"bands match the saved test evaluation for all {len(counts)} model x outcome pairs "
          f"(n and events per band, and cut-points)")
    print(f"wrote {out} ({len(rows)} rows; no patient IDs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

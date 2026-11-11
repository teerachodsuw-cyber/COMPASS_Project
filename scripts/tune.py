#!/usr/bin/env python
"""Grid tuning of learning rate x weight decay per model, on the VALIDATION split only.

Usage:
    python scripts/tune.py --dry-run                       # list configs + time estimate, no data loaded
    python scripts/tune.py --data data/processed           # real data
    python scripts/tune.py --models dynamic_deephit tracer # subset of models

Grid, run/out dirs and the dry-run speed reference: configs/tuning.yaml. Max epochs and
early-stopping patience: configs/training.yaml (training_loop), the same for every config.
One pinned split for every config (<run_dir>/split.json; --split-from <training run dir> to
reuse an existing run's split). Resumable: configs with a result.json are skipped. The test
split is never loaded. Writes <out_dir>/summary.csv and summary.md, marking each model's
winner (lowest best total val loss).
"""

import argparse
import sys

import pandas as pd

from compass.data.synthetic_generator import generate_synthetic_cohort
from compass.models.registry import MODEL_REGISTRY
from compass.training import tuning


def _fmt_s(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tuning-config", default="configs/tuning.yaml")
    p.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the configs and a run-time estimate; train nothing.")
    p.add_argument("--data", default=None, help="Processed real-data dir. Omit for synthetic.")
    p.add_argument("--n-patients", type=int, default=1000, help="Synthetic only.")
    p.add_argument("--data-seed", type=int, default=None, help="Synthetic only.")
    p.add_argument("--split-from", default=None, help="Re-use the split saved in this training run dir.")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-epochs", type=int, default=None, help="Override training_loop.max_epochs (all configs).")
    p.add_argument("--patience", type=int, default=None, help="Override training_loop.early_stopping_patience.")
    args = p.parse_args(argv)

    tuning_cfg = tuning.load_yaml(args.tuning_config)
    training_cfg = tuning.load_yaml("configs/training.yaml")
    models = args.models or tuning_cfg["models"]
    run_dir = args.run_dir or tuning_cfg["run_dir"]
    out_dir = args.out_dir or tuning_cfg["out_dir"]
    loop = training_cfg["training_loop"]
    max_epochs = args.max_epochs if args.max_epochs is not None else int(loop["max_epochs"])
    patience = args.patience if args.patience is not None else loop.get("early_stopping_patience")

    base_lr = {m: tuning.load_yaml(MODEL_REGISTRY[m][1])["learning_rate"] for m in models}
    entries = tuning.expand_grid(tuning_cfg, models, base_lr)

    if args.dry_run:
        est = tuning.estimate(entries, tuning_cfg, max_epochs, patience)
        print(f"{len(entries)} configs ({len(entries) // len(models)} per model) | max epochs {max_epochs}, "
              f"patience {patience} | same pinned split + seed for all | run dir {run_dir} | out dir {out_dir}")
        print("\n| # | Model | Config id | Learning rate | Weight decay | s/epoch (v1) | Est. typical | Est. range |")
        print("|---|---|---|---|---|---|---|---|")
        for i, e in enumerate(est, 1):
            done = " (finished, skip)" if tuning.finished_result(run_dir, e) else ""
            print(f"| {i} | {e['model']} | {e['config_id']}{done} | {e['learning_rate']:.3g} | {e['weight_decay']:.3g} | "
                  f"{e['s_per_epoch']:.1f} | {_fmt_s(e['typical_s'])} | {_fmt_s(e['min_s'])}-{_fmt_s(e['max_s'])} |")
        print("\nEstimated total (v1 speeds; typical = v1 epochs before early stopping; range = patience+1 .. max epochs):")
        for m in models:
            rows = [e for e in est if e["model"] == m]
            print(f"  {m}: typical {_fmt_s(sum(r['typical_s'] for r in rows))} "
                  f"(range {_fmt_s(sum(r['min_s'] for r in rows))}-{_fmt_s(sum(r['max_s'] for r in rows))})")
        print(f"  ALL: typical {_fmt_s(sum(r['typical_s'] for r in est))} "
              f"(range {_fmt_s(sum(r['min_s'] for r in est))}-{_fmt_s(sum(r['max_s'] for r in est))})")
        print("  (+ ~10-20 s setup per config; lower learning rates tend to run more epochs before early stopping)")
        return 0

    schema = tuning.load_yaml("configs/data_schema.yaml")
    if args.data:
        visits_df = pd.read_parquet(f"{args.data}/cohort_visits.parquet")
        outcomes_df = pd.read_parquet(f"{args.data}/cohort_outcomes.parquet")
    else:
        seed = args.data_seed if args.data_seed is not None else training_cfg["seed"]
        visits_df, outcomes_df = generate_synthetic_cohort(n_patients=args.n_patients, seed=seed)

    split = tuning.resolve_split(run_dir, visits_df, outcomes_df, training_cfg, args.split_from)
    print(f"pinned split: {len(split['train'])} train / {len(split['val'])} val (test not loaded); seed {split['seed']}")
    rows = tuning.run_grid(entries, visits_df, outcomes_df, schema, training_cfg, split, run_dir, out_dir,
                           max_epochs, patience)
    print("\n" + tuning.write_summary(rows, out_dir))
    print(f"\nwrote {out_dir}/summary.csv and {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

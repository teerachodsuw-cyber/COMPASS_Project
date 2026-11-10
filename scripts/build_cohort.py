#!/usr/bin/env python
"""Build the real-data analytic cohort (loader steps 1-7).

Usage:
    python scripts/build_cohort.py                       # defaults below
    python scripts/build_cohort.py --dry-run             # steps 1-6 only, writes nothing

1. load the .dta (convert_dates=True, convert_categoricals=False)
2-5. build_cohort(): exclusions, visit cleaning/placement, landmark eligibility, outcomes
6. validate_dataset() on the outputs -- nothing is written unless it passes
7. write cohort_visits.parquet, cohort_outcomes.parquet, flow_log.json (aggregate counts)

Outputs go only to data/processed/ (git-ignored). Prints aggregates only.
"""

import argparse
import json
import sys
from pathlib import Path

from compass.data.loader import build_cohort, load_stata, load_yaml
from compass.data.validate_schema import SchemaValidationError, validate_dataset

PROCESSED_DIR = Path("data/processed")


def format_flow_table(flow: dict) -> str:
    lines = [
        "| Step | Description | Removed | Remaining | Expected removed |",
        "|---|---|---|---|---|",
        f"| start | Patients in source file ({flow['source']['rows']:,} rows) | | {flow['source']['patients']:,} | |",
    ]
    for e in flow["exclusions"]:
        exp = f"{e['expected_removed']:,}" if "expected_removed" in e else ""
        lines.append(f"| {e['id']} | {e['label']} | {e['removed']:,} | {e['remaining']:,} | {exp} |")
    a = flow["analytic_n"]
    lines.append(f"| | Analytic n after E1-E4 | | {a['observed']:,} | expected {a['expected']:,} |" if a["expected"] else "")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dta", default="data/raw/MASLD_analysis_v1_20261105.dta")
    parser.add_argument("--mapping", default="configs/real_data_mapping.yaml")
    parser.add_argument("--schema", default="configs/data_schema.yaml")
    parser.add_argument("--out", default=str(PROCESSED_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Run steps 1-6 and print; write nothing.")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out != PROCESSED_DIR.resolve() and PROCESSED_DIR.resolve() not in out.parents:
        print(f"refusing to write outside {PROCESSED_DIR}/ (git-ignored): {args.out}", file=sys.stderr)
        return 2

    mapping, schema = load_yaml(args.mapping), load_yaml(args.schema)
    visits_df, outcomes_df, flow = build_cohort(load_stata(args.dta), mapping, schema)

    print(format_flow_table(flow))
    v = flow["visits"]
    print(f"\nvisit rows in cohort {v['rows_in_cohort']:,}; dropped after death {v['dropped_visit_after_death']:,}; "
          f"dropped after cutoff {v['dropped_after_cutoff']:,}")
    print("final:", json.dumps(flow["final"]))
    print("% missing per feature after placement:", json.dumps(flow["missing_pct_after_placement"]))

    try:
        validate_dataset(visits_df, outcomes_df, config_path=args.schema)
    except SchemaValidationError as exc:
        print(f"\nvalidate_dataset FAILED -- nothing written: {exc}", file=sys.stderr)
        return 1
    print("\nvalidate_dataset: PASSED")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    visits_df.to_parquet(out / "cohort_visits.parquet", index=False)
    outcomes_df.to_parquet(out / "cohort_outcomes.parquet", index=False)
    (out / "flow_log.json").write_text(json.dumps(flow, indent=2))
    print(f"wrote {out}/cohort_visits.parquet, cohort_outcomes.parquet, flow_log.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

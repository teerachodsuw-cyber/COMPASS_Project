"""scripts/make_table1.py on synthetic data: it runs, writes the three outputs, the split columns
add up to the overall column, and the locked-number cross-check stops a run on other data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compass.data.landmarking import landmark_candidate_mask
from compass.data.splitting import data_fingerprint
from compass.data.synthetic_generator import generate_synthetic_cohort

ROOT = Path(__file__).resolve().parents[1]
PREV = {"cad": "cad_prev", "stroke": "stroke_prev", "hf": "hf_prev", "pvd": "pvd_prev"}


def _load_script():
    spec = importlib.util.spec_from_file_location("make_table1", ROOT / "scripts" / "make_table1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_inputs(tmp_path: Path) -> list[str]:
    visits, outcomes = generate_synthetic_cohort(300, seed=11, malo_rate=0.10)
    # Keep patients with a valid landmark (the loader's E5), as in the real cohort.
    death = outcomes.set_index("patient_id")["mortality_date"].where(outcomes.set_index("patient_id")["mortality_event"] == 1)
    ok = landmark_candidate_mask(visits["visit_date"], visits["patient_id"].map(outcomes.set_index("patient_id")["index_date"]),
                                 visits["patient_id"].map(death))
    keep = set(visits.loc[ok, "patient_id"])
    visits = visits[visits["patient_id"].isin(keep)].reset_index(drop=True)
    outcomes = outcomes[outcomes["patient_id"].isin(keep)].reset_index(drop=True)

    data = tmp_path / "processed"
    data.mkdir()
    visits.to_parquet(data / "cohort_visits.parquet")
    outcomes.to_parquet(data / "cohort_outcomes.parquet")

    rng = np.random.default_rng(0)
    ids = rng.permutation(outcomes["patient_id"].to_numpy()).tolist()
    n_tr, n_va = int(0.7 * len(ids)), int(0.15 * len(ids))
    split = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:], "seed": 0,
             "fingerprint": data_fingerprint(visits, outcomes)}
    (tmp_path / "split.json").write_text(json.dumps(split))

    # Raw-file stand-in with the CVD prevalence flags, consistent with cvd_at_risk.
    prior = outcomes["cvd_at_risk"].to_numpy() == 0
    which = rng.integers(0, len(PREV), size=len(outcomes))
    raw = pd.DataFrame({"person_id": outcomes["patient_id"].astype(str)})
    for k, col in enumerate(PREV.values()):
        raw[col] = (prior & (which == k)).astype(np.int8)
    raw.to_stata(tmp_path / "raw.dta", write_index=False)

    return ["--data", str(data), "--split-file", str(tmp_path / "split.json"), "--raw", str(tmp_path / "raw.dta")]


def test_table1_runs_and_column_totals_add_up(tmp_path):
    script = _load_script()
    args = _synthetic_inputs(tmp_path)
    out = tmp_path / "tables"

    # The locked numbers belong to the real cohort: on synthetic data the run must stop, writing nothing.
    assert script.main(args + ["--out-dir", str(tmp_path / "locked")]) == 4
    assert not (tmp_path / "locked").exists()

    assert script.main(args + ["--out-dir", str(out), "--skip-locked-check"]) == 0
    for name in ("table1.docx", "table1.csv", "table1.md"):
        assert (out / name).stat().st_size > 0

    counts = json.loads((out / "table1_counts.json").read_text())
    outcomes = pd.read_parquet(tmp_path / "processed" / "cohort_outcomes.parquet")
    overall, splits = counts["overall"], [counts[c] for c in ("train", "val", "test")]
    assert overall["n"] == len(outcomes)
    for key in ("n", "male", "visits", "mortality_events", "malo_events", "cvd_events", "cvd_pool",
                "mortality_at_risk", "malo_at_risk", "cvd_at_risk",
                "mortality_events_after_entry", "malo_events_after_entry", "cvd_events_after_entry"):
        assert sum(s[key] for s in splits) == overall[key], key
    for level in overall["smoking"]:
        assert sum(s["smoking"][level] for s in splits) == overall["smoking"][level], level
    for col in [overall, *splits]:
        assert sum(col["smoking"].values()) == col["n"]  # never / former / current / missing cover everyone
    assert overall["cvd_pool"] == int(outcomes["cvd_at_risk"].sum())
    for name in ("mortality", "malo", "cvd"):
        assert overall[f"{name}_events"] == int(outcomes[f"{name}_event"].sum())
        assert overall[f"{name}_events_after_entry"] <= overall[f"{name}_events"]
    md = (out / "table1.md").read_text()
    assert f"Of these, {overall['malo_events_after_entry']:,} MALO and {overall['cvd_events_after_entry']:,} CVD events" in md
    assert "Observed follow-up, index to last visit, years, median (IQR)" in md

    # The CSV and the Word table carry the same header and the same number of rows.
    from docx import Document

    csv = pd.read_csv(out / "table1.csv")
    table = Document(out / "table1.docx").tables[0]
    assert [c.text for c in table.rows[0].cells][1:] == list(csv.columns[2:])
    assert len(table.rows) == len(csv) + 1
    assert f"(n = {overall['n']:,})" in csv.columns[2]

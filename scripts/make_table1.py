#!/usr/bin/env python
"""Table 1: baseline characteristics, overall and by the pinned data split (aggregates only).
Reads the processed cohort and the pinned split; never touches models, results or analysis code.

Usage:
    python scripts/make_table1.py                       # defaults below = the real cohort
    python scripts/make_table1.py --data data/processed --split-file runs/tune_v2/split.json \\
        --raw data/raw/MASLD_analysis_v1_20261105.dta --out-dir outputs/tables

Definitions (also printed in the table footnote):
- Cohort entry = the patient's FIRST valid landmark (compass.data.landmarking.first_landmark:
  the earliest visit on/after index_date and before death), as in the evaluation.
- Characteristics = each variable's most recent observed value on/before entry (LOCF over the
  visits up to and including the entry visit; missing if never observed). Age is the static
  baseline variable index_age (age at index_date), exactly as the models receive it.
- Prior CVD = the source's prevalent flags (cad/stroke/hf/pvd_prev), read from the raw file
  named in configs/real_data_mapping.yaml (they are not kept in data/processed); "any" must
  equal cvd_at_risk == 0 for every patient (checked).
- Outcomes and follow-up are counted from index_date (the cohort definition: events after
  index_date and on/before the cutoff), with event / competing / censoring times from
  compass.training.targets.build_targets (landmark = index_date, no window cap):
  mortality censored at the cutoff; MALO / CVD censored at min(last visit, cutoff), death
  before that competing. 10-yr cumulative incidence: Kaplan-Meier (mortality), Aalen-Johansen
  with death competing (MALO, CVD; CVD within the at-risk pool only).
- Follow-up: potential follow-up to the cutoff by reverse Kaplan-Meier (mortality follow-up,
  NHSO linkage); observed follow-up = index_date to the last visit (MALO / CVD follow-up).
- Continuous: median (IQR). Categorical: n (%) of the column total. Missing n (%) for every
  variable.

Cross-check (unless --skip-locked-check, for synthetic data only): the n per column and the
event counts must equal the locked numbers (LOCKED below); otherwise nothing is written and the
script exits 4.

Outputs in --out-dir (git-ignored): table1.docx (Word table, no colours, bold title,
footnote), table1.csv, table1.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from compass.data.landmarking import landmark_candidate_mask
from compass.data.splitting import check_fingerprint, data_fingerprint
from compass.evaluation.calibration import aalen_johansen_cif
from compass.training.targets import DAYS_PER_YEAR, build_targets, compute_last_visit_dates

PID = "patient_id"
COLUMNS = ("overall", "train", "val", "test")
COLUMN_TITLE = {"overall": "Overall", "train": "Training", "val": "Validation", "test": "Test"}
LOCKED = {
    "n": {"overall": 11655, "train": 8158, "val": 1748, "test": 1749},
    "events": {"mortality": 1379, "malo": 1286, "cvd": 1705},
    "events_after_entry": {"malo": 1207, "cvd": 1656},
    "cvd_pool": 9302,
}
OUTCOMES = ("mortality", "malo", "cvd")
OUTCOME_LABEL = {"mortality": "All-cause mortality", "malo": "MALO", "cvd": "CVD"}
NO_WINDOW_BINS = 100  # build_targets window = 100 yr: no cap at any real follow-up

# (variable, label, decimals)
CONTINUOUS = {
    "Demographics": [("index_age", "Age, years", 0)],
    "Anthropometrics and blood pressure": [
        ("height", "Height, cm", 1), ("weight", "Weight, kg", 1), ("bmi", "BMI, kg/m²", 1),
        ("sbp", "SBP, mmHg", 0), ("dbp", "DBP, mmHg", 0),
    ],
    "Laboratory values": [
        ("fbs", "FBS, mg/dL", 0), ("hba1c", "HbA1c, %", 1), ("cholesterol", "Total cholesterol, mg/dL", 0),
        ("triglyceride", "Triglycerides, mg/dL", 0), ("hdl_cholesterol", "HDL-C, mg/dL", 0),
        ("ldl_cholesterol", "LDL-C, mg/dL", 0), ("plt", "Platelets, ×10³/µL", 0),
        ("serum_creatinine", "Serum creatinine, mg/dL", 2), ("egfr", "eGFR, mL/min/1.73 m²", 0),
        ("alt", "ALT, U/L", 0), ("ast", "AST, U/L", 0),
    ],
    "FibroScan": [("e_median", "LSM, kPa", 1), ("cap_median", "CAP, dB/m", 0)],
}
MEDICATIONS = [
    ("metformin", "Metformin"), ("sulfonylureas", "Sulfonylureas"), ("pioglitazone", "Pioglitazone"),
    ("insulin", "Insulin"), ("dpp4", "DPP-4 inhibitors"), ("glp1", "GLP-1 receptor agonists"),
    ("sglt2", "SGLT2 inhibitors"), ("acei", "ACE inhibitors"), ("arb", "ARBs"), ("arni", "ARNI"),
    ("statins", "Statins"), ("fibrate", "Fibrates"), ("ezetimibe", "Ezetimibe"),
]
SMOKING = [(0, "Never"), (1, "Former"), (2, "Current")]
PRIOR_CVD = [("cad", "CAD"), ("stroke", "Stroke"), ("hf", "Heart failure"), ("pvd", "PVD")]
SNAPSHOT_COLUMNS = (
    [v for group in CONTINUOUS.values() for v, _, _ in group] + [m for m, _ in MEDICATIONS] + ["sex", "smoking_status"]
)

TITLE = "Table 1. Baseline characteristics of the study cohort, overall and by data split"
FOOTNOTE = (
    "Values are median (IQR) or n (%); percentages use the column total as the denominator, with missing "
    "values shown separately for every variable. Patients were split at random at the patient level "
    "(training 70%, validation 15%, test 15%), stratified on the three outcome indicators; no statistical "
    "tests were performed. "
    "Characteristics are each variable's most recent observed value on or before cohort entry (the first "
    "visit on or after the index date and before death); age is age at the index date. Height, weight, BMI "
    "and blood pressure were recorded once, at the first visit on or after the index date. Prior CVD: "
    "prevalent CAD, stroke, heart failure or PVD at the index date (source prevalence flags); these "
    "patients form the group not at risk for incident CVD. Potential follow-up: from the index date to the "
    "administrative cutoff (30 June 2025), median (IQR) by reverse Kaplan-Meier; mortality follow-up uses "
    "NHSO registry linkage to the cutoff. Observed follow-up: from the index date to the last visit; MALO "
    "and CVD follow-up ends at the last visit. Visits per patient: all laboratory visits in the analytic "
    "dataset, including visits before the index date. Outcomes: first event after the index date and on or "
    "before the cutoff. Of these, {malo_after} MALO and {cvd_after} CVD events occurred after the first "
    "valid landmark and were predictable by the models. MALO: compensated cirrhosis, decompensated "
    "cirrhosis, HCC or liver transplantation. CVD: CAD, stroke, heart failure or PVD, in patients without "
    "prior CVD. 10-year cumulative incidence: Kaplan-Meier for mortality; Aalen-Johansen with death as a "
    "competing event for MALO and CVD, with follow-up censored at the earlier of the last visit and the "
    "cutoff. "
    "ACE, angiotensin-converting enzyme; ALT, alanine aminotransferase; ARB, angiotensin II receptor "
    "blocker; ARNI, angiotensin receptor-neprilysin inhibitor; AST, aspartate aminotransferase; BMI, body "
    "mass index; CAD, coronary artery disease; CAP, controlled attenuation parameter; CVD, cardiovascular "
    "disease; DBP, diastolic blood pressure; DPP-4, dipeptidyl peptidase-4; eGFR, estimated glomerular "
    "filtration rate; FBS, fasting blood sugar; GLP-1, glucagon-like peptide-1; HbA1c, glycated "
    "haemoglobin; HCC, hepatocellular carcinoma; HDL-C, high-density lipoprotein cholesterol; IQR, "
    "interquartile range; LDL-C, low-density lipoprotein cholesterol; LSM, liver stiffness measurement; "
    "MALO, major adverse liver outcomes; PVD, peripheral vascular disease; SBP, systolic blood pressure; "
    "SGLT2, sodium-glucose cotransporter-2."
)


class LockedMismatch(ValueError):
    """A column n or event count differs from the locked numbers."""


# ---- inputs ---------------------------------------------------------------------------------


def load_split_ids(split_file: str, visits: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """{train, val, test: [ids]} from the pinned split; the data must match its fingerprint and
    the three splits must partition the cohort."""
    split = json.loads(Path(split_file).read_text())
    check_fingerprint(split["fingerprint"], data_fingerprint(visits, outcomes))
    ids = {k: list(split[k]) for k in ("train", "val", "test")}
    sets = [set(v) for v in ids.values()]
    cohort = set(outcomes[PID])
    if sum(map(len, sets)) != len(set().union(*sets)) or set().union(*sets) != cohort:
        raise ValueError("the pinned split does not partition the cohort (overlapping or missing patients)")
    return ids


def prior_cvd_flags(raw_path: str, mapping: dict, outcomes: pd.DataFrame) -> pd.DataFrame:
    """{cad, stroke, hf, pvd, any} prevalent flags (0/1, NaN if missing) per cohort patient, from
    the raw file. 'any' must equal cvd_at_risk == 0 (the loader's definition)."""
    pid = mapping["structure"]["patient_id"]
    comps = mapping["outcomes"]["cvd"]["components"]
    cols = {name: comps[name]["prevalent"] for name, _ in PRIOR_CVD}
    raw = pd.read_stata(raw_path, columns=[pid, *cols.values()], convert_dates=False, convert_categoricals=False)
    per_patient = raw.groupby(pid)[list(cols.values())].first().rename(columns={v: k for k, v in cols.items()})
    flags = per_patient.reindex(outcomes[PID].to_numpy())
    if flags.isna().all(axis=1).any():
        raise ValueError(f"{int(flags.isna().all(axis=1).sum())} cohort patient(s) not found in the raw file")
    flags["any"] = (flags[[n for n, _ in PRIOR_CVD]] == 1).any(axis=1).astype(int)
    expected = (outcomes.set_index(PID)["cvd_at_risk"] == 0).astype(int).reindex(flags.index)
    if not (flags["any"] == expected).all():
        raise LockedMismatch(f"prior CVD (any prevalent flag) differs from cvd_at_risk == 0 for {int((flags['any'] != expected).sum())} patient(s)")
    return flags


def entry_dates(visits: pd.DataFrame, outcomes: pd.DataFrame) -> pd.Series:
    """patient_id -> first valid landmark (cohort entry)."""
    o = outcomes.set_index(PID)
    death = o["mortality_date"].where(o["mortality_event"] == 1)
    candidate = landmark_candidate_mask(visits["visit_date"], visits[PID].map(o["index_date"]), visits[PID].map(death))
    entry = visits.loc[candidate].groupby(PID)["visit_date"].min()
    if len(entry) != len(o):
        raise ValueError(f"{len(o) - len(entry)} patient(s) have no valid landmark")
    return entry.reindex(o.index)


def entry_snapshot(visits: pd.DataFrame, outcomes: pd.DataFrame, entry: pd.Series) -> pd.DataFrame:
    """One row per patient: each variable's most recent observed value on/before the first
    valid landmark (cohort entry)."""
    o = outcomes.set_index(PID)
    v = visits.sort_values([PID, "visit_date"]).reset_index(drop=True)
    history = v.loc[v["visit_date"] <= v[PID].map(entry)]
    filled = history.groupby(PID)[SNAPSHOT_COLUMNS].ffill()
    filled[PID] = history[PID].to_numpy()
    return filled.groupby(PID).tail(1).set_index(PID).reindex(o.index)


def outcome_times(visits: pd.DataFrame, outcomes: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """{outcome: DataFrame(time_years, event_type (1 event / 2 competing / 0 censored), at_risk)}
    from index_date, via build_targets with no window cap."""
    ids = outcomes[PID].tolist()
    t = build_targets(ids, outcomes["index_date"].tolist(), outcomes, compute_last_visit_dates(visits), cutoff,
                      1.0, NO_WINDOW_BINS)
    out = {}
    for name in OUTCOMES:
        event = t[f"{name}_event_indicator"].numpy().astype(int)
        competing = t[f"{name}_competing_indicator"].numpy().astype(int)
        out[name] = pd.DataFrame(
            {"time_years": t[f"{name}_target_days"].numpy().astype(float) / DAYS_PER_YEAR,
             "event_type": event + 2 * competing,
             "at_risk": t[f"{name}_at_risk"].numpy().astype(bool)},
            index=pd.Index(ids, name=PID),
        )
        if (out[name]["time_years"] >= NO_WINDOW_BINS - 1e-6).any():
            raise ValueError("follow-up reached the build_targets window; raise NO_WINDOW_BINS")
    return out


# ---- summaries ------------------------------------------------------------------------------


def _n(x: int) -> str:
    return f"{int(x):,}"


def _n_pct(count: int, total: int) -> str:
    return f"{_n(count)} ({100.0 * count / total:.1f})" if total else "–"


def _median_iqr(values: pd.Series, decimals: int) -> str:
    x = values.dropna().astype(float)
    if x.empty:
        return "–"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:,.{decimals}f} ({q1:,.{decimals}f}–{q3:,.{decimals}f})"


def reverse_km_quartiles(time: np.ndarray, died: np.ndarray) -> tuple:
    """(Q1, median, Q3) follow-up by reverse Kaplan-Meier: being censored (alive at the cutoff)
    is the 'event', death is censoring. NaN where a quartile is not reached."""
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter().fit(np.asarray(time, float), event_observed=1 - np.asarray(died, int))
    sf = kmf.survival_function_.iloc[:, 0]

    def first_time_at_or_below(p):  # the step function's exact jump time
        hit = sf.index[sf.to_numpy() <= p]
        return float(hit[0]) if len(hit) else float("nan")

    return tuple(first_time_at_or_below(p) for p in (0.75, 0.5, 0.25))


def _years(q: tuple) -> str:
    return "–" if any(np.isnan(v) or np.isinf(v) for v in q) else f"{q[1]:.1f} ({q[0]:.1f}–{q[2]:.1f})"


def build_rows(groups: dict, snapshot: pd.DataFrame, prior: pd.DataFrame, times: dict, visits_per_patient: pd.Series,
               horizon: float, observed_followup: pd.Series, after_entry: pd.DataFrame) -> tuple[list[dict], dict]:
    """Table rows ({kind: section|row, label, indent, <column>: text}) and the aggregate counts
    used by the cross-check / test."""
    counts = {c: {} for c in COLUMNS}
    rows: list[dict] = []

    def section(label):
        rows.append({"kind": "section", "label": label, "indent": 0, **{c: "" for c in COLUMNS}})

    def row(label, fn, indent=0):
        rows.append({"kind": "row", "label": label, "indent": indent, **{c: fn(c, groups[c]) for c in COLUMNS}})

    def missing_row(var):
        row("Missing, n (%)", lambda c, ids: _n_pct(snapshot.loc[ids, var].isna().sum(), len(ids)), indent=1)

    for c in COLUMNS:
        counts[c]["n"] = len(groups[c])

    # Demographics
    section("Demographics")
    var, label, dec = CONTINUOUS["Demographics"][0]
    row(f"{label}, median (IQR)", lambda c, ids: _median_iqr(snapshot.loc[ids, var], dec))
    missing_row(var)
    row("Male sex, n (%)", lambda c, ids: _n_pct((snapshot.loc[ids, "sex"] == "male").sum(), len(ids)))
    missing_row("sex")
    row("Smoking status, n (%)", lambda c, ids: "")
    for code, name in SMOKING:
        row(name, lambda c, ids, code=code: _n_pct((snapshot.loc[ids, "smoking_status"] == code).sum(), len(ids)), indent=1)
    missing_row("smoking_status")
    for c in COLUMNS:
        s = snapshot.loc[groups[c], "smoking_status"]
        counts[c]["smoking"] = {**{name: int((s == code).sum()) for code, name in SMOKING}, "Missing": int(s.isna().sum())}
        counts[c]["male"] = int((snapshot.loc[groups[c], "sex"] == "male").sum())

    # Continuous groups
    for group in ("Anthropometrics and blood pressure", "Laboratory values", "FibroScan"):
        section(group)
        for var, label, dec in CONTINUOUS[group]:
            row(f"{label}, median (IQR)", lambda c, ids, var=var, dec=dec: _median_iqr(snapshot.loc[ids, var], dec))
            if group == "FibroScan":
                row("Available, n (%)", lambda c, ids, var=var: _n_pct(snapshot.loc[ids, var].notna().sum(), len(ids)), indent=1)
            missing_row(var)

    # Medications
    section("Medications at entry, n (%)")
    for var, label in MEDICATIONS:
        row(label, lambda c, ids, var=var: _n_pct((snapshot.loc[ids, var] == 1).sum(), len(ids)))
        missing_row(var)

    # Prior CVD
    section("Prior CVD at index, n (%)")
    row("Any", lambda c, ids: _n_pct((prior.loc[ids, "any"] == 1).sum(), len(ids)))
    row("Missing, n (%)", lambda c, ids: _n_pct(prior.loc[ids, [n for n, _ in PRIOR_CVD]].isna().all(axis=1).sum(), len(ids)), indent=1)
    for var, label in PRIOR_CVD:
        row(label, lambda c, ids, var=var: _n_pct((prior.loc[ids, var] == 1).sum(), len(ids)), indent=1)
        row("Missing, n (%)", lambda c, ids, var=var: _n_pct(prior.loc[ids, var].isna().sum(), len(ids)), indent=2)

    # Follow-up
    section("Follow-up")
    mort = times["mortality"]
    row("Potential follow-up to 30 Jun 2025 (reverse Kaplan-Meier), years, median (IQR)",
        lambda c, ids: _years(reverse_km_quartiles(mort.loc[ids, "time_years"], mort.loc[ids, "event_type"] == 1)))
    row("Missing, n (%)", lambda c, ids: _n_pct(mort.loc[ids, "time_years"].isna().sum(), len(ids)), indent=1)
    row("Observed follow-up, index to last visit, years, median (IQR)",
        lambda c, ids: _median_iqr(observed_followup.loc[ids], 1))
    row("Missing, n (%)", lambda c, ids: _n_pct(observed_followup.loc[ids].isna().sum(), len(ids)), indent=1)
    row("Visits per patient, median (IQR)", lambda c, ids: _median_iqr(visits_per_patient.loc[ids], 0))
    row("Missing, n (%)", lambda c, ids: _n_pct(visits_per_patient.loc[ids].isna().sum(), len(ids)), indent=1)
    for c in COLUMNS:
        counts[c]["visits"] = int(visits_per_patient.loc[groups[c]].sum())

    # Outcomes
    section("Outcomes")
    for name in OUTCOMES:
        t = times[name]

        def at_risk_rows(ids, t=t):
            sub = t.loc[ids]
            return sub[sub["at_risk"]]

        if name == "cvd":
            row("CVD at-risk pool (no prior CVD), n", lambda c, ids: _n((prior.loc[ids, "any"] == 0).sum()))
        row(f"{OUTCOME_LABEL[name]}, events, n", lambda c, ids, f=at_risk_rows: _n((f(ids)["event_type"] == 1).sum()))
        row(f"{int(horizon)}-year cumulative incidence, %",
            lambda c, ids, f=at_risk_rows: f"{100 * aalen_johansen_cif(f(ids)['time_years'], f(ids)['event_type'], horizon):.1f}",
            indent=1)
        row("Missing, n (%)", lambda c, ids, t=t: _n_pct(t.loc[ids, "time_years"].isna().sum(), len(ids)), indent=1)
        for c in COLUMNS:
            sub = at_risk_rows(groups[c])
            counts[c][f"{name}_events"] = int((sub["event_type"] == 1).sum())
            counts[c][f"{name}_at_risk"] = int(len(sub))
    for c in COLUMNS:
        counts[c]["cvd_pool"] = int((prior.loc[groups[c], "any"] == 0).sum())
        for name in OUTCOMES:
            counts[c][f"{name}_events_after_entry"] = int(after_entry.loc[groups[c], name].sum())
    return rows, counts


def cross_check(counts: dict, outcomes: pd.DataFrame) -> list[str]:
    """Differences from the locked numbers (column n, events in the outcome table and in the
    table, CVD pool); empty if everything matches."""
    diffs = []
    for c, n in LOCKED["n"].items():
        if counts[c]["n"] != n:
            diffs.append(f"n {COLUMN_TITLE[c]}: {counts[c]['n']:,} vs locked {n:,}")
    for name, n in LOCKED["events"].items():
        in_data = int(outcomes[f"{name}_event"].sum())
        if in_data != n:
            diffs.append(f"{name} events in cohort_outcomes: {in_data:,} vs locked {n:,}")
        if counts["overall"][f"{name}_events"] != n:
            diffs.append(f"{name} events in the table: {counts['overall'][f'{name}_events']:,} vs locked {n:,}")
    for name, n in LOCKED["events_after_entry"].items():
        if counts["overall"][f"{name}_events_after_entry"] != n:
            diffs.append(f"{name} events after the first valid landmark: {counts['overall'][f'{name}_events_after_entry']:,} vs locked {n:,}")
    if counts["overall"]["cvd_pool"] != LOCKED["cvd_pool"]:
        diffs.append(f"CVD pool: {counts['overall']['cvd_pool']:,} vs locked {LOCKED['cvd_pool']:,}")
    return diffs


# ---- writers --------------------------------------------------------------------------------


def headers(counts: dict) -> list[str]:
    return ["Characteristic"] + [f"{COLUMN_TITLE[c]} (n = {counts[c]['n']:,})" for c in COLUMNS]


def write_csv(rows: list[dict], counts: dict, path: Path) -> None:
    out = pd.DataFrame(
        [{"section": r["label"] if r["kind"] == "section" else "", "characteristic": ("  " * r["indent"]) + r["label"]
          if r["kind"] == "row" else "", **{h: r[c] for h, c in zip(headers(counts)[1:], COLUMNS)}} for r in rows]
    )
    out.to_csv(path, index=False, encoding="utf-8")


def to_markdown(rows: list[dict], counts: dict) -> str:
    hdr = headers(counts)
    lines = [f"**{TITLE}**", "", "| " + " | ".join(hdr) + " |", "|" + "|".join([":---"] + ["---:"] * len(COLUMNS)) + "|"]
    for r in rows:
        if r["kind"] == "section":
            lines.append(f"| **{r['label']}** |" + " |" * len(COLUMNS))
        else:
            label = "&nbsp;&nbsp;&nbsp;&nbsp;" * r["indent"] + r["label"]
            lines.append("| " + " | ".join([label] + [r[c] for c in COLUMNS]) + " |")
    return "\n".join(lines + ["", footnote(counts), ""])


def _set_borders(cell, **edges) -> None:
    """edges: top/bottom = True (single black line) on a python-docx cell."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "bottom"):
        if edges.get(edge):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "8")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            borders.append(el)


def write_docx(rows: list[dict], counts: dict, path: Path) -> None:
    """A real Word table (three horizontal rules, no colours, no shading), bold title above and
    the footnote below."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.left_margin = sec.right_margin = Cm(2)
    style = doc.styles["Normal"]
    style.font.name, style.font.size = "Arial", Pt(8)
    style.paragraph_format.space_after = Pt(0)

    title = doc.add_paragraph()
    run = title.add_run(TITLE)
    run.bold, run.font.size = True, Pt(10)
    title.paragraph_format.space_after = Pt(6)

    hdr = headers(counts)
    table = doc.add_table(rows=1 + len(rows), cols=len(hdr))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Cm(6.2)] + [Cm(2.7)] * len(COLUMNS)

    def fill(cell, text, bold=False, align_right=False, indent=0):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if align_right else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Cm(0.35 * indent)
        r = p.add_run(text)
        r.bold = bold

    for j, h in enumerate(hdr):
        cell = table.rows[0].cells[j]
        cell.width = widths[j]
        fill(cell, h, bold=True, align_right=j > 0)
        _set_borders(cell, top=True, bottom=True)
    for i, r in enumerate(rows, start=1):
        cells = table.rows[i].cells
        for j in range(len(hdr)):
            cells[j].width = widths[j]
        if r["kind"] == "section":
            fill(cells[0], r["label"], bold=True)
        else:
            fill(cells[0], r["label"], indent=r["indent"])
            for j, c in enumerate(COLUMNS, start=1):
                fill(cells[j], r[c], align_right=True)
        if i == len(rows):
            for cell in cells:
                _set_borders(cell, bottom=True)

    note = doc.add_paragraph(footnote(counts))
    note.paragraph_format.space_before = Pt(4)
    for r in note.runs:
        r.font.size = Pt(7)
    doc.save(path)


# ---- main -----------------------------------------------------------------------------------


def make_table(visits: pd.DataFrame, outcomes: pd.DataFrame, split_ids: dict, prior: pd.DataFrame,
               cutoff: pd.Timestamp, horizon: float) -> tuple[list[dict], dict]:
    entry = entry_dates(visits, outcomes)
    snapshot = entry_snapshot(visits, outcomes, entry)
    times = outcome_times(visits, outcomes, cutoff)
    o = outcomes.set_index(PID)
    visits_per_patient = visits.groupby(PID).size().reindex(o.index)
    observed_followup = (compute_last_visit_dates(visits).reindex(o.index) - o["index_date"]).dt.days / DAYS_PER_YEAR
    # Events (from index_date) dated after the first valid landmark: the ones the models can predict.
    after_entry = pd.DataFrame({name: (o[f"{name}_event"] == 1) & (o[f"{name}_date"] > entry) for name in OUTCOMES})
    groups = {"overall": outcomes[PID].tolist(), **split_ids}
    return build_rows(groups, snapshot, prior, times, visits_per_patient, horizon, observed_followup, after_entry)


def footnote(counts: dict) -> str:
    o = counts["overall"]
    return FOOTNOTE.format(malo_after=_n(o["malo_events_after_entry"]), cvd_after=_n(o["cvd_events_after_entry"]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--split-file", default="runs/tune_v2/split.json")
    p.add_argument("--mapping", default="configs/real_data_mapping.yaml")
    p.add_argument("--schema", default="configs/data_schema.yaml")
    p.add_argument("--raw", default=None, help="Raw .dta with the CVD prevalence flags (default: the mapping's source.file).")
    p.add_argument("--horizon", type=float, default=10.0)
    p.add_argument("--out-dir", default="outputs/tables")
    p.add_argument("--skip-locked-check", action="store_true", help="Synthetic data only: skip the locked-number cross-check.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = yaml.safe_load(Path(args.mapping).read_text())
    schema = yaml.safe_load(Path(args.schema).read_text())
    cutoff = pd.Timestamp(schema["censoring"]["administrative_cutoff"])
    visits = pd.read_parquet(Path(args.data) / "cohort_visits.parquet")
    outcomes = pd.read_parquet(Path(args.data) / "cohort_outcomes.parquet")
    split_ids = load_split_ids(args.split_file, visits, outcomes)
    try:
        prior = prior_cvd_flags(args.raw or mapping["source"]["file"], mapping, outcomes)
    except LockedMismatch as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 4

    rows, counts = make_table(visits, outcomes, split_ids, prior, cutoff, args.horizon)
    if not args.skip_locked_check:
        diffs = cross_check(counts, outcomes)
        if diffs:
            print("STOP: differs from the locked numbers -- nothing written:\n  " + "\n  ".join(diffs), file=sys.stderr)
            return 4
        print("cross-check: n per column and event counts match the locked numbers")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, counts, out / "table1.csv")
    (out / "table1.md").write_text(to_markdown(rows, counts))
    write_docx(rows, counts, out / "table1.docx")
    (out / "table1_counts.json").write_text(json.dumps(counts, indent=2))
    print(f"wrote {out}/table1.docx, table1.csv, table1.md ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

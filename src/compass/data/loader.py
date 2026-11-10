"""Real-data loader: Siriraj MASLD-T2DM .dta -> (visits_df, outcomes_df) in the pipeline's
schema (configs/data_schema.yaml), driven entirely by configs/real_data_mapping.yaml -- no
source column names are hard-coded here.

build_cohort() steps:
1. Baseline (patient-level) source columns must be constant within each patient (raises).
2. Patient exclusions in the mapping's order (E1..E4), each logged as removed / remaining.
3. Visits: drop visit_after_death rows and visits after the administrative cutoff; map
   columns and codes. [TV] per-visit columns as they are; `placement: baseline_once`
   variables only on the first visit on/after index_date (NaN elsewhere); other baseline
   variables (static attributes: age, sex, smoking) repeated on every row.
4. Patients with no valid landmark (no visit on/after index_date and before death) are
   dropped (E5) -- compass.data.landmarking.landmark_candidate_mask is the one definition.
5. Outcomes, one row per patient: <x>_dt cleared where its event flag is 0; incident
   component = event == 1 and index_date < date <= cutoff; MALO / CVD = earliest incident
   component; cvd_at_risk = 0 if any CVD prevalent flag == 1 (then cvd_event = 0);
   mortality from the death indicator (date only where it is 1).

Everything logged is an aggregate count. Error messages carry counts and column names only.
"""

import pandas as pd
import yaml

from compass.data.landmarking import landmark_candidate_mask


class CohortCountMismatch(ValueError):
    """An exclusion count or the analytic n differs from the mapping's expected value."""


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_stata(path: str) -> pd.DataFrame:
    return pd.read_stata(path, convert_dates=True, convert_categoricals=False)


def _feature_entries(mapping: dict) -> dict[str, dict]:
    """Pipeline variable name -> mapping entry, for every schema feature."""
    out = {}
    for group in ("continuous_variables", "medication_flags", "demographics"):
        out.update(mapping[group])
    return out


def _outcome_columns(mapping: dict) -> list[str]:
    cols = [mapping["outcomes"]["mortality"]["event"], mapping["outcomes"]["mortality"]["date"]]
    for outcome in ("malo", "cvd"):
        for comp in mapping["outcomes"][outcome]["components"].values():
            cols += [c for c in (comp["event"], comp["date"], comp.get("prevalent")) if c]
    return cols


def _check_baseline_constant(df: pd.DataFrame, mapping: dict) -> None:
    pid = mapping["structure"]["patient_id"]
    cols = [mapping["structure"]["index_date"], mapping["exclusion_inputs"]["liver_cancer_baseline"]]
    cols += [e["column"] for e in _feature_entries(mapping).values() if e["level"] == "baseline"]
    cols += _outcome_columns(mapping)
    cols = list(dict.fromkeys(cols))
    nunique = df.groupby(pid)[cols].nunique(dropna=False)
    varying = nunique > 1
    if varying.any().any():
        bad_cols = varying.columns[varying.any()].tolist()
        raise ValueError(
            f"{int(varying.any(axis=1).sum())} patient(s) have non-constant values in baseline column(s) {bad_cols}"
        )


def _component(pt: pd.DataFrame, comp: dict, index_date: pd.Series, cutoff: pd.Timestamp):
    """-> (event flag, date cleared where flag != 1, prior mask, incident mask)."""
    flag = pt[comp["event"]] == 1
    date = pt[comp["date"]].where(flag)
    prior = flag & (date <= index_date)
    if comp.get("prevalent"):
        prior = prior | (pt[comp["prevalent"]] == 1)
    incident = flag & (date > index_date) & (date <= cutoff)
    return flag, date, prior, incident


def _death(pt: pd.DataFrame, mapping: dict):
    m = mapping["outcomes"]["mortality"]
    flag = pt[m["event"]] == 1
    return flag, pt[m["date"]].where(flag)


def _map_codes(series: pd.Series, code_map: dict, name: str) -> pd.Series:
    mapped = series.map(code_map)
    bad = series.notna() & mapped.isna()
    if bad.any():
        codes = sorted(series[bad].unique().tolist(), key=str)
        raise ValueError(f"'{name}': {int(bad.sum())} row(s) with code(s) {codes} not in the mapping code_map {list(code_map)}")
    return mapped


def build_cohort(
    df: pd.DataFrame, mapping: dict, schema: dict, check_expected: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """-> (visits_df, outcomes_df, flow_log). See module docstring for the rules."""
    s = mapping["structure"]
    pid, vdate, idx_col = s["patient_id"], s["visit_date"], s["index_date"]
    cutoff = pd.Timestamp(schema["censoring"]["administrative_cutoff"])

    _check_baseline_constant(df, mapping)
    pt = df.groupby(pid).first()
    index_date = pt[idx_col]
    death_flag, death_date = _death(pt, mapping)

    flow = {"source": {"rows": int(len(df)), "patients": int(len(pt))}, "exclusions": [], "visits": {}, "final": {}}
    keep = pd.Series(True, index=pt.index)
    mismatches = []
    for rule in mapping["exclusions"]:
        if rule["rule"] == "prior_component":
            comps = mapping["outcomes"][rule["outcome"]]["components"]
            mask = pd.Series(False, index=pt.index)
            for name in rule["components"]:
                mask |= _component(pt, comps[name], index_date, cutoff)[2]
        elif rule["rule"] == "death_on_or_before_index":
            mask = death_flag & (death_date <= index_date)
        elif rule["rule"] == "flag_equals_one":
            mask = pt[mapping["exclusion_inputs"][rule["input"]]] == 1
        else:
            raise ValueError(f"unknown exclusion rule '{rule['rule']}' ({rule['id']})")
        removed = mask & keep
        keep &= ~removed
        entry = {"id": rule["id"], "label": rule["label"], "removed": int(removed.sum()), "remaining": int(keep.sum())}
        if "expected_removed" in rule:
            entry["expected_removed"] = rule["expected_removed"]
            if entry["removed"] != rule["expected_removed"]:
                mismatches.append(f"{rule['id']}: removed {entry['removed']}, expected {rule['expected_removed']}")
        flow["exclusions"].append(entry)
    expected_n = mapping.get("expected_analytic_n")
    flow["analytic_n"] = {"observed": int(keep.sum()), "expected": expected_n}
    if expected_n is not None and int(keep.sum()) != expected_n:
        mismatches.append(f"analytic n {int(keep.sum())}, expected {expected_n}")
    if mismatches and check_expected:
        raise CohortCountMismatch("cohort counts differ from the mapping: " + "; ".join(mismatches))

    # ---- visits -------------------------------------------------------------------------
    v = df[df[pid].isin(pt.index[keep])]
    after_death = v[mapping["exclusion_inputs"]["visit_after_death"]] == 1
    after_cutoff = v[vdate] > cutoff
    flow["visits"] = {
        "rows_in_cohort": int(len(v)),
        "dropped_visit_after_death": int(after_death.sum()),
        "dropped_after_cutoff": int((after_cutoff & ~after_death).sum()),
    }
    v = v[~after_death & ~after_cutoff].sort_values([pid, vdate])

    visits = pd.DataFrame({"patient_id": v[pid].to_numpy(), "visit_date": v[vdate].to_numpy()})
    v_index = v[idx_col].to_numpy()
    on_or_after_index = v[vdate].to_numpy() >= v_index
    # first visit on/after index_date, per patient (rows are sorted by patient, date)
    first_post_index = pd.Series(on_or_after_index).groupby(v[pid].to_numpy()).cumsum().to_numpy() == 1
    first_post_index &= on_or_after_index
    for name, entry in _feature_entries(mapping).items():
        col = v[entry["column"]].reset_index(drop=True)
        if "code_map" in entry:
            col = _map_codes(col, entry["code_map"], entry["column"])
        if entry.get("placement") == "baseline_once":
            col = col.where(first_post_index)
        visits[name] = col.to_numpy()

    # ---- landmark eligibility (E5) --------------------------------------------------------
    candidate = landmark_candidate_mask(visits["visit_date"], v_index, v[pid].map(death_date).to_numpy())
    has_candidate = pd.Series(candidate).groupby(visits["patient_id"].to_numpy()).any().reindex(pt.index[keep], fill_value=False)
    no_landmark = ~has_candidate
    keep_final = has_candidate.index[has_candidate]
    flow["exclusions"].append(
        {"id": "E5", "label": "No valid landmark (no visit on/after index_date and before death)",
         "removed": int(no_landmark.sum()), "remaining": int(len(keep_final))}
    )
    visits = visits[visits["patient_id"].isin(keep_final)].reset_index(drop=True)

    # ---- outcomes -------------------------------------------------------------------------
    p = pt.loc[keep_final]
    p_index = index_date.loc[keep_final]
    outcomes = pd.DataFrame({"patient_id": keep_final.to_numpy(), "index_date": p_index.to_numpy()})
    d_flag, d_date = death_flag.loc[keep_final], death_date.loc[keep_final]
    outcomes["mortality_event"] = d_flag.astype(int).to_numpy()
    outcomes["mortality_date"] = d_date.to_numpy()
    for outcome in ("malo", "cvd"):
        comps = mapping["outcomes"][outcome]["components"]
        incident_dates = []
        for comp in comps.values():
            _, date, _, incident = _component(p, comp, p_index, cutoff)
            incident_dates.append(date.where(incident))
        dates = pd.concat(incident_dates, axis=1)
        first = dates.min(axis=1)
        if outcome == "cvd":
            prevalent = pd.concat([p[c["prevalent"]] == 1 for c in comps.values() if c.get("prevalent")], axis=1).any(axis=1)
            outcomes["cvd_at_risk"] = (~prevalent).astype(int).to_numpy()
            first = first.where(~prevalent)
        outcomes[f"{outcome}_event"] = first.notna().astype(int).to_numpy()
        outcomes[f"{outcome}_date"] = first.to_numpy()
    outcomes = outcomes[
        ["patient_id", "index_date", "mortality_event", "mortality_date", "malo_event", "malo_date",
         "cvd_event", "cvd_date", "cvd_at_risk"]
    ]

    flow["final"] = {
        "patients": int(len(outcomes)),
        "visits": int(len(visits)),
        "deaths": int(outcomes["mortality_event"].sum()),
        "malo_events": int(outcomes["malo_event"].sum()),
        "cvd_at_risk": int(outcomes["cvd_at_risk"].sum()),
        "cvd_events": int(outcomes["cvd_event"].sum()),
    }
    flow["missing_pct_after_placement"] = {
        name: round(float(visits[name].isna().mean() * 100), 2) for name in _feature_entries(mapping)
    }
    return visits, outcomes, flow

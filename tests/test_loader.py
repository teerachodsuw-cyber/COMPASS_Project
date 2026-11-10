"""Tests for compass.data.loader.build_cohort on a synthetic mini-dataframe shaped like the
real .dta (same source column names, taken from configs/real_data_mapping.yaml). No real
data is read.
"""

import numpy as np
import pandas as pd
import pytest

from compass.data.landmarking import sample_landmark
from compass.data.loader import CohortCountMismatch, build_cohort, load_yaml
from compass.data.validate_schema import validate_dataset

MAPPING = load_yaml("configs/real_data_mapping.yaml")
SCHEMA = load_yaml("configs/data_schema.yaml")
CUTOFF = pd.Timestamp(SCHEMA["censoring"]["administrative_cutoff"])
T = pd.Timestamp
COMPS = {**MAPPING["outcomes"]["malo"]["components"], **MAPPING["outcomes"]["cvd"]["components"]}
DEATH = MAPPING["outcomes"]["mortality"]
EXCL = MAPPING["exclusion_inputs"]
FEATURES = {**MAPPING["continuous_variables"], **MAPPING["medication_flags"], **MAPPING["demographics"]}


def _patient(pid, index="2015-01-01", visits=("2015-01-01", "2016-01-01", "2017-01-01"), events=None,
             prevalent=(), death=None, **patient_cols) -> pd.DataFrame:
    """One patient's source rows, every mapped column filled like the real file:
    <x>_dt columns hold the censoring date (cutoff) unless the event flag is 1."""
    events = events or {}
    rows = []
    for k, vd in enumerate(visits):
        row = {
            MAPPING["structure"]["patient_id"]: pid,
            MAPPING["structure"]["visit_date"]: T(vd),
            MAPPING["structure"]["index_date"]: T(index),
            EXCL["liver_cancer_baseline"]: 0.0,
            EXCL["visit_after_death"]: 0,
            DEATH["event"]: int(death is not None),
            DEATH["date"]: T(death) if death else CUTOFF,
        }
        for j, (name, e) in enumerate(FEATURES.items()):
            if name == "sex":
                row[e["column"]] = 1
            elif name == "smoking_status":
                row[e["column"]] = 0.0
            elif name in MAPPING["medication_flags"]:
                row[e["column"]] = 1.0 if e["level"] == "baseline" else float(k % 2)
            elif e["level"] == "baseline":
                row[e["column"]] = 100.0 + j            # constant within patient
            else:
                row[e["column"]] = 10.0 * (k + 1) + j   # varies by visit
        for name, c in COMPS.items():
            row[c["event"]] = int(name in events)
            row[c["date"]] = T(events[name]) if name in events else (CUTOFF if name != "livertx" else pd.NaT)
            if c.get("prevalent"):
                row[c["prevalent"]] = int(name in prevalent)
        row.update(patient_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def _build(*patients, **kw):
    return build_cohort(pd.concat(patients, ignore_index=True), MAPPING, SCHEMA, check_expected=kw.get("check_expected", False))


def _flow(flow, step):
    return next(e for e in flow["exclusions"] if e["id"] == step)


# ---- exclusions -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "step,bad",
    [
        ("E1", dict(prevalent=("hcc",))),
        ("E1", dict(events={"com_cirrhosis": "2015-01-01"})),            # liver event ON index -> prior
        ("E1", dict(events={"decom_cirrhosis": "2014-06-01"})),          # before index
        ("E2", dict(events={"livertx": "2015-01-01"})),
        ("E3", dict(death="2015-01-01")),
        ("E4", {EXCL["liver_cancer_baseline"]: 1.0}),
    ],
    ids=["E1_prevalent", "E1_on_index", "E1_before_index", "E2_transplant", "E3_death_on_index", "E4_liver_cancer"],
)
def test_each_exclusion_removes_only_its_patient(step, bad):
    visits, outcomes, flow = _build(_patient(1), _patient(2, **bad))
    assert outcomes["patient_id"].tolist() == [1]
    assert set(visits["patient_id"]) == {1}
    for e in flow["exclusions"]:
        assert e["removed"] == (1 if e["id"] == step else 0), e["id"]


def test_exclusions_applied_in_order_and_logged_once():
    # patient 2 qualifies for E1 and E4: counted under E1 only
    _, _, flow = _build(_patient(1), _patient(2, prevalent=("hcc",), **{EXCL["liver_cancer_baseline"]: 1.0}))
    assert [e["id"] for e in flow["exclusions"]] == ["E1", "E2", "E3", "E4", "E5"]
    assert _flow(flow, "E1")["removed"] == 1 and _flow(flow, "E4")["removed"] == 0
    assert _flow(flow, "E4")["remaining"] == 1


def test_expected_count_mismatch_raises():
    with pytest.raises(CohortCountMismatch, match="E1: removed 0, expected 923"):
        _build(_patient(1), check_expected=True)


# ---- outcomes -------------------------------------------------------------------------


def test_dates_cleared_where_flag_is_zero():
    _, outcomes, _ = _build(_patient(1))  # every <x>_dt holds the cutoff, every flag 0
    row = outcomes.iloc[0]
    for name in ("mortality", "malo", "cvd"):
        assert row[f"{name}_event"] == 0
        assert pd.isna(row[f"{name}_date"])


def test_composite_is_earliest_incident_component():
    _, outcomes, _ = _build(
        _patient(1, events={"hcc": "2019-03-01", "com_cirrhosis": "2017-05-01", "cad": "2020-01-01", "hf": "2016-02-01"})
    )
    row = outcomes.iloc[0]
    assert row["malo_event"] == 1 and row["malo_date"] == T("2017-05-01")
    assert row["cvd_event"] == 1 and row["cvd_date"] == T("2016-02-01")


def test_event_on_index_or_after_cutoff_is_not_incident():
    # CVD event ON index_date (no prevalent flag) and a stroke after the cutoff: neither incident
    _, outcomes, _ = _build(_patient(1, events={"cad": "2015-01-01", "stroke": "2025-07-15"}))
    row = outcomes.iloc[0]
    assert row["cvd_event"] == 0 and pd.isna(row["cvd_date"])
    # ... and the next component after index is picked up instead
    _, outcomes, _ = _build(_patient(1, events={"cad": "2015-01-01", "pvd": "2018-01-01"}))
    assert outcomes.iloc[0]["cvd_date"] == T("2018-01-01")


def test_cvd_prevalent_forces_not_at_risk_and_no_event():
    _, outcomes, _ = _build(_patient(1, events={"hf": "2018-01-01"}, prevalent=("stroke",)), _patient(2))
    o = outcomes.set_index("patient_id")
    assert o.loc[1, "cvd_at_risk"] == 0 and o.loc[1, "cvd_event"] == 0 and pd.isna(o.loc[1, "cvd_date"])
    assert o.loc[2, "cvd_at_risk"] == 1


def test_mortality_from_death_indicator():
    _, outcomes, _ = _build(_patient(1, death="2018-06-01"))
    row = outcomes.iloc[0]
    assert row["mortality_event"] == 1 and row["mortality_date"] == T("2018-06-01")
    assert row["index_date"] == T("2015-01-01")


# ---- visits ---------------------------------------------------------------------------


def test_baseline_once_on_first_post_index_visit_only():
    visits, _, _ = _build(_patient(1, index="2015-01-10", visits=("2014-12-20", "2015-02-01", "2016-01-01")))
    assert visits["visit_date"].tolist() == [T("2014-12-20"), T("2015-02-01"), T("2016-01-01")]  # pre-index kept as history
    for name, e in FEATURES.items():
        col = visits[name]
        if e.get("placement") == "baseline_once":
            assert col.isna().tolist() == [True, False, True], name
        elif e["level"] == "baseline":
            assert col.notna().all() and col.nunique() == 1, name  # static attribute repeated
        else:
            assert col.notna().all(), name
    assert visits["sex"].tolist() == ["male"] * 3
    assert visits["smoking_status"].tolist() == [0, 0, 0]


def test_post_death_and_post_cutoff_visits_dropped():
    p = _patient(1, visits=("2015-01-01", "2016-01-01", "2017-01-01", "2025-07-01"), death="2016-06-01")
    p.loc[p[MAPPING["structure"]["visit_date"]] == T("2017-01-01"), EXCL["visit_after_death"]] = 1
    p.loc[p[MAPPING["structure"]["visit_date"]] == T("2025-07-01"), EXCL["visit_after_death"]] = 1
    q = _patient(2, visits=("2015-01-01", "2025-07-01"))
    visits, _, flow = _build(p, q)
    assert visits.groupby("patient_id").size().to_dict() == {1: 2, 2: 1}
    assert flow["visits"]["dropped_visit_after_death"] == 2
    assert flow["visits"]["dropped_after_cutoff"] == 1


def test_unknown_sex_code_raises():
    with pytest.raises(ValueError, match=r"'sex_code': 3 row\(s\) with code\(s\) \[3\]"):
        _build(_patient(1, sex_code=3))


def test_non_constant_baseline_column_raises():
    p = _patient(1)
    p.loc[p.index[1], MAPPING["continuous_variables"]["height"]["column"]] = 999.0
    with pytest.raises(ValueError, match=r"1 patient\(s\) have non-constant values in baseline column\(s\) \['height_clean_rc'\]"):
        _build(p, _patient(2))


# ---- landmarks ------------------------------------------------------------------------


def test_landmark_sampling_skips_pre_index_visits():
    visits = pd.DataFrame({"patient_id": [1] * 4, "visit_date": pd.to_datetime(["2014-11-01", "2014-12-15", "2015-01-10", "2016-01-01"])})
    rng = np.random.default_rng(0)
    seen = {sample_landmark(visits, rng, index_date=T("2015-01-10"), death_date=pd.NaT) for _ in range(200)}
    assert seen == {T("2015-01-10"), T("2016-01-01")}


@pytest.mark.parametrize(
    "bad",
    [
        dict(index="2015-06-01", visits=("2015-01-01", "2015-03-01")),       # every visit before index
        dict(visits=("2015-01-01", "2016-01-01"), death="2015-01-01"),       # only post-index visit is on the death date...
    ],
    ids=["all_visits_pre_index", "only_candidate_on_death_date"],
)
def test_patient_without_valid_landmark_is_dropped(bad):
    if "death" in bad:  # ...index is earlier, so the death is not on index (E3) but on the first visit
        bad = dict(bad, index="2014-12-01", visits=("2014-11-01", "2015-01-01"))
    visits, outcomes, flow = _build(_patient(1), _patient(2, **bad))
    assert outcomes["patient_id"].tolist() == [1]
    assert set(visits["patient_id"]) == {1}
    assert _flow(flow, "E5")["removed"] == 1 and _flow(flow, "E5")["remaining"] == 1


# ---- end to end -----------------------------------------------------------------------


def test_loader_output_passes_validator():
    visits, outcomes, flow = _build(
        _patient(1, events={"hcc": "2019-01-01"}),
        _patient(2, index="2015-02-01", visits=("2015-01-15", "2015-02-01", "2018-01-01"), death="2019-01-01",
                 events={"cad": "2017-01-01"}),
        _patient(3, prevalent=("cad",), **{MAPPING["demographics"]["smoking_status"]["column"]: np.nan}),
    )
    validate_dataset(visits, outcomes)
    assert flow["final"] == {"patients": 3, "visits": 9, "deaths": 1, "malo_events": 1, "cvd_at_risk": 2, "cvd_events": 1}
    assert set(flow["missing_pct_after_placement"]) == set(FEATURES)

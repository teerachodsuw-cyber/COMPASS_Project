#!/usr/bin/env python
"""Publication figures from the final evaluation outputs (aggregates only). Reads results; never
recomputes metrics, refits cut-points or touches models.

Usage:
    python scripts/make_figures.py                      # defaults below = the final test results
    python scripts/make_figures.py --eval-dir outputs/final_test --out-dir outputs/figures \\
        --flow data/processed/flow_log.json --split-file runs/tune_v2/split.json \\
        --band-assignments runs/final_test/band_assignments.csv [--include-baseline]
    # final version (Fig 2 / S3 with the baseline row, plus Fig 5):
    python scripts/make_figures.py --include-baseline --baseline-eval-dir outputs/baseline_test \\
        --serial-dir outputs/serial_test --serial-val-dir outputs/serial_val --importance-dir outputs/importance

Inputs:
- <eval-dir>/metrics.json (scripts/evaluate.py): C-index / IBS with bootstrap CIs, pairwise
  differences, calibration deciles + O/E, risk bands (n, events, observed risk, HRs).
- --flow (scripts/build_cohort.py's flow_log.json) and --split-file (split.json; only the
  number of IDs per split is read) for the cohort flow.
- --band-assignments (scripts/export_band_assignments.py: band, time, event type per at-risk
  patient, no IDs) for the risk-band curves only. Each curve's value at 10 yr and each band's n
  are checked against the saved banding table in metrics.json before anything is drawn.

Outputs in --out-dir (git-ignored): for each figure an editable vector PDF (TrueType fonts,
text kept as text) and a 300-dpi PNG preview:
  fig1_cohort_flow, fig2_cindex, fig3_calibration, fig4_risk_band_curves,
  figS1_pairwise_cindex, figS2_band_hazard_ratios, figS3_ibs (+ figS4_baseline_differences,
  fig5_serial_evaluation; see below).
--include-baseline adds a grey "Cox/Fine-Gray baseline" row (model "cox_baseline") to Fig 2 and
S3; off by default. Its rows come from --baseline-eval-dir (scripts/evaluate.py --baseline output,
e.g. outputs/baseline_test), checked to be the same split, patients, horizon and bootstrap settings.
With --baseline-eval-dir, figS4_baseline_differences (deep model minus the Cox / Fine-Gray baseline,
paired C-index / IBS differences) is drawn too; its deep-model rows must equal --eval-dir's exactly and
every plotted estimate / CI is read back and must equal the baseline JSON.
--serial-dir / --serial-val-dir add fig5_serial_evaluation from scripts/serial_evaluate.py's test
(--final) and validation metrics.json: the two runs must share models, checkpoints, settings and the
landmark Cox fit (fitted on validation, reused on test); every plotted estimate / CI is read back
from the figure and must equal the test JSON exactly.
--importance-dir adds fig6_importance_grouped and figS5_importance_top15 from
scripts/feature_importance.py's importance.json (mean drop and 2.5-97.5 percentile range per
predictor; rows ordered by the mean across the four predictors); every plotted value is read back
and must equal the JSON exactly.
Fig 4 curves stop at the last time with >= 10 patients at risk in that band (the full curves are
still checked against the saved 10-yr observed risks); numbers-at-risk tables are not truncated.

Style (JHEP Reports): Arial (fallback Liberation Sans / Helvetica), 8 pt text (6.5-7 pt only for
dense legends / numbers at risk; nothing below 6 pt), sentence case, no full stops; panel letters
Arial bold 14 pt; widths 174 mm (full) / 85 mm (single); recessive grey axes and grid; fixed
colour-blind-safe colours with a distinct marker / line style per series (never colour alone).
Every saved figure is checked for overlapping text; any overlap is reported and fails the run.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from compass.evaluation.calibration import aalen_johansen_cif  # noqa: E402

MM = 1 / 25.4
FULL_WIDTH, SINGLE_WIDTH = 174 * MM, 85 * MM
OUTCOMES = ("mortality", "malo", "cvd")
OUTCOME_LABEL = {"mortality": "Mortality", "malo": "MALO", "cvd": "CVD"}
MODELS = ("dynamic_deephit", "tracer", "survlatent_ode")
MODEL_STYLE = {
    "dynamic_deephit": {"label": "Dynamic-DeepHit", "color": "#0072B2", "marker": "o"},
    "tracer": {"label": "TraCeR", "color": "#D55E00", "marker": "s"},
    "survlatent_ode": {"label": "SurvLatent ODE", "color": "#009E73", "marker": "^"},
    "cox_baseline": {"label": "Cox/Fine-Gray baseline", "color": "#6E6E6E", "marker": "D"},
}
BAND_STYLE = (
    {"label": "Low", "short": "Low", "color": "#9E8CC9", "ls": "-"},
    {"label": "Intermediate", "short": "Int", "color": "#6A4FA8", "ls": "--"},
    {"label": "High", "short": "High", "color": "#3B1F75", "ls": ":"},
)
CINDEX_LABEL = {"mortality": "C-index (Uno)", "malo": "C-index (Wolbers)", "cvd": "C-index (Wolbers)"}
IBS_LABEL = {"mortality": "Integrated Brier score", "malo": "Competing-risk IBS (Schoop)",
             "cvd": "Competing-risk IBS (Schoop)"}
OBSERVED_LABEL = {"mortality": "Kaplan-Meier", "malo": "Aalen-Johansen", "cvd": "Aalen-Johansen"}
EXCLUSION_LABEL = {
    "E1": "Prior cirrhosis, decompensated cirrhosis or HCC",
    "E2": "Prior liver transplant",
    "E3": "Death on index date",
    "E4": "Liver cancer at baseline",
    "E5": "No valid landmark visit",
}
AXIS_GREY, GRID_GREY, TEXT_COLOUR, REF_GREY = "#8C8C8C", "#E3E3E3", "#222222", "#6E6E6E"
RISK_TIMES = (0, 2, 4, 6, 8, 10)
MIN_FONT_PT = 6.0

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.edgecolor": AXIS_GREY, "axes.linewidth": 0.6, "axes.labelcolor": TEXT_COLOUR, "text.color": TEXT_COLOUR,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4,
    "xtick.color": AXIS_GREY, "ytick.color": AXIS_GREY, "xtick.labelcolor": TEXT_COLOUR, "ytick.labelcolor": TEXT_COLOUR,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": GRID_GREY, "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.3,
    "savefig.dpi": 300, "axes.unicode_minus": True,
}


class FigureDataError(ValueError):
    """The inputs are missing something a figure needs, or disagree with each other."""


# ---- helpers ------------------------------------------------------------------------------


def _per_model(results: dict) -> dict:
    return {(r["model"], r["outcome"], r["metric_name"]): r for r in results["per_model"]}


def _models_in(results: dict, include_baseline: bool) -> list[str]:
    present = {r["model"] for r in results["per_model"]}
    models = [m for m in MODELS if m in present]
    if include_baseline:
        if "cox_baseline" not in present:
            raise FigureDataError("--include-baseline: no 'cox_baseline' model results (pass --baseline-eval-dir, "
                                  "e.g. outputs/baseline_test)")
        models.append("cox_baseline")
    return models


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _panel_letter(fig, ax, letter: str) -> None:
    """Bold 14 pt letter at the top-left of the axes' full extent (incl. tick / axis labels)."""
    r = fig.canvas.get_renderer()
    bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    top = ax.get_position().y1
    fig.text(bb.x0, top + 0.012, letter, fontsize=14, fontweight="bold", ha="left", va="bottom")


def _style_axes(ax, grid_axis: str = "x") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def drawn_texts(fig) -> list[Text]:
    """Every non-empty visible text that is actually drawn (tick labels only for ticks shown)."""
    fig.canvas.draw()
    texts = [t for t in fig.texts]
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:  # axes drawn with axis("off") show no ticks or axis labels
            texts += [ax.xaxis.label, ax.yaxis.label]
            for axis in (ax.xaxis, ax.yaxis):
                for tick in axis._update_ticks():
                    texts += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
        if ax.get_legend():
            texts += list(ax.get_legend().get_texts())
    for leg in fig.legends:
        texts += list(leg.get_texts())
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def text_problems(fig) -> list[str]:
    """Overlapping text pairs, text outside the figure, and text below the minimum size."""
    r = fig.canvas.get_renderer()
    texts = drawn_texts(fig)
    boxes = [t.get_window_extent(r) for t in texts]
    fig_box = fig.bbox
    problems = []
    for t, b in zip(texts, boxes):
        if t.get_fontsize() < MIN_FONT_PT:
            problems.append(f"text below {MIN_FONT_PT} pt: {t.get_text()!r}")
        if b.x0 < fig_box.x0 - 1 or b.y0 < fig_box.y0 - 1 or b.x1 > fig_box.x1 + 1 or b.y1 > fig_box.y1 + 1:
            problems.append(f"text outside the figure: {t.get_text()!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = boxes[i], boxes[j]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                problems.append(f"overlap: {texts[i].get_text()!r} / {texts[j].get_text()!r}")
    return problems


def save(fig, out_dir: Path, name: str) -> list[Path]:
    problems = text_problems(fig)
    if problems:
        raise FigureDataError(f"{name}: " + "; ".join(problems))
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


# ---- Fig 1: cohort flow ---------------------------------------------------------------------


def cohort_flow_counts(flow: dict, split: dict) -> dict:
    start = int(flow["source"]["patients"])
    remaining = start
    exclusions = []
    for e in flow["exclusions"]:
        if remaining - int(e["removed"]) != int(e["remaining"]):
            raise FigureDataError(f"flow log: {e['id']} counts do not add up")
        remaining = int(e["remaining"])
        exclusions.append((e["id"], EXCLUSION_LABEL.get(e["id"], e["label"]), int(e["removed"])))
    final = int(flow["final"]["patients"])
    if remaining != final:
        raise FigureDataError(f"flow log: {remaining} after exclusions but final patients {final}")
    sizes = {k: len(split[k]) for k in ("train", "val", "test")}
    if sum(sizes.values()) != final:
        raise FigureDataError(f"split sizes {sizes} do not sum to the analytic cohort {final}")
    return {"start": start, "exclusions": exclusions, "final": final, "split": sizes}


def fig_cohort_flow(counts: dict):
    w_mm, h_mm = 174, 112
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.axis("off")

    def box(cx, cy, w, h, text, ha="center", pad=None):
        """h=None: fit the box height to the text plus `pad` mm above and below."""
        x = cx if ha == "center" else cx - w / 2 + 3
        t = ax.text(x, cy, text, ha=ha, va="center", fontsize=8, linespacing=1.45)
        if h is None:
            bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
            h = (bb.y1 - bb.y0) + 2 * pad
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="white", ec=REF_GREY, lw=0.8))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": REF_GREY, "lw": 0.8, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 7})

    main_x = 52
    box(main_x, 100, 64, 12, f"Patients in source data\nn = {counts['start']:,}")
    excluded = counts["start"] - counts["final"]
    lines = [f"Excluded (n = {excluded:,})"] + [f"{i}  {label}: {n:,}" for i, label, n in counts["exclusions"]]
    box(128, 72, 84, None, "\n".join(lines), ha="left", pad=3)
    box(main_x, 46, 64, 12, f"Analytic cohort\nn = {counts['final']:,}")
    arrow(main_x, 94, main_x, 52)
    ax.plot([main_x, 128 - 42], [72, 72], color=REF_GREY, lw=0.8)
    arrow(128 - 42 - 0.01, 72, 128 - 42, 72)

    names = (("train", "Training"), ("val", "Validation"), ("test", "Test"))
    xs = (30, 87, 144)
    ax.plot([main_x, main_x], [40, 30], color=REF_GREY, lw=0.8)
    ax.plot([xs[0], xs[-1]], [30, 30], color=REF_GREY, lw=0.8)
    for x, (key, label) in zip(xs, names):
        arrow(x, 30, x, 24)
        n = counts["split"][key]
        box(x, 16, 48, 16, f"{label}\nn = {n:,} ({n / counts['final']:.1%})")
    return fig


# ---- Fig 2 / S3: per-model metric, dot and whisker ------------------------------------------


def fig_metric_forest(results: dict, metric: str, include_baseline: bool):
    rows = _per_model(results)
    models = _models_in(results, include_baseline)
    labels = CINDEX_LABEL if metric == "cindex" else IBS_LABEL
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, (58 + 4 * len(models)) * MM), sharey=True)
    fig.subplots_adjust(left=0.205 if include_baseline else 0.15, right=0.98, top=0.86, bottom=0.36, wspace=0.12)
    ys = np.arange(len(models))[::-1]
    lo_all, hi_all = [], []
    for ax, outcome in zip(axes, OUTCOMES):
        for y, m in zip(ys, models):
            r = rows.get((m, outcome, metric))
            if r is None or not _finite(r["point_estimate"]):
                ax.text(0.5, y, "Not available", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7)
                continue
            st = MODEL_STYLE[m]
            pe, lo, hi = r["point_estimate"], r["ci_lower"], r["ci_upper"]
            ax.errorbar(pe, y, xerr=[[pe - lo], [hi - pe]], fmt=st["marker"], color=st["color"], ms=4.5, elinewidth=1.1,
                        capsize=2.2, capthick=1.0, zorder=3)
            lo_all.append(lo)
            hi_all.append(hi)
        ax.set_title(OUTCOME_LABEL[outcome])
        ax.set_xlabel(labels[outcome])
        _style_axes(ax, "x")
    if metric == "cindex":
        lo_x, hi_x = 0.45, max(0.85, math.ceil((max(hi_all, default=0.8) + 0.02) * 20) / 20)
        for ax in axes:
            ax.axvline(0.5, color=REF_GREY, ls="--", lw=0.8, zorder=1)
    else:
        lo_x = max(0.0, math.floor((min(lo_all, default=0) - 0.01) * 50) / 50)
        hi_x = math.ceil((max(hi_all, default=0.2) + 0.01) * 50) / 50
    for ax in axes:
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(-0.6, len(models) - 0.4)
    axes[0].set_yticks(ys, [MODEL_STYLE[m]["label"] for m in models])
    handles = [Line2D([], [], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"], ls="-", lw=1.1, ms=4.5,
                      label=MODEL_STYLE[m]["label"]) for m in models]
    if metric == "cindex":
        handles.append(Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="No discrimination (0.5)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, bbox_to_anchor=(0.565, 0.0),
               columnspacing=1.5)
    fig.text(0.565, 0.115, "Points: estimate; bars: 95% CI (1,000 patient-level bootstrap draws)", ha="center",
             va="bottom", fontsize=7)
    fig.canvas.draw()
    for ax, letter in zip(axes, "ABC"):
        _panel_letter(fig, ax, letter)
    return fig


# ---- Fig 3: calibration ---------------------------------------------------------------------


def fig_calibration(results: dict):
    cal = results["calibration"]
    models = [m for m in MODELS if m in cal]
    fig = plt.figure(figsize=(FULL_WIDTH, 92 * MM))
    width, gap, left, bottom, height = 0.24, 0.088, 0.08, 0.36, 0.24 * FULL_WIDTH / (92 * MM)
    for i, outcome in enumerate(OUTCOMES):
        ax = fig.add_axes([left + i * (width + gap), bottom, width, height])
        top = 0.0
        for m in models:
            t = cal[m][outcome]
            xs = [g["mean_predicted"] for g in t["groups"]]
            ys = [g["observed"] for g in t["groups"]]
            top = max([top, *xs, *ys])
            st = MODEL_STYLE[m]
            oe = t["overall"]["oe_ratio"]
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], lw=1.0, ms=3.5,
                    label=f"{st['label']} (O/E {oe:.2f})" if _finite(oe) else st["label"], zorder=3)
        step = 0.1 if top * 1.05 <= 0.5 else 0.2
        lim = max(step, math.ceil(top * 1.05 / step) * step)
        ax.plot([0, lim], [0, lim], color=REF_GREY, ls="--", lw=0.8, label="Perfect calibration", zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ticks = np.round(np.arange(0, lim + 1e-9, step), 2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        _style_axes(ax, "both")
        ax.set_title(OUTCOME_LABEL[outcome])
        ax.set_xlabel("Predicted 10-yr risk")
        ax.set_ylabel(f"Observed 10-yr risk\n({OBSERVED_LABEL[outcome]})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=7, handlelength=2.0, borderaxespad=0)
    fig.text(0.5, 0.012, "Points: deciles of predicted 10-yr risk; O/E: observed / expected (mean predicted) risk",
             ha="center", va="bottom", fontsize=7)
    fig.canvas.draw()
    for ax, letter in zip(fig.axes, "ABC"):
        _panel_letter(fig, ax, letter)
    return fig


# ---- Fig 4: risk-band curves ----------------------------------------------------------------


def cumulative_incidence_curve(time, event_type, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Step curve (times, CIF) of cause 1 up to tmax: Aalen-Johansen with any other positive code
    competing; with no competing codes this is 1 - Kaplan-Meier. Same product-limit recursion as
    compass.evaluation.calibration.aalen_johansen_cif (checked against it at tmax)."""
    time = np.asarray(time, dtype=float)
    event_type = np.asarray(event_type).astype(int)
    is_event = (event_type != 0) & (time <= tmax)
    if not is_event.any():
        return np.array([0.0, tmax]), np.array([0.0, 0.0])
    t, inv = np.unique(time[is_event], return_inverse=True)
    d_all = np.bincount(inv, minlength=t.size)
    d_cause = np.bincount(inv, weights=(event_type[is_event] == 1).astype(float), minlength=t.size)
    n = time.size - np.searchsorted(np.sort(time), t, side="left")
    s_after = np.cumprod(1.0 - d_all / n)
    s_before = np.concatenate([[1.0], s_after[:-1]])
    cif = np.cumsum(s_before * d_cause / n)
    return np.concatenate([[0.0], t, [tmax]]), np.concatenate([[0.0], cif, [cif[-1]]])


MIN_AT_RISK = 10  # Fig 4 curves stop at the last time with at least this many patients at risk


def truncate_curve(t: np.ndarray, cif: np.ndarray, time: np.ndarray, min_at_risk: int = MIN_AT_RISK,
                   tmax: float = 10.0) -> tuple[np.ndarray, np.ndarray, float]:
    """(t, cif, t_end): the step curve cut at t_end = the last time with >= min_at_risk patients
    at risk (at risk at t = time >= t, as in the numbers-at-risk table), capped at tmax."""
    srt = np.sort(np.asarray(time, dtype=float))
    t_end = float(min(srt[-min_at_risk], tmax)) if srt.size >= min_at_risk else 0.0
    keep = t <= t_end
    return np.concatenate([t[keep], [t_end]]), np.concatenate([cif[keep], [cif[keep][-1]]]), t_end


def band_curves(assignments: pd.DataFrame, results: dict, horizon: int = 10) -> dict:
    """{(model, outcome): {"curves": [(t, cif)] per band (full, to the horizon), "drawn": the
    same curves truncated at MIN_AT_RISK (truncate_curve), "t_end": per band, "at_risk": [[n at
    RISK_TIMES]] per band, "hr": high-vs-low row}}; the full curves are checked against the saved
    banding table (n per band and the observed risk at the horizon must match)."""
    bands = results["banding"]["bands"]
    out = {}
    for m in MODELS:
        for o in OUTCOMES:
            saved = bands.get(m, {}).get(o)
            if saved is None:
                continue
            if isinstance(saved, dict):
                out[(m, o)] = {"status": saved.get("status", "not available")}
                continue
            sub = assignments[(assignments["model"] == m) & (assignments["outcome"] == o)]
            curves, drawn, t_ends, at_risk = [], [], [], []
            for b in range(3):
                s = sub[sub["band"] == b]
                time, et = s["time_years"].to_numpy(float), s["event_type"].to_numpy(int)
                if len(s) != saved[b]["n"]:
                    raise FigureDataError(f"{m} / {o} / band {b}: {len(s)} patients in the assignments, "
                                          f"{saved[b]['n']} in the saved banding table")
                t, cif = cumulative_incidence_curve(time, et, horizon)
                ref = aalen_johansen_cif(time, et, horizon)
                if not (abs(cif[-1] - ref) < 1e-12 and abs(ref - saved[b]["observed_risk"]) < 1e-9):
                    raise FigureDataError(f"{m} / {o} / band {b}: curve at {horizon} yr {cif[-1]:.6f} vs saved "
                                          f"observed risk {saved[b]['observed_risk']:.6f}")
                curves.append((t, cif))
                tt, cc, t_end = truncate_curve(t, cif, time, tmax=horizon)
                drawn.append((tt, cc))
                t_ends.append(t_end)
                at_risk.append([int((time >= rt).sum()) if rt > 0 else len(s) for rt in RISK_TIMES])
            out[(m, o)] = {"curves": curves, "drawn": drawn, "t_end": t_ends, "at_risk": at_risk, "hr": saved[2],
                           "n": [r["n"] for r in saved]}
    return out


def fig_risk_band_curves(curves: dict):
    """3 x 3 grid laid out in millimetres: one shared band legend at the top, per panel the band
    curves (truncated at MIN_AT_RISK patients at risk), the HR as the panel title, and the full
    numbers-at-risk table underneath."""
    W, H = 174, 240
    left, gap, right = 18, 12, 3
    pw = (W - left - right - 2 * gap) / 3
    curve_h, title_h, xlab_h, table_h, block_gap, header_h = 38, 7, 10, 12, 4, 16
    block = title_h + curve_h + xlab_h + table_h + block_gap
    fig = plt.figure(figsize=(W * MM, H * MM))

    def mm_axes(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    letters = iter("ABCDEFGHI")
    for i, o in enumerate(OUTCOMES):
        top = max([float(cif.max()) for m in MODELS for _, cif in curves.get((m, o), {}).get("curves", [])] or [0.0])
        ylim = max(0.1, math.ceil(top * 1.25 * 10) / 10)
        curve_top = H - header_h - i * block - title_h
        curve_bottom = curve_top - curve_h
        table_bottom = curve_bottom - xlab_h - table_h
        for j, m in enumerate(MODELS):
            x0 = left + j * (pw + gap)
            ax = mm_axes(x0, curve_bottom, pw, curve_h)
            tab = mm_axes(x0, table_bottom, pw, table_h)
            ax.set_xlim(-0.3, 10.3)
            ax.set_ylim(0, ylim)
            ax.set_xticks(RISK_TIMES)
            ax.set_yticks(np.round(np.arange(0, ylim + 1e-9, 0.1 if ylim <= 0.6 else 0.2), 2))
            if j > 0:
                ax.set_yticklabels([])
            _style_axes(ax, "y")
            ax.set_xlabel("Years from landmark", labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{OUTCOME_LABEL[o]}\ncumulative incidence")
            if i == 0:
                fig.text((x0 + pw / 2) / W, (H - 9) / H, MODEL_STYLE[m]["label"], ha="center", va="top",
                         fontsize=8, fontweight="bold")
            fig.text((x0 - 9) / W, (curve_top + 1) / H, next(letters), fontsize=14, fontweight="bold",
                     ha="left", va="bottom")
            tab.axis("off")
            tab.set_xlim(-0.3, 10.3)
            tab.set_ylim(0, 4)
            c = curves.get((m, o))
            if c is None or "status" in c:
                ax.text(5, ylim / 2, "Not available\n(no cut-points fitted)", ha="center", va="center", fontsize=7)
                continue
            for b, (t, cif) in enumerate(c["drawn"]):
                st = BAND_STYLE[b]
                ax.step(t, cif, where="post", color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
            hr = c["hr"]
            kind = "HR" if o == "mortality" else "Cause-specific HR"
            ax.set_title(f"{kind} high vs low\n{hr['hazard_ratio']:.2f} ({hr['hr_ci_lower']:.2f}–{hr['hr_ci_upper']:.2f})"
                         if _finite(hr["hazard_ratio"]) else f"{kind} high vs low\nnot estimable",
                         fontsize=7, linespacing=1.2, pad=3)
            tab.text(-0.3, 3.5, "Number at risk", ha="left", va="center", fontsize=6.5, fontweight="bold")
            for b in range(3):
                st, y = BAND_STYLE[b], 2.5 - b
                tab.text(-1.25, y, st["short"], ha="right", va="center", fontsize=6.5, color=st["color"])
                for x, n in zip(RISK_TIMES, c["at_risk"][b]):
                    tab.text(x, y, f"{n:,}", ha="center", va="center", fontsize=6.5, color=st["color"])
    handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=1.3, label=f"{st['label']} risk band") for st in BAND_STYLE]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.8 / H), ncol=3, fontsize=7,
               handlelength=3.0, columnspacing=2.0, borderaxespad=0)
    fig.text(0.5, 5.5 / H, "Cumulative incidence: 1 − Kaplan-Meier (mortality), Aalen-Johansen with death as a "
             "competing event (MALO, CVD); Int: intermediate band", ha="center", va="bottom", fontsize=6.5)
    fig.text(0.5, 2 / H, f"Curves truncated when fewer than {MIN_AT_RISK} patients remained at risk",
             ha="center", va="bottom", fontsize=6.5)
    return fig


# ---- S1: pairwise C-index differences -------------------------------------------------------


def _comparison_label(c: str) -> str:
    a, b = c.split("_minus_")
    return f"{MODEL_STYLE[a]['label']} − {MODEL_STYLE[b]['label']}"


def fig_pairwise_cindex(results: dict):
    rows = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in results["pairwise"]}
    comps = list(dict.fromkeys(r["comparison"] for r in results["pairwise"]))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 74 * MM), sharey=True)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.87, bottom=0.40, wspace=0.16)
    ys = np.arange(len(comps))[::-1]
    span = 0.0
    for ax, o in zip(axes, OUTCOMES):
        for y, c in zip(ys, comps):
            r = rows.get((c, o, "cindex"))
            if r is None or not _finite(r["point_estimate"]):
                continue
            pe, lo, hi = r["point_estimate"], r["ci_lower"], r["ci_upper"]
            excludes = lo > 0 or hi < 0
            ax.errorbar(pe, y, xerr=[[pe - lo], [hi - pe]], fmt="o", ms=4.5, color=TEXT_COLOUR, elinewidth=1.1,
                        capsize=2.2, mfc=TEXT_COLOUR if excludes else "white", mec=TEXT_COLOUR, zorder=3)
            span = max(span, abs(lo), abs(hi))
        ax.axvline(0, color=REF_GREY, ls="--", lw=0.8, zorder=1)
        ax.set_title(OUTCOME_LABEL[o])
        ax.set_xlabel(f"Difference in C-index\n({CINDEX_LABEL[o].split('(')[1]}")
        _style_axes(ax, "x")
    lim = math.ceil((span + 0.01) * 50) / 50
    step = next(st for st in (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if lim / st <= 2.5)  # <= 2 ticks per side
    k = math.floor((lim - 1e-9) / step)
    ticks = np.round(np.arange(-k, k + 1) * step, 2)  # interior ticks only: no labels on the panel edges
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.6, len(comps) - 0.4)
    axes[0].set_yticks(ys, [_comparison_label(c) for c in comps])
    handles = [Line2D([], [], marker="o", color=TEXT_COLOUR, mfc=TEXT_COLOUR, ls="-", lw=1.1, ms=4.5,
                      label="95% CI excludes 0"),
               Line2D([], [], marker="o", color=TEXT_COLOUR, mfc="white", ls="-", lw=1.1, ms=4.5,
                      label="95% CI includes 0"),
               Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="No difference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.62, 0.0))
    fig.text(0.62, 0.10, "Paired patient-level bootstrap (1,000 draws); positive: first model higher",
             ha="center", va="bottom", fontsize=7)
    fig.canvas.draw()
    for ax, letter in zip(axes, "ABC"):
        _panel_letter(fig, ax, letter)
    return fig


# ---- S2: band hazard ratios -----------------------------------------------------------------


def fig_band_hazard_ratios(results: dict):
    bands = results["banding"]["bands"]
    models = [m for m in MODELS if m in bands]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 70 * MM), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.36, wspace=0.12)
    ys = np.arange(len(models))[::-1].astype(float)
    lo_all, hi_all = [], []
    for ax, o in zip(axes, OUTCOMES):
        for y, m in zip(ys, models):
            rows = bands[m].get(o)
            if not isinstance(rows, list):
                continue
            for b, dy in ((1, 0.14), (2, -0.14)):
                r, st = rows[b], BAND_STYLE[b]
                if not _finite(r["hazard_ratio"]):
                    continue
                pe, lo, hi = r["hazard_ratio"], r["hr_ci_lower"], r["hr_ci_upper"]
                eb = ax.errorbar(pe, y + dy, xerr=[[pe - lo], [hi - pe]], fmt="o", ms=4, color=st["color"],
                                 elinewidth=1.2, capsize=2, zorder=3)
                eb[2][0].set_linestyle(st["ls"])
                lo_all.append(lo)
                hi_all.append(hi)
        ax.axvline(1, color=REF_GREY, ls="--", lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_title(OUTCOME_LABEL[o])
        ax.set_xlabel("Hazard ratio vs low band" if o == "mortality" else "Cause-specific HR vs low band")
        _style_axes(ax, "x")
    lo_x = min(0.8, 0.8 * min(lo_all, default=1.0))
    hi_x = max(hi_all, default=10) * 1.3
    ticks = [t for t in (1, 2, 5, 10, 20, 50) if lo_x <= t <= hi_x]
    for ax in axes:
        ax.set_xlim(lo_x, hi_x)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.6, len(models) - 0.4)
    axes[0].set_yticks(ys, [MODEL_STYLE[m]["label"] for m in models])
    handles = [Line2D([], [], color=BAND_STYLE[b]["color"], ls=BAND_STYLE[b]["ls"], marker="o", ms=4, lw=1.2,
                      label=f"{BAND_STYLE[b]['label']} vs low") for b in (1, 2)]
    handles.append(Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="HR = 1"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.56, 0.0))
    fig.text(0.56, 0.115, "Validation cut-points applied unchanged; Cox (mortality), cause-specific Cox (MALO, CVD); "
             "bars: 95% CI", ha="center", va="bottom", fontsize=7)
    fig.canvas.draw()
    for ax, letter in zip(axes, "ABC"):
        _panel_letter(fig, ax, letter)
    return fig


# ---- S4: deep model minus Cox / Fine-Gray baseline ------------------------------------------

DIFF_METRICS = ("cindex", "ibs")


def check_baseline_comparison(results: dict, baseline: dict) -> None:
    """The baseline evaluation must hold the SAME deep-model results as the locked evaluation
    (same split, patients, horizon, bootstrap; identical deep per-model rows)."""
    for key in ("split", "n_patients", "horizon_years", "n_bootstrap", "bootstrap_seed"):
        if results.get(key) != baseline.get(key):
            raise FigureDataError(f"S4: --baseline-eval-dir differs from --eval-dir in '{key}'")
    key = lambda r: (r["model"], r["outcome"], r["metric_name"])  # noqa: E731
    deep = {key(r): r for r in baseline["per_model"] if r["model"] in MODELS}
    locked = {key(r): r for r in results["per_model"] if r["model"] in MODELS}
    if deep != locked:
        raise FigureDataError("S4: the baseline evaluation's deep-model results differ from the locked evaluation")


def baseline_diff_values(baseline: dict) -> dict:
    """{(metric, outcome, model): (estimate, lower, upper)} of "<model>_minus_cox_baseline"."""
    rows = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in baseline["pairwise"]}
    out = {}
    for metric in DIFF_METRICS:
        for o in OUTCOMES:
            for m in MODELS:
                r = rows.get((f"{m}_minus_cox_baseline", o, metric))
                if r is None:
                    raise FigureDataError(f"S4: no '{m}_minus_cox_baseline' {metric} result for {o}")
                out[(metric, o, m)] = (r["point_estimate"], r["ci_lower"], r["ci_upper"])
    return out


def fig_baseline_differences(baseline: dict):
    """2 x 3 (S1 layout): row 1 ΔC-index, row 2 ΔIBS, deep model minus the Cox / Fine-Gray
    baseline, one row per deep model; filled marker = 95% CI excludes 0. Estimates / CIs are
    drawn as separate artists (gid "pe|..." / "ci|...") for check_baseline_diff_plotted."""
    values = baseline_diff_values(baseline)
    horizon = int(baseline["horizon_years"])
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 112 * MM), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.25, hspace=0.75, wspace=0.16)
    ys = np.arange(len(MODELS))[::-1].astype(float)
    for row, metric in enumerate(DIFF_METRICS):
        span = max(max(abs(values[(metric, o, m)][1]), abs(values[(metric, o, m)][2])) for o in OUTCOMES for m in MODELS)
        unit = 50 if metric == "cindex" else 500
        lim = math.ceil((span * 1.1) * unit) / unit
        step = next(st for st in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2) if lim / st <= 2.5)
        kmax = math.floor((lim - 1e-12) / step)
        ticks = np.round(np.arange(-kmax, kmax + 1) * step, 3)
        for col, o in enumerate(OUTCOMES):
            ax = axes[row, col]
            for y, m in zip(ys, MODELS):
                st = MODEL_STYLE[m]
                pe, lo, hi = values[(metric, o, m)]
                excludes = lo > 0 or hi < 0
                ax.hlines(y, lo, hi, color=st["color"], lw=1.1, zorder=3, gid=f"ci|{metric}|{o}|{m}")
                ax.plot([pe], [y], ls="none", marker=st["marker"], ms=4.5, color=st["color"], mec=st["color"],
                        mfc=st["color"] if excludes else "white", mew=0.9, zorder=4, gid=f"pe|{metric}|{o}|{m}")
            ax.axvline(0, color=REF_GREY, ls="--", lw=0.8, zorder=1)
            ax.set_xlim(-lim, lim)
            ax.set_xticks(ticks)
            ax.set_ylim(-0.6, len(MODELS) - 0.4)
            ax.set_title(OUTCOME_LABEL[o])
            if metric == "cindex":
                ax.set_xlabel(f"ΔC-index ({CINDEX_LABEL[o].split('(')[1]}")
            else:
                ax.set_xlabel("ΔIBS" if o == "mortality" else "ΔIBS (Schoop)")
            _style_axes(ax, "x")
        axes[row, 0].set_yticks(ys, [MODEL_STYLE[m]["label"] for m in MODELS])
    handles = [Line2D([], [], marker="o", color=TEXT_COLOUR, mfc=TEXT_COLOUR, ls="-", lw=1.1, ms=4.5,
                      label="95% CI excludes 0"),
               Line2D([], [], marker="o", color=TEXT_COLOUR, mfc="white", ls="-", lw=1.1, ms=4.5,
                      label="95% CI includes 0"),
               Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="No difference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.565, 0.085))
    fig.text(0.565, 0.008,
             f"Deep model minus Cox (mortality) / Fine-Gray (MALO, CVD) baseline at cohort entry (first valid landmark), "
             f"test split, {horizon}-yr horizon;\npaired patient-level bootstrap (1,000 draws). Positive ΔC-index: deep "
             f"model higher; positive ΔIBS: deep model worse.",
             ha="center", va="bottom", fontsize=6.5, linespacing=1.35)
    fig.canvas.draw()
    for ax, letter in zip(axes.flat, "ABCDEF"):
        _panel_letter(fig, ax, letter)
    return fig


def check_baseline_diff_plotted(fig, baseline: dict) -> int:
    """Every plotted S4 estimate / CI equals the baseline JSON exactly (and each is plotted once)."""
    expected = baseline_diff_values(baseline)
    seen = {}
    for ax in fig.axes:
        for line in ax.lines:
            gid = line.get_gid() or ""
            if gid.startswith("pe|"):
                seen.setdefault(gid[3:], {})["pe"] = float(line.get_xdata()[0])
        for coll in ax.collections:
            gid = coll.get_gid() or ""
            if gid.startswith("ci|"):
                seg = coll.get_segments()[0]
                seen.setdefault(gid[3:], {})["ci"] = (float(seg[0][0]), float(seg[1][0]))
    if len(seen) != len(expected):
        raise FigureDataError(f"S4: {len(seen)} plotted estimates, {len(expected)} in the JSON")
    for (metric, o, m), (pe, lo, hi) in expected.items():
        got = seen.get(f"{metric}|{o}|{m}")
        if got is None or got.get("pe") != pe or got.get("ci") != (lo, hi):
            raise FigureDataError(f"S4: plotted value differs from the JSON for {m} / {o} / {metric}")
    return len(expected)


# ---- Fig 6 / S5: permutation importance -----------------------------------------------------

IMPORTANCE_PREDICTORS = MODELS + ("cox_baseline",)
IMPORTANCE_TOP_N = 15
IMPORTANCE_FOOTNOTE = ("Permutation importance: decrease in C-index when the variable (or group) is randomly shuffled across "
                       "patients; reflects predictive reliance, not causal effect.")


def importance_order(rows: list[dict], outcome: str, top_n: int | None = None) -> list[str]:
    """Units for one outcome, highest mean importance (averaged over the four predictors)
    first; ties by name. top_n keeps the first n."""
    by_unit = {}
    for r in rows:
        if r["outcome"] == outcome:
            by_unit.setdefault(r["unit"], []).append(r["mean_drop"])
    order = sorted(by_unit, key=lambda u: (-float(np.mean(by_unit[u])), u))
    return order[:top_n] if top_n else order


def fig_importance(results: dict, kind: str, top_n: int | None = None):
    """Three stacked panels (mortality, MALO, CVD): one row per group (kind "grouped") or per
    variable (kind "individual", top_n per outcome), the four predictors dodged; point = mean
    drop over the repeats, bar = 2.5-97.5 percentile range. Artists carry gids
    ("pe|outcome|unit|predictor", "ci|...") for check_importance_plotted."""
    rows = results[kind]
    labels = {r["unit"]: r["label"] for r in rows}
    idx = {(r["outcome"], r["unit"], r["predictor"]): r for r in rows}
    orders = {o: importance_order(rows, o, top_n) for o in OUTCOMES}
    n_rows = max(len(v) for v in orders.values())
    row_h = 4.6 if n_rows <= 10 else 3.8
    W, left, right = 174, 44, 4
    panel_h, title_h, xlab_h, top_h, foot_h = n_rows * row_h + 2, 6, 11, 12, 15
    H = top_h + 3 * (title_h + panel_h + xlab_h) + foot_h
    fig = plt.figure(figsize=(W * MM, H * MM))
    offsets = dict(zip(IMPORTANCE_PREDICTORS, np.linspace(0.27, -0.27, len(IMPORTANCE_PREDICTORS))))
    letters = iter("ABC")
    for i, o in enumerate(OUTCOMES):
        bottom = H - top_h - (i + 1) * (title_h + panel_h + xlab_h) + xlab_h
        ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, panel_h / H])
        units = orders[o]
        ys = np.arange(len(units))[::-1].astype(float) + (n_rows - len(units))  # top-aligned
        lo_all, hi_all = [0.0], [0.0]
        for y, u in zip(ys, units):
            for pred in IMPORTANCE_PREDICTORS:
                r = idx.get((o, u, pred))
                if r is None:
                    continue
                st = MODEL_STYLE[pred]
                yy = y + offsets[pred]
                ax.hlines(yy, r["p2_5"], r["p97_5"], color=st["color"], lw=1.0, zorder=3, gid=f"ci|{o}|{u}|{pred}")
                ax.plot([r["mean_drop"]], [yy], ls="none", marker=st["marker"], ms=3.6, color=st["color"], zorder=4,
                        gid=f"pe|{o}|{u}|{pred}")
                lo_all.append(r["p2_5"])
                hi_all.append(r["p97_5"])
        span = max(hi_all) - min(lo_all)
        ax.set_xlim(min(lo_all) - 0.06 * span - 1e-4, max(hi_all) + 0.06 * span + 1e-4)
        ax.axvline(0, color=REF_GREY, ls="--", lw=0.8, zorder=1)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_yticks(ys, [labels[u] for u in units])
        ax.set_xlabel(f"Decrease in C-index ({CINDEX_LABEL[o].split('(')[1]}", labelpad=2)
        ax.set_title(OUTCOME_LABEL[o], loc="left", fontweight="bold")
        _style_axes(ax, "x")
        fig.text(2 / W, (bottom + panel_h + 1.5) / H, next(letters), fontsize=14, fontweight="bold", ha="left", va="bottom")
    handles = [Line2D([], [], color=MODEL_STYLE[p]["color"], marker=MODEL_STYLE[p]["marker"], ls="-", lw=1.0, ms=3.6,
                      label=MODEL_STYLE[p]["label"]) for p in IMPORTANCE_PREDICTORS]
    handles.append(Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="No reliance (0)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 1 / H), ncol=len(handles), fontsize=7,
               columnspacing=1.3, handlelength=2.0, borderaxespad=0)
    what = ("groups shuffled jointly with one shared permutation; rows ordered by mean importance across the four predictors"
            if kind == "grouped" else
            f"top {top_n} variables per outcome by mean importance across the four predictors; smoking status: its three "
            "indicators shuffled together")
    notes = [IMPORTANCE_FOOTNOTE,
             (f"Test split, first valid landmark, {int(results['horizon_years'])}-yr horizon; points: mean over "
              f"{int(results['n_repeats'])} permutations; bars: 2.5–97.5 percentile range across permutations;"),
             f"{what}."]
    fig.text(0.5, 1.5 / H, "\n".join(notes), ha="center", va="bottom", fontsize=6.5, linespacing=1.35)
    return fig


def check_importance_plotted(fig, results: dict, kind: str, top_n: int | None = None) -> int:
    """Every plotted mean / range equals the JSON exactly, the plotted units are the expected
    (ordered, top-n) ones, and nothing else is plotted. Returns the number of points checked."""
    rows = results[kind]
    idx = {(r["outcome"], r["unit"], r["predictor"]): r for r in rows}
    expected = {(o, u, p) for o in OUTCOMES for u in importance_order(rows, o, top_n) for p in IMPORTANCE_PREDICTORS
                if (o, u, p) in idx}
    seen = {}
    for ax in fig.axes:
        for line in ax.lines:
            gid = line.get_gid() or ""
            if gid.startswith("pe|"):
                seen.setdefault(tuple(gid.split("|")[1:]), {})["pe"] = float(line.get_xdata()[0])
        for coll in ax.collections:
            gid = coll.get_gid() or ""
            if gid.startswith("ci|"):
                seg = coll.get_segments()[0]
                seen.setdefault(tuple(gid.split("|")[1:]), {})["ci"] = (float(seg[0][0]), float(seg[1][0]))
    if set(seen) != expected:
        raise FigureDataError(f"importance ({kind}): plotted units differ from the JSON's ordered selection")
    for key in expected:
        r = idx[key]
        if seen[key].get("pe") != r["mean_drop"] or seen[key].get("ci") != (r["p2_5"], r["p97_5"]):
            raise FigureDataError(f"importance ({kind}): plotted value differs from the JSON for {key}")
    return len(expected)


# ---- Fig 5: serial evaluation (secondary analysis 2) ----------------------------------------

SERIAL_FULL = tuple(f"{m}_full" for m in MODELS)
SERIAL_ROW1 = SERIAL_FULL + ("landmark_cox",)
SERIAL_LABEL = {**{f"{m}_full": f"{MODEL_STYLE[m]['label']}, full history" for m in MODELS},
                "landmark_cox": "Landmark Cox/Fine-Gray"}
SERIAL_STYLE = {**{f"{m}_full": MODEL_STYLE[m] for m in MODELS}, "landmark_cox": MODEL_STYLE["cox_baseline"]}
SERIAL_SAME_RUN_KEYS = ("analysis", "data", "horizon_years", "prediction_points", "n_bootstrap", "bootstrap_seed",
                        "predictors", "model_sources", "checkpoint_sha256")


def check_serial_runs(test: dict, val: dict) -> None:
    """The serial test run must be the --final test run of the same models, checkpoints and
    landmark Cox fit as the validation run (fitted there, reused unchanged)."""
    if test.get("split") != "test" or val.get("split") != "val":
        raise FigureDataError(f"serial runs: expected splits test / val, got {test.get('split')} / {val.get('split')}")
    for key in SERIAL_SAME_RUN_KEYS:
        if test.get(key) != val.get(key):
            raise FigureDataError(f"serial test and validation runs differ in '{key}'")
    if not (test.get("checkpoints_unchanged") and val.get("checkpoints_unchanged")):
        raise FigureDataError("serial runs: checkpoints changed during a run")
    lt, lv = test["landmark_cox"], val["landmark_cox"]
    if lt["sha256"] != lv["sha256"] or lt["fitted_in_this_run"] or not lv["fitted_in_this_run"]:
        raise FigureDataError("serial runs: the landmark Cox fit was not fitted on validation and reused unchanged on test")
    for k in test["points"]:
        if test["points"][k]["n_train_evaluated"] != val["points"][k]["n_train_evaluated"]:
            raise FigureDataError(f"serial runs: different training reference at k = {k}")


def serial_values(results: dict) -> dict:
    """{(row, outcome, predictor, k): (estimate, lower, upper)} straight from the JSON: row 1 =
    C-index of the full-history deep models + landmark Cox; row 2 = C-index difference, the same
    model's full minus snapshot."""
    out = {}
    for k in results["prediction_points"]:
        p = results["points"][str(k)]
        per_model = {(r["model"], r["outcome"], r["metric_name"]): r for r in p["per_model"]}
        pairwise = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in p["pairwise"]}
        for o in OUTCOMES:
            for pred in SERIAL_ROW1:
                r = per_model[(pred, o, "cindex")]
                out[(1, o, pred, int(k))] = (r["point_estimate"], r["ci_lower"], r["ci_upper"])
            for m in MODELS:
                r = pairwise[(f"{m}_full_minus_{m}_snapshot", o, "cindex")]
                out[(2, o, f"{m}_full", int(k))] = (r["point_estimate"], r["ci_lower"], r["ci_upper"])
    return out


def serial_stars(results: dict) -> set:
    """{(outcome, "<model>_full", k)} whose paired C-index difference vs landmark Cox (2a,
    "<model>_full_minus_landmark_cox") has a 95% CI excluding 0."""
    out = set()
    for k in results["prediction_points"]:
        pairwise = {(r["comparison"], r["outcome"], r["metric_name"]): r for r in results["points"][str(k)]["pairwise"]}
        for o in OUTCOMES:
            for pred in SERIAL_FULL:
                r = pairwise[(f"{pred}_minus_landmark_cox", o, "cindex")]
                if r["ci_lower"] > 0 or r["ci_upper"] < 0:
                    out.add((o, pred, int(k)))
    return out


def _serial_flags(results: dict) -> dict:
    """{outcome: [k with < MIN_EVENTS events]} from the JSON flags."""
    return {o: [int(k) for k in results["prediction_points"] if o in results["points"][str(k)]["flags"]] for o in OUTCOMES}


def fig_serial(results: dict):
    """2 x 3: row 1 C-index (95% CI) by prediction point (deep full history + landmark Cox),
    row 2 C-index difference full minus snapshot; filled marker = CI excludes 0. Each estimate /
    CI is drawn as its own artist (gid "pe|..." / "ci|...") so check_serial_plotted can read the
    plotted numbers back."""
    values = serial_values(results)
    flags = _serial_flags(results)
    stars = serial_stars(results)
    points = [int(k) for k in results["prediction_points"]]
    horizon = int(results["horizon_years"])
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 142 * MM))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.30, hspace=0.62, wspace=0.40)
    dodge1 = dict(zip(SERIAL_ROW1, np.linspace(-0.18, 0.18, len(SERIAL_ROW1))))
    dodge2 = dict(zip(SERIAL_FULL, np.linspace(-0.12, 0.12, len(SERIAL_FULL))))
    for col, o in enumerate(OUTCOMES):
        for row, preds, dodge in ((1, SERIAL_ROW1, dodge1), (2, SERIAL_FULL, dodge2)):
            ax = axes[row - 1, col]
            for pred in preds:
                st = SERIAL_STYLE[pred]
                xs = [k + dodge[pred] for k in points]
                pes = [values[(row, o, pred, k)][0] for k in points]
                ax.plot(xs, pes, color=st["color"], lw=0.6, alpha=0.8, zorder=2)
                for x, k in zip(xs, points):
                    pe, lo, hi = values[(row, o, pred, k)]
                    ax.vlines(x, lo, hi, color=st["color"], lw=1.0, zorder=3, gid=f"ci|{row}|{o}|{pred}|{k}")
                    filled = row == 1 or lo > 0 or hi < 0
                    ax.plot([x], [pe], ls="none", marker=st["marker"], ms=4.2, color=st["color"], mec=st["color"],
                            mfc=st["color"] if filled else "white", mew=0.9, zorder=4, gid=f"pe|{row}|{o}|{pred}|{k}")
                    if row == 1 and (o, pred, k) in stars:  # 2a: CI of full minus landmark Cox excludes 0
                        ax.annotate("*", (x, hi), xytext=(0, 0.5), textcoords="offset points", ha="center",
                                    va="bottom", fontsize=8, color=st["color"], gid=f"star|{o}|{pred}|{k}")
            lows = [values[(row, o, p, k)][1] for p in preds for k in points]
            highs = [values[(row, o, p, k)][2] for p in preds for k in points]
            if row == 1:
                headroom = 0.03 if any(s[0] == o for s in stars) else 0.005  # room for the asterisks
                ax.set_ylim(math.floor((min(lows) - 0.005) * 20) / 20, math.ceil((max(highs) + headroom) * 20) / 20)
                ax.set_ylabel(f"{horizon}-yr {CINDEX_LABEL[o]}")
                ax.set_title(OUTCOME_LABEL[o])
            else:
                lim = math.ceil((max(abs(min(lows)), abs(max(highs))) + 0.005) * 50) / 50
                ax.set_ylim(-lim, lim)
                ax.axhline(0, color=REF_GREY, ls="--", lw=0.8, zorder=1)
                ax.set_ylabel("ΔC-index, full − snapshot")
                ax.set_title(f"{OUTCOME_LABEL[o]}: full history vs snapshot")
            ax.set_xlim(points[0] - 0.4, points[-1] + 0.4)
            ax.set_xticks(points, [f"{k}†" if k in flags[o] else str(k) for k in points])
            ax.set_xlabel("Prediction point, years after index")
            _style_axes(ax, "y")
    handles = [Line2D([], [], color=SERIAL_STYLE[p]["color"], marker=SERIAL_STYLE[p]["marker"], ls="-", lw=1.0, ms=4.2,
                      label=SERIAL_LABEL[p]) for p in SERIAL_ROW1]
    handles += [Line2D([], [], marker="o", color=TEXT_COLOUR, mfc=TEXT_COLOUR, ls="none", ms=4.2,
                       label="Difference: 95% CI excludes 0"),
                Line2D([], [], marker="o", color=TEXT_COLOUR, mfc="white", ls="none", ms=4.2,
                       label="Difference: 95% CI includes 0"),
                Line2D([], [], color=REF_GREY, ls="--", lw=0.8, label="No difference")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.115),
               columnspacing=1.4, handlelength=2.0)
    flagged = [(o, k) for o in OUTCOMES for k in flags[o]]
    dagger = ""
    if flagged:
        parts = [f"{results['points'][str(k)]['counts'][o]['n_events']} {OUTCOME_LABEL[o]} events at {k} yr" for o, k in flagged]
        dagger = f"† Fewer than 30 events within {horizon} yr ({', '.join(parts)}); interpret with caution."
    notes = [
        f"Prediction points: the most recent visit within the year before 1, 2 and 3 years after the index date; "
        f"{horizon}-yr horizon from that visit.",
        "Full history: all visits up to the prediction point; snapshot: a single visit holding each variable's most "
        "recent value (same checkpoints).",
        "Landmark Cox (mortality) / Fine-Gray (MALO, CVD) fitted per prediction point on the training split. "
        "Bars: 95% CI (1,000 paired patient-level bootstrap draws).",
    ] + (["* Paired difference vs landmark Cox / Fine-Gray, 95% CI excludes 0."] if stars else []) \
      + ([dagger] if dagger else [])
    fig.text(0.5, 0.008, "\n".join(notes), ha="center", va="bottom", fontsize=6.5, linespacing=1.35)
    fig.canvas.draw()
    for ax, letter in zip(axes.flat, "ABCDEF"):
        _panel_letter(fig, ax, letter)
    return fig


def check_serial_plotted(fig, results: dict) -> int:
    """Every plotted estimate and CI must equal the JSON value exactly, and every JSON value must
    be plotted once. Returns the number of (estimate, CI) pairs checked."""
    expected = serial_values(results)
    seen = {}
    for ax in fig.axes:
        for line in ax.lines:
            gid = line.get_gid() or ""
            if gid.startswith("pe|"):
                seen.setdefault(gid[3:], {})["pe"] = float(line.get_ydata()[0])
        for coll in ax.collections:
            gid = coll.get_gid() or ""
            if gid.startswith("ci|"):
                seg = coll.get_segments()[0]
                seen.setdefault(gid[3:], {})["ci"] = (float(seg[0][1]), float(seg[1][1]))
    if len(seen) != len(expected):
        raise FigureDataError(f"Fig 5: {len(seen)} plotted estimates, {len(expected)} in the JSON")
    for (row, o, pred, k), (pe, lo, hi) in expected.items():
        got = seen.get(f"{row}|{o}|{pred}|{k}")
        if got is None or got.get("pe") != pe or got.get("ci") != (lo, hi):
            raise FigureDataError(f"Fig 5: plotted value differs from the JSON for {pred} / {o} / k = {k} (row {row})")
    drawn_stars = set()
    for ax in fig.axes:
        for t in ax.texts:
            gid = t.get_gid() or ""
            if gid.startswith("star|"):
                o, pred, k = gid.split("|")[1:]
                drawn_stars.add((o, pred, int(k)))
    if drawn_stars != serial_stars(results):
        raise FigureDataError("Fig 5: asterisks differ from the JSON's 2a (full minus landmark Cox) results")
    return len(expected)


# ---- main -----------------------------------------------------------------------------------


def merge_baseline_rows(results: dict, baseline_results: dict) -> dict:
    """results with the cox_baseline per-model rows of another evaluation of the SAME split,
    patients, horizon and bootstrap draws added (deep-model rows are not touched)."""
    for key in ("split", "n_patients", "horizon_years", "n_bootstrap", "bootstrap_seed"):
        if results.get(key) != baseline_results.get(key):
            raise FigureDataError(f"--baseline-eval-dir differs from --eval-dir in '{key}'")
    rows = [r for r in baseline_results["per_model"] if r["model"] == "cox_baseline"]
    if not rows:
        raise FigureDataError("--baseline-eval-dir has no 'cox_baseline' results")
    return {**results, "per_model": [r for r in results["per_model"] if r["model"] != "cox_baseline"] + rows}


def make_all(eval_dir, out_dir, flow_path, split_path, assignments_path, include_baseline: bool = False,
             baseline_eval_dir=None, serial_dir=None, serial_val_dir=None, importance_dir=None) -> list[Path]:
    serial = None
    if serial_dir:
        if not serial_val_dir:
            raise FigureDataError("--serial-dir needs --serial-val-dir (the validation run, for the run checks)")
        serial = json.loads((Path(serial_dir) / "metrics.json").read_text())
        check_serial_runs(serial, json.loads((Path(serial_val_dir) / "metrics.json").read_text()))
    eval_dir, out_dir = Path(eval_dir), Path(out_dir)
    results = json.loads((eval_dir / "metrics.json").read_text())
    baseline = None
    if include_baseline and baseline_eval_dir:
        baseline = json.loads((Path(baseline_eval_dir) / "metrics.json").read_text())
        check_baseline_comparison(results, baseline)
        results = merge_baseline_rows(results, baseline)
    flow = json.loads(Path(flow_path).read_text())
    split = json.loads(Path(split_path).read_text())
    assignments = pd.read_csv(assignments_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    paths = []
    paths += save(fig_cohort_flow(cohort_flow_counts(flow, split)), out_dir, "fig1_cohort_flow")
    paths += save(fig_metric_forest(results, "cindex", include_baseline), out_dir, "fig2_cindex")
    paths += save(fig_calibration(results), out_dir, "fig3_calibration")
    paths += save(fig_risk_band_curves(band_curves(assignments, results)), out_dir, "fig4_risk_band_curves")
    paths += save(fig_pairwise_cindex(results), out_dir, "figS1_pairwise_cindex")
    paths += save(fig_band_hazard_ratios(results), out_dir, "figS2_band_hazard_ratios")
    paths += save(fig_metric_forest(results, "ibs", include_baseline), out_dir, "figS3_ibs")
    if baseline is not None:
        fig = fig_baseline_differences(baseline)
        check_baseline_diff_plotted(fig, baseline)
        paths += save(fig, out_dir, "figS4_baseline_differences")
    if serial is not None:
        fig = fig_serial(serial)
        check_serial_plotted(fig, serial)
        paths += save(fig, out_dir, "fig5_serial_evaluation")
    if importance_dir:
        importance = json.loads((Path(importance_dir) / "importance.json").read_text())
        for kind, top_n, name in (("grouped", None, "fig6_importance_grouped"),
                                  ("individual", IMPORTANCE_TOP_N, "figS5_importance_top15")):
            fig = fig_importance(importance, kind, top_n)
            check_importance_plotted(fig, importance, kind, top_n)
            paths += save(fig, out_dir, name)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="outputs/final_test")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--flow", default="data/processed/flow_log.json")
    p.add_argument("--split-file", default="runs/tune_v2/split.json", help="Only the number of IDs per split is read.")
    p.add_argument("--band-assignments", default="runs/final_test/band_assignments.csv")
    p.add_argument("--include-baseline", action="store_true", help="Grey baseline row in Fig 2 / S3 (model 'cox_baseline').")
    p.add_argument("--baseline-eval-dir", default=None, help="Evaluation dir holding the cox_baseline results.")
    p.add_argument("--serial-dir", default=None, help="Serial evaluation test run (scripts/serial_evaluate.py --final) -> Fig 5.")
    p.add_argument("--serial-val-dir", default=None, help="Serial evaluation validation run (checked against the test run).")
    p.add_argument("--importance-dir", default=None, help="scripts/feature_importance.py output -> Fig 6 and S5.")
    args = p.parse_args(argv)
    try:
        paths = make_all(args.eval_dir, args.out_dir, args.flow, args.split_file, args.band_assignments,
                         args.include_baseline, args.baseline_eval_dir, args.serial_dir, args.serial_val_dir,
                         args.importance_dir)
    except FigureDataError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 3
    font = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=STYLE["font.sans-serif"]))
    print(f"font: {Path(font).name}; no text overlaps; wrote {len(paths)} files to {args.out_dir}/:")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

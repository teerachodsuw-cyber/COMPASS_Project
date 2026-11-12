"""Calibration plot at the horizon: one figure per outcome, mean predicted risk (x) vs observed
risk (y; Kaplan-Meier for mortality, Aalen-Johansen with death competing for MALO / CVD) per
decile of predicted risk, one line per model, with the line of perfect calibration. Plots
aggregates only (decile means), never individual patients.
"""

from pathlib import Path

from matplotlib.figure import Figure

from compass.visualization.discrimination_calibration import _MODEL_COLORS


def plot_calibration(calibration: dict, outcome: str) -> Figure:
    """calibration: {model: {outcome: compass.evaluation.calibration.calibration_table(...)}}."""
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    top = 0.0
    estimator = horizon = None
    for model, per in calibration.items():
        table = per[outcome]
        estimator, horizon = table["estimator"], table["horizon_years"]
        xs = [g["mean_predicted"] for g in table["groups"]]
        ys = [g["observed"] for g in table["groups"]]
        top = max([top, *xs, *ys])
        ax.plot(xs, ys, "o-", color=_MODEL_COLORS.get(model), markersize=5,
                label=f"{model} (O/E {table['overall']['oe_ratio']:.2f})")
    top = min(1.0, top * 1.08) if top > 0 else 1.0
    ax.plot([0, top], [0, top], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"Mean predicted {horizon:g}-yr risk (decile)")
    ax.set_ylabel(f"Observed {horizon:g}-yr risk ({estimator})")
    ax.set_title(f"Calibration: {outcome}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def save_calibration_plots(calibration: dict, outcomes, out_dir) -> list[Path]:
    """One PNG per outcome: <out_dir>/calibration_<outcome>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for outcome in outcomes:
        path = out_dir / f"calibration_{outcome}.png"
        plot_calibration(calibration, outcome).savefig(path, dpi=150)
        paths.append(path)
    return paths

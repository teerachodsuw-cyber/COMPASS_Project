"""Discrimination-calibration scatter: x = time-dependent (Uno's) C-index (higher better),
y = integrated Brier score (lower better), one subplot per outcome, one point per model with
bootstrap CI error bars.
"""

import matplotlib.figure
import matplotlib.pyplot as plt

_MODEL_COLORS = {
    "dynamic_deephit": "#1f77b4",
    "tracer": "#ff7f0e",
    "survlatent_ode": "#2ca02c",
    "cox_baseline": "#6E6E6E",
}


def plot_discrimination_calibration_scatter(
    bootstrap_results: dict,
    outcomes: list = ("mortality", "malo", "cvd"),
    horizon: int = 10,
) -> matplotlib.figure.Figure:
    """bootstrap_results: {model_name: bootstrap_ci_single_model(...) DataFrame}, one entry
    per model, each with columns [outcome, horizon, metric_name, point_estimate, ci_lower,
    ci_upper] (see compass.evaluation.bootstrap.bootstrap_ci_single_model).
    """
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5 * len(outcomes), 5), squeeze=False)
    axes = axes[0]

    for ax, outcome in zip(axes, outcomes):
        for model_name, df in bootstrap_results.items():
            sub = df[(df["outcome"] == outcome) & (df["horizon"] == horizon)]
            cindex_row = sub[sub["metric_name"] == "cindex"].iloc[0]
            ibs_row = sub[sub["metric_name"] == "ibs"].iloc[0]

            x = cindex_row["point_estimate"]
            x_err = [[max(0.0, x - cindex_row["ci_lower"])], [max(0.0, cindex_row["ci_upper"] - x)]]
            y = ibs_row["point_estimate"]
            y_err = [[max(0.0, y - ibs_row["ci_lower"])], [max(0.0, ibs_row["ci_upper"] - y)]]

            color = _MODEL_COLORS.get(model_name)
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", markersize=8, label=model_name, color=color, capsize=4)

        ax.set_title(f"{outcome} @ {horizon}yr")
        ax.set_xlabel("Time-dependent C-index (higher is better)")
        ax.set_ylabel("Integrated Brier Score (lower is better)")

    axes[0].legend(loc="best")
    fig.tight_layout()
    return fig

import pandas as pd
from typing import List

import plotly.graph_objects as go


"""
benchmark_columns = ["116", "230", "665"]
target_column = "640"
"""

def generate_distribution_comparison(
    data_df: pd.DataFrame,
    benchmark_columns: List[str],
    target_column: str,
) -> go.Figure:
    """
    Plot min/max range and median across load profiles, sorted by median,
    with one highlighted target profile.

    Notes
    -----
    - benchmark_columns and target_column are expected WITHOUT 'load_' prefix
      (e.g. "116", "230", "665"; target "640").
    - Returns a Plotly Figure for direct use in Dash (dcc.Graph).
    """

    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    # Basic input checks (fail fast, clear error)
    missing = [c for c in (ref_cols + [target_col]) if c not in data_df.columns]
    if missing:
        raise KeyError(f"Missing columns in data_df: {missing}")

    # Colors (kept from original)
    c_min = "#4C78A8"     # blue
    c_max = "#54A24B"     # green
    c_med = "#F58518"     # orange
    c_target = "#E45756"  # red
    c_band = "lightgrey"  # fill

    # Reference statistics
    ref_mins = data_df[ref_cols].min().rename("min")
    ref_maxs = data_df[ref_cols].max().rename("max")
    ref_meds = data_df[ref_cols].median().rename("median")

    refs = pd.concat([ref_mins, ref_meds, ref_maxs], axis=1)
    refs.loc[:, "name"] = benchmark_columns
    # Index at this point is like "load_116"... we want ids "116" for robust lookup
    refs.index = refs.index.str.replace("load_", "", regex=False)

    # Target statistics
    target_min = data_df[target_col].min()
    target_max = data_df[target_col].max()
    target_med = data_df[target_col].median()

    # Add target (index = "640" etc.)
    refs.loc[target_column] = {
        "min": target_min,
        "median": target_med,
        "max": target_max,
        "name": target_column,
    }

    # Sort by median (same behavior as original)
    refs = refs.sort_values("median")

    # X axis: ordinal index 0..N-1 (same as original)
    x = list(range(len(refs)))

    fig = go.Figure()

    # Band between min and max
    fig.add_trace(
        go.Scatter(
            x=x,
            y=refs["min"].values,
            mode="lines",
            name="min",
            line=dict(color=c_min, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=refs["max"].values,
            mode="lines",
            name="max",
            line=dict(color=c_max, width=2),
            fill="tonexty",
            fillcolor="rgba(220,220,220,0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=refs["median"].values,
            mode="lines",
            name="median",
            line=dict(color=c_med, width=2),
        )
    )

    # Highlight target
    if target_column in refs.index:
        x_target = int(refs.index.get_loc(target_column))
        y_tmin = float(refs.loc[target_column, "min"])
        y_tmed = float(refs.loc[target_column, "median"])
        y_tmax = float(refs.loc[target_column, "max"])

        # dashed vertical line
        fig.add_shape(
            type="line",
            x0=x_target,
            x1=x_target,
            y0=y_tmin,
            y1=y_tmax,
            line=dict(color=c_target, width=2, dash="dash"),
        )

        # target points
        fig.add_trace(
            go.Scatter(
                x=[x_target, x_target, x_target],
                y=[y_tmin, y_tmed, y_tmax],
                mode="markers",
                name="target",
                marker=dict(color=c_target, size=9),
            )
        )

        # match original: only label target on x-axis
        fig.update_xaxes(
            tickmode="array",
            tickvals=[x_target],
            ticktext=[target_column],
        )

    fig.update_layout(
        title=" ",
        xaxis_title="Profil-Index (sortiert nach Median)",
        yaxis_title="Leistung (kW)",
        margin=dict(l=50, r=30, t=55, b=45),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08, yanchor="bottom"),
        template="plotly_dark",
        height=520,
    )

    return fig

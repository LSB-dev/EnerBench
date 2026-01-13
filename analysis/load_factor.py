import pandas as pd
from typing import List
from matplotlib import pyplot as plt

"""
benchmark_columns = ["116", "230", "665"]
target_column = "640"
"""

def generate_distribution_comparison(data_df:  pd.DataFrame, benchmark_columns: List[str], target_column: str):
    """
    Plot min/max range and median across load profiles, sorted by median,
    with one highlighted target profile.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    # Colors
    c_min = "#4C78A8"  # blue
    c_max = "#54A24B"  # green
    c_med = "#F58518"  # orange
    c_target = "#E45756"  # red
    c_band = "lightgrey"  # fill

    # Reference statistics
    ref_mins = data_df[ref_cols].min().rename("min")
    ref_maxs = data_df[ref_cols].max().rename("max")
    ref_meds = data_df[ref_cols].median().rename("median")

    refs = pd.concat([ref_mins, ref_meds, ref_maxs], axis=1)
    refs.loc[:,"name"] = benchmark_columns

    # Target statistics
    target_min = data_df[target_col].min()
    target_max = data_df[target_col].max()
    target_med = data_df[target_col].median()

    # Add target to dataset
    refs.loc[target_column] = {
        "min": target_min,
        "median": target_med,
        "max": target_max,
        "name": target_column
    }

    refs = refs.sort_values("median")

    # Order references
    x = range(len(refs))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, refs["min"].values, color=c_min, linewidth=2, label="min")
    ax.plot(x, refs["max"].values, color=c_max, linewidth=2, label="max")
    ax.fill_between(x, refs["min"].values, refs["max"].values, color=c_band, alpha=0.35)
    ax.plot(x, refs["median"].values, color=c_med, linewidth=2, label="median")

    # Highlight target
    x_target =  refs.index.get_loc(target_column)
    ax.vlines(
        x_target,
        ymin=refs.loc[target_column, "min"],
        ymax=refs.loc[target_column, "max"],
        colors=c_target,
        linestyles="dashed",
        linewidth=2,
    )
    ax.scatter(
        [x_target] * 3,
        refs.loc[target_column, ["min", "median", "max"]],
        color=c_target,
        zorder=5,
        label="_nolegend_",
    )


    ax.set_xticks([x_target])
    ax.set_xticklabels([target_column])

    ax.set_ylabel("Load / kW")
    ax.set_title("Min/Max Range With Median Across Load Profiles(sorted by median)")
    ax.legend()
    fig.tight_layout()

    return fig


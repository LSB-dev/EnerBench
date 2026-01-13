import pandas as pd
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from helpers.weather_mapping import WEATHER_LABELS

# coco and dow won't be used:
# coco is not continuous
# no explanation about dow

"""
benchmark_columns = ["116", "230", "665"]
target_column = "640"
"""

# Colors
c_min = "#4C78A8"  # blue
c_max = "#54A24B"  # green
c_med = "#F58518"  # orange
c_target = "#E45756"  # red
c_band = "lightgrey"  # fill

def weather_r2(df, load_id, weather_vars):
    y = df[f"load_{load_id}"].to_numpy()
    X = df[[f"{v}_{load_id}" for v in weather_vars]].to_numpy()

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]

    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)

    return pd.Series(y).corr(pd.Series(y_hat), method="spearman")

def plot_dependency_boxplots(
    all_corr: pd.DataFrame,
    target_col: str,
    target_column: str,
    title: str = "Weather–load dependency across load profiles",
):
    """
    Boxplot of dependency values per variable across reference load profiles,
    with target profile highlighted by a horizontal line.

    Parameters
    ----------
    all_corr : DataFrame
        Rows = variables, columns = load profiles.
    target_col : str
        Column name of target load profile.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    df = all_corr.copy()

    ref_cols = [c for c in df.columns if c != target_col]


    # Data for boxplots (list of arrays, one per variable)
    data = [df.loc[var, ref_cols].dropna().values for var in df.index]
    target_vals = df[target_col].values
    x = np.arange(len(data))


    fig, ax = plt.subplots(figsize=(10, 4))

    # Dotted baseline at 0 (behind everything)
    ax.axhline(0, linestyle=":", linewidth=1.5, color="#9CA3AF", zorder=0)

    # Boxplot
    """bp = ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=c_band, alpha=0.6),
        medianprops=dict(color=c_med, linewidth=2),  # orange median
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )"""

    # Violinplot
    vp = ax.violinplot(
        data,
        positions=x,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor("lightgrey")
        body.set_alpha(0.5)

    vp["cmedians"].set_color("lightgrey")
    vp["cmedians"].set_linewidth(2)

    # --- scatter reference points inside violins ---
    rng = np.random.default_rng(42)

    for i, vals in enumerate(data):
        jitter = rng.normal(loc=0.0, scale=0.04, size=len(vals))
        ax.scatter(
            np.full_like(vals, x[i]) + jitter,
            vals,
            s=12,
            color=c_min,
            alpha=0.6,
            zorder=3,
        )

    # Target lines
    ax.plot(x, target_vals, linewidth=2.0, alpha=0.9, zorder=6, color=c_target)
    ax.scatter(x, target_vals, s=90, marker="D", color=c_target, zorder=7)

    # Set weather labels on x
    xticklabels = [WEATHER_LABELS.get(v, v) for v in df.index]
    ax.set_xticks(np.arange(len(df.index)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.set_ylabel("Spearman correlation (ρ)")
    ax.set_title(title)

    ax.set_ylim(*(-1,1))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgrey", edgecolor="none", alpha=0.5,
              label="Reference distribution"),
        Line2D([0], [0], color=c_med, lw=2, label="Reference median"),
        Line2D([0], [0], marker="D", color=c_target,
               markerfacecolor="#E45756", markersize=8,
               label=target_column),
    ]

    ax.legend(
        handles=legend_elements
    )

    fig.tight_layout()

    return fig

def generate_distribution_comparison(data_df:  pd.DataFrame, benchmark_columns: List[str], target_column: str):

    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    weather_vars = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']

    # Drop weather vars with high proportion of missing values
    miss_prop = data_df.isna().sum()/len(data_df)
    high_missing = miss_prop[miss_prop > 0.3].index
    base_vars_high_missing = (
        high_missing
        .str.rsplit("_", n=1)
        .str[0]
    )
    weather_vars_to_drop = sorted(
        set(base_vars_high_missing).intersection(weather_vars)
    )
    weather_vars = [
        v for v in weather_vars
        if v not in weather_vars_to_drop
    ]

    all_cols = ref_cols + [target_col]
    all_col_names = benchmark_columns + [target_column]

    all_corr = []
    for i in range(len(all_cols)):
        corr = data_df[[all_cols[i]] + [f"{v}_{all_col_names[i]}" for v in weather_vars]].corr(method="spearman")
        corr = corr[all_cols[i]].drop(all_cols[i])
        corr.index = weather_vars
        all_corr.append(corr)

    all_corr = pd.concat(all_corr, axis = 1)

    all_corr = all_corr.dropna(how="all")

    # R2 from linear regression
    all_r2 = []
    for col_name in all_col_names:
        all_r2.append(weather_r2(data_df, col_name, weather_vars = all_corr.index))

    all_corr.loc["Overall Weather Dependence",:] = all_r2

    fig = plot_dependency_boxplots(all_corr, target_col, target_column, title = "Weather–load dependency across load profiles")

    return fig



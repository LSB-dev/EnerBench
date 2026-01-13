# Responsible:  LSB

import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from analysis.self_similarity.metrics import evaluation_metrics
from shared_util import check_all_columns_are_in_df

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _lag_similarity(df: pd.DataFrame, metric="sMAPE", lags=96):
    shifted_df = df.shift(lags)[lags:]
    original_df = df[lags:]
    assert metric in evaluation_metrics
    metric_func = evaluation_metrics[metric]
    scores = {
        col: metric_func(original_df[col], shifted_df[col]) for col in original_df.columns
    }
    return scores


def _lag_1d_similarity(df: pd.DataFrame, metric="sMAPE"):
    return _lag_similarity(df, metric=metric, lags=96)


def _lag_1w_similarity(df: pd.DataFrame, metric="sMAPE"):
    return _lag_similarity(df, metric=metric, lags=96 * 7)


def check_all_columns_have_no_nan(data_df, reference_columns, target_column):
    all_cols = reference_columns + [target_column]
    for i, (col_hasna, col) in enumerate(zip(data_df[all_cols].isna().any(axis=0), all_cols)):
        assert not col_hasna, f"Column '{col}' has missing values: {data_df[col].isna().sum()}"


def generate_self_similarity_plot(data_df: pd.DataFrame, reference_columns: List[str], target_column: str):
    # check inputs
    check_all_columns_are_in_df(data_df, reference_columns, target_column)
    check_all_columns_have_no_nan(data_df, reference_columns, target_column)

    # do some work
    lag_1_scores = _lag_1d_similarity(data_df[reference_columns + [target_column]])
    lag_7_scores = _lag_1w_similarity(data_df[reference_columns + [target_column]])
    print(lag_1_scores)

    target_color = "#ff0000"
    best_lag_scores = {col: min(lag_1_scores[col], lag_7_scores[col]) for col in lag_1_scores}

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Similarity Scores (Lag 1d)",
            "Similarity Scores (Lag 1w)",
            "Similarity Scores (Overall)"
        ],
        vertical_spacing=0.08
    )

    def add_box_and_target(fig, scores, row):
        data = [scores[col] for col in reference_columns]

        # Boxplot
        fig.add_trace(
            go.Box(
                x=data,
                orientation="h",
                name="Reference",
                marker_color="lightblue",
                boxmean=True,
                showlegend=(row == 1)
            ),
            row=row,
            col=1
        )

        # Vertikale Target-Linie
        fig.add_shape(
            type="line",
            x0=scores[target_column],
            x1=scores[target_column],
            y0=0,
            y1=1,
            xref=f"x{row}" if row > 1 else "x",
            yref="paper",
            line=dict(color=target_color, width=2, dash="dash"),
        )

    add_box_and_target(fig, lag_1_scores, row=1)
    add_box_and_target(fig, lag_7_scores, row=2)
    add_box_and_target(fig, best_lag_scores, row=3)

    # Dashboard-farbliches Alignment (wie in app.py)
    BG = "#0b0f14"
    GRID = "rgba(255,255,255,0.10)"
    AXIS = "rgba(255,255,255,0.25)"
    TEXT = "#ffffff"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT),
        height=520,
        boxmode="group",
        title=dict(text=" ", x=0.02, xanchor="left", font=dict(color=TEXT)),
        margin=dict(l=60, r=20, t=60, b=55),
        showlegend=False,
        ),

    # Achsen + Grid passend dunkel
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT), row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT), row=r, col=1)

    fig.update_xaxes(title_text="Density", row=3, col=1)

    return fig


if __name__ == '__main__':
    import pandas as pd
    from constants import DEFAULT_SAMPLE_FILE

    df = pd.read_csv(DEFAULT_SAMPLE_FILE)

    # drop rows with missing values
    df = df.iloc[:35136]
    print(df.head())
    print(df.tail())

    target_column = "load_665"
    reference_columns = [f"load_{id}" for id in [116, 230, 640]]  # ["load_116", "load_230", "load_640"]

    plt_instance = generate_self_similarity_plot(df, reference_columns, target_column)
    plt.show()

# Responsible:  LSB
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from analysis.self_similarity.metrics import evaluation_metrics
from shared_util import check_all_columns_are_in_df

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _lag_similarity(df: pd.DataFrame, metric="sMAPE", lags=96) -> dict:
    shifted_df = df.shift(lags)[lags:]
    original_df = df[lags:]
    assert metric in evaluation_metrics
    metric_func = evaluation_metrics[metric]
    scores = {
        col: metric_func(original_df[col], shifted_df[col]) for col in original_df.columns
    }
    return scores


def _lag_1d_similarity(df: pd.DataFrame, metric="sMAPE") -> dict:
    return _lag_similarity(df, metric=metric, lags=96)


def _lag_1w_similarity(df: pd.DataFrame, metric="sMAPE") -> dict:
    return _lag_similarity(df, metric=metric, lags=96 * 7)


def check_all_columns_have_no_nan(data_df, reference_columns, target_column):
    all_cols = reference_columns + [target_column]
    for i, (col_hasna, col) in enumerate(zip(data_df[all_cols].isna().any(axis=0), all_cols)):
        assert not col_hasna, f"Column '{col}' has missing values: {data_df[col].isna().sum()}"


def _get_quantile_rank(lag_scores: dict, target_column_str: str)-> float:
    distribution = np.sort(np.array(list(lag_scores.values())))
    target_value = lag_scores[target_column_str]

    position = np.searchsorted(distribution, target_value, side='left')
    relative_position = position / len(lag_scores)
    return relative_position


def generate_self_similarity_plot(data_df: pd.DataFrame, reference_columns: List[str], target_column: str):
    # check inputs
    check_all_columns_are_in_df(data_df, reference_columns, target_column)
    check_all_columns_have_no_nan(data_df, reference_columns, target_column)

    # do some work
    lag_1_scores = _lag_1d_similarity(data_df[reference_columns + [target_column]])
    lag_7_scores = _lag_1w_similarity(data_df[reference_columns + [target_column]])
    best_lag_scores = {col: min(lag_1_scores[col], lag_7_scores[col]) for col in lag_1_scores}

    target_color = "#ff0000"

    # -----------------------------
    # Setup Figure mit 2 Subplots
    # -----------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[
            "Einzelne Scores",
            "Gesamt (Verteilung)"
        ],
        vertical_spacing=0.2
    )

    # -----------------------------
    # Obere Boxplots: Täglich & Wöchentlich
    # -----------------------------
    def add_box(fig, scores, label, y_pos):
        data = [scores[col] for col in reference_columns]
        fig.add_trace(
            go.Box(
                x=data,
                y=[y_pos] * len(data),
                orientation="h",
                name=label,
                marker_color="lightblue",
                boxmean=True,
                width=0.3,
                showlegend=False
            ),
            row=1,
            col=1
        )
        fig.add_shape(
            type="line",
            x0=scores[target_column],
            x1=scores[target_column],
            y0=-0.35 + y_pos,
            y1=0.35 + y_pos,
            xref="x1",
            yref="y1",
            showlegend=False,
            line=dict(color=target_color, width=2, dash="dash"),
        )
    add_box(fig, lag_1_scores, "Täglich", y_pos=0)
    add_box(fig, lag_7_scores, "Wöchentlich", y_pos=1)
    fig.update_yaxes(tickvals=["0", "1"], ticktext=["Täglich", "Wöchentlich"], row=1, col=1)

    # -----------------------------
    # Unteres Histogramm / Verteilung für Gesamt
    # -----------------------------
    hist_data = [best_lag_scores[col] for col in reference_columns]
    fig.add_trace(
        go.Box(
                x=hist_data,
                y=[0] * len(hist_data),
                orientation="h",
                width=0.3,
                marker_color="lightgreen",
                boxmean=True,
                showlegend=False
        ),
        row=2, col=1
    )
    fig.add_shape(
        type="line",
        x0=best_lag_scores[target_column],
        x1=best_lag_scores[target_column],
        name="Ihr Lastprofil",
        showlegend=True,
        y0=-0.4,
        y1=0.4,
        xref="x2",
        yref="y2",
        line=dict(color=target_color, width=2, dash="dash"),
    )

    # -----------------------------
    # Styling (dunkles Dashboard)
    # -----------------------------
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
        showlegend=True,
    )

    fig.update_yaxes(tickvals=["0"], ticktext=["Gesamt"], row=2, col=1)

    # Achsen + Grid
    for r in (1, 2):
        fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT), row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=TEXT), row=r, col=1)

    fig.update_xaxes(title_text="Selbstähnlichkeit des Signals (je kleiner desto höher die Selbstähnlichkeit)", row=2,
                     col=1)

    # -----------------------------
    # Calculte Explanation text
    # -----------------------------
    weekly_quantile_rank = _get_quantile_rank(lag_7_scores, target_column)
    daily_quantile_rank = _get_quantile_rank(lag_1_scores, target_column)
    overall_quantile_rank = _get_quantile_rank(best_lag_scores, target_column)

    if weekly_quantile_rank <= 0.1:
        weekly_summary = "sehr hoch"
    elif weekly_quantile_rank <= 0.2:
        weekly_summary = "hoch"
    elif weekly_quantile_rank > 0.2 and weekly_quantile_rank < 0.8:
        weekly_summary = "im durchschnittlichen Bereich"
    elif weekly_quantile_rank >= 0.80 and weekly_quantile_rank < 0.9:
        weekly_summary = "niedrig"
    elif weekly_quantile_rank >= 0.9:
        weekly_summary = "sehr niedrig"
    else:
        weekly_summary = "unbekannt"

    if daily_quantile_rank <= 0.1:
        daily_summary = "sehr hoch"
    elif daily_quantile_rank <= 0.2:
        daily_summary = "hoch"
    elif daily_quantile_rank > 0.2 and daily_quantile_rank < 0.8:
        daily_summary = "im durchschnittlichen Bereich"
    elif daily_quantile_rank >= 0.80 and daily_quantile_rank < 0.9:
        daily_summary = "niedrig"
    elif daily_quantile_rank >= 0.9:
        daily_summary = "sehr niedrig"
    else:
        daily_summary = "unbekannt"

    if overall_quantile_rank <= 0.1:
        total_summary = "sehr hoch (in den Top-10% aller Unternehmen). Eine Prognose nur basierend auf historischen Werten kann ausreichend sein."
    elif overall_quantile_rank <= 0.2:
        total_summary = "hoch (in den Top-20% aller Unternehmen). Eine Prognose nur basierend auf historischen Werten kann ausreichend sein."
    elif overall_quantile_rank > 0.2 and overall_quantile_rank < 0.8:
        total_summary = "im durchschnittlichen Bereich. Eine Prognose nur basierend auf historischen Werten ist ungenau nicht empfohlen, weitere Varaiblen sind hierfür benötigt."
    elif overall_quantile_rank >= 0.80 and overall_quantile_rank < 0.9:
        total_summary = "niedrig (in den niedrigsten 20% aller Unternehmen). Eine Prognose nur basierend auf historischen Werten ist ungenau und wird nicht empfohlen, weitere Varaiblen sind hierfür benötigt."
    elif overall_quantile_rank >= 0.9:
        total_summary = "sehr niedrig (in den niedrigsten 10% aller Unternehmen). Eine Prognose nur basierend auf historischen Werten ist sehr ungenau und wird nicht empfohlen, weitere Varaiblen sind hierfür benötigt."
    else:
        total_summary = "unbekannt"

    interpretation_str = f"Die wöchentliche Selbstähnlichkeit ist {weekly_summary}. Die tägliche Selbstähnlichkeit ist {daily_summary}. Die Gesamt-Selbstähnlichkeit ist {total_summary}."

    # remove later, once interpretation string is added.
    print(interpretation_str)

    return fig, interpretation_str

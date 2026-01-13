# Responsible:  LSB

import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from analysis.self_similarity.metrics import evaluation_metrics
from shared_util import check_all_columns_are_in_df


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

    target_color = "r"
    best_lag_scores = {col: min(lag_1_scores[col], lag_7_scores[col]) for col in lag_1_scores}

    y_min = 0.5
    y_max = 1.5

    # create and return result plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

    data = [lag_1_scores[key] for key in reference_columns]
    axes[0].boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0].set_title(f"Similarity Scores (Lag 1d)")
    axes[0].set_ylabel(f"")
    axes[0].set_xlabel(f"Density")
    axes[0].vlines(x=lag_1_scores[target_column], ymin=y_min, ymax=y_max, colors=target_color, linestyle="--", linewidth=2,
                   label='Target')

    data = [lag_7_scores[key] for key in reference_columns]
    axes[1].boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[1].set_title(f"Similarity Scores (Lag 1w)")
    axes[1].set_ylabel(f"")
    axes[1].set_xlabel(f"Density")
    axes[1].vlines(x=lag_7_scores[target_column], ymin=y_min, ymax=y_max, colors=target_color, linestyle="--", linewidth=2,
                   label='Target')

    data = [best_lag_scores[key] for key in reference_columns]
    axes[2].boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[2].set_title(f"Similarity Scores (Overall)")
    axes[2].set_ylabel(f"")
    axes[2].set_xlabel(f"Density")
    axes[2].vlines(x=best_lag_scores[target_column], ymin=y_min, ymax=y_max, colors=target_color, linestyle="--", linewidth=2,
                   label='Target')


    plt.legend()
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

# Responsible:  LSB

import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from analysis.self_similarity.metrics import evaluation_metrics
from shared_util import check_all_columns_are_in_df


def _lag_1d_similarity(df: pd.DataFrame, metric="sMAPE"):
    shifted_df = df.shift(96)[96:]
    original_df = df[96:]
    assert metric in evaluation_metrics
    metric_func = evaluation_metrics[metric]
    scores = {
        col: metric_func(original_df[col], shifted_df[col]) for col in original_df.columns
    }
    return scores


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
    print(lag_1_scores)

    # create and return result plot
    fig = plt.figure(figsize=(20, 10))
    plt.hist( [lag_1_scores[key] for key in reference_columns], label='Scores (the smaller the better)')
    plt.vlines(x=lag_1_scores[target_column], ymin=0, ymax=1, colors='r', label='Target-Column')
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

import pandas as pd
import numpy as np
from typing import List
from matplotlib import pyplot as plt

"""
benchmark_columns = ["116", "230", "665"]
target_column = "640"
"""

def generate_distribution_comparison(data_df:  pd.DataFrame, benchmark_columns: List[str], target_column: str):

    ref_cols = [f"load_{x}" for x in benchmark_columns]
    target_col = f"load_{target_column}"

    weather_vars = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco', 'dow']

    all_cols = ref_cols + [target_col]
    all_col_names = benchmark_columns + [target_column]

    all_corr = []
    for i in range(len(all_cols)):
        corr = data_df[[all_cols[i]] + [f"{v}_{all_col_names[i]}" for v in weather_vars]].corr(method="spearman")
        corr = corr[all_cols[i]].drop(all_cols[i])
        corr.index = weather_vars
        all_corr.append(corr)

    all_corr = pd.concat(all_corr, axis = 1)

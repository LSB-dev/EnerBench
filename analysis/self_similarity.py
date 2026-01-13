# TODO LSB
from turtle import pd
from typing import List
import matplotlib.pyplot as plt

def generate_self_similary_plot(data_df:  pd.DataFrame, benchmark_columns: List[str], target_column: str):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([1,2,3], [4,5,6], label='nice one!')

    return fig


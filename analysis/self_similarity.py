# TODO LSB
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

def generate_self_similary_plot(data_df: pd.DataFrame, reference_columns: List[str], target_column: str):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([1,2,3,4], [4,5,8,5], label='nice one!')

    return fig



if __name__ == '__main__':
    # test functionality
    import pandas as pd
    import numpy as np
    from constants import DEFAULT_SAMPLE_FILE

    df = pd.read_csv(DEFAULT_SAMPLE_FILE)
    print(df.head())

    target_column = "load_665"
    reference_columns = [r"load_{id}" for id in [116, 230, 640, ]] # ["load_116", "load_230", "load_640"]

    plt_instance = generate_self_similary_plot(df, reference_columns, 'load_665')
    plt.show()
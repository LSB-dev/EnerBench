import pandas as pd
from typing import List


def check_all_columns_are_in_df(data_df: pd.DataFrame, reference_columns: List[str], target_column: str):
    # check for reference columns
    for col in reference_columns:
        assert col in data_df.columns, f"Reference-Column '{col}' not found in dataframe. Available columns: {data_df.columns}"

    # check for target column
    assert target_column in data_df.columns, f"Target-Column '{target_column}' not found in dataframe. Available columns: {data_df.columns}"
    assert target_column not in reference_columns, f"Target-Column '{target_column}' is also a Reference-Column"

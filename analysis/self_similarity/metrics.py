"""
    version 1.0.3
    2025-01-28
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, \
    root_mean_squared_error
import numpy as np


def MAPE_plain(y, y_pred):
    "in 0.xxx"
    return mean_absolute_percentage_error(y, y_pred)


def MAPE(y, y_pred):
    "in %"
    return 100 * MAPE_plain(y, y_pred)


def relMAE(y, y_pred):
    """
    relative Mean Absolute Error
    @param y: the original, reference value
    @param y_pred: the estimation
    @return: af funciton calculating relMAE
    """
    return 100 * mean_absolute_error(y, y_pred) / np.mean(y)


def RMSE(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    return np.sqrt(mse)


def sMAPE(y, y_pred):
    """
    (the absolute sMAPE version)
    :param y: the original, reference value
    :param y_pred: the estimation
    :return: a value > 0, example: 13.1257878, indicating roughly 13% error.
    """
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    assert isinstance(y, np.ndarray), f"Wrong type: '{type(y)}', should ne 'np.ndarray'."
    assert isinstance(y_pred, np.ndarray), f"Wrong type: '{type(y_pred)}', should ne 'np.ndarray'."

    nenner = (np.abs(y) + np.abs(y_pred)) / 2
    mask = nenner != 0
    zaehler = np.abs(y - y_pred)
    smape = np.mean(zaehler[mask] / nenner[mask])
    return 100 * smape


# call with argument (y, y')'
evaluation_metrics = {
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error,
    "MAE": mean_absolute_error,
    "MAPE": MAPE,  # in %, i.e., typically 0... 100
    "sMAPE": sMAPE,  # in %, i.e., typically 0... 100
    "relMAE": relMAE  # in %, i.e.,  typically 0... 100
}

# Code Description:
# This code contains several functions to calculate different error metrics
# used to evaluate the performance of forecasting models. The metrics calculated
# include:
# - RRMSE (Relative Root Mean Squared Error)
# - PBE (Percentage Bias Error)
# - POCID (Percentage of Correctly Identified Direction)
# - MASE (Mean Absolute Scaled Error)
# These functions take true and predicted values as inputs and return the respective error metric.

import numpy as np
from sklearn.metrics import mean_squared_error as mse

def rrmse(y_true, y_pred, mean_y_true_serie_completa):
    """
    Calculate Relative Root Mean Squared Error (RRMSE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - mean_y_true_serie_completa: float, mean value of the true series

    Returns:
    - float, RRMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rrmse_value = rmse / mean_y_true_serie_completa
 
    return rrmse_value

def pbe(y_true, y_pred):
    """
    Calculate Percentage Bias Error (PBE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - float, PBE value
    """
    return 100 * (np.sum(y_true - y_pred) / np.sum(y_true))

def pocid(y_true, y_pred):
    """
    Calculate Percentage of Correctly Predicted Direction (POCID).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - float, POCID value
    """
    n = len(y_true)
    D = [1 if (y_pred[i] - y_pred[i-1]) * (y_true[i] - y_true[i-1]) > 0 else 0 for i in range(1, n)]
    POCID = 100 * np.sum(D) / (n-1)

    return POCID

def mase(y_true, y_pred, y_baseline):
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - y_baseline: array-like, baseline (naive) values

    Returns:
    - float, MASE value
    """
    mae_pred = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_baseline))
    result = mae_pred / mae_naive
    return result
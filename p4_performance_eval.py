from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from p3_arima_model_training import arima_forecast

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, mape


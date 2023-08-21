from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from p3_arima_model_training import arima_forecast

"""# Sample Data (replace this with your data)
# df = pd.read_csv('path_to_your_data.csv', index_col='date', parse_dates=True)

# Splitting data
#train_size = int(0.8 * len(df))
#train, test = df[:train_size], df[train_size:]

# ====== ARIMA (for univariate series) ======
# Assuming 'column1' is the time series you want to forecast with ARIMA
model_arima = ARIMA(train['column1'], order=(5,1,0))
model_fit_arima = model_arima.fit(disp=0)
forecast_arima = model_fit_arima.forecast(steps=len(test))[0]
"""



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, mape



# TODO ask Jörg about implementation. Necessary?
"""
# Average 10 last day test vs 10 last days forecast
average_last3_days_test = np.mean(test_data['close'][-10:])
average_last3_days_forecast = np.mean(forecast[:, 0][-10:])
average_last3_days_diff = round \
    (abs(average_last3_days_forecast - average_last3_days_test) / average_last3_days_test * 100, 2)
    """

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from p2_data_validation import merge_data_one_ticker
from p3_arima_data_preparation import arima_df_filled_nas, arima_df_normalized, arima_split_data

def arima_train(df_train_scaled, df_test_scaled, ticker, arima_order=(2, 0, 1)):
    
    # df_train_scaled, df_test_scaled = df_normalized(ticker)

    #manual fit
    model = ARIMA(df_train_scaled['close'], order = arima_order)
    model.fit()

    return model


def arima_forecast(df_train_scaled, df_test_scaled, ticker, arima_order=(2, 0, 1)):
    # df_train_scaled, df_test_scaled = df_normalized(ticker)

    # TODO why do we concatenate them?
    # Extend the training data to include the test data
    extended_train = pd.concat([df_train_scaled['close'], df_test_scaled['close']])

    # Refit the ARIMA model using the extended training data
    model_fit_extended = ARIMA(extended_train, order = arima_order).fit()

    # Forecast for the next month (30 days)
    forecast_steps = 30
    forecast = model_fit_extended.forecast(steps=forecast_steps)

    return forecast


if __name__ == '__main__':
    ticker = "BTC-USD"
    arima_order=(2, 0, 1)

    df_arima = merge_data_one_ticker(ticker)
    df_arima = arima_df_filled_nas(df_arima)
    train, test = arima_split_data(df_arima)
    df_train_scaled, df_test_scaled = arima_df_normalized(train, test)
    arima_model = arima_train(df_train_scaled, df_test_scaled, ticker, arima_order)
    arima_forecast = arima_forecast(df_train_scaled, df_test_scaled, ticker, arima_order)
    print(arima_forecast)

# TODO reverse the transformation so that we can forecast future prices

# TODO I get this error message ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
# TODO create performance evaluation for ARIMA
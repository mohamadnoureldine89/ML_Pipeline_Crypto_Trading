import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from p2_data_validation import merge_data_one_ticker
from p3_arima_data_preparation import arima_df_filled_nas, arima_df_normalized, arima_split_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta


def arima_train(df_train_scaled, forecast_steps=30, arima_order=(2, 0, 1)):

    model = ARIMA(df_train_scaled['close'], order=arima_order).fit()
    forecast_arima = model.forecast(forecast_steps)

    return forecast_arima


def arima_forecast(df_arima, forecast_steps=30, arima_order=(2, 0, 1)):

    # Extend the training data to include the test data
    extended_train = arima_df_filled_nas(df_arima)

    # Normalize
    scaler = MinMaxScaler()

    # Train goes with fit_transform
    extended_train_scaled = scaler.fit_transform(extended_train[['close']])

    # Convert NP array to PD DF
    extended_train_scaled_df = pd.DataFrame(extended_train_scaled, columns=['close'])

    # Refit the ARIMA model using the extended training data
    model_fit_extended = ARIMA(extended_train_scaled_df['close'], order=arima_order).fit()

    # Forecast for the n days (30 by default)
    forecast = model_fit_extended.forecast(steps=forecast_steps)

    # Reshape for inverse transformation
    forecast_2d = np.array(forecast).reshape(-1, 1)

    # Reverse the transformation
    forecast_original_scale = scaler.inverse_transform(forecast_2d)

    forecast_original = forecast_original_scale.flatten()

    # Array of dates starting from tomorrow
    dates = pd.date_range(start=datetime.today().date() + timedelta(days=1), periods=30)

    # Convert to DataFrame
    df = pd.DataFrame({'Date': dates, 'Value': forecast_original})

    return df

if __name__ == '__main__':
    ticker = "BTC-USD"
    arima_order_test = (2, 0, 1)

    df_arima = merge_data_one_ticker(ticker)
    df_arima = arima_df_filled_nas(df_arima)
    train, test = arima_split_data(df_arima)
    df_train_scaled, df_test_scaled = arima_df_normalized(train, test)
    arima_model = arima_train(df_train_scaled, arima_order=arima_order_test)
    arima_forecast_prices = arima_forecast(df_arima, forecast_steps=30, arima_order=arima_order_test)
    print(arima_forecast_prices)



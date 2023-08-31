from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import os
from datetime import datetime, timedelta
from p1_data_ingestion import ingest_to_aws, ticker_table, ticker_selection, load_env_variables_from_file, upload_df
from p2_data_validation import merge_data_one_ticker, validate_data, column_names
from p3_arima_data_preparation import arima_df_filled_nas, arima_df_normalized, arima_split_data
from p3_arima_model_training import arima_train, arima_forecast
from p3_var_data_preparation import var_df_filled_nas, perform_adf_test
from p3_var_model_training import fit_VAR, forecast_VAR
from p4_performance_eval import evaluate_model

def forecast_crypto_prices(ticker, START_DATE="2020-01-01"):
    
    ###################################
    # 3.a ARIMA data preparation
    ###################################
    df_arima = merge_data_one_ticker(ticker)
    df_arima = arima_df_filled_nas(df_arima)

    ###################################
    # 3.a ARIMA model training
    ###################################
    arima_order_test = (2, 0, 1)
    days_forecast = 30
    arima_forecast_prices_beyond_test = arima_forecast(df_arima, forecast_steps=days_forecast, arima_order=arima_order_test)

    ###################################
    # 3.b VAR data preparation
    ###################################
    df_var = merge_data_one_ticker(ticker)
    df_var = var_df_filled_nas(df_var)

    ###################################
    # 3.b VAR model training
    ###################################

    cols = len(column_names) + 1  # for now we consider all columns
    order = 11  # best order value calculated previously
    start_date = datetime.strptime("2022-01-31", "%Y-%m-%d")  # calculated as best start date

    forecast_var_beyond_test_set = forecast_VAR(df_var, cols, order, start_date, potential_columns=column_names)

    return arima_forecast_prices_beyond_test, forecast_var_beyond_test_set


if __name__ == "__main__":
    START_DATE = "2020-01-01"
    ###################################
    # 1. Data ingestion - upload to AWS
    ###################################
    ingest_to_aws(START_DATE)

    # Array of dates starting from tomorrow
    dates = pd.date_range(start=datetime.today().date() + timedelta(days=1), periods=30)

    # create empty dataframes for arima and var forecast with dates of next 30 days
    arima_forecast_df = pd.DataFrame({'Date': dates})
    var_forecast_df = pd.DataFrame({'Date': dates})

    for ticker, name in zip(ticker_selection['ticker'], ticker_selection['name']):
        print(f"Ticker {ticker} being processed")
        
        arima_forecast_prices_beyond_test, var_forecast_beyond_test_set = forecast_crypto_prices(ticker, START_DATE)
        
        # ARIMA Model
        arima_forecast_prices_beyond_test = arima_forecast_prices_beyond_test["Value"]
        arima_forecast_prices_beyond_test = pd.DataFrame({f"{name}": arima_forecast_prices_beyond_test})
        arima_forecast_df = pd.concat([arima_forecast_df, arima_forecast_prices_beyond_test], axis=1)

        # VAR Model
        var_forecast_beyond_test_set = var_forecast_beyond_test_set["Value"]
        var_forecast_beyond_test_set = pd.DataFrame({f"{name}": var_forecast_beyond_test_set})
        var_forecast_df = pd.concat([var_forecast_df, var_forecast_beyond_test_set], axis=1)


        print(f"Ticker {ticker} complete")
    upload_df(arima_forecast_df, "arima_forecast")
    upload_df(var_forecast_df, "var_forecast")





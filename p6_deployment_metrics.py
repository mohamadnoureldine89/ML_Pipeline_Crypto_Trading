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
import numpy as np

def evaluate_model_performance(ticker):
    
    ###################################
    # 3.a ARIMA data preparation
    ###################################
    df_arima = merge_data_one_ticker(ticker)
    df_arima = arima_df_filled_nas(df_arima)
    train, test = arima_split_data(df_arima)
    df_train_scaled, df_test_scaled = arima_df_normalized(train, test)

    ###################################
    # 3.a ARIMA model training
    ###################################
    arima_order_test = (2, 0, 1)
    days_forecast = 30
    forecast_arima_test_set = arima_train(df_train_scaled, forecast_steps=len(test['close'].values),
                                          arima_order=arima_order_test)

    ###################################
    # 4.a ARIMA performance evaluation
    ###################################
    arima_metrics = evaluate_model(test['close'].values, forecast_arima_test_set)

    ###################################
    # 3.b VAR data preparation
    ###################################
    df_var = merge_data_one_ticker(ticker)
    df_var = var_df_filled_nas(df_var)
    perform_adf_test(df_var)

    ###################################
    # 3.b VAR model training
    ###################################
    # TODO update configurations here?

    # configure test and train periods
    test_duration = 60  # change for longer/ shorter test periods (days)
    test_start_date = datetime.strptime('2023-06-10', "%Y-%m-%d")  # start of test period
    test_end_date = test_start_date + timedelta(days=test_duration)  # end of test period
    train_end_date = test_start_date - timedelta(days=1)

    cols = len(column_names) + 1  # TODO for now we consider all columns
    order = 11  # best order value calculated previously
    start_date = datetime.strptime("2022-01-31", "%Y-%m-%d")  # calculated as best start date

    # for now I consider all columns
    dict_fit_VAR_output = fit_VAR(df_var, train_end_date, test_start_date, test_end_date, cols, order, start_date,
                                  column_names)

    forecast_var_test_set = dict_fit_VAR_output["forecast_test_set"]

    ###################################
    # 4.b VAR performance evaluation
    ###################################
    var_test_data = dict_fit_VAR_output["test_data"]
    var_metrics = evaluate_model(var_test_data['close'].values, forecast_var_test_set)

    return arima_metrics, var_metrics


if __name__ == "__main__":

    metrics = ["mae_arima", "mse_arima", "rmse_arima", "mape_arima", "mae_var", "mse_var", "rmse_var", "mape_var"]
    # create an empty dataframe for arima and var performance metrics
    metrics_df = pd.DataFrame({'Metric_name': metrics})

    for ticker, name in zip(ticker_selection['ticker'], ticker_selection['name']):
        print(f"Ticker {ticker} being processed")
        
        arima_metrics, var_metrics = evaluate_model_performance(ticker)

        # ARIMA Model
        mae_arima, mse_arima, rmse_arima, mape_arima = arima_metrics

        # VAR Model
        mae_var, mse_var, rmse_var, mape_var = var_metrics

        metrics_column = np.array([mae_arima, mse_arima, rmse_arima, mape_arima, mae_var, mse_var, rmse_var, mape_var])
        metrics_column = pd.DataFrame({f"{name}": metrics_column})


        metrics_df = pd.concat([metrics_df, metrics_column], axis=1)


        print(f"Ticker {ticker} complete")
        
    upload_df(metrics_df, "metrics")









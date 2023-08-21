from sqlalchemy import create_engine
import pandas as pd
import os
from datetime import datetime, timedelta
from p1_data_ingestion import ingest_to_aws
from p2_data_validation import merge_data_one_ticker, validate_data, column_names
from p3_arima_data_preparation import arima_df_filled_nas, arima_df_normalized, arima_split_data
from p3_arima_model_training import arima_train, arima_forecast
from p3_var_data_preparation import var_df_filled_nas, perform_adf_test
from p3_var_model_training import fit_VAR

# from p3_var_model_training import
from p4_performance_eval import evaluate_model

def load_env_variables_from_file(file_path):
    """
    Load environment variables from a file env.txt and export them as env variables

    """
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

def fetch_tickers():

    load_env_variables_from_file('env.txt')
    # Load credentials from environment variables
    db_endpoint = os.environ.get("DB_ENDPOINT")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")

    # Create a database URL
    db_url = f"postgresql://{db_user}:{db_password}@{db_endpoint}:{db_port}/{db_name}"

    table_name = "crypto_data"

    # Using a context manager to handle the engine
    with create_engine(db_url).connect() as connection:
        query = f"SELECT DISTINCT ticker FROM {table_name}"
        df_data = pd.read_sql(query, connection)
        df_data = df_data["ticker"].tolist()

    return df_data

def forecast_crypto_prices(ticker, START_DATE="2020-01-01", nb_currencies=50):
    ###################################
    # 1. Data ingestion - upload to AWS
    ###################################
    ingest_to_aws(nb_currencies, START_DATE)

    ###################################
    # 2. Data validation
    ###################################
    # Pull crypto, market and sentiment data from DB and merge it into a single DF
    df_crypto_sa_id = merge_data_one_ticker(ticker)

    # Validate the data (missing values, correlation heatmap)
    validate_data(df_crypto_sa_id)

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
    arima_forecast_prices_beyond_test = arima_forecast(df_arima, forecast_steps=days_forecast,
                                                       arima_order=arima_order_test)
    print(forecast_arima_test_set)

    ###################################
    # 4.a ARIMA performance evaluation
    ###################################
    mae_arima, mse_arima, rmse_arima, mape_arima = evaluate_model(test['close'].values, forecast_arima_test_set)
    print(f"ARIMA - MAE: {mae_arima}, MSE: {mse_arima}, RMSE: {rmse_arima}, MAPE: {mape_arima}%")

    ###################################
    # 3.b VAR data preparation
    ###################################
    df_var = merge_data_one_ticker(ticker)
    df_var = var_df_filled_nas(df_var)
    perform_adf_test(df_var)

    ###################################
    # 3.b VAR model training
    ###################################

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

    forecast_var_test_set = dict_fit_VAR_output["forecast"]
    print(forecast_var_test_set)

    ###################################
    # 4.b VAR performance evaluation
    ###################################
    var_test_data = dict_fit_VAR_output["test_data"]

    mae_var, mse_var, rmse_var, mape_var = evaluate_model(var_test_data['close'].values, forecast_var_test_set['close'])
    print(f"VAR - MAE: {mae_var}, MSE: {mse_var}, RMSE: {rmse_var}, MAPE: {mape_var}%")


if __name__ == "__main__":
    START_DATE = "2020-01-01"
    nb_currencies = 5
    ticker_list = fetch_tickers()
    for ticker in ticker_list:
        print(f"Ticker {ticker} being processed")
        forecast_crypto_prices(ticker, START_DATE, nb_currencies)
        print(f"Ticker {ticker} complete")





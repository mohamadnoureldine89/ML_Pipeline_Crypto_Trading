from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from p3_var_data_preparation import var_df_filled_nas
from p2_data_validation import merge_data_one_ticker, column_names

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def fit_VAR(df, train_end_date, test_start_date, test_end_date, cols, order, start_date, potential_columns=column_names):
    
    df.index = pd.to_datetime(df.index)
    start_date = pd.Timestamp(start_date)
    train_end_date = pd.Timestamp(train_end_date)

    # Split data into train and validation sets
    train_data = df.loc[start_date:train_end_date]
    test_data = df.loc[test_start_date:test_end_date]

    # TODO before fitting, we need to normalize the data as many columns are not stationery
    
    # Fit VAR model
    model = VAR(train_data[potential_columns[:cols]])
    model_fit = model.fit(order)

    # Forecast using the trained model
    forecast = model_fit.forecast(model_fit.endog, steps=len(test_data)) 
    
    # Replace values less than 0 with 0 (prices can't be negative)
    forecast[forecast < 0] = 0

    # TODO Calculate RMSE -> MAPE seems more suitable ?
    # rmse = np.sqrt(mean_squared_error(test_data['close'], forecast[:, -1]))
                
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_data['close'], forecast[:, -1])

    # Average 10 last day test vs 10 last days forecast   
    average_last3_days_test = np.mean(test_data['close'][-10:])
    average_last3_days_forecast = np.mean(forecast[:, 0][-10:])
    average_last3_days_diff = round(abs(average_last3_days_forecast - average_last3_days_test) / average_last3_days_test * 100, 2)

    return {
        'model_fit': model_fit,
        'forecast':forecast,
        'mape': mape,
        'avg_last_10_days_diff': average_last3_days_diff
    }


if __name__ == "__main__":

    # configure test and train periods
    test_duration = 60  # change for longer/ shorter test periods (days)
    test_start_date = datetime.strptime('2023-06-10', "%Y-%m-%d") # start of test period
    test_end_date = test_start_date + timedelta(days=test_duration) # end of test period
    train_end_date = test_start_date - timedelta(days=1)

    ticker = "BTC-USD"
    df = merge_data_one_ticker(ticker)
    df = var_df_filled_nas(df)

    cols = len(column_names) + 1 # TODO for now we consider all columns
    order = 11 # best order value calculated previously
    start_date = datetime.strptime("2022-01-31", "%Y-%m-%d") # calculated as best start date

    # for now I consider all columns
    dict_fit_VAR_output = fit_VAR(df, train_end_date, test_start_date, test_end_date, cols, order, start_date, column_names)

    print(dict_fit_VAR_output)

# TODO here I get ValueWarning: No frequency information was provided, so inferred frequency D will be used.

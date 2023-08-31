from statsmodels.tsa.stattools import adfuller
from p2_data_validation import merge_data_one_ticker, column_names

# ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1] <= 0.05  # p-value

def var_df_filled_nas(df):

    # fill NAs with next value
    df.ffill(inplace=True)

    # fill the rest with previous value
    df.bfill(inplace=True)

    missing_values = df.isnull().sum()
    # print(missing_values)

    return df

def perform_adf_test(df):

    # Test for Stationarity with Augmented Dickey–Fuller (ADF) Test:
    for column in df.columns:
        is_stationary = adf_test(df[column])
        print(f"{column} is {'stationary' if is_stationary else 'non-stationary'}")

# FYI: We don't split the data and normalize here but later during model training

if __name__ == "__main__":
    ticker = 'BTC-USD'
    df = merge_data_one_ticker(ticker)
    df = var_df_filled_nas(df)
    perform_adf_test(df)

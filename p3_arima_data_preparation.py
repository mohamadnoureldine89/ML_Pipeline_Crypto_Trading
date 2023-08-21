from p2_data_validation import merge_data_one_ticker
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from p2_data_validation import column_names

# Fill NAs with next value or previous value
def arima_df_filled_nas(df):

    # fill NAs with next value
    df.ffill(inplace=True)

    # fill the rest with previous value
    df.bfill(inplace=True)

    missing_values = df.isnull().sum()
    print(missing_values)

    return df


def arima_split_data(df):
    
    # Split the data
    train_size = int(len(df) * 0.8)

    # Train and test samples
    train = df[:train_size]
    test = df[train_size:]

    return train, test


# Perform data normalization AFTER splitting to avoid data leakage
def arima_df_normalized(train, test):

    scaler = MinMaxScaler()

    # train goes with fit_transform
    train_scaled = scaler.fit_transform(train)

    # convert NP array to PD DF
    df_train_scaled = pd.DataFrame(train_scaled, columns=column_names)

    # test goes with transform
    test_scaled = scaler.transform(test)

    # convert NP array to PD DF
    df_test_scaled = pd.DataFrame(test_scaled, columns=column_names)

    return df_train_scaled, df_test_scaled


if __name__ == "__main__":
    ticker = 'BTC-USD'
    df = merge_data_one_ticker(ticker)
    df = arima_df_filled_nas(df)
    train, test = arima_split_data(df)
    df_train_scaled, df_test_scaled = arima_df_normalized(train, test)


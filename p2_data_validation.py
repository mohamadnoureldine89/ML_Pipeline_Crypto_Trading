import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# TODO maybe we want to log the validations in a log file? 

column_names = ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'sa_score1',
                    'sa_score2', 'sa_score3', 'nasdaq_close', 'nasdaq_volume', 'ftse_close',
                    'ftse_volume', 'crude_close', 'crude_volume', 'stoxx50e_close',
                    'stoxx50e_volume', 'gold_close', 'gold_volume', 'nikkei_close',
                    'nikkei_volume', 'hsi_close', 'hsi_volume']

def load_env_variables_from_file(file_path):
    """
    Load environment variables from a file env.txt and export them as env variables
    
    """
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

def fetch_db():

    load_env_variables_from_file('env.txt')
    # Load credentials from environment variables 
    # Make sure to load the write env variables before running the coe
    db_endpoint = os.environ.get("DB_ENDPOINT")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")

    # Create a database URL
    db_url = f"postgresql://{db_user}:{db_password}@{db_endpoint}:{db_port}/{db_name}"

    table_names = ["sentiment_data", "crypto_data", "index_data"]
    df_data = {}

    # Using a context manager to handle the engine
    with create_engine(db_url).connect() as connection:
        for table_name in table_names:
            query = f"SELECT * FROM {table_name}"
            df_data[table_name] = pd.read_sql(query, connection)
    
    return df_data

def merge_data_one_ticker(ticker):
    # fetch crypto data for specific ticker
    df_data = fetch_db()
    df_crypto = df_data["crypto_data"]
    df_crypto = df_crypto[df_crypto["ticker"] == ticker]

    # fetch sentiment analysis and merge
    df_senti_crypt = df_data["sentiment_data"]
    df_crypto_sa = df_crypto.merge(df_senti_crypt, on='date', how='left')

    # Fetch index data and merge
    df_index_data = df_data["index_data"]
    df_crypto_sa_id = df_crypto_sa.merge(df_index_data, on='date', how='left')

    # Set date as index
    df_crypto_sa_id = df_crypto_sa_id.set_index('date')

    # Drop the ticker column
    df_crypto_sa_id = df_crypto_sa_id.drop('ticker', axis=1)

    # Convert volume into float
    df_crypto_sa_id['volume'] = pd.to_numeric(df_crypto_sa_id['volume'], errors='coerce')

    # Rename columns
    df_crypto_sa_id.columns = column_names

    return df_crypto_sa_id

def validate_data(df):
    
    print(df.describe())
    print(df.info())

    # Missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # Correlation matrix
    correlation_matrix = df.corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.show()

if __name__ == "__main__":
    ticker = 'BTC-USD'
    df_crypto_sa_id = merge_data_one_ticker(ticker)
    validate_data(df_crypto_sa_id)





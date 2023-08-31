import pandas as pd
from yahoo_fin.stock_info import get_data
from yahooquery import Screener
import psycopg2
import io
from senti_crypt import get_senti_crypt
import yfinance as yf
import os
from datetime import datetime, timedelta

def load_env_variables_from_file(file_path):
    """
    Load environment variables from a file env.txt and export them as env variables
    
    """
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

def fetch_ticker_data(symbol, start_date, end_date=None, index_as_date=False, interval="1d"):
    """Retrieve the data of a single ticker."""
    response = get_data(symbol, start_date, end_date, index_as_date, interval)
    df = pd.DataFrame(response)
    df = df[['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']]
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_sentiment_data():
    """Retrieve sentiment analysis data."""
    df_senti_crypt = get_senti_crypt()
    df_senti_crypt = df_senti_crypt[["date", "score1", "score2", "score3"]]
    df_senti_crypt['date'] = pd.to_datetime(df_senti_crypt['date'])
    return df_senti_crypt

def fetch_index_data(index_tickers, start_date, end_date=None):
    """Fetch historical data for each provided index."""
    all_index_data = pd.DataFrame()
    for ticker in index_tickers:
        index_data = yf.download(ticker, start=start_date, end=end_date)
        index_data = index_data[['Close', 'Volume']]
        index_data.columns = [f"{ticker}_Close", f"{ticker}_Volume"]
        all_index_data = pd.concat([all_index_data, index_data], axis=1)
    all_index_data.reset_index(inplace=True)
    all_index_data.rename(columns={'Date': 'date'}, inplace=True)
    return all_index_data

def clean_column_name(column_name):
    return column_name.replace('^', '').replace('-', '_').replace('=', '')

# Database Operations
def to_sql(df, map_dict):
    cols_map = zip(df.columns, df.dtypes.replace(map_dict))
    sql_cols = ", ".join("{} {}".format(n, d) for (n, d) in cols_map)
    return sql_cols

def create_table(cursor, table_name, sql_cols):
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    query = f""" 
            CREATE TABLE {table_name} (
            {sql_cols}
            )
            """
    cursor.execute(query)

def upload_data(cursor, df, table_name):
    schema_name = 'public'
    csv_file = io.StringIO()
    df.to_csv(csv_file, header=df.columns, index=False, encoding='utf-8')
    csv_file.seek(0)
    sql_statement = f"""
        COPY {schema_name}.{table_name} FROM STDIN WITH 
            CSV
            HEADER
            DELIMITER AS ','
        """
    cursor.copy_expert(sql=sql_statement, file=csv_file)
    csv_file.close()

# TODO replace the 3 functions above
def upload_df(df, table_name):

    load_env_variables_from_file('env.txt')
    # Load credentials from environment variables 
    # Make sure to load the write env variables before running the coe
    db_endpoint = os.environ.get("DB_ENDPOINT")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")

    # Configuration Settings
    DB_CONFIG = {
        'endpoint': db_endpoint,
        'port': db_port,
        'name': db_name,
        'user': db_user,
        'password': db_password
    }
    # Establish DB connection
    connection = psycopg2.connect(
        host=DB_CONFIG['endpoint'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['name'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    cursor = connection.cursor()
    
    map_dict = {
        "datetime64[ns]": "DATE",
        "float64": "NUMERIC",
        "object": "VARCHAR",
        "int64": "BIGINT"
    }
    
    cols_map = zip(df.columns, df.dtypes.replace(map_dict))
    sql_cols = ", ".join("{} {}".format(n, d) for (n, d) in cols_map)

    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    query = f""" 
            CREATE TABLE {table_name} (
            {sql_cols}
            )
            """
    cursor.execute(query)

    schema_name = 'public'
    csv_file = io.StringIO()
    df.to_csv(csv_file, header=df.columns, index=False, encoding='utf-8')
    csv_file.seek(0)
    sql_statement = f"""
        COPY {schema_name}.{table_name} FROM STDIN WITH 
            CSV
            HEADER
            DELIMITER AS ','
        """
    cursor.copy_expert(sql=sql_statement, file=csv_file)
    csv_file.close()

    # Close connection
    connection.commit()
    cursor.close()
    connection.close()
   


ticker_selection = {
        'ticker': ['ETH-USD', 'BTC-USD', 'DOGE-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'XRP-USD', 'BNB-USD', 'TRX-USD', 'DAI-USD', 'AVAX-USD'],
        'name': ['Ethereum', 'Bitcoin', 'Doge', 'Cardano', 'Solana', 'Polkadot', 'XRP', 'BNB', 'Tron', 'Dai', 'Avalanche']
    }

def ticker_table():
    
    # Create DataFrame
    ticker_table_df = pd.DataFrame(ticker_selection)
    return ticker_table_df



def ingest_to_aws(START_DATE):

    END_DATE = None
    INDEX_AS_DATE = False
    INTERVAL = "1d"

    MAP_DICT = {
        "datetime64[ns]": "DATE",
        "float64": "NUMERIC",
        "object": "VARCHAR",
        "int64": "BIGINT"
    }

    load_env_variables_from_file('env.txt')
    # Load credentials from environment variables 
    # Make sure to load the write env variables before running the coe
    db_endpoint = os.environ.get("DB_ENDPOINT")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")

    # Configuration Settings
    DB_CONFIG = {
        'endpoint': db_endpoint,
        'port': db_port,
        'name': db_name,
        'user': db_user,
        'password': db_password
    }
    # Establish DB connection
    connection = psycopg2.connect(
        host=DB_CONFIG['endpoint'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['name'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    cursor = connection.cursor()

    if connection:
        # Create a table for selected tickers
        ticker_table_df = ticker_table()
        sql_cols = to_sql(ticker_table_df, MAP_DICT)
        create_table(cursor, "ticker_list", sql_cols)
        upload_data(cursor, ticker_table_df, "ticker_list")

        # create a single table to upload
        column_names = ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker']
        df_all_currencies = pd.DataFrame(columns=column_names)

        for symbol in ticker_table_df.ticker:
            df_all_currencies = pd.concat([df_all_currencies, 
            fetch_ticker_data(symbol, START_DATE, END_DATE, INDEX_AS_DATE, INTERVAL)])
        sql_cols = to_sql(df_all_currencies, MAP_DICT)
        create_table(cursor, "crypto_data", sql_cols)
        upload_data(cursor, df_all_currencies, "crypto_data")

        # Fetch and store sentiment data
        df_sentiment = fetch_sentiment_data()
        sql_cols = to_sql(df_sentiment, MAP_DICT)
        create_table(cursor, "sentiment_data", sql_cols)
        upload_data(cursor, df_sentiment, "sentiment_data")

        # Fetch and store index data
        indices = ["^IXIC", "^FTSE", "CL=F", "^STOXX50E", "GC=F", "^N225", "^HSI"]
        df_index = fetch_index_data(indices, START_DATE, END_DATE)
        df_index.columns = [clean_column_name(col_name) for col_name in df_index.columns]

        sql_cols = to_sql(df_index, MAP_DICT)
        create_table(cursor, "index_data", sql_cols)
        upload_data(cursor, df_index, "index_data")

        # Close connection
        connection.commit()
        cursor.close()
        connection.close()
    else:
        print("Connection Error, Please Fix!")

if __name__ == '__main__':
    START_DATE = "2020-01-01"
    ingest_to_aws(START_DATE)

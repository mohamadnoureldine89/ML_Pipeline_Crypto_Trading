# Cryptocurrency Analysis with ARIMA and VAR Models

This project provides tools for fetching, validating, and analyzing cryptocurrency data using ARIMA and VAR models.

## Features

- **Data Ingestion**: Fetch data from various sources and upload it to an AWS database.
- **Data Validation**: Pull cryptocurrency, market, and sentiment data from the database, then merge and validate it.
- **ARIMA Model**:
  - **Data Preparation**: Prepare data specifically for the ARIMA model.
  - **Model Training**: Train the ARIMA model using the prepared data and forecast future prices.
- **VAR Model**:
  - **Data Preparation**: Prepare data specifically for the VAR model.
  - **Model Training**: Train the VAR model using the prepared data.

## How to Run

1. **Set Your Cryptocurrency of Choice**:
   By default, the ticker is set to 'BTC-USD'. You can change this by updating the `ticker` variable.
   
2. **Data Ingestion**:
   Uncomment the `ingest_to_aws` function call to ingest data to AWS. Set the starting date and the number of currencies to fetch.

3. **Data Validation**:
   Run the `merge_data_one_ticker` and `validate_data` functions to validate your data.

4. **ARIMA Model**:
   - Prepare the data using the provided functions.
   - Train the ARIMA model using the specified order.
   - Print the forecasted values.

5. **VAR Model**:
   - Prepare the data using the provided functions.
   - Configure the train and test periods as well as other parameters.
   - Train the VAR model.
   - Print the output.

To execute the script, run:

```
python main.py
```

## Dependencies

- Will add the requirements.txt later
- The functions from modules like `p1_data_ingestion`, `p2_data_validation`, and so on need to be available.

## Notes

- Make sure to set up and configure your AWS credentials if you're ingesting data to AWS. Save them in a file called `env.txt`. It should look like this:
DB_ENDPOINT=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
- The script provides TODO notes, especially around the data ingestion part. Make sure to check and act on them as necessary.

---

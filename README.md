# Cryptocurrency Analysis with ARIMA and VAR Models

This project provides tools for fetching, validating, and analyzing cryptocurrency data using ARIMA and VAR models.

## Stages

- **Data Ingestion**: Fetch data from various sources and upload it to an AWS database.
- **Data Validation**: Pull cryptocurrency, market, and sentiment data from the database, then merge and validate it.
- **ARIMA Model**:
  - **Data Preparation**: Prepare data specifically for the ARIMA model.
  - **Model Training**: Train the ARIMA model using the prepared data and forecast future prices.
- **VAR Model**:
  - **Data Preparation**: Prepare data specifically for the VAR model.
  - **Model Training**: Train the VAR model using the prepared data.
  - **Model Optimization**: TODO, optimize the parameters of the VAR model

## How to Run

1. In a file named env.txt, save your connection configurations to AWS. It should looks like this:
DB_ENDPOINT=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=

2. Set up your venv as per requirements.txt.

3. Run p5_ml_all_currencies. Mind the if name == main statement.

---

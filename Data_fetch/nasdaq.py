#nasdaq.py 

import logging
import requests
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

def fetch_data_from_nasdaq(ticker_symbols, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for specific symbols using the NASDAQ API.

    Parameters:
        ticker_symbols (list): List of stock symbols to fetch data for.
        api_key (str): API key for NASDAQ.
        csv_dir (str): Directory to save the fetched data as CSV files.
        start_date (str, optional): Start date for data retrieval (YYYY-MM-DD).
        end_date (str, optional): End date for data retrieval (YYYY-MM-DD).

    Returns:
        list: List of file paths of the saved CSV files.
    """
    file_paths = []

    for symbol in ticker_symbols:
        symbol = symbol.strip()
        url = construct_api_url(symbol, api_key, start_date, end_date)

        try:
            response = get_data_from_api(url, symbol)
            if response is None:
                continue

            df = pd.DataFrame(response['data'])
            file_path = f'{csv_dir}/{symbol}_nasdaq.csv'
            save_data_to_csv(df, file_path)
            file_paths.append(file_path)

        except requests.exceptions.RequestException as e:
            handle_request_exception(symbol, e)
        except Exception as e:
            handle_exception(symbol, e)

    return file_paths

def construct_api_url(symbol, api_key, start_date, end_date):
    # Construct API URL based on provided parameters
    url = f'https://dataondemand.nasdaq.com/api/v1/historical/{symbol}?apiKey={api_key}'
    if start_date:
        url += f"&startDate={start_date}"
    if end_date:
        url += f"&endDate={end_date}"
    return url

def get_data_from_api(url, symbol):
    # Make API request and handle errors
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        if 'data' not in data:
            logger.warning(f"No time series data found for symbol {symbol} using NASDAQ API")
            return None
        return data
    except requests.exceptions.RequestException as e:
        return None

def save_data_to_csv(data, file_path):
    # Save given data to a CSV file
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")

def handle_request_exception(symbol, error):
    # Handle request exceptions and log error
    logger.error(f"Error fetching data for symbol {symbol} using NASDAQ API: {error}")

def handle_exception(symbol, error):
    # Handle unexpected exceptions and log error
    logger.error(f"An unexpected error occurred for symbol {symbol}: {error}")


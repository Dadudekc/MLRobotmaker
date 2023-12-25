# nasdaq.py
import pandas as pd
import logging
import requests

# Setup logging
logger = logging.getLogger(__name__)

def fetch_data_from_nasdaq(ticker_symbols, config, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for a specific symbol using the NASDAQ API.

    Parameters:
        symbol (str): Stock symbol to fetch data for.
        api_key (str): API key for NASDAQ.
        csv_directory (str): Directory to save the fetched data as a CSV file.

    Returns:
        str: File path of the saved CSV file, or None if an error occurs.
    """
    url = f'https://dataondemand.nasdaq.com/api/v1/historical/{symbol}?apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        data = response.json()
        if 'data' not in data:
            logger.warning(f"No time series data found for symbol {symbol} using NASDAQ API")
            return None

        df = pd.DataFrame(data['data'])
        file_path = f'{csv_directory}/{symbol}_nasdaq.csv'
        save_data_to_csv(df, file_path)
        logger.info(f"Data for symbol {symbol} saved to {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for symbol {symbol} using NASDAQ API: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred for symbol {symbol}: {e}")
        return None

def save_data_to_csv(data, file_path):
    """
    Save given data to a CSV file.

    Args:
        data (pd.DataFrame): Data to be saved.
        file_path (str): Path of the CSV file to save data.
    """
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")

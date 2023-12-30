# nasdaq.py
import pandas as pd
import logging
import requests

# Setup logging
logger = logging.getLogger(__name__)

def fetch_data_from_nasdaq(ticker_symbols, config, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for specific symbols using the NASDAQ API.

    Parameters:
        ticker_symbols (list): List of stock symbols to fetch data for.
        api_key (str): API key for NASDAQ.
        csv_dir (str): Directory to save the fetched data as CSV files.

    Returns:
        list: List of file paths of the saved CSV files.
    """
    file_paths = []

    for symbol in ticker_symbols:
        symbol = symbol.strip()
        url = f'https://dataondemand.nasdaq.com/api/v1/historical/{symbol}?apiKey={api_key}'

        if start_date:
            url += f"&startDate={start_date}"
        if end_date:
            url += f"&endDate={end_date}"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors

            data = response.json()
            if 'data' not in data:
                logger.warning(f"No time series data found for symbol {symbol} using NASDAQ API")
                continue

            df = pd.DataFrame(data['data'])
            file_path = f'{csv_dir}/{symbol}_nasdaq.csv'
            save_data_to_csv(df, file_path)
            logger.info(f"Data for symbol {symbol} saved to {file_path}")
            file_paths.append(file_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for symbol {symbol} using NASDAQ API: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for symbol {symbol}: {e}")

    return file_paths

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

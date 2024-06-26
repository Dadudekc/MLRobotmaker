#data_fetch.py

"""
This script is designed to fetch stock data from various APIs (AlphaVantage, Polygon.io, and NASDAQ).
It reads the required configuration from a file called 'config.ini', which should include API keys and other necessary settings.
The fetched data is then saved in CSV format in a specified directory.
"""

# 1. Imports
import os
import requests
import logging
import pandas as pd
import shutil
import configparser


# Define API URLs
API_URLS = {
    "alphavantage": "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}",
    "polygonio": "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/2000-01-01/2023-08-06?apiKey={}",
    "nasdaq": "https://dataondemand.nasdaq.com/api/v1/historical/{}?apiKey={}"
}



# 2. Configuration and Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(file_path):
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        print(f"The configuration file {file_path} was not found.")
        return None
    config.read(file_path)
    return config

config_file_path = r'C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MLRobot\config.ini'
config = load_config(config_file_path)

def validate_config(config):
    """
    Validate the presence and correctness of required configurations.

    Args:
        config (configparser.ConfigParser): The loaded configuration object.

    Returns:
        bool: True if validation is successful, False otherwise.
    """
    required_settings = {
        'API': ['alphavantage', 'polygonio', 'nasdaq'],
        'Settings': ['csv_directory']
    }

    for section, keys in required_settings.items():
        if not config.has_section(section):
            logger.error(f"Missing section: '{section}' in config file.")
            return False
        for key in keys:
            if not config.has_option(section, key):
                logger.error(f"Missing key: '{key}' in section: '{section}' in config file.")
                return False

    return True

# At the start of your script, after loading the config:
config = load_config(config_file_path)
if config is None or not validate_config(config):
    print("Configuration validation failed. Please check the config.ini file.")
    exit(1)


# 3. Utility Functions


def read_api_keys_from_config():
    """
    Retrieve API keys for various data sources from the configuration file.

    Returns:
        tuple: Contains API keys for AlphaVantage, PolygonIO, and NASDAQ respectively.
    """
    config = configparser.ConfigParser()  # This is the correct way to use ConfigParser
    config.read('config.ini')
    alpha_vantage_api_key = config.get('API', 'alphavantage')
    polygonio_api_key = config.get('API', 'polygonio')
    nasdaq_api_key = config.get('API', 'nasdaq')
    return alpha_vantage_api_key, polygonio_api_key, nasdaq_api_key


def get_csv_directory_from_config():
    """
    Retrieve the CSV directory path from the configuration file.

    Returns:
        str: The path of the CSV directory.
    """
    try:
        csv_directory = config.get('Settings', 'csv_directory')
        return csv_directory
    except configparser.NoOptionError:
        logger.error("CSV directory setting not found in the configuration file.")
        return None

import os

def get_full_path(relative_or_absolute_path):
    """
    Convert a relative path from the config file to an absolute path.
    If the path is already absolute, it is returned as is.

    Args:
        relative_or_absolute_path (str): The path from the config file.

    Returns:
        str: The absolute path.
    """
    if os.path.isabs(relative_or_absolute_path):
        return relative_or_absolute_path
    else:
        # Combine the relative path with the directory of the config file
        base_path = os.path.dirname(os.path.abspath(config_file_path))
        return os.path.join(base_path, relative_or_absolute_path)

# When retrieving the CSV directory path:
csv_directory = get_full_path(config.get('Settings', 'csv_directory'))


# 4.1 Fetch Data from Generic API

def fetch_data_from_api(ticker, api_key, csv_directory, selected_api):
    """
    Fetch stock data for a specific symbol from an API and save it as a CSV file.

    Parameters:
        api_url (str): The API URL endpoint.
        symbol (str): The stock symbol to fetch data for.
        api_key (str): The API key for authentication.
        csv_directory (str): Directory to save the fetched data as a CSV file.

    Returns:
        None: If an error occurs or no data is found.
        str: The file path of the saved CSV file if successful.
    """
    # Fetch the URL template from a dictionary of API URLs based on the selected API
    api_url = API_URLS[selected_api].format(symbol, api_key)

    try:
        # Log the initiation of data fetching
        logger.info(f"Initiating data fetch for symbol: {symbol}")
        print(f"Fetching data for symbol: {symbol}")

        # Fetch the data from the API
        response = requests.get(api_url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        # Validate the fetched data
        if not data:
            logger.warn(f"No data found for symbol: {symbol}")
            print(f"No data found for symbol: {symbol}")
            return None

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Save the data as a CSV file
        file_path = os.path.join(csv_directory, f"{symbol}.csv")
        df.to_csv(file_path, index=False)

        # Log successful data fetching
        logger.info(f"Data saved to {file_path}")
        print(f"Data saved to {file_path}")

        return file_path

    except requests.RequestException as e:
        logger.error(f"An error occurred while fetching data: {e}")
        print(f"An error occurred while fetching data for symbol: {symbol}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred for symbol: {symbol}")
        return None


# 4.2 Fetch Historical Data using AlphaVantage

def load_historical_data_alpha_vantage(symbols, api_key, csv_directory):
    """
    Fetch historical stock data for a list of symbols using the AlphaVantage API.

    Parameters:
        symbols (list): List of stock symbols to fetch data for.
        api_key (str): The API key for AlphaVantage.
        csv_directory (str): Directory to save the fetched data as a CSV file.

    Returns:
        dict: Dictionary with stock symbols as keys and corresponding data as DataFrames.
    """
    stock_data = {}
    for symbol in symbols:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors

            data = response.json()
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                logger.warning(f"No time series data found for symbol {symbol} using AlphaVantage")
                continue

            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'date'

            file_path = os.path.join(csv_directory, f'{symbol}_data.csv')
            df.to_csv(file_path)

            logger.info(f"Data for symbol {symbol} saved to {file_path}")
            stock_data[symbol] = df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for symbol {symbol} using AlphaVantage: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for symbol {symbol}: {e}")

    return stock_data

# 4.3 Fetch Historical Data using Polygon.io

def load_historical_data_polygonio(symbols, api_key, csv_directory):
    """
    Fetch historical stock data for a list of symbols using the Polygon.io API.

    Parameters:
        symbols (list): List of stock symbols to fetch data for.
        api_key (str): The API key for Polygon.io.
        csv_directory (str): Directory to save the fetched data as a CSV file.

    Returns:
        dict: Dictionary with stock symbols as keys and corresponding data as DataFrames.
    """
    stock_data = {}
    for symbol in symbols:
        url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2000-01-01/2023-08-06?apiKey={api_key}'
        try:
            response = requests.get(url)
            data = response.json()

            if not data.get('results', []):
                logger.warning(f"No time series data found for symbol {symbol} using Polygon.io")
                continue

            df = pd.DataFrame(data['results'])
            file_path = os.path.join(csv_directory, f'{symbol}_data.csv')
            df.to_csv(file_path)

            logger.info(f"Data for symbol {symbol} saved to {file_path}")
            stock_data[symbol] = df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for symbol {symbol} using Polygon.io: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for symbol {symbol}: {e}")

    return stock_data

# 4.4 Fetch Historical Data using NASDAQ API

def load_historical_data_nasdaq(symbols, api_key, csv_directory):
    """
    Fetch historical stock data for a list of symbols using the NASDAQ API.

    Parameters:
        symbols (list): List of stock symbols to fetch data for.
        api_key (str): The API key for NASDAQ.
        csv_directory (str): Directory to save the fetched data as a CSV file.

    Returns:
        dict: Dictionary with stock symbols as keys and corresponding data as DataFrames.
    """
    stock_data = {}
    for symbol in symbols:
        url = f'https://dataondemand.nasdaq.com/api/v1/historical/{symbol}?apiKey={api_key}'
        try:
            response = requests.get(url)
            data = response.json()

            if 'data' not in data:
                logger.warning(f"No time series data found for symbol {symbol} using NASDAQ API")
                continue

            df = pd.DataFrame(data['data'])
            file_path = os.path.join(csv_directory, f'{symbol}_data.csv')
            df.to_csv(file_path)

            logger.info(f"Data for symbol {symbol} saved to {file_path}")
            stock_data[symbol] = df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for symbol {symbol} using NASDAQ API: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for symbol {symbol}: {e}")

    return stock_data

#4.5 csv sorting

def manage_csv_files(csv_directory):
    # Create 'format1', 'format2', and 'unknown format' folders if they don't exist
    format1_folder = os.path.join(csv_directory, 'format1')
    format2_folder = os.path.join(csv_directory, 'format2')
    unknown_format_folder = os.path.join(csv_directory, 'unknown_format')  # New folder for unknown formats

    for folder in [format1_folder, format2_folder, unknown_format_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Iterate through CSV files in the main directory
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_directory, filename)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Determine the destination folder based on the format
            format1_columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n']
            format2_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in format1_columns):
                destination_folder = format1_folder
            elif all(col in df.columns for col in format2_columns):
                destination_folder = format2_folder
            else:
                destination_folder = unknown_format_folder  # Move to unknown format folder if format is not recognized

            # Move the file to the appropriate folder
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(file_path, destination_path)
            print(f"File {filename} moved to {destination_folder}")

def main(symbols, start_date=None, end_date=None, selected_api=None):
    """
    Main function to fetch stock data for a list of symbols using multiple data sources.
    """
    # Retrieve the CSV directory path from the config file
    csv_directory = get_csv_directory_from_config()
    if not csv_directory:
        print("Error: CSV directory path not found in the config file.")
        return

    # Validate and ensure the CSV directory exists or create it
    if not os.path.exists(csv_directory):
        try:
            os.makedirs(csv_directory, exist_ok=True)
        except OSError as e:
            print(f"An error occurred while creating the directory: {e}")
            return
        
    # Initialize variables for API keys
    alpha_vantage_api_key, polygonio_api_key, nasdaq_api_key = read_api_keys_from_config()

    # Initialize a dictionary to store the fetched data
    stock_data = {}

    # Logic to fetch data based on the selected API or all APIs
    for symbol in symbols:
        if selected_api:
            # Use the selected API's key and URL template
            api_key = config.get('API', selected_api)
            api_url = API_URLS[selected_api]
            fetched_data = fetch_data_from_api(symbol, api_key, csv_directory, selected_api)
            if fetched_data:
                stock_data[symbol] = fetched_data
        else:
            # Fetch from each API if no specific API is selected
            av_data = load_historical_data_alpha_vantage([symbol], alpha_vantage_api_key, csv_directory)
            po_data = load_historical_data_polygonio([symbol], polygonio_api_key, csv_directory)
            nasdaq_data = load_historical_data_nasdaq([symbol], nasdaq_api_key, csv_directory)
            stock_data.update(av_data or {})
            stock_data.update(po_data or {})
            stock_data.update(nasdaq_data or {})

    # Print symbols for which data fetch failed from all sources
    failed_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    for symbol in failed_symbols:
        print(f"Data fetch failed for stock: {symbol}")

    # Call the function to manage CSV files
    manage_csv_files(csv_directory)

if __name__ == "__main__":
    # Direct execution logic (e.g., for testing)
    symbols_input = input("Enter stock symbols separated by comma: ") or 'AAPL,MSFT'
    symbols = symbols_input.split(',')

    main(symbols)

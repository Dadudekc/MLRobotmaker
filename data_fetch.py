
from configparser import ConfigParser
import logging

def read_config(config_file_path='config.ini'):
    config = ConfigParser()
    try:
        if config.read(config_file_path) == []:
            logging.error(f"Error reading the configuration file: {config_file_path}")
            return None
        return config
    except Exception as e:
        logging.error(f"An exception occurred while reading the config file: {e}")
        return None



# Load configuration from config.ini
config = read_config('config.ini')
api_key = config.get('API', 'AlphaVantage', fallback='Your_Default_API_Key')
csv_directory = config.get('csv_directory', 'directory_path', fallback='Your_Default_CSV_Directory')


import os
import shutil
import pandas as pd
import requests
import logging
from configparser import ConfigParser

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data_from_api(api_url, symbol, api_key, csv_directory):
    """Fetches data from the API and saves it as a CSV file.
    
    Parameters:
        api_url (str): The API URL endpoint (loaded from config).
        symbol (str): The stock symbol to fetch data for.
        api_key (str): The API key for authentication (loaded from config).
        csv_directory (str): The directory to save the CSV file (loaded from config).
        
    Returns:
        None: If an error occurs.
        str: The file path of the saved CSV file.
    """
    try:
        # Log the initiation of data fetching
        logger.info(f"Initiating data fetch for symbol: {symbol}")
        print(f"Fetching data for symbol: {symbol}")
        
        # Fetch the data from the API
        response = requests.get(api_url.format(symbol, api_key))
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



def load_historical_data_alpha_vantage(symbols, api_key, csv_directory):
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



def load_historical_data_polygonio(symbols, api_key, csv_directory):
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

    return stock_data

def load_historical_data_nasdaq(symbols, api_key, csv_directory):
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

    return stock_data

def read_api_keys_from_config():
    config = ConfigParser()
    config.read('config.ini')
    alpha_vantage_api_key = config.get('API', 'AlphaVantage')
    polygonio_api_key = config.get('API', 'PolygonIO')
    nasdaq_api_key = config.get('API', 'NASDAQ')
    return alpha_vantage_api_key, polygonio_api_key, nasdaq_api_key

def main():
    # Define the directory to save CSV files
    csv_directory = r'C:\Users\Dagurlkc\OneDrive\Desktop\MyAIRobot\csv_files'

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    # Retrieve API keys from config.ini
    alpha_vantage_api_key, polygonio_api_key, nasdaq_api_key = read_api_keys_from_config()

    # Define symbols and API keys
    symbols = ['TSLA', 'SQ', 'KO', 'SPY', 'F', 'ARKK', 'QQQ', 'AAPL', 'AMZN']

    # Define the stock_data dictionary
    stock_data = {}

    # Attempt to fetch data from AlphaVantage for missing symbols
    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    alpha_vantage_data = load_historical_data_alpha_vantage(missing_symbols, alpha_vantage_api_key, csv_directory)
    stock_data.update(alpha_vantage_data)

    # Attempt to fetch data from Polygon.io for missing symbols
    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    polygonio_data = load_historical_data_polygonio(missing_symbols, polygonio_api_key, csv_directory)
    stock_data.update(polygonio_data)

    # Attempt to fetch data from NASDAQ API for missing symbols
    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    nasdaq_data = load_historical_data_nasdaq(missing_symbols, nasdaq_api_key, csv_directory)
    stock_data.update(nasdaq_data)

    # Handle data fetch failures for specific stocks
    failed_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    for symbol in failed_symbols:
        print(f"Data fetch failed for stock: {symbol}")

def main():
    csv_directory = 'csv_files'
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    alpha_vantage_api_key, polygonio_api_key, nasdaq_api_key = read_api_keys_from_config()
    symbols = ['TSLA', 'SQ', 'KO', 'SPY', 'F', 'ARKK', 'QQQ', 'AAPL', 'AMZN']

    stock_data = {}

    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    alpha_vantage_data = load_historical_data_alpha_vantage(missing_symbols, alpha_vantage_api_key, csv_directory)
    stock_data.update(alpha_vantage_data)

    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    polygonio_data = load_historical_data_polygonio(missing_symbols, polygonio_api_key, csv_directory)
    stock_data.update(polygonio_data)

    missing_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    nasdaq_data = load_historical_data_nasdaq(missing_symbols, nasdaq_api_key, csv_directory)
    stock_data.update(nasdaq_data)

    failed_symbols = [symbol for symbol in symbols if symbol not in stock_data]
    for symbol in failed_symbols:
        print(f"Data fetch failed for stock: {symbol}")

if __name__ == "__main__":
    main()

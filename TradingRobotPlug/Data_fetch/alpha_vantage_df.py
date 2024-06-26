#alpha_vantage_df.py

import os
import requests
import pandas as pd
import configparser
import logging
from typing import List, Optional

class AlphaVantageDataFetcher:
    def __init__(self, config_path: str, csv_dir: str):
        self.csv_dir = csv_dir
        self.logger = self.setup_logger()  # Initialize logger first
        self.config = self.load_config(config_path)
        self.api_key = self.get_api_key()
        self.base_url = "https://www.alphavantage.co/query"

    def setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("AlphaVantageDataFetcher")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

        log_file_path = os.path.join(os.path.dirname(__file__), "data_fetcher.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_config(self, config_path: str) -> Optional[configparser.ConfigParser]:
        config = configparser.ConfigParser()
        try:
            with open(config_path, 'r') as f:
                config.read_file(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            return None
        except configparser.Error as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            return None

    # ... [rest of the methods remain the same]


    def load_config(self, config_path: str) -> Optional[configparser.ConfigParser]:
        config = configparser.ConfigParser()
        try:
            with open(config_path, 'r') as f:
                config.read_file(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
        except configparser.Error as e:
            self.logger.error(f"Error parsing configuration file: {e}")
        return None

    def get_api_key(self) -> Optional[str]:
        if self.config is None:
            self.logger.error("Configuration is not loaded.")
            return None

        try:
            return self.config['API']['alphavantage']
        except KeyError:
            self.logger.error("Alpha Vantage API key not found in config file.")
        return None


    def fetch_data(self, ticker_symbols: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        if start_date is None:
            start_date = "2023-01-01"  # default start date
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        all_data = pd.DataFrame()

        for symbol in ticker_symbols:
            data = self.fetch_data_for_symbol(symbol, start_date, end_date)
            if data is not None:
                all_data = pd.concat([all_data, data])

        return all_data

    def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            # Debug: Print API Key and Base URL
            print("Debug: API Key:", self.api_key)
            print("Debug: Base URL:", self.base_url)

            response = requests.get(self.base_url, params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",  # Fetch full-length time series
                "datatype": "json",  # Fetch data in JSON format
            })
            response.raise_for_status()

            # Check if the response contains valid JSON data
            if response.text:
                data = response.json()
                
                if "Time Series (Daily)" not in data:
                    self.logger.warning(f"No data found for symbol: {symbol}")
                    return None

                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index = pd.to_datetime(df.index)  # Convert index to datetime
                
                # Sort and ensure a monotonic index
                df = df.sort_index()
                
                df.index.name = "date"
                df['symbol'] = symbol

                # Perform date-based slicing
                filtered_df = df.loc[start_date:end_date]

                return filtered_df
            else:
                self.logger.warning(f"No JSON data in the response for symbol: {symbol}")
                return None

        except requests.RequestException as e:
            self.logger.error(f"Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for symbol {symbol}: {e}")
            return None



    def validate_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """
        Validates that the start and end dates are within the DataFrame's date range.
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if df.index.min() > start_date or df.index.max() < end_date:
            self.logger.warning(f"Requested date range {start_date} to {end_date} is out of available data range for {df['symbol'].iloc[0]}")
            return False
        return True

    def save_data_to_csv(self, data: pd.DataFrame, symbol: str) -> None:
        if data.empty:
            self.logger.warning(f"No data to save for symbol: {symbol}")
            return

        try:
            file_path = os.path.join(self.csv_dir, f"{symbol}_data.csv")
            data.to_csv(file_path)
            self.logger.info(f"Data for {symbol} saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol} to CSV: {e}")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, 'config.ini')
        csv_dir = os.path.join(script_dir, "Data_fetch")

        data_fetcher = AlphaVantageDataFetcher(config_path, csv_dir)
        ticker_symbols = ["AAPL"]
        start_date = "2022-01-01"
        end_date = "2022-12-31"

        fetched_data = data_fetcher.fetch_data(ticker_symbols, start_date, end_date)

        if not fetched_data.empty:
            data_fetcher.save_data_to_csv(fetched_data, "AAPL")
            print(f"Data fetched and saved for {ticker_symbols}")
        else:
            print("No data fetched.")
    except Exception as e:
        print(f"An error occurred: {e}")
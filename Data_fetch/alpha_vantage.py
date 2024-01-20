#alpha_vantage.py

import os
import requests
import pandas as pd
import logging
from typing import List

class AlphaVantageDataFetcher:
    def __init__(self, api_key: str, csv_dir: str):
        self.api_key = api_key
        self.csv_dir = csv_dir
        self.base_url = "https://www.alphavantage.co/query"
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("AlphaVantageDataFetcher")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

        # Create a log file in the same directory as the script
        log_file_path = os.path.join(os.path.dirname(__file__), "data_fetcher.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def fetch_data(self, ticker_symbols: List[str]) -> pd.DataFrame:
        all_data = pd.DataFrame()

        for symbol in ticker_symbols:
            data = self._fetch_data_for_symbol(symbol)
            if data is not None:
                all_data = pd.concat([all_data, data])

        return all_data

    def _fetch_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        try:
            api_params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
            }

            response = requests.get(self.base_url, params=api_params)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "date"
            df['symbol'] = symbol

            return df

        except requests.RequestException as e:
            self.logger.error(f"Error fetching data for symbol {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for symbol {symbol}: {e}")
            return None

if __name__ == "__main__":
    ticker_symbols = ["AAPL"]
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    csv_dir = "/path/to/your/csv/directory"

    if api_key is None:
        print("Error: Alpha Vantage API key not set in environment variable ALPHA_VANTAGE_API_KEY")
    else:
        data_fetcher = AlphaVantageDataFetcher(api_key, csv_dir)
        fetched_data = data_fetcher.fetch_data(ticker_symbols)
        if not fetched_data.empty:
            print(fetched_data.head())

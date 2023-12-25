#alpha_vantage.py

import requests
import pandas as pd
import logging
import os
from alpha_vantage.timeseries import TimeSeries

# Setting up logging
logger = logging.getLogger(__name__)

def fetch_data_from_alpha_vantage(ticker_symbols, config, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch stock data for specific symbols from AlphaVantage API.

    Parameters:
        ticker_symbols (list): The stock symbols to fetch data for.
        api_key (str): The API key for AlphaVantage.
        csv_dir (str): Directory to save the fetched data as CSV files.
    """
    all_data = pd.DataFrame()

    for symbol in ticker_symbols:
        api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"

        try:
            logger.info(f"Fetching data for symbol: {symbol} from AlphaVantage")
            response = requests.get(api_url)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()

            # Check if data is received
            if "Time Series (Daily)" not in data:
                logger.warning(f"No data found for symbol: {symbol}")
                continue

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "date"
            df['symbol'] = symbol  # Add a column for the symbol

            all_data = pd.concat([all_data, df])

        except requests.RequestException as e:
            logger.error(f"Error fetching data from AlphaVantage for symbol {symbol}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching data from AlphaVantage for symbol {symbol}: {e}")

    return all_data

#example usage
if __name__ == "__main__":
    ticker_symbols = ["AAPL"]  # Example list of symbols
    api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
    csv_dir = "/path/to/your/csv/directory"
    config = {}  # Assuming you have a config dictionary, or load it from a file

    # Assuming start_date and end_date are not needed for this test, you can set them to None
    start_date = None
    end_date = None

    fetch_data_from_alpha_vantage(ticker_symbols, config, api_key, csv_dir, start_date, end_date)


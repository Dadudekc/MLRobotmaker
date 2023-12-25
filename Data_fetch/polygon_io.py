#polygon_io

import pandas as pd
import requests
from datetime import datetime

def fetch_data_from_polygon(ticker_symbols, config, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for multiple symbols using the Polygon.io API.

    Args:
        ticker_symbols (list of str): List of stock symbols to fetch data for.
        config (dict): Configuration dictionary.
        api_key (str): API key for Polygon.io.
        csv_dir (str): Directory to save CSV files.
        start_date (str, optional): Start date for data fetching in YYYY-MM-DD format.
        end_date (str, optional): End date for data fetching in YYYY-MM-DD format. Defaults to current date.

    Returns:
        dict of pd.DataFrame: Dictionary where each key is a symbol and value is a DataFrame containing the fetched stock data.
    """

    # Default end date to current date if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data_frames = {}

    for symbol in ticker_symbols:
        # Construct the URL for the API call
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"

        # Make the API request
        response = requests.get(url)
        if response.status_code == 200:
            # Process the response data
            data = response.json()
            df = pd.DataFrame(data.get('results', []))

            # Debugging log for DataFrame type
            print(f"Type of fetched data for {symbol}: {type(df)}")

            if isinstance(df, pd.DataFrame):
                # Convert timestamps to readable dates, and perform any additional processing
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.drop(['t'], axis=1, inplace=True)

                # Add DataFrame to the dictionary
                data_frames[symbol] = df
            else:
                print(f"Fetched data for {symbol} is not a DataFrame.")
        else:
            print(f"Failed to fetch data for {symbol}. HTTP Status Code: {response.status_code}")

    return data_frames

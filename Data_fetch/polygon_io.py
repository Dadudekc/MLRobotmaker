#polygon_io

# Import necessary libraries at the top of the file
import os
import pandas as pd
import requests
from datetime import datetime

def fetch_data_from_polygon(ticker_symbols, api_key, csv_dir, start_date=None, end_date=None):
    """
    Fetch historical stock data for multiple symbols using the Polygon.io API.

    Args:
        ticker_symbols (list of str): List of stock symbols to fetch data for.
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
        # Construct the CSV file path using f-strings for clarity
        csv_file_path = os.path.join(csv_dir, f"{symbol}_data.csv")

        if os.path.exists(csv_file_path):
            # Load data from the existing CSV file with automatic date parsing
            df = pd.read_csv(csv_file_path, index_col=0, parse_dates=['date'])
            data_frames[symbol] = df
            print(f"Loaded data for {symbol} from the CSV file.")
        else:
            # Fetch data for the symbol and handle exceptions
            df = fetch_data_for_symbol(symbol, api_key, start_date, end_date)
            
            if df is not None:
                # Save the data to the CSV file for future use with index
                df.to_csv(csv_file_path, index=True)
                
                # Add DataFrame to the dictionary
                data_frames[symbol] = df

    return data_frames

def fetch_data_for_symbol(symbol, api_key, start_date, end_date):
    """
    Fetch historical stock data for a single symbol using the Polygon.io API.

    Args:
        symbol (str): Stock symbol to fetch data for.
        api_key (str): API key for Polygon.io.
        start_date (str): Start date for data fetching in YYYY-MM-DD format.
        end_date (str): End date for data fetching in YYYY-MM-DD format.

    Returns:
        pd.DataFrame or None: DataFrame containing the fetched stock data or None if the fetch fails.
    """
    # Construct the URL for the API call
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"

    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request fails

        # Process the response data
        data = response.json()
        results = data.get('results', [])

        if isinstance(results, list):
            # Convert timestamps to readable dates
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.drop(['t'], axis=1, inplace=True)
            
            return df
        else:
            print(f"Fetched data for {symbol} is not in the expected format.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {symbol}. Error: {str(e)}")
        return None

if __name__ == "__main__":
    ticker_symbols = ["AAPL", "MSFT"]  # Replace with your desired symbols
    api_key = "YOUR_API_KEY"  # Replace with your Polygon.io API key
    csv_dir = "data"  # Replace with your desired CSV directory
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    data_frames = fetch_data_from_polygon(ticker_symbols, api_key, csv_dir, start_date, end_date)

    # You can now work with the fetched data frames as needed.

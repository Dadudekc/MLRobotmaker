#async_threaded_data_fetch.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from data_fetch import fetch_data_from_api  # Ensure this is the correct import

# Additional arguments required by fetch_data_from_api
API_KEY = "your_api_key_here"
CSV_DIRECTORY = "path_to_csv_directory"
SELECTED_API = "name_of_selected_api"

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

async def async_data_fetch(ticker, api_key, csv_directory, selected_api):
    try:
        if is_debug:
            print(f"Debug: Fetching data for {ticker} using API: {selected_api}")
        data = await fetch_data_from_api(ticker, api_key, csv_directory, selected_api)
        if is_debug:
            print(f"Debug: Fetched data for {ticker}: {data}")
    except Exception as e:
        if is_debug:
            print(f"Debug: Error fetching data for {ticker}: {e}")

def run_async_data_fetch(ticker, api_key, csv_directory, selected_api):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_data_fetch(ticker, api_key, csv_directory, selected_api))

def start_async_data_fetch(ticker_list, api_key, csv_directory, selected_api):
    max_threads = min(len(ticker_list), 5)
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(run_async_data_fetch, ticker, api_key, csv_directory, selected_api) for ticker in ticker_list]


if __name__ == "__main__":
    ticker_list = ['AAPL', 'MSFT', 'GOOGL']
    create_directory('output')
    API_KEY = "your_api_key_here"
    CSV_DIRECTORY = "path_to_csv_directory"
    SELECTED_API = "name_of_selected_api"
    start_async_data_fetch(ticker_list, API_KEY, CSV_DIRECTORY, SELECTED_API)

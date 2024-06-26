#main.py

import os
import sys
import traceback

# Get the directory containing 'main.py' (the script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project directory by going up one level from the script's location
project_dir = os.path.dirname(script_dir)

# Append the project directory to the Python path
sys.path.append(project_dir)

from Data_fetch import alpha_vantage_df, polygon_io, nasdaq
from Utilities import config_utils, logging_utils, file_management
from Data_fetch.alpha_vantage_df import AlphaVantageDataFetcher

def main(csv_dir='default_csv_dir', ticker_symbols=['default_symbol'], start_date='default_start_date', end_date='default_end_date', selected_api='Alpha Vantage'):
    try:

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set up the relative path to config.ini
        config_path = os.path.join(script_dir, '..', 'config.ini')


        # Create an instance of AlphaVantageDataFetcher
        av_fetcher = AlphaVantageDataFetcher(config_path, csv_dir)

        # Fetch and save data based on the selected API
        for symbol in ticker_symbols:
            try:
                file_path = os.path.join(csv_dir, f'{symbol}_data.csv')
                if selected_api == 'Alpha Vantage':
                    av_data = av_fetcher.fetch_data([symbol], start_date, end_date)
                    if av_data is not None:
                        av_fetcher.save_data_to_csv(av_data, symbol)
                elif selected_api == 'Polygon':
                    # Add the logic for Polygon API
                    pass
                elif selected_api == 'Nasdaq':
                    # Add the logic for Nasdaq API
                    pass
                else:
                    print(f"Unsupported API selected: {selected_api}")
                    return
            except Exception as e:
                traceback.print_exc()
                print(f"Error during data fetching for {symbol}: {e}")

    except Exception as e:
        traceback.print_exc()
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    # Call the main function with your desired parameters
    main(csv_dir=os.path.join(project_dir, 'Data_fetch'), ticker_symbols=['AAPL', 'MSFT'], start_date='2022-01-01', end_date='2022-12-31', selected_api='Alpha Vantage')
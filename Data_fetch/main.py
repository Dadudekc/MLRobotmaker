#main.py

# Importing necessary modules
from . import alpha_vantage, polygon_io, nasdaq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utilities import config_utils, logging_utils, file_management

def main(csv_dir='default_csv_dir', ticker_symbols=['default_symbol'], start_date='default_start_date', end_date='default_end_date', selected_api='default_api'):
    # Adjusted the path to reference the parent directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config.ini')
    
    # Load and validate the configuration
    config = config_utils.load_config(config_path)
    if not config_utils.validate_config(config):
        print("Configuration validation failed.")
        return

    # Set up logging
    logging_utils.setup_logging(config)

    # Fetch the API key from the configuration based on the selected API
    api_keys = {
        'AlphaVantage': config['API']['alphavantage'],
        'Polygon': config['API']['polygonio'],
        'Nasdaq': config['API']['nasdaq']
    }
    print(f"Loaded API keys from configuration: {api_keys}")
    api_key = api_keys.get(selected_api, None)
    if not api_key:
        print(f"API key for {selected_api} not found in configuration.")
        return

    # Define symbols to fetch data for
    symbols = ticker_symbols if ticker_symbols else ['AAPL', 'MSFT', 'GOOGL']

    # Fetch data from different APIs based on selected_api
    if selected_api == 'AlphaVantage':
        av_data = alpha_vantage.fetch_data_from_alpha_vantage(ticker_symbols, config, api_key, csv_dir, start_date, end_date)
        print(f"Type of data from AlphaVantage API: {type(av_data)}")  # Debug print statement
        file_path = os.path.join(csv_dir, 'alpha_vantage_data.csv')
        file_management.save_data_to_csv(av_data, file_path)
    elif selected_api == 'Polygon':
        po_data = polygon_io.fetch_data_from_polygon(ticker_symbols, config, api_key, csv_dir, start_date, end_date)
        print(f"Type of data from Polygon.io API: {type(po_data)}")  # Debug print statement
        file_path = os.path.join(csv_dir, 'polygon_io_data.csv')
        file_management.save_data_to_csv(po_data, file_path)
    elif selected_api == 'Nasdaq':
        nasdaq_data = nasdaq.fetch_data_from_nasdaq(ticker_symbols, config, api_key, csv_dir, start_date, end_date)
        print(f"Type of data from Nasdaq API: {type(nasdaq_data)}")  # Debug print statement
        file_path = os.path.join(csv_dir, 'nasdaq_data.csv')
        file_management.save_data_to_csv(nasdaq_data, file_path)


    # Additional operations can be added as needed

if __name__ == '__main__':
    main()

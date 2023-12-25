#config_handling.py

# Section 1:Configuration Module

# --- Standard Library Imports ---
import warnings
import json
import os
import sys
import logging
import configparser
import glob
from pathlib import Path

# --- External Library Imports ---
import pandas as pd

# --- Configuration Settings and Logging ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")
warnings.simplefilter(action='ignore', category=Warning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini')
loading_path = config['Paths']['loading_path']
saving_path = config['Paths']['saving_path']

# --- Utility Functions for Configuration ---

def get_format_path(format_number):
    """
    Function to get the path for a specified data format.
    
    Args:
    format_number (int): The number representing the data format.

    Returns:
    Path: The path for the specified data format.
    """
    project_dir = Path(__file__).resolve().parent
    config_file_path = project_dir / 'config.ini'
    
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if format_number == 1:
        return project_dir / config.get('Paths', 'format1processeddata', fallback='csv_files/format1')
    elif format_number == 2:
        return project_dir / config.get('Paths', 'format2processeddata', fallback='csv_files/format2')

def detect_format(file_path):
    """
    Detects the format of the data file.

    Args:
    file_path (str): Path to the data file.

    Returns:
    int: Format number of the file.
    """
    df = pd.read_csv(file_path)

    if "Meta Data" in df.columns:
        return 1
    elif set(["h", "l", "o", "v", "c"]).issubset(df.columns):
        return 2
    else:
        return 0

def detect_and_transform_format(file_path):
    """
    Transforms the data file to a standard format if needed.

    Args:
    file_path (str): Path to the data file.

    Returns:
    str: Path to the transformed file.
    """
    df = pd.read_csv(file_path)

    if "Meta Data" in df.columns:
        # Transformation for the first alternative format
        data = [json.loads(row) for row in df[df.columns[-1]]]
        new_df = pd.DataFrame(data)
        new_df.columns = ['open', 'high', 'low', 'close', 'volume']
        new_df['date'] = df[df.columns[2]].values

    elif "h" in df.columns and "l" in df.columns and "o" in df.columns and "v" in df.columns and "c" in df.columns:
        # Transformation for the second alternative format
        new_df = df.rename(columns={"h": "high", "l": "low", "o": "open", "v": "volume", "c": "close"})
        new_df['date'] = df.index

    else:
        new_df = df  # No transformation needed

    new_file_path = Path(file_path).with_name("transformed_" + Path(file_path).name)
    new_df.to_csv(new_file_path, index=False)

    return new_file_path

def handle_nan_in_dataframe(df):
    """
    Handles NaN values in the dataframe by filling them with appropriate values.

    Args:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The dataframe with NaN values handled.
    """
    for column in ['high', 'low', 'close']:
        df[column].fillna(method='ffill', inplace=True)
        df[column].fillna(method='bfill', inplace=True)
    return df

def fetch_csv_files_and_tickers(loading_path):
    """
    Fetches CSV files and extracts ticker symbols from the filenames.

    Args:
    loading_path (str): Directory path where CSV files are stored.

    Returns:
    list: List of CSV file paths.
    list: List of ticker symbols extracted from file names.
    """
    csv_files = glob.glob(os.path.join(loading_path, '*.csv'))
    tickers = [os.path.basename(file).split('_')[0] for file in csv_files]
    return csv_files, tickers

def read_configuration_settings():
    """
    Reads the configuration settings from the config.ini file.

    Returns:
    dict: Dictionary containing the configuration settings.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    config_settings = {
        'loading_path': config['Paths']['loading_path'],
        'saving_path': config['Paths']['saving_path']
    }
    return config_settings

def load_user_settings(settings_file='config.ini'):
    """
    Loads user settings from a configuration file.
    """
    config = configparser.ConfigParser()
    config.read(settings_file)
    return dict(config['UserSettings'])

# Example usage
user_settings = load_user_settings()
# Use user_settings in your data processing functions

if __name__ == "__main__":
    # Read the configuration settings
    config_settings = read_configuration_settings()

    # Fetch CSV files and tickers
    csv_files, tickers = fetch_csv_files_and_tickers(config_settings['loading_path'])

    # Now csv_files and tickers can be used further in the script


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


from configparser import ConfigParser



import os
import pandas as pd
import ta
import logging
import configparser

from keras.models import load_model  # Import the load_model function
from alpha_vantage.timeseries import TimeSeries
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from config.ini
config = read_config('config.ini')  # Updated code using the read_config function  # Adjust the path to your config file

# Separate Configuration Settings
def load_config():
    return {
        'api_key': config.get('API', 'AlphaVantage', fallback='Your_Default_API_Key'),
        'models_directory': config.get('Paths', 'models_directory', fallback='Your_Default_Models_Directory'),
        'grandfather_models_directory': config.get('Paths', 'grandfather_models_directory', fallback='Your_Default_Grandfather_Models_Directory'),
        'start_date': config.get('Backtesting', 'start_date', fallback='Your_Default_Start_Date'),
        'end_date': config.get('Backtesting', 'end_date', fallback='Your_Default_End_Date')
    }

# Updated backtest_regression function with move accuracy

def backtest_regression(models, symbols, start_date, end_date, api_key):
    backtest_results = {}  # Initialize an empty dictionary to store the backtest results
    
    # Initialize the Alpha Vantage API for fetching stock data
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    for symbol in symbols:
        try:
            # Fetch stock data for the symbol between start_date and end_date
            data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Apply the loaded model to the fetched data to get the predicted prices or signals
            if symbol in models:
                model = models[symbol]
                signals = model.predict(data)
                
                # Calculate daily returns based on the predicted prices
                daily_returns = np.diff(signals) / signals[:-1]
                
                # Calculate the portfolio value based on daily returns
                portfolio_value = np.cumprod(1 + daily_returns)
                
                # Calculate the original accuracy if actual returns are available
                if 'actual_returns' in data.columns:
                    actual_returns = np.diff(data['actual_returns'].values) / data['actual_returns'].values[:-1]
                    correct_predictions = np.sum(np.sign(daily_returns) == np.sign(actual_returns))
                    total_predictions = len(daily_returns)
                    original_accuracy = (correct_predictions / total_predictions) * 100
                else:
                    original_accuracy = None
                
                # Calculate move accuracy
                move_accuracy = np.sum(np.sign(daily_returns) == np.sign(actual_returns)) / len(daily_returns) * 100
                
                # Store these calculated metrics in the dictionary for the symbol
                backtest_results[symbol] = {
                    'daily_returns': daily_returns,
                    'portfolio_value': portfolio_value,
                    'original_accuracy': original_accuracy,
                    'move_accuracy': move_accuracy
                }
                
        except Exception as e:
            print(f"An error occurred while backtesting the symbol {symbol}: {e}")
    
    return backtest_results

# Function for loading models from specified directories
def load_models(model_directory):
    loaded_models = {}
    model_files = os.listdir(model_directory)
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(model_directory, model_file)
        try:
            if model_file.endswith('.h5'):
                loaded_model = load_model(model_path)  # Load the model using load_model for .h5 files
            else:
                loaded_model = joblib.load(model_path)  # Keep joblib for other formats
            loaded_models[model_name] = loaded_model
            print(f"Loaded model '{model_name}' from '{model_path}'")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
    return loaded_models

def preprocess_data(backtest_data_folder):
    backtest_data_files = [file for file in os.listdir(backtest_data_folder) if file.endswith('.csv')]
    for csv_file in backtest_data_files:
        file_path = os.path.join(backtest_data_folder, csv_file)
        data = pd.read_csv(file_path)
        
        # Calculate technical indicators
        data['SMA_10'] = ta.trend.SMAIndicator(data['close'], window=10).sma_indicator()
        data['EMA_10'] = ta.trend.EMAIndicator(data['close'], window=10).ema_indicator()
        data['Price_RoC'] = ta.momentum.ROCIndicator(data['close'], window=10).roc()
        data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=10).rsi()
        data['MACD'] = ta.trend.MACD(data['close']).macd()
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        data['Bollinger_Mid'] = bollinger.bollinger_mavg()
        
        # Save the processed data to a new CSV file
        processed_file_path = os.path.join(backtest_data_folder, 'processed_data', f'{os.path.splitext(csv_file)[0]}_processed_data.csv')
        data.to_csv(processed_file_path, index=False)
        logger.info(f"Processed data saved to: {processed_file_path}")

# Main Function
def main():
    config_settings = load_config()
    preprocess_data(r'C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MyAIRobot\backtest_data')

    regression_models = load_models(config_settings['models_directory'])  # Assuming regression models
    symbols_to_test = ['TSLA']
    start_date = config_settings['start_date']
    end_date = config_settings['end_date']

    for symbol in symbols_to_test:
        print(f"Backtesting for symbol: {symbol}")

        backtest_results = backtest_regression(regression_models, [symbol], start_date, end_date, config_settings['api_key'])
        print("Backtesting Results:")
        print(backtest_results)

if __name__ == "__main__":
    main()

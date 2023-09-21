
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



# Load configuration from config.ini
config = read_config('config.ini')


#data_processing1

import pandas as pd
import ta  
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    format1_folder = config.get('Paths', 'format1processeddata', fallback='Your_Default_Format1_Folder')
    
    # Get a list of CSV files in the format1 folder
    format1_files = [file for file in os.listdir(format1_folder) if file.endswith('.csv')]
    
    # Loop through each CSV file and calculate indicators
    for csv_file in format1_files:
        file_path = os.path.join(format1_folder, csv_file)
        
        # Load the CSV file as a DataFrame
        data = pd.read_csv(file_path)
        
        # Calculate RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['c']).rsi()

        # Calculate Stochastic Oscillator
        data['stoch_oscillator'] = ta.momentum.StochasticOscillator(data['h'], data['l'], data['c']).stoch()

        # Calculate EMA (20-day)
        data['ema_20'] = ta.trend.EMAIndicator(data['c'], window=20).ema_indicator()

        # Calculate Bollinger Bands
        data['bb_bands'] = ta.volatility.BollingerBands(data['c']).bollinger_mavg()

        # Calculate Average True Range (ATR)
        data['atr'] = ta.volatility.AverageTrueRange(data['h'], data['l'], data['c']).average_true_range()

        # Calculate MACD
        data['macd'] = ta.trend.MACD(data['c']).macd()
        
        # Create a subfolder for processed data if it doesn't exist
        processed_data_folder = os.path.join(format1_folder, 'f1_processed_data')
        os.makedirs(processed_data_folder, exist_ok=True)
        
        # Save the processed data to a new CSV file
        processed_file_path = os.path.join(processed_data_folder, f'{os.path.splitext(csv_file)[0]}_processed_data.csv')
        data.to_csv(processed_file_path, index=False)
        
        logger.info(f"Processed data saved to: {processed_file_path}")

if __name__ == "__main__":
    main()

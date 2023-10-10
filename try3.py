#data_processing1.py

# Section 1: Imports and Initial Configurations

import os
import sys
import logging
import configparser
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.utils import dropna
from ta.trend import CCIIndicator, PSARIndicator
from pathlib import Path
from ta.trend import IchimokuIndicator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_format1_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(base_path, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return os.path.join(base_path, config.get('Paths', 'format1processeddata', fallback='csv_files/format1'))

format1_path = get_format1_path()

import glob

csv_files = glob.glob(os.path.join(format1_path, "*.csv"))
tickers = [os.path.basename(file).split('_')[0] for file in csv_files]



import os
import sys
import logging
import configparser
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.utils import dropna
from ta.trend import CCIIndicator, PSARIndicator
from pathlib import Path
from ta.trend import IchimokuIndicator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_format1_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(base_path, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return os.path.join(base_path, config.get('Paths', 'format1processeddata', fallback='csv_files/format1'))

format1_path = get_format1_path()

import glob

csv_files = glob.glob(os.path.join(format1_path, "*.csv"))
tickers = [os.path.basename(file).split('_')[0] for file in csv_files]



import os
import sys
import logging
import configparser
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.utils import dropna
from ta.trend import CCIIndicator, PSARIndicator
from pathlib import Path
from ta.trend import IchimokuIndicator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_format2_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(base_path, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return os.path.join(base_path, config.get('Paths', 'format2processeddata', fallback='csv_files/format2'))

format2_path = get_format2_path()

import glob

csv_files = glob.glob(os.path.join(format2_path, "*.csv"))
tickers = [os.path.basename(file).split('_')[0] for file in csv_files]

#Data Processing and Feature Selection Functions

def process_data(ticker, features_to_add):
    df = pd.read_csv(os.path.join(format1_path, f"{ticker}_data.csv"))

    if 'ATR' in features_to_add:
        df = add_atr(df)
    if 'MA' in features_to_add:
        df = add_moving_average(df)
    if 'EMA' in features_to_add:
        df = add_exponential_moving_average(df)
    if 'BB' in features_to_add:
        df = add_bollinger_bands(df)
    if 'SO' in features_to_add:
        df = add_stochastic_oscillator(df)
    if 'MAE' in features_to_add:
        df = add_mae(df)
    if 'RVI' in features_to_add:
        df = add_rvi(df)
    if 'PVT' in features_to_add:
        df = add_pvt(df)
    if 'Chaikin_Oscillator' in features_to_add:
        df = add_chaikin_oscillator(df)
    if 'DPO' in features_to_add:
        df = add_dpo(df)
    if 'Williams_R' in features_to_add:
        df = add_williams_r(df)
    if 'Elder_Ray_Index' in features_to_add:
        df = add_elder_ray_index(df)
    if 'ROC' in features_to_add:
        df = add_roc(df)
    if 'OBV' in features_to_add:
        df = add_obv(df)
    if 'MFI' in features_to_add:
        df = add_mfi(df)
    if 'PPO' in features_to_add:
        df = add_ppo(df)
    if 'CCI' in features_to_add:
        df = add_cci(df)
    if 'Stoch_RSI' in features_to_add:
        df = add_stoch_rsi(df)
    if 'Ultimate_Oscillator' in features_to_add:
        df = add_ultimate_oscillator(df)
    if 'PSAR' in features_to_add:
        df = add_psar(df)
    if 'ADL' in features_to_add:
        df = add_adl(df)
    if 'ADX' in features_to_add:
        df = add_adx(df)
    if 'ForceIndex' in features_to_add:
        df = add_force_index(df)
    if 'KeltnerChannel' in features_to_add:
        df = add_keltner_channel(df)
    if 'CoppockCurve' in features_to_add:
        df = add_coppock_curve(df)
    if 'Ichimoku' in features_to_add:
        df = add_ichimoku(df)
    if 'ZigZag' in features_to_add:
        df = add_zigzag(df)
    if 'HMA' in features_to_add:
        df = add_hma(df)
    if 'QStick' in features_to_add:
        df = add_qstick(df)
    if 'TRIX' in features_to_add:
        df = add_trix(df)
    if 'Vortex' in features_to_add:
        df = add_vortex(df)
    df.dropna(inplace=True)

    output_filename = os.path.join(format1_path, f"{ticker}_processed_data.csv")
    df.to_csv(output_filename, index=False)
    logging.info(f"Processed and saved enriched data to {output_filename}")

def select_features(features_dict):
    selected_features = []
    print("Please select the technical indicators you want to add:")
    for key, value in features_dict.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number(s) of the indicator (comma-separated), 'All', 'Done' or 'Quit' to finish: ").strip().lower()
        if choice == 'all':
            return list(features_dict.values())[:-1]
        elif choice == 'quit' or choice == 'done' or choice == '22':
            break
        else:
            choices = [ch.strip() for ch in choice.split(',')]
            for ch in choices:
                if ch in features_dict:
                    selected_features.append(features_dict[ch])
                else:
                    print("Invalid choice, please try again.")

    return selected_features

#Data Processing and Feature Selection Functions

def process_data(ticker, features_to_add):
    df = pd.read_csv(os.path.join(format2_path, f"{ticker}_data.csv"))

    if 'ATR' in features_to_add:
        df = add_atr(df)
    if 'MA' in features_to_add:
        df = add_moving_average(df)
    if 'EMA' in features_to_add:
        df = add_exponential_moving_average(df)
    if 'BB' in features_to_add:
        df = add_bollinger_bands(df)
    if 'SO' in features_to_add:
        df = add_stochastic_oscillator(df)
    if 'MAE' in features_to_add:
        df = add_mae(df)
    if 'RVI' in features_to_add:
        df = add_rvi(df)
    if 'PVT' in features_to_add:
        df = add_pvt(df)
    if 'Chaikin_Oscillator' in features_to_add:
        df = add_chaikin_oscillator(df)
    if 'DPO' in features_to_add:
        df = add_dpo(df)
    if 'Williams_R' in features_to_add:
        df = add_williams_r(df)
    if 'Elder_Ray_Index' in features_to_add:
        df = add_elder_ray_index(df)
    if 'ROC' in features_to_add:
        df = add_roc(df)
    if 'OBV' in features_to_add:
        df = add_obv(df)
    if 'MFI' in features_to_add:
        df = add_mfi(df)
    if 'PPO' in features_to_add:
        df = add_ppo(df)
    if 'CCI' in features_to_add:
        df = add_cci(df)
    if 'Stoch_RSI' in features_to_add:
        df = add_stoch_rsi(df)
    if 'Ultimate_Oscillator' in features_to_add:
        df = add_ultimate_oscillator(df)
    if 'PSAR' in features_to_add:
        df = add_psar(df)
    if 'ADL' in features_to_add:
        df = add_adl(df)
    if 'ADX' in features_to_add:
        df = add_adx(df)
    if 'ForceIndex' in features_to_add:
        df = add_force_index(df)
    if 'KeltnerChannel' in features_to_add:
        df = add_keltner_channel(df)
    if 'CoppockCurve' in features_to_add:
        df = add_coppock_curve(df)
    if 'Ichimoku' in features_to_add:
        df = add_ichimoku(df)
    if 'ZigZag' in features_to_add:
        df = add_zigzag(df)
    if 'HMA' in features_to_add:
        df = add_hma(df)
    if 'QStick' in features_to_add:
        df = add_qstick(df)
    if 'TRIX' in features_to_add:
        df = add_trix(df)
    if 'Vortex' in features_to_add:
        df = add_vortex(df)
    df.dropna(inplace=True)

    output_filename = os.path.join(format2_path, f"{ticker}_processed_data.csv")
    df.to_csv(output_filename, index=False)
    logging.info(f"Processed and saved enriched data to {output_filename}")

def select_features(features_dict):
    selected_features = []
    print("Please select the technical indicators you want to add:")
    for key, value in features_dict.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number(s) of the indicator (comma-separated), 'All', 'Done' or 'Quit' to finish: ").strip().lower()
        if choice == 'all':
            return list(features_dict.values())[:-1]
        elif choice == 'quit' or choice == 'done' or choice == '22':
            break
        else:
            choices = [ch.strip() for ch in choice.split(',')]
            for ch in choices:
                if ch in features_dict:
                    selected_features.append(features_dict[ch])
                else:
                    print("Invalid choice, please try again.")

    return selected_features


# Section 4: Main Execution

def main():
    format_choice = input("Choose data format (1 or 2): ")
    
    if format_choice == "1":
        features_dict = {
            # Add feature definitions here
        }
        selected_features_to_add = select_features(features_dict)
        for ticker in tickers:
            process_data(ticker, selected_features_to_add)
    elif format_choice == "2":
        features_dict = {
            # Add feature definitions here
        }
        selected_features_to_add = select_features(features_dict)
        for ticker in tickers:
            process_data(ticker, selected_features_to_add)
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
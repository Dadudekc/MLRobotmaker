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

# Section 2: Technical Indicator Functions

def add_moving_average(df, window_size=10):
    df[f'SMA_{window_size}'] = df['c'].rolling(window=window_size).mean()
    return df

def add_exponential_moving_average(df, window_size=10):
    df[f'EMA_{window_size}'] = df['c'].ewm(span=window_size, adjust=False).mean()
    return df

def add_bollinger_bands(df, window_size=10):
    df['Bollinger_High'] = df['c'].rolling(window=window_size).mean() + (df['c'].rolling(window=window_size).std() * 2)
    df['Bollinger_Low'] = df['c'].rolling(window=window_size).mean() - (df['c'].rolling(window=window_size).std() * 2)
    df['Bollinger_Mid'] = df['c'].rolling(window=window_size).mean()
    return df

def add_stochastic_oscillator(df, window_size=14):
    df['Lowest'] = df['l'].rolling(window=window_size).min()
    df['Highest'] = df['h'].rolling(window=window_size).max()
    df['Stochastic'] = 100 * ((df['c'] - df['Lowest']) / (df['Highest'] - df['Lowest']))
    df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3).mean()
    return df

def add_atr(df, window_size=14):
    indicator = AverageTrueRange(df['h'], df['l'], df['c'], window_size)
    df['ATR'] = indicator.average_true_range()
    return df

def add_mae(df, window_size=10, percentage=0.025):  # 2.5% is a common default
    SMA = df['c'].rolling(window=window_size).mean()
    df['MAE_Upper'] = SMA + (SMA * percentage)
    df['MAE_Lower'] = SMA - (SMA * percentage)
    return df

def add_rvi(df):
    df['RVI'] = (df['c'] - df['o']) / (df['h'] - df['l'])
    return df

def add_pvt(df):
    df['PVT'] = ((df['c'] - df['c'].shift(1)) / df['c'].shift(1)) * df['v'] + df['v'].shift(1)
    df['PVT'] = df['PVT'].fillna(0)
    return df

def compute_adl(df):
    clv = ((df['c'] - df['l']) - (df['h'] - df['c'])) / (df['h'] - df['l'])
    clv = clv.fillna(0)  # Replace NaNs with 0
    adl = (clv * df['v']).cumsum()
    return adl

def add_chaikin_oscillator(df):
    adl = compute_adl(df)
    df['Chaikin_Oscillator'] = adl.ewm(span=3).mean() - adl.ewm(span=10).mean()
    return df

def add_dpo(df, window=20):
    displaced_ma = df['c'].shift(int(window/2) + 1).rolling(window=window).mean()
    df['DPO'] = df['c'] - displaced_ma
    return df

def add_williams_r(df, window=14):
    highest_high = df['h'].rolling(window=window).max()
    lowest_low = df['l'].rolling(window=window).min()
    df['Williams_R'] = -100 * (highest_high - df['c']) / (highest_high - lowest_low)
    return df

def add_elder_ray_index(df, window=13):
    ema = df['c'].ewm(span=window).mean()
    df['Bull_Power'] = df['h'] - ema
    df['Bear_Power'] = df['l'] - ema
    return df

def add_roc(df, window=10):
    df['ROC'] = (df['c'] - df['c'].shift(window)) / df['c'].shift(window) * 100
    return df

def add_obv(df):
    df['OBV'] = (df['v'] * (~df['c'].diff().le(0) * 2 - 1)).cumsum()
    return df

def add_mfi(df, window=14):
    typical_price = (df['h'] + df['l'] + df['c']) / 3
    money_flow = typical_price * df['v']
    positive_flow = money_flow.where(df['c'] > df['c'].shift(1), 0).rolling(window=window).sum()
    negative_flow = money_flow.where(df['c'] < df['c'].shift(1), 0).rolling(window=window).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df['MFI'] = mfi
    return df

def add_ppo(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df['c'].ewm(span=fast_period).mean()
    slow_ema = df['c'].ewm(span=slow_period).mean()
    df['PPO'] = (fast_ema - slow_ema) / slow_ema * 100
    df['PPO_Signal'] = df['PPO'].ewm(span=signal_period).mean()
    df['PPO_Histogram'] = df['PPO'] - df['PPO_Signal']
    return df

def add_cci(df, window=20):
    tp = (df['h'] + df['l'] + df['c']) / 3
    cci = (tp - tp.rolling(window).mean()) / (0.015 * tp.rolling(window).std())
    df['CCI'] = cci
    return df

def add_stoch_rsi(df, window=14):
    delta = df['c'].diff(1)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

    df['Stoch_RSI'] = stoch_rsi
    return df

def add_ultimate_oscillator(df):
    bp = df['c'] - df[['c', 'l']].min(axis=1)
    tr = df[['h', 'l']].max(axis=1) - df[['c', 'l']].min(axis=1)
    avg7 = (bp.rolling(7).sum() / tr.rolling(7).sum())
    avg14 = (bp.rolling(14).sum() / tr.rolling(14).sum())
    avg28 = (bp.rolling(28).sum() / tr.rolling(28).sum())
    ultimate = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    df['Ultimate_Oscillator'] = ultimate
    return df

def add_psar(df, step=0.02, max_step=0.2):
    indicator = PSARIndicator(df['h'], df['l'], df['c'], step=step, max_step=max_step)
    df['PSAR'] = indicator.psar()
    return df

def add_adl(df):
    clv = ((df['c'] - df['l']) - (df['h'] - df['c'])) / (df['h'] - df['l'])
    adl = (clv * df['v']).cumsum()
    df['ADL'] = adl
    return df

from ta.trend import ADXIndicator

def add_adx(df, window=14):
    adx = ADXIndicator(df['h'], df['l'], df['c'], window)
    df['ADX'] = adx.adx()
    return df

def add_force_index(df, window=13):
    fi = df['c'].diff() * df['v']
    df['ForceIndex'] = fi.rolling(window).mean()
    return df

def add_keltner_channel(df, window=20, multiplier=2):
    ma = df['c'].rolling(window=window).mean()
    atr = AverageTrueRange(df['h'], df['l'], df['c'], window).average_true_range()

    df['Keltner_Channel_High'] = ma + (multiplier * atr)
    df['Keltner_Channel_Low'] = ma - (multiplier * atr)
    df['Keltner_Channel_Mid'] = ma
    return df

def add_coppock_curve(df, window1=14, window2=11, window3=10):
    roc1 = ((df['c'] - df['c'].shift(window1)) / df['c'].shift(window1)) * 100
    roc2 = ((df['c'] - df['c'].shift(window2)) / df['c'].shift(window2)) * 100

    coppock = (roc1 + roc2).rolling(window=window3).mean()
    df['CoppockCurve'] = coppock
    return df



def add_ichimoku(df):
    ichimoku = IchimokuIndicator(df['h'], df['l'], visual=False)
    df['Ichimoku_a'] = ichimoku.ichimoku_a()
    df['Ichimoku_b'] = ichimoku.ichimoku_b()
    df['Ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    return df

def add_hma(df, window=14):
    wma_half_length = df['c'].rolling(window=int(window / 2)).mean()
    wma_full_length = df['c'].rolling(window=window).mean()
    df['HMA'] = ((2 * wma_half_length - wma_full_length).rolling(window=int(window ** 0.5)).mean()).shift(-int(window ** 0.5) + 1)
    return df

def add_qstick(df, window=8):
    df['QStick'] = (df['c'] - df['o']).rolling(window=window).mean()
    return df

def add_trix(df, window=15):
    ema1 = df['c'].ewm(span=window).mean()
    ema2 = ema1.ewm(span=window).mean()
    ema3 = ema2.ewm(span=window).mean()
    df['TRIX'] = (ema3 - ema3.shift(1)) / ema3.shift(1)
    return df

from ta.trend import VortexIndicator

def add_vortex(df, window=14):
    vortex = VortexIndicator(df['h'], df['l'], df['c'], window)
    df['Vortex_Positive'] = vortex.vortex_indicator_pos()
    df['Vortex_Negative'] = vortex.vortex_indicator_neg()
    return df


# Section 3: Data Processing and Feature Selection Functions

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

# Section 4: Main Execution

if __name__ == "__main__":
    features_dict = {
        "1": "MA",
        "2": "EMA",
        "3": "BB",
        "4": "RSI",
        "5": "MACD",
        "6": "SO",
        "7": "ATR",
        "7": "MAE",
        "8": "RVI",
        "9": "PVT",
        "10": "Chaikin_Oscillator",
        "11": "DPO",
        "12": "Williams_R",
        "14": "Elder_Ray_Index",
        "15": "ROC",
        "16": "OBV",
        "17": "MFI",
        "18": "PPO",
        "19": "CCI",
        "20": "Stoch_RSI",
        "21": "Ultimate_Oscillator",
        "22": "PSAR",
        "23": "ADL",
        "24": "ADX",
        "25": "ForceIndex",
        "26": "KeltnerChannel",
        "27": "CoppockCurve",
        "28": "Ichimoku",
        "29": "HMA",
        "30": "QStick",
        "31": "TRIX",
        "32": "Vortex",
        "33": "Quit"
    }

    selected_features_to_add = select_features(features_dict)
    for ticker in tickers:
        process_data(ticker, selected_features_to_add)


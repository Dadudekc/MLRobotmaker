#data_processing2.py

# Imports and Initial Configurations
import warnings
import json
import os
import numpy as np
import sys
import logging
import configparser
import glob
import pandas as pd
from pathlib import Path
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.utils import dropna
from ta.trend import CCIIndicator, PSARIndicator, IchimokuIndicator, ADXIndicator, VortexIndicator
import talib

warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta.trend")
warnings.simplefilter(action='ignore', category=Warning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini')
loading_path = config['Paths']['loading_path']
saving_path = config['Paths']['saving_path']
csv_files = glob.glob(os.path.join(loading_path, '*.csv'))
tickers = [os.path.basename(file).split('_')[0] for file in csv_files]  # Adjust the split logic based on your file naming convention

# Utility Functions
def get_format_path(format_number):
    project_dir = Path(__file__).resolve().parent
    config_file_path = project_dir / 'config.ini'
    
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if format_number == 1:
        return project_dir / config.get('Paths', 'format1processeddata', fallback='csv_files/format1')
    elif format_number == 2:
        return project_dir / config.get('Paths', 'format2processeddata', fallback='csv_files/format2')

def handle_nan_in_dataframe(df):
    for column in ['h', 'l', 'c', 'high', 'low', 'close']:
        if column in df.columns:
            df[column].fillna(method='ffill', inplace=True)  
            df[column].fillna(method='bfill', inplace=True) 
    return df


def detect_format(file_path):
    df = pd.read_csv(file_path)

    # Check if the file is in the first alternative format
    if "Meta Data" in df.columns:
        return 1

    # Check if the file is in the second alternative format
    elif set(["h", "l", "o", "v", "c"]).issubset(df.columns):
        return 2

    # Default format
    else:
        return 0  # Or any default format number you wish to assign


def detect_and_transform_format(file_path):
    df = pd.read_csv(file_path)

    # Check if the file is in the first alternative format
    if "Meta Data" in df.columns:
        # Extract and transform the JSON strings into a DataFrame
        data = [json.loads(row) for row in df[df.columns[-1]]]
        new_df = pd.DataFrame(data)
        new_df.columns = ['open', 'high', 'low', 'close', 'volume']
        new_df['date'] = df[df.columns[2]].values  # Assuming date is in the 3rd column

    # Check if the file is in the second alternative format
    elif "h" in df.columns and "l" in df.columns and "o" in df.columns and "v" in df.columns and "c" in df.columns:
        new_df = df.rename(columns={"h": "high", "l": "low", "o": "open", "v": "volume", "c": "close"})
        new_df['date'] = df.index  # Assuming the index is the date

    # If the file is already in the desired format
    else:
        new_df = df  # No transformation needed

    # Reorder the columns to the desired format
    new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']]

    # Save the transformed DataFrame to a new CSV
    new_file_path = Path(file_path).with_name("transformed_" + Path(file_path).name)
    new_df.to_csv(new_file_path, index=False)

    return new_file_path


# Technical Indicator Functions (Add only the unique ones from both scripts)
#Section 1: Moving Averages

def add_moving_average(df, window_size=10):
    df[f'SMA_{window_size}'] = df['close'].rolling(window=window_size).mean()
    return df

def add_exponential_moving_average(df, window_size=10):
    df[f'EMA_{window_size}'] = df['close'].ewm(span=window_size, adjust=False).mean()
    return df

#Section 2: Bollinger band

def add_bollinger_bands(df, window_size=10):
    df['Bollinger_High'] = df['close'].rolling(window=window_size).mean() + (df['close'].rolling(window=window_size).std() * 2)
    df['Bollinger_Low'] = df['close'].rolling(window=window_size).mean() - (df['close'].rolling(window=window_size).std() * 2)
    df['Bollinger_Mid'] = df['close'].rolling(window=window_size).mean()
    return df

#Section 3: Ocsillators

def add_relative_strength_index(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi.fillna(0)
    return df

def add_stochastic_oscillator(df, window_size=14):
    df['Lowest'] = df['low'].rolling(window=window_size).min()
    df['Highest'] = df['high'].rolling(window=window_size).max()
    df['Stochastic'] = 100 * ((df['close'] - df['Lowest']) / (df['Highest'] - df['Lowest']))
    df['Stochastic_Signal'] = df['Stochastic'].rolling(window=3).mean()
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = fast_ema - slow_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_commodity_channel_index(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
    df['CCI'] = cci
    return df

def add_williams_r(df, window=14):
    highest_high = df['high'].rolling(window=window).max()
    lowest_low = df['low'].rolling(window=window).min()
    df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    return df

def add_rate_of_change(df, window=10):
    df['ROC'] = ((df['close'] - df['close'].shift(window)) / df['close'].shift(window)) * 100
    return df

def add_money_flow_index(df, window=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window).sum()
    negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df['MFI'] = mfi
    return df

#Section 4: Volatility

def add_average_true_range(df, window_size=14):
    indicator = AverageTrueRange(df['high'], df['low'], df['close'], window_size)
    df['ATR'] = indicator.average_true_range()
    return df

def add_keltner_channel(df, window=20, multiplier=2):
    ma = df['close'].rolling(window=window).mean()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window).average_true_range()
    df['Keltner_Channel_High'] = ma + (multiplier * atr)
    df['Keltner_Channel_Low'] = ma - (multiplier * atr)
    df['Keltner_Channel_Mid'] = ma
    return df

def add_standard_deviation(df, window_size=20):
    df['Standard_Deviation'] = df['close'].rolling(window=window_size).std()
    return df

def add_historical_volatility(df, window_size=20):
    log_return = np.log(df['close'] / df['close'].shift(1))
    df['Historical_Volatility'] = log_return.rolling(window=window_size).std() * np.sqrt(window_size)
    return df

def add_chandelier_exit(df, window=22, multiplier=3):
    highest_high = df['high'].rolling(window=window).max()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window).average_true_range()
    df['Chandelier_Exit_Long'] = highest_high - atr * multiplier
    return df

#Section 5: Trend Indicators

def add_moving_average_envelope(df, window_size=10, percentage=0.025):
    SMA = df['close'].rolling(window=window_size).mean()
    df['MAE_Upper'] = SMA + (SMA * percentage)
    df['MAE_Lower'] = SMA - (SMA * percentage)
    return df

def add_adx(df, window=14):
    adx = ADXIndicator(df['high'], df['low'], df['close'], window)
    df['ADX'] = adx.adx()
    return df

def add_ichimoku_cloud(df):
    nine_period_high = df['high'].rolling(window=9).max()
    nine_period_low = df['low'].rolling(window=9).min()
    df['Ichimoku_Conversion_Line'] = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = df['high'].rolling(window=26).max()
    twenty_six_period_low = df['low'].rolling(window=26).min()
    df['Ichimoku_Base_Line'] = (twenty_six_period_high + twenty_six_period_low) / 2

    df['Ichimoku_Leading_Span_A'] = ((df['Ichimoku_Conversion_Line'] + df['Ichimoku_Base_Line']) / 2).shift(26)

    fifty_two_period_high = df['high'].rolling(window=52).max()
    fifty_two_period_low = df['low'].rolling(window=52).min()
    df['Ichimoku_Leading_Span_B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    df['Ichimoku_Lagging_Span'] = df['close'].shift(-26)
    return df

def add_parabolic_sar(df, step=0.02, max_step=0.2):
    indicator = PSARIndicator(df['high'], df['low'], df['close'], step=step, max_step=max_step)
    df['PSAR'] = indicator.psar()
    return df

def calculate_historical_volatility(df, window):
    log_return = np.log(df['close'] / df['close'].shift(1))
    return log_return.rolling(window=window).std() * np.sqrt(window)

def determine_threshold(df, dynamic, fixed_threshold, vol_window):
    if dynamic:
        volatility = calculate_historical_volatility(df, window=vol_window)
        return volatility * fixed_threshold
    else:
        return fixed_threshold

def confirm_peak_trough(df, index, lookback):
    if index < lookback or index > len(df) - lookback:
        return False
    for i in range(1, lookback + 1):
        if df['close'][index - i] > df['close'][index] or df['close'][index + i] > df['close'][index]:
            return False
    return True

def add_zigzag_indicator(df, dynamic_threshold, fixed_threshold, lookback, vol_window):
    df['ZigZag'] = np.nan
    threshold = determine_threshold(df, dynamic=dynamic_threshold, fixed_threshold=fixed_threshold, vol_window=vol_window)
    
    last_peak = last_trough = df['close'][0]
    for i in range(1, len(df)):
        if abs(df['close'][i] - last_peak) > threshold[i] and df['close'][i] < last_peak:
            if confirm_peak_trough(df, i, lookback):
                df['ZigZag'][i] = last_peak = df['close'][i]
        elif abs(df['close'][i] - last_trough) > threshold[i] and df['close'][i] > last_trough:
            if confirm_peak_trough(df, i, lookback):
                df['ZigZag'][i] = last_trough = df['close'][i]

    return df

#Section 6: Volume Indicators

def add_on_balance_volume(df):
    df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
    return df

def add_vwap(df):
    df['Cumulative_Volume_Price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
    df['Cumulative_Volume'] = df['volume'].cumsum()
    df['VWAP'] = df['Cumulative_Volume_Price'] / df['Cumulative_Volume']
    return df

def add_accumulation_distribution_line(df):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv.fillna(0, inplace=True)
    df['ADL'] = (clv * df['volume']).cumsum()
    return df

def add_chaikin_money_flow(df, window=20):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv.fillna(0, inplace=True)
    money_flow_volume = clv * df['volume']
    df['CMF'] = money_flow_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    return df

def add_volume_oscillator(df, short_window=12, long_window=26):
    short_vol = df['volume'].rolling(window=short_window).mean()
    long_vol = df['volume'].rolling(window=long_window).mean()
    df['Volume_Oscillator'] = short_vol - long_vol
    return df

#Section 7: Other Indicators

def add_awesome_oscillator(df, short_window=5, long_window=34):
    mid_price = (df['high'] + df['low']) / 2
    df['Awesome_Oscillator'] = mid_price.rolling(window=short_window).mean() - mid_price.rolling(window=long_window).mean()
    return df

def add_trix(df, window=15):
    ema1 = df['close'].ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    df['TRIX'] = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    return df

def add_standard_pivot_points(df):
    df['Pivot_Point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['Resistance1'] = 2 * df['Pivot_Point'] - df['low'].shift(1)
    df['Support1'] = 2 * df['Pivot_Point'] - df['high'].shift(1)
    # Add Resistance2, Support2, Resistance3, Support3 as needed
    return df

def add_elders_force_index(df, window=13):
    df['Force_Index'] = df['close'].diff(1) * df['volume']
    df['Force_Index_EMA'] = df['Force_Index'].ewm(span=window, adjust=False).mean()
    return df

def add_hull_moving_average(df, window=9):
    wma_half_length = df['close'].rolling(window=int(window / 2)).mean()
    wma_full_length = df['close'].rolling(window=window).mean()
    df['HMA'] = (2 * wma_half_length - wma_full_length).rolling(window=int(np.sqrt(window))).mean()
    return df

def add_detrended_price_oscillator(df, window=20):
    displaced_ma = df['close'].shift(int(window / 2 + 1)).rolling(window=window).mean()
    df['DPO'] = df['close'] - displaced_ma
    return df

#even more
def add_gann_high_low_activator(df, window=14):
    df['Gann_High'] = df['high'].rolling(window=window).max()
    df['Gann_Low'] = df['low'].rolling(window=window).min()
    df['Gann_High_Low_Activator'] = (df['Gann_High'] + df['Gann_Low']) / 2
    return df

def add_fisher_transform(df, window=9):
    value = 2 * ((df['close'] - df['low'].rolling(window=window).min()) / 
                 (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()) - 1)
    df['Fisher'] = 0.5 * np.log((1 + value) / (1 - value)).rolling(window=window).mean()
    return df

def add_coppock_curve(df, w1=10, w2=14, w3=11):
    roc1 = ((df['close'] - df['close'].shift(w1)) / df['close'].shift(w1)) * 100
    roc2 = ((df['close'] - df['close'].shift(w2)) / df['close'].shift(w2)) * 100
    df['Coppock_Curve'] = (roc1 + roc2).rolling(window=w3).mean()
    return df

def add_ultimate_oscillator(df):
    bp = df['close'] - df[['close', 'low']].min(axis=1)
    tr = df[['high', 'low']].max(axis=1) - df[['close', 'low']].min(axis=1)
    avg7 = (bp.rolling(7).sum() / tr.rolling(7).sum())
    avg14 = (bp.rolling(14).sum() / tr.rolling(14).sum())
    avg28 = (bp.rolling(28).sum() / tr.rolling(28).sum())
    df['Ultimate_Oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    return df

def add_supertrend(df, atr_multiplier=3, atr_window=7):
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_window).average_true_range()
    upper_band = (df['high'] + df['low']) / 2 + (atr_multiplier * atr)
    lower_band = (df['high'] + df['low']) / 2 - (atr_multiplier * atr)
    # Further logic to calculate SuperTrend...
    return df

def add_fibonacci_retracement_levels(df):
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    df['Fib_23.6%'] = max_price - diff * 0.236
    df['Fib_38.2%'] = max_price - diff * 0.382
    df['Fib_50%'] = max_price - diff * 0.5
    df['Fib_61.8%'] = max_price - diff * 0.618
    df['Fib_78.6%'] = max_price - diff * 0.786
    return df

def add_macd_histogram(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = fast_ema - slow_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_woodie_pivot_points(df):
    df['Pivot_Point_Woodie'] = (df['high'].shift(1) + df['low'].shift(1) + 2 * df['close'].shift(1)) / 4
    df['Woodie_R1'] = 2 * df['Pivot_Point_Woodie'] - df['low'].shift(1)
    df['Woodie_R2'] = df['Pivot_Point_Woodie'] + df['high'].shift(1) - df['low'].shift(1)
    df['Woodie_S1'] = 2 * df['Pivot_Point_Woodie'] - df['high'].shift(1)
    df['Woodie_S2'] = df['Pivot_Point_Woodie'] - df['high'].shift(1) + df['low'].shift(1)
    return df

def add_camarilla_pivot_points(df):
    df['Pivot_Point_Camarilla'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['Camarilla_R1'] = df['close'].shift(1) + 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 12
    df['Camarilla_R2'] = df['close'].shift(1) + 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 6
    df['Camarilla_R3'] = df['close'].shift(1) + 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 4
    df['Camarilla_R4'] = df['close'].shift(1) + 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 2
    df['Camarilla_S1'] = df['close'].shift(1) - 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 12
    df['Camarilla_S2'] = df['close'].shift(1) - 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 6
    df['Camarilla_S3'] = df['close'].shift(1) - 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 4
    df['Camarilla_S4'] = df['close'].shift(1) - 1.1 * (df['high'].shift(1) - df['low'].shift(1)) / 2
    return df

# Process Data Function
def process_data(ticker, features_to_add):
    try:
        logger.info(f"Processing data for ticker: {ticker}")
        original_file_path = os.path.join(loading_path, f"{ticker}_data.csv")

        # Skip processing if the file doesn't exist or is not correctly named
        if not os.path.exists(original_file_path) or not ticker.isalpha():
            logger.warning(f"Skipping {ticker}: File does not exist or incorrect ticker format")
            return

        # Detect the format dynamically and transform if necessary
        transformed_file_path = detect_and_transform_format(original_file_path)

        # Check if the transformed file exists
        file_path = transformed_file_path if os.path.exists(transformed_file_path) else original_file_path

        df = pd.read_csv(file_path)
        df = handle_nan_in_dataframe(df)

        # Apply selected features/technical indicators
        for feature in features_to_add:
            if feature == "Simple Moving Average (SMA)":
                df = add_moving_average(df)
            elif feature == "Exponential Moving Average (EMA)":
                df = add_exponential_moving_average(df)
            elif feature == "Bollinger Bands":
                df = add_bollinger_bands(df)
            elif feature == "Relative Strength Index (RSI)":
                df = add_relative_strength_index(df)
            elif feature == "Stochastic Oscillator":
                df = add_stochastic_oscillator(df)
            elif feature == "MACD":
                df = add_macd(df)
            elif feature == "Commodity Channel Index (CCI)":
                df = add_commodity_channel_index(df)
            elif feature == "Williams %R":
                df = add_williams_r(df)
            elif feature == "Rate of Change (ROC)":
                df = add_rate_of_change(df)
            elif feature == "Money Flow Index (MFI)":
                df = add_money_flow_index(df)
            elif feature == "Average True Range (ATR)":
                df = add_average_true_range(df)
            elif feature == "Keltner Channel":
                df = add_keltner_channel(df)
            elif feature == "Standard Deviation":
                df = add_standard_deviation(df)
            elif feature == "Historical Volatility":
                df = add_historical_volatility(df)
            elif feature == "Chandelier Exit":
                df = add_chandelier_exit(df)
            elif feature == "Moving Average Envelope (MAE)":
                df = add_moving_average_envelope(df)
            elif feature == "Average Directional Index (ADX)":
                df = add_adx(df)
            elif feature == "Ichimoku Cloud":
                df = add_ichimoku_cloud(df)
            elif feature == "Parabolic SAR":
                df = add_parabolic_sar(df)
            elif feature == "ZigZag Indicator":
                df = add_zigzag_indicator(df, dynamic_threshold, fixed_threshold, lookback, vol_window)
            elif feature == "On-Balance Volume (OBV)":
                df = add_on_balance_volume(df)
            elif feature == "Volume Weighted Average Price (VWAP)":
                df = add_vwap(df)
            elif feature == "Accumulation/Distribution Line (ADL)":
                df = add_accumulation_distribution_line(df)
            elif feature == "Chaikin Money Flow (CMF)":
                df = add_chaikin_money_flow(df)
            elif feature == "Volume Oscillator":
                df = add_volume_oscillator(df)
            elif feature == "Awesome Oscillator":
                df = add_awesome_oscillator(df)
            elif feature == "TRIX":
                df = add_trix(df)
            elif feature == "Standard Pivot Points":
                df = add_standard_pivot_points(df)
            elif feature == "Elders Force Index":
                df = add_elders_force_index(df)
            elif feature == "Hull Moving Average (HMA)":
                df = add_hull_moving_average(df)
            elif feature == "Detrended Price Oscillator (DPO)":
                df = add_detrended_price_oscillator(df)
            elif feature == "Gann High Low Activator":
                df = add_gann_high_low_activator(df)
            elif feature == "Fisher Transform":
                df = add_fisher_transform(df)
            elif feature == "Coppock Curve":
                df = add_coppock_curve(df)
            elif feature == "Ultimate Oscillator":
                df = add_ultimate_oscillator(df)
            elif feature == "SuperTrend":
                df = add_supertrend(df)
            elif feature == "Fibonacci Retracement Levels":
                df = add_fibonacci_retracement_levels(df)
            elif feature == "MACD Histogram":
                df = add_macd_histogram(df)
            elif feature == "Woodie Pivot Points":
                df = add_woodie_pivot_points(df)
            elif feature == "Camarilla Pivot Points":
                df = add_camarilla_pivot_points(df)

        output_path = saving_path
        output_filename = os.path.join(output_path, f"{ticker}_processed_data.csv")
        
        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        df.to_csv(output_filename, index=False)
        logger.info(f"Processed and saved enriched data to {output_filename}")

    except Exception as e:
        logger.error(f"Error while processing {ticker}. Error: {e}", exc_info=True)



def add_relative_volatility_index(df, window=14):
    std_dev = df['close'].diff().rolling(window=window).std()
    df['RVI'] = std_dev / std_dev.rolling(window=window).mean() * 100
    return df

# Feature Selection Function (Use either of the script's logic)
def select_features():
    print("The default option is to select all features.")
    override_default = input("Would you like to make a custom selection? (yes/no): ").strip().lower()

    if override_default != 'yes':
        # Return all features except the "Quit" option
        return [value for key, value in features_dict.items() if key != "41"]  # Assuming "41" is the key for "Quit"

    selected_features = []
    print("Please select the technical indicators you want to add:")
    
    for key, value in features_dict.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number(s) of the indicator (comma-separated) or 'Quit' to finish: ").strip().lower()
        
        if choice in ['quit', '41']: 
            break
        else:
            choices = [ch.strip() for ch in choice.split(',')]
            for ch in choices:
                if ch in features_dict and ch != "41":  # Exclude "Quit"
                    selected_features.append(features_dict[ch])
                else:
                    print(f"Invalid choice '{ch}', please try again.")

    return selected_features

# Main Execution Block
if __name__ == "__main__":
    features_dict = {
        "1": "Simple Moving Average (SMA)",
        "2": "Exponential Moving Average (EMA)",
        "3": "Bollinger Bands",
        "4": "Relative Strength Index (RSI)",
        "5": "Stochastic Oscillator",
        "6": "MACD",
        "7": "Commodity Channel Index (CCI)",
        "8": "Williams %R",
        "9": "Rate of Change (ROC)",
        "10": "Money Flow Index (MFI)",
        "11": "Average True Range (ATR)",
        "12": "Keltner Channel",
        "13": "Standard Deviation",
        "14": "Historical Volatility",
        "15": "Chandelier Exit",
        "16": "Moving Average Envelope (MAE)",
        "17": "Average Directional Index (ADX)",
        "18": "Ichimoku Cloud",
        "19": "Parabolic SAR",
        "20": "ZigZag Indicator",
        "21": "On-Balance Volume (OBV)",
        "22": "Volume Weighted Average Price (VWAP)",
        "23": "Accumulation/Distribution Line (ADL)",
        "24": "Chaikin Money Flow (CMF)",
        "25": "Volume Oscillator",
        "26": "Awesome Oscillator",
        "27": "TRIX",
        "28": "Standard Pivot Points",
        "29": "Elders Force Index",
        "30": "Hull Moving Average (HMA)",
        "31": "Detrended Price Oscillator (DPO)",
        "32": "Gann High Low Activator",
        "33": "Fisher Transform",
        "34": "Coppock Curve",
        "35": "Ultimate Oscillator",
        "36": "SuperTrend",
        "37": "Fibonacci Retracement Levels",
        "38": "MACD Histogram",
        "39": "woodie Pivot Points",
        "40": "Camarilla Pivot Points",
        "41": "Quit"

    }

def select_features(features_dict):
    selected_features = []
    print("Please select the technical indicators you want to add:")
    
    for key, value in features_dict.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number(s) of the indicator (comma-separated), 'All', 'Done' or 'Quit' to finish: ").strip().lower()
        
        if choice == 'all':
            return list(features_dict.values())[:-1]
        elif choice in ['quit', 'done', '33']: 
            break
        else:
            choices = [ch.strip() for ch in choice.split(',')]
            for ch in choices:
                if ch in features_dict:
                    selected_features.append(features_dict[ch])
                else:
                    print("Invalid choice, please try again.")

    return selected_features

# Main Execution Block
if __name__ == "__main__":
    # Default values
    dynamic_threshold_default = True
    fixed_threshold_default = 0.05  # Example default value
    lookback_default = 14  # Example default value
    vol_window_default = 21  # Example default value

    # Ask user if they want to override defaults
    override_defaults = input("Override default settings? (yes/no): ").lower() == 'yes'

    if override_defaults:
        dynamic_threshold = input("Use dynamic threshold? (yes/no): ").lower() == 'yes'
        fixed_threshold = float(input("Enter fixed threshold percentage (e.g., 0.05 for 5%): "))
        lookback = int(input("Enter lookback period for confirmation: "))
        vol_window = int(input("Enter volatility window (for dynamic threshold): "))
    else:
        # Use default values
        dynamic_threshold = dynamic_threshold_default
        fixed_threshold = fixed_threshold_default
        lookback = lookback_default
        vol_window = vol_window_default

    # Allow the user to select the desired technical indicators
    selected_features_to_add = select_features(features_dict)

    # Process data for each ticker using the selected indicators
    for ticker in tickers:
        process_data(ticker, selected_features_to_add)

# data_processing2

import pandas as pd
import os
import logging
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define format2 folder path
format2_path = "C:\\Users\Dagurlkc\\OneDrive\\Desktop\\DaDudeKC\\MyAIRobot\\csv_files\\format2"

# Function to add features and technical indicators
def add_features_and_indicators(df):
    # Simple Moving Average (SMA) for 10 days
    sma_10 = SMAIndicator(df["close"], window=10)
    df["SMA_10"] = sma_10.sma_indicator()
    
    # Exponential Moving Average (EMA) for 10 days
    ema_10 = EMAIndicator(df["close"], window=10)
    df["EMA_10"] = ema_10.ema_indicator()
    
    # Price Rate of Change (RoC)
    roc = ROCIndicator(df["close"], window=10)
    df["Price_RoC"] = roc.roc()
    
    # Relative Strength Index (RSI)
    rsi = RSIIndicator(df["close"], window=10)
    df["RSI"] = rsi.rsi()
    
    # Moving Average Convergence Divergence (MACD)
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    
    # Bollinger Bands
    bollinger = BollingerBands(df["close"])
    df["Bollinger_High"] = bollinger.bollinger_hband()
    df["Bollinger_Low"] = bollinger.bollinger_lband()
    df["Bollinger_Mid"] = bollinger.bollinger_mavg()
    
    return df

# Function to process a single CSV file
def process_csv_file(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add features and indicators
        df = add_features_and_indicators(df)
        
        # Save the enriched DataFrame back to the same CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Processed and saved enriched data to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while processing {file_path}: {e}")

# Main function
def main():
    for filename in os.listdir(format2_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(format2_path, filename)
            process_csv_file(file_path)

if __name__ == "__main__":
    main()

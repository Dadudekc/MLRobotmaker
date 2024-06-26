#data_processing.py
import os
import sys
import pandas as pd
from Data_processing import data_io 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utilities import config_manager
from Data_processing import technical_indicators as ti
import logging
from Utilities.Utils import log_message


# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data(ticker):
    try:
        # Initialize Configuration Manager
        config = config_manager.ConfigManager()

        # Load paths from configuration
        loading_path = config.get_path('Paths', 'loading_path', fallback='default/loading/path')
        saving_path = config.get_path('Paths', 'saving_path', fallback='default/saving/path')

        # File path for the ticker
        file_path = os.path.join(loading_path, f"{ticker}_data.csv")

        # Load and transform data
        df = data_io.load_transformed_data(file_path)

        # Apply Technical Indicators
        df = ti.TechnicalIndicators.add_moving_average(df)
        df = ti.TechnicalIndicators.add_bollinger_bands(df)
        df = ti.TechnicalIndicators.add_exponential_moving_average(df)
        df = ti.TechnicalIndicators.add_stochastic_oscillator(df)
        df = ti.TechnicalIndicators.calculate_macd_components(df)
        df = ti.TechnicalIndicators.add_average_true_range(df)
        df = ti.TechnicalIndicators.add_relative_strength_index(df)
        df = ti.TechnicalIndicators.add_commodity_channel_index(df)
        df = ti.TechnicalIndicators.add_williams_r(df)
        df = ti.TechnicalIndicators.add_rate_of_change(df)
        df = ti.TechnicalIndicators.add_money_flow_index(df)
        df = ti.TechnicalIndicators.add_standard_deviation(df)
        df = ti.TechnicalIndicators.add_historical_volatility(df)
        df = ti.TechnicalIndicators.add_chandelier_exit(df)
        df = ti.TechnicalIndicators.add_keltner_channel(df)
        df = ti.TechnicalIndicators.add_moving_average_envelope(df)
        df = ti.TechnicalIndicators.add_adx(df)
        df = ti.TechnicalIndicators.add_ichimoku_cloud(df)
        df = ti.TechnicalIndicators.add_parabolic_sar(df)
        df = ti.TechnicalIndicators.add_fisher_transform(df)
        df = ti.TechnicalIndicators.add_coppock_curve(df)
        df = ti.TechnicalIndicators.add_ultimate_oscillator(df)
        df = ti.TechnicalIndicators.add_supertrend(df)  # Ensure implementation
        df = ti.TechnicalIndicators.add_fibonacci_retracement_levels(df)
        df = ti.TechnicalIndicators.add_macd_histogram(df)
        df = ti.TechnicalIndicators.add_woodie_pivot_points(df)
        df = ti.TechnicalIndicators.add_camarilla_pivot_points(df)
        df = ti.TechnicalIndicators.add_detrended_price_oscillator(df)
        df = ti.TechnicalIndicators.add_gann_high_low_activator(df)
        df = ti.TechnicalIndicators.add_elders_force_index(df)
        df = ti.TechnicalIndicators.add_hull_moving_average(df)
        df = ti.TechnicalIndicators.add_on_balance_volume(df)
        df = ti.TechnicalIndicators.add_vwap(df)
        df = ti.TechnicalIndicators.add_accumulation_distribution_line(df)
        df = ti.TechnicalIndicators.add_chaikin_money_flow(df)
        df = ti.TechnicalIndicators.add_volume_oscillator(df)
        df = ti.TechnicalIndicators.add_awesome_oscillator(df)
        df = ti.TechnicalIndicators.add_trix(df)
        df = ti.TechnicalIndicators.add_standard_pivot_points(df)

        # Generate the new file name based on the original file name
        base_name = os.path.basename(file_path)
        new_file_name = base_name.replace(f"{ticker}_data", f"{ticker}_processed_data")

        # Determine the full path for the new file
        processed_file_path = os.path.join(saving_path, new_file_name)

        
        # Save the processed data using the new file path
        data_io.save_transformed_data(df, processed_file_path)


        logger.info(f"Data processing completed for ticker: {ticker}. Saved as {new_file_name}")

    except Exception as e:
        logger.error(f"Error processing data for ticker {ticker}: {e}", exc_info=True)

def main():
    # Load tickers from a file or a list
    tickers = []

    for ticker in tickers:
        process_data(ticker)

if __name__ == "__main__":
    main()

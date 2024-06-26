#model_management.py

#Part 1: Imports and DataLoader Class

import os
import glob
import pandas as pd
import configparser
import json

class DataLoader:
    def __init__(self, config):
        self.loading_path = config['Paths']['loading_path']

    def load_data(self, file_path):
        # Check if the provided path is absolute. If not, construct the full path.
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.loading_path, file_path)
        
        # Check if the file exists
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Data file not found at {file_path}.")

    def get_tickers(self):
        csv_files = glob.glob(os.path.join(self.loading_path, '*.csv'))
        return [os.path.basename(file).split('_')[0] for file in csv_files]

#Part 2: DataTransformer Class

class DataTransformer:
    @staticmethod
    def transform_format(df):
        if "Meta Data" in df.columns:
            data = [json.loads(row) for row in df[df.columns[-1]]]
            new_df = pd.DataFrame(data)
            new_df.columns = ['open', 'high', 'low', 'close', 'volume']
            new_df['date'] = df[df.columns[2]].values

        elif set(["h", "l", "o", "v", "c"]).issubset(df.columns):
            new_df = df.rename(columns={"h": "high", "l": "low", "o": "open", "v": "volume", "c": "close"})
            new_df['date'] = df.index

        else:
            new_df = df

        new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return new_df

#Part 3: FeatureEngineering Class

import logging

class FeatureEngineering:
    def __init__(self, features_to_add):
        self.features_to_add = features_to_add
        # Dictionary mapping features to their respective functions
        self.feature_functions = {
            "Simple Moving Average (SMA)": lambda df: TechnicalIndicators.add_moving_average(df, window_size=10),
            "Exponential Moving Average (EMA)": lambda df: TechnicalIndicators.add_exponential_moving_average(df, window_size=10),
            "Bollinger Bands": lambda df: TechnicalIndicators.add_bollinger_bands(df, window_size=20, std_multiplier=2),
            "Stochastic Oscillator": lambda df: TechnicalIndicators.add_stochastic_oscillator(df, window_size=14),
            "MACD": lambda df: TechnicalIndicators.calculate_macd_components(df),
            "Average True Range (ATR)": lambda df: TechnicalIndicators.add_average_true_range(df, window_size=14),
            "Relative Strength Index (RSI)": lambda df: TechnicalIndicators.add_relative_strength_index(df, window=14),
            "Williams %R": lambda df: TechnicalIndicators.add_williams_r(df, window=14),
            "Rate of Change": lambda df: TechnicalIndicators.add_rate_of_change(df, window=10),
            "Money Flow Index": lambda df: TechnicalIndicators.add_money_flow_index(df, window=14),
            "Keltner Channel": lambda df: TechnicalIndicators.add_keltner_channel(df, window=20, multiplier=2),
            "Standard Deviation": lambda df: TechnicalIndicators.add_standard_deviation(df, window_size=20),
            "Historical Volatility": lambda df: TechnicalIndicators.add_historical_volatility(df, window_size=20),
            "Chandelier Exit": lambda df: TechnicalIndicators.add_chandelier_exit(df, window=22, multiplier=3),
            "Moving Average Envelope": lambda df: TechnicalIndicators.add_moving_average_envelope(df, window_size=10, percentage=0.025),
            "ADX": lambda df: TechnicalIndicators.add_adx(df, window=14),
            "Ichimoku Cloud": lambda df: TechnicalIndicators.add_ichimoku_cloud(df),
            "Parabolic SAR": lambda df: TechnicalIndicators.add_parabolic_sar(df),
            "ZigZag Indicator": lambda df: TechnicalIndicators.add_zigzag_indicator(df),
            "On-Balance Volume": lambda df: TechnicalIndicators.add_on_balance_volume(df),
            "VWAP": lambda df: TechnicalIndicators.add_vwap(df),
            "Accumulation/Distribution Line": lambda df: TechnicalIndicators.add_accumulation_distribution_line(df),
            "Chaikin Money Flow": lambda df: TechnicalIndicators.add_chaikin_money_flow(df),
            "Volume Oscillator": lambda df: TechnicalIndicators.add_volume_oscillator(df),
            "Awesome Oscillator": lambda df: TechnicalIndicators.add_awesome_oscillator(df),
            "TRIX": lambda df: TechnicalIndicators.add_trix(df),
            "Standard Pivot Points": lambda df: TechnicalIndicators.add_standard_pivot_points(df),
            "Elders Force Index": lambda df: TechnicalIndicators.add_elders_force_index(df),
            "Hull Moving Average": lambda df: TechnicalIndicators.add_hull_moving_average(df),
            "Detrended Price Oscillator": lambda df: TechnicalIndicators.add_detrended_price_oscillator(df)
            # Add more mappings as needed
        }


    def apply_features(self, df):
        for feature in self.features_to_add:
            function = self.feature_functions.get(feature)
            if function:
                df = function(df)
            else:
                logging.warning(f"Feature '{feature}' is not recognized.")
        return df

#Part 4: ModelManager Class

class ModelManager:
    def __init__(self, config, features_dict):
        self.data_loader = DataLoader(config)
        self.data_transformer = DataTransformer()
        self.features_dict = features_dict

    def process_ticker_data(self, ticker, selected_features):
        try:
            df = self.data_loader.load_data(ticker)
            df = self.data_transformer.transform_format(df)
            feature_engineer = FeatureEngineering(selected_features)
            df = feature_engineer.apply_features(df)
            self.save_processed_data(df, ticker)
        except Exception as e:
            logging.error(f"Error processing data for {ticker}: {e}")

    def select_features(self):
        features_dict = {
            "1": "Simple Moving Average (SMA)",
            "2": "Exponential Moving Average (EMA)",
            "3": "Bollinger Bands",
            "4": "Stochastic Oscillator",
            "5": "MACD",
            "6": "Average True Range (ATR)",
            "7": "Relative Strength Index (RSI)",
            "8": "Williams %R",
            "9": "Rate of Change",
            "10": "Money Flow Index",
            "11": "Keltner Channel",
            "12": "Standard Deviation",
            "13": "Historical Volatility",
            "14": "Chandelier Exit",
            "15": "Moving Average Envelope",
            "16": "ADX",
            "17": "Ichimoku Cloud",
            "18": "Parabolic SAR",
            "19": "ZigZag Indicator",
            "20": "On-Balance Volume",
            "21": "VWAP",
            "22": "Accumulation/Distribution Line",
            "23": "Chaikin Money Flow",
            "24": "Volume Oscillator",
            "25": "Awesome Oscillator",
            "26": "TRIX",
            "27": "Standard Pivot Points",
            "28": "Elders Force Index",
            "29": "Hull Moving Average",
            "30": "Detrended Price Oscillator",
            "quit": "Quit Selection"
        }

        print("By default, all features are selected.")
        override_default = input("Would you like to customize the feature selection? (yes/no): ").strip().lower()

        # If the user does not want to customize, return all features
        if override_default != 'yes':
            return list(features_dict.values())

        print("Please enter the numbers of the features you wish to apply (e.g., 1,3,5). Type 'quit' to finish:")
        for key, value in features_dict.items():
            print(f"{key}: {value}")

        selected_features = []
        while True:
            choices = input("Enter feature numbers (comma-separated): ").strip().lower()
            if choices == 'quit':
                break
            for choice in choices.split(','):
                feature = features_dict.get(choice.strip())
                if feature and feature not in selected_features:
                    selected_features.append(feature)
                elif not feature:
                    logging.warning(f"Invalid selection: {choice}")

        return selected_features

    def save_processed_data(self, df, ticker):
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        output_file = os.path.join(self.saving_path, f"{ticker}_processed.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Processed data saved for {ticker} in {output_file}")

    def run(self):
        selected_features = self.select_features()
        for ticker in self.data_loader.get_tickers():
            self.process_ticker_data(ticker, selected_features)

#Part 5: Main Execution

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('config.ini')
    features_dict = ()  # Define the features dictionary

    model_manager = ModelManager(config, features_dict)
    model_manager.run()



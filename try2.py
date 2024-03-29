#data_processing_tab.py
#intergrating share data store into data_processing_tab

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import configparser
from Data_processing.technical_indicators import TechnicalIndicators
from Utilities.utils import MLRobotUtils
import os
from Utilities.shared_data_store import SharedDataStore, Observer

class DataProcessingTab:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.is_debug_mode = False  # Initialize debug mode to False
        self.shared_data_store = SharedDataStore()
        # Initialize MLRobotUtils with debug mode state
        self.utils = MLRobotUtils(is_debug_mode=self.is_debug_mode)

        # Initialize GUI components
        self.setup_gui()

    def setup_gui(self):
        # Frame for CSV File Selection
        file_frame = tk.Frame(self.parent)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(file_frame, text="Select CSV File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_entry = tk.Entry(file_frame, width=50)
        self.file_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.browse_button = tk.Button(file_frame, text="Browse", command=lambda: self.utils.browse_file(self.file_path_entry))
        self.browse_button.pack(side=tk.LEFT)

        # Frame for Features/Indicators
        features_frame = tk.Frame(self.parent)
        features_frame.pack(padx=10, pady=(0, 5))

        tk.Label(features_frame, text="Select Features/Indicators:").pack(anchor=tk.W)
        self.features_listbox = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, height=15)
        self.features_listbox.pack(side=tk.LEFT, padx=(0, 10))

        # Populate the listbox with features/indicators
        features = [
            "Simple Moving Average (SMA)",
            "Exponential Moving Average (EMA)",
            "Bollinger Bands",
            "Stochastic Oscillator",
            "MACD",
            "Average True Range (ATR)",
            "Relative Strength Index (RSI)",
            "Commodity Channel Index (CCI)",
            "Williams %R",
            "Rate of Change (ROC)",
            "Money Flow Index (MFI)",
            "Standard Deviation",
            "Historical Volatility",
            "Chandelier Exit",
            "Keltner Channel",
            "Moving Average Envelope (MAE)",
            "Average Directional Index (ADX)",
            "Ichimoku Cloud",
            "Parabolic SAR",
            "Zigzag Indicator",
            "On-Balance Volume (OBV)",
            "Volume Weighted Average Price (VWAP)",
            "Accumulation/Distribution Line (ADL)",
            "Chaikin Money Flow (CMF)",
            "Volume Oscillator",
            "Awesome Oscillator",
            "TRIX",
            "Standard Pivot Points",
            "Fibonacci Retracements"
        ]

        for feature in features:
            self.features_listbox.insert(tk.END, feature)

        # Buttons for Select All and Unselect All
        buttons_frame = tk.Frame(features_frame)
        buttons_frame.pack(side=tk.RIGHT)
        self.select_all_button = tk.Button(buttons_frame, text="Select All", command=self.select_all_features)
        self.select_all_button.pack()
        self.unselect_all_button = tk.Button(buttons_frame, text="Unselect All", command=self.unselect_all_features)
        self.unselect_all_button.pack()

        # Debug Mode Button
        self.debug_mode_button = tk.Button(self.parent, text="Debug Mode: OFF", command=self.toggle_debug_mode)

        self.debug_mode_button.pack(pady=(5, 10))

        # Start Processing Button
        self.start_processing_button = tk.Button(self.parent, text="Start Processing", command=self.process_data)
        self.start_processing_button.pack(pady=(0, 5))
  
        if hasattr(self, 'log_message'):
            self.log_message.delete('1.0', tk.END)  # Clear existing content
        else:
            # Create the Data Processing Log frame and ScrolledText only if it does not exist
            log_frame = tk.Frame(self.parent)
            log_frame.pack(padx=10, pady=(0, 5))
            tk.Label(log_frame, text="Data Processing Log:").pack(anchor=tk.W)
            self.log_message = scrolledtext.ScrolledText(log_frame, height=10)
            self.log_message.pack(fill='both', expand=True)

    def toggle_debug_mode(self):
        # Toggle debug mode state
        self.is_debug_mode = not self.is_debug_mode

        # Update the button text to reflect the current state
        if self.is_debug_mode:
            self.debug_mode_button.config(text="Debug Mode: ON")
        else:
            self.debug_mode_button.config(text="Debug Mode: OFF")

    # Here, you can add additional actions to be performed when toggling debug mode

    def select_all_features(self):
        # Select all items in the features_listbox
        for i in range(self.features_listbox.size()):
            self.features_listbox.selection_set(i)

    def unselect_all_features(self):
        # Deselect all items in the features_listbox
        self.features_listbox.selection_clear(0, tk.END)

    def process_data(self):
        file_path = self.file_path_entry.get()
        selected_features = [self.features_listbox.get(idx) for idx in self.features_listbox.curselection()]

        if not file_path:
            messagebox.showwarning("Input Error", "Please select a CSV file.")
            self.log_message.insert(tk.END, "Input Error: Please select a CSV file.\n")
            return

        if not selected_features:
            messagebox.showwarning("Input Error", "Please select at least one feature/indicator.")
            self.log_message.insert(tk.END, "Input Error: Please select at least one feature/indicator.\n")
            return

        try:
            df = pd.read_csv(file_path)
            self.standardize_column_names(df)  # Standardize column names if necessary
            df.fillna(method='ffill', inplace=True)  # Forward fill to handle NaN values
            df.fillna(method='bfill', inplace=True)  # Backward fill as a fallback
            self.log_message.insert(tk.END, "CSV file loaded successfully.\n")

            # Apply selected features
            for feature in selected_features:
                self.apply_feature(df, feature)

            # Save the processed data
            save_path_dir = self.config.get('SAVE_PATH_SECTION', 'save_path_dir')
            save_path = self.utils.auto_generate_save_path(file_path, save_path_dir)
            df.to_csv(save_path, index=False)
            self.utils.log_message(f"Processed data saved to '{save_path}'.", self.log_message)
            self.log_message.insert(tk.END, "Data processing completed.\n")
        except Exception as e:
            self.log_message.insert(tk.END, f"Error during data processing: {str(e)}\n")
            messagebox.showerror("Processing Error", str(e))


    def apply_feature(self, df, feature_name):
        """
        Apply the given feature to the DataFrame.
        
        Args:
        df (pd.DataFrame): The DataFrame to which the feature will be applied.
        feature_name (str): The name of the feature to apply.

        Returns:
        pd.DataFrame: The DataFrame with the applied feature.
        
        Raises:
        ValueError: If the feature method does not return a DataFrame.
        """
        try:
            # Dictionary mapping feature names to their corresponding methods
            feature_methods = {
                "Simple Moving Average (SMA)": TechnicalIndicators.add_moving_average,
                "Exponential Moving Average (EMA)": TechnicalIndicators.add_exponential_moving_average,
                "Bollinger Bands": TechnicalIndicators.add_bollinger_bands,
                "Stochastic Oscillator": TechnicalIndicators.add_stochastic_oscillator,
                "MACD": TechnicalIndicators.calculate_macd_components,
                "Average True Range (ATR)": TechnicalIndicators.add_average_true_range,
                "Relative Strength Index (RSI)": TechnicalIndicators.add_relative_strength_index,
                "Commodity Channel Index (CCI)": TechnicalIndicators.add_commodity_channel_index,
                "Williams %R": TechnicalIndicators.add_williams_r,
                "Rate of Change (ROC)": TechnicalIndicators.add_rate_of_change,
                "Money Flow Index (MFI)": TechnicalIndicators.add_money_flow_index,
                "Standard Deviation": TechnicalIndicators.add_standard_deviation,
                "Historical Volatility": TechnicalIndicators.add_historical_volatility,
                "Chandelier Exit": TechnicalIndicators.add_chandelier_exit,
                "Keltner Channel": TechnicalIndicators.add_keltner_channel,
                "Moving Average Envelope (MAE)": TechnicalIndicators.add_moving_average_envelope,
                "Average Directional Index (ADX)": TechnicalIndicators.add_adx,
                "Ichimoku Cloud": TechnicalIndicators.add_ichimoku_cloud,
                "Parabolic SAR": TechnicalIndicators.add_parabolic_sar,
                "Zigzag Indicator": TechnicalIndicators.add_zigzag_indicator,  # Assuming this method exists in TechnicalIndicators
                "On-Balance Volume (OBV)": TechnicalIndicators.add_on_balance_volume,
                "Volume Weighted Average Price (VWAP)": TechnicalIndicators.add_vwap,
                "Accumulation/Distribution Line (ADL)": TechnicalIndicators.add_accumulation_distribution_line,
                "Chaikin Money Flow (CMF)": TechnicalIndicators.add_chaikin_money_flow,
                "Volume Oscillator": TechnicalIndicators.add_volume_oscillator,
                "Awesome Oscillator": TechnicalIndicators.add_awesome_oscillator,
                "TRIX": TechnicalIndicators.add_trix,
                "Standard Pivot Points": TechnicalIndicators.add_standard_pivot_points,
                "Fibonacci Retracements": TechnicalIndicators.add_fibonacci_retracement_levels,
                # ... add any additional features and their corresponding methods here ...
            }

            # Check if the feature is in the feature methods dictionary
            if feature_name in feature_methods:
                # Get the method corresponding to the feature name
                feature_method = feature_methods[feature_name]
                # Apply the feature method to the DataFrame
                result = feature_method(df)

                # Ensure the returned result is a DataFrame
                if not isinstance(result, pd.DataFrame):
                    raise ValueError(f"Expected a DataFrame from {feature_name}, but got {type(result)}.")

                # Debug mode logging
                if self.is_debug_mode:
                    self.utils.log_message(f"Applied {feature_name}.", self.log_message)
                    self.utils.log_message(f"DataFrame columns after processing {feature_name}: {result.columns}", self.log_message)
                    self.utils.log_message(f"DataFrame sample after processing {feature_name}:\n{result.head()}", self.log_message)

                return result  # Return the DataFrame with the applied feature

            else:
                # Log a message if the feature method is not found
                self.utils.log_message(f"Feature method for '{feature_name}' not found.", self.log_message)

        except Exception as e:
            # Log the error and rethrow if in debug mode
            if self.is_debug_mode:
                self.utils.log_message(f"Error applying {feature_name}: {str(e)}", self.log_message, self.is_debug_mode)

            raise  # Rethrow the exception to handle it in the calling method

    def standardize_column_names(self, df):
        column_renames = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        df.rename(columns=column_renames, inplace=True)

def main():
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')  # Replace with your config file path

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("ML Robot")

    # Create an instance of DataProcessingTab, passing both root and config
    app = DataProcessingTab(root, config)

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()

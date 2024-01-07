#data_processing_tab.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import configparser
from Data_processing.technical_indicators import TechnicalIndicators
from Utils import MLRobotUtils, update_status, auto_generate_save_path

class DataProcessingTab:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.debug_mode = False  # Initialize debug mode to False

        # Initialize MLRobotUtils with debug mode state
        self.utils = MLRobotUtils(is_debug_mode=self.debug_mode)

        # Initialize GUI components
        self.setup_gui()

    def setup_gui(self):
        # Frame for CSV File Selection
        file_frame = tk.Frame(self.parent)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(file_frame, text="Select CSV File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_entry = tk.Entry(file_frame, width=50)
        self.file_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.browse_button = tk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        # Frame for Features/Indicators
        features_frame = tk.Frame(self.parent)
        features_frame.pack(padx=10, pady=(0, 5))

        tk.Label(features_frame, text="Select Features/Indicators:").pack(anchor=tk.W)
        self.features_listbox = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, height=15)
        self.features_listbox.pack(side=tk.LEFT, padx=(0, 10))


        # Buttons for Start Processing, Clear Logs, and Toggle Debug Mode
        buttons_frame = tk.Frame(self.parent)
        buttons_frame.pack(pady=5)
        self.start_button = tk.Button(buttons_frame, text="Start Processing", command=self.process_data)
        self.start_button.pack(side=tk.LEFT, fill='x', padx=(0, 5))
        self.clear_logs_button = tk.Button(buttons_frame, text="Clear Logs", command=self.clear_logs)
        self.clear_logs_button.pack(side=tk.LEFT)



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
            "Zigzag Indicator",  # Added Zigzag Indicator
            "On-Balance Volume (OBV)",
            "Volume Weighted Average Price (VWAP)",
            "Accumulation/Distribution Line (ADL)",
            "Chaikin Money Flow (CMF)",
            "Volume Oscillator",
            "Awesome Oscillator",
            "TRIX",  # Added TRIX
            "Standard Pivot Points"  # Added Standard Pivot Points
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

        # Toggle Debug Mode Button
        self.debug_mode_button = tk.Button(self.parent, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.pack()
            
        if hasattr(self, 'log_text'):
            self.log_text.delete('1.0', tk.END)  # Clear existing content
        else:
            # Create the Data Processing Log frame and ScrolledText only if it does not exist
            log_frame = tk.Frame(self.parent)
            log_frame.pack(padx=10, pady=(0, 5))
            tk.Label(log_frame, text="Data Processing Log:").pack(anchor=tk.W)
            self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
            self.log_text.pack(fill='both', expand=True)

    def browse_file(self):
        # Open a file dialog to choose a CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        # If a file is selected, update the file_path_entry with this path
        if file_path:
            self.file_path_entry.delete(0, tk.END)  # Clear any existing text in the entry
            self.file_path_entry.insert(0, file_path)  # Insert the selected file path

    def clear_logs(self):
        if hasattr(self, 'log_text'):
            self.log_text.delete('1.0', tk.END)  # Clear the content of the log text widget


    def toggle_debug_mode(self):
        # Toggle debug mode state
        self.debug_mode = not self.debug_mode

        # Update the MLRobotUtils instance with the new debug mode state
        self.utils.is_debug_mode = self.debug_mode

        # Update the button text to reflect the current state
        if self.debug_mode:
            self.debug_mode_button.config(text="Debug Mode: ON")
        else:
            self.debug_mode_button.config(text="Debug Mode: OFF")

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
            self.log_text.insert(tk.END, "Input Error: Please select a CSV file.\n")
            return

        if not selected_features:
            messagebox.showwarning("Input Error", "Please select at least one feature/indicator.")
            self.log_text.insert(tk.END, "Input Error: Please select at least one feature/indicator.\n")
            return

        try:
            df = pd.read_csv(file_path)
            self.standardize_column_names(df)  # Standardize column names if necessary
            df.fillna(method='ffill', inplace=True)  # Forward fill to handle NaN values
            df.fillna(method='bfill', inplace=True)  # Backward fill as a fallback
            self.log_text.insert(tk.END, "CSV file loaded successfully.\n")

        except Exception as e:
            self.log_text.insert(tk.END, f"Error loading CSV file: {str(e)}\n")
            messagebox.showerror("Loading Error", f"Error loading CSV file: {str(e)}")
            return

        # Apply selected features
        for feature in selected_features:
            try:
                # Debugging: Print DataFrame structure and sample data before processing
                if self.debug_mode:
                    self.utils.log_message(f"DataFrame columns before processing {feature}: {df.columns}", self.log_text)
                    self.utils.log_message(f"DataFrame sample before processing {feature}:\n{df.head()}", self.log_text)

                    if feature == "Simple Moving Average (SMA)":
                        df = TechnicalIndicators.add_moving_average(df)
                    elif feature == "Exponential Moving Average (EMA)":
                        df = TechnicalIndicators.add_exponential_moving_average(df)
                    elif feature == "Bollinger Bands":
                        df = TechnicalIndicators.add_bollinger_bands(df)
                    elif feature == "Stochastic Oscillator":
                        df = TechnicalIndicators.add_stochastic_oscillator(df)
                    elif feature == "MACD":
                        df = TechnicalIndicators.calculate_macd_components(df)
                    elif feature == "Average True Range (ATR)":
                        df = TechnicalIndicators.add_average_true_range(df)
                    elif feature == "Relative Strength Index (RSI)":
                        df = TechnicalIndicators.add_relative_strength_index(df)
                    elif feature == "Commodity Channel Index (CCI)":
                        df = TechnicalIndicators.add_commodity_channel_index(df)
                    elif feature == "Williams %R":
                        df = TechnicalIndicators.add_williams_r(df)
                    elif feature == "Rate of Change (ROC)":
                        df = TechnicalIndicators.add_rate_of_change(df)
                    elif feature == "Money Flow Index (MFI)":
                        df = TechnicalIndicators.add_money_flow_index(df)
                    elif feature == "Standard Deviation":
                        df = TechnicalIndicators.add_standard_deviation(df)
                    elif feature == "Historical Volatility":
                        df = TechnicalIndicators.add_historical_volatility(df)
                    elif feature == "Chandelier Exit":
                        df = TechnicalIndicators.add_chandelier_exit(df)
                    elif feature == "Keltner Channel":
                        df = TechnicalIndicators.add_keltner_channel(df)
                    elif feature == "Moving Average Envelope (MAE)":
                        df = TechnicalIndicators.add_moving_average_envelope(df)
                    elif feature == "Average Directional Index (ADX)":
                        df = TechnicalIndicators.add_adx(df)
                    elif feature == "Ichimoku Cloud":
                        df = TechnicalIndicators.add_ichimoku_cloud(df)
                    elif feature == "Parabolic SAR":
                        # Debugging: Print message before applying Parabolic SAR
                        if self.debug_mode:
                            self.utils.log_message("Applying Parabolic SAR...", self.log_text)

                        df = TechnicalIndicators.add_parabolic_sar(df)

                        # Debugging: Print DataFrame structure and sample data after applying Parabolic SAR
                        if self.debug_mode:
                            self.utils.log_message("Parabolic SAR applied.", self.log_text)
                            self.utils.log_message(f"DataFrame columns after applying Parabolic SAR: {df.columns}", self.log_text)
                            self.utils.log_message(f"DataFrame sample after applying Parabolic SAR:\n{df.head()}", self.log_text)

                        if hasattr(TechnicalIndicators, 'add_zigzag_indicator'):
                            df = TechnicalIndicators.add_zigzag_indicator(df)
                        else:
                            self.utils.log_message("Zigzag indicator method not found in TechnicalIndicators.", self.log_text)
                    elif feature == "On balance Volume":
                        df = TechnicalIndicators.add_on_balance_volume(df)
                    elif feature == "Volume Weighted Average Price (VWAP)":
                        df = TechnicalIndicators.add_vwap(df)
                    elif feature == "Accumulation/Distribution Line (ADL)":
                        df = TechnicalIndicators.add_accumulation_distribution_line(df)
                    elif feature == "Chaikin Money Flow (CMF)":
                        df = TechnicalIndicators.add_chaikin_money_flow(df)
                    elif feature == "Volume Oscillator":
                        df = TechnicalIndicators.add_volume_oscillator(df)
                    elif feature == "Awesome Oscillator":
                        df = TechnicalIndicators.add_awesome_oscillator(df)
                    elif feature == "TRIX":
                        df = TechnicalIndicators.add_trix(df)
                    elif feature == "Standard Pivot Points":
                        df = TechnicalIndicators.add_standard_pivot_points(df)
                    # ... additional features as needed ...

                    # Log processing message
                    self.utils.log_message(f"Applied {feature}.", self.log_text)
            except Exception as feature_error:
                self.utils.log_message(f"Error applying {feature}: {str(feature_error)}", self.log_text)
                continue  # Optionally, continue with the next feature

        try:
            # Assuming 'save_path_config' is the configuration key for the save path
            save_path_dir = self.config.get('SAVE_PATH_SECTION', 'save_path_dir')
            
            # Generate the save path
            save_path = auto_generate_save_path(file_path, save_path_dir)
            
            df.to_csv(save_path, index=False)
            self.utils.log_message(f"Processed data saved to '{save_path}'.", self.log_text)
            self.log_text.insert(tk.END, "Data processing completed.\n")  # Log the completion message
        except Exception as e:
            self.utils.log_message(f"Error during data processing: {str(e)}", self.log_text)
            messagebox.showerror("Processing Error", str(e))
# Function to generate the save path based on file_path and config

    def generate_save_path(file_path, config):
        # Extract the directory and filename from the input file path
        directory, filename = os.path.split(file_path)

        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)

        # Optionally, you can use a directory specified in the config
        # For example, if your config has a section "SAVE_PATH_SECTION" with a key "save_path_dir"
        save_directory = config.get('SAVE_PATH_SECTION', 'save_path_dir', fallback=directory)

        # Construct the new filename by appending '_processed'
        new_filename = f"{name}_processed{extension}"

        # Combine the directory and new filename to form the full save path
        save_path = os.path.join(save_directory, new_filename)

        return save_path
    
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

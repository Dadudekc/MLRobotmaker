#MLROBOTtk.py

# Section 1: Imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkcalendar import Calendar, DateEntry
import threading
import configparser
import traceback
import os
import pandas as pd
import logging
# Import modules from your project structure
from Data_fetch.main import main as fetch_main
from Data_fetch import polygon_io
from Data_processing.technical_indicators import TechnicalIndicators
from Data_processing.data_processing import process_data
from model_development.model_training import train_model, create_neural_network, save_model, perform_hyperparameter_tuning, bayesian_hyperparameter_tuning, CustomHyperModel
from model_development import model_training
from model_development.model_management import DataLoader, DataTransformer, FeatureEngineering, ModelManager
from model_development.model_development import split_and_scale_data, train_and_evaluate_model, evaluate_model
from model_development.data_preprocessing import preprocess_data




# Global debug flag
is_debug_mode = False

def configure_application():
    config = configparser.ConfigParser()

    def select_directory(entry):
        directory = filedialog.askdirectory()
        if is_debug_mode:\

            
            log_message(f"Debug: Selected directory - {directory}")
        entry.delete(0, tk.END)
        entry.insert(0, directory)

    def save_preferences():
        if is_debug_mode:
            log_message("Debug: Saving application preferences")
        # Save preferences related to data directories
        config['DataDirectories']['DataFetchDirectory'] = data_fetch_entry.get()
        config['DataDirectories']['DataProcessingDirectory'] = data_processing_entry.get()
        config['DataDirectories']['ModelTrainingDirectory'] = model_training_entry.get()

        # Save last directory preference
        config['DEFAULT']['LastDirectory'] = directory_entry.get()
        
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    config_dialog = tk.Tk()
    config_dialog.title("Configuration Wizard")

    # Data directories configuration
    tk.Label(config_dialog, text="Data Fetch Directory:").pack()
    data_fetch_entry = tk.Entry(config_dialog)
    data_fetch_entry.pack()
    tk.Button(config_dialog, text="Browse", command=lambda: select_directory(data_fetch_entry)).pack()

    tk.Label(config_dialog, text="Data Processing Directory:").pack()
    data_processing_entry = tk.Entry(config_dialog)
    data_processing_entry.pack()
    tk.Button(config_dialog, text="Browse", command=lambda: select_directory(data_processing_entry)).pack()

    tk.Label(config_dialog, text="Model Training Directory:").pack()
    model_training_entry = tk.Entry(config_dialog)
    model_training_entry.pack()
    tk.Button(config_dialog, text="Browse", command=lambda: select_directory(model_training_entry)).pack()

    # Last directory entry
    tk.Label(config_dialog, text="Last Directory:").pack()
    directory_entry = tk.Entry(config_dialog)
    directory_entry.pack()
    tk.Button(config_dialog, text="Browse", command=lambda: select_directory(directory_entry)).pack()

    save_button = tk.Button(config_dialog, text="Save Preferences", command=save_preferences)
    save_button.pack()

    config_dialog.mainloop()

def toggle_debug_mode(debug_button):
    global is_debug_mode
    is_debug_mode = not is_debug_mode
    debug_button.config(text=f"Debug Mode: {'ON' if is_debug_mode else 'OFF'}")
    print(f"Debug Mode is now {'ON' if is_debug_mode else 'OFF'}")


def setup_data_fetch_tab(tab):
    config = configparser.ConfigParser()

    def log_message(message, log_text=None):
        global is_debug_mode
        if is_debug_mode:
            print(message)

        if log_text:
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, message + "\n")
            log_text.config(state=tk.DISABLED)


    # Define directory_entry
    directory_entry = tk.Entry(tab)
    directory_entry.pack()

    # Function to browse directory
    def browse_directory():
        directory = filedialog.askdirectory()
        if directory:  # Check if a directory was selected
            directory_entry.delete(0, tk.END)
            directory_entry.insert(0, directory)
            if is_debug_mode:
                log_message(f"Debug: Directory selected - {directory}")


    # Button to trigger directory browsing
    browse_button = tk.Button(tab, text="Browse", command=browse_directory)
    browse_button.pack()

    def save_preferences():
        config['DEFAULT']['LastDirectory'] = directory_entry.get()
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def load_preferences():
        config.read('config.ini')
        default_dir = config.get('csv_directory', 'directory_path', fallback='')
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, default_dir)

    def fetch_data():
        csv_dir = directory_entry.get()
        ticker_symbols = tickers_entry.get().split(',')
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()
        selected_api = api_var.get()

        if is_debug_mode:
            log_message(f"Debug: Fetch request received")
            log_message(f"Debug: CSV Directory - {csv_dir}")
            log_message(f"Debug: Tickers - {ticker_symbols}")
            log_message(f"Debug: Start Date - {start_date}, End Date - {end_date}")
            log_message(f"Debug: API Selected - {selected_api}")

        if not (csv_dir and ticker_symbols):
            messagebox.showwarning("Input Error", "Please specify the CSV directory and stock symbols.")
            return

        fetch_button.config(state=tk.DISABLED)
        status_label.config(text="Fetching data...")
        log_message("Initiating data fetch... Please wait while data is being fetched.")
        threading.Thread(target=lambda: fetch_data_threaded(csv_dir, ticker_symbols, start_date, end_date, selected_api)).start()


    def fetch_data_threaded(csv_dir, ticker_symbols, start_date, end_date, selected_api):
        try:
            if is_debug_mode:
                log_message(f"Debug: Starting data fetch for symbols: {ticker_symbols}")
                log_message(f"Debug: Date range from {start_date} to {end_date}")
                log_message(f"Debug: Using API {selected_api}")
                log_message(f"Debug: CSV directory set to {csv_dir}")

            fetch_main(csv_dir, ticker_symbols, start_date, end_date, selected_api)
            
            if is_debug_mode:
                log_message("Debug: Data fetch completed successfully.")

            status_label.config(text="Data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            
            if is_debug_mode:
                log_message("Debug: An error occurred during data fetch")
                log_message(f"Debug: Error details - {error_message}")
                print(f"Debug: Error occurred during data fetch - {error_message}")  # Print error to terminal in debug mode

            messagebox.showerror("Error", error_message)
        finally:
            fetch_button.config(state=tk.NORMAL)

    def clear_logs():
        log_text.config(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.config(state=tk.DISABLED)

    # GUI Elements for Data Fetch Tab
    tk.Label(tab, text="Stock Tickers (comma separated):").pack()
    tickers_entry = tk.Entry(tab)
    tickers_entry.pack()

    tk.Label(tab, text="Start Date:").pack()
    start_date_entry = DateEntry(tab, width=12, background='darkblue', foreground='white', borderwidth=2)
    start_date_entry.pack()

    tk.Label(tab, text="End Date:").pack()
    end_date_entry = DateEntry(tab, width=12, background='darkblue', foreground='white', borderwidth=2)
    end_date_entry.pack()

    tk.Label(tab, text="Select API:").pack()
    api_var = tk.StringVar()
    api_dropdown = ttk.Combobox(tab, textvariable=api_var, values=["AlphaVantage", "polygonio", "Nasdaq"])
    api_dropdown.pack()

    fetch_button = tk.Button(tab, text="Fetch Data", command=fetch_data)
    fetch_button.pack()

    status_label = tk.Label(tab, text="")
    status_label.pack()

    log_label = tk.Label(tab, text="Logs:")
    log_label.pack()
    log_text = scrolledtext.ScrolledText(tab, height=10)
    log_text.pack()

    clear_logs_button = tk.Button(tab, text="Clear Logs", command=clear_logs)
    clear_logs_button.pack()

    ttk.Button(tab, text="Save Preferences", command=save_preferences).pack()



# Section 3: Data Processing Tab Setup


def process_data_wrapper(save_path_entry, file_path_entry, features_listbox, status_output, log_text):
    file_path = file_path_entry.get()
    selected_features = [features_listbox.get(idx) for idx in features_listbox.curselection()]

    if not file_path:
        messagebox.showwarning("Input Error", "Please select a CSV file.")
        return

    if not selected_features:
        messagebox.showwarning("Input Error", "Please select at least one feature/indicator.")
        return
    
    # Extracting ticker symbol from the filename
    # Assuming the file name is something like 'AAPL_data.csv'
    ticker_symbol = os.path.basename(file_path).split('_')[0].upper()

    try:
        log_message("Starting data processing...", log_text)
        update_status(status_output, "Processing...")

        # Read the CSV file
        df = pd.read_csv(file_path)
        log_message("CSV file loaded successfully.", log_text)

        required_columns = ['close', 'high']
        missing_columns = [col for col in required_columns if col not in df.columns.str.lower()]
        if missing_columns:
            error_msg = f"Error: Missing required columns in CSV file: {', '.join(missing_columns)}"
            log_message(error_msg, log_text)
            messagebox.showerror("Data Processing Error", error_msg)
            return

        # Apply selected features/indicators
        for feature in selected_features:
            if is_debug_mode:
                print(f"Applying feature: {feature}")
                print(f"DataFrame shape before applying {feature}: {df.shape}")
                print(f"DataFrame sample before applying {feature}:\n{df.head()}")

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
                df = TechnicalIndicators.add_parabolic_sar(df)
            elif feature == "Zigzag Indicator":
                df = TechnicalIndicators.add_zigzag_indicator(df)
            elif feature == "On-Balance Volume (OBV)":
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

            if is_debug_mode:
                print(f"DataFrame shape after applying {feature}: {df.shape}")
                print(f"DataFrame sample after applying {feature}:\n{df.head()}")

# Optional: Display processed data in the GUI

        # Add this function outside of process_data_wrapper, at the top level of your script
        def auto_generate_save_path(input_file_path, config):
            base_dir = config.get('DataDirectories', 'DataProcessingDirectory', fallback='default_directory')
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.basename(input_file_path).replace('.csv', f'_processed_{timestamp}.csv')
            return os.path.join(base_dir, filename)

        # Inside the process_data_wrapper function
        new_save_path = auto_generate_save_path(file_path, config)

        # Debugging log to check the type of df
        log_message(f"Type of df before saving: {type(df)}", log_text)

        # Check if df is a DataFrame
        if isinstance(df, pd.DataFrame):
            os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
            try:
                df.to_csv(new_save_path, index=False)
            except Exception as e:
                log_message(f"Error during file save: {str(e)}", log_text)
                messagebox.showerror("File Save Error", str(e))
                return

                log_message(f"Processed data saved to '{new_save_path}'.", log_text)
            else:
                log_message("Error: Data is not in DataFrame format.", log_text)
                messagebox.showerror("Data Processing Error", "Data is not in DataFrame format.")

        log_message(f"Processed data saved to '{new_save_path}'.", log_text)

        update_status(status_output, "Data processing completed.")
    
    except Exception as e:
        log_message(f"Error during data processing: {str(e)}", log_text)
        update_status(status_output, "Error during data processing.")
        messagebox.showerror("Processing Error", str(e))

        if is_debug_mode:
            print(f"Debug: Error during data processing - {str(e)}") 

# Helper functions for logging and status updates
def log_message(message, log_text=None):
    global is_debug_mode
    if is_debug_mode:
        print(message)

    if log_text:
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.config(state=tk.DISABLED)


    # The existing functionality to update the GUI log
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message + "\n")
    log_text.config(state=tk.DISABLED)

def update_status(status_output, message):
    status_output.config(state=tk.NORMAL)
    status_output.delete(1.0, tk.END)
    status_output.insert(tk.END, message + "\n")
    status_output.config(state=tk.DISABLED)

def clear_logs(log_text):
    log_text.config(state=tk.NORMAL)
    log_text.delete(1.0, tk.END)
    log_text.config(state=tk.DISABLED)

def setup_data_processing_tab(tab):
    config = configparser.ConfigParser()
    config.read('config.ini')

    # GUI Elements Creation
    tk.Label(tab, text="Data Processing").pack()

    # Frame for CSV File Selection
    file_frame = tk.Frame(tab)
    file_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(file_frame, text="Select CSV File:").pack(side=tk.LEFT)
    file_path_entry = tk.Entry(file_frame, width=50)
    file_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10,0))

    def select_all_features():
    # Select all items in the features listbox
        for idx in range(features_listbox.size()):
            features_listbox.select_set(idx)

    def browse_file():
        default_dir = config.get('DataDirectories', 'DataProcessingDirectory', fallback='')
        filepath = filedialog.askopenfilename(initialdir=default_dir, 
                                              filetypes=[("CSV Files", "*.csv")])
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, filepath)

    browse_button = tk.Button(file_frame, text="Browse", command=browse_file)
    browse_button.pack(side=tk.LEFT, padx=(0,10))

    # Selected Features/Indicators
    tk.Label(tab, text="Select Features/Indicators:").pack()
    features_listbox = tk.Listbox(tab, selectmode=tk.MULTIPLE)
    features_listbox.pack()

    # Button to select all features
    select_all_button = tk.Button(tab, text="Select All Features", command=select_all_features)
    select_all_button.pack()

    # Populate the listbox with features/indicators
    features = ["Simple Moving Average (SMA)", "Exponential Moving Average (EMA)",
                "Bollinger Bands", "Stochastic Oscillator", "MACD", 
                "Average True Range (ATR)", "Relative Strength Index (RSI)",
                "Commodity Channel Index (CCI)", "Williams %R", 
                "Rate of Change (ROC)", "Money Flow Index (MFI)",
                "Standard Deviation", "Historical Volatility", 
                "Chandelier Exit", "Keltner Channel", "Moving Average Envelope (MAE)",
                "Average Directional Index (ADX)", "Ichimoku Cloud", 
                "Parabolic SAR", "On-Balance Volume (OBV)",
                "Volume Weighted Average Price (VWAP)", "Accumulation/Distribution Line (ADL)",
                "Chaikin Money Flow (CMF)", "Volume Oscillator", "Awesome Oscillator"]
    for feature in features:
        features_listbox.insert(tk.END, feature)

    # Status Output
    status_output = tk.Text(tab, height=5)
    status_output.pack(fill='both', expand=True, padx=5, pady=5)

    # Data Processing Log
    log_label = tk.Label(tab, text="Data Processing Log:")
    log_label.pack()
    log_text = scrolledtext.ScrolledText(tab, height=10)
    log_text.pack()

    # Add GUI elements for saving processed data
    tk.Label(tab, text="Save Processed Data As:").pack()
   # Change save_path_entry to a Label for preview
    save_path_label = tk.Label(tab, text="Save path will be shown here after processing")
    save_path_label.pack(fill='x', padx=10, pady=5)

    def browse_save_path():
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            # Check if the user selected a file path (not canceled)
            save_path_label.config(text=f"File will be saved as: {filepath}")

        save_path_entry.delete(0, tk.END)
        save_path_entry.insert(0, filepath)


    save_browse_button = tk.Button(tab, text="Browse", command=browse_save_path)
    save_browse_button.pack(side=tk.LEFT, padx=(0,10))

    # Event Handlers - Update process_data function as needed
    start_button = tk.Button(tab, text="Start Processing", 
                            command=lambda: process_data_wrapper(save_path_entry, file_path_entry, features_listbox, status_output, log_text))
    start_button.pack(fill='x', padx=5, pady=5)
    clear_logs_button = tk.Button(tab, text="Clear Logs", command=lambda: clear_logs(log_text))
    clear_logs_button.pack()

    # Update the label to display the save path
    save_path_label.config(text=f"File will be saved as: {file_path_entry}")


#section 4: model training tab setup
# Global list to store models
trained_models = []
def setup_model_training_tab(tab):
    # Read configuration settings
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Data File Selection
    def browse_data_file():
        default_dir = config.get('DataDirectories', 'DataProcessingDirectory', fallback='')
        filepath = filedialog.askopenfilename(initialdir=default_dir, filetypes=[("CSV Files", "*.csv")])
        data_file_entry.delete(0, tk.END)
        data_file_entry.insert(0, filepath)

    data_file_frame = tk.Frame(tab)
    data_file_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(data_file_frame, text="Select Data File:").pack(side=tk.LEFT)
    data_file_entry = tk.Entry(data_file_frame, width=50)
    data_file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10, 0))
    data_file_browse_button = tk.Button(data_file_frame, text="Browse", command=browse_data_file)
    data_file_browse_button.pack(side=tk.LEFT, padx=(0, 10))

    # Define scaler options
    scaler_options = ['standard', 'minmax', 'robust', 'quantile', 'power', 'normalizer', 'maxabs']

    # Add Scaler Selection Dropdown to the GUI
    tk.Label(tab, text="Select Scaler Type:").pack()
    scaler_type_var = tk.StringVar()
    scaler_type_dropdown = ttk.Combobox(tab, textvariable=scaler_type_var, values=scaler_options)
    scaler_type_dropdown.pack()

    # Model Selection
    tk.Label(tab, text="Select Model Type:").pack()
    model_type_var = tk.StringVar()
    model_type_dropdown = ttk.Combobox(tab, textvariable=model_type_var, values=["linear_regression", "random_forest", "neural_network"])
    model_type_dropdown.pack()

    # Training Parameters
    tk.Label(tab, text="Epochs:").pack()
    epochs_entry = tk.Entry(tab)
    epochs_entry.pack()

    # Progress Bar
    progress = ttk.Progressbar(tab, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.pack()

    # Model Information Text
    model_info_text = tk.Text(tab, height=5)
    model_info_text.pack()

    # Error Label
    error_label = tk.Label(tab, text="", fg="red")
    error_label.pack()

    # Log Text
    log_text = scrolledtext.ScrolledText(tab, height=10)
    log_text.pack()

    def log_training_message(message, log_text):
        # Log to console if debug mode is active
        if is_debug_mode:
            print(message)

        # Log to the Tkinter text widget
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.config(state=tk.DISABLED)
        log_text.yview(tk.END)

    

    #Start_training function
    def start_training():
        data_file_path = data_file_entry.get()
        model_type = model_type_var.get()
        epochs = int(epochs_entry.get())

        # Define test_size and scaler_type
        test_size = 0.2  # Example: 20% of data used for testing
        scaler_type = scaler_type_var.get()  # Get the selected scaler type

        data_loader = DataLoader(config)
        try:
            log_training_message("Loading data...", log_text)
            print("Loading data...")  # Print statement for debugging

            data = data_loader.load_data(data_file_path)
            log_training_message(f"Data loaded successfully. Data shape: {data.shape}", log_text)
            print(f"Data loaded successfully. Data shape: {data.shape}")  # Print statement for debugging

            log_training_message(f"Raw data preview:\n{data.head()}", log_text)
            print(f"Raw data preview:\n{data.head()}")  # Print statement for debugging

            # Start preprocessing
            log_training_message("Starting preprocessing...", log_text)
            print("Starting preprocessing...")  # Print statement for debugging

            # Call preprocess_data with the correct arguments
            data = preprocess_data(data, fill_method='mean', date_column='date')

            log_training_message("Preprocessing completed.", log_text)
            print("Preprocessing completed.")  # Print statement for debugging

            log_training_message(f"Data preview after preprocessing:\n{data.head()}", log_text)
            print(f"Data preview after preprocessing:\n{data.head()}")  # Print statement for debugging

            # Prepare data for model training
            X = data.drop('close', axis=1)
            y = data['close']

            log_training_message("Splitting and scaling data...", log_text)
            print("Splitting and scaling data...")  # Print statement for debugging

            # Split and scale the data
            global X_test, y_test
            X_train, X_test, y_train, y_test = split_and_scale_data(X, y, test_size, scaler_type)

            log_training_message("Data splitting and scaling completed.", log_text)
            print("Data splitting and scaling completed.")  # Print statement for debugging

            # Start model training
            log_training_message("Starting model training...", log_text)
            print("Starting model training...")  # Print statement for debugging

            global trained_models
            model, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type, epochs)
            trained_models.append(model)
            display_model_info(model) 
            log_training_message("Model training completed.", log_text)
            log_training_message(f"Model Evaluation Metrics: {metrics}", log_text)
            print("Model training completed.")  # Print statement for debugging
            print(f"Model Evaluation Metrics: {metrics}")  # Print statement for debugging

            global trained_model
            trained_model = model

        except Exception as e:
            log_training_message(f"Error in model training: {str(e)}", log_text)
            print(f"Debug: Error in model training - {str(e)}")  # Print error to terminal in debug mode


    # Save Model Function
    def save_model():
        if trained_model is not None:
            file_path = filedialog.asksaveasfilename(filetypes=[("Model Files", "*.model")])
            if file_path:
                model_training.save_model(trained_model, file_path)

    # Load Model Function
    def load_model():
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.model")])
        if file_path:
            global trained_model
            trained_model = model_training.load_model(file_path)
            display_model_info(trained_model)

    # Function to compare models
    def compare_trained_models():
        try:
            global trained_models, X_test, y_test

            if len(trained_models) > 1:
                for model in trained_models:
                    evaluation_results = evaluate_model(model, X_test, y_test, task_type='regression')
                    print(evaluation_results)
                # You can extend this loop to compare models based on specific metrics
            else:
                print("Not enough models to compare. Train more models.")
            
            print("Comparing models...")

        except Exception as e:
            print(f"Error during model comparison: {e}")
            raise


    # GUI Element for Training
    start_training_button = tk.Button(tab, text="Start Training", command=lambda: threading.Thread(target=start_training).start())
    start_training_button.pack()

    save_model_button = tk.Button(tab, text="Save Model", command=save_model)
    load_model_button = tk.Button(tab, text="Load Model", command=load_model)
    compare_models_button = tk.Button(tab, text="Compare Models", command=compare_trained_models)
    save_model_button.pack()
    load_model_button.pack()
    compare_models_button.pack()
    
    # Display Model Information
    def display_model_info(model):
        info = f"Model Type: {model.__class__.__name__}\n"
        # Add other model information here
        model_info_text.delete(1.0, tk.END)
        model_info_text.insert(tk.END, info)

    # Function to Update GUI with Error
    def update_gui_with_error(error_message, tab):
        #GUI error message update logic here
        error_label = tk.Label(tab, text="", fg="red")
        error_label.config(text=f"Error: {error_message}")
        error_label.pack()

#section 5: strategy testing tab setup
#section 6: deployment tab setup

# Section 7: Main window setup and tab control

# Main window setup and tab control
root = tk.Tk()
root.title("Trading Robot Development Platform")

# Create the tab control
tab_control = ttk.Notebook(root)

# Create tabs
data_fetch_tab = ttk.Frame(tab_control)
data_processing_tab = ttk.Frame(tab_control)
model_training_tab = ttk.Frame(tab_control)
strategy_testing_tab = ttk.Frame(tab_control)
deployment_tab = ttk.Frame(tab_control)

# Add tabs to the tab control in the desired order top to bottom = left to right
tab_control.add(data_fetch_tab, text='Data Fetching')
tab_control.add(data_processing_tab, text='Data Processing')
tab_control.add(model_training_tab, text='Model Training')
tab_control.add(strategy_testing_tab, text='Strategy Testing')
tab_control.add(deployment_tab, text='Deployment')

# Configuration Wizard Button
configure_button = tk.Button(root, text="Configure Application", command=configure_application)
configure_button.pack()

# In your main window setup
debug_button = tk.Button(root, text="Debug Mode: OFF", command=lambda: toggle_debug_mode(debug_button))
debug_button.pack()

# Setup each tab
setup_data_fetch_tab(data_fetch_tab)
setup_data_processing_tab(data_processing_tab)
setup_model_training_tab(model_training_tab)

# # setup_strategy_testing_tab(strategy_testing_tab)
# setup_deployment_tab(deployment_tab)

# Pack the tab control
tab_control.pack(expand=1, fill="both")

# Run the application
root.mainloop()

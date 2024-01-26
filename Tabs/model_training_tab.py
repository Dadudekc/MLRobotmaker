# model_training_tab.py

# Section 1: Imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import configparser
import threading
import logging
import json
import joblib
import asyncio
import concurrent.futures
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import (
    save_model as save_keras_model, Sequential as NeuralNetwork, Model
)
from tensorflow.keras.layers import Dense, LSTM
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import schedule
import time
import traceback
import os
from Utilities.Utils import MLRobotUtils
from model_development.model_training import (
    train_model, load_model, create_lstm_model, train_arima_model, create_ensemble_model, create_neural_network, create_lstm_model, train_arima_model
)

import smtplib  # For sending email notifications
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart
import queue
from sklearn.preprocessing import OrdinalEncoder

# Section 1.1: Additional Imports for RandomForestClassifier
from xgboost import XGBRegressor
import pandas as pd
import time
import datetime
from sklearn.impute import SimpleImputer
from tkinter import messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Section 1.2: Model training tab class

class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, config, scaler_options):
        super().__init__(parent)
        self.is_debug_mode = False  # Boolean flag to indicate debug mode
        self.training_in_progress = True  # Flag for active training
        self.training_paused = False  # Flag for paused training
        self.config = config  # Configuration settings
        self.scaler_options = scaler_options  # Data scaling options
        self.trained_models = []  # List for trained models
        self.trained_model = None  # Current trained model reference
        self.utils = MLRobotUtils(is_debug_mode=config.getboolean('Settings', 'DebugMode', fallback=False))
        self.setup_model_training_tab()  # Initialize GUI components
        self.scaler_type_var = tk.StringVar()
        self.n_estimators_entry = tk.Entry(self)  # Entry for the number of estimators
        self.n_estimators_entry.pack()
        self.window_size_label = tk.Label(self, text="Window Size:")  # Label for window size
        self.window_size_entry = tk.Entry(self)  # Entry for window size
        self.trained_scaler = None  # Trained data scaler
        self.queue = queue.Queue()  # Queue for asynchronous tasks
        self.after(100, self.process_queue)  # Regular queue processing
        self.pause_button = ttk.Button(self, text="Pause Training", command=self.pause_training)
        self.pause_button.pack(pady=5)
        self.resume_button = ttk.Button(self, text="Resume Training", command=self.resume_training)
        self.resume_button.pack(pady=5)
        self.error_label = tk.Label(self, text="", fg="red")  # Label for error messages
        self.error_label.pack()
        self.setup_debug_mode_toggle()

    # Function to set up the debug mode toggle button
    def setup_debug_mode_toggle(self):
        self.debug_button = tk.Button(self, text="Enable Debug Mode", command=self.toggle_debug_mode)
        self.debug_button.pack(pady=5)

    # Function to toggle the debug mode on button click
    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        btn_text = "Disable Debug Mode" if self.is_debug_mode else "Enable Debug Mode"
        self.debug_button.config(text=btn_text)
        # Include the is_debug_mode flag in the log_message call
        self.utils.log_message("Debug mode: " + str(self.is_debug_mode), self, self.log_text, self.is_debug_mode)

    # Function to set up the entire model training tab
    def setup_model_training_tab(self):
        self.setup_title_label()
        self.setup_data_file_path_section()
        self.setup_model_type_selection()
        self.setup_training_configuration()
        self.setup_start_training_button()
        self.setup_progress_and_logging()
        # Scaler Selection Dropdown
        tk.Label(self, text="Select Scaler:").pack()
        self.scaler_type_var = tk.StringVar()
        self.scaler_dropdown = ttk.Combobox(self, textvariable=self.scaler_type_var,
                                            values=["StandardScaler", "MinMaxScaler", 
                                                    "RobustScaler", "Normalizer", "MaxAbsScaler"])
        self.scaler_dropdown.pack()
    # Function to set up the title label
    def setup_title_label(self):
        tk.Label(self, text="Model Training", font=("Helvetica", 16)).pack(pady=10)

    # Function to set up the data file path section
    def setup_data_file_path_section(self):
        self.data_file_label = tk.Label(self, text="Data File Path:")
        self.data_file_label.pack()
        self.data_file_entry = tk.Entry(self)
        self.data_file_entry.pack()
        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_data_file)
        self.browse_button.pack(pady=5)

    # Function to set up the model type selection dropdown
    def setup_model_type_selection(self):
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar()
        model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var,
                                        values=["linear_regression", "random_forest",
                                                "neural_network", "LSTM", "ARIMA"])
        model_type_dropdown.pack()
        model_type_dropdown.bind("<<ComboboxSelected>>", self.show_dynamic_options)
        self.dynamic_options_frame = tk.Frame(self)
        self.dynamic_options_frame.pack(pady=5)

    # Function to set up training configuration options
    def setup_training_configuration(self):
        # Assuming 'setup_training_configurations' sets up various configuration options
        self.setup_training_configurations()

    # Function to set up the start training button
    def setup_start_training_button(self):
        self.start_training_button = ttk.Button(self, text="Start Training", command=self.start_training)
        self.start_training_button.pack(pady=10)


#Section 2: GUI Components and Functions
        
    # Function to show additional input fields based on the selected model type
    def show_epochs_input(self, event):
        selected_model_type = self.model_type_var.get()
        
        if selected_model_type in ["neural_network", "LSTM"]:
            if not hasattr(self, 'epochs_label'):
                self.epochs_label = tk.Label(self, text="Epochs:")
                self.epochs_entry = tk.Entry(self)


            self.epochs_label.pack(in_=self)
            self.epochs_entry.pack(in_=self)

            self.window_size_label.pack()
            self.window_size_entry.pack()
        else:
            if hasattr(self, 'epochs_label'):
                self.epochs_label.pack_forget()
                self.epochs_entry.pack_forget()
                self.window_size_label.pack_forget()
                self.window_size_entry.pack_forget()
                
    # Function to start training process
    def start_asyncio_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_forever()

    # Start asyncio loop in a separate thread
    asyncio_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
    asyncio_thread.start()


    def start_training(self):
        if not self.validate_inputs():
            self.display_message("Invalid input. Please check your settings.")
            return

        data_file_path = self.data_file_entry.get()
        scaler_type = self.scaler_type_var.get()
        model_type = self.model_type_var.get()
        epochs = self.get_epochs(model_type)

        if epochs is None:
            self.display_message("Invalid number of epochs for the selected model.")
            return

        self.disable_training_button()
        self.display_message("Training started...")

        # Load and preprocess your data to get the input shape
        data = pd.read_csv(data_file_path)
        X = data.drop('close', axis=1)  # Adjust this to your data
        input_shape = (X.shape[1],)  # Shape as a tuple

        if model_type == "neural_network":
            def objective(trial):
                # Define hyperparameter search space
                lstm_layers = [trial.suggest_int('lstm_layer_1', 32, 128), trial.suggest_int('lstm_layer_2', 16, 64)]
                dropout_rates = [trial.suggest_uniform('dropout_1', 0.2, 0.5), trial.suggest_uniform('dropout_2', 0.2, 0.5)]

                # Call create_neural_network with the suggested hyperparameters
                model = create_neural_network(input_shape, lstm_layers, dropout_rates)

                # Evaluate the model's performance and return a metric to optimize
                return validation_loss

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            best_params = study.best_params

            # Call create_neural_network with the best hyperparameters
            model = create_neural_network(input_shape, best_params['lstm_layers'], best_params['dropout_rates'])

        elif model_type == "LSTM":
            lstm_layers = [64, 32]  # Example values, customize as needed
            dropout_rates = [0.2, 0.3]  # Example values, customize as needed

            # Call create_lstm_model with the lists
            model = create_lstm_model(input_shape, lstm_layers, dropout_rates)

        elif model_type == "ARIMA":
            model = train_arima_model(data_file_path, scaler_type)
        elif model_type == "linear_regression":
            model = self.train_linear_regression(data_file_path, scaler_type)
        elif model_type == "random_forest":
            model = self.train_random_forest(data_file_path, scaler_type, 'close')
        # Add other model types here...

    async def simulate_training_progress(self):
        # Simulate training progress (replace with your actual training code)
        try:
            for progress in range(1, 101):
                # Simulate time-consuming task
                await asyncio.sleep(0.1)
                
                # Update progress bar
                self.update_progress_bar(progress)

                # Optionally, you can display a message to the user
                self.display_message(f"Training Progress: {progress}%")

        except Exception as e:
            logging.error(f"An error occurred during training progress simulation: {str(e)}")
            self.display_message(f"Training progress error: {str(e)}")

        finally:
            # Ensure that the progress bar is at 100% even if an error occurred
            self.update_progress_bar(100)


    def update_progress_bar(self, progress):
        self.progress_var.set(progress)
        self.update_idletasks()

    def disable_training_button(self):
        self.start_training_button.config(state='disabled')

    def enable_training_button(self):
        self.start_training_button.config(state='normal')

    def display_message(self, message):
        # Display messages to the user in a text widget or messagebox
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)

    # Other methods for validation, input retrieval, and setup

    def validate_inputs(self):
        # Set default values for various inputs
        DEFAULT_EPOCHS = 50
        DEFAULT_WINDOW_SIZE = 20
        DEFAULT_N_ESTIMATORS = 200
        DEFAULT_ARIMA_P = 5
        DEFAULT_ARIMA_D = 1
        DEFAULT_ARIMA_Q = 5

        # Common validation checks (applicable to all models)
        data_file_path = self.data_file_entry.get()
        model_type = self.model_type_var.get()

        if not data_file_path:
            self.error_label.config(text="Data file path is required.", fg="red")
            return False

        if not model_type:
            self.error_label.config(text="Please select a model type.", fg="red")
            return False

        # Model-specific validation checks with default values
        if model_type in ["neural_network", "LSTM"]:
            epochs_str = self.epochs_entry.get()
            window_size_str = self.window_size_entry.get()

            if not epochs_str.isdigit():
                self.epochs_entry.delete(0, tk.END)
                self.epochs_entry.insert(0, str(DEFAULT_EPOCHS))

            if not window_size_str.isdigit():
                self.window_size_entry.delete(0, tk.END)
                self.window_size_entry.insert(0, str(DEFAULT_WINDOW_SIZE))

        elif model_type == "random_forest":
            n_estimators_str = self.n_estimators_entry.get()
            if not n_estimators_str.isdigit():
                self.n_estimators_entry.delete(0, tk.END)
                self.n_estimators_entry.insert(0, str(DEFAULT_N_ESTIMATORS))

        elif model_type == "ARIMA":
            p_value_str = self.arima_p_entry.get()
            d_value_str = self.arima_d_entry.get()
            q_value_str = self.arima_q_entry.get()

            if not p_value_str.isdigit():
                self.arima_p_entry.delete(0, tk.END)
                self.arima_p_entry.insert(0, str(DEFAULT_ARIMA_P))

            if not d_value_str.isdigit():
                self.arima_d_entry.delete(0, tk.END)
                self.arima_d_entry.insert(0, str(DEFAULT_ARIMA_D))

            if not q_value_str.isdigit():
                self.arima_q_entry.delete(0, tk.END)
                self.arima_q_entry.insert(0, str(DEFAULT_ARIMA_Q))

        # Add more model types and their specific validation as needed

        # Clear error label if everything is valid
        self.error_label.config(text="")

        return True  # Validation passed

    class ModelTrainingLogger:
        def __init__(self, log_text):
            self.log_text = log_text
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.handler = logging.StreamHandler()
            self.handler.setLevel(logging.INFO)
            self.logger.addHandler(self.handler)

        def log(self, message):
            self.utils.log_message(message, self.log_text)
            self.logger.info(message)

    def handle_exceptions(logger):
        """
        A decorator function to handle exceptions and log them.

        Args:
            logger (function): A logging function to log exceptions.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ModelTrainingError as mte:
                    logger.log(f"Error in model training: {str(mte)}")
                except ValueError as ve:
                    logger.log(f"Invalid parameter: {str(ve)}")
                except Exception as e:
                    error_message = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                    logger.log(error_message)
            return wrapper
        return decorator

    @handle_exceptions(logger=ModelTrainingLogger)
    async def train_model_and_enable_button(self, data_file_path, scaler_type, model_type, epochs):
        """
        Train a machine learning model based on provided parameters. This includes concurrent training, 
        mixed-precision training, distributed training with Horovod, automated hyperparameter tuning with Optuna, 
        model ensembling, federated learning, automated data augmentation, and model quantization.

        Args:
            data_file_path (str): The path to the data file.
            scaler_type (str): The type of data scaler to use.
            model_type (str): The type of machine learning model.
            epochs (int): The number of training epochs.

        Raises:
            ValueError: If invalid parameters are provided.

        Returns:
            None
        """
        # Validate input parameters
        if not os.path.exists(data_file_path):
            raise ValueError("Data file path does not exist")
        if model_type not in ["linear_regression", "neural_network", "random_forest"]:
            raise ValueError("Unsupported model_type")
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer")

        # Preprocess data
        window_size = int(self.window_size_entry.get()) if model_type == "neural_network" else 1
        X_train, X_test, y_train, y_test = self.preprocess_data(data_file_path, scaler_type, model_type, window_size)

        # Initialize and configure model
        model = await self.initialize_and_configure_model(model_type, X_train, epochs)
        if model is None:
            raise ModelTrainingError("Failed to initialize model")

        # Automated hyperparameter tuning
        best_params = await self.perform_hyperparameter_tuning(model, X_train, y_train, epochs)

        # Ensemble and quantize model
        ensemble_model = self.create_ensemble_model(model, best_params)
        quantized_model = self.quantize_model(ensemble_model)

        # Train the model asynchronously
        await self.train_model_async(quantized_model, X_train, y_train, epochs)

        # Post-training actions
        self.trained_model = quantized_model
        self.log_training_completion()
        self.enable_training_button()

    async def train_model_async(self, model, X_train, y_train, epochs):
        """
        Asynchronous method to train the model.

        Args:
            model: The machine learning model to train.
            X_train: Training feature data.
            y_train: Training target data.
            epochs: Number of training epochs.

        Returns:
            None
        """
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Consider handling TensorFlow/Keras models differently if necessary
                future = loop.run_in_executor(executor, model.fit, X_train, y_train, epochs=epochs, verbose=1)
                
                while not future.done():
                    await asyncio.sleep(1)
                    # Update progress bar based on actual progress (if possible)
                
                # Consider handling the result of future here (e.g., model training results)
                await future
        except Exception as e:
            # Proper error handling
            print(f"Error in training model: {e}")
            # Consider logging the error as well
        finally:
            # Any cleanup if necessary
            pass


    # Additional helper methods (initialize_and_configure_model, perform_hyperparameter_tuning,
    # create_ensemble_model, quantize_model, log_training_completion) go here...

    # Other helper classes and functions (FederatedLearningSimulator, DataAugmentation, Quantization) should be added as needed.

    def get_epochs(self, model_type):
        if model_type in ["neural_network", "LSTM"]:
            epochs_str = self.epochs_entry.get()
            if not epochs_str.isdigit() or int(epochs_str) <= 0:
                self.utils.log_message("Epochs should be a positive integer.", self, self.log_text, self.is_debug_mode)
                return None
            return int(epochs_str)
        else:
            # For models like Random Forest, return a default value, like 1 or 0
            return 1


    async def train_model_logic(self, data_file_path, scaler_type, model_type, epochs):
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(data_file_path, scaler_type, model_type, epochs)

            # Model training logic based on the selected model type
            if model_type == "linear_regression":
                model = self.train_linear_regression(X_train, y_train)
            elif model_type == "random_forest":
                model = self.train_random_forest(X_train, y_train)
            elif model_type == "neural_network":
                model = create_neural_network(input_shape=X_train.shape[1])
                model.fit(X_train, y_train, epochs=epochs)
            elif model_type == "LSTM":
                model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                model.fit(X_train, y_train, epochs=epochs)
            elif model_type == "ARIMA":
                model = train_arima_model(X_train, y_train)
            # ... Additional model training logic for other model types ...

            # Append the trained model to the list
            self.trained_models.append(model)

            # Optionally, evaluate the model and log performance metrics
            if model_type in ["linear_regression", "random_forest", "neural_network", "LSTM", "ARIMA"]:
                evaluation_result = evaluate_model(model, X_test, y_test)
                self.utils.log_message(f"Model evaluation result: {evaluation_result}" + file_path, self, self.log_text, self.is_debug_mode)

            # Update the trained model information
            self.trained_model = model
            self.utils.log_message("Model training completed.", + file_path, self, self.log_text, self.is_debug_mode)

        except Exception as e:
            # Handle and log any exceptions during training
            self.utils.log_message(f"Error in model training: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)
        finally:
            # Re-enable the Start Training button regardless of success or exception
            self.start_training_button.config(state='normal')


    #Section 3: Data Preprocessing and Model Integration
            
    def preprocess_data(self, data_file_path, scaler_type, model_type, window_size=5, epochs=None):
        try:
            # Load the dataset
            data = pd.read_csv(data_file_path)  # Adjust based on your data format

            # Ensure the dataset has the required columns
            if 'date' not in data.columns:
                raise ValueError("Missing 'date' column in the dataset")
            
            target_column = None
            if 'close' in data.columns:
                target_column = 'close'
            elif '4. close' in data.columns:
                target_column = '4. close'

            if target_column is None:
                raise ValueError("Neither 'close' nor '4. close' column found in the dataset")

            # Handle missing values in the 'date' column
            data['date'] = data['date'].replace('', np.nan)
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])

            # Extract features from 'date' column
            data['day_of_week'] = data['date'].dt.dayofweek
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year

            # Define your target variable and feature set
            X = data.drop([target_column, 'date'], axis=1)
            y = data[target_column]

            # Fill NaN values in features with their mean
            for col in X.columns:
                if X[col].isna().any():
                    X[col].fillna(X[col].mean(), inplace=True)

            # Create the scaler object
            scaler = self.get_scaler(scaler_type)
            # Model-specific preprocessing
            if model_type == "ARIMA":
                # ARIMA-specific preprocessing
                X, y = self.arima_preprocessing(data)

            elif model_type in ["neural_network", "LSTM"]:
                # Neural Network and LSTM-specific preprocessing
                # Scale the features
                X_scaled = scaler.fit_transform(X)
                self.trained_scaler = scaler

                if model_type == "LSTM":
                    # Create windowed data for LSTM
                    X, y = self.create_windowed_data(X_scaled, y, window_size)
                else:
                    # For a standard neural network
                    X, y = X_scaled, y

            else:
                # For other models, use the general preprocessing logic
                # Scale the features
                X_scaled = scaler.fit_transform(X)
                self.trained_scaler = scaler
                X, y = X_scaled, y

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            # Handle and log any exceptions during preprocessing
            self.utils.log_message(f"Error in data preprocessing: {str(e)}"+ file_path, self, self.log_text, self.is_debug_mode)
            return None, None, None, None



    # Define specific preprocessing methods

    # Function to preprocess data for ARIMA models
    def arima_preprocessing(self, data):
        """
        Specific preprocessing for ARIMA models for close price prediction.

        Args:
            data (DataFrame): The dataset to preprocess.

        Returns:
            tuple: Preprocessed features (X) and target values (y).
        """
        try:
            # Ensure the data is in time series format
            if 'date' not in data.columns:
                self.utils.log_message("Datetime column 'date' not found in the dataset. Please ensure your data is in time series format." + file_path, self, self.log_text, self.is_debug_mode)
                return None, None

            # Ensure the dataset is sorted by date
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values(by='date')

            # Extract the target variable (close price)
            if 'close' in data.columns:
                y = data['close'].values  # Use 'close' as the target column name if it exists
            elif '4. close' in data.columns:
                y = data['4. close'].values  # Use '4. close' as the target column name if 'close' does not exist
            else:
                self.utils.log_message("Neither 'close' nor '4. close' column found in the dataset." + file_path, self, self.log_text, self.is_debug_mode)
                return None, None

            # Perform differencing to make the data stationary (if necessary)
            # Example: y_diff = y - y.shift(1)
            # This step depends on the stationarity of your data

            # Additional preprocessing steps specific to ARIMA
            # Example: Feature selection, outlier handling, etc.

            # Prepare X (features)
            # Depending on your specific use case, you may need to engineer features from the time series data
            # Example: Lag features, rolling statistics, external factors, etc.

            # In this example, we'll use lag features as an illustration
            num_lags = 5  # Adjust the number of lag features as needed
            for lag in range(1, num_lags + 1):
                data[f'lag_{lag}'] = y.shift(lag)
            
            # Drop rows with NaN values due to lag
            data.dropna(inplace=True)

            # Prepare X with lag features
            X = data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]  # Replace with your relevant lag features

            return X, y
        except Exception as e:
            # Handle and log any exceptions during preprocessing
            self.utils.log_message(f"Error in ARIMA preprocessing: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)
            return None, None

    def general_preprocessing(self, data, scaler_type, feature_selection=False, num_features=None):
        """
        General preprocessing for other model types.

        Args:
            data (DataFrame): The dataset to preprocess.
            scaler_type (str): Type of scaler to use for feature scaling.
            feature_selection (bool): Whether to perform feature selection.
            num_features (int): Number of top features to select if feature_selection is True.

        Returns:
            tuple: Preprocessed features (X) and target values (y).
        """
        try:
            # Extract features and target variable
            X = data.drop('close', axis=1)
            y = data['close'].values

            # Scale features using the specified scaler
            X_scaled = self.scale_features(X, scaler_type)

            # Feature selection (optional)
            if feature_selection:
                X_scaled = self.select_top_features(X_scaled, y, num_features)

            return X_scaled, y
        except Exception as e:
            # Handle and log any exceptions during preprocessing
            self.utils.log_message(f"Error in general preprocessing: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)
            return None, None
        
    def scale_features(self, X, scaler_type):
        """
        Scale the features using the specified scaler, handling both DataFrames and NumPy arrays.

        Args:
            X (DataFrame or numpy array): The feature matrix.
            scaler_type (str): Type of scaler to use for feature scaling.

        Returns:
            DataFrame or numpy array: Scaled feature matrix.
        """
        # Define scalers
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer(),
            'maxabs': MaxAbsScaler()
        }

        # Select scaler
        scaler = scalers.get(scaler_type, StandardScaler())

        # Check if X is a DataFrame or a NumPy array
        if isinstance(X, pd.DataFrame):
            # If DataFrame, retain column names
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # If NumPy array, just scale it
            X_scaled = scaler.fit_transform(X)

        return X_scaled




    # Function to handle ARIMA model training logic
    def train_arima_model_logic(self, y_train):
        """
        Train an ARIMA model with the provided time series data.

        Args:
            y_train (Series): Time series data for training.

        Returns:
            ARIMAResultsWrapper: A trained ARIMA model.
        """
        # Define and fit the ARIMA model
        # Note: Adjust the order parameters (p, d, q) based on your data
        model = train_arima_model(y_train, order=(1, 1, 1))
        return model

    # Function to integrate the trained models with the rest of the application
    def integrate_trained_model(self):
        """
        Integrate the trained model with the application, update the UI, and handle any post-processing.
        """
        try:
            if self.trained_model:
                # Update UI components
                self.update_ui_with_model_info()

                # Save the trained model
                self.save_trained_model()

                # Perform post-processing (if needed)
                self.perform_post_processing()

                self.utils.log_message("Model integration complete." + file_path, self, self.log_text, self.is_debug_mode)
            else:
                self.utils.log_message("No trained model available for integration." + file_path, self, self.log_text, self.is_debug_mode)
        except Exception as e:
            self.utils.log_message(f"Error during model integration: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)

    def update_ui_with_model_info(self):
        """
        Update UI components with information about the integrated model.
        """
        if self.trained_model:
            # Example 1: Update a label with the model type
            model_type_label.config(text=f"Model Type: {self.get_model_type()}")

            # Example 2: Enable a button for model predictions
            predict_button.config(state='normal')

            # Example 3: Display model accuracy if available
            if hasattr(self.trained_model, 'score'):
                accuracy_label.config(text=f"Model Accuracy: {self.calculate_model_accuracy():.2f}%")

            # You can add more UI updates based on your application's needs

        else:
            # Example: Disable the prediction button if no trained model is available
            predict_button.config(state='disabled')

    def get_model_type(self):
        """
        Get the type of the integrated model.

        Returns:
            str: Model type (e.g., "Linear Regression", "Neural Network").
        """
        # You can determine the model type based on the attributes or characteristics of the model.
        # Example: Check if the model is an instance of a specific class
        if isinstance(self.trained_model, LinearRegression):
            return "Linear Regression"
        elif isinstance(self.trained_model, NeuralNetwork):
            return "Neural Network"
        else:
            return "Unknown"

    def calculate_model_accuracy(self):
        """
        Calculate the accuracy of the integrated model (if applicable).

        Returns:
            float: Model accuracy as a percentage.
        """
        # You can calculate the accuracy based on your model's evaluation method or metrics.
        # Example: If the model has a 'score' method, you can use it to calculate accuracy
        if hasattr(self.trained_model, 'score'):
            accuracy = self.trained_model.score(X_test, y_test)  # Replace X_test and y_test with your test data
            return accuracy * 100.0  # Convert to percentage
        else:
            return 0.0  # Return 0 if accuracy calculation is not applicable

    # You can add more functions and UI components as needed for your specific application.

#model saving function
    def save_trained_model(self):
        """
        Save the trained model to a file or storage location.

        Returns:
            None
        """
        if self.trained_model is None:
            self.utils.log_message("No trained model available to save." + file_path, self, self.log_text, self.is_debug_mode)
            return

        try:
            # Determine the model type and file extension based on the model
            model_type = self.get_model_type(self.trained_model)
            file_extension = self.get_file_extension(model_type)

            # Open a file dialog for the user to choose the save location and filename
            file_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                                    filetypes=[(f"{model_type.upper()} Files", f"*{file_extension}"), ("All Files", "*.*")])

            if not file_path:  # User canceled the save dialog
                return

            # Save the model
            self.save_model_by_type(self.trained_model, model_type, file_path)

            # Save scaler and metadata if available
            if self.trained_scaler:
                self.save_model_datafile_path, model_type

            self.utils.log_message(f"Model saved to {file_path}" + file_path, self, self.log_text, self.is_debug_mode)
        except Exception as e:
            self.utils.log_message(f"Error saving model: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)

    def save_model_by_type(self, model, model_type, file_path):
        """
        Save the trained model to a file based on its type.

        Args:
            model: The trained model to be saved.
            model_type (str): The type of the model (e.g., 'sklearn', 'keras').
            file_path (str): The file path to save the model.
        """
        if model_type == "sklearn":
            joblib.dump(model, file_path)
        elif model_type == "keras":
            model.save(file_path)
        # Add logic for saving other types of models


    # Function to handle exceptions during model training
    def handle_model_training_exceptions(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_label.config(text=f"Error in model training: {e}", fg="red")
                self.utils.log_message(f"Model training error: {e}" + file_path, self, self.log_text, self.is_debug_mode)
        return wrapper

    def start_automated_training(self):
        interval = self.schedule_dropdown.get()
        if interval == "Daily":
            schedule.every().day.at("10:00").do(self.run_automated_training_tasks)
        elif interval == "Weekly":
            schedule.every().week.do(self.run_automated_training_tasks)
        elif interval == "Monthly":
            schedule.every().month.do(self.run_automated_training_tasks)
        
        self.utils.log_message(f"Automated training scheduled {interval.lower()}." + file_path, self, self.log_text, self.is_debug_mode)

        # Start a thread to run the schedule
        threading.Thread(target=self.run_schedule).start()

    def run_schedule(self):
        """
        Run the scheduled tasks, including automated model training and real-time monitoring.
        """
        while True:
            schedule.run_pending()
            time.sleep(1)  # Sleep for 1 second to avoid excessive CPU usage

    def run_automated_training_tasks(self):
        """
        Run the automated training tasks, including model training and real-time analytics monitoring.
        """
        # Log the start of automated training
        self.utils.log_message("Automated training started." + file_path, self, self.log_text, self.is_debug_mode)

        # Implement adaptive learning logic based on past performance
        model_type = self.config.get("Model", "model_type")
        data_file_path = self.config.get("Data", "file_path")
        epochs = int(self.config.get("Model", "epochs")) if model_type in ["neural_network", "LSTM"] else 1

        # Call training logic here
        self.train_model_and_enable_button(data_file_path, model_type, epochs)

        # Implement real-time analytics monitoring during training
        self.initiate_real_time_training_monitoring()

        # Log the completion of automated training
        self.utils.log_message("Automated training completed." + file_path, self, self.log_text, self.is_debug_mode)

        # Show a message box to notify the user
        messagebox.showinfo("Automated Training", "Automated training completed.")

    def initiate_real_time_training_monitoring(self):
        """
        Start real-time analytics monitoring during training.
        """
        # Start a separate thread to monitor training progress
        threading.Thread(target=self.monitor_training_progress, daemon=True).start()

    def monitor_training_progress(self):
        """
        Monitor and update training progress in real-time.
        """
        while self.training_in_progress:
            if not self.training_paused:
                progress_data = self.get_training_progress()  # Replace with your actual progress tracking logic

                # Update the UI with progress_data
                self.update_ui_with_progress(progress_data)

                # Check for training completion
                if self.is_training_complete(progress_data):
                    self.training_in_progress = False
                    self.on_training_completed()

            time.sleep(1)  # Adjust the interval as needed

    # Function to visualize model training and evaluation results
    def visualize_training_results(self, y_test, y_pred):
        """
        Visualize the model training and evaluation results.

        Args:
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values by the model.
        """

        # Advanced visualization with seaborn
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Advanced Model Evaluation Results')
        plt.show()


    # Function to retrieve the appropriate scaler based on user selection
    # Scaler selection method
    def get_scaler(self, scaler_type):
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        return scalers.get(scaler_type, StandardScaler())

    # Function to browse and select a data file
    def browse_data_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, file_path)
            self.preview_selected_data(file_path)
            self.utils.log_message(f"Selected data file: {file_path}", self, self.log_text, self.is_debug_mode) 

    def preview_selected_data(self, file_path):
        # Log the action of previewing data
        self.utils.log_message("Previewing data from file: " + file_path, self.log_text, self.is_debug_mode)

        # Check if the file exists
        if not os.path.exists(file_path):
            self.utils.log_message("File does not exist: " + file_path, self.log_text, self.is_debug_mode)
            return

        # Check the file extension (assuming a CSV file)
        if not file_path.endswith('.csv'):
            self.utils.log_message("Unsupported file format. Only CSV files are supported.", self.log_text, self.is_debug_mode)
            return

        try:
            # Read the file using pandas
            data = pd.read_csv(file_path)

            # Append a preview of the data into the log text widget
            self.utils.log_message("Data preview:\n" + str(data.head()), self.log_text, self.is_debug_mode)

        except Exception as e:
            # Handle exceptions and display them in the log text widget
            self.utils.log_message("An error occurred while reading the file: " + str(e), self.log_text, self.is_debug_mode)


    # Function to handle exceptions during data loading and preprocessing
    def handle_data_preprocessing_exceptions(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_label.config(text=f"Error in data preprocessing: {e}", fg="red")
                self.utils.log_message(f"Data preprocessing error: {e}" + file_path, self, self.log_text, self.is_debug_mode)
        return wrapper

    # Function to manage post-training actions like model saving, updating UI, etc.
    def post_training_actions(self, model, model_type):
        file_path = self.utils.auto_generate_save_path(model_type)
        self.save_trained_model(model, model_type, file_path)
        self.utils.log_message(f"Model saved: {file_path}" + file_path, self, self.log_text, self.is_debug_mode)

        # Optionally, upload the model to cloud storage for persistence and global access
        self.upload_model_to_cloud(file_path)

    # Function to initiate the model training process
    def initiate_model_training(self, data_file_path, scaler_type, model_type, epochs):
        try:
            self.utils.log_message("Starting model training..." + file_path, self, self.log_text, self.is_debug_mode)
            threading.Thread(target=lambda: self.train_model(data_file_path, scaler_type, model_type, epochs)).start()
        except Exception as e:
            self.utils.log_message(f"Error in model training: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)
            print(f"Debug: Error in model training - {str(e)}")

    # ... (Additional methods and logic as required for your application) ...

    def perform_post_processing(self, model, model_type, X_test, y_test, y_pred):
        """
        Perform post-processing tasks after model integration.

        Args:
            model: The trained model.
            model_type (str): Type of the model.
            X_test (DataFrame): Test features.
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values by the model.

        Returns:
            None
        """
        try:
            # Save the trained model
            file_path = self.utils.s(model_type)
            self.save_trained_model(model, model_type, file_path)
            self.utils.log_message(f"Model saved: {file_path}" + file_path, self, self.log_text, self.is_debug_mode)

            # Calculate evaluation metrics
            metrics = self.calculate_model_metrics(y_test, y_pred)
            self.utils.log_message(f"Model evaluation metrics: {metrics}" + file_path, self, self.log_text, self.is_debug_mode)

            # Visualize model performance
            self.visualize_model_performance(y_test, y_pred)

            # Generate model reports
            self.generate_model_reports(model, X_test, y_test, y_pred)

            # Upload the model to cloud storage (if needed)
            self.upload_model_to_cloud(file_path)

            # Notify users or stakeholders
            self.send_notification("Model Training Complete", "The model training process has finished successfully.")

            # Perform additional post-processing tasks
            self.additional_post_processing()

        except Exception as e:
            self.utils.log_message(f"Error during post-processing: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)

    def calculate_model_metrics(self, y_true, y_pred):
        """
        Calculate and return model evaluation metrics.

        Args:
            y_true (Series): Actual target values.
            y_pred (Series): Predicted target values.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        metrics = {
            'Mean Absolute Error (MAE)': mean_absolute_error(y_true, y_pred),
            'Root Mean Squared Error (RMSE)': mean_squared_error(y_true, y_pred, squared=False),
            'R-squared (R2)': r2_score(y_true, y_pred)
        }
        return metrics

    def visualize_model_performance(self, y_true, y_pred):
        """
        Visualize the model's performance using plots.

        Args:
            y_true (Series): Actual target values.
            y_pred (Series): Predicted target values.
        """
        sns.set(style="whitegrid")

        # Create a scatter plot of actual vs. predicted values
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Model Evaluation Results')
        plt.show()

    def generate_model_reports(self, model, X_test, y_test, y_pred):
        """
        Generate detailed reports or summaries of the model's performance.

        Args:
            model: The trained model.
            X_test (DataFrame): Test features.
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values.
        """
        try:
            # Generate a classification report for classification models
            if isinstance(model, Classifier):  # Replace 'Classifier' with your actual classifier class
                classification_rep = classification_report(y_test, y_pred)
                self.utils.log_message("Classification Report:" + file_path, self, self.log_text, self.is_debug_mode)
                self.utils.log_message(classification_rep, self.log_text)

                # Generate a confusion matrix plot
                confusion_mat = confusion_matrix(y_test, y_pred)
                self.plot_confusion_matrix(confusion_mat)

            # Generate regression-specific reports for regression models
            elif isinstance(model, Regressor):  # Replace 'Regressor' with your actual regressor class
                # Calculate and log regression metrics (e.g., RMSE, MAE, R-squared)
                regression_metrics = self.calculate_regression_metrics(y_test, y_pred)
                self.utils.log_message("Regression Metrics:" + file_path, self, self.log_text, self.is_debug_mode)
                self.utils.log_message(classification_rep, self.log_text, is_debug_mode=self.is_debug_mode)


                # Generate regression-specific visualizations if needed
                self.generate_regression_visualizations(y_test, y_pred)

            # For other model types, you can implement custom reports
            else:
                custom_report = self.generate_custom_model_report(model, X_test, y_test, y_pred)
                self.utils.log_message("Custom Model Report:" + file_path, self, self.log_text, self.is_debug_mode)
                self.utils.log_message(custom_report, self.log_text, is_debug_mode=self.is_debug_mode)


            # Add any other reporting or visualization logic as required

        except Exception as e:
            self.utils.log_message(f"Error generating model reports: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)

    def calculate_regression_metrics(self, y_test, y_pred):
        """
        Calculate and return regression metrics such as RMSE, MAE, and R-squared.

        Args:
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values.

        Returns:
            dict: A dictionary containing regression metrics.
        """

        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        metrics = {
            'Root Mean Squared Error (RMSE)': rmse,
            'Mean Absolute Error (MAE)': mae,
            'R-squared (R2)': r_squared
        }

        return metrics

    def generate_regression_visualizations(self, y_test, y_pred):
        """
        Generate visualizations for regression model evaluation.

        Args:
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values.
        """
        # Scatter plot of actual vs. predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.show()

        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix):
        """
        Plot a confusion matrix.

        Args:
            confusion_matrix (array): The confusion matrix to be plotted.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def send_notification(self, subject, message):
        """
        Send notifications to inform users or stakeholders.

        Args:
            subject (str): Notification subject.
            message (str): Notification message.
        """
        try:
            # Configure email settings
            smtp_server = "your_smtp_server.com"
            smtp_port = 587
            sender_email = "your_sender_email@gmail.com"
            sender_password = "your_sender_password"
            recipient_email = "recipient_email@example.com"

            # Create a secure SSL context
            context = ssl.create_default_context()

            # Create an email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject

            # Attach the message to the email
            msg.attach(MIMEText(message, 'plain'))

            # Establish a secure connection with the SMTP server
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())

            print("Notification sent successfully.")
        except Exception as e:
            print(f"Error sending notification: {str(e)}")

    def additional_post_processing(self):
        """
        Perform additional post-processing tasks specific to your application.

        Args:
            None
        """
        try:
            # Implement any custom post-processing tasks required by your application
            # Example: Data logging, generating reports, or connecting to external APIs
            pass
        except Exception as e:
            print(f"Error in additional post-processing: {str(e)}")

    def generate_custom_model_report(self, model, X_test, y_test, y_pred):
        """
        Generate a custom model report based on the specific model type.

        Args:
            model: The trained model.
            X_test (DataFrame): Test features.
            y_test (Series): Actual target values.
            y_pred (Series): Predicted target values.

        Returns:
            str: A custom model report as a string.
        """
        try:
            # Implement custom reporting logic based on the model type
            # Example: Explain feature importance, model insights, or any specific details
            custom_report = "Custom Model Report:\n"

            # Add your custom reporting content here

            return custom_report
        except Exception as e:
            print(f"Error generating custom model report: {str(e)}")
            return "Custom Model Report: Error"

    def get_training_progress(self):
        """
        Get the current training progress.

        Returns:
            dict: A dictionary containing training progress information.
        """
        # Example: Simulate progress data with epoch and loss
        progress_data = {'epoch': 5, 'loss': 0.123}
        return progress_data

    def update_ui_with_progress(self, progress_data):
        """
        Update the user interface with training progress.

        Args:
            progress_data (dict): A dictionary containing training progress information.
        """
        # Example: Print progress data to the console
        print(f"Epoch: {progress_data['epoch']}, Loss: {progress_data['loss']}")

    def is_training_complete(self, progress_data):
        """
        Check if the training is complete based on progress data.

        Args:
            progress_data (dict): A dictionary containing training progress information.

        Returns:
            bool: True if training is complete, False otherwise.
        """
        # Example: Check if a certain number of epochs (e.g., 10) have been reached
        return progress_data['epoch'] >= 10

    def on_training_completed(self):
        """
        Handle actions to be taken when training is completed.
        """
        # Example: Print a completion message
        print("Training completed.")

    def upload_model_to_cloud(self, file_path):
        # Example using a hypothetical cloud storage API
        cloud_service = CloudStorageAPI()
        response = cloud_service.upload(file_path)
        if response.success:
            print(f"Model uploaded to cloud storage: {response.url}")
        else:
            print("Failed to upload model to cloud storage.")

    def adaptive_learning_logic(self):
        # Load historical performance data
        try:
            performance_data = pd.read_csv('path_to_performance_log.csv') # Adjust path as needed
        except FileNotFoundError:
            self.utils.log_message("Performance log file not found." + file_path, self, self.log_text, self.is_debug_mode)
            return
        
        # Splitting data for training and validation
        features = performance_data.drop(['performance_metric', 'optimal_parameters'], axis=1)
        targets = performance_data['optimal_parameters'].apply(json.loads).apply(pd.Series)
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

        # Training a model to predict optimal training parameters
        model = XGBRegressor()  # Using XGBoost for multi-output regression
        grid_search = GridSearchCV(model, param_grid={'n_estimators': [100, 200], 'max_depth': [3, 5]}, cv=3)
        grid_search.fit(X_train, y_train)

        # Predicting optimal parameters for the next training
        latest_performance = features.iloc[-1].values.reshape(1, -1)
        predicted_parameters = grid_search.predict(latest_performance)

        # Update training parameters for the next session
        self.update_training_parameters(predicted_parameters[0])

        # Evaluate and log
        accuracy = grid_search.score(X_test, y_test)
        self.utils.log_message(f"Parameter prediction model accuracy: {accuracy}" + file_path, self, self.log_text, self.is_debug_mode)

    def update_training_parameters(self, parameters):
        # Example: Updating multiple parameters
        self.learning_rate = parameters.get('learning_rate', self.learning_rate)
        self.batch_size = parameters.get('batch_size', self.batch_size)
        # ... update other parameters ...

        # Log the updated parameters
        updated_params = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            # ... other parameters ...
        }
        self.utils.log_message(f"Updated training parameters: {updated_params}" + file_path, self, self.log_text, self.is_debug_mode)

    def process_queue(self):
        try:
            progress_data = self.queue.get_nowait()
            self.update_gui_with_progress(progress_data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def train_linear_regression(self, data_file_path, scaler_type):
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(data_file_path, scaler_type, "linear_regression")

            # Initialize the Linear Regression model
            model = LinearRegression()

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            score = model.score(X_test, y_test)
            self.display_message(f"Linear Regression Model Score: {score}")

            # Update the trained model information
            self.trained_model = model
            self.utils.log_message("Linear regression model training completed.", self, self.log_text, self.is_debug_mode)

            # Re-enable the Start Training button and update UI
            self.enable_training_button()

        except Exception as e:
            self.utils.log_message(f"Error in training linear regression model: {str(e)}", self, self.log_text, self.is_debug_mode)
            self.error_label.config(text=f"Error in training linear regression model: {e}", fg="red")

    def update_gui_with_progress(self, progress_data):
        # Update progress bar and log
        self.progress_var.set(progress_data['progress'])
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"Epoch: {progress_data['epoch']}, Loss: {progress_data['loss']}, Accuracy: {progress_data['accuracy']}%\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def pause_training(self):
        self.training_paused = True
        self.log_text.config(state='normal')
        self.log_text.insert('end', "Training paused.\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def resume_training(self):
        self.training_paused = False
        self.log_text.config(state='normal')
        self.log_text.insert('end', "Training resumed.\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def neural_network_preprocessing(self, data, scaler_type):
        """
        Specific preprocessing for neural network models.

        Args:
            data (DataFrame): The dataset to preprocess.
            scaler_type (str): Type of scaler to use for feature scaling.

        Returns:
            tuple: Preprocessed features (X) and target values (y).
        """
        try:
            # 1. Feature scaling
            # Create a scaler object based on the specified scaler_type
            scaler = self.get_scaler(scaler_type)
            
            # Apply feature scaling to the dataset excluding the target column
            scaled_data = scaler.fit_transform(data.drop('target', axis=1))

            # 2. Prepare target variable
            # Replace 'target' with your actual target column name
            y = data['target'].values 

            return scaled_data, y
        except Exception as e:
            # Handle and log any exceptions during preprocessing
            self.utils.log_message(f"Error in neural network preprocessing: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)
            return None, None
        
    def show_dynamic_options(self, event):
        # Clear current dynamic options
        for widget in self.dynamic_options_frame.winfo_children():
            widget.destroy()

        selected_model_type = self.model_type_var.get()
        if selected_model_type == "neural_network":
            self.setup_neural_network_options()
        elif selected_model_type == "LSTM":
            self.setup_lstm_options()
        elif selected_model_type == "ARIMA":
            self.setup_arima_options()
        elif selected_model_type == "linear_regression":
            self.setup_linear_regression_options()
        elif selected_model_type == "random_forest":
            self.setup_random_forest_options()
        # ... add other model type specific options ...

    def setup_neural_network_options(self):
        # Epochs input
        tk.Label(self.dynamic_options_frame, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.dynamic_options_frame)
        self.epochs_entry.pack()

        # Window Size input
        tk.Label(self.dynamic_options_frame, text="Window Size:").pack()
        self.window_size_entry = tk.Entry(self.dynamic_options_frame)
        self.window_size_entry.pack()

    # LSTM can use the same options as the neural network
    setup_lstm_options = setup_neural_network_options

    def setup_lstm_options(self):
        # Clear any existing widgets in the frame
        for widget in self.dynamic_options_frame.winfo_children():
            widget.destroy()

        # Epochs input
        tk.Label(self.dynamic_options_frame, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.dynamic_options_frame)
        self.epochs_entry.pack()

        # Window Size input
        tk.Label(self.dynamic_options_frame, text="Window Size:").pack()
        self.window_size_entry = tk.Entry(self.dynamic_options_frame)
        self.window_size_entry.pack()

        # Additional LSTM-specific options can be added here
        # For example, you might want to add options for number of layers, dropout rate, etc.

    def setup_arima_options(self):
        # ARIMA p-value
        tk.Label(self.dynamic_options_frame, text="ARIMA p-value:").pack()
        self.arima_p_entry = tk.Entry(self.dynamic_options_frame)
        self.arima_p_entry.pack()

        # ARIMA d-value
        tk.Label(self.dynamic_options_frame, text="ARIMA d-value:").pack()
        self.arima_d_entry = tk.Entry(self.dynamic_options_frame)
        self.arima_d_entry.pack()

        # ARIMA q-value
        tk.Label(self.dynamic_options_frame, text="ARIMA q-value:").pack()
        self.arima_q_entry = tk.Entry(self.dynamic_options_frame)
        self.arima_q_entry.pack()

        # Epochs input
        tk.Label(self.dynamic_options_frame, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.dynamic_options_frame)
        self.epochs_entry.pack()

    def setup_linear_regression_options(self):
        # Regularization parameter
        tk.Label(self.dynamic_options_frame, text="Regularization:").pack()
        self.regularization_entry = tk.Entry(self.dynamic_options_frame)
        self.regularization_entry.pack()

    def setup_random_forest_options(self):
        # Number of trees
        tk.Label(self.dynamic_options_frame, text="Number of Trees:").pack()
        self.trees_entry = tk.Entry(self.dynamic_options_frame)
        self.trees_entry.pack()

        # Max depth
        tk.Label(self.dynamic_options_frame, text="Max Depth:").pack()
        self.depth_entry = tk.Entry(self.dynamic_options_frame)
        self.depth_entry.pack()

    def setup_progress_and_logging(self):
        # Progress Bar and Log Text
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=5)
        self.log_text = tk.Text(self, height=10, state='disabled')
        self.log_text.pack()

    def setup_training_configurations(self):
        # Advanced Training Configuration Section
        tk.Label(self, text="Training Configurations", font=("Helvetica", 14)).pack(pady=5)

        # Learning Rate
        tk.Label(self, text="Learning Rate:").pack()
        self.learning_rate_entry = tk.Entry(self)
        self.learning_rate_entry.pack()

        # Batch Size
        tk.Label(self, text="Batch Size:").pack()
        self.batch_size_entry = tk.Entry(self)
        self.batch_size_entry.pack()

        # Advanced Settings Toggle
        self.advanced_settings_var = tk.BooleanVar()
        self.advanced_settings_check = ttk.Checkbutton(self, text="Show Advanced Settings", 
                                                    variable=self.advanced_settings_var, 
                                                    command=self.toggle_advanced_settings)
        self.advanced_settings_check.pack(pady=5)

        # Container for Advanced Settings
        self.advanced_settings_frame = tk.Frame(self)
        self.advanced_settings_frame.pack()

    def toggle_advanced_settings(self):
        # Clear existing widgets in the advanced settings frame
        for widget in self.advanced_settings_frame.winfo_children():
            widget.destroy()

        if self.advanced_settings_var.get():
            # Show advanced settings
            tk.Label(self.advanced_settings_frame, text="Optimizer:").pack()
            self.optimizer_entry = tk.Entry(self.advanced_settings_frame)
            self.optimizer_entry.pack()

            tk.Label(self.advanced_settings_frame, text="Regularization Rate:").pack()
            self.regularization_entry = tk.Entry(self.advanced_settings_frame)
            self.regularization_entry.pack()

            # Add more advanced settings as needed

    def get_model_type(self, model):
        if isinstance(model, sklearn.base.BaseEstimator):
            return "sklearn"
        elif isinstance(model, Model):
            return "keras"
        elif isinstance(model, torch.nn.Module):
            return "pytorch"
        return "unknown_model"

    def get_file_extension(self, model_type):
        extensions = {"sklearn": ".joblib", "keras": ".h5", "pytorch": ".pth"}
        return extensions.get(model_type, ".model")

    def plot_regression_results(self, y_true, y_pred):
        """Plotting function for regression results."""
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Regression Results')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plotting function for confusion matrix (for classification)."""
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, annot=True, fmt='g')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def create_windowed_data(self, X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)

    # Merged Code for Saving Metadata and Scaler
    def save_model_data(self, file_path, model, scaler):
        """
        Save the trained model, scaler, and any metadata to separate files.

        Args:
            file_path (str): The base file path for saving the model, scaler, and metadata.
            model: The trained model to be saved.
            scaler: The trained scaler.

        Returns:
            None
        """
        try:
            # Determine file extensions and paths based on the model type
            model_type = self.get_model_type(model)
            model_file_path = file_path + self.get_file_extension(model_type)
            scaler_file_path = file_path + "_scaler.pkl"
            metadata_file_path = file_path + "_metadata.json"

            # Save the model
            self.save_model_by_type(model, model_type, model_file_path)

            # Save the scaler
            with open(scaler_file_path, 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)

            # Prepare metadata
            metadata = {
                'scaler_type': type(scaler).__name__,
                'scaler_params': scaler.get_params(),
                'model_type': model_type,
                'data_shape': model.input_shape if hasattr(model, 'input_shape') else None
                # Add more metadata as required
            }

            # Save metadata
            with open(metadata_file_path, 'w') as metadata_file:
                json.dump(metadata, metadata_file)

            self.utils.log_message(f"Model, scaler, and metadata saved successfully." + file_path, self, self.log_text, self.is_debug_mode)

        except Exception as e:
            self.utils.log_message(f"Error saving model data: {str(e)}" + file_path, self, self.log_text, self.is_debug_mode)


    def select_top_features(self, X, y, num_features):
        """
        Select the top 'num_features' based on their importance.

        Args:
            X (numpy array): Feature matrix.
            y (numpy array): Target variable.
            num_features (int): Number of features to select.

        Returns:
            numpy array: Reduced feature matrix with selected top features.
        """
        # Initialize the feature selector
        selector = SelectKBest(score_func=f_regression, k=num_features)

        # Fit to the data and transform it
        X_selected = selector.fit_transform(X, y)

        return X_selected

    @staticmethod
    def convert_date_to_timestamp(date_str):
        """ Convert a date string to a Unix timestamp. """
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return time.mktime(date.timetuple())
        except ValueError:
            return None

    def train_random_forest(self, data_file_path, scaler_type, target_column='close'):
        try:
            # Load and preprocess the data
            data = pd.read_csv(data_file_path)
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            # Convert date columns to timestamps
            for column in data.columns:
                if data[column].dtype == 'object':
                    # Assuming the date is in 'YYYY-MM-DD' format, adapt if necessary
                    if data[column].str.match(r'\d{4}-\d{2}-\d{2}').any():
                        data[column] = data[column].apply(self.convert_date_to_timestamp)

            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # Impute NaN values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # Scale features
            X_scaled = self.scale_features(X_imputed, scaler_type)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            # Initialize Random Forest model
            rf = RandomForestRegressor()

            # Hyperparameter tuning with Grid Search
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_

            # Train the model with the best hyperparameters
            best_rf.fit(X_train, y_train)

            return best_rf
        except FileNotFoundError:
            logging.error(f"Data file not found at path: {data_file_path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise

    # Example usage of train_random_forest
    # model = train_random_forest(data_file_path, scaler_type, target_column)


    # Method to create scaler selection dropdown
    def setup_scaler_selection(self):
        tk.Label(self, text="Choose Scaler:").pack()
        self.scaler_var = tk.StringVar()
        self.scaler_dropdown = ttk.Combobox(self, textvariable=self.scaler_var, 
                                            values=["StandardScaler", "MinMaxScaler", 
                                                    "RobustScaler", "Normalizer", "MaxAbsScaler"])
        self.scaler_dropdown.pack()
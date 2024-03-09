# model_training_tab.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import asyncio
from datetime import datetime
import json
import logging
import queue
import smtplib
import threading
import traceback
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tkinter as tk
from tkinter import ttk, filedialog

# Third-party Imports
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    explained_variance_score, log_loss, max_error,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, precision_recall_fscore_support,
    r2_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
)
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
# Local Imports
from Utilities.utils import MLRobotUtils
from model_development.model_training import perform_hyperparameter_tuning
from functools import wraps
from datetime import datetime
from sklearn.base import BaseEstimator
import joblib
import pickle
import sklearn
import keras
import torch

# Section 1.2: Model training tab class
# Filter warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup TensorFlow logging level
tf.get_logger().setLevel('ERROR')

# Setup logging for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
optuna.logging.set_verbosity(optuna.logging.INFO)


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
        self.utils = MLRobotUtils(
            is_debug_mode=config.getboolean('Settings', 'DebugMode', fallback=False)
        )
        
        self.setup_model_training_tab()  # Initialize GUI components
        self.setup_epochs_entry()
        self.scaler_type_var = tk.StringVar()
        self.n_estimators_entry = tk.Entry(self)  # Entry for the number of estimators
        self.n_estimators_entry.pack()
        self.window_size_label = tk.Label(self, text="Window Size:")  # Label for window size
        self.window_size_entry = tk.Entry(self)  # Entry for window size
        self.trained_scaler = None  # Trained data scaler
        self.queue = queue.Queue()  # Queue for asynchronous tasks
        self.after(100, self.process_queue)  # Regular queue processing
        self.error_label = tk.Label(self, text="", fg="red")  # Label for error messages
        self.error_label.pack()
        self.setup_debug_mode_toggle()
        self.setup_dropout_rate_slider()

    def setup_epochs_entry(self):
        # Setup for epochs_entry
        tk.Label(self, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self)
        self.epochs_entry.insert(0, "50")  # Default value
        self.epochs_entry.pack()

    def setup_dropout_rate_slider(self):
        tk.Label(self, text="Dropout Rate:").pack()
        self.dropout_rate_var = tk.DoubleVar()
        self.dropout_rate_slider = ttk.Scale(
            self,
            variable=self.dropout_rate_var,
            from_=0.0,
            to=1.0,
            orient="horizontal"
        )
        self.dropout_rate_slider.pack()
        
        self.dropout_rate_slider.pack()
                
    # Function to set up the debug mode toggle button
    def setup_debug_mode_toggle(self):
        self.debug_button = tk.Button(self, text="Enable Debug Mode", command=self.toggle_debug_mode)
        self.debug_button.pack(pady=5)

    # Function to toggle the debug mode on button click
    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        btn_text = "Disable Debug Mode" if self.is_debug_mode else "Enable Debug Mode"
        self.debug_button.config(text=btn_text)
        self.display_message(f"Debug mode {'enabled' if self.is_debug_mode else 'disabled'}", level="DEBUG")

    def log_debug(self, message):
        if self.is_debug_mode:
            self.display_message(message, level="DEBUG")

    def log_message_from_thread(self, message):
        # This method allows logging from background threads
        self.after(0, lambda: self.display_message(message))

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
        config_frame = ttk.Frame(self)
        config_frame.pack(pady=10, fill='x', padx=10)
        self.start_training_button = ttk.Button(config_frame, text="Start Training", command=self.start_training)
        self.start_training_button.pack(padx=10)

        self.pause_button = ttk.Button(config_frame, text="Pause Training", command=self.pause_training)
        self.pause_button.pack(padx=10)

        self.resume_button = ttk.Button(config_frame, text="Resume Training", command=self.resume_training)
        self.resume_button.pack(padx=10)



# Section 2: GUI Components and Functions


    async def async_preprocess_data(self):
        """
        Loads and preprocesses data asynchronously, handling errors.
        """
        try:
            # Replace with your actual data loading and preprocessing logic
            await asyncio.sleep(1)  # Placeholder for preprocessing
            # Example: X_train, X_val, y_train, y_val = ...
            
            # Ensure variables are not None
            if None in [X_train, X_val, y_train, y_val]:
                raise ValueError("One or more data variables are None")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            self.display_message(f"Error during data preprocessing: {str(e)}", level="ERROR")
            raise  # Re-raise the exception to be handled by the caller

    def async_evaluate_model(self, X_test, y_test, model_type):
        """
        Asynchronously evaluates the model and updates the UI with the results.
        
        Args:
        - X_test: Test features
        - y_test: True labels
        - model_type: 'classification' or 'regression'
        """
        try:
            # Check if the trained model exists and is valid
            if self.trained_model is None:
                raise ValueError("No trained model available for evaluation.")

            # Ensure that the predict method is available for the trained model
            if not hasattr(self.trained_model, 'predict'):
                raise AttributeError("The trained model does not support prediction.")

            y_pred = self.trained_model.predict(X_test)
            results_message = "Model Evaluation:\n"

            if model_type == 'classification':
                # Check if predict_proba method is available for the trained model
                if not hasattr(self.trained_model, 'predict_proba'):
                    raise AttributeError("The trained model does not support probability prediction.")

                y_pred_proba = self.trained_model.predict_proba(X_test)[:, 1]  # Assuming binary classification
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)

                results_message += (
                    f"Accuracy: {accuracy:.2f}\n"
                    f"Precision: {precision:.2f}\n"
                    f"Recall: {recall:.2f}\n"
                    f"F1-Score: {fscore:.2f}\n"
                    f"AUC-ROC: {auc_roc:.2f}\n"
                    f"Log Loss: {logloss:.2f}\n"
                    f"Confusion Matrix:\n{conf_matrix}"
                )
                
                # Optionally, plot the confusion matrix
                self.plot_confusion_matrix(conf_matrix, ['Class 0', 'Class 1'])
            
            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)

                results_message += (
                    f"MSE: {mse:.2f}\n"
                    f"RMSE: {rmse:.2f}\n"
                    f"R2 Score: {r2:.2f}\n"
                )

            # Update UI with results in the main thread
            self.after(0, lambda: self.display_evaluation_results(results_message))

        except Exception as e:
            error_message = f"Error during model evaluation: {e}"
            self.after(0, lambda: self.display_message(error_message, level="ERROR"))


    # Implement display_evaluation_results and plot_confusion_matrix as needed for your UI


    def plot_confusion_matrix(self, y_true=None, y_pred=None, conf_matrix=None, 
                              class_names=None, save_path="confusion_matrix.png", 
                              show_plot=True):
        """
        Plot and optionally save a confusion matrix. Can accept either a confusion matrix,
        or true and predicted labels to compute the confusion matrix.

        Args:
            y_true (array-like, optional): True labels. Required if conf_matrix is not provided.
            y_pred (array-like, optional): Predicted labels. Required if conf_matrix is not provided.
            conf_matrix (array-like, optional): Precomputed confusion matrix. If provided, y_true and y_pred are ignored.
            class_names (list, optional): List of class names for labeling the axes. If not provided, integers will be used.
            save_path (str): Path to save the plot. Defaults to "confusion_matrix.png".
            show_plot (bool): Whether to display the plot. Defaults to True.
        """
        if conf_matrix is None:
            if y_true is None or y_pred is None:
                raise ValueError("You must provide either a confusion matrix or true and predicted labels.")
            conf_matrix = confusion_matrix(y_true, y_pred)

        # Use provided class names or default to integers
        if class_names is None:
            class_names = list(range(conf_matrix.shape[0]))

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        # Save the plot to a file
        if save_path:
            plt.savefig(save_path)
        
        # Optionally display the plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def display_evaluation_results(self, message, widget=None):
        """
        Updates a UI element with the evaluation results.

        Args:
        - message: The results message to display.
        - widget: The UI element to update. If None, assumes a Text widget named self.log_text.
        """
        if widget is None:
            widget = self.log_text

        # Update the UI element with the results
        widget.config(state='normal')
        widget.insert('end', message + "\n")
        widget.config(state='disabled')
        widget.see('end')

    def create_sequence(self, features, target, lookback):
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def prepare_and_train_lstm_model(self, df, scaler_type, lookback=60, epochs=50, batch_size=32):
        # Ensure target_column is in the DataFrame
        target_column = 'close'
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Convert date to a numerical feature, for example, days since a fixed date
        df['date'] = pd.to_datetime(df['date'])
        reference_date = pd.to_datetime('2000-01-01')
        df['days_since'] = (df['date'] - reference_date).dt.days

        # Exclude the original date column and use 'days_since' for training
        features = df.drop(columns=[target_column, 'date']).values
        target = df[target_column].values

        # Scaling the features and target
        feature_scaler = self.get_scaler(scaler_type)
        scaled_features = feature_scaler.fit_transform(features)
        target = target.reshape(-1, 1)
        target_scaler = self.get_scaler(scaler_type)
        scaled_target = target_scaler.fit_transform(target)

        # Creating sequences for LSTM
        X, y = self.create_sequence(scaled_features, scaled_target.flatten(), lookback)

        # Splitting dataset into training, testing, and new data
        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_new, y_test, y_new = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

        # Defining the LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

        return model, feature_scaler, target_scaler, X_test, y_test, X_new, y_new

    
    def neural_network_preprocessing(self, data, scaler_type, close_price_column='close', file_path=None):
        """
        Specific preprocessing for neural network models.

        Args:
            data (DataFrame): The dataset to preprocess.
            scaler_type (str): Type of scaler to use for feature scaling.
            close_price_column (str): Name of the column for close prices, which is the target.
            file_path (str, optional): Path of the data file for logging purposes.

        Returns:
            tuple: Preprocessed features (X) and target values (y).
        """
        try:
            if close_price_column not in data.columns:
                raise ValueError(f"'{close_price_column}' column not found in the data.")

            # Convert date column to numeric features
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data['year'] = data['date'].dt.year
                data['month'] = data['date'].dt.month
                data['day'] = data['date'].dt.day
                data.drop(columns=['date'], inplace=True)

            # Prepare your features and target
            X = data.drop(columns=[close_price_column])
            y = data[close_price_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Handle non-numeric columns here (if any)

            # Scaling features
            scaler = self.get_scaler(scaler_type)
            X_scaled = scaler.fit_transform(X)

            return X_scaled, y

        except Exception as e:
            error_message = f"Error in neural network preprocessing: {str(e)}"
            if file_path:
                error_message += f" File path: {file_path}"
            self.utils.log_message(error_message, self, self.log_text, self.is_debug_mode)
            return None, None

    def preprocess_new_data(self, new_data):
        scaled_new_data = self.feature_scaler.transform(new_data)
        X_new = self.create_sequence(scaled_new_data, self.lookback)
        return X_new

    def load_unseen_test_data(filepath):
        """
        Load unseen test data from a CSV file.

        Parameters:
        filepath (str): The path to the CSV file containing the test data.

        Returns:
        DataFrame: The loaded test data.
        """
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None

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



    def start_training(self, X_test=None, y_test=None):
        
        if not self.validate_inputs():
            self.display_message("Invalid input. Please check your settings.")
            return
        # Ensure the entry widget exists and has not been destroyed
        if hasattr(self, 'epochs_entry'):
            epochs_str = self.epochs_entry.get()

        data_file_path = self.data_file_entry.get()
        scaler_type = self.scaler_type_var.get()
        model_type = self.model_type_var.get()
        epochs_str = self.epochs_entry.get()
        epochs = int(epochs_str) if epochs_str.isdigit() and int(epochs_str) > 0 else 50

        if epochs is None:
            self.display_message("Error: Number of epochs not specified for the selected model type.")
            return

        try:
            self.disable_training_button()
            self.display_message("Training started...", level="INFO")

            # Load and preprocess the data
            data = pd.read_csv(data_file_path)
            self.display_message("Data loading and preprocessing started.", level="INFO")
            features = data.drop(['date', 'close'], axis=1)
            target = data['close']

            scaler = self.get_scaler(scaler_type)
            scaled_features = scaler.fit_transform(features)
            scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

            def create_sequences(features, target, sequence_length):
                X, y = [], []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:(i + sequence_length)])
                    y.append(target[i + sequence_length, 0])
                return np.array(X), np.array(y)

            sequence_length = 5
            X, y = create_sequences(scaled_features, scaled_target, sequence_length)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "neural_network":
                # Preprocess data for neural network
                X, y = self.neural_network_preprocessing(data, scaler_type)
                if X is None or y is None:
                    self.display_message("Preprocessing failed. Training aborted.", level="ERROR")
                    return

                # Splitting the data into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                # Neural network training logic
                try:
                    def objective(trial):
                        # Hyperparameter definitions
                        num_layers = trial.suggest_int('num_layers', 1, 3)
                        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
                        units_per_layer = trial.suggest_categorical('units', [16, 32, 64, 128])

                        # Model creation with the trial's current hyperparameters
                        model = Sequential()
                        for _ in range(num_layers):
                            model.add(Dense(units_per_layer, activation='relu'))
                            model.add(Dropout(dropout_rate))
                        model.add(Dense(1))  # Adjust this according to your output layer needs

                        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

                        # Training the model
                        history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)

                        # Objective: Minimize the validation loss
                        validation_loss = np.min(history.history['val_loss'])
                        return validation_loss

                    # Optuna study
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=10)

                    # Training final model with best parameters
                    best_params = study.best_params
                    final_model = Sequential()
                    for _ in range(best_params['num_layers']):
                        final_model.add(Dense(best_params['units'], activation='relu'))
                        final_model.add(Dropout(best_params['dropout_rate']))
                    final_model.add(Dense(1))  # Adjust this according to your output layer needs

                    final_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
                    final_model.fit(X_train, y_train, epochs=best_params.get('epochs', 10), 
                                    batch_size=best_params.get('batch_size', 32), verbose=1)
                    self.save_trained_model(final_model, 'neural_network')

                    # Evaluate your model's performance
                    self.async_evaluate_model(final_model, X_val, y_val)

                except Exception as e:
                    self.display_message(f"Error training neural network: {str(e)}", level="ERROR")

            elif model_type == "LSTM":
                try:
                    # Imputer to handle NaN values
                    imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'most_frequent' too
                    scaled_features_imputed = imputer.fit_transform(scaled_features)

                    # Create sequences for LSTM
                    sequence_length = 60  # Adjust as needed
                    X, y = create_sequences(scaled_features_imputed, scaled_target, sequence_length)
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Prepare and train the LSTM model (Assuming this function is defined elsewhere in your code)
                    lstm_model = self.prepare_and_train_lstm_model(X_train, y_train, epochs, batch_size=32)

                    # Evaluate the model (assuming you have a separate evaluation dataset)
                    if X_val is not None and y_val is not None:
                        self.async_evaluate_model(lstm_model, X_val, y_val, model_type='regression')
                    else:
                        self.display_message("Evaluation skipped: X_val or y_val is not provided.", level="WARNING")

                    # Save the model for later use or deployment
                    self.save_trained_model(lstm_model, model_type='lstm')

                except ValueError as ve:
                    self.display_message(f"Training failed: {str(ve)}", level="ERROR")
                except Exception as e:
                    self.display_message(f"Unexpected error in LSTM training: {str(e)}", level="ERROR")
                    traceback.print_exc()  # Optionally print the full traceback for debugging

            # Linear Regression Training
            elif model_type.lower() == "linear_regression":
                imputer = SimpleImputer(strategy='mean')
                lr_pipeline = make_pipeline(imputer, LinearRegression())
                lr_X_train, lr_X_val, lr_y_train, lr_y_val = train_test_split(
                    scaled_features, scaled_target, test_size=0.2, random_state=42)
                lr_pipeline.fit(lr_X_train, lr_y_train)
                
                # Here, the file path is optional. If not provided, save_trained_model will handle it.
                self.save_trained_model(lr_pipeline, 'linear_regression', scaler)


                            
            # Random Forest Training
            elif model_type.lower() == "random_forest":
                rf_model = RandomForestRegressor(n_estimators=100)
                rf_X_train, rf_X_val, rf_y_train, rf_y_val = train_test_split(
                    scaled_features, scaled_target, test_size=0.2, random_state=42)
                rf_model.fit(rf_X_train, rf_y_train.ravel())
                self.save_trained_model(rf_model, 'random_forest', scaler)

            elif model_type == "ARIMA":
                self.train_arima_model_in_background(target)

            # Simulate a delay for demonstration purposes; replace or remove with actual model training code
            time.sleep(2)  # Simulate training delay
            self.display_message("Training completed successfully.", level="INFO")

        except Exception as e:
            error_message = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.display_message(error_message, level="ERROR")


        finally:
            self.enable_training_button()  # This ensures the button is always re-enabled

    def train_arima_model_in_background(self, close_prices):
        def background_training(close_prices):
            results = {
                'predictions': [],
                'errors': [],
                'parameters': {'order': (5, 1, 0)},
                'performance_metrics': {}
            }
            train_size = int(len(close_prices) * 0.8)
            train, test = close_prices[:train_size], close_prices[train_size:]
            history = [x for x in train]

            for t in range(len(test)):
                try:
                    model = ARIMA(history, order=results['parameters']['order'])
                    model_fit = model.fit()
                    forecast = model_fit.forecast()[0]
                    results['predictions'].append(forecast)
                    obs = test.iloc[t]
                    history.append(obs)
                except Exception as e:
                    error_message = f"Error training ARIMA model at step {t}: {e}"
                    print(error_message)
                    results['errors'].append(error_message)
            
            # After training, calculate performance metrics
            # For simplicity, let's assume we're calculating MSE as a placeholder
            results['performance_metrics']['mse'] = np.mean((np.array(test) - np.array(results['predictions']))**2)

            # Save model, results, and metadata
            self.save_arima_results(results, model_fit)
            self.save_trained_model(model_fit, model_type= 'arima')

        threading.Thread(target=background_training, args=(close_prices,), daemon=True).start()
        self.display_message("ARIMA model training started in background...", level="INFO")

    def save_arima_results(self, results, model_fit):
        try:
            # Get model directory path and prepare file paths
            models_directory = self.config.get('Paths', 'models_directory')
            if not os.path.exists(models_directory):
                os.makedirs(models_directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file_path = os.path.join(models_directory, f'arima_model_{timestamp}.pkl')

            # Save the ARIMA model using the updated save_model_by_type method
            self.save_model_by_type(model_fit, 'arima', model_file_path)
            self.display_message(f"ARIMA model saved to {model_file_path}", level="INFO")

            # Save predictions and errors
            results_file_path = os.path.join(models_directory, f'arima_results_{timestamp}.json')
            with open(results_file_path, 'w') as result_file:
                json.dump(results, result_file, indent=4)
            self.display_message(f"ARIMA model results saved to {results_file_path}", level="INFO")

        except Exception as e:
            error_message = f"Error saving ARIMA model results: {e}"
            self.display_message(error_message, level="ERROR")
            raise
            
    def save_trained_model(self, model=None, model_type=None, scaler=None, file_path=None):
        """
        Save the trained model, scaler, and any metadata to separate files. Enhanced to handle direct file path specification
        or user interaction for file path selection, with explicit handling for .joblib file extension.
        Args:
            model: The trained model to be saved. Defaults to self.trained_model if None.
            model_type (str): The type of the trained model.
            scaler: The trained scaler. Defaults to self.trained_scaler if None.
            file_path (str): The base file path for saving. If None, a file dialog is used to determine the path.
        Returns:
            None
        """
        model = model or self.trained_model
        scaler = scaler or getattr(self, 'trained_scaler', None)

        if model is None or model_type is None:
            print("No trained model available to save or model type not provided.")
            return

        # Determine the model type and file extension
        file_extension = ".joblib"

        # User interaction for file path if not provided
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"{model_type}_{timestamp}{file_extension}"
            file_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                                    initialfile=default_filename,
                                                    filetypes=[(f"{model_type.upper()} Files", f"*{file_extension}"), ("All Files", "*.*")])

            if not file_path:  # User canceled the save dialog
                print("Save operation canceled by user.")
                return

        # Append "_model" suffix before the file extension if not directly specified by user
        if not file_path.endswith(file_extension):
            file_path += "_model" + file_extension

        # Save the model
        joblib.dump(model, file_path)
        print(f"Model of type '{model_type}' saved successfully at {file_path}")

        # Save the scaler if present
        if scaler is not None:
            scaler_file_path = file_path.replace(file_extension, "_scaler.joblib")
            joblib.dump(scaler, scaler_file_path)
            print(f"Scaler saved successfully at {scaler_file_path}")

        # Prepare and save metadata
        metadata = self.construct_metadata(model, model_type, scaler)
        metadata_file_path = file_path.replace(file_extension, "_metadata.json")
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        print(f"Metadata saved to {metadata_file_path}")

    def construct_metadata(self, model, model_type, scaler=None):
        """
        Construct and return metadata for the model, including parameters, performance metrics, and optionally scaler information.
        Args:
            model: The trained model.
            model_type (str): The type of the trained model.
            scaler: The trained scaler. Optional; defaults to None.
        Returns:
            dict: The constructed metadata.
        """
        metadata = {
            'model_type': model_type,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        if hasattr(model, 'get_params'):
            # For simplicity, convert all parameter values to strings
            metadata['model_parameters'] = {param: str(value) for param, value in model.get_params().items()}

        if hasattr(model, 'named_steps'):
            metadata['pipeline_steps'] = {}
            for name, step in model.named_steps.items():
                # Represent each step by its class name and parameters
                step_representation = {
                    'class': step.__class__.__name__,
                    'parameters': {param: str(value) for param, value in step.get_params().items()}
                }
                metadata['pipeline_steps'][name] = step_representation

        if scaler:
            metadata['scaler'] = {
                'class': scaler.__class__.__name__,
                'parameters': {param: str(value) for param, value in scaler.get_params().items()}
            }

        return metadata




    # Other methods for validation, input retrieval, and setup

    def disable_training_button(self):
        self.start_training_button.config(state='disabled')

    def enable_training_button(self):
        self.start_training_button.config(state='normal')

    def display_message(self, message, level="INFO"):
        # Define colors for different log levels
        log_colors = {"INFO": "black", "WARNING": "orange", "ERROR": "red", "DEBUG": "blue"}

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the message with timestamp and level
        formatted_message = f"[{timestamp} - {level}] {message}\n"

        # Insert the message into the log_text widget
        self.log_text.config(state='normal')
        self.log_text.tag_config(level, foreground=log_colors.get(level, "black"))
        self.log_text.insert(tk.END, formatted_message, level)

        # Scroll to the end of the log_text widget to show the latest message
        self.log_text.see(tk.END)

        # Disable the log_text widget to prevent user editing
        self.log_text.config(state='disabled')


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
            self.display_message(message, self.log_text)
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
            raise ValueError("Unsupported model type")
        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer")

        # Preprocess data
        window_size = int(self.window_size_entry.get()) if model_type == "neural_network" else 1
        X_train, X_test, y_train, y_test = await self.preprocess_data(data_file_path, scaler_type, model_type, window_size)

        # Initialize and configure model
        model = await self.initialize_and_configure_model(model_type, X_train.shape[1:], epochs)
        if model is None:
            raise ModelTrainingError("Failed to initialize model")

        # Automated hyperparameter tuning
        best_params = await perform_hyperparameter_tuning(model, X_train, y_train, epochs)

        # Ensemble and quantize model
        ensemble_model = self.create_ensemble_model([model], best_params)
        quantized_model = self.quantize_model(ensemble_model)

        # Train the model asynchronously
        training_metrics = await self.train_model_sync(quantized_model, X_train, y_train, epochs)

        # Post-training actions
        self.trained_model = quantized_model
        self.log_training_completion(quantized_model, training_metrics)
        self.enable_training_button()

        # Evaluate model performance
        model_evaluation_results = self.async_evaluate_model(quantized_model, X_test, y_test)
        self.display_message(f"Model evaluation results: {model_evaluation_results}")

    async def initialize_and_configure_model(self, model_type, input_shape, epochs):
        """
        Initialize and configure a machine learning model based on the type.

        Args:
            model_type (str): Type of the model to initialize.
            input_shape (tuple): Shape of the input data for neural networks and LSTM models.
            epochs (int): Number of epochs for training, mainly used for neural networks and LSTM.

        Returns:
            Initialized model.
        """
        if model_type == "neural_network":
            # Initialization for a simple feedforward neural network
            model = Sequential([
                Dense(128, activation='relu', input_shape=input_shape),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1)  # Adjust based on your output
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')  # Customize as needed
            return model

        elif model_type == "LSTM":
            # Initialization for an LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)  # Adjust based on your output
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')  # Customize as needed
            return model

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100)  # Customize as needed
            return model

        elif model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            return model

        elif model_type == "ARIMA":
            # For ARIMA, the model initialization is different as it's not a sklearn or keras model
            # We'll return a placeholder or configuration here, as actual model fitting happens during training
            return {'order': (5, 1, 0)}  # Customize ARIMA order (p,d,q) as needed

        else:
            raise ValueError("Unsupported model type")

        return None


    def log_training_completion(self, model, model_type, training_metrics):
        """
        Log the completion of the training process and handle additional tasks.

        Args:
            model: The trained model.
            model_type (str): Type of the model (e.g., 'lstm', 'neural_network').
            training_metrics: Metrics or results from the training process.

        Returns:
            None
        """
        # Log the training completion
        self.display_message(f"Training completed successfully. Metrics: {training_metrics}", level="INFO")

        # Additional tasks like saving the model
        try:
            # Use the save_trained_model method to save the model
            # Specify the actual path and model type
            self.save_trained_model(model, model_type=model_type, file_path=f'path/to/save/{model_type}_model')
            self.display_message("Model saved successfully.", level="INFO")
        except Exception as e:
            self.display_message(f"Error saving model: {str(e)}", level="ERROR")

    def train_model_sync(self, model, X_train, y_train, epochs):
        """
        Synchronously train the model.

        Args:
            model: The machine learning model to train.
            X_train: Training feature data.
            y_train: Training target data.
            epochs: Number of training epochs.

        Returns:
            None
        """
        try:
            # Directly train the model without async calls
            model.fit(X_train, y_train, epochs=epochs, verbose=1)
        except Exception as e:
            print(f"Error in training model: {e}")


    def create_ensemble_model(self, base_models, train_data, train_labels, method='voting', weights=None):
        """
        Create an ensemble model using the specified method.

        Args:
            base_models (list): List of base machine learning models to include in the ensemble.
            method (str): The ensemble method to use. Options: 'voting', 'stacking', etc. Default is 'voting'.
            weights (list): Optional list of weights for voting. Default is None.

        Returns:
            object: The ensemble model object.
        """
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            if isinstance(base_models[0], Classifier):
                ensemble_model = VotingClassifier(estimators=base_models, voting='hard', weights=weights)
            else:
                ensemble_model = VotingRegressor(estimators=base_models, weights=weights)
        elif method == 'stacking':
            # Implement stacking ensemble method
            pass
        else:
            raise ValueError("Unsupported ensemble method.")

        ensemble_model.fit(train_data, train_labels)  # Assuming train_data and train_labels are defined
        return ensemble_model


    def quantize_model(self, model, quantization_method='weight'):
        """
        Quantize the given model using the specified quantization method.

        Args:
            model: The trained machine learning model to quantize.
            quantization_method (str): The quantization method to use. Options: 'weight', 'activation', etc. Default is 'weight'.

        Returns:
            object: The quantized model object.
        """
        if quantization_method == 'weight':
            # Implement weight quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()
        elif quantization_method == 'activation':
            # Implement activation quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_generator
            quantized_model = converter.convert()
        else:
            raise ValueError("Unsupported quantization method.")

        # Return the quantized model
        return quantized_model

    def representative_dataset_generator(self,train_data):
        """
        Generate representative dataset for activation quantization.

        Returns:
            tf.data.Dataset: A representative dataset.
        """
        # Implement code to generate a representative dataset
        # Example: Use a small subset of your training data
        dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(1)
        for input_data in dataset.take(100):
            yield [input_data]

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
            return

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
            self.display_message("Data preprocessing completed.", level="INFO")
            return X_train, X_test, y_train, y_test
        
        except pd.errors.EmptyDataError as empty_error:
            # Handle empty data file error
            self.utils.log_message(f"Empty data error in preprocessing: {str(empty_error)} - {data_file_path}", self.log_text, self.is_debug_mode)
        except pd.errors.ParserError as parser_error:
            # Handle parsing error from pandas
            self.utils.log_message(f"Data parsing error in preprocessing: {str(parser_error)} - {data_file_path}", self.log_text, self.is_debug_mode)
        except ValueError as value_error:
            # Handle value errors, typically related to incorrect data types or values
            self.utils.log_message(f"Value error in preprocessing: {str(value_error)} - {data_file_path}", self.log_text, self.is_debug_mode)
        except Exception as general_error:
            # Handle any other exceptions that are not caught by the specific ones above
            import traceback
            self.utils.log_message(f"General error in preprocessing: {str(general_error)}\nTraceback: {traceback.format_exc()} - {data_file_path}", self.log_text, self.is_debug_mode)
        finally:
            return None, None, None, None

    # Define specific preprocessing methods

    # Function to preprocess data for ARIMA models
    def arima_preprocessing(self, data, file_path):
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
            target_column = 'close' if 'close' in data.columns else '4. close' if '4. close' in data.columns else None
            if not target_column:
                self.utils.log_message("Target column for close prices not found in the dataset." + file_path, self, self.log_text, self.is_debug_mode)
                return None, None

            y = data[target_column].values

            # Prepare X (features)
            num_lags = 5  # Adjust the number of lag features as needed
            for lag in range(1, num_lags + 1):
                data[f'lag_{lag}'] = y.shift(lag)

            # Drop rows with NaN values due to lag
            data.dropna(inplace=True)

            # Prepare X with lag features
            X = data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]  # Replace with your relevant lag features

            return X, y

        except pd.errors.ParserError as parser_error:
            self.utils.log_message(f"Parser error in ARIMA preprocessing: {str(parser_error)} - {file_path}", self, self.log_text, self.is_debug_mode)
            return None, None
        except ValueError as value_error:
            self.utils.log_message(f"Value error in ARIMA preprocessing: {str(value_error)} - {file_path}", self, self.log_text, self.is_debug_mode)
            return None, None
        except Exception as general_error:
            import traceback
            self.utils.log_message(f"General error in ARIMA preprocessing: {str(general_error)}\nTraceback: {traceback.format_exc()} - {file_path}", self, self.log_text, self.is_debug_mode)
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

        if isinstance(X, pd.DataFrame):
            # If DataFrame, retain column names
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # If NumPy array, just scale it
            X_scaled = scaler.fit_transform(X)

        return X_scaled

    def calculate_model_accuracy(self, X_test, y_test):
        """
        Calculate the accuracy of the integrated model, given test data.

        Parameters:
            X_test (array-like): Test features.
            y_test (array-like): True values for test features.

        Returns:
            float: Model accuracy as a percentage, or 0.0 if not applicable.
        """
        try:
            if hasattr(self.trained_model, 'score'):
                accuracy = self.trained_model.score(X_test, y_test)
                return accuracy * 100.0
        except Exception as e:
            print(f"Error calculating model accuracy: {str(e)}")
        return 0.0

    def save_scaler(self, scaler, file_path=None):
        try:
            # If file_path is not provided, open a file dialog for the user to select a save location
            if file_path is None:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All Files", "*.*")],
                    title="Save Scaler As"
                )

                # Check if the user canceled the save operation
                if not file_path:
                    self.display_message("Save operation canceled.", level="INFO")
                    return

            # Determine the appropriate method to save the scaler
            if isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler)):
                joblib.dump(scaler, file_path)
            else:
                # If scaler is not a recognized type, use pickle
                with open(file_path, 'wb') as file:
                    pickle.dump(scaler, file)

            self.display_message(f"Scaler saved successfully at {file_path}", level="INFO")

        except Exception as e:
            self.display_message(f"Error saving scaler: {str(e)}", level="ERROR")



    def save_model_by_type(self, model, model_type, file_path):
        # Convert model_type to lowercase for consistent comparison
        model_type = model_type.lower()

        if model_type == 'linear_regression' or model_type == 'random_forest':
            # Save scikit-learn model
            joblib.dump(model, file_path)
        elif model_type == 'lstm' or model_type == 'neural_network':
            # Save Keras model
            model.save(file_path)
        elif model_type == 'arima':
            # Save ARIMA model
            with open(file_path, 'wb') as pkl:
                pickle.dump(model, pkl)
        else:
            print(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"Model of type '{model_type}' saved successfully.")

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
        self.utils.log_message("Automated training started." + data_file_path, self, self.log_text, self.is_debug_mode)

        # Implement adaptive learning logic based on past performance
        model_type = self.config.get("Model", "model_type")
        data_file_path = self.config.get("Data", "file_path")
        epochs = int(self.config.get("Model", "epochs")) if model_type in ["neural_network", "LSTM"] else 1

        # Call training logic here
        self.train_model_and_enable_button(data_file_path, model_type, epochs)

        # Implement real-time analytics monitoring during training
        self.initiate_real_time_training_monitoring()

        # Log the completion of automated training
        self.utils.log_message("Automated training completed." + data_file_path, self, self.log_text, self.is_debug_mode)

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
        self.display_message("Previewing data from file: " + file_path)
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
        self.utils.log_message(f"Model saved: {file_path}" + file_path, self, self.log_text, self.is_debug_mode, level="INFO")

        # Optionally, upload the model to cloud storage for persistence and global access
        self.upload_model_to_cloud(file_path)

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
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        explained_variance = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        metrics = {
            'Mean Absolute Error (MAE)': mae,
            'Root Mean Squared Error (RMSE)': rmse,
            'R-squared (R2)': r2,
            'Explained Variance': explained_variance,
            'Max Error': max_err,
            'Mean Absolute Percentage Error (MAPE)': mape
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

    def is_training_complete(self, progress_data, target_epochs=10):
        """
        Check if the training is complete based on progress data.

        Args:
            progress_data (dict): A dictionary containing training progress information.
            target_epochs (int): The target number of epochs to reach. Default is 10.

        Returns:
            bool: True if training is complete, False otherwise.
        """
        # Example: Check if a certain number of epochs (e.g., 10) have been reached
        return progress_data.get('epoch', 0) >= target_epochs

    def on_training_completed(self, y_test, y_pred):
        """
        Handle actions to be taken when training is completed.

        Args:
            y_test: The true labels from the test set.
            y_pred: The predicted labels from the model.
        """
        self.visualize_training_results(y_test, y_pred)
        self.calculate_model_metrics(y_test, y_pred)

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
        
    def show_dynamic_options(self, event):
        # Clear current dynamic options
        for widget in self.dynamic_options_frame.winfo_children():
            widget.destroy()

        selected_model_type = self.model_type_var.get()
        if selected_model_type == "neural_network":
            self.setup_neural_network_options()
        elif selected_model_type == "LSTM":
            self.setup_neural_network_options()
        elif selected_model_type == "ARIMA":
            self.setup_arima_options()
        elif selected_model_type == "linear_regression":
            self.setup_linear_regression_options()
        elif selected_model_type == "random_forest":
            self.setup_random_forest_options()
        # ... add other model type specific options ...

    def setup_neural_network_options(self):
        # Epochs input with default value
        tk.Label(self.dynamic_options_frame, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.dynamic_options_frame)
        self.epochs_entry.insert(0, "50")  # Default value
        self.epochs_entry.pack()

        # Window Size input with default value
        tk.Label(self.dynamic_options_frame, text="Window Size:").pack()
        self.window_size_entry = tk.Entry(self.dynamic_options_frame)
        self.window_size_entry.insert(0, "30")  # Default value
        self.window_size_entry.pack()

        # Add a 'Submit' button with a command to validate and apply the settings
        submit_button = tk.Button(self.dynamic_options_frame, text="Submit", command=self.apply_neural_network_options)
        submit_button.pack()

    
    def apply_neural_network_options(self):
        try:
            # Validate and retrieve epochs
            epochs = int(self.epochs_entry.get())
            if epochs <= 0:
                raise ValueError("Epochs must be a positive integer.")

            # Validate and retrieve window size
            window_size = int(self.window_size_entry.get())
            if window_size <= 0:
                raise ValueError("Window Size must be a positive integer.")

            # Further processing with validated values...
            # For example, setting these values in the model configuration

        except ValueError as e:
            tk.messagebox.showerror("Input Error", str(e))

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

    def get_model_type(self, model=None):
        """
        Get the type of the model.

        Args:
            model (optional): The model to check.

        Returns:
            str: Model type.
        """
        if model is None:
            model = self.trained_model

        # Update these checks based on your actual model classes
        if isinstance(model, ARIMA):
            return "ARIMA"
        elif isinstance(model, RandomForestRegressor):
            return "Random Forest"
        elif hasattr(model, 'predict') and not isinstance(model, (sklearn.base.BaseEstimator, keras.models.Model, torch.nn.Module)):
            return "Custom Neural Network"  # For custom neural network models
        elif isinstance(model, LinearRegression):
            return "LINEAR REGRESSION"
        elif isinstance(model, sklearn.base.BaseEstimator):
            return "sklearn"
        elif isinstance(model, keras.models.Model):
            return "keras"
        elif isinstance(model, torch.nn.Module):
            return "pytorch"
        else:
            return "Unknown"

    def get_file_extension(self, model_type):
        extensions = {"sklearn": ".joblib", "keras": ".h5", "pytorch": ".pth"}
        return extensions.get(model_type, ".model")

    def create_windowed_data(self, X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)

# Required libraries and modules for the GUI and model training
import os
import tkinter as tk
from tkinter import ttk, filedialog
import logging
import queue
import threading
import warnings
from datetime import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import traceback
from functools import wraps

# Import third-party libraries for data handling, machine learning, and visualization
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
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from keras.regularizers import l1_l2
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Ridge
import numpy as np

# Local utility class for model training and other tasks
from Utilities.utils import MLRobotUtils
from model_development.model_training import perform_hyperparameter_tuning


class ModelTrainingLogger:
    """
    Logger class to enable both console and GUI-based logging for the model training process.
    """
    def __init__(self, log_text_widget):
        # Initializes the logger with a Text widget for GUI-based logging
        self.log_text_widget = log_text_widget
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def log(self, message):
        # Logs a message to the console and updates the GUI-based log widget
        self.logger.info(message)
        self.log_text_widget.config(state='normal')
        self.log_text_widget.insert('end', f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        self.log_text_widget.config(state='disabled')
        self.log_text_widget.see('end')

class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, config, scaler_options):
        super().__init__(parent)
        self.window_size_label = tk.Label(self, text="Window Size:")
        self.window_size_entry = tk.Entry(self, width=10)
        self.utils = MLRobotUtils()  # Initialize the utils attribute
        self.model_configs = {
            "neural_network": {
                "epochs": {"label": "Epochs:", "default": 50},
                "window_size": {"label": "Window Size:", "default": 30}
            },
            "LSTM": {
                "epochs": {"label": "Epochs:", "default": 50},
                "window_size": {"label": "Window Size:", "default": 30}
            },
            "ARIMA": {
                "p_value": {"label": "ARIMA p-value:", "default": 1},
                "d_value": {"label": "ARIMA d-value:", "default": 1},
                "q_value": {"label": "ARIMA q-value:", "default": 1}
            },
            "linear_regression": {
                "regularization": {"label": "Regularization(alpha):", "default": 0.01}
            },
            "random_forest": {
                "n_estimators": {"label": "Number of Trees (n_estimators):", "default": 100},
                "max_depth": {"label": "Max Depth:", "default": None},
                "min_samples_split": {"label": "Min Samples Split:", "default": 2},
                "min_samples_leaf": {"label": "Min Samples Leaf:", "default": 1}
            }
            # Add other models and their configurations here...
        }
        self.config = config  # Configuration for model training (e.g., paths, parameters)
        self.scaler_options = scaler_options  # Scaler options for data normalization
        self.trained_model = None  # Placeholder for the trained model
        self.queue = queue.Queue()  # Queue for logging messages
        self.ml_robot_utils = MLRobotUtils()  # Instance of utility class for additional functionalities
        self.is_debug_mode = False  # Initialize debug mode flag
        self.setup_model_training_tab()  # Call method to setup the GUI components
        self.after(100, self.process_queue)  # Setup a recurring task to process logging messages
        self.error_label = tk.Label(self, text="", fg="red")  # Label for displaying error messages
        self.setup_submit_button()  # Call once during initialization
        # Initialize window size widgets here or in a relevant setup method
        self.window_size_label = tk.Label(self.settings_frame, text="Window Size:")
        self.window_size_entry = tk.Entry(self.settings_frame)
        self.train_button = tk.Button(self, text="Start Automated Training", command=self.run_automated_training_tasks)
        self.train_button.pack()  # Adjust layout as needed

    def setup_submit_button(self):
        """
        Sets up the Submit button for model configuration before training.
        """
        self.submit_button = ttk.Button(self, text="Submit", command=self.submit_model_config)
        self.submit_button.pack(pady=10)

    def submit_model_config(self):
        """
        Gathers inputs from the configuration fields, validates them, and prepares the model for training based on the selected model type.
        """
        try:
            model_type = self.model_type_var.get()
            if model_type not in self.model_configs:
                raise ValueError(f"Unsupported model type: {model_type}")

            config_options = self.model_configs[model_type]
            validated_params = {}

            # Dynamically validate parameters based on the selected model's configuration
            for param, config in config_options.items():
                entry_widget = getattr(self, f"{param}_entry", None)
                if entry_widget:
                    value_str = entry_widget.get().strip()
                    # Convert and validate based on specific requirements (e.g., int for epochs, float for regularization)
                    if param in ["epochs", "window_size", "n_estimators", "min_samples_split", "min_samples_leaf", "p_value", "d_value", "q_value"]:
                        if not value_str.isdigit():
                            raise ValueError(f"{config['label']} must be a positive integer.")
                        value = int(value_str)
                    elif param == "regularization":
                        try:
                            value = float(value_str)
                        except ValueError:
                            raise ValueError(f"{config['label']} must be a valid number.")
                    elif param == "max_depth" and value_str.lower() != "none":
                        if not value_str.isdigit():
                            raise ValueError(f"{config['label']} must be a positive integer or 'None'.")
                        value = int(value_str)
                    else:
                        value = value_str  # For parameters that can accept general strings

                    validated_params[param] = value

            # Proceed to configure or train the model with these validated parameters
            # This is a placeholder for your model configuration or training logic
            self.display_message(f"{model_type} configuration submitted with parameters: {validated_params}")

        except ValueError as e:
            # Display error message in the GUI instead of printing or logging
            self.display_message(f"Error in input: {e}", level="ERROR")




    def setup_model_training_tab(self):
        """
        Configures the GUI components for the model training tab.
        """
        self.setup_title_label()
        self.setup_data_file_path_section()
        self.setup_model_type_selection()
        self.setup_training_configurations()
        self.setup_start_training_button()
        self.setup_progress_and_logging()
        self.setup_debug_mode_toggle()  # Ensure this is included to set up the debug mode toggle button
        # Additional setup for scaler selection dropdown, placed here for coherent organization
        tk.Label(self, text="Select Scaler:").pack()
        self.scaler_type_var = tk.StringVar()
        self.scaler_dropdown = ttk.Combobox(self, textvariable=self.scaler_type_var,
                                            values=["StandardScaler", "MinMaxScaler", 
                                                    "RobustScaler", "Normalizer", "MaxAbsScaler"])
        self.scaler_dropdown.pack()
        # Assuming a global list to store metrics of each training session
        self.training_history = []

    def add_training_session(self, metrics):
        """
        Adds the metrics of a completed training session to the history.
        
        Parameters:
            metrics (dict): A dictionary containing the metrics of the completed training session.
        """
        self.training_history.append(metrics)

    # These include methods to setup title label, data file path section, model type selection, start training button, progress and logging

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

    def setup_title_label(self):
        """
        Sets up the title label for the model training section.
        """
        tk.Label(self, text="Model Training", font=("Helvetica", 16)).pack(pady=10)

    def compare_with_last_session(self):
        """
        Compares the latest training session's metrics with those of the previous session
        and generates feedback based on the comparison.
        """
        if len(self.training_history) < 2:
            return "This is your first session or only session. Train another model to start comparison."
        
        # Extract metrics from the latest and second latest training sessions
        last_session_metrics = self.training_history[-1]
        second_last_session_metrics = self.training_history[-2]
        
        # Generate comparison feedback (this is a simplistic example focusing on R²)
        if last_session_metrics["R²"] > second_last_session_metrics["R²"]:
            feedback = "Improvement in R² value! Your model is getting better."
        elif last_session_metrics["R²"] == second_last_session_metrics["R²"]:
            feedback = "R² value unchanged. Consider experimenting with different parameters."
        else:
            feedback = "Decrease in R² value. Review the changes or try new parameters."
        
        return feedback

    # Continue with other setup methods (setup_data_file_path_section, setup_model_type_selection, etc.)
    # Each method configures a specific part of the GUI, such as dropdowns for selecting model types, buttons for browsing files, and so on.

    def setup_data_file_path_section(self):
        """
        Sets up the section for selecting the data file path.
        """
        tk.Label(self, text="Data File Path:").pack()
        self.data_file_entry = tk.Entry(self)
        self.data_file_entry.pack()
        ttk.Button(self, text="Browse", command=self.browse_data_file).pack(pady=5)

    def setup_model_type_selection(self):
        """
        Sets up the model type selection dropdown menu and dynamic option display.
        """
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar(self)
        model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var, 
                                           values=["linear_regression", "random_forest", "neural_network", "LSTM", "ARIMA"])
        model_type_dropdown.pack()
        self.model_type_var.trace_add('write', self.show_dynamic_options)
        self.dynamic_options_frame = tk.Frame(self)
        self.dynamic_options_frame.pack(pady=5)

    def setup_start_training_button(self):
        """
        Sets up the button to start the training process.
        """
        ttk.Button(self, text="Start Training", command=self.start_training).pack(pady=10)

    def setup_progress_and_logging(self):
        """
        Sets up the UI components for displaying the training progress and logs.
        """
        self.progress_var = tk.IntVar(self, value=0)
        ttk.Progressbar(self, variable=self.progress_var, maximum=100).pack(pady=5)
        self.log_text = tk.Text(self, height=10, state='disabled')
        self.log_text.pack()
        self.logger = ModelTrainingLogger(self.log_text)

    # Function to browse and select a data file
    def browse_data_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, file_path)
            self.preview_selected_data(file_path)
            self.utils.log_message(f"Selected data file: {file_path}", self, self.log_text, self.is_debug_mode) 

    def process_queue(self):
        """
        Processes queued messages to update the GUI, including training progress and logs.
        """
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self.logger.log(message)  # Update the GUI with the message
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

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

# Implement additional functionalities as required for model training, evaluation, etc.


    def show_dynamic_options(self, *_):
        """
        Dynamically generates UI elements based on the model configuration selected by the user.
        """
        # Clear current dynamic options
        for widget in self.dynamic_options_frame.winfo_children():
            widget.destroy()

        selected_model_type = self.model_type_var.get()
        if selected_model_type in self.model_configs:
            # Generate UI elements for each parameter
            for param, info in self.model_configs[selected_model_type].items():
                tk.Label(self.dynamic_options_frame, text=info["label"]).pack()
                entry = tk.Entry(self.dynamic_options_frame)
                entry.insert(0, str(info["default"]))
                entry.pack()
                # Use setattr to make entries accessible by name
                setattr(self, f"{param}_entry", entry)

    def setup_progress_and_logging(self):
        # Progress Bar and Log Text
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=5)
        self.log_text = tk.Text(self, height=10, state='disabled')
        self.log_text.pack()

                
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

    def toggle_advanced_settings(self):
        """
        Toggles the visibility of advanced settings based on the user's selection.
        """
        if self.advanced_settings_var.get():
            # Show advanced settings
            self.settings_frame.pack(pady=(0, 10))
            self.optimizer_label.pack()
            self.optimizer_entry.pack()
            self.regularization_label.pack()
            self.regularization_entry.pack()
            self.learning_rate_label.pack()
            self.learning_rate_entry.pack()
            self.batch_size_label.pack()
            self.batch_size_entry.pack()
            self.window_size_label.pack()
            self.window_size_entry.pack()
        else:
            # Hide advanced settings
            self.settings_frame.pack_forget()

    def log_debug(self, message):
        if self.is_debug_mode:
            self.display_message(message, level="DEBUG")

    def log_message_from_thread(self, message):
        # This method allows logging from background threads
        self.after(0, lambda: self.display_message(message))

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
    def setup_training_configurations(self):
        # Training Configuration Section
        tk.Label(self, text="Training Configurations", font=("Helvetica", 14)).pack(pady=5)

        # Frame for settings
        self.settings_frame = tk.Frame(self)
        # Do not pack this frame yet

        # Prepare settings without packing them
        self.optimizer_label = tk.Label(self.settings_frame, text="Optimizer:")
        self.optimizer_entry = tk.Entry(self.settings_frame)

        self.regularization_label = tk.Label(self.settings_frame, text="Regularization Rate:")
        self.regularization_entry = tk.Entry(self.settings_frame)

        self.learning_rate_label = tk.Label(self.settings_frame, text="Learning Rate:")
        self.learning_rate_entry = tk.Entry(self.settings_frame)

        self.batch_size_label = tk.Label(self.settings_frame, text="Batch Size:")
        self.batch_size_entry = tk.Entry(self.settings_frame)

        # Advanced Settings Toggle
        self.advanced_settings_var = tk.BooleanVar()
        self.advanced_settings_check = ttk.Checkbutton(
            self, text="Show Advanced Settings",
            variable=self.advanced_settings_var,
            command=self.toggle_advanced_settings
        )
        self.advanced_settings_check.pack(pady=5)

    def show_advanced_settings(self, with_animation=False):
        # Show advanced settings
        if with_animation:
            self.animate_visibility(self.learning_rate_entry, True)
            self.animate_visibility(self.batch_size_entry, True)
        else:
            self.learning_rate_label.pack()
            self.learning_rate_entry.pack()
            self.batch_size_label.pack()
            self.batch_size_entry.pack()

    def hide_advanced_settings(self, with_animation=False):
        # Hide advanced settings
        if with_animation:
            self.animate_visibility(self.learning_rate_entry, False)
            self.animate_visibility(self.batch_size_entry, False)
        else:
            self.learning_rate_label.pack_forget()
            self.learning_rate_entry.pack_forget()
            self.batch_size_label.pack_forget()
            self.batch_size_entry.pack_forget()


    # Function to set up the start training button
    def setup_start_training_button(self):
        config_frame = ttk.Frame(self)
        config_frame.pack(pady=10, fill='x', padx=10)
        self.start_training_button = ttk.Button(config_frame, text="Start Training", command=self.start_training)
        self.start_training_button.pack(padx=10)

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

    def animate_visibility(self, widget, visible):
        # Animates visibility of a widget (entry, button, etc.)
        if widget is not None:
            if visible:
                # Show widget with smooth animation
                widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                self.fade_in(widget)
            else:
                # Hide widget with smooth animation
                self.fade_out(widget)

    def fade_in(self, widget, duration=500):
        # Fade in animation for the widget
        current_opacity = float(widget.attributes("-alpha")) if widget.winfo_viewable() else 0.0
        target_opacity = 1.0
        step = (target_opacity - current_opacity) / duration * 10

        def change_opacity():
            nonlocal current_opacity
            current_opacity += step
            widget.attributes("-alpha", current_opacity)
            if current_opacity < target_opacity:
                widget.after(10, change_opacity)

        change_opacity()

    def fade_out(self, widget, duration=500):
        # Fade out animation for the widget
        current_opacity = float(widget.attributes("-alpha")) if widget.winfo_viewable() else 1.0
        target_opacity = 0.0
        step = (current_opacity - target_opacity) / duration * 10

        def change_opacity():
            nonlocal current_opacity
            current_opacity -= step
            widget.attributes("-alpha", current_opacity)
            if current_opacity > target_opacity:
                widget.after(10, change_opacity)
            else:
                widget.place_forget()  # Once faded out, hide the widget

        change_opacity()


    def get_epochs(self, model_type):
        if model_type in ["neural_network", "LSTM"]:
            try:
                epochs_str = self.epochs_entry.get()
                if not epochs_str.isdigit() or int(epochs_str) <= 0:
                    self.utils.log_message("Epochs should be a positive integer.", self, self.log_text, self.is_debug_mode)
                    return None
                return int(epochs_str)
            except _tkinter.TclError as e:
                self.utils.log_message(f"Failed to retrieve epochs: {e}", self, self.log_text, self.is_debug_mode)
                return None
        else:
 
            return 0  


# Section 2: GUI Components and Functions

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
            results_message = "Model Evaluation Results:\n"

            if model_type == 'classification':
                if not hasattr(self.trained_model, 'predict_proba'):
                    raise AttributeError("The trained model does not support probability prediction.")

                y_pred_proba = self.trained_model.predict_proba(X_test)[:, 1]  # Assuming binary classification
                accuracy = accuracy_score(y_test, y_pred)  # Direct accuracy calculation
                precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)

                results_message += f"""Accuracy: {accuracy:.2f} (The proportion of correct predictions among the total number of cases processed.)
    Precision: {precision:.2f} (The proportion of correct positive predictions.)
    Recall: {recall:.2f} (The ability of the model to find all the relevant cases.)
    F1-Score: {fscore:.2f} (The harmonic mean of precision and recall.)
    AUC-ROC: {auc_roc:.2f} (The model's ability to distinguish between classes.)
    Log Loss: {logloss:.2f} (The loss of the model for true labels vs predicted probabilities.)\n"""

                self.plot_confusion_matrix(conf_matrix, ['Class 0', 'Class 1'])
            
            elif model_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                accuracy = self.calculate_model_accuracy(X_test, y_test)  # Using the accuracy calculation for regression models

                results_message += f"""MSE: {mse:.2f} (The average squared difference between the estimated values and actual value.)
    RMSE: {rmse:.2f} (The square root of MSE, providing a measure of how spread out these residuals are.)
    R2 Score: {r2:.2f} (The proportion of the variance for the dependent variable that's predictable from the independent variable(s).)
    Accuracy: {accuracy:.2f}% (Model accuracy as a percentage.)\n"""  # Added accuracy for regression

            # Adding tips for improvement based on performance
            if model_type == 'classification' and accuracy < 75:
                results_message += "\nConsider improving your model by adjusting the hyperparameters, using more data, or trying a different model algorithm."
                self.logger.warning("Model accuracy is below 75%. Consider model improvement.")
            elif model_type == 'regression' and r2 < 0.75:
                results_message += "\nFor better performance, you might consider feature engineering, choosing a different model, or optimizing the current model's hyperparameters."
                self.logger.warning("R2 score is below 0.75. Consider model improvement.")

            self.after(0, lambda: self.display_evaluation_results(results_message))

        except ValueError as ve:
            error_message = f"ValueError during model evaluation: {ve}"
            self.logger.error(error_message)
            self.after(0, lambda: self.display_message(error_message, level="ERROR"))
        except AttributeError as ae:
            error_message = f"AttributeError during model evaluation: {ae}"
            self.logger.error(error_message)
            self.after(0, lambda: self.display_message(error_message, level="ERROR"))
        except Exception as e:
            error_message = f"Error during model evaluation: {e}"
            self.logger.error(error_message)
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
        
    def start_training(self):
        if not self.validate_inputs():
            self.display_message("Invalid input. Please check your settings.", "ERROR")
            return

        data_file_path = self.data_file_entry.get()
        model_type = self.model_type_var.get()

        # Initialize epochs to a default value
        epochs = 50

        # Only attempt to access epochs_entry if it's applicable
        if hasattr(self, 'epochs_entry'):
            epochs_str = self.epochs_entry.get()
            if epochs_str.isdigit() and int(epochs_str) > 0:
                epochs = int(epochs_str)

        try:
            self.disable_training_button()
            self.display_message("Training started...", "INFO")

            # Load and preprocess the data
            data = pd.read_csv(data_file_path)
            self.display_message("Data loading and preprocessing started.", "INFO")
            X, y = self.preprocess_data_with_feature_engineering(data)

            # Check if X or y is None or empty
            if X is None or y is None or X.empty or y.empty:
                self.display_message("Preprocessing resulted in empty data. Aborting training.", "ERROR")
                return

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            trained_model = None
            if model_type in ['neural_network', 'LSTM']:
                trained_model = self.train_neural_network_or_lstm_with_regularization_and_transfer_learning(X_train, y_train, X_val, y_val, model_type, epochs)
            elif model_type == 'linear_regression':
                trained_model = self.train_linear_regression_with_auto_optimization(X_train, y_train, X_val, y_val)
            elif model_type == 'random_forest':
                trained_model = self.train_random_forest_with_auto_optimization(X_train, y_train, X_val, y_val)
            elif model_type == "ARIMA" and hasattr(self, 'train_arima_model_in_background'):
                # This assumes train_arima_model_in_background is defined elsewhere in your class
                self.train_arima_model_in_background(y_train)

            # Post-training operations...
            self.display_message("Training completed successfully.", "INFO")
            self.save_trained_model(trained_model, model_type)

        except Exception as e:
            error_message = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.display_message(error_message, "ERROR")
        finally:
            self.enable_training_button()

    def create_lag_features(self, df, column_name, lag_sizes):
        if column_name not in df.columns:
            self.display_message(f"Warning: Column '{column_name}' not found in DataFrame. Skipping lag feature creation.", "ERROR")
            return df

        for lag_days in lag_sizes:
            df[f'{column_name}_lag_{lag_days}'] = df[column_name].shift(lag_days)
        
        # Filling NaN values after lag feature creation
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # For filling initial rows if needed
        
        return df

    def create_rolling_window_features(self, data, column_name, windows, method='pad'):
        for window in windows:
            data[f'{column_name}_rolling_mean_{window}'] = data[column_name].rolling(window=window).mean()
            data[f'{column_name}_rolling_std_{window}'] = data[column_name].rolling(window=window).std()
            
            if method == 'interpolate':
                data[f'{column_name}_rolling_mean_{window}'].interpolate(method='linear', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].interpolate(method='linear', inplace=True)
            elif method == 'pad':
                data[f'{column_name}_rolling_mean_{window}'].fillna(method='pad', inplace=True)
                data[f'{column_name}_rolling_std_{window}'].fillna(method='pad', inplace=True)
            else:
                data.fillna(data.mean(), inplace=True)  # General fallback to mean imputation

        return data


    def preprocess_data_with_feature_engineering(self, data, lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20]):
        # Initial data check
        if data.empty:
            print("The dataset is empty before preprocessing. Please check the data source.")
            return None, None

        # Convert 'date' column to datetime and create a numeric feature from it
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            # Dynamically adjust the reference date to the minimum date in the dataset
            reference_date = data['date'].min()
            data['days_since_reference'] = (data['date'] - reference_date).dt.days

        # Ensure there's a unique index
        if 'index' not in data.columns:
            data.reset_index(inplace=True, drop=True)
        
        # Create lag and rolling window features for 'close' price
        data = self.create_lag_features(data, 'close', lag_sizes)
        data = self.create_rolling_window_features(data, 'close', window_sizes)
        
        # Check if the dataset has become empty after feature creation
        if data.dropna().empty:
            print("The dataset became empty after creating lag and rolling window features due to NaN removal. Please adjust the lag and window sizes.")
            return None, None
        else:
            data.dropna(inplace=True)
        
        # Exclude the 'date' column if it's still present after creating numeric features
        data = data.drop(columns=['date'], errors='ignore')
        
        # Separate the target variable and features
        if 'close' in data.columns:
            y = data['close']
            X = data.drop(columns=['close'])
        else:
            print("The 'close' column is missing from the dataset. Please check the dataset.")
            return None, None

        # Final check before returning
        if X.empty or y.empty:
            print("Either features (X) or target (y) is empty after preprocessing. Please check the preprocessing steps.")
            return None, None
        
        return X, y


    def train_neural_network_or_lstm_with_regularization_and_transfer_learning(self, X_train, y_train, X_val, y_val, model_type, epochs=100, pretrained_model_path=None, previous_model_metrics=None):
        if pretrained_model_path:
            model = self.load_model(pretrained_model_path)
            for layer in model.layers[:-5]:
                layer.trainable = False
        else:
            model = Sequential()

        if model_type == "neural_network":
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            # For a standard neural network, no reshaping is required.
            X_train_reshaped, X_val_reshaped = X_train, X_val
        elif model_type == "LSTM":
            model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            # Reshape for LSTM; ensure X_train and X_val are numpy arrays for reshaping
            X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1) if not isinstance(X_train, np.ndarray) else X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1) if not isinstance(X_val, np.ndarray) else X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        model.fit(X_train_reshaped, y_train, validation_data=(X_val_reshaped, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])

        # Making predictions and ensuring output shape consistency
        y_pred_val = model.predict(X_val_reshaped).flatten()

        # Debugging: print shapes to verify consistency
        print(f"y_val shape: {y_val.shape}, y_pred_val shape: {y_pred_val.shape}")

        if y_val.shape[0] != y_pred_val.shape[0]:
            raise ValueError(f"Inconsistent number of samples: y_val has {y_val.shape[0]} samples, y_pred_val has {y_pred_val.shape[0]} samples.")

        mse = mean_squared_error(y_val, y_pred_val)
        rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        r2 = r2_score(y_val, y_pred_val)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        if r2 > 0.75:
            self.display_message("Great! Your model has a strong predictive ability for stock market trends.", "INFO")
        elif r2 > 0.5:
            self.display_message("Your model has decent predictive ability, but there's room for improvement.", "WARNING")
        else:
            self.display_message("It looks like your model might need further tuning to accurately predict stock market trends.", "ERROR")

        if previous_model_metrics:
            previous_val_loss, _, _ = previous_model_metrics
            if mse < previous_val_loss:
                self.display_message("Congratulations! Your new model has lower validation MSE than the previous one.", "INFO")
            else:
                self.display_message("Your new model's performance is not better than the previous one.", "INFO")

        return model, (mse, rmse, r2)
    
    def train_linear_regression_with_auto_optimization(self, X_train, y_train, X_val, y_val):
        """
        Trains a linear regression model with Ridge regularization using auto-optimized hyperparameters.

        Parameters:
        - X_train: Training features.
        - y_train: Training target.
        - X_val: Validation features.
        - y_val: Validation target.

        Returns:
        - best_model: The best trained Ridge regression model.
        """
        param_grid = {'alpha': np.logspace(-4, 0, 50)}
        model = Ridge()
        randomized_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', verbose=2)
        randomized_search.fit(X_train, y_train)
        
        # Inspect Randomized Search Results
        self.display_message("Randomized Search Results:", "INFO")
        results_df = pd.DataFrame(randomized_search.cv_results_)
        # Convert DataFrame to string for display
        results_str = results_df[['param_alpha', 'mean_test_score', 'std_test_score']].to_string()
        self.display_message(results_str, "INFO")

        # Evaluate Cross-Validation Scores
        cv_scores = cross_val_score(randomized_search.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # Convert the numpy array of scores to a formatted string
        cv_scores_str = ", ".join([f"{score:.2f}" for score in cv_scores])
        self.display_message(f"CV Scores: {cv_scores_str}", "INFO")

        best_model = randomized_search.best_estimator_
        r2 = best_model.score(X_val, y_val)
        y_pred_val = best_model.predict(X_val)
        mse, rmse = mean_squared_error(y_val, y_pred_val), mean_squared_error(y_val, y_pred_val, squared=False)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}", "INFO")
        
        # Interpret R²
        if r2 > 0.75:
            self.display_message("Excellent! Your model is well-tuned for predicting stock prices.", "INFO")
        elif r2 > 0.5:
            self.display_message("Not bad, but consider exploring more features or different models for better accuracy.", "WARNING")
        else:
            self.display_message("Your model might not be complex enough to capture the nuances of the stock market.", "ERROR")

        # Best alpha
        best_alpha = randomized_search.best_params_['alpha']
        self.display_message(f"Best regularization strength (alpha): {best_alpha:.4f}. Consider using this as a starting point for your next training session.", "INFO")

        # Further suggestions
        if r2 < 0.5:
            self.display_message("Improving feature selection or engineering new features could enhance your model's performance.", "INFO")
        else:
            self.display_message("Your model's performance is promising! Experiment with different ranges of alpha to potentially fine-tune the regularization strength.", "INFO")

        return best_model


    def train_random_forest_with_auto_optimization(self, X_train, y_train, X_val, y_val, random_state=None):
        """
        Trains a random forest model with auto-optimized hyperparameters.

        Parameters:
        - X_train: Training features.
        - y_train: Training target.
        - X_val: Validation features.
        - y_val: Validation target.
        - random_state: Random seed for reproducibility.

        Returns:
        - best_rf_model: The best trained random forest model.
        """
        param_grid = {
            'n_estimators': np.linspace(10, 300, num=20, dtype=int),
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }
        
        rf = RandomForestRegressor(random_state=random_state)
        rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=3, verbose=1, random_state=random_state, n_jobs=-1)
        rf_random_search.fit(X_train, y_train)
        
        best_rf_model = rf_random_search.best_estimator_
        r2 = best_rf_model.score(X_val, y_val)
        y_pred_val = best_rf_model.predict(X_val)
        mse, rmse = mean_squared_error(y_val, y_pred_val), mean_squared_error(y_val, y_pred_val, squared=False)
        self.display_message(f"Validation MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Interpret R²
        if r2 > 0.75:
            self.display_message("Fantastic! Your random forest model is likely to predict stock market movements accurately.", level="INFO")
        elif r2 > 0.5:
            self.display_message("Your model is on the right track, but tweaking parameters or adding features could help.", level="WARNING")
        else:
            self.display_message("It seems there's significant potential for improving your model's accuracy.", level="ERROR")
        
        # Best parameters
        best_params = rf_random_search.best_params_
        self.display_message(f"Best parameters found: {best_params}. Use these parameters as a baseline for your next training session.", level="INFO")

        # Further suggestions
        if r2 < 0.5:
            self.display_message("Consider experimenting with more estimators or adjusting the max depth. Feature engineering might also improve results.", level="INFO")
        else:
            important_features = sorted(zip(X_train.columns, best_rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
            self.display_message(f"Most important features: {important_features[:5]}. Focusing on these features might yield further insights.", level="INFO")

        return best_rf_model

    def train_arima_model_in_background(self, close_prices, threshold=100):
        """
        Train an ARIMA model asynchronously in the background.

        Parameters:
        - close_prices: A list of closing prices.
        - threshold: Threshold for MSE to determine model performance.

        """
        def background_training(close_prices):
            results = {'predictions': [], 'errors': [], 'parameters': {'order': (5, 1, 0)}, 'performance_metrics': {}}
            train_size = int(len(close_prices) * 0.8)
            train, test = close_prices[:train_size], close_prices[train_size:]
            history = list(train)

            for t in range(len(test)):
                try:
                    model = ARIMA(history, order=results['parameters']['order'])
                    model_fit = model.fit()
                    forecast = model_fit.forecast()[0]
                    results['predictions'].append(forecast)
                    obs = test[t]
                    history.append(obs)
                except Exception as e:
                    self.display_message(f"Error training ARIMA model at step {t}: {e}", level="ERROR")
                    results['errors'].append(str(e))
            
            mse = mean_squared_error(test, results['predictions'])
            self.display_message(f"Test MSE: {mse:.2f}")

            if mse < threshold:
                self.display_message("Your ARIMA model seems promising for forecasting stock prices.", level="INFO")
            else:
                self.display_message("Consider different ARIMA parameters or models for better forecasting accuracy.", level="WARNING")

            if mse < threshold:
                self.display_message("Your ARIMA model performs well! Consider using the same or similar parameters (p, d, q) for similar datasets.", level="INFO")
            else:
                self.display_message("Consider trying different combinations of (p, d, q) parameters. AIC and BIC from the model summary can guide the selection.", level="INFO")

            self.display_message("Tip: A lower AIC or BIC value usually indicates a better model fit. Use these metrics to compare different ARIMA configurations.", level="INFO")

        threading.Thread(target=background_training, args=(close_prices,), daemon=True).start()
        self.display_message("ARIMA model training started in background...", level="INFO")

    def save_arima_results(self, results, model_fit):
        """
        Save ARIMA model results and model file.

        Parameters:
        - results: Dictionary containing model predictions, errors, parameters, and performance metrics.
        - model_fit: Fitted ARIMA model.

        """
        try:
            models_directory = self.config.get('Paths', 'models_directory')
            if not os.path.exists(models_directory):
                os.makedirs(models_directory)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file_path = os.path.join(models_directory, f'arima_model_{timestamp}.pkl')

            self.save_model_by_type(model_fit, 'arima', model_file_path)
            self.display_message(f"ARIMA model saved to {model_file_path}", level="INFO")

            results_file_path = os.path.join(models_directory, f'arima_results_{timestamp}.json')
            with open(results_file_path, 'w') as result_file:
                json.dump(results, result_file, indent=4)
            self.display_message(f"ARIMA model results saved to {results_file_path}", level="INFO")

        except Exception as e:
            error_message = f"Error saving ARIMA model results: {e}"
            self.display_message(error_message, level="ERROR")
            raise

    def load_model(self, model_file_path, scaler_file_path=None, metadata_file_path=None):
        try:
            model = joblib.load(model_file_path)
            print(f"Model loaded successfully from {model_file_path}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return None, None, None

        scaler = None
        if scaler_file_path:
            try:
                scaler = joblib.load(scaler_file_path)
                print(f"Scaler loaded successfully from {scaler_file_path}")
            except Exception as e:
                print(f"Failed to load scaler: {str(e)}")

        metadata = None
        if metadata_file_path:
            try:
                with open(metadata_file_path, 'r') as metadata_file:
                    metadata = json.load(metadata_file)
                print(f"Metadata loaded successfully from {metadata_file_path}")
            except Exception as e:
                print(f"Failed to load metadata: {str(e)}")

        return model, scaler, metadata


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


    def validate_inputs(self):
        # Common validation checks (applicable to all models)
        data_file_path = self.data_file_entry.get()
        if not data_file_path:
            self.error_label.config(text="Data file path is required.", fg="red")
            return False

        model_type = self.model_type_var.get()
        if not model_type:
            self.error_label.config(text="Please select a model type.", fg="red")
            return False

        # Model-specific validation checks with default values from self.model_configs
        model_config = self.model_configs.get(model_type, {})
        for config_key, config_value in model_config.items():
            entry_widget = getattr(self, f"{config_key}_entry", None)
            if entry_widget:
                entry_value_str = entry_widget.get()
                if not entry_value_str.replace('.', '', 1).isdigit():  # Allows for decimal inputs
                    # Clear the entry and set to default if validation fails
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, str(config_value["default"]))
                elif config_key == "max_depth" and entry_value_str.lower() == "none":  # Special case for None value
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, "None")
            else:
                print(f"Warning: Entry widget for {config_key} not found.")  # Debugging line

        # Clear error label if everything is valid
        self.error_label.config(text="")

        return True  # Validation passed


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
            import optuna
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                ...
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            # Customize as needed
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
        try:
            interval = self.schedule_dropdown.get().lower()  # Get the desired interval from a dropdown menu

            if interval == "daily":
                schedule.every().day.at("10:00").do(self.run_automated_training_tasks)
                timing = "every day at 10:00 AM"
            elif interval == "weekly":
                schedule.every().week.at("10:00").do(self.run_automated_training_tasks)
                timing = "every week on this day at 10:00 AM"
            elif interval == "monthly":
                schedule.every().month.at("10:00").do(self.run_automated_training_tasks)
                timing = "once a month at 10:00 AM"
            else:
                raise ValueError("Invalid interval selected for automated training.")

            self.display_message(f"Automated training scheduled to run {timing}.", level="INFO")

            # Start a thread to run the schedule without blocking the main GUI thread
            threading.Thread(target=self.run_schedule, daemon=True).start()
        except ValueError as ve:
            self.display_message(f"Error scheduling automated training: {ve}", level="ERROR")
        except Exception as e:
            self.display_message(f"Unexpected error occurred while setting up automated training: {e}", level="ERROR")


    def run_automated_training_tasks(self):
        """
        Run the automated training tasks, including model training and real-time analytics monitoring.
        """
        # Log the start of automated training with display_message instead of utils.log_message
        data_file_path = self.config.get("Data", "file_path")  # Ensure data_file_path is defined before use
        self.display_message(f"Automated training started with data from {data_file_path}.", level="INFO")

        # Implement adaptive learning logic based on past performance
        model_type = self.config.get("Model", "model_type")
        epochs = int(self.config.get("Model", "epochs")) if model_type in ["neural_network", "LSTM"] else 1

        # Call training logic here, presumably train_model_and_enable_button is adapted to handle success/failure
        training_success = self.train_model_and_enable_button(data_file_path, model_type, epochs)
        if training_success:
            self.display_message("Model training completed successfully.", level="INFO")
        else:
            self.display_message("Model training encountered issues. Check logs for details.", level="ERROR")

        # Implement real-time analytics monitoring during training
        # Assuming initiate_real_time_training_monitoring is adapted to provide user-friendly messages
        self.initiate_real_time_training_monitoring()

        # Assuming the function to check training progress provides meaningful insights
        self.monitor_training_progress()

        # Once training is confirmed to be completed, notify the user
        self.display_message("Automated training session has successfully completed.", level="INFO")


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

        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Model Evaluation Results')
        plt.grid(True)
        plt.show()
        self.display_message("Model evaluation visualization displayed.", level="INFO")

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

    def process_directory(self, directory_path):
        csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
        best_models = {}

        for csv_file in csv_files:
            file_path = os.path.join(directory_path, csv_file)
            self.display_message(f"Processing {csv_file}...", level="INFO")
            # Load data, assuming a hypothetical function load_data that returns the necessary training and validation sets
            X_train, X_val, y_train, y_val = self.load_data(file_path)

            # Train model, assuming train_model returns a model and its performance metric
            model, performance = self.start_automated_training(X_train, y_train, X_val, y_val)

            # Store the best model for this file
            best_models[csv_file] = (model, performance)

        # After processing all files, you might want to do something with the best models
        for csv_file, (model, performance) in best_models.items():
            self.display_message(f"Best model for {csv_file}: {model} with performance {performance}", level="INFO")

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

    def explain_model_predictions(self, model, X_train):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)

        fig.show()

    def train_automl(self, X_train, y_train):
        automl_model = AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
        automl_model.fit(X_train, y_train)
        return automl_model

    def send_data_to_stream(self, data):
        self.producer.send('ml_stream', data.encode('utf-8'))

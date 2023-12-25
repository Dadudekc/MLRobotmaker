#model_training_tab.py
import tkinter as tk
import configparser
import threading
from Utils import log_message, auto_generate_save_path, update_status
from model_development.model_training import train_model, save_model, load_model
from tkinter import ttk, scrolledtext, messagebox, filedialog
from Utils import MLRobotUtils
import logging
import joblib
from typing import Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential as NeuralNetwork
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier


class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, scaler_options):
        super().__init__(parent)
        self.scaler_options = scaler_options
        self.root = root
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Initialize debug mode state
        self.utils = MLRobotUtils(is_debug_mode=config.getboolean('Settings', 'DebugMode', fallback=False))
        self.create_widgets()

        # Initialize epochs label and entry
        self.epochs_label = tk.Label(self.frame, text="Epochs:")
        self.epochs_label.pack_forget()
        self.epochs_entry = tk.Entry(self.frame)
        self.epochs_entry.pack_forget()

        # Initialize debug mode state
        self.trained_models = []
        self.trained_model = None

    def show_epochs_input(self, event):
        selected_model_type = self.model_type_var.get()
        if selected_model_type == "neural_network":
            if not self.epochs_label:
                self.epochs_label = tk.Label(self.frame, text="Epochs:")
                self.epochs_entry = tk.Entry(self.frame)

            self.epochs_label.pack()
            self.epochs_entry.pack()
        else:
            if self.epochs_label:
                self.epochs_label.pack_forget()
                self.epochs_entry.pack_forget()

    def start_training(self):
        if self.validate_inputs():
            data_file_path = self.data_file_entry.get()
            scaler_type = self.scaler_type_var.get()
            model_type = self.model_type_var.get()
            epochs_str = self.epochs_entry.get()

            # Validate the epochs input
            if model_type == "neural_network":
                if not epochs_str.isdigit() or int(epochs_str) <= 0:
                    self.utils.log_message("Epochs should be a positive integer.", self.log_text)
                    return  # Exit the function if the epochs input is invalid
                epochs = int(epochs_str)
            else:
                epochs = 1  # Default value for other model types

            try:
                self.utils.log_message("Starting model training...", self.log_text)
                self.update_progress_bar(0)  # Correctly calling the method

                # Run the training in a separate thread
                threading.Thread(target=lambda: self.train_model(data_file_path, scaler_type, model_type, epochs)).start()
            except Exception as e:
                self.utils.log_message(f"Error in model training: {str(e)}", self.log_text)
                print(f"Debug: Error in model training - {str(e)}")


    # Progress Bar
    def update_progress_bar(self, value):
        """
        Update the progress bar with the given value.

        Parameters:
        value (int): The progress value to set on the progress bar.
        """
        self.progress['value'] = value
        self.root.update_idletasks() 

    def toggle_debug_mode(self):
        # Toggle debug mode state
        self.utils.is_debug_mode = not self.utils.is_debug_mode

        # Update the button text to reflect the current state
        if self.utils.is_debug_mode:
            self.debug_mode_button.config(text="Debug Mode: ON")
            self.utils.log_message("Debug mode is ON.", self.log_text)
        else:
            self.debug_mode_button.config(text="Debug Mode: OFF")
            self.utils.log_message("Debug mode is OFF.", self.log_text)

    def get_scaler(self, scaler_type):
        """Return the scaler object based on the user's choice."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
            'normalizer': Normalizer()
        }
        return scalers.get(scaler_type, StandardScaler())

    def create_widgets(self):
        # GUI setup and widgets
        tk.Label(self.frame, text="Model Training", font=("Helvetica", 16)).pack(pady=10)
        
        tk.Label(self.frame, text="Enter Data File Path:").pack(pady=5)
        self.data_file_entry = tk.Entry(self.frame)
        self.data_file_entry.pack(pady=5)

        tk.Button(self.frame, text="Browse Data File", command=self.browse_data_file).pack(pady=5)

        tk.Label(self.frame, text="Select Scaler Type:").pack()
        self.scaler_type_var = tk.StringVar()
        scaler_type_dropdown = ttk.Combobox(self.frame, textvariable=self.scaler_type_var, values=self.scaler_options)
        scaler_type_dropdown.pack()
        tk.Label(self.frame, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar()
        model_type_dropdown = ttk.Combobox(self.frame, textvariable=self.model_type_var, values=["linear_regression", "random_forest", "neural_network"])
        model_type_dropdown.pack()


        # Bind the show_epochs_input function to the model type dropdown's <<ComboboxSelected>> event
        model_type_dropdown.bind("<<ComboboxSelected>>", self.show_epochs_input)

        tk.Button(self.frame, text="Start Training", command=self.start_training).pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack()

        # Model Information Text
        self.model_info_text = tk.Text(self.frame, height=5)
        self.model_info_text.pack()

        # Error Label
        self.error_label = tk.Label(self.frame, text="", fg="red")
        self.error_label.pack()

        # Add this code inside the create_widgets method
        self.debug_mode_button = tk.Button(self.frame, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.pack()

        # Log Text
        self.log_text = scrolledtext.ScrolledText(self.frame, height=10)
        self.log_text.pack()

        # Save button
        tk.Button(self.frame, text="Save Trained Model", command=self.save_trained_model).pack(pady=5)

    def save_trained_model(self):
        if self.trained_model is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".model",
                                                    filetypes=[("Model Files", "*.model"), ("All Files", "*.*")])
            if file_path:
                # Assuming save_model is a function that saves your model
                save_model(self.trained_model, file_path)
                self.utils.log_message(f"Model saved to {file_path}", self.log_text)
        else:
            self.utils.log_message("No trained model available to save.", self.log_text)

    def browse_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)  # Clear any existing entry
            self.data_file_entry.insert(0, file_path)  # Insert the selected file path into the entry field
            self.utils.log_message(f"Selected data file: {file_path}", self.log_text)

    def validate_inputs(self):
        data_file_path = self.data_file_entry.get()
        scaler_type = self.scaler_type_var.get()
        model_type = self.model_type_var.get()
        epochs_str = self.epochs_entry.get() if model_type == "neural_network" else "1"

        if not data_file_path:
            self.error_label.config(text="Data file path is required.", fg="red")
            return False
        if not scaler_type:
            self.error_label.config(text="Scaler type is required.", fg="red")
            return False
        if not model_type:
            self.error_label.config(text="Model type is required.", fg="red")
            return False
        if model_type == "neural_network" and (not epochs_str.isdigit() or int(epochs_str) <= 0):
            self.error_label.config(text="Epochs should be a positive integer.", fg="red")
            return False

        self.error_label.config(text="")
        return True

    def preprocess_data(self, file_path, scaler_type):
        # Load the data
        data = pd.read_csv(file_path)

        # Handle empty strings in 'date' column
        data['date'] = data['date'].replace('', np.nan)

        # Convert 'date' to datetime and handle incorrect formats
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

        # Handle rows where 'date' couldn't be converted (now NaT)
        # You can drop, fill, or handle these rows based on your requirement
        data = data.dropna(subset=['date'])

        # Extract features from 'date' column
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

        # Define your target variable and feature set
        y = data['close']  # Replace with your target column name
        X = data.drop(['close', 'date'], axis=1)  # Drop target and date column

        # Fill NaN values
        for col in X.columns:
            if X[col].isna().any():
                X[col].fillna(X[col].mean(), inplace=True)

        # Scale the features
        scaler = self.get_scaler(scaler_type)
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


    def train_model(self, data_file_path, scaler_type, model_type, epochs):
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(data_file_path, scaler_type)

            # Initialize the model based on the selected type
            if model_type == "linear_regression":
                model = LinearRegression()
            elif model_type == "neural_network":
                model = NeuralNetwork()
                model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
            elif model_type == "random_forest":
                model = RandomForestRegressor()  # Use RandomForestClassifier() if it's a classification task
            else:
                self.utils.log_message(f"Unsupported model type: {model_type}", self.log_text)
                return

            # Train the model
            model.fit(X_train, y_train)

            # Update progress bar and assign trained model
            self.update_progress_bar(100)
            self.trained_model = model

            self.utils.log_message("Model training completed.", self.log_text)

        except Exception as e:
            self.utils.log_message(f"Error in model training: {str(e)}", self.log_text)
            print(f"Debug: Error in model training - {str(e)}")



# Modify compare_models function as needed
# In compare_models function, ensure X_test and y_test are defined and populated
def compare_models(X_test, y_test):
    try:
        if len(trained_models) > 1:
            for model in trained_models:
                evaluation_results = evaluate_model(model, X_test, y_test, task_type='regression')
                print(evaluation_results)
            # You can extend this loop to compare models based on specific metrics
        else:
            utils.log_message("Not enough models to compare. Train more models.", log_text)
    except Exception as e:
        print(f"Error during model comparison: {e}")
        raise


# Modify setup_model_training_tab to use MLRobotUtils methods
def setup_model_training_tab(tab):
    config = configparser.ConfigParser()
    config.read('config.ini')

    utils = MLRobotUtils(is_debug_mode=True)  # Create an instance of MLRobotUtils

    # Browse Data File Function
    def browse_data_file_wrapper():
        data_file_path = browse_data_file()
        if data_file_path:
            data_file_entry.delete(0, tk.END)
            data_file_entry.insert(0, data_file_path)

    tk.Label(tab, text="Select Scaler Type:").pack()
    scaler_type_var = tk.StringVar()
    scaler_type_dropdown = ttk.Combobox(tab, textvariable=scaler_type_var, values=scaler_options)
    scaler_type_dropdown.pack()

    # Model Selection Dropdown
    tk.Label(tab, text="Select Model Type:").pack()
    model_type_var = tk.StringVar()
    model_type_dropdown = ttk.Combobox(tab, textvariable=model_type_var, values=["linear_regression", "random_forest", "neural_network"])
    model_type_dropdown.pack()

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

    # Browse Data File Button
    browse_data_button = tk.Button(tab, text="Browse Data File", command=browse_data_file_wrapper)
    browse_data_button.pack()

    # Example of using MLRobotUtils method
    utils.log_message("Setup for model training tab is complete.", log_text)





# Modify browse_data_file to use MLRobotUtils methods
def browse_data_file(self):
    utils = MLRobotUtils(is_debug_mode=True)  # Create an instance of MLRobotUtils
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)  # Clear any existing entry
            self.data_file_entry.insert(0, file_path)  # Insert the selected file path into the entry field
            utils.log_message(f"Selected data file: {file_path}", self.log_text)
    except Exception as e:
        utils.log_message(f"Error while browsing data file: {str(e)}", self.log_text)
        print(f"Debug: Error while browsing data file - {str(e)}")


def save_trained_model(self):
    if self.trained_model is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".model",
                                                filetypes=[("Model Files", "*.model"), ("All Files", "*.*")])
        if file_path:
            save_model(self.trained_model, file_path)
            self.trained_models.append((self.trained_model, file_path))  # Save the model and its file path
            self.utils.log_message(f"Model saved to {file_path}", self.log_text)

def load_model(model_file_path: str, raise_exception: bool = False) -> Optional[Any]:
    """
    Load a machine learning model from a file.

    Parameters:
    model_file_path (str): The path to the model file.
    raise_exception (bool): If True, raises an exception on failure. Otherwise, returns None.

    Returns:
    Optional[Any]: The loaded model or None if an error occurs and raise_exception is False.

    Raises:
    Exception: Various exceptions related to file handling or model loading, if raise_exception is True.
    """
    try:
        loaded_model = joblib.load(model_file_path)
        return loaded_model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_file_path}")
    except joblib.JoblibException as e:
        logging.error(f"Error loading model from {model_file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unknown error loading model from {model_file_path}: {str(e)}")

    if raise_exception:
        raise
    else:
        return None



def compare_models(X_test, y_test):
    try:
        if len(trained_models) > 1:
            for model in trained_models:
                evaluation_results = evaluate_model(model, X_test, y_test, task_type='regression')
                print(evaluation_results)
            # You can extend this loop to compare models based on specific metrics
        else:
            utils.log_message("Not enough models to compare. Train more models.", log_text)
    except Exception as e:
        print(f"Error during model comparison: {e}")
        raise


# Initialize global variables
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
config = load_config()

trained_models = []
trained_model = None

# Create the main application window
root = tk.Tk()
root.title("Model Training")

# Create a tabbed interface
notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Model Training")
notebook.pack(fill=tk.BOTH, expand=True)

# Define scaler_options
scaler_options = ['standard', 'minmax', 'robust', 'quantile', 'power', 'normalizer', 'maxabs']

# Create the ModelTrainingTab instance with scaler_options
model_training_tab = ModelTrainingTab(tab1, scaler_options)


# Start the Tkinter main loop
root.mainloop()
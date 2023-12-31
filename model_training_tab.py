#model_training_tab.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import tkinter as tk
import configparser
import threading
from Utils import log_message, auto_generate_save_path, update_status
from model_development.model_training import train_model, load_model
import joblib
from tensorflow.keras.models import save_model as save_keras_model
import torch
import sklearn
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
from tensorflow.keras.models import Sequential as NeuralNetwork, Model
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import LSTM


class ModelTrainingTab(tk.Frame):
    def __init__(self, parent, config, scaler_options):
        super().__init__(parent)
        self.config = config
        self.scaler_options = scaler_options
        self.trained_models = []
        self.trained_model = None
        self.utils = MLRobotUtils(is_debug_mode=config.getboolean('Settings', 'DebugMode', fallback=False))
        # Initialize Window Size Label and Entry
        self.window_size_label = tk.Label(self, text="Window Size:")
        self.window_size_entry = tk.Entry(self)
        self.trained_scaler = None  # Initialize trained_scaler
        self.setup_model_training_tab()

    def setup_model_training_tab(self):
        # GUI setup and widgets
        tk.Label(self, text="Model Training", font=("Helvetica", 16)).pack(pady=10)

        # Data File Path
        tk.Label(self, text="Enter Data File Path:").pack(pady=5)
        self.data_file_entry = tk.Entry(self)
        self.data_file_entry.pack(pady=5)
        tk.Button(self, text="Browse Data File", command=self.browse_data_file).pack(pady=5)

        # Scaler Type
        tk.Label(self, text="Select Scaler Type:").pack()
        self.scaler_type_var = tk.StringVar()
        scaler_type_dropdown = ttk.Combobox(self, textvariable=self.scaler_type_var, values=self.scaler_options)
        scaler_type_dropdown.pack()

        # Model Type
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar()
        model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var, values=["linear_regression", "random_forest", "neural_network"])
        model_type_dropdown.pack()
        model_type_dropdown.bind("<<ComboboxSelected>>", self.show_epochs_input)

        # Start Training Button
        self.start_training_button = tk.Button(self, text="Start Training", command=self.start_training)
        self.start_training_button.pack(pady=10)


        #Model Info, Error Label, Log Text, Save Button
        self.error_label = tk.Label(self, text="", fg="red")
        self.error_label.pack()
        self.log_text = scrolledtext.ScrolledText(self, height=15)
        self.log_text.pack()
        tk.Button(self, text="Save Trained Model", command=self.save_trained_model).pack(pady=5)

        # Debug Mode Button
        self.debug_mode_button = tk.Button(self, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.pack()

        # Initialize epochs label and entry
        self.epochs_label = tk.Label(self, text="Epochs:")
        self.epochs_label.pack_forget()
        self.epochs_entry = tk.Entry(self)
        self.epochs_entry.pack_forget()
        self.window_size_label.pack_forget()
        self.window_size_entry.pack_forget()


    def show_epochs_input(self, event):
        selected_model_type = self.model_type_var.get()
        if selected_model_type == "neural_network":
            if not self.epochs_label:
                self.epochs_label = tk.Label(self, text="Epochs:")
                self.epochs_entry = tk.Entry(self)

            self.epochs_label.pack()
            self.epochs_entry.pack()
            self.window_size_label.pack()
            self.window_size_entry.pack()

        else:
            if self.epochs_label:
                self.epochs_label.pack_forget()
                self.epochs_entry.pack_forget()

    def start_training(self):
        if not self.validate_inputs():
            return  # Exit if inputs are not valid

        data_file_path = self.data_file_entry.get()
        scaler_type = self.scaler_type_var.get()
        model_type = self.model_type_var.get()
        epochs = self.get_epochs(model_type)

        if epochs is None:
            return  # Exit if epochs are invalid for neural network

        self.start_training_button.config(state='disabled')
        threading.Thread(target=lambda: self.train_model_and_enable_button(data_file_path, scaler_type, model_type, epochs)).start()
        
    def train_model_and_enable_button(self, data_file_path, scaler_type, model_type, epochs):
        try:
            # Load and preprocess data
            window_size = int(self.window_size_entry.get()) if self.model_type_var.get() == "neural_network" else 1
            X_train, X_test, y_train, y_test = self.preprocess_data(data_file_path, scaler_type, model_type, window_size)

            # Initialize and train the model based on the selected type
            if model_type == "linear_regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
            elif model_type == "neural_network":
                model = NeuralNetwork()
                model.add(LSTM(64, activation='relu', input_shape=(window_size, X_train.shape[2])))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=epochs)
            elif model_type == "random_forest":
                model = RandomForestRegressor()
                model.fit(X_train, y_train)

            # Update UI elements to reflect that training is complete
            self.trained_model = model
            self.utils.log_message("Model training completed.", self.log_text)

        except Exception as e:
            # Log any exceptions that occur during model training
            self.utils.log_message(f"Error in model training: {str(e)}", self.log_text)
            print(f"Debug: Error in model training - {str(e)}")
        finally:
            # Re-enable the Start Training button regardless of whether training succeeded or an exception was caught
            self.start_training_button.config(state='normal')


    def get_epochs(self, model_type):
        if model_type != "neural_network":
            return 1  # Default value for other model types

        epochs_str = self.epochs_entry.get()
        if not epochs_str.isdigit() or int(epochs_str) <= 0:
            self.utils.log_message("Epochs should be a positive integer.", self.log_text)
            return None
        return int(epochs_str)

    def initiate_model_training(self, data_file_path, scaler_type, model_type, epochs):
        try:
            self.utils.log_message("Starting model training...", self.log_text)
            threading.Thread(target=lambda: self.train_model(data_file_path, scaler_type, model_type, epochs)).start()
        except Exception as e:
            self.utils.log_message(f"Error in model training: {str(e)}", self.log_text)
            print(f"Debug: Error in model training - {str(e)}")

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

    def save_trained_model(self):
        if self.trained_model is not None:
            model_type = "unknown_model"
            if isinstance(self.trained_model, sklearn.base.BaseEstimator):
                model_type = "sklearn"
                file_extension = ".joblib"
            elif isinstance(self.trained_model, Model):
                model_type = "keras"
                file_extension = ".h5"
            elif isinstance(self.trained_model, torch.nn.Module):
                model_type = "pytorch"
                file_extension = ".pth"
            else:
                self.utils.log_message("Unsupported model type for saving.", self.log_text)
                return

            # Ask for filename from the user
            file_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                                    filetypes=[(f"{model_type.upper()} Files", f"*{file_extension}"), ("All Files", "*.*")])
            if file_path:
                # Save the model
                if file_extension == ".joblib":
                    joblib.dump(self.trained_model, file_path)
                elif file_extension == ".h5":
                    save_keras_model(self.trained_model, file_path)
                elif file_extension == ".pth":
                    torch.save(self.trained_model.state_dict(), file_path)

                self.utils.log_message(f"Model saved to {file_path}", self.log_text)

                # Save the scaler with a more descriptive name
                if self.trained_scaler is not None:
                    scaler_type = type(self.trained_scaler).__name__.lower()
                    scaler_file_path = f"{file_path}_{model_type}_{scaler_type}_scaler.joblib"
                    joblib.dump(self.trained_scaler, scaler_file_path)
                    self.utils.log_message(f"Scaler saved to {scaler_file_path}", self.log_text)
        else:
            self.utils.log_message("No trained model available to save.", self.log_text)



    def browse_data_file(self):
        # Open a file dialog and update the data file entry with the selected file path
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)  # Clear any existing text
            self.data_file_entry.insert(0, file_path)  # Insert the selected file path
            self.utils.log_message(f"Selected data file: {file_path}", self.log_text)


    def validate_inputs(self):
        data_file_path = self.data_file_entry.get()
        scaler_type = self.scaler_type_var.get()
        model_type = self.model_type_var.get()
        epochs_str = self.epochs_entry.get() if model_type == "neural_network" else "1"
        window_size_str = self.window_size_entry.get()

        # Various checks for empty inputs
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
        if model_type == "neural_network" and (not window_size_str.isdigit() or int(window_size_str) <= 0):
            self.error_label.config(text="Window size should be a positive integer.", fg="red")
            return False

        # Check if window size is too large for the dataset
        if model_type == "neural_network":
            try:
                data_length = pd.read_csv(data_file_path).shape[0]
                if int(window_size_str) >= data_length:
                    self.error_label.config(text="Window size too large for the dataset.", fg="red")
                    return False
            except Exception as e:
                self.utils.log_message(f"Error checking data length: {str(e)}", self.log_text)
                return False

        self.error_label.config(text="")
        return True


    def preprocess_data(self, file_path, scaler_type, model_type, window_size=5):
        # Load the data
        data = pd.read_csv(file_path)

        # Handle empty strings in 'date' column
        data['date'] = data['date'].replace('', np.nan)

        # Convert 'date' to datetime and handle incorrect formats
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

        # Handle rows where 'date' couldn't be converted (now NaT)
        data = data.dropna(subset=['date'])

        # Extract features from 'date' column
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

        # Define your target variable and feature set
        if 'close' in data.columns:
            y = data['close']  # Use 'close' as your target column name if it exists
            X = data.drop(['close', 'date'], axis=1)  # Drop 'close' and 'date' columns
        elif '4. close' in data.columns:
            y = data['4. close']  # Use '4. close' as your target column name if 'close' does not exist
            X = data.drop(['4. close', 'date'], axis=1)  # Drop '4. close' and 'date' columns
        else:
            raise ValueError("Neither 'close' nor '4. close' column found in the dataset")


        # Fill NaN values
        for col in X.columns:
            if X[col].isna().any():
                X[col].fillna(X[col].mean(), inplace=True)

        # Create the scaler object
        scaler = self.get_scaler(scaler_type)
        # Scale the features
        X_scaled = scaler.fit_transform(X)
        # Store the used scaler object
        self.trained_scaler = scaler

        # Split the data into training and testing sets
        if model_type == "neural_network":
            # Check if dataset is large enough for the given window size
            if len(X_scaled) > window_size:
                X, y = self.create_windowed_data(X_scaled, y, window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                self.utils.log_message("Window size is too large for the dataset.", self.log_text)
                return None, None, None, None  # Or handle this case as you see fit
        else:
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
            self.trained_model = model
            self.trained_scaler = scaler

            self.utils.log_message("Model training completed.", self.log_text)

        except Exception as e:
            self.utils.log_message(f"Error in model training: {str(e)}", self.log_text)
            print(f"Debug: Error in model training - {str(e)}")

    def compare_models(self, X_test, y_test):
        try:
            if len(self.trained_models) > 1:
                for model in self.trained_models:
                    evaluation_results = self.evaluate_model(model, X_test, y_test)
                    print(evaluation_results)
                # Further logic...
            else:
                self.utils.log_message("Not enough models to compare. Train more models.", self.log_text)
        except Exception as e:
            print(f"Error during model comparison: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test, model_type):
        """
        Evaluate the model with more metrics and provide visualizations.

        Parameters:
        - model: The trained model.
        - X_test: Test features.
        - y_test: True values for test features.
        - model_type: Type of the model ('regression' or 'classification').
        """

        predictions = model.predict(X_test)

        # For Regression
        if model_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            self.plot_regression_results(y_test, predictions)  # Visualization for regression
            return {"Mean Squared Error": mse, "R^2 Score": r2}
        
        # For Classification
        elif model_type == 'classification':
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            self.plot_confusion_matrix(y_test, predictions)  # Visualization for classification
            return {"Accuracy": accuracy, "Classification Report": report}
        else:
            self.utils.log_message("Invalid model type provided.", self.log_text)
            return {}

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

    def browse_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)  # Clear any existing entry
            self.data_file_entry.insert(0, file_path)  # Insert the selected file path into the entry field
            self.utils.log_message(f"Selected data file: {file_path}", self.log_text)

    def plot_regression_results(self, y_true, y_pred):
        """Plotting function for regression results."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Regression Results: True vs Predicted')
        plt.show()

    def create_windowed_data(self, X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)

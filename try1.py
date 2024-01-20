#model_evaluation_tab.py

#part 1

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model as load_keras_model
import torch  # assuming PyTorch is also used
import logging
import traceback
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import json

class ModelEvaluationTab(tk.Frame):
    def __init__(self, parent, is_debug_mode):
        super().__init__(parent)
        self.is_debug_mode = is_debug_mode
        self.setup_ui()
        self.configure_logging()
        self.configure_styles()
        #add a style configuration for the DebugMode.TButton
        style = ttk.Style()
        style.configure('DebugMode.TButton', background='green', foreground='white')  # Configure your desired style

    def configure_styles(self):
        style = ttk.Style()
        style.configure('DebugMode.TButton', background='green', foreground='white')

    def setup_ui(self):
        # Model Selection
        ttk.Label(self, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.model_paths_frame = ttk.Frame(self)
        self.model_paths_frame.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        self.model_path_entries = []
        self.add_model_path_entry()  # Add the first entry widget

        self.add_model_button = ttk.Button(self, text="Add Another Model", command=self.add_model_path_entry)
        self.add_model_button.grid(row=0, column=2, padx=10, pady=5, sticky='ew')


        # Data Selection
        ttk.Label(self, text="Select Data:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.data_path_entry = ttk.Entry(self)
        self.data_path_entry.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
        self.data_browse_button = ttk.Button(self, text="Browse", command=self.browse_data)
        self.data_browse_button.grid(row=1, column=2, padx=10, pady=5, sticky='ew')

        # Results Display
        self.results_text = tk.Text(self, height=15)
        self.results_text.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')
        self.rowconfigure(3, weight=1)
        self.columnconfigure(1, weight=1)

        # Log Text Widget for Debugging
        self.log_text = scrolledtext.ScrolledText(self, height=10)
        self.log_text.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Debug Mode Toggle Button
        self.debug_mode_button = ttk.Button(self, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Compare Models Button
        self.compare_models_button = ttk.Button(self, text="Compare Models", command=self.compare_models)
        self.compare_models_button.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Evaluate Model Button
        self.evaluate_model_button = ttk.Button(self, text="Evaluate Model", command=self.initiate_model_evaluation)
        self.evaluate_model_button.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Data Summary Text Widget
        self.data_summary_text = tk.Text(self, height=5)
        self.data_summary_text.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        debug_status = "ON" if self.is_debug_mode else "OFF"
        self.log_message(f"Debug mode turned {debug_status}")
        self.debug_mode_button.config(style='DebugMode.TButton' if self.is_debug_mode else 'TButton', text="Debug Mode: ON" if self.is_debug_mode else "Debug Mode: OFF")


    def log_message(self, message):
        if self.is_debug_mode:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.yview(tk.END)

#part 2
    def browse_model(self, entry):
        self.log_message("Browsing for model file")
        file_path = filedialog.askopenfilename(
            filetypes=[("Scikit-learn Model", "*.joblib"),
                    ("Keras Model", "*.h5"),
                    ("PyTorch Model", "*.pth"),
                    ("All Files", "*.*")]
        )
        if file_path and self.validate_model_file(file_path):
            entry.delete(0, tk.END)
            entry.insert(0, file_path)
            self.log_message(f"Model file selected: {file_path}")
        else:
            messagebox.showerror("Invalid File", "The selected file is not a valid model.")

    def browse_data(self):
        self.log_message("Browsing for data file")
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            self.data_path_entry.delete(0, tk.END)
            self.data_path_entry.insert(0, file_path)
            self.log_message(f"Data file selected: {file_path}")

    def load_model(self, filename):
        self.log_message(f"Attempting to load model from {filename}")
        try:
            if filename.endswith('.joblib'):
                # Loading joblib model (typically scikit-learn models)
                try:
                    model = joblib.load(filename)
                    self.log_message(f"Joblib model loaded: {filename}")
                    return model
                except Exception as e:
                    self.log_error(f"Error loading joblib model: {e}")
                    return None
            elif filename.endswith('.h5'):
                return load_keras_model(filename)
            elif filename.endswith('.pth'):
                # Assuming PyTorch model loading mechanism
                model = torch.load(filename)
                model.eval()  # Set the model to evaluation mode
                return model
            else:
                self.log_message("Unsupported file format for model.")
                return None
        except Exception as e:
            self.log_error(e)
            return None

    def log_error(self, e):
        error_message = str(e)
        traceback_message = traceback.format_exc()
        self.log_message(f"Error: {error_message}\nTraceback: {traceback_message}")

#part 3
    def display_all_results(self, model_results):
        for model_path, results in model_results.items():
            display_text = f"Results for {model_path}:\n"
            display_text += "\n".join([f"{key}: {value}" for key, value in results.items()])
            display_text += "\n\n"
            self.results_text.insert(tk.END, display_text)

    def evaluate_model(self):
        model_results = {}
        for entry in self.model_path_entries:
            model_path = entry.get()
            if not model_path:
                continue

            data_path = self.data_path_entry.get()
            if not data_path:
                messagebox.showerror("Error", "Please select both a model and a dataset.")
                continue

            try:
                model = self.load_model(model_path)
                if model is None:
                    continue  # Skip if the model couldn't be loaded

                data = pd.read_csv(data_path)
                X, y = data.iloc[:, :-1], data.iloc[:, -1]
                X = np.array(X, dtype=np.float32)  # Convert X to float32


                # Depending on the model type, you may need to preprocess the data differently
                # e.g., scaling for certain models, encoding categorical variables, etc.

                # Handle prediction based on model type
            try:
                predictions = model.predict(X)
            except Exception as e:
                self.log_error(f"Error during model prediction: {e}")
                return

                model_type = 'classification' if self.is_classification(model) else 'regression'

                evaluation_results = self.evaluate_and_display_model(model, X, y, model_type)
                model_results[model_path] = evaluation_results

            except Exception as e:
                self.log_error(e)

        # Process and display all collected evaluation results
        self.display_all_results(model_results)

# Here you would add logic to handle different model types and data formats.
# Implement parallel processing if needed.


        try:
            self.log_message("Loading model for evaluation...")
            model = self.load_model(model_path)
            self.log_message(f"Model loaded from {model_path}")

            self.log_message("Loading data for evaluation...")
            data = pd.read_csv(data_path)
            self.log_message(f"Data loaded from {data_path}")

            X, y = data.iloc[:, :-1], data.iloc[:, -1]
            X = np.array(X, dtype=np.float32)  # Convert X to float32

            predictions = model.predict(X)
            model_type = 'classification' if self.is_classification(model) else 'regression'

            evaluation_results = self.evaluate_and_display_model(model, X, y, model_type)
            self.log_model_metrics(y, predictions)  # Log model metrics after evaluation
            self.log_message(f"Evaluation completed. Model type: {model_type}")

            # Collect all evaluation_results in a dictionary or list
            # After the loop, process and display these results


        except Exception as e:
            self.log_error(e)

    def evaluate_and_display_model(self, model, X_test, y_test, model_type):
        self.log_message(f"Evaluating model of type: {model_type}")
        predictions = model.predict(X_test)

        if model_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            self.log_message(f"Regression Model Evaluated: MSE={mse}, R2={r2}")
            return {"Mean Squared Error": mse, "R^2 Score": r2}

        elif model_type == 'classification':
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            report = classification_report(y_test, predictions)
            self.log_message(f"Classification Model Evaluated: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}")
            return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Report": report}

    def display_results(self, results, model_type):
        self.log_message(f"Displaying results for {model_type} model")
        display_text = "\n".join([f"{key}: {value}" for key, value in results.items()])
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, display_text)

#part 4

    def is_classification(self, model):
        self.log_message("Determining if the model is for classification")
        try:
            if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
                return True
            if isinstance(model, RandomForestClassifier) or (isinstance(model, Sequential) and hasattr(model, 'predict_classes')):
                return True
        except Exception as e:
            self.log_message(f"Exception in is_classification: {e}")
        return messagebox.askyesno("Model Type", "Is this a classification model?")

    def log_model_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.log_message(f"Model Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO if self.is_debug_mode else logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
    def add_model_path_entry(self):
        model_frame = ttk.Frame(self.model_paths_frame)
        model_frame.pack(side='top', fill='x', expand=True, padx=5, pady=2)
        MAX_MODEL_COUNT = 3  # Set this to your desired limit

        if len(self.model_path_entries) < MAX_MODEL_COUNT:
            entry = ttk.Entry(model_frame)
            entry.pack(side='left', fill='x', expand=True, padx=5, pady=2)
            self.model_path_entries.append(entry)
        else:
            messagebox.showinfo("Limit Reached", "Maximum number of models reached.")


        browse_button = ttk.Button(model_frame, text="Browse", command=lambda: self.browse_model(entry))
        browse_button.pack(side='right', padx=5, pady=2)


    def display_data_summary(self, data_path):
        data = pd.read_csv(data_path)
        summary = data.describe().to_string()
        preview = data.head().to_string()
        self.data_summary_text.delete('1.0', tk.END)
        self.data_summary_text.insert(tk.END, "Summary:\n" + summary + "\n\nPreview:\n" + preview)


    def plot_data(self, data, plot_type):
        try:
            if plot_type == 'histogram':
                data.hist(figsize=(10, 8))
            elif plot_type == 'boxplot':
                data.plot(kind='box', figsize=(10, 8), subplots=True, layout=(2,2), sharex=False, sharey=False)
            # Add more plot types as needed

            plt.show()  # Display the plot
        except Exception as e:
            self.log_error(e)

    def calculate_custom_metrics(self, model, X, y):
        try:
            if isinstance(model, RandomForestRegressor):
                # Example: Calculating Feature Importances for Random Forest
                importances = model.feature_importances_
                return {"Feature Importances": importances}
            elif isinstance(model, LinearRegression):
                # Example: Coefficients for Linear Regression
                coefficients = model.coef_
                return {"Coefficients": coefficients}
            # Add more model types and their specific metrics here
            else:
                self.log_message("No custom metrics available for this model type.")
                return {}
        except Exception as e:
            self.log_error(e)
            return {}


    def visualize_feature_importance(self, model, feature_names):
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Plot the feature importances
                plt.figure()
                plt.title("Feature Importances")
                plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
                plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
                plt.xlim([-1, X.shape[1]])
                plt.show()
            else:
                self.log_message("This model does not have feature importances.")
        except Exception as e:
            self.log_error(e)

    def validate_model_file(self, file_path):
        # Check if the file is a scaler file
        if "scaler" in file_path:
            self.log_message("File identified as a scaler, not a model: " + file_path)
            return False

        # Validate based on extension
        valid_extensions = ['.joblib', '.h5', '.pth']
        if any(file_path.endswith(ext) for ext in valid_extensions):
            return True
        else:
            self.log_message("Invalid model file extension for file: " + file_path)
            return False

    def calculate_custom_metrics(self, model, X, y):
        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
            feature_names = X.columns
            return {"Feature Importances": dict(zip(feature_names, importances))}
        # Add more conditions for other model types
        else:
            return {"Info": "No custom metrics available for this model type."}

    def compare_models(self):
        # Check if there are multiple models to compare
        if len(self.model_path_entries) < 2:
            self.log_message("Not enough models for comparison. Please evaluate at least two models.")
            return

        self.log_message("Comparing Models...")

        # Check if the JSON file with model evaluation results exists
        json_file_path = 'model_evaluation_results.json'
        if not os.path.exists(json_file_path):
            self.log_message(f"Model evaluation results file not found: {json_file_path}")
            return  # Exit the function if file does not exist

        # Load evaluation results from JSON file
        try:
            with open(json_file_path, 'r') as file:
                model_evaluation_results = json.load(file)
        except Exception as e:
            self.log_message(f"Error reading model evaluation results: {e}")
            return

        # Retrieve and compare the performance metrics for each model
        model_performance = {}
        for entry in self.model_path_entries:
            model_path = entry.get()
            if not model_path or model_path not in model_evaluation_results:
                continue

            model_performance[model_path] = model_evaluation_results[model_path]

        # Comparing models based on a specific metric, e.g., accuracy
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get('Accuracy', 0), reverse=True)
        comparison_results = "\n".join([f"{model}: Accuracy = {metrics.get('Accuracy', 0)}" for model, metrics in sorted_models])

        self.log_message("Model Comparison Results (Based on Accuracy):")
        self.log_message(comparison_results)
        # You can extend this logic to compare based on other metrics as well


    def get_model_performance(self, model_path):
        # Retrieve the model's performance metrics from a stored file or dictionary
        # This could be a JSON file, a database, or a Python dictionary
        try:
            # Example: Loading from a JSON file where each model's path is a key
            with open('model_evaluation_results.json', 'r') as file:
                model_evaluation_results = json.load(file)

            if model_path in model_evaluation_results:
                return model_evaluation_results[model_path]
            else:
                self.log_message(f"No evaluation data found for model: {model_path}")
                return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}  # Return default values if no data found
        except Exception as e:
            self.log_message(f"Error accessing evaluation data: {str(e)}")
            return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}  # Return default values in case of error


    def initiate_model_evaluation(self):
        # This method simply calls the existing evaluate_model method
        self.evaluate_model()

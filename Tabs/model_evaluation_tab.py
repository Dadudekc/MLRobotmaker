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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler
from tensorflow.keras.models import Sequential


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
        self.results_text = tk.Text(self)
        self.results_text_scrollbar = tk.Scrollbar(self, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=self.results_text_scrollbar.set)
        self.results_text_scrollbar.grid(row=3, column=4, sticky='ns')
        self.results_text.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        self.rowconfigure(3, weight=1)
        self.columnconfigure(1, weight=1)

        # Log Text Widget for Debugging
        self.log_text = scrolledtext.ScrolledText(self, height=10)
        self.log_text.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Debug Mode Toggle Button
        self.debug_mode_button = ttk.Button(self, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Create a new frame for model configuration
        self.config_frame = ttk.Frame(self)
        self.config_frame.grid(row=9, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Add input fields for model parameters (e.g., sliders, dropdowns, text entries)
        # Example:
        self.param1_entry = ttk.Entry(self.config_frame)
        self.param1_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        # Add a button to apply configurations
        self.apply_config_button = ttk.Button(self.config_frame, text="Apply Configurations", command=self.apply_configurations)
        self.apply_config_button.grid(row=0, column=2, padx=10, pady=5, sticky='ew')

        # Compare Models Button
        self.compare_models_button = ttk.Button(self, text="Compare Models", command=self.compare_models)
        self.compare_models_button.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Evaluate Model Button
        self.evaluate_model_button = ttk.Button(self, text="Evaluate Model", command=self.initiate_model_evaluation)
        self.evaluate_model_button.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Data Summary Text Widget
        # Create a new frame for visualizations
        self.visualization_frame = ttk.Frame(self)
        self.visualization_frame.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Add buttons for different types of plots
        self.plot_histogram_button = ttk.Button(self.visualization_frame, text="Plot Histogram", command=lambda: self.plot_data('histogram'))
        self.plot_histogram_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        self.plot_boxplot_button = ttk.Button(self.visualization_frame, text="Plot Boxplot", command=lambda: self.plot_data('boxplot'))
        self.plot_boxplot_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        # Initialize and place the data summary text widget
        self.data_summary_text = tk.Text(self, height=5)
        self.data_summary_text.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Add a button to visualize feature importance
        self.visualize_feature_importance_button = ttk.Button(self, text="Visualize Feature Importance", command=self.visualize_feature_importance)
        self.visualize_feature_importance_button.grid(row=11, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Add a button to display data summary
        self.display_data_summary_button = ttk.Button(self, text="Display Data Summary", command=self.display_data_summary)
        self.display_data_summary_button.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        
        # Add a button to calculate custom metrics
        self.calculate_custom_metrics_button = ttk.Button(self, text="Calculate Custom Metrics", command=self.calculate_custom_metrics)
        self.calculate_custom_metrics_button.grid(row=12, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        
        # Add buttons to plot different types of data
        self.plot_histogram_button = ttk.Button(self, text="Plot Histogram", command=lambda: self.plot_data('histogram'))
        self.plot_histogram_button.grid(row=9, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        self.plot_boxplot_button = ttk.Button(self, text="Plot Boxplot", command=lambda: self.plot_data('boxplot'))
        self.plot_boxplot_button.grid(row=10, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Add a button to retrieve model performance
        self.get_model_performance_button = ttk.Button(self, text="Get Model Performance", command=self.get_model_performance)
        self.get_model_performance_button.grid(row=13, column=0, columnspan=3, padx=10, pady=5, sticky='ew')


    def apply_configurations(self):
        # Retrieve configuration parameters and their expected types
        config_params = self.get_config_params()  

        model_configurations = {}

        for param, expected_type in config_params.items():
            # Assume each parameter has a corresponding entry widget named as '{param}_entry'
            entry_widget = getattr(self, f"{param}_entry")
            entry_value = entry_widget.get()

            try:
                # Convert the parameter value to its expected type
                converted_value = expected_type(entry_value)
                model_configurations[param] = converted_value

                # Optionally, update the UI to reflect successful configuration
                entry_widget.config(background="white")
            except ValueError:
                # Indicate error in UI and log message
                entry_widget.config(background="red")
                messagebox.showerror("Error", f"Invalid value for {param}. Expected type: {expected_type.__name__}.")
                return

        # Apply the configurations to the model (example for a scikit-learn model)
        # self.my_model.set_params(**model_configurations)

        # Log the applied configurations
        config_str = ", ".join(f"{key} = {value}" for key, value in model_configurations.items())
        self.log_message(f"Configurations applied: {config_str}")

    def get_config_params(self):
        # Define the expected configuration parameters and their data types
        return {
            'learning_rate': float,  # Example for a float parameter
            'max_depth': int,        # Example for an integer parameter
            'activation_function': str  # Example for a string parameter
            # Add more parameters as needed
        }


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
            # Create a Treeview widget for displaying results in a table format
            self.results_table = ttk.Treeview(self)
            self.results_table["columns"] = ("accuracy", "precision", "recall", "f1_score")
            for col in self.results_table["columns"]:
                self.results_table.heading(col, text=col, command=lambda _col=col: self.treeview_sort_column(_col, False))

            # Define a method to handle sorting (you'll need to define this method)
            def treeview_sort_column(self, col, reverse):
                data_list = [(self.results_table.set(k, col), k) for k in self.results_table.get_children('')]
                data_list.sort(reverse=reverse)

                # Rearrange items in sorted positions
                for index, (val, k) in enumerate(data_list):
                    self.results_table.move(k, '', index)

                # Reverse sort next time
                self.results_table.heading(col, command=lambda _col=col: self.treeview_sort_column(_col, not reverse))


    def evaluate_model(self, model, X_test, y_test, model_type):
        """
        Evaluate the model with more metrics and provide visualizations.
        Parameters:
        - model: The trained model or the path to the model.
        - X_test: Test features.
        - y_test: True values for test features.
        - model_type: Type of the model ('regression' or 'classification').
        """
        try:
            # Load the model if a path is provided
            if isinstance(model, str):
                model = self.load_model(model)

            # Preprocess data
            X_preprocessed = self.preprocess_data(X_test, model)

            # Predictions
            predictions = model.predict(X_preprocessed)

            # Evaluate Metrics
            if model_type == 'regression':
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                # Visualization for regression
                self.plot_regression_results(y_test, predictions)
                return {"Mean Squared Error": mse, "R^2 Score": r2}
            elif model_type == 'classification':
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions)
                # Visualization for classification
                self.plot_confusion_matrix(y_test, predictions)
                return {"Accuracy": accuracy, "Classification Report": report}
            else:
                self.log_error("Invalid model type provided.")
                return {}
        except Exception as e:
            self.log_error(f"Error during model evaluation: {e}")
            return {}


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

    def plot_data(self, plot_type):
        data_path = self.data_path_entry.get()
        if not data_path:
            messagebox.showerror("Error", "No data file selected.")
            return

        data = pd.read_csv(data_path)
        if plot_type == 'histogram':
            fig, ax = plt.subplots()
            data.hist(ax=ax)
            plt.title("Histogram")
        elif plot_type == 'boxplot':
            fig, ax = plt.subplots()
            data.plot(kind='box', ax=ax)
            plt.title("Boxplot")

        # Create a new Tkinter Toplevel window for the plot
        plot_window = tk.Toplevel(self)
        plot_window.title(f"{plot_type.title()} Plot")

        # Embed the plot in the Tkinter window using FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


    def calculate_custom_metrics(self, model, X, y):
        try:

            if isinstance(model, RandomForestRegressor):
                importances = model.feature_importances_
                feature_names = X.columns
                return {"Feature Importances": dict(zip(feature_names, importances))}
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
                # Create a new tkinter Toplevel window
                visualization_window = tk.Toplevel(self)
                visualization_window.title("Data Visualization")

                # Create a figure for Matplotlib
                fig, ax = plt.subplots()

                # Add your plotting logic here
                # For example: data.hist(ax=ax) or data.plot(ax=ax)

                # Create a canvas and add the figure to it
                canvas = FigureCanvasTkAgg(fig, master=visualization_window)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)

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
        """
        Compare multiple trained models.
        """
        try:
            if len(self.model_path_entries) < 2:
                self.log_message("Not enough models for comparison. Please evaluate at least two models.")
                return

            # Load data for comparison
            data_path = self.data_path_entry.get()
            if not data_path:
                messagebox.showerror("Error", "Please select a dataset.")
                return
            data = pd.read_csv(data_path)
            X, y = data.iloc[:, :-1], data.iloc[:, -1]

            # Initialize comparison results
            comparison_results = {}

            for entry in self.model_path_entries:
                model_path = entry.get()
                if model_path:
                    model_type = 'classification' if self.is_classification(model_path) else 'regression'
                    evaluation_results = self.evaluate_model(model_path, X, y, model_type)
                    comparison_results[model_path] = evaluation_results

            # Display comparison results
            for model_path, results in comparison_results.items():
                self.log_message(f"Results for {model_path}: {results}")
            # Extend this to display in the UI as needed

        except Exception as e:
            self.log_error(f"Error during model comparison: {e}")


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

    def preprocess_data(self, X, model):
        try:
            # Example: Identify categorical and numeric columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Define transformers for different column types
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
                ('scaler', StandardScaler())])  # Scale numeric values

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One-hot encode categorical values

            # Combine transformers into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)])

            # Apply the preprocessing to the dataset
            X_preprocessed = preprocessor.fit_transform(X)

            return pd.DataFrame(X_preprocessed)
        except Exception as e:
            self.log_error(f"Error during data preprocessing: {e}")
            return None

    def custom_prediction_method(self, model, X, batch_size=None):
        """
        Extended method for custom predictions, handling different types of models.
        Supports batch processing for large datasets.
        """
        try:
            model_type = str(type(model)).lower()

            # TensorFlow/Keras Models
            if "tensorflow" in model_type or "keras" in model_type:
                if batch_size:
                    # Process in batches for large datasets
                    predictions = np.vstack([model.predict_on_batch(X[i:i+batch_size]) for i in range(0, len(X), batch_size)])
                else:
                    predictions = model.predict(X)

            # PyTorch Models
            elif "torch" in model_type:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float()
                    if next(model.parameters()).is_cuda:
                        X_tensor = X_tensor.to('cuda')
                    # Batch processing for PyTorch
                    if batch_size:
                        predictions = np.vstack([model(X_tensor[i:i+batch_size]).cpu().numpy() for i in range(0, len(X_tensor), batch_size)])
                    else:
                        predictions = model(X_tensor).cpu().numpy()

            # Scikit-learn Models
            elif "sklearn" in model_type:
                predictions = model.predict(X)

            # XGBoost Models
            elif "xgboost" in model_type:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                predictions = model.predict(dmatrix)

            # Add more elif blocks for other types of models, such as LightGBM, CatBoost, etc.

            else:
                self.log_error(f"Unsupported model type: {model_type}")
                return None

            return predictions

        except Exception as e:
            self.log_error(f"Error during custom prediction: {e}")
            return None

def is_classification(model):
    self.log_message("Determining if the model is for classification")
    try:
        if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
            return True
        if isinstance(model, RandomForestClassifier) or (isinstance(model, Sequential) and hasattr(model, 'predict_classes')):
            return True
    except Exception as e:
        self.log_message(f"Exception in is_classification: {e}")
    return messagebox.askyesno("Model Type", "Is this a classification model?")
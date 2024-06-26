#trade_analyzer_tab.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib  # For sklearn models
import torch  # For PyTorch models
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import json

class TradeAnalyzerTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.models = {'rf': None, 'nn': None, 'pytorch': None}
        self.scaler = None
        self.metadata = None
        self.data = None
        self.apply_theme()
        self.create_widgets()
        logging.basicConfig(level=logging.INFO)

        self.metadata_button = ttk.Button(self, text="Load Metadata", command=self.load_metadata)
        self.metadata_button.grid(row=5, column=0, columnspan=2, pady=5)


    def apply_theme(self):
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # Example theme

    def create_widgets(self):
        self.create_model_load_buttons()
        self.create_data_load_button()
        self.create_predict_button()
        self.create_results_display()
        self.create_tabbed_results()

    def create_model_load_buttons(self):
        model_types = {'Random Forest': 'rf', 'Neural Network': 'nn', 'PyTorch': 'pytorch'}
        for idx, (name, type) in enumerate(model_types.items()):
            ttk.Label(self, text=f"Select {name} Model:").grid(row=idx, column=0, sticky='w')
            button = ttk.Button(self, text="Browse", command=lambda t=type: self.load_model(t), style='Custom.TButton')
            button.grid(row=idx, column=1)

    def create_data_load_button(self):
        ttk.Label(self, text="Select Trading Data:").grid(row=3, column=0, sticky='w')
        self.data_button = ttk.Button(self, text="Browse", command=self.load_data)
        self.data_button.grid(row=3, column=1)

    def create_predict_button(self):
        self.predict_button = ttk.Button(self, text="Make Predictions", command=self.async_make_predictions)
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=5)

    def create_results_display(self):
        self.results_text = tk.Text(self, height=10, width=50)
        self.results_text.grid(row=5, column=0, columnspan=2, pady=5)

    def create_tabbed_results(self):
        self.notebook = ttk.Notebook(self)
        self.results_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.logs_tab, text="Logs")
        self.notebook.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")

        self.logs_text = tk.Text(self.logs_tab, height=10, width=50)
        self.logs_text.pack(expand=True, fill='both')

    def async_make_predictions(self):
        self.predict_button['state'] = 'disabled'
        threading.Thread(target=self.make_predictions, daemon=True).start()

    def load_model(self, model_type):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                if model_type == 'rf':
                    self.models['rf'] = joblib.load(file_path)
                elif model_type == 'nn':
                    self.models['nn'] = tf.keras.models.load_model(file_path)
                elif model_type == 'pytorch':
                    # Assuming PyTorch model is saved with torch.save(model.state_dict())
                    self.models['pytorch'] = torch.load(file_path)
                    self.models['pytorch'].eval()  # Set to evaluation mode
                self.update_ui('log', f"{model_type.upper()} model loaded successfully.")
            except Exception as e:
                self.update_ui('log', f"Failed to load {model_type.upper()} model: {e}")

    def load_metadata(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.metadata = json.load(f)
                self.scaler = joblib.load(self.metadata['scaler_path'])
                self.update_ui('log', "Metadata loaded successfully.")
            except Exception as e:
                self.update_ui('log', f"Failed to load metadata: {e}")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.update_ui('log', "Data loaded successfully.")
            except Exception as e:
                self.update_ui('log', f"Failed to load data: {e}")

    def preprocess_data_for_model(self, model_type, data):
        if self.scaler:
            data = self.scaler.transform(data)
        # Additional preprocessing steps based on metadata
        # Implement specific preprocessing steps for each model if needed
        return data

    def make_predictions(self):
        if self.data is not None and any(self.models.values()):
            try:
                preprocessed_data = self.preprocess_data_for_model('model_type', self.data)  # Replace 'model_type' as needed
                # Implement the prediction logic for each model
                # Update the UI with results
            except Exception as e:
                self.update_ui('log', f"Error in making predictions: {e}")
        else:
            self.update_ui('log', "Please load both model and data.")
        self.update_ui('enable_button', "")

    def update_ui(self, action, message):
        if action == 'log':
            self.logs_text.insert(tk.END, message + "\n")
        elif action == 'result':
            self.results_text.insert(tk.END, message + "\n")
        elif action == 'enable_button':
            self.predict_button['state'] = 'normal'
        # Add more actions as needed

    def display_results_graph(self, actual_data, predicted_data):
        fig, ax = plt.subplots()
        ax.plot(actual_data, label='Actual Prices')
        ax.plot(predicted_data, label='Predicted Prices')
        ax.set_title('Stock Prices Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        # Displaying the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.results_tab)
        canvas.draw()
        canvas.get_tk_widget().pack()


import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import keras.models

class HistoricalDataManager:
    def __init__(self):
        self.historical_data = {}

    def import_data(self, selected_asset, update_ui_callback):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.historical_data[selected_asset] = data
                update_ui_callback(f"Data imported from {file_path}.")
            except Exception as e:
                update_ui_callback(f"Failed to import data: {e}.")

    def calculate_performance_metrics(self, selected_asset, update_ui_callback):
        asset_data = self.historical_data.get(selected_asset)
        if asset_data is not None:
            total_return = (asset_data['Close'].iloc[-1] / asset_data['Close'].iloc[0] - 1) * 100
            update_ui_callback(f"Total Return for {selected_asset}: {total_return:.2f}%")
        else:
            update_ui_callback(f"No historical data available for {selected_asset}.")

class ModelManager:
    def __init__(self):
        self.model = None

    def select_model(self, update_ui_callback):
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5"), ("Model Files", "*.model")])
        if model_path:
            try:
                self.model = keras.models.load_model(model_path)
                update_ui_callback(f"Model loaded successfully from {model_path}.")
            except Exception as e:
                update_ui_callback(f"Failed to load model: {e}.")

    def predict(self, data):
        if self.model:
            return self.model.predict(data)
        return None

class RiskManagementTab(ttk.Frame):
    def __init__(self, parent, trained_model=None):  # Adding trained_model parameter with default value of None
        super().__init__(parent)
        self.data_manager = HistoricalDataManager()
        self.model_manager = ModelManager()
        self.selected_asset = tk.StringVar(value="None")
        self.trained_model = trained_model  # Use this attribute as needed
        self.create_widgets()


    def create_widgets(self):
        self.setup_data_import_button()
        self.setup_metrics_button()
        self.setup_model_selection_button()

    def update_ui(self, message):
        print(message)  # Placeholder for actual UI update

    def setup_data_import_button(self):
        ttk.Button(self, text="Import Historical Data",
                   command=lambda: self.data_manager.import_data(
                       self.selected_asset.get(), self.update_ui)).pack()

    def setup_metrics_button(self):
        ttk.Button(self, text="Calculate Metrics",
                   command=lambda: self.data_manager.calculate_performance_metrics(
                       self.selected_asset.get(), self.update_ui)).pack()

    def setup_model_selection_button(self):
        ttk.Button(self, text="Select Trained Model",
                   command=lambda: self.model_manager.select_model(self.update_ui)).pack()

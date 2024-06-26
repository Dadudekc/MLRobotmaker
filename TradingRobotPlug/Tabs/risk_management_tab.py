import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import keras.models
import joblib
import pickle
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from datetime import datetime

# Assume these are imported correctly
from risk_management_resources.HistoricalDataManager import HistoricalDataManager
from risk_management_resources.ModelManager import ModelManager

# GUI Components
class RiskManagementTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.data_manager = HistoricalDataManager()
        self.model_manager = ModelManager()
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

        ttk.Label(control_frame, text="Asset:").pack(side=tk.LEFT)
        self.asset_entry = ttk.Entry(control_frame)
        self.asset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(control_frame, text="Import Data", command=self.import_data).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Predict", command=self.predict).pack(side=tk.LEFT)

    def load_model(self):
        def async_load():
            self.model_manager.load_model(self.update_ui)
        threading.Thread(target=async_load).start()

    def import_data(self):
        def async_import():
            selected_asset = self.asset_entry.get()
            self.data_manager.import_data(selected_asset, self.update_ui)
        threading.Thread(target=async_import).start()

    def predict(self):
        try:
            input_data = float(self.asset_entry.get())  # Convert input to float
            self.model_manager.predict(input_data, self.update_ui)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a numeric value.")

    def validate_asset_input(self):
        if not self.asset_entry.get().strip():
            messagebox.showerror("Error", "Asset name cannot be empty.")
            return False
        return True

    def update_ui(self, message):
        messagebox.showinfo("Info", message)

class ChartTab(ttk.Frame):
    def __init__(self, parent, model_manager):
        super().__init__(parent)
        self.data_manager = HistoricalDataManager()
        self.model_manager = model_manager
        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_type_var = tk.StringVar(value='candle')
        plot_options = ttk.OptionMenu(self, self.plot_type_var, 'candle', 'candle', 'line', 'ohlc')
        plot_options.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.current_index = 0
        self.data = None

        # Control to enter/load data
        self.load_data_button = ttk.Button(self, text="Load Data", command=self.load_data)
        self.load_data_button.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Navigation buttons
        nav_frame = ttk.Frame(self)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(nav_frame, text="<<", command=self.prev_candle).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text=">>", command=self.next_candle).pack(side=tk.LEFT)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path, parse_dates=True, index_col=0)  # Assuming datetime index
            self.current_index = 0
            self.update_chart()

    def update_chart(self):
        if self.data is not None and not self.data.empty:
            display_data = self.data.iloc[:self.current_index+1]
            if not display_data.empty:
                self.ax.clear()
                mpf.plot(display_data, ax=self.ax, type=self.plot_type_var.get(), style='charles', volume=True)
                
                # Plotting model predictions if available
                if self.model_manager.model:
                    predictions = self.model_manager.model.predict(display_data[['Close']])
                    self.ax.plot(display_data.index, predictions, label='Predictions', color='red')
                    self.ax.legend()

                self.canvas.draw()

    def next_candle(self):
        if self.data is not None and self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.update_chart()

    def prev_candle(self):
        if self.data is not None and self.current_index > 0:
            self.current_index -= 1
            self.update_chart()

# Main application setup
if __name__ == "__main__":
    root = tk.Tk()
    tab_control = ttk.Notebook(root)
    risk_management_tab = RiskManagementTab(tab_control)
    tab_control.add(risk_management_tab, text="Risk Management")
    chart_tab = ChartTab(tab_control)  
    tab_control.pack(expand=1, fill="both")
    root.mainloop()

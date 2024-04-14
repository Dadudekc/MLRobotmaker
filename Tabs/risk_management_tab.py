import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import keras.models
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class HistoricalDataManager:
    def __init__(self):
        self.historical_data = {}

    def import_data(self, selected_asset, update_ui_callback, update_assets_callback=None):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.historical_data[selected_asset] = data
                if update_assets_callback:
                    update_assets_callback()  # Update the dropdown list if callback is provided
                update_ui_callback(f"Data imported for {selected_asset} from {file_path}.")
            except Exception as e:
                update_ui_callback(f"Failed to import data: {e}")

    def calculate_performance_metrics(self, selected_asset, update_ui_callback):
        asset_data = self.historical_data.get(selected_asset)
        if asset_data is not None:
            total_return = (asset_data['Close'].iloc[-1] / asset_data['Close'].iloc[0] - 1) * 100
            update_ui_callback(f"Total Return for {selected_asset}: {total_return:.2f}%")
        else:
            update_ui_callback("No historical data available for {selected_asset}.")

    def calculate_risk_metrics(self, selected_asset, update_ui_callback):
        asset_data = self.historical_data.get(selected_asset)
        if asset_data is not None:
            daily_returns = asset_data['Close'].pct_change()
            var_95 = np.percentile(daily_returns.dropna(), 5)
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            cumulative_returns = (1 + daily_returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns / peak) - 1
            max_drawdown = drawdown.min()
            update_ui_callback(f"VaR (95%): {var_95:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}, Maximum Drawdown: {max_drawdown:.2%}")
        else:
            update_ui_callback("No historical data available for {selected_asset}.")

    def calculate_position_size(self, selected_asset, account_equity, risk_per_trade, update_ui_callback):
        asset_data = self.historical_data.get(selected_asset)
        if asset_data is not None:
            daily_returns = asset_data['Close'].pct_change()
            var_95 = np.percentile(daily_returns.dropna(), 5)
            position_size = abs((account_equity * risk_per_trade) / var_95)
            update_ui_callback(f"Recommended position size for {selected_asset}: {position_size:.2f}")
        else:
            update_ui_callback("No historical data available for {selected_asset}.")

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
                update_ui_callback(f"Failed to load model: {e}")

class RiskManagementTab(ttk.Frame):
    def __init__(self, parent, trained_model=None):
        super().__init__(parent)
        self.data_manager = HistoricalDataManager()
        self.model_manager = ModelManager()
        self.selected_asset = tk.StringVar()
        self.account_equity = tk.DoubleVar(value=10000.0)
        self.risk_per_trade = tk.DoubleVar(value=0.01)
        self.trained_model = trained_model
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Select Asset:").pack()
        self.asset_dropdown = ttk.Combobox(self, textvariable=self.selected_asset)
        self.asset_dropdown.pack()
        self.update_asset_dropdown()  # Ensure the dropdown is updated on init

        self.setup_data_import_button()
        self.setup_metrics_button()
        self.setup_risk_metrics_button()
        self.setup_position_sizing_inputs()
        self.setup_model_selection_button()
        self.setup_plot_frame()

    def update_asset_dropdown(self):
        assets = list(self.data_manager.historical_data.keys())
        self.asset_dropdown['values'] = assets
        if assets:
            self.selected_asset.set(assets[0])

    def setup_data_import_button(self):
        ttk.Button(self, text="Import Historical Data", command=lambda: self.data_manager.import_data(self.asset_dropdown.get(), self.update_ui, self.update_asset_dropdown)).pack()

    def setup_metrics_button(self):
        ttk.Button(self, text="Calculate Performance Metrics", command=lambda: self.data_manager.calculate_performance_metrics(self.selected_asset.get(), self.update_ui)).pack()

    def setup_risk_metrics_button(self):
        ttk.Button(self, text="Calculate Risk Metrics", command=lambda: self.data_manager.calculate_risk_metrics(self.selected_asset.get(), self.update_ui)).pack()

    def setup_position_sizing_inputs(self):
        ttk.Label(self, text="Account Equity:").pack()
        ttk.Entry(self, textvariable=self.account_equity).pack()
        ttk.Label(self, text="Risk Per Trade (%):").pack()
        ttk.Entry(self, textvariable=self.risk_per_trade).pack()
        ttk.Button(self, text="Calculate Position Size", command=lambda: self.data_manager.calculate_position_size(self.selected_asset.get(), self.account_equity.get(), self.risk_per_trade.get() / 100, self.update_ui)).pack()

    def setup_model_selection_button(self):
        ttk.Button(self, text="Select Trained Model", command=lambda: self.model_manager.select_model(self.update_ui)).pack()

    def setup_plot_frame(self):
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Risk Management")
    RiskManagementTab(root).pack(fill="both", expand=True)
    root.mainloop()

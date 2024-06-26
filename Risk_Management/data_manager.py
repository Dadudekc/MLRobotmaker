# data_manager.py

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from tkinter import filedialog

class HistoricalDataManager:
    def __init__(self, alpha_vantage_api_key):
        self.historical_data = {}
        self.column_aliases = {
            'Close': ['close', 'closing_price', 'Close', 'end_of_day_price', 'CLOSE', '4.Close', '4. close']
        }
        self.ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')

    def import_data(self, selected_asset, update_ui_callback):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.transform_data(data)
                self.historical_data[selected_asset] = data
                update_ui_callback(f"Data imported for {selected_asset} from {file_path}.")
            except Exception as e:
                update_ui_callback(f"Failed to import data: {e}")

    def transform_data(self, data):
        for canonical, aliases in self.column_aliases.items():
            for alias in aliases:
                if alias in data.columns:
                    data.rename(columns={alias: canonical}, inplace=True)
                    break
        data['Close'] = data['Close'].fillna(method='ffill')
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Daily Return'] = data['Close'].pct_change()

    def get_data(self, symbol):
        if symbol in self.historical_data:
            return self.historical_data[symbol]

        try:
            yf_data = yf.download(symbol, period="1y")
            yf_data['Daily Return'] = yf_data['Close'].pct_change()
            self.historical_data[symbol] = yf_data
            return yf_data
        except Exception as e:
            print(f"Error fetching data from yFinance: {e}")
            return pd.DataFrame()

    def get_alpha_vantage_data(self, symbol):
        try:
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')
            data['Daily Return'] = data['4. close'].pct_change()
            self.historical_data[symbol] = data
            return data
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()

    def calculate_metrics(self, selected_asset, update_ui_callback):
        data = self.historical_data.get(selected_asset, pd.DataFrame())
        if not data.empty:
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            mean_return = data['Daily Return'].mean()
            volatility = data['Daily Return'].std()
            metrics_text = f"Metrics for {selected_asset}:\nTotal Return: {total_return:.2f}%\nMean Return: {mean_return:.4f}\nVolatility: {volatility:.4f}"
            update_ui_callback(metrics_text)
        else:
            update_ui_callback(f"No data available for {selected_asset}. Please import data first.")

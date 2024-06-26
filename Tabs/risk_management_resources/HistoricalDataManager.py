import pandas as pd
from tkinter import filedialog

class HistoricalDataManager:
    def __init__(self):
        self.historical_data = {}
        self.column_aliases = {'Close': ['close', 'closing_price', 'Close', 'end_of_day_price']}

    def import_data(self, selected_asset, callback):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                data = self._transform_columns(data)
                self.historical_data[selected_asset] = data
                callback(f"Data for {selected_asset} imported successfully from {file_path}.")
            except Exception as e:
                callback(f"Failed to import data: {e}")

    def _transform_columns(self, df):
        for canonical, aliases in self.column_aliases.items():
            for alias in aliases:
                if alias in df.columns:
                    df.rename(columns={alias: canonical}, inplace=True)
        return df

    def get_asset_data(self, asset):
        return self.historical_data.get(asset, pd.DataFrame())

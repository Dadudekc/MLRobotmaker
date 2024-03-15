#trade_analysis_tab.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TradingAnalysisTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.trade_data = pd.DataFrame()
        self.create_widgets()

    def create_widgets(self):
        # Load Data Button
        self.load_data_button = tk.Button(self, text="Load Trade Data", command=self.load_trade_data)
        self.load_data_button.pack(pady=5)

        # Frame for Matplotlib Chart
        self.chart_frame = tk.Frame(self)
        self.chart_frame.pack(fill='both', expand=True)
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

        # Trade Analysis Section
        self.analysis_text = tk.Text(self, height=10, width=50)
        self.analysis_text.pack()

        # Strategy Suggestions Section
        self.strategy_text = tk.Text(self, height=10, width=50)
        self.strategy_text.pack()

    def load_trade_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.trade_data = pd.read_csv(file_path)
            self.analyze_trades()
            self.plot_trade_data()

    def analyze_trades(self):
        if self.trade_data.empty:
            messagebox.showwarning("Warning", "No data loaded.")
            return

        # Example Analysis (replace with real analysis)
        self.analysis_text.insert(tk.END, "Performing analysis on trade data...\n")
        profit_loss = self.trade_data['Exit Price'] - self.trade_data['Entry Price']
        avg_profit_loss = profit_loss.mean()
        self.analysis_text.insert(tk.END, f"Average Profit/Loss per Trade: {avg_profit_loss:.2f}\n")

        # Simulated Strategy Suggestion
        self.strategy_text.insert(tk.END, "Strategy Suggestion based on analysis:\n")
        self.strategy_text.insert(tk.END, "Adjust entry strategy to target higher average profit.\n")

    def plot_trade_data(self):
        if self.trade_data.empty:
            return

        self.ax.clear()
        self.trade_data['Entry Price'].plot(ax=self.ax, label='Entry Price')
        self.trade_data['Exit Price'].plot(ax=self.ax, label='Exit Price')
        self.ax.legend()
        self.ax.set_title('Entry and Exit Prices Over Time')
        self.canvas.draw()
        
class RiskAssessment:
    def __init__(self, historical_data):
        self.historical_data = historical_data

    def calculate_volatility(self):
        return self.historical_data['Close'].pct_change().std()

    def value_at_risk(self, confidence_level=0.95):
        if not isinstance(self.historical_data, pd.DataFrame):
            raise ValueError("Historical data is not in a pandas DataFrame format.")
        
        pct_changes = self.historical_data['Close'].pct_change()
        return pct_changes.quantile(1 - confidence_level)

    def expected_shortfall(self, confidence_level=0.95):
        var = self.value_at_risk(confidence_level)
        return self.historical_data['Close'].pct_change()[self.historical_data['Close'].pct_change() <= var].mean()

    def assess_risk(self):
        volatility = self.calculate_volatility()
        var = self.value_at_risk()
        es = self.expected_shortfall()
        risk_metric = {
            "volatility": volatility,
            "value_at_risk": var,
            "expected_shortfall": es
        }
        return risk_metric

class RiskManagementJournal:
    def __init__(self):
        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)

    def display_journal(self):
        for entry in self.entries:
            print(entry)

    def export_to_csv(self, path):
        pd.DataFrame(self.entries).to_csv(path, index=False)

    def import_from_csv(self, path):
        self.entries = pd.read_csv(path).to_dict('records')

class CustomRiskStrategies:
    def __init__(self, strategy_name, parameters):
        self.strategy_name = strategy_name
        self.parameters = parameters

    def apply_strategy(self, historical_data):
        # Example strategy: Moving Average Crossover
        if self.strategy_name == "Moving Average Crossover":
            short_window = self.parameters.get('short_window', 40)
            long_window = self.parameters.get('long_window', 100)
            signals = pd.DataFrame(index=historical_data.index)
            signals['signal'] = 0.0

            signals['short_mavg'] = historical_data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
            signals['long_mavg'] = historical_data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
            signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
            signals['positions'] = signals['signal'].diff()
            return signals
        else:
            raise ValueError("Strategy not recognized")

def main():
    root = tk.Tk()
    root.title("Trading Analysis")
    TradingAnalysisTab(root).pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()

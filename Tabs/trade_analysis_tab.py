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

def main():
    root = tk.Tk()
    root.title("Trading Analysis")
    TradingAnalysisTab(root).pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()

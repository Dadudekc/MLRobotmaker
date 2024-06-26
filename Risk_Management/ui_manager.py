# ui_manager.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from config import ASSET_VALUES
from data_manager import HistoricalDataManager
from model_manager import ModelManager
from trade_simulator import TradeSimulator
import logging


class ModelEvaluationApp(tk.Frame):
    def __init__(self, parent, is_debug_mode):
        super().__init__(parent)
        self.is_debug_mode = is_debug_mode
        self.data_manager = HistoricalDataManager('YOUR_ALPHA_VANTAGE_API_KEY')
        self.model_manager = ModelManager()
        self.trade_simulator = TradeSimulator()
        self.setup_ui()
        self.configure_logging()

    def setup_ui(self):
        # Model Management
        model_frame = ttk.LabelFrame(self, text="Model Management")
        model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.model_path_entry = ttk.Entry(model_frame)
        self.model_path_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        self.load_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Data Management
        data_frame = ttk.LabelFrame(self, text="Data Management")
        data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(data_frame, text="Select Data:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.data_path_entry = ttk.Entry(data_frame)
        self.data_path_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        ttk.Button(data_frame, text="Browse", command=self.browse_data).grid(row=0, column=2, padx=10, pady=5, sticky='ew')

        ttk.Label(data_frame, text="Select Asset:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.asset_dropdown = ttk.Combobox(data_frame, values=list(ASSET_VALUES.keys()), state="readonly")
        self.asset_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
        ttk.Button(data_frame, text="Import Data", command=self.import_data).grid(row=1, column=2, padx=10, pady=5, sticky='ew')

        self.calculate_metrics_button = ttk.Button(data_frame, text="Calculate Metrics", command=self.calculate_metrics)
        self.calculate_metrics_button.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        self.display_asset_details_button = ttk.Button(data_frame, text="Display Asset Details", command=self.display_asset_details)
        self.display_asset_details_button.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Backtesting
        backtesting_frame = ttk.LabelFrame(self, text="Backtesting")
        backtesting_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Button(backtesting_frame, text="Run Backtest", command=self.run_backtest).grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        # Trading Simulation
        trade_frame = ttk.LabelFrame(self, text="Trading Simulation")
        trade_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(trade_frame, text="Select Asset:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.trade_asset_dropdown = ttk.Combobox(trade_frame, values=list(ASSET_VALUES.keys()), state="readonly")
        self.trade_asset_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        ttk.Label(trade_frame, text="Amount:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.amount_entry = ttk.Entry(trade_frame)
        self.amount_entry.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        ttk.Label(trade_frame, text="Trade Type:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.trade_type_var = tk.StringVar()
        self.trade_type_buy = ttk.Radiobutton(trade_frame, text="Buy", variable=self.trade_type_var, value="buy")
        self.trade_type_buy.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.trade_type_sell = ttk.Radiobutton(trade_frame, text="Sell", variable=self.trade_type_var, value="sell")
        self.trade_type_sell.grid(row=2, column=2, padx=5, pady=5, sticky='w')

        ttk.Label(trade_frame, text="Option Type (optional):").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.option_type_var = tk.StringVar()
        self.option_type_call = ttk.Radiobutton(trade_frame, text="Call", variable=self.option_type_var, value="call")
        self.option_type_call.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.option_type_put = ttk.Radiobutton(trade_frame, text="Put", variable=self.option_type_var, value="put")
        self.option_type_put.grid(row=3, column=2, padx=5, pady=5, sticky='w')

        self.simulate_trade_button = ttk.Button(trade_frame, text="Simulate Trade", command=self.simulate_trade)
        self.simulate_trade_button.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        self.portfolio_value_button = ttk.Button(trade_frame, text="Calculate Portfolio Value", command=self.calculate_portfolio_value)
        self.portfolio_value_button.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        self.trade_report_button = ttk.Button(trade_frame, text="Generate Trade Report", command=self.generate_trade_report)
        self.trade_report_button.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        # Results Display
        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.results_text = tk.Text(results_frame, height=5)
        self.results_text.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')
        self.results_text_scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=self.results_text_scrollbar.set)
        self.results_text_scrollbar.grid(row=0, column=1, sticky='ns')

        # Debug Mode Toggle Button
        self.debug_mode_button = ttk.Button(self, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.grid(row=5, column=0, padx=10, pady=5, sticky='ew')

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO if self.is_debug_mode else logging.ERROR, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_message(self, message):
        if self.is_debug_mode:
            self.results_text.insert(tk.END, message + "\n")
            self.results_text.yview(tk.END)

    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        debug_status = "ON" if self.is_debug_mode else "OFF"
        self.log_message(f"Debug mode turned {debug_status}")
        self.debug_mode_button.config(text=f"Debug Mode: {debug_status}")

    def browse_model(self):
        self.log_message("Browsing for model file")
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.h5"), ("Pickle Files", "*.pkl"), ("Joblib Files", "*.joblib")]
        )
        if file_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, file_path)
            self.load_model(file_path)

    def load_model(self, file_path=None):
        self.log_message("Loading model")
        if not file_path:
            file_path = self.model_path_entry.get()
        if file_path:
            self.model_manager.load_model(self.update_ui)

    def browse_data(self):
        self.log_message("Browsing for data file")
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            self.data_path_entry.delete(0, tk.END)
            self.data_path_entry.insert(0, file_path)
            self.log_message(f"Data file selected: {file_path}")

    def import_data(self):
        selected_asset = self.asset_dropdown.get()
        if selected_asset in ASSET_VALUES:
            self.data_manager.import_data(selected_asset, self.update_ui)
        else:
            self.update_ui("Invalid asset selected. Please select a valid asset.")

    def calculate_metrics(self):
        selected_asset = self.asset_dropdown.get()
        if selected_asset in ASSET_VALUES:
            self.data_manager.calculate_metrics(selected_asset, self.update_ui)
        else:
            self.update_ui("Invalid asset selected. Please select a valid asset.")

    def display_asset_details(self):
        selected_asset = self.asset_dropdown.get()
        if selected_asset in ASSET_VALUES:
            asset_info = ASSET_VALUES[selected_asset]
            details = (
                f"ID: {asset_info['id']}\n"
                f"Risk Factor: {asset_info['risk_factor']}\n"
                f"Market Segment: {asset_info['market_segment']}\n"
                f"Sector: {asset_info['sector']}\n"
                f"Beta: {asset_info['beta']}\n"
                f"Dividend Yield: {asset_info['dividend_yield']}%\n"
                f"Market Cap: {asset_info['market_cap']}\n"
            )
            options_details = "Options:\n" + "\n".join(
                [f"  - Strike Price: {opt['strike_price']}, Expiry Date: {opt['expiry_date']}, Type: {opt['option_type']}"
                 for opt in asset_info['options']]
            )
            self.update_ui(details + options_details)
        else:
            self.update_ui("Invalid asset selected. Please select a valid asset.")

    def simulate_trade(self):
        selected_asset = self.trade_asset_dropdown.get()
        amount = self.amount_entry.get()
        trade_type = self.trade_type_var.get()
        trade_option_type = self.option_type_var.get() if self.option_type_var.get() else None

        if selected_asset in ASSET_VALUES:
            asset_id = ASSET_VALUES[selected_asset]['id']
            risk_factor = ASSET_VALUES[selected_asset]['risk_factor']
            price = self.data_manager.historical_data[selected_asset]['Close'].iloc[-1]
            try:
                amount = float(amount)
                self.trade_simulator.simulate_trade(selected_asset, amount, trade_type, price, risk_factor, trade_option_type)
                self.update_ui(f"Trade simulated: {trade_type.capitalize()} {amount} of {selected_asset} at {price}")
            except ValueError:
                self.update_ui("Invalid amount. Please enter a numeric value.")
        else:
            self.update_ui("Invalid asset selected. Please select a valid asset.")

    def calculate_portfolio_value(self):
        portfolio_value = self.trade_simulator.calculate_portfolio_value()
        self.update_ui(f"Total portfolio value: {portfolio_value}")

    def generate_trade_report(self):
        report = self.trade_simulator.generate_trade_report()
        report_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if report_file_path:
            report.to_csv(report_file_path, index=False)
            self.update_ui(f"Trade report saved to {report_file_path}")

    def run_backtest(self):
        selected_asset = self.asset_dropdown.get()
        if selected_asset in ASSET_VALUES:
            historical_data = self.data_manager.get_data(selected_asset)
            model = self.model_manager.model
            if model:
                self.trade_simulator.backtest(model, historical_data, selected_asset)
                final_value = self.trade_simulator.calculate_portfolio_value()
                self.update_ui(f"Final portfolio value after backtest: {final_value}")
            else:
                self.update_ui("No model loaded. Please load a model first.")
        else:
            self.update_ui("Invalid asset selected. Please select a valid asset.")

    def backtest_model(self, model, historical_data):
        if historical_data.empty:
            self.update_ui("No historical data available. Please import data first.")
            return

        X = historical_data.drop(columns=['Close']).values
        y_true = historical_data['Close'].values
        y_pred = model.predict(X)

        if isinstance(model, (keras.models.Model, torch.nn.Module, joblib.Memory)):
            self.evaluate_regression_model(y_true, y_pred)
        elif isinstance(model, (torch.nn.Module, joblib.Memory)):
            self.evaluate_classification_model(y_true, y_pred)

        self.plot_predictions(y_true, y_pred)

    def plot_predictions(self, y_true, y_pred):
        fig, ax = plt.subplots()
        ax.plot(y_true, label='True Values')
        ax.plot(y_pred, label='Predicted Values')
        ax.legend()
        ax.set_title("True vs. Predicted Values")
        plot_window = tk.Toplevel(self)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def evaluate_regression_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        metrics_text = f"Mean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}\nMean Absolute Error: {mae:.2f}"
        self.update_ui(metrics_text)

    def evaluate_classification_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        metrics_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
        self.update_ui(metrics_text)

    def update_ui(self, message):
        self.log_message(message)
        messagebox.showinfo("Information", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelEvaluationApp(root, is_debug_mode=True)
    app.pack(fill="both", expand=True)
    root.mainloop()

#data_fetch_tab.py

from Utils import log_message
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkcalendar import DateEntry
import threading
import configparser
from alpha_vantage.timeseries import TimeSeries
import time

# Import your specific modules for data fetching
from Data_fetch.main import main as fetch_main

class DataFetchTab:
    def __init__(self, parent, config, is_debug_mode):
        self.parent = parent
        self.config = config
        self.is_debug_mode = is_debug_mode
        self.setup_data_fetch_tab()

    def setup_data_fetch_tab(self):
        # Set up the Data Fetch tab UI elements here
        tk.Label(self.parent, text="Stock Tickers (comma separated):").pack()
        self.tickers_entry = tk.Entry(self.parent)
        self.tickers_entry.pack()

        tk.Label(self.parent, text="Start Date:").pack()
        self.start_date_entry = DateEntry(self.parent, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.start_date_entry.pack()

        tk.Label(self.parent, text="End Date:").pack()
        self.end_date_entry = DateEntry(self.parent, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.end_date_entry.pack()

        tk.Label(self.parent, text="Select API:").pack()
        self.api_var = tk.StringVar()
        self.api_dropdown = ttk.Combobox(self.parent, textvariable=self.api_var, values=["AlphaVantage", "polygonio", "Nasdaq"])
        self.api_dropdown.pack()

        self.fetch_button = tk.Button(self.parent, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.pack()

        self.status_label = tk.Label(self.parent, text="")
        self.status_label.pack()

        self.log_text = scrolledtext.ScrolledText(self.parent, height=10)
        self.log_text.pack()

        self.fetch_all_button = tk.Button(self.parent, text="Fetch All Data", command=self.fetch_all_data)
        self.fetch_all_button.pack()

        # Toggle Debug Mode Button
        self.debug_mode_button = tk.Button(self.parent, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.pack()


    def fetch_data(self):
        if not self.validate_inputs():
            return

        csv_dir = self.tickers_entry.get()
        ticker_symbols = self.tickers_entry.get().split(',')
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        selected_api = self.api_var.get()

        if self.is_debug_mode:
            log_message("Debug: Fetch request received", self.log_text)
            log_message(f"Debug: CSV Directory - {csv_dir}", self.log_text)
            log_message(f"Debug: Tickers - {ticker_symbols}", self.log_text)
            log_message(f"Debug: Start Date - {start_date}, End Date - {end_date}", self.log_text)
            log_message(f"Debug: API Selected - {selected_api}", self.log_text)


        self.fetch_button.config(state=tk.DISABLED)
        self.status_label.config(text="Fetching data...")
        threading.Thread(target=lambda: self.fetch_data_threaded(csv_dir, ticker_symbols, start_date, end_date, selected_api)).start()

    def fetch_all_data(self):
        try:
            if self.is_debug_mode:
                log_message("Debug: Fetch All request received", self.log_text)

            self.fetch_button.config(state=tk.DISABLED)
            self.fetch_all_button.config(state=tk.DISABLED)
            self.status_label.config(text="Fetching all available data...")


            # Retrieve the Alpha Vantage API key from the config file
            api_key = self.config.get('API_KEYS', 'AlphaVantage', fallback='YOUR_DEFAULT_API_KEY')

            # Initialize Alpha Vantage TimeSeries
            ts = TimeSeries(key=api_key, output_format='pandas')

            # Fetch the tickers entered by the user
            tickers = self.tickers_entry.get().split(',')

            # Validate that tickers have been entered
            if not tickers or not all(tickers):
                messagebox.showwarning("Input Error", "Please specify stock symbols.")
                self.fetch_button.config(state=tk.NORMAL)
                self.fetch_all_button.config(state=tk.NORMAL)
                return

            for ticker in tickers:
                data, meta_data = ts.get_daily(symbol=ticker.strip(), outputsize='full')
                file_name = f'{ticker.strip()}_daily_data.csv'
                data.to_csv(file_name)
                time.sleep(12)  # Sleep to avoid hitting API rate limit

            self.status_label.config(text="All data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            if self.is_debug_mode:
                log_message("Debug: An error occurred during data fetch", self.log_text)
                log_message(f"Debug: Error details - {error_message}", self.log_text)
            messagebox.showerror("Error", error_message)

        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)
            
    def validate_inputs(self):
        ticker_symbols = self.tickers_entry.get().split(',')
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        if not ticker_symbols or not all(ticker_symbols):
            messagebox.showwarning("Input Error", "Please specify stock symbols.")
            return False

        if start_date > end_date:
            messagebox.showwarning("Input Error", "End date must be after start date.")
            return False

        return True

    def fetch_data_threaded(self, csv_dir, ticker_symbols, start_date, end_date, selected_api):
        try:
            fetch_main(csv_dir, ticker_symbols, start_date, end_date, selected_api)
            if self.is_debug_mode:
                log_message("Debug: Data fetch completed successfully.", self.log_text, self.is_debug_mode)
            self.status_label.config(text="Data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            if self.is_debug_mode:
                log_message("Debug: An error occurred during data fetch", self.log_text, self.is_debug_mode)
                log_message(f"Debug: Error details - {error_message}", self.log_text, self.is_debug_mode)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)

    def toggle_debug_mode(self):
        # Toggle debug mode state
        self.is_debug_mode = not self.is_debug_mode

        # Update the button text to reflect the current state
        if self.is_debug_mode:
            self.debug_mode_button.config(text="Debug Mode: ON")
        else:
            self.debug_mode_button.config(text="Debug Mode: OFF")


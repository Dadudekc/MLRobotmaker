#data_fetch_tab.py

#Part 1: Import Statements and Initial Class Definition

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
from pyttktooltip import ToolTip  # Additional import for tooltips
import threading
import configparser
import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from Utils import log_message
from Data_fetch.main import main as fetch_main

class DataFetchTab:
    def __init__(self, parent, config, is_debug_mode):
        self.parent = parent
        self.config = config
        self.is_debug_mode = is_debug_mode
        self.setup_data_fetch_tab()

#Part 2: Setup Function with Improved Layout and Styles
        
    def setup_data_fetch_tab(self):
        # Apply theme and styles
        style = ttk.Style()
        style.theme_use('clam')  # or other themes like 'alt', 'default', 'classic'
        style.configure('TButton', font=('Arial', 10), borderwidth='1')
        style.map('TButton', foreground=[('active', '!disabled', 'green')], background=[('active', 'black')])

        # Input Frame
        input_frame = ttk.Frame(self.parent)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Label(input_frame, text="Stock Tickers (comma separated):").pack(side=tk.LEFT, padx=5)
        self.tickers_entry = tk.Entry(input_frame)
        self.tickers_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ToolTip(self.tickers_entry, text='Enter stock tickers separated by commas')

        tk.Label(input_frame, text="Start Date:").pack(side=tk.LEFT, padx=5)
        self.start_date_entry = DateEntry(input_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.start_date_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(input_frame, text="End Date:").pack(side=tk.LEFT, padx=5)
        self.end_date_entry = DateEntry(input_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.end_date_entry.pack(side=tk.LEFT, padx=5)

        self.api_var = tk.StringVar()
        self.api_dropdown = ttk.Combobox(input_frame, textvariable=self.api_var, values=["AlphaVantage", "polygonio", "Nasdaq"])
        self.api_dropdown.pack(side=tk.LEFT, padx=5)
        ToolTip(self.api_dropdown, text='Select the API to fetch data')

        self.fetch_button = tk.Button(input_frame, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.pack(side=tk.RIGHT, padx=5)

        # Directory Frame
        dir_frame = ttk.Frame(self.parent)
        dir_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(dir_frame, text="Save Directory:").pack(side=tk.LEFT, padx=5)
        self.save_directory_entry = tk.Entry(dir_frame)
        self.save_directory_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(dir_frame, text="Browse", command=self.browse_save_directory).pack(side=tk.RIGHT, padx=5)

        # Status and Log
        self.status_label = tk.Label(self.parent, text="")
        self.status_label.pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(self.parent, height=10)
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

#Part 3: Remaining Class Methods
        
    def browse_save_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_directory_entry.delete(0, tk.END)
            self.save_directory_entry.insert(0, directory)

    def fetch_data(self):
        if not self.validate_inputs():
            return

        # Get user input and split it into ticker symbols
        user_input = self.tickers_entry.get()
        ticker_symbols = user_input.split(',')
        csv_dir = self.save_directory_entry.get()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        selected_api = self.api_var.get()

        # Log messages
        if self.is_debug_mode:
            log_message("Debug: Fetch request received", self.log_text)
            # ... other log messages ...

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

            api_key = self.config.get('API_KEYS', 'AlphaVantage', fallback='YOUR_DEFAULT_API_KEY')
            ts = TimeSeries(key=api_key, output_format='pandas')
            tickers = self.tickers_entry.get().split(',')

            if not tickers or not all(tickers):
                messagebox.showwarning("Input Error", "Please specify stock symbols.")
                return

            for ticker in tickers:
                ticker = ticker.strip()
                if not ticker:
                    continue

                data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
                file_name = f'{ticker}_daily_data.csv'
                save_dir = self.save_directory_entry.get()

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file_path = os.path.join(save_dir, file_name)
                data.to_csv(file_path)
                time.sleep(12)

                if self.is_debug_mode:
                    log_message(f"Data for {ticker} saved to {file_path}", self.log_text)

            self.status_label.config(text="All data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            log_message(f"Error: {error_message}", self.log_text)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)


    def fetch_data_threaded(self, csv_dir, ticker_symbols, start_date, end_date, selected_api):
        try:
            data = fetch_main(csv_dir, ticker_symbols, start_date, end_date, selected_api)
            save_dir = self.save_directory_entry.get()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for ticker in ticker_symbols:
                ticker = ticker.strip()
                file_name = f'{ticker}_data.csv'
                file_path = os.path.join(save_dir, file_name)

                if os.path.exists(file_path):
                    existing_data = pd.read_csv(file_path)
                    combined_data = pd.concat([existing_data, data]).drop_duplicates().reset_index(drop=True)
                else:
                    combined_data = data

                combined_data.to_csv(file_path, index=False)
                log_message(f"Data for {ticker} saved to {file_path}", self.log_text)

            self.status_label.config(text="Data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            log_message(f"Error: {error_message}", self.log_text)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)


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


    def toggle_debug_mode(self):
        self.is_debug_mode = not self.is_debug_mode
        button_text = "Debug Mode: ON" if self.is_debug_mode else "Debug Mode: OFF"
        self.debug_mode_button.config(text=button_text)
        if self.is_debug_mode:
            log_message("Debug mode activated", self.log_text)
        else:
            log_message("Debug mode deactivated", self.log_text)

#Part 4: Display and Plotting Methods
    def display_data_preview(self, data):
        if data is not None:
            self.log_text.insert(tk.END, data.head().to_string())
        else:
            log_message("No data to display", self.log_text)

    def save_data_to_csv(data_frame, ticker_symbol):
        filename = f"{ticker_symbol}_data.csv"
        file_path = os.path.join("C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot/csv_files/format2", filename)
        data_frame.to_csv(file_path, index=False)
        print(f"Data for {ticker_symbol} saved to {file_path}")

    def log_fetch_status(self, message, is_success=True):
        """Logs the status of data fetching operations."""
        if is_success:
            log_message(f"Success: {message}", self.log_text)
        else:
            log_message(f"Error: {message}", self.log_text)

    def update_status_label(self, message):
        """Updates the status label with a given message."""
        self.status_label.config(text=message)

    def download_data_as_csv(self, data, file_name):
        """Downloads the fetched data as a CSV file."""
        try:
            file_path = os.path.join(self.save_directory_entry.get(), file_name)
            if file_path.endswith('.json'):
                data.to_json(file_path)
            elif file_path.endswith('.xlsx'):
                data.to_excel(file_path)
            else:
                data.to_csv(file_path)
            log_message(f"Data saved as {file_path}", self.log_text)
        except Exception as e:
            log_message(f"Error in saving file: {str(e)}", self.log_text)

    def plot_data(self, data):
        """Plots the fetched data."""
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        fig, ax = plt.subplots()
        data.plot(kind='line', ax=ax)
        self.canvas = FigureCanvasTkAgg(fig, master=self.plotting_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


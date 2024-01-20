#data_fetch_tab.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import logging
import sys
import os
import time
from tkcalendar import DateEntry
from Utilities.Utils import MLRobotUtils
from Data_fetch.main import main as fetch_main

class DataFetchTab:
    def __init__(self, parent, config, is_debug_mode):
        self.parent = parent
        self.config = config
        self.is_debug_mode = is_debug_mode

        # Initialize the MLRobotUtils with the debug mode state
        self.utils = MLRobotUtils(is_debug_mode=self.is_debug_mode)

        # Initialize with dynamic values or defaults
        self.csv_dir = self.config.get('DataDirectories', 'CSVDirectory', fallback='path/to/csv_dir')
        self.ticker_symbols = self.config.get('DataSettings', 'TickerSymbols', fallback='AAPL,GOOGL').split(',')
        self.start_date = self.config.get('DataSettings', 'StartDate', fallback='2023-01-01')
        self.end_date = self.config.get('DataSettings', 'EndDate', fallback='2023-12-31')
        self.selected_api = self.config.get('DataSettings', 'SelectedAPI', fallback='some_api')
        self.log_text = self.config.get('Logging', 'LogFile', fallback='log.txt')

        # Call the method to set up the UI components
        self.setup_data_fetch_tab()

    def setup_data_fetch_tab(self):
        # Create the main frame for the data fetch tab
        main_frame = ttk.Frame(self.parent)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Input Parameters Frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Labels and Entries
        # Create a label and an entry widget for stock tickers
        self.tickers_entry = ttk.Entry(input_frame, width=30)
        self.start_date_entry = DateEntry(input_frame, width=30)  # Initialize start_date_entry
        self.end_date_entry = DateEntry(input_frame, width=30)    # Initialize end_date_entry
        self.api_dropdown = ttk.Combobox(input_frame, values=["Alpha Vantage", "Polygon.io", "Nasdaq"])  # Initialize api_dropdown

        ttk.Label(input_frame, text="Stock Tickers (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        entries = [self.tickers_entry, self.start_date_entry, self.end_date_entry, self.api_dropdown]

        # Define the labels here
        labels = ["Stock Tickers (comma separated):", "Start Date:", "End Date:", "Select API:"]

        for i, label_text in enumerate(labels):
            ttk.Label(input_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=10, pady=10)
            entries[i].grid(row=i, column=1, padx=10, pady=10, sticky="ew")

        # Fetch Buttons
        self.fetch_button = ttk.Button(input_frame, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=len(labels) + 1, column=0, padx=5, pady=10, sticky='ew')

        # Save Directory Frame
        dir_frame = ttk.LabelFrame(main_frame, text="Save Directory", padding=10)
        dir_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Create and assign the save_directory_entry widget
        self.save_directory_entry = ttk.Entry(dir_frame, width=30)
        self.save_directory_entry.grid(row=0, column=0, padx=5, sticky="ew")

        # Browse Button for save directory
        ttk.Button(dir_frame, text="Browse", command=self.utils.browse_save_directory).grid(row=0, column=1, padx=10, pady=10)

        # Status and Log Frame
        status_log_frame = ttk.Frame(main_frame)
        status_log_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        main_frame.rowconfigure(2, weight=1)
        status_log_frame.columnconfigure(0, weight=1)

        # Create and assign the status_label widget
        self.status_label = ttk.Label(status_log_frame, text="")
        self.status_label.grid(row=0, column=0, padx=5, sticky="ew")

        # Create and assign the log_text widget
        self.log_text = tk.Text(status_log_frame, wrap=tk.WORD, height=10, width=40)
        self.log_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create a frame for the candlestick chart
        self.chart_frame = tk.Frame(self.parent)
        self.chart_frame.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

        # Initialize 'data' to None
        self.data = None

        # Create an instance of CandlestickChart
        self.candlestick_chart = None

        # Create "Create Chart" button
        self.create_chart_button = ttk.Button(input_frame, text="Create Chart", command=self.candlestick_chart)
        self.create_chart_button.grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky='ew')

        # Fetch All Button
        self.fetch_all_button = ttk.Button(input_frame, text="Fetch All Data", command=self.fetch_all_data)
        self.fetch_all_button.grid(row=len(labels), columnspan=2, padx=5, pady=10)

        # Debug Mode Button
        self.debug_mode_button = ttk.Button(main_frame, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.grid(row=3, column=0, pady=10)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(main_frame, mode="determinate")  # Change mode to "determinate"
        self.progress_bar.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        # Status Label
        self.status_label.config(text="Ready to fetch data...")
        self.status_label.grid(row=5, column=0, padx=5, sticky="ew")

        # Additional UI elements can be set up here

    def fetch_main(self):
        """Main method to handle data fetching logic."""
        try:
            # Assuming fetch_main is a function that fetches data based on the parameters
            data = fetch_main(self.csv_dir, self.ticker_symbols, self.start_date, self.end_date, self.selected_api)

            # Process and save the data
            # Implement your data processing and saving logic here

            self.utils.log_message("Data fetch and processing completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            self.utils.log_message(error_message)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)
            self.progress_bar.stop()
            self.status_label.config(text="Data fetch completed")

    def fetch_data(self):
        """Button callback to start data fetching process."""
        self.fetch_button.config(state=tk.DISABLED)
        self.fetch_all_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_label.config(text="Fetching data...")
        threading.Thread(target=self.fetch_main).start()

    def fetch_data_threaded(self):
        """Start the data fetching process in a threaded manner."""
        # This method will start the fetch_main method in a separate thread
        self.fetch_button.config(state=tk.DISABLED)
        self.fetch_all_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_label.config(text="Fetching data...")
        threading.Thread(target=self.fetch_main).start()

    def validate_inputs(self):
        """Validate user inputs before fetching data."""
        ticker_symbols = self.tickers_entry.get().split(',')
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        if not ticker_symbols or not all(ticker_symbols):
            messagebox.showwarning("Input Error", "Please specify at least one stock symbol.")
            return False

        if start_date is None or end_date is None:
            messagebox.showwarning("Input Error", "Please select valid start and end dates.")
            return False

        if start_date > end_date:
            messagebox.showwarning("Input Error", "Start date must be before end date.")
            return False

        return True

    def toggle_debug_mode(self):
        """Toggle the debug mode state and update the UI accordingly."""
        self.is_debug_mode = not self.is_debug_mode
        button_text = "Debug Mode: ON" if self.is_debug_mode else "Debug Mode: OFF"
        self.debug_mode_button.config(text=button_text)

        # Log the state change for debugging purposes
        if self.is_debug_mode:
            self.utils.log_message("Debug mode enabled", self.log_text,self.log_text_widget, self.is_debug_mode)
        else:
            self.utils.log_message("Debug mode disabled", self.log_text, self.is_debug_mode)

        # Optionally, perform any additional actions required when toggling debug mode


    def fetch_all_data(self):
        try:
            if self.is_debug_mode:
                self.utils.log_message("Debug: Fetch All request received", self.log_text)
            self.fetch_button.config(state=tk.DISABLED)
            self.fetch_all_button.config(state=tk.DISABLED)
            # Start the progress bar here
            if self.progress_bar:
                self.progress_bar["value"] = 0  # Reset progress to 0
                self.progress_bar.start()

                self.progress_bar.start()
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

            for idx, ticker in enumerate(tickers):
                ticker = ticker.strip()
                if not ticker:
                    continue

                # Update progress status
                progress_message = f"Fetching data for {ticker} ({idx + 1}/{len(tickers)})"
                self.status_label.config(text=progress_message)

                # Fetch data for each ticker symbol
                data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
                file_name = f'{ticker}_daily_data.csv'
                save_dir = self.save_directory_entry.get()
                if not save_dir:
                    messagebox.showwarning("Save Directory", "Please specify a save directory.")
                    continue

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file_path = os.path.join(save_dir, file_name)
                data.to_csv(file_path)
                time.sleep(12)  # Sleep to avoid hitting API rate limit

                if self.is_debug_mode:
                    self.utils.log_message(f"Data for {ticker} saved to {file_path}", self.log_text)

            self.status_label.config(text="All data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            self.utils.log_message(f"Debug: An error occurred during data fetch", self.log_text, self.is_debug_mode)

            self.utils.log_message(f"Debug: Error details - {error_message}", self.log_text)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)


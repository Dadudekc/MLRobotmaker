#data_fetch_tab.py
#attempting to integrate shared data store for better data 
# Import necessary libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
import threading
import configparser
import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import logging
import sys
sys.path.append(r'C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MLRobot')
from Utilities.utils import MLRobotUtils
from Data_fetch.main import main as fetch_main
from Data_fetch.alpha_vantage_df import AlphaVantageDataFetcher
from Data_processing.visualization import ChartCreator
from Utilities.shared_data_store import Observer, SharedDataStore

# Define the project directory here
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the path to the config file
config_path = os.path.join(project_dir, 'config.ini')

class DataFetchTab(tk.Frame, Observer):
    def __init__(self, parent, shared_data_store, config=None, is_debug_mode=False):
        super().__init__(parent)
        self.shared_data_store = SharedDataStore()
        self.shared_data_store.register_observer(self, interest='dataset_update')  # Only interested in dataset updates
        self.config = config
        self.is_debug_mode = is_debug_mode
        self.api_var = tk.StringVar()

        # Initialize other attributes...
        self.tickers_entry = None
        self.start_date_entry = None
        self.end_date_entry = None
        self.api_dropdown = None
        self.save_directory_entry = None
        self.utils = MLRobotUtils()
        self.chart_frame = None
        self.candlestick_chart = None

        # Call methods to setup the tab's UI components
        self.setup_data_fetch_tab()

    def setup_data_fetch_tab(self):
        # Create the main frame for the data fetch tab
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.pack(fill='both', expand=True)  # Make sure to pack or grid it within self
        # Input Parameters Frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Labels and Entries
        # Create a label and an entry widget for stock tickers
        self.tickers_entry = ttk.Entry(input_frame, width=30)
        self.start_date_entry = DateEntry(input_frame, width=30)  # Initialize start_date_entry
        self.end_date_entry = DateEntry(input_frame, width=30)    # Initialize end_date_entrysa
        self.api_dropdown = ttk.Combobox(input_frame, textvariable=self.api_var, values=["Alpha Vantage", "Polygon.io", "Nasdaq"])
        self.api_dropdown.bind('<<ComboboxSelected>>', lambda e: self.api_var.set(self.api_dropdown.get()))



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
        ttk.Button(dir_frame, text="Browse", command=self.browse_save_directory).grid(row=0, column=1, padx=10, pady=10)

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
        self.chart_frame = tk.Frame(self)
        self.chart_frame.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

        # Initialize 'data' to None
        self.fetched_data = None

        # Create an instance of CandlestickChart
        self.candlestick_chart = None

        # Create "Create Chart" button
        self.create_chart_button = ttk.Button(input_frame, text="Create Chart", command=self.generate_candlestick_chart)

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

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            return file_path
        return None
    
    def browse_save_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_directory_entry.delete(0, tk.END)
            self.save_directory_entry.insert(0, directory)

    def update(self, message, event_type):
        # Implementing the observer update method
        # Make sure to update the UI in a thread-safe way
        def ui_update():
            if event_type == "dataset_update":
                self.refresh_data_view(message)
            elif event_type == "model_update":
                # Handle model updates if necessary
                pass

        self.after(0, ui_update)  # Schedule the UI update to be run in the main thread

    def refresh_data_view(self, message):
        # This method updates parts of the UI related to data display, such as refreshing a data table or list.
        # Implementation depends on your UI design and requirements
        self.status_label.config(text=message)

    def fetch_data(self):
        if not self.validate_inputs():
            return

        try:
            # Initialize or reset the progress bar
            self.progress_bar.start()

            # Get user input
            user_input = self.tickers_entry.get()
            ticker_symbols = user_input.split(',')
            csv_dir = self.save_directory_entry.get()
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            selected_api = self.api_dropdown.get()

            # Create an instance of AlphaVantageDataFetcher
            av_fetcher = AlphaVantageDataFetcher(config_path, csv_dir)

            # Initialize an empty DataFrame to store combined data
            combined_data = pd.DataFrame()

            # Fetch and save data for each symbol
            for symbol in ticker_symbols:
                symbol = symbol.strip()
                if not symbol:
                    continue

                if selected_api == 'Alpha Vantage':
                    av_data = av_fetcher.fetch_data([symbol], start_date, end_date)
                    if av_data is not None:
                        # Convert columns to numeric and handle non-numeric values
                        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                        for column in numeric_columns:
                            av_data[column] = pd.to_numeric(av_data[column], errors='coerce')
                        av_data.fillna(method='ffill', inplace=True)

                        # Append this symbol's data to the combined DataFrame
                        combined_data = pd.concat([combined_data, av_data])

                        # Save individual symbol data to CSV
                        av_fetcher.save_data_to_csv(av_data, symbol)

            # Store the combined fetched data
            self.fetched_data = combined_data
            self.status_label.config(text="Data fetch completed")
            self.shared_data_store.add_dataset("fetched_data_name", fetched_data)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.progress_bar.stop()
            self.progress_bar['value'] = 0  # Reset progress bar to 0

    def fetch_all_data(self):
        try:
            if self.is_debug_mode:
                self.utils.log_message("Debug: Fetch All request received", self.log_text, self.is_debug_mode)
            self.fetch_button.config(state=tk.DISABLED)
            self.fetch_all_button.config(state=tk.DISABLED)
            # Start the progress bar here
            if self.progress_bar:
                self.progress_bar["value"] = 0  # Reset progress to 0
                self.progress_bar.start()

                self.progress_bar.start()
                self.status_label.config(text="Fetching all available data...")
 
            # Retrieve the Alpha Vantage API key from the config file
            api_key = self.config.get('API', 'alphavantage', fallback='YOUR_DEFAULT_API_KEY')

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
                    self.utils.log_message(f"Data for {ticker} saved to {file_path}", self.log_text, self.is_debug_mode)

            
            self.status_label.config(text="All data fetch completed")
            self.shared_data_store.add_dataset("fetched_data_name", fetched_data)

        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            self.utils.log_message(f"Debug: An error occurred during data fetch", self.log_text, self.is_debug_mode)

            self.utils.log_message(f"Debug: Error details - {error_message}", self.log_text, self.is_debug_mode)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)
                
    def fetch_data_threaded(self, csv_dir, ticker_symbols, start_date, end_date, selected_api):
        try:
            # Initialize a list to store error messages
            error_messages = []

            # Initialize progress_percent to 0 at the start
            progress_percent = 0

            # Get the save directory
            save_dir = self.save_directory_entry.get()
            if not save_dir:
                messagebox.showwarning("Save Directory", "Please specify a save directory.")
                return

            # Check if the save directory is specified and exists, create if not
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Create a determinate progress bar for tracking the merge progress
            self.progress_bar.config(mode="determinate")
            self.status_label.config(text="Merging data... 0%")
            self.parent.update_idletasks()  # Force update the UI

            # Log the inputs
            self.utils.log_message(f"Fetching data for tickers: {ticker_symbols}, start date: {start_date}, end date: {end_date}, API: {selected_api}", self.log_text, self.is_debug_mode)

            # Iterate through each ticker symbol to handle each data file
            for idx, ticker in enumerate(ticker_symbols):
                ticker = ticker.strip()
                if not ticker:
                    continue

                # Update progress_percent for each ticker
                progress_percent = int((idx + 1) / len(ticker_symbols) * 100)

                # Log the fetch attempt
                self.utils.log_message(f"Fetching data for {ticker}: Start Date - {start_date}, End Date - {end_date}, API - {selected_api}", self.log_text, self.is_debug_mode)

                # Fetch the data
                data = fetch_main(csv_dir, [ticker], start_date, end_date, selected_api)

                # Log if data is empty or not
                if data is not None and not data.empty:
                    self.utils.log_message(f"Data successfully fetched for {ticker}", self.log_text, self.is_debug_mode)
                else:
                    self.utils.log_message(f"No data fetched for {ticker}", self.log_text, self.is_debug_mode)

                file_name = f'{ticker}_data.csv'
                file_path = os.path.join(save_dir, file_name)

                try:
                    # Check if data is not None before saving
                    if data is not None and not data.empty:
                        # Save the data to a CSV file
                        data.to_csv(file_path, index=False)
                        self.utils.log_message(f"Data for {ticker} saved to {file_path}", self.log_text, self.is_debug_mode)
                    else:
                        raise ValueError(f"No data returned for ticker {ticker}")

                    # Update progress bar and label
                    self.progress_bar["value"] = progress_percent
                    self.status_label.config(text=f"Merging data... {progress_percent}%")
                    self.parent.update_idletasks()  # Force update the UI
                except Exception as e:
                    # Handle errors for individual ticker symbols
                    error_messages.append(f"Error for {ticker}: {str(e)}")
                    # Log the error
                    self.log_fetch_status(f"Error for {ticker}: {str(e)}", is_success=False)

            # Update the status label
            if error_messages:
                error_message = "Errors occurred during data merge:\n" + "\n".join(error_messages)
                messagebox.showerror("Error", error_message)
                self.status_label.config(text="Data merge completed with errors")
            else:
                self.status_label.config(text="Data merge completed")

        except Exception as e:
            # Handle errors and update status/progress accordingly
            error_message = f"Error during data fetch: {str(e)}"
            self.utils.log_message(f"Error: {error_message}", self.log_text, self.is_debug_mode)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)
            # Stop the progress bar here
            self.progress_bar.stop()
            self.progress_bar['value'] = 100  # Reset progress bar to 0
            self.status_label.config(text="Data fetch completed")
            self.progress_bar.destroy()


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

        # Toggle the debug mode flag
        self.is_debug_mode = not self.is_debug_mode

        # Update the button text based on the new state
        button_text = "Debug Mode: ON" if self.is_debug_mode else "Debug Mode: OFF"

        # Update style dynamically
        style = ttk.Style()
        if self.is_debug_mode:
            # Apply custom style for debug mode
            style.configure("DebugMode.TButton", background="green", foreground="white", font=('Helvetica', 10, 'bold'))
            self.debug_mode_button.config(style="DebugMode.TButton")
        else:
            # Revert to original style
            orig_background, orig_foreground = self.original_button_style
            style.configure("TButton", background=orig_background, foreground=orig_foreground)
            self.debug_mode_button.config(style="TButton")

        self.debug_mode_button.config(text=button_text)

        # Log the state change for debugging purposes
        log_message = "Debug mode turned on" if self.is_debug_mode else "Debug mode turned off"
        print(log_message)  # Replace with a proper logging mechanism if available

        # Optionally, perform any additional actions required when toggling debug mode
        # For example, update other UI elements or internal state

    def log_fetch_status(self, message, is_success=True):
        """Logs the status of data fetching operations."""
        # Define a logger with a custom format and file handler
        logger = logging.getLogger("fetch_status")
        logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        
        # Create a file handler for storing logs in a file
        file_handler = logging.FileHandler("fetch_status.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)

        if is_success:
            logger.info(f"Success: {message}")
            self.utils.log_message(f"Success: {message}", self.log_text, self.is_debug_mode)
        else:
            logger.error(f"Error: {message}")
            self.utils.log_message(f"Error: {message}", self.log_text, self.is_debug_mode)
                
    def generate_candlestick_chart(self):
        try:
            # Check if the fetched data is available
            if self.fetched_data is None or self.fetched_data.empty:
                raise ValueError("No data available for creating the chart. Fetch data first.")

            # Create an instance of ChartCreator with the fetched data
            chart_creator = ChartCreator(self.parent, data=self.fetched_data)
            title = "Candlestick Chart"
            save_path = os.path.join(self.save_directory_entry.get(), "candlestick_chart.png")
            chart_creator.create_candlestick_chart(title=title, save_path=save_path)

            # Update the status label
            self.status_label.config(text=f"Candlestick chart created and saved as {save_path}")
        except Exception as e:
            # Handle errors if any
            error_message = f"Error creating candlestick chart: {str(e)}"
            self.utils.log_message(error_message, self.log_text, self.is_debug_mode)
            messagebox.showerror("Error", error_message)

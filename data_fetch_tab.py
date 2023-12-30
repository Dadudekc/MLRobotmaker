import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
import threading
import configparser
import os
import time
import pandas as pd  # Added import for pandas
from alpha_vantage.timeseries import TimeSeries
from Utils import log_message

# Import your specific modules for data fetching
from Data_fetch.main import main as fetch_main

class DataFetchTab:
    def __init__(self, parent, config, is_debug_mode):
        self.parent = parent
        self.config = config
        self.is_debug_mode = is_debug_mode
        self.setup_data_fetch_tab()

    def setup_data_fetch_tab(self):
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(input_frame, text="Stock Tickers (comma separated):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.tickers_entry = ttk.Entry(input_frame, width=30)
        self.tickers_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(input_frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_date_entry = DateEntry(input_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="End Date:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.end_date_entry = DateEntry(input_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Select API:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_var = tk.StringVar()
        self.api_dropdown = ttk.Combobox(input_frame, textvariable=self.api_var, values=["AlphaVantage", "polygonio", "Nasdaq"])
        self.api_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Fetch Button
        self.fetch_button = ttk.Button(input_frame, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=4, column=0, padx=5)

        dir_frame = ttk.LabelFrame(main_frame, text="Save Directory", padding=10)
        dir_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.save_directory_entry = ttk.Entry(dir_frame, width=30)
        self.save_directory_entry.grid(row=0, column=0, padx=5, sticky="ew")
        ttk.Button(dir_frame, text="Browse", command=self.browse_save_directory).grid(row=0, column=1, padx=5, pady=5)

        status_log_frame = ttk.Frame(main_frame)
        status_log_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        main_frame.rowconfigure(2, weight=1)
        status_log_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(status_log_frame, text="")
        self.status_label.grid(row=0, column=0, padx=5, sticky="ew")
        self.log_text = scrolledtext.ScrolledText(status_log_frame, height=10)
        self.log_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.plotting_frame = ttk.Frame(main_frame, padding=10)
        self.plotting_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=4, column=0, padx=10, pady=5, sticky="e")

        # Fetch All Button (Moved next to Fetch Button)
        self.fetch_all_button = ttk.Button(input_frame, text="Fetch All Data", command=self.fetch_all_data)
        self.fetch_all_button.grid(row=4, column=1, padx=5)

        # Debug Mode Button (Centered at the bottom)
        self.debug_mode_button = ttk.Button(main_frame, text="Toggle Debug Mode", command=self.toggle_debug_mode)
        self.debug_mode_button.grid(row=5, column=0, pady=10)

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

        if self.is_debug_mode:
            log_message("Debug: Fetch request received", self.log_text)
            log_message(f"Debug: CSV Directory - {csv_dir}", self.log_text)
            log_message(f"Debug: Tickers - {ticker_symbols}", self.log_text)
            log_message(f"Debug: Start Date - {start_date}, End Date - {end_date}", self.log_text)
            log_message(f"Debug: API Selected - {selected_api}", self.log_text)

        # Disable the fetch button and update status
        self.fetch_button.config(state=tk.DISABLED)
        self.status_label.config(text="Fetching data...")

        # Start a new thread to fetch data
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

            for ticker in tickers:  # Use 'tickers' instead of 'ticker_symbols'
                ticker = ticker.strip()
                if not ticker:
                    continue

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
                    log_message(f"Data for {ticker} saved to {file_path}", self.log_text)

            self.status_label.config(text="All data fetch completed")
        except Exception as e:
            error_message = f"Error during data fetch: {str(e)}"
            log_message(f"Debug: An error occurred during data fetch", self.log_text)
            log_message(f"Debug: Error details - {error_message}", self.log_text)
            messagebox.showerror("Error", error_message)
        finally:
            self.fetch_button.config(state=tk.NORMAL)
            self.fetch_all_button.config(state=tk.NORMAL)

    def fetch_data_threaded(self, csv_dir, ticker_symbols, start_date, end_date, selected_api):
        try:
            # Fetch the data
            data = fetch_main(csv_dir, ticker_symbols, start_date, end_date, selected_api)

            # Get the save directory
            save_dir = self.save_directory_entry.get()
            if not save_dir:
                messagebox.showwarning("Save Directory", "Please specify a save directory.")
                return

            # Check if the save directory is specified and exists, create if not
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Iterate through each ticker symbol to handle each data file
            for ticker in ticker_symbols:
                ticker = ticker.strip()
                if not ticker:
                    continue

                file_name = f'{ticker}_data.csv'
                file_path = os.path.join(save_dir, file_name)

                # Check for existing data and merge if necessary
                if os.path.exists(file_path):
                    existing_data = pd.read_csv(file_path)
                    combined_data = pd.concat([existing_data, data]).drop_duplicates().reset_index(drop=True)
                else:
                    combined_data = data

                # Save the merged data
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
        """Validate user inputs before fetching data."""
        ticker_symbols = self.tickers_entry.get().split(',')
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        # Check if ticker symbols are provided
        if not ticker_symbols or not all(ticker_symbols):
            messagebox.showwarning("Input Error", "Please specify stock symbols.")
            return False

        # Check if start date is before end date
        if start_date > end_date:
            messagebox.showwarning("Input Error", "End date must be after start date.")
            return False

        return True

    def toggle_debug_mode(self):
        """Toggle the debug mode state and update the UI accordingly."""
        # Toggle the debug mode flag
        self.is_debug_mode = not self.is_debug_mode

        # Update the button text based on the new state
        button_text = "Debug Mode: ON" if self.is_debug_mode else "Debug Mode: OFF"
        self.debug_mode_button.config(text=button_text)

        # Optional: Log the change in debug mode state
        if self.is_debug_mode:
            log_message("Debug mode activated", self.log_text)
        else:
            log_message("Debug mode deactivated", self.log_text)

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

    def display_data_preview(self, data):
        """Displays a preview of the fetched data."""
        if data is not None:
            self.log_text.insert(tk.END, data.head().to_string())
        else:
            log_message("No data to display", self.log_text)

    def plot_data(self, data):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        fig, ax = plt.subplots()
        data.plot(kind='line', ax=ax)  # Adjust plot type and settings as needed
        self.canvas = FigureCanvasTkAgg(fig, master=self.plotting_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def save_data_to_csv(data_frame, ticker_symbol):
        # Format the filename with the stock ticker
        filename = f"{ticker_symbol}_data.csv"
        file_path = os.path.join("C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot/csv_files/format2", filename)

        # Save the DataFrame to a CSV file
        data_frame.to_csv(file_path, index=False)
        print(f"Data for {ticker_symbol} saved to {file_path}")

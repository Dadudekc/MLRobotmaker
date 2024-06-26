import os
import configparser
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import datetime

class MLRobotUtils:
    def __init__(self, is_debug_mode=False):
        self.is_debug_mode = is_debug_mode

    def log_message(self, message, root_window, log_text_widget=None):
        if self.is_debug_mode:
            print(message)  # Log to the console

        if log_text_widget and isinstance(log_text_widget, tk.Text) and root_window:
            def append_message():
                log_text_widget.config(state='normal')
                log_text_widget.insert(tk.END, message + "\n")
                log_text_widget.config(state='disabled')
                log_text_widget.see(tk.END)

            root_window.after(0, append_message)

    def get_model_types(self):
        """Return a list of supported model types."""
        return ['linear_regression', 'random_forest', 'lstm', 'neural_network', 'arima']
    
    def select_directory(self, entry):
        directory = filedialog.askdirectory()
        if self.is_debug_mode:
            self.log_message(f"Debug: Selected directory - {directory}", None)
        entry.delete(0, tk.END)
        entry.insert(0, directory)

    def save_preferences(self, config, data_fetch_entry, data_processing_entry, model_training_entry, directory_entry):
        config['DataDirectories']['DataFetchDirectory'] = data_fetch_entry.get()
        config['DataDirectories']['DataProcessingDirectory'] = data_processing_entry.get()
        config['DataDirectories']['ModelTrainingDirectory'] = model_training_entry.get()
        config['DEFAULT']['LastDirectory'] = directory_entry.get()
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def browse_directory(self, entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)
            if self.is_debug_mode:
                self.log_message(f"Debug: Directory selected - {directory}", None)

    def auto_generate_save_path(self, input_file_path, base_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))

        if extension.lower() != '.csv':
            raise ValueError("Input file is not a CSV file.")

        new_filename = f"{base_name}_processed_{timestamp}.csv"
        return os.path.join(base_dir, new_filename)

    def generate_save_path(self, file_path, config):
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        save_directory = config.get('SAVE_PATH_SECTION', 'save_path_dir', fallback=directory)
        new_filename = f"{name}_processed{extension}"
        save_path = os.path.join(save_directory, new_filename)
        return save_path

    def update_status(self, status_output, message):
        status_output.config(state=tk.NORMAL)
        status_output.delete(1.0, tk.END)
        status_output.insert(tk.END, message + "\n")
        status_output.config(state=tk.DISABLED)

    def browse_data_file(self, data_file_entry):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            data_file_entry.delete(0, tk.END)
            data_file_entry.insert(0, file_path)

    def load_configuration(self, config_file='config.ini'):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            logging.error(f"Configuration file does not exist: {config_file}")
            raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
        config.read(config_file)
        logging.info("Configuration file loaded successfully.")
        return config

    def setup_logging(self, level=logging.INFO):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    def validate_section_keys(self, config, section, required_keys):
        missing_keys = [key for key in required_keys if key not in config[section]]
        if missing_keys:
            raise ValueError(f"Missing required config key(s) in '{section}': {', '.join(missing_keys)}")

    def save_data_to_csv(self, data_frame, ticker_symbol):
        filename = f"{ticker_symbol}_data.csv"
        file_path = os.path.join("C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot/csv_files/format2", filename)
        data_frame.to_csv(file_path, index=False)
        print(f"Data for {ticker_symbol} saved to {file_path}")

    def plot_data(self, data):
        try:
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            fig, ax = plt.subplots()
            data.plot(kind='line', ax=ax)
            self.canvas = FigureCanvasTkAgg(fig, master=self.plotting_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            error_message = f"Error in plotting data: {str(e)}"
            self.log_message(error_message, self.log_text)
            messagebox.showerror("Error", error_message)

    def display_data_preview(self, data):
        if data is not None:
            self.log_text.insert(tk.END, data.head().to_string())
        else:
            self.log_message("No data to display", self.log_text)

    def download_data_as_csv(self, data, file_name):
        try:
            file_path = os.path.join(self.save_directory_entry.get(), file_name)
            if file_path.endswith('.json'):
                data.to_json(file_path)
            elif file_path.endswith('.xlsx'):
                data.to_excel(file_path)
            else:
                data.to_csv(file_path)
            self.log_message(f"Data saved as {file_path}", self.log_text)
        except Exception as e:
            error_message = f"Error in saving data as {file_name}: {str(e)}"
            self.log_message(error_message, self.log_text)
            messagebox.showerror("Error", error_message)

    def update_status_label(self, message):
        self.status_label.config(text=message)

    def create_candlestick_chart(self):
        if self.chart_frame:
            self.chart_frame.destroy()
        self.chart_frame = tk.Frame(self.parent)
        self.chart_frame.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

        file_path = self.select_file()
        if file_path:
            data = pd.read_csv(file_path)
            self.candlestick_chart = CandlestickChart(self.chart_frame, data)
            self.candlestick_chart.create_chart()

            canvas = FigureCanvasTkAgg(self.candlestick_chart.fig, master=self.chart_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def browse_file(self, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def clear_logs(self):
        if hasattr(self, 'log_text'):
            self.log_text.delete('1.0', tk.END)

    def toggle_debug_mode(self, debug_mode_button):
        self.is_debug_mode = not self.is_debug_mode
        print(f"Debug mode set to: {self.is_debug_mode}")

        if self.is_debug_mode:
            debug_mode_button.config(text="Debug Mode: ON")
            print("Debug mode is now ON")
        else:
            debug_mode_button.config(text="Debug Mode: OFF")
            print("Debug mode is now OFF")

    def browse_save_directory(self):
        save_dir = filedialog.askdirectory()
        if save_dir:
            self.csv_dir = save_dir
            self.save_dir_label.config(text=f"Save Directory: {save_dir}")

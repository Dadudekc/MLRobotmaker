import os
import configparser
import logging
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import datetime

class MLRobotUtils:
    def __init__(self, is_debug_mode=False):
        self.is_debug_mode = is_debug_mode

    def log_message(self, message, tk_parent, log_text_widget, is_debug_mode=False):
        if is_debug_mode:
            print(message)

        def append_message():
            log_text_widget.config(state='normal')
            log_text_widget.insert(tk.END, message + "\n")
            log_text_widget.config(state='disabled')
            log_text_widget.see(tk.END)

        # Ensure the GUI update is done in a thread-safe manner
        tk_parent.after(0, append_message)


    def select_directory(self, entry):
        directory = filedialog.askdirectory()
        if self.is_debug_mode:
            self.log_message(f"Debug: Selected directory - {directory}")
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
                self.log_message(f"Debug: Directory selected - {directory}")
                    
    def log_message(self, message, tk_parent, log_text_widget, is_debug_mode=False):
        if log_text_widget:
            log_text_widget.config(state=tk.NORMAL)
            log_text_widget.insert(tk.END, message + "\n")
            log_text_widget.config(state=tk.DISABLED)

        # Check if debug mode is active
        if is_debug_mode:
            print(message)





    def auto_generate_save_path(input_file_path, base_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Extract the base name (without extension) and extension
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))

        # Check if the original file is a CSV
        if extension.lower() != '.csv':
            raise ValueError("Input file is not a CSV file.")

        # Create a new filename with the timestamp
        new_filename = f"{base_name}_processed_{timestamp}.csv"
        
        return os.path.join(base_dir, new_filename)

    # Function to generate the save path based on file_path and config

    def generate_save_path(file_path, config):
        # Extract the directory and filename from the input file path
        directory, filename = os.path.split(file_path)

        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)

        # Optionally, you can use a directory specified in the config
        # For example, if your config has a section "SAVE_PATH_SECTION" with a key "save_path_dir"
        save_directory = config.get('SAVE_PATH_SECTION', 'save_path_dir', fallback=directory)

        # Construct the new filename by appending '_processed'
        new_filename = f"{name}_processed{extension}"

        # Combine the directory and new filename to form the full save path
        save_path = os.path.join(save_directory, new_filename)

        return save_path

    def update_status(status_output, message):
        status_output.config(state=tk.NORMAL)
        status_output.delete(1.0, tk.END)
        status_output.insert(tk.END, message + "\n")
        status_output.config(state=tk.DISABLED)

    def browse_data_file(data_file_entry):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            data_file_entry.delete(0, tk.END)  # Clear any existing entry
            data_file_entry.insert(0, file_path)  # Insert the selected file path into the entry field

    def load_configuration(config_file='config.ini'):
        """
        Load a configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            ConfigParser: The configuration object.
        """
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            logging.error(f"Configuration file does not exist: {config_file}")
            raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
        config.read(config_file)
        logging.info("Configuration file loaded successfully.")
        return config

    def setup_logging(level=logging.INFO):
        """
        Set up basic logging configuration.

        Args:
            level (int): Logging level.
        """
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    def validate_section_keys(config, section, required_keys):
        """
        Validate that all necessary keys are present in a specified configuration section.

        Args:
            config (ConfigParser object): Configuration object to validate.
            section (str): Section name to check in the configuration.
            required_keys (list): Required keys to check for in the section.

        Raises:
            ValueError: If any required keys are missing.
        """
        missing_keys = [key for key in required_keys if key not in config[section]]
        if missing_keys:
            raise ValueError(f"Missing required config key(s) in '{section}': {', '.join(missing_keys)}")

    def save_data_to_csv(data_frame, ticker_symbol):
        # Format the filename with the stock ticker
        filename = f"{ticker_symbol}_data.csv"
        file_path = os.path.join("C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot/csv_files/format2", filename)

        # Save the DataFrame to a CSV file
        data_frame.to_csv(file_path, index=False)
        print(f"Data for {ticker_symbol} saved to {file_path}")

    def plot_data(self, data):
        try:
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            fig, ax = plt.subplots()
            data.plot(kind='line', ax=ax)  # Adjust plot type and settings as needed
            self.canvas = FigureCanvasTkAgg(fig, master=self.plotting_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        except Exception as e:
            error_message = f"Error in plotting data: {str(e)}"
            log_message(error_message, self.log_text)
            messagebox.showerror("Error", error_message)

    def display_data_preview(self, data):
        """Displays a preview of the fetched data."""
        if data is not None:
            self.log_text.insert(tk.END, data.head().to_string())
        else:
            log_message("No data to display", self.log_text)

    def download_data_as_csv(self, data, file_name):
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
            error_message = f"Error in saving data as {file_name}: {str(e)}"
            log_message(error_message, self.log_text)
            messagebox.showerror("Error", error_message)

    def update_status_label(self, message):
        """Updates the status label with a given message."""
        self.status_label.config(text=message)

    def create_candlestick_chart(self):
        if self.chart_frame:
            self.chart_frame.destroy()
        self.chart_frame = tk.Frame(self.parent)
        self.chart_frame.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

        file_path = self.select_file()
        if file_path:
            data = pd.read_csv(file_path)
            self.candlestick_chart = CandlestickChart(self.chart_frame, data)  # Pass the loaded data here
            self.candlestick_chart.create_chart()  # Ensure this method uses the data for chart creation

            # Display the chart in the Tkinter window
            canvas = FigureCanvasTkAgg(self.candlestick_chart.fig, master=self.chart_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def browse_file(self):
        # Open a file dialog to choose a CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        # If a file is selected, update the file_path_entry with this path
        if file_path:
            self.file_path_entry.delete(0, tk.END)  # Clear any existing text in the entry
            self.file_path_entry.insert(0, file_path)  # Insert the selected file path

    def clear_logs(self):
        if hasattr(self, 'log_text'):
            self.log_text.delete('1.0', tk.END)  # Clear the content of the log text widget


    def toggle_debug_mode(self):
        # Toggle debug mode state
        self.debug_mode = not self.debug_mode

        # Update the MLRobotUtils instance with the new debug mode state
        self.utils.is_debug_mode = self.debug_mode

        # Update the button text to reflect the current state
        if self.debug_mode:
            self.debug_mode_button.config(text="Debug Mode: ON")
        else:
            self.debug_mode_button.config(text="Debug Mode: OFF")

    def browse_save_directory(self):
        """Open a file dialog to select the save directory."""
        save_dir = filedialog.askdirectory()
        if save_dir:
            self.csv_dir = save_dir  # Update the save directory
            self.save_dir_label.config(text=f"Save Directory: {save_dir}")

# Utils.py

import os
import datetime
import tkinter as tk
import pandas as pd
import logging
from tkinter import filedialog
import configparser 

class MLRobotUtils:
    def __init__(self, is_debug_mode=False):
        self.is_debug_mode = is_debug_mode

    def log_message(self, message, log_text=None):
        if self.is_debug_mode:
            print(message)

        if log_text:
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, message + "\n")
            log_text.config(state=tk.DISABLED)

    def select_directory(self, entry):
        directory = filedialog.askdirectory()
        if self.is_debug_mode:
            self.log_message(f"Debug: Selected directory - {directory}")
        entry.delete(0, tk.END)
        entry.insert(0, directory)

    def save_preferences(self, config):
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

def log_message(message, log_text=None, is_debug_mode=False):
    if log_text:
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.config(state=tk.DISABLED)
    
    # Check if debug mode is active
    if is_debug_mode:
        print(message)

def auto_generate_save_path(input_file_path, base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.basename(input_file_path).replace('.csv', f'_processed_{timestamp}.csv')
    return os.path.join(base_dir, filename)

def update_status(status_output, message):
    status_output.config(state=tk.NORMAL)
    status_output.delete(1.0, tk.END)
    status_output.insert(tk.END, message + "\n")
    status_output.config(state=tk.DISABLED)

def browse_data_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data_file_entry.delete(0, tk.END)  # Clear any existing entry
        data_file_entry.insert(0, file_path)  # Insert the selected file path into the entry field

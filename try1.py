#model_training_tab....adding hypermodel tuning...


import tkinter as tk
from tkinter import ttk
import configparser
from data_fetch_tab import DataFetchTab
from data_processing_tab import DataProcessingTab
from model_training_tab import ModelTrainingTab

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Create the main Tkinter window
root = tk.Tk()
root.title("My Main Application")

# Create the tab control
tabControl = ttk.Notebook(root)

# Data Fetch Tab
data_fetch_tab = ttk.Frame(tabControl)
tabControl.add(data_fetch_tab, text='Data Fetch')
is_debug_mode = config.getboolean('Settings', 'DebugMode', fallback=False)
data_fetch_tab_instance = DataFetchTab(data_fetch_tab, config, is_debug_mode)

# Data Processing Tab
data_processing_tab = ttk.Frame(tabControl)
tabControl.add(data_processing_tab, text='Data Processing')
data_processing_tab_instance = DataProcessingTab(data_processing_tab, config)

# Model Training Tab
model_training_tab = ttk.Frame(tabControl)
tabControl.add(model_training_tab, text='Model Training')
scaler_options = ['standard', 'minmax', 'robust', 'normalizer', 'maxabs']

# Create an instance of ModelTrainingTab and pack it inside its parent frame
model_training_tab_instance = ModelTrainingTab(model_training_tab, config, scaler_options)
model_training_tab_instance.pack(fill="both", expand=True)  # This ensures the tab fills its container

# Pack the tab control and run the main loop
tabControl.pack(expand=1, fill="both")
root.mainloop()

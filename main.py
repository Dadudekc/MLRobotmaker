# C:\DaDudeKC\MLRobot\main.py
# Main script 

import os
import tkinter as tk
from tkinter import ttk
import configparser
from Tabs.data_fetch_tab import DataFetchTab
from Tabs.data_processing_tab import DataProcessingTab
from Tabs.model_training_tab import ModelTrainingTab
from Risk_Management.risk_management_main import ModelEvaluationApp
import keras

# Global variable to store the trained model
trained_model = None

# Set the working directory
def set_working_directory(path):
    os.chdir(path)
    print("Current Working Directory: ", os.getcwd())

# Load the configuration file
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Create the main Tkinter window
def create_main_window(title):
    root = tk.Tk()
    root.title(title)
    return root

# Add tabs to the notebook
def add_tabs_to_notebook(tab_control, config, is_debug_mode):
    # Data Fetch Tab
    data_fetch_tab = ttk.Frame(tab_control)
    tab_control.add(data_fetch_tab, text='Data Fetch')
    DataFetchTab(data_fetch_tab, config, is_debug_mode)

    # Data Processing Tab
    data_processing_tab = ttk.Frame(tab_control)
    tab_control.add(data_processing_tab, text='Data Processing')
    DataProcessingTab(data_processing_tab, config)

    # Model Training Tab
    model_training_tab = ttk.Frame(tab_control)
    tab_control.add(model_training_tab, text='Model Training')
    scaler_options = ['standard', 'minmax', 'robust', 'normalizer', 'maxabs']
    ModelTrainingTab(model_training_tab, config, scaler_options).pack(fill="both", expand=True)

    # Risk Management Tab
    risk_management_tab = ttk.Frame(tab_control)
    tab_control.add(risk_management_tab, text='Risk Management')
    risk_management_tab_instance = ModelEvaluationApp(risk_management_tab, is_debug_mode)
    risk_management_tab_instance.pack(fill="both", expand=True)
    
    return risk_management_tab_instance

def main():
    # Define the path
    path = r"C:\DaDudeKC\MLRobot"
    set_working_directory(path)

    # Load configuration
    config = load_config()
    is_debug_mode = config.getboolean('Settings', 'DebugMode', fallback=False)

    # Create the main Tkinter window
    root = create_main_window("ML Robot Management Application")

    # Create the tab control
    tabControl = ttk.Notebook(root)

    # Add tabs to the notebook
    add_tabs_to_notebook(tabControl, config, is_debug_mode)

    # Pack the tab control and run the main loop
    tabControl.pack(expand=1, fill="both")
    root.mainloop()

if __name__ == "__main__":
    main()
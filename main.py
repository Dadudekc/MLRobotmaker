# Main script
# Define the path you want to navigate to
import os
path = r"C:\Users\Dagurlkc\OneDrive\Desktop\DaDudeKC\MLRobot"

# Change the current working directory to the specified path
os.chdir(path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

import tkinter as tk
from tkinter import ttk
import configparser
from tkinter.filedialog import askopenfilename  # Import askopenfilename
from Tabs.data_fetch_tab import DataFetchTab
from Tabs.data_processing_tab import DataProcessingTab
from Tabs.model_training_tab import ModelTrainingTab
from Tabs.model_evaluation_tab import ModelEvaluationTab
from Tabs.trade_analysis_tab import TradingAnalysisTab  
from Tabs.trade_description_analyzer_tab import TradeDescriptionAnalyzerTab
from Tabs.trade_analyzer_tab import TradeAnalyzerTab
from Tabs.risk_management_tab import RiskManagementTab
import keras  # Import Keras if not already imported


# Global variable to store the trained model
trained_model = None

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Create the main Tkinter window
root = tk.Tk()
root.title("Look...I did a thing")

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

# Create the Risk Management Tab
risk_management_tab = ttk.Frame(tabControl)
tabControl.add(risk_management_tab, text='Risk Management')

# Initialize RiskManagementTab without trained_model
risk_management_tab_instance = RiskManagementTab(risk_management_tab, None)  # Passing None for trained_model

# Function to select a trained model
def select_trained_model():
    global trained_model  # Use the global variable
    model_path = askopenfilename(filetypes=[("Model Files", "*.h5")])
    if model_path:
        try:
            trained_model = keras.models.load_model(model_path)
            risk_management_tab_instance.set_trained_model(trained_model)
        except Exception as e:
            print(f"Error loading model: {e}")  # Provide feedback
            risk_management_tab_instance.set_trained_model(None)

# Bind the select trained model function to a button or menu item in the Risk Management Tab
button_select_model = ttk.Button(risk_management_tab_instance, text="Select Trained Model", command=select_trained_model)
button_select_model.pack()
risk_management_tab_instance.pack(fill="both", expand=True)

# Model Evaluation Tab
model_evaluation_tab = ttk.Frame(tabControl)  # Define the model_evaluation_tab frame first
tabControl.add(model_evaluation_tab, text='Model Evaluation')

# Create an instance of ModelEvaluationTab and pass is_debug_mode
model_evaluation_tab_instance = ModelEvaluationTab(model_evaluation_tab, is_debug_mode)
model_evaluation_tab_instance.pack(fill="both", expand=True)  # Pack the instance

# Trade Analysis Tab
trade_analysis_tab = ttk.Frame(tabControl)
tabControl.add(trade_analysis_tab, text='Trade Analysis')  # Add a new tab for Trade Analysis
trade_analysis_tab_instance = TradingAnalysisTab(trade_analysis_tab)  # Create an instance of TradingAnalysisTab
trade_analysis_tab_instance.pack(fill="both", expand=True)  # Pack the instance

# Trade Description Analyzer Tab
trade_description_tab = ttk.Frame(tabControl)
tabControl.add(trade_description_tab, text='Trade Description Analyzer')

# Create an instance of TradeDescriptionAnalyzerTab and pack it inside its parent frame
trade_description_tab_instance = TradeDescriptionAnalyzerTab(trade_description_tab)
trade_description_tab_instance.pack(fill="both", expand=True)

# Pack the tab control and run the main loop
tabControl.pack(expand=1, fill="both")

root.mainloop()
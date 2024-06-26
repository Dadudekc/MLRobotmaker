# risk_management_main.py

import os
import sys
from tkinter import Tk

# Ensure the Risk_Management directory is in the sys.path
sys.path.append(os.path.dirname(__file__))

from config import load_config, ASSET_VALUES
from data_manager import HistoricalDataManager
from model_manager import ModelManager
from trade_simulator import TradeSimulator
from ui_manager import ModelEvaluationApp

def main():
    # Define the path
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)

    # Load configuration
    config = load_config()
    is_debug_mode = config.getboolean('Settings', 'DebugMode', fallback=False)

    # Create the main Tkinter window
    root = Tk()
    root.title("Risk Management Application")

    # Initialize and pack the ModelEvaluationApp
    app = ModelEvaluationApp(root, is_debug_mode)
    app.pack(fill="both", expand=True)

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()

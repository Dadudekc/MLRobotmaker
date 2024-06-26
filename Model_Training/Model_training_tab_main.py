import tkinter as tk
from utilities import MLRobotUtils
from gui_module import ModelTrainingTab

def main():
    root = tk.Tk()
    root.title("Model Training Application")
    
    # Configuration and scaler options for demonstration
    config = {"Paths": {"models_directory": "./models"}}
    scaler_options = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"]
    
    model_training_tab = ModelTrainingTab(root, config, scaler_options)
    model_training_tab.pack(expand=True, fill="both")
    
    root.mainloop()

if __name__ == "__main__":
    main()

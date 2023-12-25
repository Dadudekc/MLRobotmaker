#attempt to make main to bring all tabs together...they seem to need restructuring...

import tkinter as tk
from tkinter import ttk

# Import your custom tab classes
from model_training_tab import ModelTrainingTab
from data_processing_tab import DataProcessingTab
from data_fetch_tab import DataFetchTab

def main():
    try:
        root = tk.Tk()
        root.title("MLRobot Application")

        print("Creating notebook...")
        notebook = ttk.Notebook(root)

        print("Initializing tabs...")
        # Define your scaler options here
        scaler_options = ['standard', 'minmax', 'robust', 'normalizer', 'maxabs']

        # Initialize the tabs with the necessary arguments
        data_fetch_tab = DataFetchTab(notebook)

        print("Adding tabs to the notebook...")

        notebook.add(data_fetch_tab, text="Data Fetching")

        print("Packing notebook into the main window...")
        notebook.pack(fill=tk.BOTH, expand=True)

        print("Entering main loop...")
        root.mainloop()
        print("Main loop exited.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()

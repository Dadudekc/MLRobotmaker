#risk_management_tab.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
import tkinter.filedialog as filedialog  # Import filedialog
from risk_assessment import RiskAssessment
from risk_management_journal import RiskManagementJournal
from custom_risk_strategies import CustomRiskStrategies

class RiskManagementTab(ttk.Frame):
    def __init__(self, parent, trained_model):
        super().__init__(parent)
        self.trained_model = None  # Initialize the trained model to None initially
        self.selected_asset = tk.StringVar()  # Store the selected asset here
        self.historical_data = {}  # Store historical data for multiple assets
        self.create_widgets()

    # Part 2: User Interface Widgets
    def create_widgets(self):
        risk_frame = ttk.LabelFrame(self, text="Risk Management")
        risk_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Entry fields for stop-loss and take-profit levels
        self.stop_loss_entry = self.create_label_entry(risk_frame, "Stop Loss (%)", row=0)
        self.take_profit_entry = self.create_label_entry(risk_frame, "Take Profit (%)", row=1)

        # Label to display selected asset
        self.selected_asset_label = ttk.Label(risk_frame, text="Selected Asset: None")
        self.selected_asset_label.grid(row=8, columnspan=2, padx=10, pady=5)

        # Radio buttons for selecting assets
        assets = ["Asset 1", "Asset 2", "Asset 3"]  # Add your asset names here
        for idx, asset in enumerate(assets):
            ttk.Radiobutton(risk_frame, text=asset, variable=self.selected_asset, value=asset, command=self.update_selected_asset).grid(row=9, column=idx + 1)

        # Button to import historical trading data
        import_data_button = ttk.Button(risk_frame, text="Import Historical Data", command=self.import_historical_data)
        import_data_button.grid(row=7, columnspan=2, padx=10, pady=10)

        # Button to calculate performance metrics
        calculate_metrics_button = ttk.Button(risk_frame, text="Calculate Metrics", command=self.calculate_performance_metrics)
        calculate_metrics_button.grid(row=10, columnspan=2, padx=10, pady=10)

        # Button to select a trained model
        select_model_button = ttk.Button(risk_frame, text="Select Trained Model", command=self.select_trained_model)
        select_model_button.grid(row=13, columnspan=2, padx=10, pady=10)

        # Button to simulate risk management strategies
        simulate_button = ttk.Button(risk_frame, text="Simulate Strategies", command=self.simulate_strategies)
        simulate_button.grid(row=11, columnspan=2, padx=10, pady=10)

        # Label to display risk-related information
        self.risk_info_label = ttk.Label(risk_frame, text="")
        self.risk_info_label.grid(row=12, columnspan=2, padx=10, pady=5)

    def create_label_entry(self, parent, label_text, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        entry = ttk.Entry(parent)
        entry.grid(row=row, column=1, padx=10, pady=5)
        return entry

    # Part 3: Importing Historical Data
    def import_historical_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        
        if file_path:
            try:
                self.historical_data[self.selected_asset.get()] = pd.read_csv(file_path)
                self.risk_info_label.config(text="Historical data imported successfully.")
            except Exception as e:
                self.risk_info_label.config(text=f"Failed to import historical data: {e}")

    # Part 4: Calculating Performance Metrics
    def calculate_performance_metrics(self):
        if self.selected_asset is not None:
            asset_data = self.historical_data.get(self.selected_asset.get())
            if asset_data is not None:
                # Implement logic to calculate performance metrics here
                # Example: Calculate and display the total return
                total_return = (asset_data['Close'][-1] / asset_data['Close'][0] - 1) * 100
                self.risk_info_label.config(text=f"Total Return: {total_return:.2f}%")
            else:
                self.risk_info_label.config(text="No historical data available for the selected asset.")
        else:
            self.risk_info_label.config(text="Please select an asset before calculating metrics.")

    # Part 5: Multiple Assets and Asset Selection
    def update_selected_asset(self):
        self.selected_asset_label.config(text=f"Selected Asset: {self.selected_asset.get()}")

    # Part 6: Simulation
    def simulate_strategies(self):
        if self.selected_asset is not None:
            asset_data = self.historical_data.get(self.selected_asset.get())
            if asset_data is not None:
                # Implement simulation logic here using historical data for the selected asset
                # Display simulation results based on risk management strategies
                self.risk_info_label.config(text="Simulation results displayed here.")
            else:
                self.risk_info_label.config(text="No historical data available for simulation.")
        else:
            self.risk_info_label.config(text="Please select an asset before simulating.")

    # Part 7: Model Selection
    def setup_select_trained_model_function(self, select_trained_model_function):
        self.select_trained_model_function = select_trained_model_function            

    def select_trained_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])
        if model_path:
            try:
                # Load the selected trained model
                self.trained_model = keras.models.load_model(model_path)
                self.risk_info_label.config(text="Trained model selected successfully.")
            except Exception as e:
                self.risk_info_label.config(text=f"Failed to load the selected model: {e}")
    
    # Method to set the trained model       
    def set_trained_model(self, trained_model):
        self.trained_model = trained_model

    # Part 8: Model Initialization
    def initialize_trained_model(self):
        if self.trained_model is not None:
            # Model is already selected, no need to re-initialize
            return
        else:
            # Initialize the model if not already done
            self.select_trained_model()
# Usage example in your main script:
# Create the Risk Management Tab and pass the trained model
#risk_management_tab_instance = RiskManagementTab(tabControl, trained_model)
#tabControl.add(risk_management_tab_instance, text='Risk Management')
            #just remove hashtags
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import logging
import sys
sys.path.append('C:/Users/Dagurlkc/OneDrive/Desktop/DaDudeKC/MLRobot/')
from modularizing_model_training_tab_project.ModelConfigManager import ModelConfigManager

# Adjusted import statements based on provided directory structure
from modularizing_model_training_tab_project.ModelConfigManager import ModelConfigManager
from modularizing_model_training_tab_project.ModelManager import ModelManager
from modularizing_model_training_tab_project.ModelTrainer import ModelTrainer
from modularizing_model_training_tab_project.UnifiedLogger import UnifiedLogger
from modularizing_model_training_tab_project.ModelManager import ModelManager
from Utilities.utils import MLRobotUtils

class ModelTrainingTab(tk.Frame):
    """
    A Tkinter Frame dedicated to configuring, running model training, and evaluating performance within a GUI.
    """
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.utils = MLRobotUtils()  # Utility class for fetching model types and options
        self.logger = UnifiedLogger()  # Advanced logging for both console and GUI
        self.training_manager = ModelManager(self.config, self.utils)
        self.setup_gui()


    def setup_gui(self):
        """ Set up GUI components within the tab. """
        self.pack(fill=tk.BOTH, expand=True)
        self.setup_title_label()
        self.setup_data_file_path_section()
        self.setup_model_type_selection()
        self.setup_start_training_button()

    def setup_title_label(self):
        """ Create a title label for the tab. """
        tk.Label(self, text="Model Training", font=("Helvetica", 16)).pack(pady=10)

    def setup_data_file_path_section(self):
        """ Create input fields for data file path and a browse button. """
        tk.Label(self, text="Data File Path:").pack()
        self.data_file_entry = tk.Entry(self)
        self.data_file_entry.pack()
        tk.Button(self, text="Browse", command=self.browse_data_file).pack(pady=5)

    def setup_model_type_selection(self):
        """ Create a dropdown menu for selecting the model type. """
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar()
        model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var,
                                           values=self.utils.get_model_types())
        model_type_dropdown.pack()
        model_type_dropdown.bind("<<ComboboxSelected>>", self.update_dynamic_options)

    def setup_start_training_button(self):
        """ Create a button to start the training process. """
        self.start_training_button = ttk.Button(self, text="Start Training", command=self.start_training)
        self.start_training_button.pack(pady=10)

    def update_dynamic_options(self, event):
        """ Update GUI based on the selected model type with dynamic options. """
        model_type = self.model_type_var.get()
        options = self.utils.get_model_options(model_type)
        self.display_dynamic_options(options)

    def display_dynamic_options(self, options):
        """ Display dynamic options for configuration based on model type. """
        if hasattr(self, 'dynamic_options_frame'):
            self.dynamic_options_frame.destroy()
        self.dynamic_options_frame = tk.Frame(self)
        self.dynamic_options_frame.pack()
        for option, info in options.items():
            tk.Label(self.dynamic_options_frame, text=info['label']).pack()
            entry = tk.Entry(self.dynamic_options_frame)
            entry.insert(0, info['default'])
            entry.pack()

    def start_training(self):
        """ Validate inputs and start model training in a new thread to keep UI responsive. """
        if not self.validate_inputs():
            return
        data_file_path = self.data_file_entry.get()
        model_type = self.model_type_var.get()
        threading.Thread(target=lambda: self.run_training(data_file_path, model_type), daemon=True).start()

    def run_training(self, data_file_path, model_type):
        """ Handles the actual training process. """
        self.training_manager.train_model(data_file_path, model_type)
        messagebox.showinfo("Training", "Model training completed successfully.")

    def browse_data_file(self):
        """ Allow users to browse and select a data file. """
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, file_path)
            self.logger.log(f"Selected data file: {file_path}")

    def validate_inputs(self):
        """ Check if the data file path and model type are correctly set. """
        if not os.path.exists(self.data_file_entry.get()):
            messagebox.showerror("Validation Error", "Data file path does not exist.")
            self.logger.log("Data file path is required.", level=logging.ERROR)
            return False
        if not self.model_type_var.get():
            messagebox.showerror("Validation Error", "Please select a model type.")
            self.logger.log("Please select a model type.", level=logging.ERROR)
            return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTrainingTab(root, config={'data_path': 'data/', 'model_path': 'models/'})
    app.pack(expand=True, fill='both')
    root.mainloop()
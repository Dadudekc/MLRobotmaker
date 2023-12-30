# data_processing_tab.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import configparser
from Data_processing.technical_indicators import TechnicalIndicators
from Utils import MLRobotUtils, update_status, auto_generate_save_path

class DataProcessingTab:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.debug_mode = False  # Initialize debug mode to False

        # Initialize MLRobotUtils with debug mode state
        self.utils = MLRobotUtils(is_debug_mode=self.debug_mode)

        # Initialize GUI components
        self.setup_gui()

    def setup_gui(self):
        # Frame for CSV File Selection
        file_frame = tk.Frame(self.parent)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(file_frame, text="Select CSV File:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_path_entry = tk.Entry(file_frame, width=50)
        self.file_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.browse_button = tk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        # Frame for Features/Indicators
        features_frame = tk.Frame(self.parent)
        features_frame.pack(padx=10, pady=(0, 5))

        tk.Label(features_frame, text="Select Features/Indicators:").pack(anchor=tk.W)
        self.features_listbox = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, height=15)
        self.features_listbox.pack(side=tk.LEFT, padx=(0, 10))

        # Buttons for Select All and Unselect All
        buttons_frame = tk.Frame(features_frame)
        buttons_frame.pack(side=tk.RIGHT)
        self.select_all_button = tk.Button(buttons_frame, text="Select All", command=self.select_all_features)
        self.select_all_button.pack()
        self.unselect_all_button = tk.Button(buttons_frame, text="Unselect All", command=self.unselect_all_features)
        self.unselect_all_button.pack()

        # Status Output
        self.status_output = tk.Text(self.parent, height=5)
        self.status_output.pack(fill='both', expand=True, padx=10, pady=5)

        # Frame for Data Processing Options
        options_frame = tk.Frame(self.parent)
        options_frame.pack(padx=10, pady=5)

        # Checkboxes for Data Processing Options
        self.normalize_var = tk.BooleanVar()
        self.normalize_checkbox = tk.Checkbutton(options_frame, text="Normalize Data", variable=self.normalize_var)
        self.normalize_checkbox.pack(side=tk.LEFT)

        self.scale_var = tk.BooleanVar()
        self.scale_checkbox = tk.Checkbutton(options_frame, text="Scale Data", variable=self.scale_var)
        self.scale_checkbox.pack(side=tk.LEFT)

        # Process Data Button
        self.process_button = tk.Button(self.parent, text="Process Data", command=self.process_data)
        self.process_button.pack(padx=10, pady=5)

        # Output Scrolled Text
        self.output_text = scrolledtext.ScrolledText(self.parent, wrap=tk.WORD, width=60, height=10)
        self.output_text.pack(fill='both', expand=True, padx=10, pady=5)

    def browse_file(self):
        # Method for opening a file dialog to select a data file
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            # Update the selected file label
            self.selected_file_label.config(text=f"Selected File: {file_path}")
            # Add code to load and process the selected file (not shown in this snippet)

    def select_all_features(self):
        # Method for selecting all features in the listbox
        self.features_listbox.select_set(0, tk.END)

    def unselect_all_features(self):
        # Method for unselecting all features in the listbox
        self.features_listbox.selection_clear(0, tk.END)

    def update_output(self, text):
        # Method for updating the output text in the GUI
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def toggle_debug_mode(self):
        # Method for toggling debug mode on/off
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.update_output("Debug mode is ON.")
        else:
            self.update_output("Debug mode is OFF")
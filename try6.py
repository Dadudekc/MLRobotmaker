import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pickle
from model_compression_utils import compress_model, prepare_model_for_deployment

class ModelDeploymentTab(tk.Frame):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.setup_ui()

    def setup_ui(self):
        # Title
        ttk.Label(self, text="Model Deployment and Optimization", font=("Helvetica", 16)).pack(pady=10)
        
        # Model Selection
        ttk.Label(self, text="Select Model:").pack()
        self.model_path_entry = ttk.Entry(self, width=50)
        self.model_path_entry.pack(side=tk.LEFT, expand=True, padx=10)
        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_model_file)
        self.browse_button.pack(side=tk.RIGHT, padx=10)

        # Model Compression Section
        ttk.Label(self, text="Model Compression Options:", font=("Helvetica", 14)).pack(pady=(20, 5))
        self.compression_frame = ttk.Frame(self)
        self.compression_frame.pack(fill=tk.X, padx=10)
        
        self.quantization_check = tk.BooleanVar()
        self.pruning_check = tk.BooleanVar()
        
        ttk.Checkbutton(self.compression_frame, text="Quantization", variable=self.quantization_check).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(self.compression_frame, text="Pruning", variable=self.pruning_check).grid(row=1, column=0, sticky=tk.W)
        
        # Compression & Preparation Button
        self.compress_button = ttk.Button(self, text="Compress & Prepare Model", command=self.compress_and_prepare_model)
        self.compress_button.pack(pady=20)

        # Log Text
        self.log_text = tk.Text(self, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def browse_model_file(self):
        file_path = filedialog.askopenfilename(title="Select a Model File",
                                               filetypes=(("Model files", "*.h5;*.pkl;*.pt"), ("All files", "*.*")))
        if file_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, file_path)

    def compress_and_prepare_model(self):
        model_path = self.model_path_entry.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file.")
            return

        # Perform model compression based on user selections
        compression_options = {
            "quantization": self.quantization_check.get(),
            "pruning": self.pruning_check.get(),
        }
        try:
            compressed_model_path = compress_model(model_path, compression_options)
            self.log(f"Model compressed successfully: {compressed_model_path}")

            # Prepare the model for deployment
            deployment_ready_path = prepare_model_for_deployment(compressed_model_path)
            self.log(f"Model is ready for deployment: {deployment_ready_path}")

            messagebox.showinfo("Success", "Model compression and preparation completed successfully.")
        except Exception as e:
            self.log(f"Error during model compression/preparation: {str(e)}")
            messagebox.showerror("Error", "An error occurred. Check the logs for details.")

    def log(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

# Assume compress_model and prepare_model_for_deployment are utility functions you'll implement or import.
# They should handle the specifics of compression (quantization, pruning) and packaging the model for deployment.

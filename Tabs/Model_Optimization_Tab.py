#Model_Optimization_Tab

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from model_training import create_neural_network, bayesian_hyperparameter_tuning

class ModelOptimizationTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Model Selection Dropdown
        tk.Label(self, text="Select Model Type:").pack()
        self.model_type_var = tk.StringVar()
        self.model_type_dropdown = ttk.Combobox(self, textvariable=self.model_type_var, values=["Random Forest", "Neural Network"])
        self.model_type_dropdown.pack()
        self.model_type_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)

        # Hyperparameter Controls Frame
        self.hyperparam_frame = ttk.Frame(self)
        self.hyperparam_frame.pack()

        # Results Visualization Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # Tuning Button
        self.tune_button = ttk.Button(self, text="Start Tuning", command=self.start_tuning)
        self.tune_button.pack()

    def on_model_select(self, event=None):
        model_type = self.model_type_var.get()
        self.build_hyperparam_controls(model_type)

    def build_hyperparam_controls(self, model_type):
        # Clear existing controls
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        if model_type == "Random Forest":
            # Add hyperparameter controls specific to Random Forest
            ttk.Label(self.hyperparam_frame, text="n_estimators").pack()
            self.n_estimators_entry = ttk.Entry(self.hyperparam_frame)
            self.n_estimators_entry.pack()

            ttk.Label(self.hyperparam_frame, text="max_depth").pack()
            self.max_depth_entry = ttk.Entry(self.hyperparam_frame)
            self.max_depth_entry.pack()

        elif model_type == "Neural Network":
            # Add hyperparameter controls specific to Neural Network
            ttk.Label(self.hyperparam_frame, text="Layers").pack()
            self.layers_entry = ttk.Entry(self.hyperparam_frame)
            self.layers_entry.pack()

            ttk.Label(self.hyperparam_frame, text="Units per Layer").pack()
            self.units_entry = ttk.Entry(self.hyperparam_frame)
            self.units_entry.pack()

    def start_tuning(self):
        model_type = self.model_type_var.get()

        # Perform hyperparameter tuning based on selected model type
        if model_type == "Random Forest":
            # Implement Random Forest tuning logic
            pass  # Placeholder
        elif model_type == "Neural Network":
            # Implement Neural Network tuning logic
            pass  # Placeholder

        # Update the plot with results
        # self.update_plot_with_results()

    def update_plot_with_results(self):
        # Placeholder function to update plot based on tuning results
        pass

    def perform_random_forest_tuning(self):
        # Extract hyperparameters
        n_estimators = int(self.n_estimators_entry.get())
        max_depth = int(self.max_depth_entry.get()) if self.max_depth_entry.get() else None

        # Define the model and parameter grid
        model = RandomForestRegressor()
        param_grid = {'n_estimators': [n_estimators], 'max_depth': [max_depth]}

        # Perform Grid Search (or use RandomizedSearchCV/BayesSearchCV)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        # grid_search.fit(X_train, y_train)  # Fit with your data

        # Save best model and update plot
        self.best_model = grid_search.best_estimator_
        self.update_plot_with_results(grid_search)

    def perform_neural_network_tuning(self):
        # Extract hyperparameters
        layers = int(self.layers_entry.get())
        units = int(self.units_entry.get())

        # Define the model and tuning strategy (using Keras Tuner)
        hypermodel = create_neural_network(input_shape=X_train.shape[1], layers=layers, units=units)
        # tuner = RandomSearch(...)  # Configure Keras Tuner
        # tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

        # Save best model and update plot
        # self.best_model = tuner.get_best_models(num_models=1)[0]
        # self.update_plot_with_results(tuner)

    def update_plot_with_results(self, tuner):
        # Example: Plot accuracy over different hyperparameter values
        # Assuming 'tuner' has the results data
        self.ax.clear()
        results_df = pd.DataFrame(tuner.cv_results_)
        results_df.plot(x='param_n_estimators', y='mean_test_score', ax=self.ax)
        self.canvas.draw()

    def save_best_model(self):
        if not self.best_model:
            messagebox.showerror("Error", "No model to save.")
            return

        # Save the best model using the save_model function from model_training.py
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib")
        if file_path:
            save_model(self.best_model, file_path)
            messagebox.showinfo("Success", "Best model saved successfully.")

# Add the save button and connect it to the save function
self.save_model_button = ttk.Button(self, text="Save Best Model", command=self.save_best_model)
self.save_model_button.pack()

# Main Application Setup
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model Optimization Tab")
    ModelOptimizationTab(root).pack(fill="both", expand=True)
    root.mainloop()

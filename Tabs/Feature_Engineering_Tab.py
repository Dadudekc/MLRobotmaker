#Feature_Engineering_Tab

import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_selection import SelectKBest, mutual_info_regression

class FeatureEngineeringTab(tk.Frame):
    def __init__(self, parent, dataframe):
        super().__init__(parent)
        self.dataframe = dataframe  # The dataset for feature engineering

        # Feature Creation Section
        tk.Label(self, text="Feature Creation").pack()
        self.column1_dropdown = ttk.Combobox(self, values=self.dataframe.columns.tolist())
        self.column1_dropdown.pack()
        self.operation_dropdown = ttk.Combobox(self, values=["+", "-", "*", "/"])
        self.operation_dropdown.pack()
        self.column2_dropdown = ttk.Combobox(self, values=self.dataframe.columns.tolist())
        self.column2_dropdown.pack()
        ttk.Button(self, text="Create Feature", command=self.create_feature).pack()

        # Feature Selection Section
        tk.Label(self, text="Feature Selection").pack()
        self.feature_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.feature_listbox.pack()
        [self.feature_listbox.insert(tk.END, col) for col in self.dataframe.columns]
        ttk.Button(self, text="Select Features", command=self.select_features).pack()

        # Feature Importance Visualization
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

    def create_feature(self):
        col1 = self.column1_dropdown.get()
        col2 = self.column2_dropdown.get()
        operation = self.operation_dropdown.get()

        if col1 and col2 and operation:
            try:
                if operation == "+":
                    self.dataframe[f"{col1}_{operation}_{col2}"] = self.dataframe[col1] + self.dataframe[col2]
                elif operation == "-":
                    self.dataframe[f"{col1}_{operation}_{col2}"] = self.dataframe[col1] - self.dataframe[col2]
                elif operation == "*":
                    self.dataframe[f"{col1}_{operation}_{col2}"] = self.dataframe[col1] * self.dataframe[col2]
                elif operation == "/":
                    self.dataframe[f"{col1}_{operation}_{col2}"] = self.dataframe[col1] / self.dataframe[col2]
                messagebox.showinfo("Success", "Feature created successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_features(self):
        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        X = self.dataframe[selected_features]
        y = self.dataframe['target_column']  # Replace 'target_column' with your actual target column name

        feature_selector = SelectKBest(mutual_info_regression, k='all').fit(X, y)
        self.update_feature_importance_plot(feature_selector.scores_)

    def update_feature_importance_plot(self, scores):
        self.ax.clear()
        self.ax.bar(range(len(scores)), scores)
        self.ax.set_title("Feature Importances")
        self.canvas.draw()

# Example Usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Feature Engineering Tab")

    # Load your dataset here
    # dataframe = pd.read_csv('your_dataset.csv')
    dataframe = pd.DataFrame(np.random.rand(100, 5), columns=[f'col{i}' for i in range(5)])  # Example dataframe
    FeatureEngineeringTab(root, dataframe).pack(fill="both", expand=True)

    root.mainloop()

#Automated_Model_Training_Tab.py

import tkinter as tk
from tkinter import ttk, messagebox
import schedule
import threading
import time

class AutomatedModelTrainingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        tk.Label(self, text="Automated Model Training").pack(pady=10)

        # Scheduling Options
        tk.Label(self, text="Train Every:").pack()
        self.schedule_dropdown = ttk.Combobox(self, values=["Daily", "Weekly", "Monthly"])
        self.schedule_dropdown.pack()
        
        # Start Training Button
        self.start_training_button = ttk.Button(self, text="Start Automated Training", command=self.start_automated_training)
        self.start_training_button.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

    def start_automated_training(self):
        interval = self.schedule_dropdown.get()
        
        # Define the training function
        def train_model():
            # Implement model training logic
            # Update status_label with progress
            pass

        # Schedule the training
        if interval == "Daily":
            schedule.every().day.do(train_model)
        elif interval == "Weekly":
            schedule.every().week.do(train_model)
        elif interval == "Monthly":
            schedule.every().month.do(train_model)
        
        # Run the schedule in a separate thread
        threading.Thread(target=self.run_schedule).start()
        messagebox.showinfo("Scheduled", f"Model training scheduled {interval.lower()}.")

    def run_schedule(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

# Example Usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automated Model Training Tab")
    AutomatedModelTrainingTab(root).pack(fill="both", expand=True)
    root.mainloop()

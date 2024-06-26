import logging
from datetime import datetime
import tkinter as tk

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnifiedLogger:
    """
    Unified Logger class to enable console, file, and GUI-based logging for various processes.
    """
    def __init__(self, log_text_widget=None, log_file='application.log'):
        """
        Initializes the logger with optional GUI logging capabilities and log file specification.

        Args:
            log_text_widget (tk.Text, optional): Text widget for displaying logs in the GUI.
            log_file (str, optional): Path to the log file for storing log messages.
        """
        self.log_text_widget = log_text_widget
        self.logger = logging.getLogger('UnifiedLogger')
        self.logger.setLevel(logging.INFO)

        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler setup
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message, level=logging.INFO):
        """
        Logs a message at the specified level across all configured handlers including GUI.

        Args:
            message (str): Message to log.
            level (int): Logging level (e.g., logging.INFO, logging.ERROR).
        """
        self.logger.log(level, message)

        # If a GUI text widget is provided, update it as well
        if self.log_text_widget:
            self.update_gui_log(message, level)

    def update_gui_log(self, message, level):
        """
        Updates the GUI log widget with the provided message and formats the display based on the log level.

        Args:
            message (str): Message to be displayed on the GUI.
            level (int): Logging level for color coding or formatting purposes.
        """
        if self.log_text_widget:
            formatted_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', formatted_message)
            self.log_text_widget.config(state='disabled')
            self.log_text_widget.see('end')

# Example usage of the UnifiedLogger
if __name__ == "__main__":
    root = tk.Tk()
    text_widget = tk.Text(root, state='disabled', height=10)
    text_widget.pack(padx=10, pady=10)
    logger = UnifiedLogger(log_text_widget=text_widget)
    
    # Simulate logging
    logger.log("This is an info message.", logging.INFO)
    logger.log("This is a warning message.", logging.WARNING)
    logger.log("This is an error message.", logging.ERROR)

    tk.Button(root, text="Close", command=root.destroy).pack()
    root.mainloop()

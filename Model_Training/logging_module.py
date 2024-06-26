import logging
from datetime import datetime
import tkinter as tk

class ModelTrainingLogger:
    def __init__(self, log_widget=None):
        self.log_widget = log_widget
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        if self.log_widget:
            widget_handler = WidgetHandler(self.log_widget)
            widget_handler.setFormatter(formatter)
            self.logger.addHandler(widget_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

class WidgetHandler(logging.Handler):
    def __init__(self, widget):
        logging.Handler.__init__(self)
        self.widget = widget

    def emit(self, record):
        log_entry = self.format(record)
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, log_entry + '\n')
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

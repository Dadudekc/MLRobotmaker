import tkinter as tk
from tkinter import ttk

class GUIUtils:
    """
    A class to encapsulate GUI utilities for creating widgets and performing GUI operations.
    """
    @staticmethod
    def create_label(parent, text, **kwargs):
        label = ttk.Label(parent, text=text, **kwargs)
        return label

    @staticmethod
    def create_entry(parent, **kwargs):
        entry = ttk.Entry(parent, **kwargs)
        return entry

    @staticmethod
    def create_button(parent, text, command, **kwargs):
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        return button

    @staticmethod
    def create_checkbox(parent, text, variable, **kwargs):
        checkbox = ttk.Checkbutton(parent, text=text, variable=variable, **kwargs)
        return checkbox

    @staticmethod
    def create_combobox(parent, values, **kwargs):
        combobox = ttk.Combobox(parent, values=values, **kwargs)
        return combobox

    @staticmethod
    def center_window(window, width=500, height=300):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        window.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    @staticmethod
    def show_error_message(parent, message):
        tk.messagebox.showerror("Error", message, parent=parent)

    @staticmethod
    def update_progressbar(progressbar, value):
        progressbar['value'] = value
        progressbar.update_idletasks()

    @staticmethod
    def animate_widget_visibility(widget, target_state):
        if target_state:
            widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            GUIUtils.fade_in(widget)
        else:
            GUIUtils.fade_out(widget)

    @staticmethod
    def fade_in(widget, duration=500):
        increment = 100 / (duration / 10)
        alpha = 0
        widget.attributes("-alpha", alpha)

        def step():
            nonlocal alpha
            if alpha < 1:
                alpha += increment / 100
                widget.attributes("-alpha", alpha)
                widget.after(10, step)

        step()

    @staticmethod
    def fade_out(widget, duration=500):
        decrement = 100 / (duration / 10)
        alpha = 1
        widget.attributes("-alpha", alpha)

        def step():
            nonlocal alpha
            if alpha > 0:
                alpha -= decrement / 100
                widget.attributes("-alpha", alpha)
                widget.after(10, step)
            else:
                widget.place_forget()

        step()

#ui_components.py

import tkinter as tk  # Assuming you're using Tkinter for your GUI
from tkinter import ttk

# Function to create tooltips for UI elements
def create_tooltip(widget, text):
    tooltip = ttk.Label(widget, text=text, background='lightyellow', padding=(5, 2))
    tooltip_id = widget.bind("<Enter>", lambda event, tooltip=tooltip: show_tooltip(event, tooltip))
    widget.bind("<Leave>", lambda event, tooltip=tooltip, id=tooltip_id: hide_tooltip(event, tooltip, id))

def show_tooltip(event, tooltip):
    tooltip.update_idletasks()
    tooltip.place(x=event.widget.winfo_rootx() + 25, y=event.widget.winfo_rooty() - 25)

def hide_tooltip(event, tooltip, id):
    tooltip.place_forget()
    event.widget.unbind("<Enter>")
    event.widget.unbind("<Leave>")

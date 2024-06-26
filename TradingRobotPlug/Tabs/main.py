# main.py

import tkinter as tk
from tkinter import ttk
from gui.frames import MainFrame
from gui.gui_utils import center_window

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Model Training Application")

    # Configure the main window
    root.geometry("800x600")  # Set the size of the main window
    center_window(root, 800, 600)  # Center the window on the screen

    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')  # or 'alt', 'default', 'classic', 'vista', etc.

    # Initialize the main frame (which includes all tabs and other components)
    main_frame = MainFrame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()

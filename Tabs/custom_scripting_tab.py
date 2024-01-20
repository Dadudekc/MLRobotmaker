#custom_scripting_tab.py

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import sys

class CustomScriptingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        tk.Label(self, text="Custom Scripting").pack(pady=10)

        # Code Editor
        self.code_editor = scrolledtext.ScrolledText(self, height=15)
        self.code_editor.pack()

        # Execute Button
        self.execute_button = ttk.Button(self, text="Execute Code", command=self.execute_code)
        self.execute_button.pack(pady=10)

        # Output Display
        self.output_display = scrolledtext.ScrolledText(self, height=15, state='disabled')
        self.output_display.pack()

    def execute_code(self):
        code = self.code_editor.get("1.0", tk.END)
        try:
            # Execute the code and capture the output
            exec(code)
        except Exception as e:
            self.output_display.config(state='normal')
            self.output_display.insert(tk.END, str(e) + '\n')
            self.output_display.config(state='disabled')

# Example Usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Custom Scripting Tab")
    CustomScriptingTab(root).pack(fill="both", expand=True)
    root.mainloop()

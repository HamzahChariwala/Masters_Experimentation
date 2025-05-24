#!/usr/bin/env python3
"""
Simple test script to verify Tkinter is working correctly.
"""

import tkinter as tk
import sys

print(f"Python version: {sys.version}")
print("Attempting to create Tkinter window...")

# Create a basic Tkinter window
root = tk.Tk()
root.title("Tkinter Test")
root.geometry("300x200")

# Add a label
label = tk.Label(root, text="If you can see this, Tkinter is working!")
label.pack(pady=50)

print("Tkinter window created, entering mainloop...")
# Run the main loop
root.mainloop()
print("Mainloop exited.") 
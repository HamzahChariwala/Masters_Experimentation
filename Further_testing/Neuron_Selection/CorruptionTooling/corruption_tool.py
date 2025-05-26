"""
Corruption Tool for selecting and modifying neural network input states.

This tool loads filtered states from an agent, displays them in a GUI,
allows selection of cells to corrupt, and saves both the original and
corrupted states to separate files.
"""

import os
import json
import random
import tkinter as tk
from tkinter import messagebox
import numpy as np
import argparse
import sys
from typing import Dict, List, Tuple, Set, Any

class StateCorruptionTool:
    def __init__(self, agent_path: str, seed: int = 42):
        """Initialize the corruption tool with the agent path and random seed."""
        self.agent_path = os.path.abspath(agent_path)
        self.filtered_states_path = os.path.join(self.agent_path, 'filtered_states.json')
        
        # Create activation_inputs directory if it doesn't exist
        self.activation_inputs_dir = os.path.join(self.agent_path, 'activation_inputs')
        os.makedirs(self.activation_inputs_dir, exist_ok=True)
        
        # Set paths for output files in the activation_inputs directory
        self.clean_inputs_path = os.path.join(self.activation_inputs_dir, 'clean_inputs.json')
        self.corrupted_inputs_path = os.path.join(self.activation_inputs_dir, 'corrupted_inputs.json')
        
        print(f"Loading states from: {self.filtered_states_path}")
        print(f"Output files will be saved to: {self.activation_inputs_dir}")
        
        # Initialize random generator with seed
        self.rng = random.Random(seed)
        
        # Load filtered states
        self.states = self._load_states()
        self.total_states = len(self.states)
        
        print(f"Loaded {self.total_states} states from filtered states file")
        
        # Keep track of seen states
        self.seen_states: Set[str] = set()
        self.current_key = None
        self.current_array = None
        
        # Initialize clean and corrupted inputs dictionaries
        self.clean_inputs = {}
        self.corrupted_inputs = {}
        
        # Initialize selected cells
        self.selected_cells = []
        
        # Track if there are unsaved changes
        self.has_unsaved_changes = False
        
    def _load_states(self) -> Dict[str, Any]:
        """Load the filtered states from the JSON file."""
        if not os.path.exists(self.filtered_states_path):
            raise FileNotFoundError(f"Filtered states file not found: {self.filtered_states_path}")
        
        with open(self.filtered_states_path, 'r') as f:
            return json.load(f)
    
    def _save_inputs(self):
        """Save the clean and corrupted inputs to JSON files."""
        # Save clean inputs
        with open(self.clean_inputs_path, 'w') as f:
            json.dump(self.clean_inputs, f, indent=2)
        
        # Save corrupted inputs
        with open(self.corrupted_inputs_path, 'w') as f:
            json.dump(self.corrupted_inputs, f, indent=2)
        
        self.has_unsaved_changes = False
        
    def get_next_state(self) -> Tuple[str, List[float]]:
        """Get the next random state that hasn't been seen yet."""
        available_keys = [k for k in self.states.keys() if k not in self.seen_states]
        
        if not available_keys:
            return None, None
        
        # Select a random key
        key = self.rng.choice(available_keys)
        self.seen_states.add(key)
        
        # Get the input array
        input_array = self.states[key]["input"]
        
        return key, input_array
    
    def run_gui(self):
        """Launch the GUI for state corruption."""
        self.root = tk.Tk()
        self.root.title("State Corruption Tool")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with state info
        self.header_frame = tk.Frame(main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.state_info_label = tk.Label(self.header_frame, text="", justify=tk.LEFT, anchor="w", font=("Arial", 10))
        self.state_info_label.pack(fill=tk.X)
        
        # Action number label
        self.action_label = tk.Label(self.header_frame, text="", justify=tk.LEFT, anchor="w", font=("Arial", 10, "bold"))
        self.action_label.pack(fill=tk.X)
        
        # Grids frame (contains both grids side by side)
        grids_frame = tk.Frame(main_frame)
        grids_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left grid frame
        self.left_grid_frame = tk.Frame(grids_frame, borderwidth=2, relief=tk.GROOVE)
        self.left_grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right grid frame
        self.right_grid_frame = tk.Frame(grids_frame, borderwidth=2, relief=tk.GROOVE)
        self.right_grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        skip_button = tk.Button(button_frame, text="Skip", command=self.skip_state)
        skip_button.pack(side=tk.LEFT, padx=(0, 5))
        
        terminate_button = tk.Button(button_frame, text="Terminate", command=self.terminate)
        terminate_button.pack(side=tk.LEFT)
        
        save_button = tk.Button(button_frame, text="✓", command=self.save_state, bg="green", fg="white")
        save_button.pack(side=tk.RIGHT)
        
        # Progress indicator
        self.progress_label = tk.Label(main_frame, text="", anchor="e")
        self.progress_label.pack(fill=tk.X, pady=(5, 0))
        
        # Load the first state
        self.load_next_state()
        
        # Start the GUI main loop
        self.root.protocol("WM_DELETE_WINDOW", self.terminate)
        self.root.mainloop()
    
    def load_next_state(self):
        """Load the next state into the GUI."""
        # Clear selected cells
        self.selected_cells = []
        
        # Get next state
        key, input_array = self.get_next_state()
        
        if key is None:
            messagebox.showinfo("Completed", "All states have been processed.")
            self.terminate()
            return
        
        self.current_key = key
        self.current_array = input_array
        
        # Update header with state key and first 8 values
        header_text = f"Key: {key}\nFirst 8 values: {input_array[:8]}"
        self.state_info_label.config(text=header_text)
        
        # Get action and additional state info if available
        state_data = self.states[key]
        action_info = ""
        
        if "action_taken" in state_data:
            action_info += f"Action: {state_data['action_taken']} | "
        
        if "risky_diagonal" in state_data:
            action_info += f"Risky Diagonal: {'Yes' if state_data['risky_diagonal'] else 'No'} | "
        
        if "next_cell_is_lava" in state_data:
            action_info += f"Next Cell is Lava: {'Yes' if state_data['next_cell_is_lava'] else 'No'}"
        
        self.action_label.config(text=action_info)
        
        # Split the remaining array into two square grids
        remaining_values = input_array[8:]
        grid_size = int(np.sqrt(len(remaining_values) // 2))
        
        # Split into two halves
        first_half = remaining_values[:grid_size*grid_size]
        second_half = remaining_values[grid_size*grid_size:]
        
        # Reshape into square grids
        grid1 = np.array(first_half).reshape(grid_size, grid_size)
        grid2 = np.array(second_half).reshape(grid_size, grid_size)
        
        # Clear existing grid frames
        for widget in self.left_grid_frame.winfo_children():
            widget.destroy()
        for widget in self.right_grid_frame.winfo_children():
            widget.destroy()
        
        # Draw the grids
        self._draw_grid(self.left_grid_frame, grid1, 0)
        self._draw_grid(self.right_grid_frame, grid2, 1)
        
        # Update progress
        progress = f"States: {len(self.seen_states)}/{self.total_states} | Saved: {len(self.clean_inputs)}"
        self.progress_label.config(text=progress)
    
    def _draw_grid(self, frame, grid_data, grid_index):
        """Draw a grid in the specified frame."""
        grid_size = len(grid_data)
        
        # Calculate cell size based on frame size
        cell_size = min(30, 400 // grid_size)
        
        # Create a canvas for the grid
        canvas_width = cell_size * grid_size
        canvas_height = cell_size * grid_size
        
        canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="white")
        canvas.pack(expand=True)
        
        # Draw grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                # Determine cell color
                value = grid_data[i][j]
                color = "black" if value == 1 else "white"
                
                # Check if cell is selected
                cell_id = (grid_index, i, j)
                if cell_id in self.selected_cells:
                    color = "grey"
                
                # Draw the cell
                rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                
                # Bind click event
                canvas.tag_bind(rect_id, "<Button-1>", 
                               lambda event, g=grid_index, r=i, c=j, rect=rect_id, canv=canvas: 
                               self._on_cell_click(g, r, c, rect, canv))
    
    def _on_cell_click(self, grid_index, row, col, rect_id, canvas):
        """Handle cell click event."""
        cell_id = (grid_index, row, col)
        
        # Toggle selection
        if cell_id in self.selected_cells:
            self.selected_cells.remove(cell_id)
            
            # Restore original color
            grid_size = int(np.sqrt((len(self.current_array) - 8) // 2))
            if grid_index == 0:
                idx = row * grid_size + col
                value = self.current_array[8 + idx]
            else:
                idx = row * grid_size + col
                value = self.current_array[8 + grid_size*grid_size + idx]
            
            color = "black" if value == 1 else "white"
            canvas.itemconfig(rect_id, fill=color)
        else:
            self.selected_cells.append(cell_id)
            canvas.itemconfig(rect_id, fill="grey")
    
    def save_state(self):
        """Save the current state as clean and create corrupted version."""
        if self.current_key is None:
            return
        
        # Add the clean input to the clean inputs dictionary
        self.clean_inputs[self.current_key] = {"input": self.current_array}
        
        # Create corrupted array (copy of original)
        corrupted_array = self.current_array.copy()
        
        # Apply corruptions
        for grid_index, row, col in self.selected_cells:
            grid_size = int(np.sqrt((len(self.current_array) - 8) // 2))
            
            if grid_index == 0:
                idx = 8 + row * grid_size + col
            else:
                idx = 8 + grid_size*grid_size + row * grid_size + col
            
            # Flip the value (0 -> 1, 1 -> 0)
            corrupted_array[idx] = 1 - corrupted_array[idx]
        
        # Add the corrupted input to the corrupted inputs dictionary
        self.corrupted_inputs[self.current_key] = {"input": corrupted_array}
        
        # Mark that we have changes to save
        self.has_unsaved_changes = True
        
        # Save to disk immediately
        self._save_inputs()
        
        # Load next state
        self.load_next_state()
    
    def skip_state(self):
        """Skip the current state and load the next one."""
        self.load_next_state()
    
    def terminate(self):
        """Save any unsaved inputs and close the GUI."""
        if self.has_unsaved_changes:
            self._save_inputs()
        
        # Show summary
        processed = len(self.seen_states)
        saved = len(self.clean_inputs)
        messagebox.showinfo("Summary", 
                            f"Session complete!\n"
                            f"States processed: {processed}/{self.total_states}\n"
                            f"States saved: {saved}\n"
                            f"Files saved to:\n"
                            f"- {self.clean_inputs_path}\n"
                            f"- {self.corrupted_inputs_path}")
        
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description='State Corruption Tool')
    parser.add_argument('--path', required=True, help='Path to the agent directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # Create and run the tool
        tool = StateCorruptionTool(args.path, args.seed)
        tool.run_gui()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
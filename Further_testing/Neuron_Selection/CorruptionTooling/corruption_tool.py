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
import re
from typing import Dict, List, Tuple, Set, Any, Optional
import math

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
        
        # Track the agent position, orientation, and goal position
        self.agent_pos = None
        self.agent_dir = None
        self.goal_pos = [9, 9]  # Default goal position at [9,9]
        self.new_goal_pos = None  # Position of the user-selected goal
        self.env_size = 11  # Default environment size, will be updated based on env ID
        
        # Enable debugging mode
        self.debug = True
        
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
    
    def _parse_state_key(self, key: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Parse the state key to extract agent position and orientation.
        Format example: "MiniGridS11N5-v0-81107-4,1,2-0655"
        - "MiniGridS11N5-v0" is the env ID (11 is env size)
        - "-81107" is the env seed (not relevant)
        - "-4,1,2" contains agent position and orientation (x=4, y=1, theta=2)
        - "-0655" is a unique identifier
        """
        # First try to extract environment size from the env ID
        env_size_match = re.search(r'MiniGrid[^0-9]*(\d+)', key)
        if env_size_match:
            self.env_size = int(env_size_match.group(1))
            if self.debug:
                print(f"Extracted environment size: {self.env_size} from key")
        
        # Extract agent position and orientation
        # Look for the pattern "-X,Y,Z-" where X,Y,Z are digits
        agent_pos_match = re.search(r'-(\d+),(\d+),(\d+)-', key)
        if agent_pos_match:
            x = int(agent_pos_match.group(1))
            y = int(agent_pos_match.group(2))
            theta = int(agent_pos_match.group(3))
            return x, y, theta
        
        # Fallback pattern matching
        if "-" in key:
            parts = key.split("-")
            for part in parts:
                if re.match(r'^\d+,\d+,\d+$', part):
                    coords = part.split(",")
                    if len(coords) == 3:
                        return int(coords[0]), int(coords[1]), int(coords[2])
        
        print(f"Warning: Could not parse position from key: {key}")
        return None, None, None
    
    def _calculate_goal_values(self, agent_pos, agent_dir, goal_pos):
        """Calculate the goal-related values (first 8 entries) based on agent and goal positions."""
        if agent_pos is None or agent_dir is None or goal_pos is None:
            print("Warning: Missing position data for goal value calculation")
            # Return default values if we don't have all necessary information
            return [0.0] * 8
        
        # Debug info
        print(f"Agent pos: {agent_pos}, dir: {agent_dir}, goal pos: {goal_pos}")
        
        # Convert to numpy arrays and ensure they're floats
        agent_pos = np.array(agent_pos, dtype=float)
        goal_pos = np.array(goal_pos, dtype=float)
        
        # Vector from agent to goal
        vec_to_goal = goal_pos - agent_pos
        print(f"Vector to goal: {vec_to_goal}")
        
        # Agent's facing direction vector
        dir_vectors = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])   # Up
        }
        # Handle out-of-range directions by defaulting to 0 (right)
        if agent_dir not in dir_vectors:
            print(f"Warning: Unknown agent direction {agent_dir}, defaulting to 0 (right)")
            agent_dir = 0
            
        facing_vec = dir_vectors[agent_dir]
        print(f"Facing vector: {facing_vec}")
        
        # Normalize vectors
        norm_vec_to_goal = np.linalg.norm(vec_to_goal)
        norm_facing_vec = np.linalg.norm(facing_vec)
        
        # Initialize default values
        angle = 0.0
        rotation = np.array([0, 0], dtype=np.uint8)
        direction_vector = np.array([0.0, 0.0], dtype=np.float32)
        four_way_goal_direction = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        four_way_angle_alignment = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        if norm_vec_to_goal > 0 and norm_facing_vec > 0:
            # Compute angle between facing vector and vector to goal
            dot_product = np.dot(facing_vec, vec_to_goal)
            angle = np.arccos(np.clip(dot_product / (norm_vec_to_goal * norm_facing_vec), -1.0, 1.0))
            print(f"Angle: {angle} radians, {angle * 180 / np.pi} degrees")
            
            # Determine rotation direction using cross product
            cross = facing_vec[0] * vec_to_goal[1] - facing_vec[1] * vec_to_goal[0]
            print(f"Cross product: {cross}")
            
            if cross > 0:
                rotation = np.array([1, 0], dtype=np.uint8)  # Left
                print("Rotation: Left")
            elif cross < 0:
                rotation = np.array([0, 1], dtype=np.uint8)  # Right
                print("Rotation: Right")
            else:
                # When cross product is zero, angle is either 0 or π
                if dot_product > 0:
                    angle = 0.0
                    rotation = np.array([0, 0], dtype=np.uint8)  # Aligned
                    print("Rotation: Aligned")
                else:
                    angle = np.pi
                    rotation = np.array([1, 1], dtype=np.uint8)  # Default to Left
                    print("Rotation: Reversed")
            
            # Normalize direction vector
            direction_vector = vec_to_goal / norm_vec_to_goal
            print(f"Normalized direction vector: {direction_vector}")
            
            # Create four-way direction representation (non-negative values only)
            # Order: [right, left, down, up]
            right = max(0, direction_vector[0])  # Positive x = right
            left = max(0, -direction_vector[0])  # Negative x = left
            down = max(0, direction_vector[1])   # Positive y = down
            up = max(0, -direction_vector[1])    # Negative y = up
            
            four_way_goal_direction = np.array([right, left, down, up], dtype=np.float32)
            print(f"Four-way goal direction: {four_way_goal_direction}")
            
            # Create four-way angle alignment representation based on angle and cross product
            # This is a simplified version of the alignment calculation in GoalAngleDistanceWrapper
            if angle <= np.pi / 6:  # Within ±30 degrees
                aligned_value = 1.0 - (angle / (np.pi / 6))
                turn_value = angle / (np.pi / 6)
                
                if cross > 0:  # Left turn needed
                    four_way_angle_alignment = np.array([aligned_value, turn_value, 0.0, 0.0], dtype=np.float32)
                elif cross < 0:  # Right turn needed
                    four_way_angle_alignment = np.array([aligned_value, 0.0, turn_value, 0.0], dtype=np.float32)
                else:  # Perfect alignment
                    four_way_angle_alignment = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # Simplified for other angle ranges
                if cross > 0:  # Left turn needed
                    four_way_angle_alignment = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                elif cross < 0:  # Right turn needed
                    four_way_angle_alignment = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                else:  # Reversed
                    four_way_angle_alignment = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            print(f"Four-way angle alignment: {four_way_angle_alignment}")
        else:
            print(f"Warning: Zero vector encountered. norm_vec_to_goal={norm_vec_to_goal}, norm_facing_vec={norm_facing_vec}")
        
        # Combine the values into the first 8 entries
        # First 2: direction vector (normalized)
        # Next 2: rotation (one-hot)
        # Next 4: four-way angle alignment
        goal_values = [
            float(direction_vector[0]),
            float(direction_vector[1]),
            float(rotation[0]),
            float(rotation[1]),
            float(four_way_angle_alignment[0]),
            float(four_way_angle_alignment[1]),
            float(four_way_angle_alignment[2]),
            float(four_way_angle_alignment[3])
        ]
        
        print(f"Final goal values: {goal_values}")
        return goal_values
        
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
        self.root.geometry("1000x700")
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with state info
        self.header_frame = tk.Frame(main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.state_info_label = tk.Label(self.header_frame, text="", justify=tk.LEFT, anchor="w", font=("Arial", 10))
        self.state_info_label.pack(fill=tk.X)
        
        # Values display
        self.values_frame = tk.Frame(main_frame)
        self.values_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Original values
        original_frame = tk.Frame(self.values_frame)
        original_frame.pack(fill=tk.X)
        
        tk.Label(original_frame, text="Original Values:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.original_values_label = tk.Label(original_frame, text="", font=("Arial", 10))
        self.original_values_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Recalculated values (when a new goal is selected)
        recalculated_frame = tk.Frame(self.values_frame)
        recalculated_frame.pack(fill=tk.X)
        
        tk.Label(recalculated_frame, text="Recalculated Values:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.recalculated_values_label = tk.Label(recalculated_frame, text="", font=("Arial", 10))
        self.recalculated_values_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Action number label
        self.action_label = tk.Label(self.header_frame, text="", justify=tk.LEFT, anchor="w", font=("Arial", 10, "bold"))
        self.action_label.pack(fill=tk.X)
        
        # Grids frame (contains all grids side by side)
        grids_frame = tk.Frame(main_frame)
        grids_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left grid frame
        self.left_grid_frame = tk.Frame(grids_frame, borderwidth=2, relief=tk.GROOVE)
        self.left_grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Center grid frame
        self.center_grid_frame = tk.Frame(grids_frame, borderwidth=2, relief=tk.GROOVE)
        self.center_grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5))
        
        # Right grid frame (environment view)
        self.right_grid_frame = tk.Frame(grids_frame, borderwidth=2, relief=tk.GROOVE)
        self.right_grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Labels for each grid
        tk.Label(self.left_grid_frame, text="Grid 1 (Select to flip bits)", font=("Arial", 9)).pack(anchor="n")
        tk.Label(self.center_grid_frame, text="Grid 2 (Select to flip bits)", font=("Arial", 9)).pack(anchor="n")
        tk.Label(self.right_grid_frame, text="Environment (Select new goal position)", font=("Arial", 9)).pack(anchor="n")
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        skip_button = tk.Button(button_frame, text="Skip", command=self.skip_state)
        skip_button.pack(side=tk.LEFT, padx=(0, 5))
        
        terminate_button = tk.Button(button_frame, text="Terminate", command=self.terminate)
        terminate_button.pack(side=tk.LEFT)
        
        # Reset goal button
        reset_goal_button = tk.Button(button_frame, text="Reset Goal", command=self.reset_goal)
        reset_goal_button.pack(side=tk.LEFT, padx=(5, 0))
        
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
    
    def reset_goal(self):
        """Reset the goal position to the default (9,9)."""
        self.new_goal_pos = None
        self.load_current_state()
    
    def load_next_state(self):
        """Load the next state into the GUI."""
        # Clear selected cells
        self.selected_cells = []
        self.new_goal_pos = None
        
        # Get next state
        key, input_array = self.get_next_state()
        
        if key is None:
            messagebox.showinfo("Completed", "All states have been processed.")
            self.terminate()
            return
        
        self.current_key = key
        self.current_array = input_array
        
        # Parse agent position and orientation from the key
        x, y, theta = self._parse_state_key(key)
        if x is not None and y is not None and theta is not None:
            self.agent_pos = [x, y]
            self.agent_dir = theta
            if self.debug:
                print(f"Parsed agent position: {self.agent_pos}, direction: {self.agent_dir} from key: {key}")
        else:
            print(f"Could not parse agent position and direction from key: {key}")
            # Set default position if parsing fails
            self.agent_pos = [5, 5]  # Center of grid
            self.agent_dir = 0       # Facing right
        
        self.load_current_state()
    
    def load_current_state(self):
        """Refresh the current state display with updated information."""
        if self.current_key is None or self.current_array is None:
            return
        
        # Original goal values (first 8 entries)
        original_goal_values = self.current_array[:8]
        
        # Recalculated goal values if a new goal is selected
        recalculated_goal_values = None
        if self.new_goal_pos is not None:
            recalculated_goal_values = self._calculate_goal_values(
                self.agent_pos, self.agent_dir, self.new_goal_pos
            )
        
        # Update header with state key and parsed information
        agent_info = ""
        if self.agent_pos is not None and self.agent_dir is not None:
            dir_names = {0: "Right", 1: "Down", 2: "Left", 3: "Up"}
            agent_info = f" | Agent: ({self.agent_pos[0]}, {self.agent_pos[1]}), Facing: {dir_names.get(self.agent_dir, str(self.agent_dir))}"
        
        header_text = f"Key: {self.current_key}{agent_info}"
        self.state_info_label.config(text=header_text)
        
        # Update value displays
        original_text = f"[{', '.join(f'{v:.4f}' for v in original_goal_values)}]"
        self.original_values_label.config(text=original_text)
        
        if recalculated_goal_values:
            recalc_text = f"[{', '.join(f'{v:.4f}' for v in recalculated_goal_values)}]"
            self.recalculated_values_label.config(text=recalc_text)
        else:
            self.recalculated_values_label.config(text="(No new goal selected)")
        
        # Get action and additional state info if available
        state_data = self.states[self.current_key]
        action_info = ""
        
        if "action_taken" in state_data:
            action_info += f"Action: {state_data['action_taken']} | "
        
        if "risky_diagonal" in state_data:
            action_info += f"Risky Diagonal: {'Yes' if state_data['risky_diagonal'] else 'No'} | "
        
        if "next_cell_is_lava" in state_data:
            action_info += f"Next Cell is Lava: {'Yes' if state_data['next_cell_is_lava'] else 'No'}"
        
        self.action_label.config(text=action_info)
        
        # Split the array into two square grids (skipping the first 8 values)
        remaining_values = self.current_array[8:]
        grid_size = int(np.sqrt(len(remaining_values) // 2))
        
        # Split into two halves
        first_half = remaining_values[:grid_size*grid_size]
        second_half = remaining_values[grid_size*grid_size:]
        
        # Reshape into square grids
        grid1 = np.array(first_half).reshape(grid_size, grid_size)
        grid2 = np.array(second_half).reshape(grid_size, grid_size)
        
        # Clear existing grid frames
        for widget in self.left_grid_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        for widget in self.center_grid_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        for widget in self.right_grid_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        
        # Draw the grids
        self._draw_grid(self.left_grid_frame, grid1, 0)
        self._draw_grid(self.center_grid_frame, grid2, 1)
        self._draw_environment_grid(self.right_grid_frame)
        
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
    
    def _draw_environment_grid(self, frame):
        """Draw the environment grid showing agent and goal positions."""
        env_size = self.env_size
        
        # Calculate cell size based on frame size
        cell_size = min(30, 400 // env_size)
        
        # Create a canvas for the grid
        canvas_width = cell_size * env_size
        canvas_height = cell_size * env_size
        
        canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="white")
        canvas.pack(expand=True)
        
        # Draw grid cells
        for i in range(env_size):
            for j in range(env_size):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                # Default cell color
                color = "white"
                outline = "gray"
                
                # Draw the cell
                rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=outline)
                
                # Add coordinate labels in small font to help with debugging
                canvas.create_text(x1 + cell_size/2, y1 + cell_size/2, text=f"{j},{i}", 
                                  font=("Arial", max(5, int(cell_size/5))), fill="lightgray")
                
                # Bind click event for selecting a new goal position
                canvas.tag_bind(rect_id, "<Button-1>", 
                               lambda event, r=i, c=j: 
                               self._on_env_cell_click(r, c))
        
        # Mark the default goal position (9,9) - now as a black cell
        goal_x, goal_y = self.goal_pos
        if 0 <= goal_x < env_size and 0 <= goal_y < env_size:
            x1, y1 = goal_x * cell_size, goal_y * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
        
        # Mark the new goal position if selected - now as a grey cell
        if self.new_goal_pos:
            new_goal_x, new_goal_y = self.new_goal_pos
            if 0 <= new_goal_x < env_size and 0 <= new_goal_y < env_size:
                x1, y1 = new_goal_x * cell_size, new_goal_y * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill="grey", outline="darkgrey")
                
                if self.debug:
                    print(f"New goal position: {self.new_goal_pos}")
        
        # Mark the agent position and orientation if available - now just an arrow with no background
        if self.agent_pos is not None and self.agent_dir is not None:
            agent_x, agent_y = self.agent_pos
            
            if self.debug:
                print(f"Drawing agent at position: {agent_x}, {agent_y} with direction: {self.agent_dir}")
            
            # Only draw if within environment bounds
            if 0 <= agent_x < env_size and 0 <= agent_y < env_size:
                x1, y1 = agent_x * cell_size, agent_y * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                # Calculate center and endpoints for the arrow
                center_x, center_y = x1 + cell_size/2, y1 + cell_size/2
                arrow_length = cell_size * 0.7  # Make arrow longer (70% of cell size)
                
                # Calculate endpoint based on direction
                if self.agent_dir == 0:  # Right
                    end_x, end_y = center_x + arrow_length/2, center_y
                    start_x, start_y = center_x - arrow_length/2, center_y
                elif self.agent_dir == 1:  # Down
                    end_x, end_y = center_x, center_y + arrow_length/2
                    start_x, start_y = center_x, center_y - arrow_length/2
                elif self.agent_dir == 2:  # Left
                    end_x, end_y = center_x - arrow_length/2, center_y
                    start_x, start_y = center_x + arrow_length/2, center_y
                elif self.agent_dir == 3:  # Up
                    end_x, end_y = center_x, center_y - arrow_length/2
                    start_x, start_y = center_x, center_y + arrow_length/2
                
                # Draw the arrow (from start to end)
                canvas.create_line(
                    start_x, start_y, end_x, end_y, 
                    fill="black", 
                    width=max(2, int(cell_size/7)), 
                    arrow=tk.LAST,
                    arrowshape=(cell_size/4, cell_size/3, cell_size/6)
                )
            else:
                print(f"Warning: Agent position {agent_x}, {agent_y} is outside the environment bounds {env_size}x{env_size}")
    
    def _on_cell_click(self, grid_index, row, col, rect_id, canvas):
        """Handle cell click event for the input grids."""
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
    
    def _on_env_cell_click(self, row, col):
        """Handle cell click event for the environment grid."""
        # Set the new goal position
        self.new_goal_pos = [col, row]  # Note: grid coordinates are (col, row)
        print(f"New goal position selected: {self.new_goal_pos}")
        
        # Refresh the display
        self.load_current_state()
    
    def save_state(self):
        """Save the current state as clean and create corrupted version."""
        if self.current_key is None:
            return
        
        # Create a copy of the original array
        clean_array = self.current_array.copy()
        
        # Create corrupted array (copy of original)
        corrupted_array = self.current_array.copy()
        
        # Apply bit flips
        for grid_index, row, col in self.selected_cells:
            grid_size = int(np.sqrt((len(self.current_array) - 8) // 2))
            
            if grid_index == 0:
                idx = 8 + row * grid_size + col
            else:
                idx = 8 + grid_size*grid_size + row * grid_size + col
            
            # Flip the value (0 -> 1, 1 -> 0)
            corrupted_array[idx] = 1 - corrupted_array[idx]
        
        # Apply goal position changes if a new goal was selected
        if self.new_goal_pos is not None:
            # Recalculate the first 8 values
            new_goal_values = self._calculate_goal_values(
                self.agent_pos, self.agent_dir, self.new_goal_pos
            )
            
            # Replace the first 8 values in the corrupted array
            corrupted_array[:8] = new_goal_values
        
        # Add the clean input to the clean inputs dictionary
        self.clean_inputs[self.current_key] = {"input": clean_array}
        
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
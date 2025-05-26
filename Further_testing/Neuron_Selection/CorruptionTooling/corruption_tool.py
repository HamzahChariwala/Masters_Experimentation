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
import torch
import importlib.util
import traceback

# Absolute path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# First ensure the project root is in the path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

# Import the feature extractor directly using importlib.util
feature_extractor_path = os.path.join(project_root, "Environment_Tooling/BespokeEdits/FeatureExtractor.py")
if os.path.exists(feature_extractor_path):
    print(f"Found feature extractor at: {feature_extractor_path}")
    spec = importlib.util.spec_from_file_location("FeatureExtractor", feature_extractor_path)
    feature_extractor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_extractor_module)
    
    # Check if the module has the CustomCombinedExtractor class
    if hasattr(feature_extractor_module, "CustomCombinedExtractor"):
        CustomCombinedExtractor = feature_extractor_module.CustomCombinedExtractor
        print("Successfully imported CustomCombinedExtractor")
        
        # Add the module to sys.modules to ensure it's available for model loading
        sys.modules["Environment_Tooling.BespokeEdits.FeatureExtractor"] = feature_extractor_module
        FEATURE_EXTRACTOR_IMPORTED = True
    else:
        print("Error: CustomCombinedExtractor class not found in module")
        FEATURE_EXTRACTOR_IMPORTED = False
else:
    print(f"Error: Feature extractor file not found at {feature_extractor_path}")
    FEATURE_EXTRACTOR_IMPORTED = False

# Only import stable_baselines3 after setting up the environment
from stable_baselines3 import DQN  # Changed from PPO to DQN

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
        
        # Variables for PyTorch hooks
        self.hook_handle = None
        self.captured_q_values = None
        
        # Load the model for inference
        try:
            # Look for the model file within the directory
            model_path = self.agent_path
            # Check if there's a .zip file in the directory that could be the model
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    if file.endswith('.zip'):
                        model_path = os.path.join(model_path, file)
                        break
            
            # Define custom objects for loading
            custom_objects = {}
            if 'FEATURE_EXTRACTOR_IMPORTED' in globals() and FEATURE_EXTRACTOR_IMPORTED:
                custom_objects["features_extractor_class"] = CustomCombinedExtractor
            
            print(f"Loading model from {model_path}...")
            self.model = DQN.load(model_path, custom_objects=custom_objects)
            print("Model loaded successfully")
            print(f"Model policy type: {type(self.model.policy)}")
            
            # Initialize logits
            self.original_logits = None
            self.corrupted_logits = None
            
        except Exception as e:
            print(f"Warning: Could not load model for inference: {e}")
            print("Using simulated logits for model predictions")
            self.model = None
            self.original_logits = None
            self.corrupted_logits = None
        
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
        self.root.geometry("1000x800")  # Increased height for logits section
        
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
        
        # Logits display frame - just a regular frame without background color
        logits_outer_frame = tk.Frame(main_frame)
        logits_outer_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Tables container
        tables_frame = tk.Frame(logits_outer_frame)
        tables_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Dark gray color for headers
        header_color = "#333333"
        
        # Original Q-values column (LEFT)
        orig_frame = tk.Frame(tables_frame)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Header box with color
        orig_header_frame = tk.Frame(orig_frame, bg=header_color, relief=tk.RAISED, borderwidth=1)
        orig_header_frame.pack(fill=tk.X)
        tk.Label(orig_header_frame, text="Original Q-values", font=("Arial", 10, "bold"), 
               bg=header_color, fg="white", anchor="center").pack(fill=tk.X, ipady=2)
        
        # Content box
        self.original_text = tk.Text(orig_frame, height=8, width=20, font=("Courier New", 12, "bold"), 
                                  wrap=tk.WORD, bg="white", fg="black", borderwidth=1, relief=tk.SOLID)
        self.original_text.pack(fill=tk.BOTH)
        self.original_text.insert(tk.END, "Initializing...\n")
        
        # Changes column (MIDDLE)
        diff_frame = tk.Frame(tables_frame)
        diff_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Header box with color
        diff_header_frame = tk.Frame(diff_frame, bg=header_color, relief=tk.RAISED, borderwidth=1)
        diff_header_frame.pack(fill=tk.X)
        tk.Label(diff_header_frame, text="Changes", font=("Arial", 10, "bold"), 
               bg=header_color, fg="white", anchor="center").pack(fill=tk.X, ipady=2)
        
        # Content box
        self.diff_text = tk.Text(diff_frame, height=8, width=20, font=("Courier New", 12, "bold"), 
                              wrap=tk.WORD, bg="white", fg="black", borderwidth=1, relief=tk.SOLID)
        self.diff_text.pack(fill=tk.BOTH)
        self.diff_text.insert(tk.END, "Initializing...\n")
        
        # Corrupted Q-values column (RIGHT)
        corrupt_frame = tk.Frame(tables_frame)
        corrupt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Header box with color
        corrupt_header_frame = tk.Frame(corrupt_frame, bg=header_color, relief=tk.RAISED, borderwidth=1)
        corrupt_header_frame.pack(fill=tk.X)
        tk.Label(corrupt_header_frame, text="Corrupted Q-values", font=("Arial", 10, "bold"), 
               bg=header_color, fg="white", anchor="center").pack(fill=tk.X, ipady=2)
        
        # Content box
        self.corrupted_text = tk.Text(corrupt_frame, height=8, width=20, font=("Courier New", 12, "bold"), 
                                   wrap=tk.WORD, bg="white", fg="black", borderwidth=1, relief=tk.SOLID)
        self.corrupted_text.pack(fill=tk.BOTH)
        self.corrupted_text.insert(tk.END, "Initializing...\n")
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left-side buttons
        left_buttons = tk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        skip_button = tk.Button(left_buttons, text="Skip", command=self.skip_state,
                              bg="#f1f1f1", fg="black", relief=tk.GROOVE)
        skip_button.pack(side=tk.LEFT, padx=(0, 5))
        
        terminate_button = tk.Button(left_buttons, text="Terminate", command=self.terminate,
                                   bg="#f1f1f1", fg="black", relief=tk.GROOVE)
        terminate_button.pack(side=tk.LEFT)
        
        # Reset goal button
        reset_goal_button = tk.Button(left_buttons, text="Reset Goal", command=self.reset_goal,
                                    bg="#f1f1f1", fg="black", relief=tk.GROOVE)
        reset_goal_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Update logits button
        update_logits_button = tk.Button(left_buttons, text="Update Predictions", command=self._update_logits_display,
                                       bg="#f1f1f1", fg="black", relief=tk.GROOVE)
        update_logits_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Save button with improved styling and contrast
        save_button = tk.Button(button_frame, text="✓", command=self.save_state, 
                              bg="#cc4778", fg="black", font=("Arial", 12, "bold"),
                              width=3, height=1)
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
        
        # Update logits after resetting goal
        self._update_logits_display()
    
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
        
        # Compute and display logits for the initial state
        self._update_logits_display()
    
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
        
        # Define specific colors for each grid
        if grid_index == 0:
            # First grid (barrier mask): dark purple for all 1s
            value_color = "#3b0f70"
        else:
            # Second grid (lava mask): purple for all 1s
            value_color = "#8c2981"
        
        # Draw grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                # Determine cell color
                value = grid_data[i][j]
                
                # Convert value to color (use assigned color for 1s, white for 0s)
                if value == 1:
                    color = value_color
                else:
                    color = "white"
                
                # Check if cell is selected
                cell_id = (grid_index, i, j)
                if cell_id in self.selected_cells:
                    color = "grey"  # Grey for selected cells
                
                # Draw the cell with gray outline
                rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                
                # Bind click event
                canvas.tag_bind(rect_id, "<Button-1>", 
                               lambda event, g=grid_index, r=i, c=j, rect=rect_id, canv=canvas: 
                               self._on_cell_click(g, r, c, rect, canv))
    
    def _draw_environment_grid(self, frame):
        """Draw the environment grid showing agent and goal positions."""
        env_size = self.env_size
        
        # Calculate cell size based on frame size (match other grids)
        grid_size = int(np.sqrt((len(self.current_array) - 8) // 2))
        cell_size = min(30, 400 // grid_size)
        
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
                
                # Create a 1-cell border with orange-red
                if i == 0 or i == env_size-1 or j == 0 or j == env_size-1:
                    color = "#dd4a68"  # Orange-red for border
                
                # Draw the cell
                rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                
                # Bind click event for selecting a new goal position
                canvas.tag_bind(rect_id, "<Button-1>", 
                               lambda event, r=i, c=j: 
                               self._on_env_cell_click(r, c))
        
        # Mark the default goal position with pale orange
        goal_x, goal_y = self.goal_pos
        if 0 <= goal_x < env_size and 0 <= goal_y < env_size:
            x1, y1 = goal_x * cell_size, goal_y * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, fill="#fe9f6d", outline="gray")  # Pale orange
        
        # Mark the new goal position if selected with a darker color
        if self.new_goal_pos:
            new_goal_x, new_goal_y = self.new_goal_pos
            if 0 <= new_goal_x < env_size and 0 <= new_goal_y < env_size:
                x1, y1 = new_goal_x * cell_size, new_goal_y * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill="#ff7f50", outline="gray")  # Brighter orange
                
                if self.debug:
                    print(f"New goal position: {self.new_goal_pos}")
        
        # Mark the agent position and orientation
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
                
                # Draw the arrow (black)
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
                # First grid (barrier mask): dark purple for 1s
                color = "#3b0f70" if value == 1 else "white"
            else:
                idx = row * grid_size + col
                value = self.current_array[8 + grid_size*grid_size + idx]
                # Second grid (lava mask): purple for 1s
                color = "#8c2981" if value == 1 else "white"
                
            canvas.itemconfig(rect_id, fill=color)
        else:
            self.selected_cells.append(cell_id)
            canvas.itemconfig(rect_id, fill="grey")
        
        # Update logits automatically when cells are clicked
        self._update_logits_display()
    
    def _on_env_cell_click(self, row, col):
        """Handle cell click event for the environment grid."""
        # Set the new goal position
        self.new_goal_pos = [col, row]  # Note: grid coordinates are (col, row)
        print(f"New goal position selected: {self.new_goal_pos}")
        
        # Refresh the display
        self.load_current_state()
        
        # Update logits automatically when new goal is selected
        self._update_logits_display()
    
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

    def _inspect_model_policy(self):
        """Inspect and print the structure of the model policy."""
        if self.model is None:
            print("No model available to inspect")
            return
        
        try:
            print("\n=== Model Policy Inspection ===")
            print(f"Policy type: {type(self.model.policy)}")
            
            # Check observation space
            if hasattr(self.model, 'observation_space'):
                print(f"Observation space: {self.model.observation_space}")
            
            # Check if this is a MultiInputPolicy
            if hasattr(self.model.policy, 'features_extractor'):
                fe = self.model.policy.features_extractor
                print(f"Features extractor: {type(fe)}")
                if hasattr(fe, 'mlp'):
                    print(f"  Has MLP: {fe.mlp}")
                if hasattr(fe, '_features_dim'):
                    print(f"  Features dim: {fe._features_dim}")
            
            # Check q_net structure
            if hasattr(self.model.policy, 'q_net'):
                q_net = self.model.policy.q_net
                print(f"Q-Net: {type(q_net)}")
                print(f"  {q_net}")
            
            print("=== End of Inspection ===\n")
        except Exception as e:
            print(f"Error during model inspection: {e}")
    
    def _setup_hooks(self):
        """Setup PyTorch hooks to capture Q-values from the model."""
        if self.model is None or not hasattr(self.model, 'policy') or not hasattr(self.model.policy, 'q_net'):
            print("Cannot setup hooks: model or q_net not available")
            return False
        
        # Storage for captured values
        self.captured_q_values = None
        
        # Define hook function to capture output
        def q_values_hook(module, input, output):
            self.captured_q_values = output.detach().cpu().numpy()
            print(f"Hook captured Q-values: {self.captured_q_values}")
        
        # Register hook on the final layer of q_net
        # From inspection, we know q_net is a Sequential with the last layer being the Q-value output
        if hasattr(self.model.policy.q_net, 'q_net'):
            # Get the sequential q_net inside the QNetwork wrapper
            q_net_sequential = self.model.policy.q_net.q_net
            
            # Register hook on the last layer
            last_layer = q_net_sequential[-1]  # Last layer in Sequential
            self.hook_handle = last_layer.register_forward_hook(q_values_hook)
            print(f"Registered hook on final layer: {last_layer}")
            return True
        
        print("Could not register hook: q_net structure not as expected")
        return False
    
    def _compute_logits(self, input_array):
        """Compute model logits (Q-values) for the given input array."""
        if self.model is None:
            # Fallback: Generate simulated logits when model can't be loaded
            return self._generate_simulated_logits(input_array)
        
        try:
            # First time, inspect the model to understand its structure
            if not hasattr(self, '_model_inspected'):
                self._inspect_model_policy()
                self._model_inspected = True
                # Setup hooks to capture Q-values
                self._hooks_setup = self._setup_hooks()
            
            # Reset captured values
            self.captured_q_values = None
            
            # Convert input array to tensor
            input_tensor = torch.FloatTensor(input_array).unsqueeze(0)
            
            # Create observation dict as required by MultiInputPolicy
            observation = {"MLP_input": input_tensor}
            
            with torch.no_grad():
                # Use predict method which will trigger the forward hooks
                action, _states = self.model.predict(observation, deterministic=True)
                print(f"Predicted action: {action}")
                
                # Check if we captured Q-values through the hook
                if self.captured_q_values is not None:
                    return self.captured_q_values.squeeze()
                else:
                    print("Warning: Hook did not capture Q-values")
                    return self._generate_simulated_logits(input_array)
                
        except Exception as e:
            print(f"Error computing Q-values: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            # Fallback to simulated logits if there's an error
            return self._generate_simulated_logits(input_array)
    
    def _generate_simulated_logits(self, input_array):
        """Generate simulated logits based on the input array for visualization purposes."""
        # Use a simple algorithm to generate consistent but reactive logits
        # This ensures changes to the input will affect the displayed values
        
        # Ensure input_array is a numpy array
        input_array = np.array(input_array)
        
        # Sum different sections of the input array with different weights
        goal_info = np.sum(input_array[:8]) * 0.5  # First 8 values (goal information)
        
        grid_size = int(np.sqrt((len(input_array) - 8) // 2))
        grid1_sum = np.sum(input_array[8:8+grid_size*grid_size]) * 0.3
        grid2_sum = np.sum(input_array[8+grid_size*grid_size:]) * 0.2
        
        # Base logits
        base_logits = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Modify logits based on weighted sums
        base_logits[0] = 0.5 + goal_info + grid1_sum * 0.1  # Left
        base_logits[1] = 0.3 + goal_info * 0.8 + grid2_sum * 0.2  # Right
        base_logits[2] = 0.7 + grid1_sum * 0.2 + grid2_sum * 0.3  # Forward
        base_logits[3] = 0.1 + (grid1_sum + grid2_sum) * 0.1  # Stay
        
        # Add some randomness but maintain consistency for the same input
        # Use a simple hash function instead of trying to hash the array directly
        input_hash = int(sum([float(x) * (i+1) for i, x in enumerate(input_array)]) * 1000)
        rng = np.random.RandomState(input_hash % 10000)
        random_noise = rng.normal(0, 0.1, 4)
        
        # Add noise and ensure some variation between actions
        logits = base_logits + random_noise
        
        return logits
    
    def _update_logits_display(self):
        """Update the logits display based on current input and corruptions."""
        # Compute logits for original input
        if self.current_array is not None:
            self.original_logits = self._compute_logits(self.current_array)
            
            # Create corrupted array for logit computation
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
            
            # Compute logits for corrupted input
            self.corrupted_logits = self._compute_logits(corrupted_array)
            
            # Update the display
            self._display_logits()
    
    def _display_logits(self):
        """Update the display to show Q-values from the model."""
        # Clear existing text in all widgets
        self.original_text.config(state=tk.NORMAL)  # Make writable
        self.corrupted_text.config(state=tk.NORMAL)
        self.diff_text.config(state=tk.NORMAL)
        
        self.original_text.delete(1.0, tk.END)
        self.corrupted_text.delete(1.0, tk.END)
        self.diff_text.delete(1.0, tk.END)
        
        # Always show headers
        self.original_text.insert(tk.END, "Original Q-values:\n")
        self.corrupted_text.insert(tk.END, "Corrupted Q-values:\n")
        self.diff_text.insert(tk.END, "Changes:\n")
        
        if self.original_logits is None and self.corrupted_logits is None:
            self.original_text.insert(tk.END, "No values available")
            self.corrupted_text.insert(tk.END, "No values available")
            self.diff_text.insert(tk.END, "No values available")
            return
        
        # Function to safely convert any value to an array
        def ensure_array(val):
            if val is None:
                return None
            val_array = np.atleast_1d(val)  # Convert scalars to 1D arrays
            if len(val_array) == 0:  # Handle empty arrays
                return np.array([0.0])
            return val_array
        
        # Ensure our values are arrays
        orig_qvals = ensure_array(self.original_logits)
        corrupt_qvals = ensure_array(self.corrupted_logits)
        
        # Original Q-values
        if orig_qvals is not None:
            best_idx = np.argmax(orig_qvals)
            for i, val in enumerate(orig_qvals):
                text = f"Action {i}: {val:.4f}"
                if i == best_idx:
                    text += " *"  # Asterisk marks best
                self.original_text.insert(tk.END, text + "\n")
        else:
            self.original_text.insert(tk.END, "N/A\n")
            
        # Corrupted Q-values
        if corrupt_qvals is not None:
            best_idx = np.argmax(corrupt_qvals)
            for i, val in enumerate(corrupt_qvals):
                text = f"Action {i}: {val:.4f}"
                if i == best_idx:
                    text += " *"  # Asterisk marks best
                self.corrupted_text.insert(tk.END, text + "\n")
        else:
            self.corrupted_text.insert(tk.END, "N/A\n")
            
        # Differences
        if orig_qvals is not None and corrupt_qvals is not None:
            min_len = min(len(orig_qvals), len(corrupt_qvals))
            diffs = corrupt_qvals[:min_len] - orig_qvals[:min_len]
            
            # Find max absolute difference
            max_diff_idx = np.argmax(np.abs(diffs))
            
            for i in range(min_len):
                diff_val = diffs[i]
                if diff_val > 0:
                    diff_marker = "↑"  # Up arrow for increase
                elif diff_val < 0:
                    diff_marker = "↓"  # Down arrow for decrease
                else:
                    diff_marker = " "
                    
                text = f"Action {i}: {diff_val:+.4f}{diff_marker}"
                if i == max_diff_idx and abs(diff_val) > 0.0001:
                    text += " *"  # Asterisk marks largest change
                self.diff_text.insert(tk.END, text + "\n")
        else:
            self.diff_text.insert(tk.END, "N/A\n")
        
        # Make all text widgets read-only
        self.original_text.config(state=tk.DISABLED)
        self.corrupted_text.config(state=tk.DISABLED)
        self.diff_text.config(state=tk.DISABLED)
        
        # Force update
        self.root.update()

def main():
    parser = argparse.ArgumentParser(description='State Corruption Tool')
    parser.add_argument('--path', required=True, help='Path to the agent directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--disable-logits', action='store_true', help='Disable logits calculation (use if model loading fails)')
    
    args = parser.parse_args()
    
    try:
        # Create and run the tool
        tool = StateCorruptionTool(args.path, args.seed)
        
        # Disable model loading if requested
        if args.disable_logits:
            tool.model = None
            print("Logits calculation disabled")
            
        tool.run_gui()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
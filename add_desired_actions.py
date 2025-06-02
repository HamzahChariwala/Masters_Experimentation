#!/usr/bin/env python3
"""
Add Desired Actions - Augment alter states with reference actions from dijkstra agents.

This script takes the alter states from optimization filtering and adds a 'desired_action' 
field by looking up what action a specified dijkstra agent (ruleset) took in the same 
environment and state. Optionally includes a GUI for visualizing and interacting with states.
"""

import os
import sys
import json
import argparse
import math
import random
from typing import Dict, Any, Optional, Tuple, List

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
    with open(file_path, 'w') as f:
        formatted_json = "{\n"
        entries = []
        
        for key, value in data.items():
            entry = f'  "{key}": {json.dumps(value)}'
            entries.append(entry)
        
        formatted_json += ",\n".join(entries)
        formatted_json += "\n}"
        f.write(formatted_json)

def parse_alter_state_key(key: str) -> tuple:
    """
    Parse an alter state key to extract environment and state components.
    
    Args:
        key: Key like "MiniGrid-LavaCrossingS11N5-v0-81103-2,1,1-0001"
        
    Returns:
        Tuple of (environment, state_key) where environment is the env name 
        and state_key is the state coordinates
    """
    # Split by last dash to separate counter suffix
    base_key = '-'.join(key.split('-')[:-1])
    
    # Split by the last dash again to separate state from environment  
    parts = base_key.split('-')
    
    # The state key is the last part, environment is everything before
    state_key = parts[-1]
    environment = '-'.join(parts[:-1])
    
    return environment, state_key

def get_dijkstra_action(env_name: str, state_key: str, ruleset: str, 
                       dijkstra_base_path: str) -> Optional[int]:
    """
    Get the action taken by a dijkstra agent for a specific environment and state.
    
    Args:
        env_name: Environment name (e.g., "MiniGrid-LavaCrossingS11N5-v0-81103")
        state_key: State coordinates (e.g., "2,1,1") 
        ruleset: Dijkstra agent ruleset (e.g., "standard")
        dijkstra_base_path: Base path to dijkstra evaluation files
        
    Returns:
        Action number if found, None otherwise
    """
    dijkstra_file = os.path.join(dijkstra_base_path, f"{env_name}.json")
    
    if not os.path.exists(dijkstra_file):
        return None
    
    try:
        dijkstra_data = load_json_file(dijkstra_file)
        
        # Navigate to the performance data for the specified ruleset
        if 'performance' not in dijkstra_data or ruleset not in dijkstra_data['performance']:
            return None
            
        ruleset_data = dijkstra_data['performance'][ruleset]
        
        # Get the action for the specific state
        if state_key in ruleset_data:
            state_array = ruleset_data[state_key]
            # Action is at index 7 in the array
            action = state_array[7]
            return action if action is not None else None
        
        return None
        
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None

def parse_input_grid(input_array: list) -> Tuple[int, list, list]:
    """
    Parse the input array to extract grid size and lava/barrier information.
    
    Args:
        input_array: The input array from state data
        
    Returns:
        Tuple of (grid_size, lava_grid, barrier_grid)
    """
    # Skip first 8 elements, then remaining elements are split between lava and barriers
    grid_data = input_array[8:]
    grid_elements = len(grid_data) // 2
    grid_size = int(math.sqrt(grid_elements))
    
    # Extract lava and barrier grids
    lava_flat = grid_data[:grid_elements]
    barrier_flat = grid_data[grid_elements:]
    
    # Convert to 2D grids
    lava_grid = []
    barrier_grid = []
    
    for i in range(grid_size):
        lava_row = lava_flat[i * grid_size:(i + 1) * grid_size]
        barrier_row = barrier_flat[i * grid_size:(i + 1) * grid_size]
        lava_grid.append(lava_row)
        barrier_grid.append(barrier_row)
    
    return grid_size, lava_grid, barrier_grid

def convert_desired_action_to_array(alter_states: Dict[str, Any]) -> None:
    """Convert desired_action from int to array format for all states."""
    for state_data in alter_states.values():
        if 'desired_action' in state_data:
            desired_action = state_data['desired_action']
            if isinstance(desired_action, int):
                state_data['desired_action'] = [desired_action]
            elif not isinstance(desired_action, list):
                state_data['desired_action'] = []

def randomize_state_order(state_keys: List[str], seed: int = 811) -> List[str]:
    """
    Randomize the order of state keys using the 4-digit suffix as unique identifier.
    
    Args:
        state_keys: List of state keys
        seed: Random seed for reproducible results
        
    Returns:
        Randomized list of state keys
    """
    # Extract 4-digit suffix from each key and pair with original key
    keyed_states = []
    for key in state_keys:
        # Extract last 4 digits (after last dash)
        suffix = key.split('-')[-1]
        try:
            unique_id = int(suffix)
        except ValueError:
            # Fallback if suffix is not numeric
            unique_id = hash(suffix) % 10000
        keyed_states.append((unique_id, key))
    
    # Set seed and randomize based on unique IDs
    random.seed(seed)
    random.shuffle(keyed_states)
    
    # Return just the keys in randomized order
    return [key for _, key in keyed_states]

class StateVisualizerGUI:
    """GUI for visualizing and interacting with alter states."""
    
    def __init__(self, alter_states: Dict[str, Any]):
        try:
            import tkinter as tk
            from tkinter import messagebox
        except ImportError:
            raise ImportError("tkinter is required for GUI mode. Install with: pip install tk")
        
        self.tk = tk
        self.messagebox = messagebox
        self.alter_states = alter_states
        self.state_keys = randomize_state_order(list(alter_states.keys()))
        self.current_index = 0
        self.user_choices = {}  # Will store user decisions for each state
        self.terminated = False
        self.action_buttons = []  # Store action button references
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("State Visualizer")
        self.root.geometry("700x800")
        
        # Create UI elements
        self.setup_ui()
        
        # Display first state
        self.display_current_state()
    
    def setup_ui(self):
        """Setup the GUI elements."""
        # Title label
        self.title_label = self.tk.Label(self.root, text="", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=10)
        
        # Info label
        self.info_label = self.tk.Label(self.root, text="", font=("Arial", 10))
        self.info_label.pack(pady=5)
        
        # Grid frame
        self.grid_frame = self.tk.Frame(self.root)
        self.grid_frame.pack(pady=20)
        
        # Action selection frame
        self.action_frame = self.tk.Frame(self.root)
        self.action_frame.pack(pady=10)
        
        self.action_label = self.tk.Label(self.action_frame, text="Action Selection:", font=("Arial", 11, "bold"))
        self.action_label.pack()
        
        self.action_buttons_frame = self.tk.Frame(self.action_frame)
        self.action_buttons_frame.pack(pady=5)
        
        # Manual verification frame
        self.verification_frame = self.tk.Frame(self.action_frame)
        self.verification_frame.pack(pady=5)
        
        self.manually_verified_var = self.tk.BooleanVar()
        self.verification_checkbox = self.tk.Checkbutton(
            self.verification_frame,
            text="Manually Verified",
            variable=self.manually_verified_var,
            command=self.toggle_manual_verification,
            font=("Arial", 10)
        )
        self.verification_checkbox.pack()
        
        # Button frame
        self.button_frame = self.tk.Frame(self.root)
        self.button_frame.pack(pady=20)
        
        # Buttons
        self.skip_button = self.tk.Button(
            self.button_frame, 
            text="Skip", 
            command=self.skip_current,
            font=("Arial", 12),
            bg="orange",
            width=10
        )
        self.skip_button.pack(side=self.tk.LEFT, padx=10)
        
        self.accept_button = self.tk.Button(
            self.button_frame, 
            text="Accept", 
            command=self.accept_current,
            font=("Arial", 12),
            bg="green",
            width=10
        )
        self.accept_button.pack(side=self.tk.LEFT, padx=10)
        
        self.terminate_button = self.tk.Button(
            self.button_frame, 
            text="Terminate & Save", 
            command=self.terminate_and_save,
            font=("Arial", 12),
            bg="red",
            width=15
        )
        self.terminate_button.pack(side=self.tk.LEFT, padx=10)
        
        # Progress label
        self.progress_label = self.tk.Label(self.root, text="", font=("Arial", 10))
        self.progress_label.pack(pady=10)
    
    def create_action_buttons(self, state_data: Dict[str, Any]):
        """Create action selection buttons."""
        # Clear existing action buttons
        for widget in self.action_buttons_frame.winfo_children():
            widget.destroy()
        self.action_buttons = []
        
        action_taken = state_data.get('action_taken')
        desired_actions = state_data.get('desired_action', [])
        if not isinstance(desired_actions, list):
            desired_actions = [desired_actions] if desired_actions is not None else []
        
        # Update manual verification checkbox
        manually_verified = state_data.get('manually_verified', False)
        self.manually_verified_var.set(manually_verified)
        
        # Create 5 action buttons (indices 0-4)
        for action_idx in range(5):
            is_taken = action_idx == action_taken
            is_desired = action_idx in desired_actions
            
            # Determine button color and state
            if is_taken:
                bg_color = "lightgray"
                button_state = "disabled"
            elif is_desired:
                bg_color = "lightblue"
                button_state = "normal"
            else:
                bg_color = "white"
                button_state = "normal"
            
            # Create button with proper command binding - made narrower
            def make_command(idx):
                return lambda: self.toggle_action(idx)
            
            button = self.tk.Button(
                self.action_buttons_frame,
                text=str(action_idx),
                width=6,
                height=2,
                bg=bg_color,
                state=button_state,
                font=("Arial", 12),
                command=make_command(action_idx)
            )
            button.pack(side=self.tk.LEFT, padx=5)
            self.action_buttons.append(button)
    
    def toggle_manual_verification(self):
        """Toggle the manual verification flag."""
        state_key = self.state_keys[self.current_index]
        state_data = self.alter_states[state_key]
        state_data['manually_verified'] = self.manually_verified_var.get()
        
        # Update info display
        self.update_current_display()
    
    def toggle_action(self, action_idx: int):
        """Toggle selection of an action."""
        state_key = self.state_keys[self.current_index]
        state_data = self.alter_states[state_key]
        
        # Ensure desired_action is a list
        if 'desired_action' not in state_data:
            state_data['desired_action'] = []
        elif not isinstance(state_data['desired_action'], list):
            current = state_data['desired_action']
            state_data['desired_action'] = [current] if current is not None else []
        
        desired_actions = state_data['desired_action']
        
        # Toggle the action
        if action_idx in desired_actions:
            desired_actions.remove(action_idx)
        else:
            desired_actions.append(action_idx)
        
        # Update the info display and button colors
        self.update_current_display()
    
    def update_current_display(self):
        """Update the current state display without rebuilding the entire grid."""
        if self.current_index >= len(self.state_keys):
            return
            
        state_key = self.state_keys[self.current_index]
        state_data = self.alter_states[state_key]
        
        # Update info text
        desired_actions = state_data.get('desired_action', [])
        if not isinstance(desired_actions, list):
            desired_actions = [desired_actions] if desired_actions is not None else []
        
        info_text = f"Grid Size: 7x7 | " \
                   f"Action Taken: {state_data.get('action_taken', 'N/A')} | " \
                   f"Desired Actions: {desired_actions} | " \
                   f"Manually Verified: {state_data.get('manually_verified', False)}"
        self.info_label.config(text=info_text)
        
        # Recreate action buttons to reflect changes
        self.create_action_buttons(state_data)
    
    def display_current_state(self):
        """Display the current state in the GUI."""
        if self.current_index >= len(self.state_keys):
            self.show_completion()
            return
        
        state_key = self.state_keys[self.current_index]
        state_data = self.alter_states[state_key]
        
        # Update title and info
        self.title_label.config(text=f"State: {state_key}")
        
        # Parse input to get grid information
        input_array = state_data['input']
        grid_size, lava_grid, barrier_grid = parse_input_grid(input_array)
        
        desired_actions = state_data.get('desired_action', [])
        if not isinstance(desired_actions, list):
            desired_actions = [desired_actions] if desired_actions is not None else []
        
        info_text = f"Grid Size: {grid_size}x{grid_size} | " \
                   f"Action Taken: {state_data.get('action_taken', 'N/A')} | " \
                   f"Desired Actions: {desired_actions} | " \
                   f"Manually Verified: {state_data.get('manually_verified', False)}"
        self.info_label.config(text=info_text)
        
        # Clear previous grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Create grid
        cell_size = 40
        for i in range(grid_size):
            for j in range(grid_size):
                # Determine cell color - FIXED: barrier=gray, lava=black
                if barrier_grid[i][j] == 1:
                    color = "gray"   # Wall/barrier
                elif lava_grid[i][j] == 1:
                    color = "black"  # Lava
                else:
                    color = "white"  # Empty space
                
                cell = self.tk.Frame(
                    self.grid_frame,
                    width=cell_size,
                    height=cell_size,
                    bg=color,
                    relief="solid",
                    borderwidth=1
                )
                cell.grid(row=i, column=j, padx=1, pady=1)
                cell.grid_propagate(False)
                
                # Add agent arrow at middle of first column (j=0, i=grid_size//2)
                if j == 0 and i == grid_size // 2:
                    arrow_label = self.tk.Label(
                        cell,
                        text="▶",
                        font=("Arial", 24, "bold"),
                        bg=color,
                        fg="lightgray"
                    )
                    arrow_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Create action selection buttons
        self.create_action_buttons(state_data)
        
        # Update progress
        progress_text = f"State {self.current_index + 1} of {len(self.state_keys)}"
        self.progress_label.config(text=progress_text)
    
    def skip_current(self):
        """Skip the current state."""
        state_key = self.state_keys[self.current_index]
        self.user_choices[state_key] = "skipped"
        self.current_index += 1
        self.display_current_state()
    
    def accept_current(self):
        """Accept the current state."""
        state_key = self.state_keys[self.current_index]
        self.user_choices[state_key] = "accepted"
        self.current_index += 1
        self.display_current_state()
    
    def terminate_and_save(self):
        """Terminate the process and save changes."""
        result = self.messagebox.askyesno(
            "Terminate", 
            f"Are you sure you want to terminate?\n"
            f"Processed: {self.current_index}/{len(self.state_keys)} states\n"
            f"This will save current progress."
        )
        if result:
            self.terminated = True
            self.root.quit()
    
    def show_completion(self):
        """Show completion message."""
        self.messagebox.showinfo(
            "Complete", 
            f"All {len(self.state_keys)} states have been processed!"
        )
        self.root.quit()
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()
        return self.user_choices, self.terminated

def add_desired_actions(alter_states_path: str, dijkstra_base_path: str, 
                       ruleset: str = "standard", gui_mode: bool = False) -> Dict[str, int]:
    """
    Add desired actions to alter states based on dijkstra agent behavior.
    
    Args:
        alter_states_path: Path to the alter states JSON file
        dijkstra_base_path: Path to directory containing dijkstra evaluation files
        ruleset: Dijkstra agent ruleset to use as reference
        gui_mode: Whether to enable GUI visualization mode
        
    Returns:
        Dictionary with statistics about the operation
    """
    print(f"Loading alter states from: {alter_states_path}")
    alter_states = load_json_file(alter_states_path)
    
    stats = {
        'total_states': len(alter_states),
        'actions_found': 0,
        'actions_missing': 0,
        'environments_processed': set()
    }
    
    print(f"Processing {stats['total_states']} alter states with ruleset '{ruleset}'")
    
    # Add desired actions first and convert to array format
    for state_key, state_data in alter_states.items():
        # Parse the state key to get environment and coordinates
        environment, coordinates = parse_alter_state_key(state_key)
        stats['environments_processed'].add(environment)
        
        # Get the desired action from dijkstra agent
        desired_action = get_dijkstra_action(environment, coordinates, ruleset, dijkstra_base_path)
        
        if desired_action is not None:
            state_data['desired_action'] = [desired_action]  # Store as array
            stats['actions_found'] += 1
        else:
            state_data['desired_action'] = []  # Empty array if no action found
            stats['actions_missing'] += 1
            print(f"Warning: No action found for {environment} state {coordinates}")
    
    # Convert any existing desired_action values to array format
    convert_desired_action_to_array(alter_states)
    
    # Run GUI mode if enabled
    gui_stats = {}
    if gui_mode:
        print("\nStarting GUI visualization mode...")
        print("Instructions:")
        print("- Black tiles = lava")
        print("- Gray tiles = walls/barriers")
        print("- White tiles = empty space")
        print("- Gray arrow = agent position")
        print("- Click action numbers to add/remove desired actions")
        print("- Gray action button = action taken (unclickable)")
        print("- Blue action button = currently selected desired action")
        print("- Skip: Skip current state")
        print("- Accept: Accept current state")
        print("- Terminate & Save: Stop and save progress")
        
        try:
            gui = StateVisualizerGUI(alter_states)
            user_choices, terminated = gui.run()
            
            gui_stats = {
                'gui_enabled': True,
                'states_viewed': len([k for k, v in user_choices.items() if v in ['accepted', 'skipped']]),
                'states_accepted': len([k for k, v in user_choices.items() if v == 'accepted']),
                'states_skipped': len([k for k, v in user_choices.items() if v == 'skipped']),
                'terminated_early': terminated
            }
            
            print(f"\nGUI session completed:")
            print(f"- States viewed: {gui_stats['states_viewed']}")
            print(f"- States accepted: {gui_stats['states_accepted']}")
            print(f"- States skipped: {gui_stats['states_skipped']}")
            print(f"- Terminated early: {gui_stats['terminated_early']}")
            
        except ImportError as e:
            print(f"GUI mode failed: {e}")
            gui_stats = {'gui_enabled': False, 'error': str(e)}
        except Exception as e:
            print(f"GUI error: {e}")
            gui_stats = {'gui_enabled': False, 'error': str(e)}
    
    # Convert set to count for final stats
    stats['environments_processed'] = len(stats['environments_processed'])
    stats.update(gui_stats)
    
    print(f"Saving updated alter states to: {alter_states_path}")
    save_json_file(alter_states, alter_states_path)
    
    return stats

def main():
    """Main entry point for adding desired actions."""
    parser = argparse.ArgumentParser(
        description='Add desired actions from dijkstra agents to alter states.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add standard dijkstra actions to alter states
  python add_desired_actions.py --alter_path ../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/optimisation_states/alter/states.json --ruleset standard
  
  # Use conservative dijkstra agent
  python add_desired_actions.py --alter_path ../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/optimisation_states/alter/states.json --ruleset conservative
  
  # Enable GUI visualization mode
  python add_desired_actions.py --alter_path ../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/optimisation_states/alter/states.json --ruleset standard --gui
  
  # Specify custom dijkstra path
  python add_desired_actions.py --alter_path path/to/alter/states.json --dijkstra_path path/to/dijkstra/files --ruleset standard
        """
    )
    
    parser.add_argument('--alter_path', required=True,
                       help='Path to the alter states JSON file')
    parser.add_argument('--dijkstra_path', default="Behaviour_Specification/Evaluations",
                       help='Path to directory containing dijkstra evaluation files (default: Behaviour_Specification/Evaluations)')
    parser.add_argument('--ruleset', default="standard",
                       choices=['standard', 'conservative', 'dangerous_1', 'dangerous_2', 'dangerous_3', 'dangerous_4', 'dangerous_5'],
                       help='Dijkstra agent ruleset to use as reference (default: standard)')
    parser.add_argument('--gui', action='store_true',
                       help='Enable GUI visualization mode')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.alter_path):
        print(f"Error: Alter states file not found: {args.alter_path}")
        return 1
        
    if not os.path.exists(args.dijkstra_path):
        print(f"Error: Dijkstra directory not found: {args.dijkstra_path}")
        return 1
    
    try:
        # Add desired actions
        stats = add_desired_actions(args.alter_path, args.dijkstra_path, args.ruleset, args.gui)
        
        # Print summary
        print("\n" + "="*60)
        print("ADD DESIRED ACTIONS SUMMARY")
        print("="*60)
        print(f"Alter states file: {args.alter_path}")
        print(f"Dijkstra ruleset: {args.ruleset}")
        print(f"GUI mode: {args.gui}")
        print(f"Total states processed: {stats['total_states']}")
        print(f"Actions found: {stats['actions_found']}")
        print(f"Actions missing: {stats['actions_missing']}")
        print(f"Environments processed: {stats['environments_processed']}")
        print(f"Success rate: {stats['actions_found']/stats['total_states']*100:.1f}%")
        
        if args.gui and stats.get('gui_enabled'):
            print(f"GUI states viewed: {stats.get('states_viewed', 0)}")
            print(f"GUI states accepted: {stats.get('states_accepted', 0)}")
            print(f"GUI states skipped: {stats.get('states_skipped', 0)}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
# In this file we will implement the visualisation of the path taken by the agent in the environment.
# This should work by taking a specified file in the JSON format and plotting the path taken by the agent in the environment.
# For now, we want to plot the whole environment, even if the agent has a prtial view
# Lets start with random colours for now for each type of object and we can iterate on the format later

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import os

def convert_to_array(obj):
    """Convert a nested list to a NumPy array."""
    return np.array(obj)

def visualize_combined(agent_positions, env_img, actions=None):
    """
    Create a single plot showing the environment as a grid:
      - Walls (0) are shown in dark (dimgray).
      - Empty cells (1) are shown in light (lightgrey).
      - Special cells (2) are shown in a different color (orange).
      - Goal cells (3) are shown with a light green background.
    Also, overlay the agent's red path with markers,
    and mark the start and goal positions with green circles.
    
    agent_positions: list of (row, col) coordinates.
    env_img: 2D array for the environment (0 = wall, 1 = empty, 2 = special, 3 = goal).
    actions: list of actions taken by the agent.
    """
    # Print debug information about agent positions
    print(f"Number of agent positions: {len(agent_positions)}")
    for i, pos in enumerate(agent_positions):
        action_info = f", Action: {actions[i]}" if actions and i < len(actions) else ""
        print(f"Position {i}: {pos}{action_info}")
    
    # Check if the agent is moving or staying in the same position
    if len(agent_positions) > 1:
        same_position = True
        first_pos = agent_positions[0]
        for pos in agent_positions[1:]:
            if not np.array_equal(pos, first_pos):
                same_position = False
                break
        
        if same_position:
            print("WARNING: Agent appears to be staying in the same position for all steps!")
            if actions:
                # Count and display each action type
                action_counts = {}
                for action in actions:
                    if action in action_counts:
                        action_counts[action] += 1
                    else:
                        action_counts[action] = 1
                print("Action counts:", action_counts)
        else:
            print("Agent moves between positions")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Find goal cell position (value 3)
    goal_coords = np.argwhere(env_img == 3)
    goal_position = None
    if goal_coords.size > 0:
        goal_position = goal_coords[0]  # Take the first goal cell if multiple exist
        goal_row, goal_col = goal_position
        print(f"Goal position: ({goal_row}, {goal_col})")
    else:
        print("No goal position (value 3) found in the environment")
    
    # Updated colormap: 
    # 0 (wall) -> dimgray
    # 1 (empty) -> lightgrey
    # 2 (special) -> orange
    # 3 (goal) -> light green
    cmap = mcolors.ListedColormap(['dimgray', 'lightgrey', 'orange', 'lightgreen'])
    
    if env_img is not None:
        # Print environment grid for debugging
        print("Environment grid shape:", env_img.shape)
        print("Unique values in environment:", np.unique(env_img))
        
        # Keep the original values including 3 for goal
        env_display = np.copy(env_img)
        
        ax.imshow(env_display, cmap=cmap, origin='upper')
        
        # Draw grid lines with minor ticks.
        nrows, ncols = env_img.shape
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    if agent_positions:
        positions = np.array(agent_positions)
        xs = positions[:, 1]  # X corresponds to column index.
        ys = positions[:, 0]  # Y corresponds to row index.
        
        # Plot the agent path in red.
        ax.plot(xs, ys, marker='o', color='red', label='Agent path')
        for i, (x, y) in enumerate(zip(xs, ys), start=0):
            action_text = str(i)
            if actions and i-1 < len(actions) and i > 0:  # Skip action for position 0
                action_text += f" (a:{actions[i-1]})"
            ax.text(x, y, action_text, color='blue', fontsize=8)
        
        # Mark start position with a green circle and label.
        start_x, start_y = xs[0], ys[0]
        ax.scatter([start_x], [start_y], color='green', s=100, label='Start', edgecolors='black')
        ax.text(start_x, start_y, "Start", color='green', fontsize=10, fontweight='bold')
    
    # Mark goal position based on the environment grid value of 3
    if goal_position is not None:
        goal_y, goal_x = goal_position  # Row, column to y, x
        ax.scatter([goal_x], [goal_y], color='green', s=100, label='Goal', edgecolors='black')
        ax.text(goal_x, goal_y, "Goal", color='green', fontsize=10, fontweight='bold')
    
    plt.title("Agent Path Visualisation")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    # Determine the correct path to the log file
    # This adjusts the path whether we're running from the project root or the VisualisationTools directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, "AgentTesting", "agent_run_log_simple_lava.json")
    
    print(f"Loading log data from: {log_path}")
    
    # Load the JSON log data.
    with open(log_path, "r") as f:
        log_data = json.load(f)
    
    # Sort action keys assuming keys like "action 0", "action 1", etc.
    action_keys = sorted(log_data.keys(), key=lambda k: int(k.split()[1]))
    print(f"Found {len(action_keys)} action steps in the log")
    
    agent_positions = []  # To store agent positions from each action.
    agent_actions = []    # To store actions taken by the agent.
    env_img = None        # Save environment image from the FIRST valid new_image entry.
    
    for key in action_keys:
        entry = log_data[key]
        
        # Extract the action taken (skip for action 0 which is the initial state)
        if key != "action 0" and "action" in entry:
            action = entry["action"]
            agent_actions.append(action)
            print(f"{key}: Action taken: {action}")
        
        ld = entry.get("info", {}).get("log_data", {})
        if "new_image" in ld:
            # Convert new_image list to a NumPy array (expected shape: (2, H, W)).
            new_image = convert_to_array(ld["new_image"])
            if new_image.ndim == 3 and new_image.shape[0] == 2:
                agent_channel = new_image[0]
                env_channel = new_image[1]
                print(f"{key}: Agent channel shape: {agent_channel.shape}")
            else:
                agent_channel = new_image
                env_channel = new_image
                print(f"{key}: Using single channel, shape: {new_image.shape}")
            
            # Save the environment image from the FIRST valid new_image entry.
            if env_img is None:
                env_img = env_channel
                print(f"Saved environment image from {key}, shape: {env_img.shape}")
                
            # Determine the agent's position by finding nonzero pixels in agent_channel.
            coords = np.argwhere(agent_channel > 0)
            if coords.size > 0:
                pos = coords.mean(axis=0)  # (row, col)
                agent_positions.append(pos)
                
                # Print position with action if available
                last_action = agent_actions[-1] if agent_actions else None
                action_info = f", Action: {last_action}" if last_action is not None else ""
                print(f"{key}: Agent position: {pos}{action_info}")
            else:
                print(f"{key}: No agent position found (no nonzero pixels)")
        else:
            print(f"No new_image found in {key}")
    
    # Add explanation of action numbers
    print("\nAction mapping:")
    print("0: Move forward")
    print("1: Turn left")
    print("2: Turn right")
    print("3: Diagonal move left")
    print("4: Diagonal move right")
    
    visualize_combined(agent_positions, env_img, agent_actions)

if __name__ == "__main__":
    main()
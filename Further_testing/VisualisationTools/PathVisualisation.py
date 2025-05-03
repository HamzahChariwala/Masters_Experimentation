# In this file we will implement the visualisation of the path taken by the agent in the environment.
# This should work by taking a specified file in the JSON format and plotting the path taken by the agent in the environment.
# For now, we want to plot the whole environment, even if the agent has a prtial view
# Lets start with random colours for now for each type of object and we can iterate on the format later

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def convert_to_array(obj):
    """Convert a nested list to a NumPy array."""
    return np.array(obj)

def visualize_combined(agent_positions, env_img):
    """
    Create a single plot showing the environment as a grid:
      - Walls (0) are shown in dark (dimgray).
      - Empty cells (1) are shown in light (lightgrey).
    Also, overlay the agent's red path with markers,
    and mark the start and goal positions with green circles.
    
    agent_positions: list of (row, col) coordinates.
    env_img: 2D array for the environment (0 = wall, 1 = empty; cell value 3 already converted).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Updated colormap: 0 (wall) -> dimgray, 1 (empty) -> lightgrey.
    cmap = mcolors.ListedColormap(['dimgray', 'lightgrey'])
    
    if env_img is not None:
        ax.imshow(env_img, cmap=cmap, origin='upper')
        
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
            ax.text(x, y, str(i), color='blue', fontsize=8)
        
        # Mark start position with a green circle and label.
        start_x, start_y = xs[0], ys[0]
        ax.scatter([start_x], [start_y], color='green', s=100, label='Start', edgecolors='black')
        ax.text(start_x, start_y, "Start", color='green', fontsize=10, fontweight='bold')
        
        # Mark goal position (last in sequence) with a green circle and label.
        goal_x, goal_y = xs[-1], ys[-1]
        ax.scatter([goal_x], [goal_y], color='green', s=100, label='Goal', edgecolors='black')
        ax.text(goal_x, goal_y, "Goal", color='green', fontsize=10, fontweight='bold')
    
    plt.title("Agent Path Visualisation")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    # Load the JSON log data.
    with open("AgentTesting/agent_run_log.json", "r") as f:
        log_data = json.load(f)
    
    # Sort action keys assuming keys like "action 0", "action 1", etc.
    action_keys = sorted(log_data.keys(), key=lambda k: int(k.split()[1]))
    
    agent_positions = []  # To store agent positions from each action.
    env_img = None        # Save environment image from the FIRST valid new_image entry.
    
    for key in action_keys:
        entry = log_data[key]
        ld = entry.get("info", {}).get("log_data", {})
        if "new_image" in ld:
            # Convert new_image list to a NumPy array (expected shape: (2, H, W)).
            new_image = convert_to_array(ld["new_image"])
            if new_image.ndim == 3 and new_image.shape[0] == 2:
                agent_channel = new_image[0]
                env_channel = new_image[1]
            else:
                agent_channel = new_image
                env_channel = new_image
            
            # Save the environment image from the FIRST valid new_image entry.
            if env_img is None:
                env_img = env_channel
                # Convert any goal cell (value 3) to 1 so that the goal is light grey.
                env_img = np.where(env_img == 3, 1, env_img)
                
            # Determine the agent's position by finding nonzero pixels in agent_channel.
            coords = np.argwhere(agent_channel > 0)
            if coords.size > 0:
                pos = coords.mean(axis=0)  # (row, col)
                agent_positions.append(pos)
        else:
            print(f"No new_image found in {key}")
    
    visualize_combined(agent_positions, env_img)

if __name__ == "__main__":
    main()
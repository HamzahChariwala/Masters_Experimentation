#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLOR_TO_IDX, IDX_TO_COLOR

def render_environment_structure(env_id, seed=12345):
    """
    Create and render a MiniGrid environment to visualize its structure.
    
    Parameters:
    ----------
    env_id : str
        Environment ID (e.g., "MiniGrid-LavaCrossingS11N5-v0")
    seed : int
        Random seed for the environment
    """
    # Create the environment
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Reset with the specified seed
    env.reset(seed=seed)
    
    # Get the grid object
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height
    
    # Create a representation of the grid
    structure = np.zeros((height, width), dtype=str)
    structure.fill(' ')  # Empty cells
    
    # Fill in the grid contents
    goal_pos = None
    lava_positions = []
    
    for i in range(width):
        for j in range(height):
            cell = grid.get(i, j)
            if cell is not None:
                if cell.type == 'wall':
                    structure[j, i] = 'W'
                elif cell.type == 'goal':
                    structure[j, i] = 'G'
                    goal_pos = (i, j)
                elif cell.type == 'lava':
                    structure[j, i] = 'L'
                    lava_positions.append((i, j))
                else:
                    structure[j, i] = cell.type[0].upper()
    
    # Mark the agent position
    agent_pos = tuple(env.agent_pos)  # Convert NumPy array to tuple
    structure[agent_pos[1], agent_pos[0]] = 'A'
    
    # Print the environment structure
    print(f"\nEnvironment: {env_id}, Seed: {seed}")
    print(f"Grid size: {width}x{height}")
    print(f"Goal position: {goal_pos}")
    print(f"Agent starting position: {agent_pos}")
    print(f"Number of lava cells: {len(lava_positions)}")
    
    # Print the grid
    print("\nGrid Structure:")
    print("  ", end="")
    for i in range(width):
        print(f"{i:2d}", end=" ")
    print()
    
    for j in range(height):
        print(f"{j:2d}", end=" ")
        for i in range(width):
            print(f" {structure[j, i]} ", end="")
        print()
    
    # Visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get RGB rendering
    img = env.render()
    ax.imshow(img)
    ax.set_title(f"Environment: {env_id}, Seed: {seed}")
    
    plt.tight_layout()
    plt.savefig(f"{env_id.replace('-', '_')}_seed_{seed}.png")
    print(f"\nSaved visualization to {env_id.replace('-', '_')}_seed_{seed}.png")
    
    # Close the environment
    env.close()
    
    # Return the grid structure and positions for comparison
    return structure, goal_pos, lava_positions, agent_pos

def generate_distribution_array(width, height, goal_pos, lava_positions):
    """
    Generate a mock distribution array for the given environment structure.
    
    Parameters:
    ----------
    width, height : int
        Grid dimensions
    goal_pos : tuple
        Position of the goal (x, y)
    lava_positions : list
        List of lava positions [(x1, y1), (x2, y2), ...]
    
    Returns:
    -------
    numpy.ndarray
        Distribution array with zeros for goal and lava positions
    """
    # Create a uniform distribution
    grid = np.ones((height, width)) / (width * height)
    
    # Zero out goal position
    if goal_pos:
        grid[goal_pos[1], goal_pos[0]] = 0.0
    
    # Zero out lava positions
    for x, y in lava_positions:
        grid[y, x] = 0.0
    
    # Renormalize
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
        
    return grid

def print_numeric_distribution(grid, goal_pos=None, lava_positions=None, title=None):
    """
    Print a numeric distribution array with formatting.
    """
    if title:
        print(f"\n=== {title} ===")
    
    height, width = grid.shape
    
    if lava_positions is None:
        lava_positions = []
    
    print("\n  Probability Distribution Array:")
    
    # For larger grids, use more compact formatting
    is_large_grid = width > 8 or height > 8
    cell_format = "{:.2e}" if is_large_grid else "{:.5f}"
    column_width = 6 if is_large_grid else 8
    
    # Print column headers
    header = "    "
    for x in range(width):
        if is_large_grid:
            header += f"{x:5d} "
        else:
            header += f"{x:7d} "
    print(header)
    
    # Print horizontal line
    print("    " + "-" * (column_width * width))
    
    # Format the probabilities with special symbols for goal and lava
    for y in range(height):
        row = f"{y:2d} |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position
                row += " GOAL " if is_large_grid else "  GOAL  "
            elif (x, y) in lava_positions:
                # Lava position
                row += " LAVA " if is_large_grid else "  LAVA  "
            else:
                # Regular cell with probability value
                prob = grid[y, x]
                row += f" {cell_format.format(prob)}"
        print(row)
    
    # Print legend
    print("\n  Legend:")
    print("  - GOAL: Goal position (zero probability)")
    print("  - LAVA: Lava cell (zero probability)")
    print("  - Values represent spawn probabilities (uniform distribution)")

if __name__ == "__main__":
    # Define environment and seed
    env_id = "MiniGrid-LavaCrossingS11N5-v0"
    seed = 12345  # Use the same seed as in config.yaml
    
    # Render the environment and get its structure
    structure, goal_pos, lava_positions, agent_pos = render_environment_structure(env_id, seed)
    
    # Create and print a mock distribution
    width, height = len(structure[0]), len(structure)
    grid = generate_distribution_array(width, height, goal_pos, lava_positions)
    
    # Print a representation similar to what we have in our visualization
    print_numeric_distribution(grid, goal_pos, lava_positions, 
                              title="Actual Environment Structure") 
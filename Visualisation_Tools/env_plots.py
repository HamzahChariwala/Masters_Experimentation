import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import minigrid
import os


def plot_lavacrossing_with_agent(seed, agent_x, agent_y, agent_theta, env_id="MiniGrid-LavaCrossingS11N5-v0", save_path=None, window_size=7):
    """
    Create a visualization of a lavacrossing environment with agent position and partial observability.
    
    Args:
        seed (int): Seed for environment generation
        agent_x (int): Agent x position
        agent_y (int): Agent y position  
        agent_theta (int): Agent orientation (0=North, 1=East, 2=South, 3=West)
        env_id (str): Environment ID (default: MiniGrid-LavaCrossingS11N5-v0)
        save_path (str): Path to save the plot. If None, saves to raw_envs/
        window_size (int): Size of observation window (default: 7)
        
    Returns:
        str: Path where the plot was saved
    """
    # Create environment and reset with seed
    env = gym.make(env_id, render_mode='rgb_array')
    obs, info = env.reset(seed=seed)
    
    # Extract grid structure
    if 'log_data' in info and 'new_image' in info['log_data']:
        grid_data = info['log_data']['new_image'][1]
        grid_array = np.array(grid_data)
    else:
        # Fallback: extract from environment grid directly
        grid_array = extract_grid_from_env(env)
    
    height, width = grid_array.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Define colors
    wall_color = '#5d5d5d'      # Slightly darker grey
    floor_color = '#f0f0f0'     # Slightly lighter gray
    lava_color = '#a0a0a0'      # Grey between floor and walls
    grid_color = '#000000'      # Faint black for borders
    goal_color = '#000000'      # Black for goal circle
    
    # Calculate observation window bounds
    # Agent is positioned on the edge of the window in the center
    half_window = window_size // 2
    
    if agent_theta == 0:  # North - agent at bottom center of window
        obs_min_x = agent_x - half_window
        obs_max_x = agent_x + half_window
        obs_min_y = agent_y - (window_size - 1)
        obs_max_y = agent_y
    elif agent_theta == 1:  # East - agent at left center of window
        obs_min_x = agent_x
        obs_max_x = agent_x + (window_size - 1)
        obs_min_y = agent_y - half_window
        obs_max_y = agent_y + half_window
    elif agent_theta == 2:  # South - agent at top center of window
        obs_min_x = agent_x - half_window
        obs_max_x = agent_x + half_window
        obs_min_y = agent_y
        obs_max_y = agent_y + (window_size - 1)
    elif agent_theta == 3:  # West - agent at right center of window
        obs_min_x = agent_x - (window_size - 1)
        obs_max_x = agent_x
        obs_min_y = agent_y - half_window
        obs_max_y = agent_y + half_window
    
    # Draw each cell
    for y in range(height):
        for x in range(width):
            cell_value = grid_array[y, x]
            
            # Determine base cell color
            if cell_value == 0:  # Wall
                cell_color = wall_color
            elif cell_value == 1:  # Floor
                cell_color = floor_color
            elif cell_value == 2:  # Lava
                cell_color = lava_color
            elif cell_value == 3:  # Goal
                cell_color = floor_color  # Base floor color, will add circle
            else:
                cell_color = floor_color  # Default to floor
            
            # Check if cell is outside observation window
            outside_window = (x < obs_min_x or x > obs_max_x or 
                            y < obs_min_y or y > obs_max_y)
            
            # Draw base cell
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, 
                                   edgecolor=grid_color, facecolor=cell_color, alpha=0.8)
            ax.add_patch(rect)
            
            # Add goal circle
            if cell_value == 3:  # Goal - add black circle
                circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                                      facecolor=goal_color, alpha=0.9)
                ax.add_patch(circle)
            
            # Add darkening overlay for cells outside observation window
            if outside_window:
                overlay = patches.Rectangle((x, y), 1, 1, linewidth=0, 
                                          facecolor='black', alpha=0.4)
                ax.add_patch(overlay)
    
    # Draw agent as triangular arrowhead
    arrow_size = 0.3
    arrow_center_x = agent_x + 0.5
    arrow_center_y = agent_y + 0.5
    
    # Define triangle vertices based on theta (accounting for inverted y-axis)
    if agent_theta == 0:  # North (up on screen)
        # Triangle pointing up
        triangle_points = [
            (arrow_center_x, arrow_center_y - arrow_size),  # tip
            (arrow_center_x - arrow_size/2, arrow_center_y + arrow_size/2),  # bottom left
            (arrow_center_x + arrow_size/2, arrow_center_y + arrow_size/2)   # bottom right
        ]
    elif agent_theta == 1:  # East (right)
        # Triangle pointing right
        triangle_points = [
            (arrow_center_x + arrow_size, arrow_center_y),  # tip
            (arrow_center_x - arrow_size/2, arrow_center_y - arrow_size/2),  # top left
            (arrow_center_x - arrow_size/2, arrow_center_y + arrow_size/2)   # bottom left
        ]
    elif agent_theta == 2:  # South (down on screen)
        # Triangle pointing down
        triangle_points = [
            (arrow_center_x, arrow_center_y + arrow_size),  # tip
            (arrow_center_x - arrow_size/2, arrow_center_y - arrow_size/2),  # top left
            (arrow_center_x + arrow_size/2, arrow_center_y - arrow_size/2)   # top right
        ]
    elif agent_theta == 3:  # West (left)
        # Triangle pointing left
        triangle_points = [
            (arrow_center_x - arrow_size, arrow_center_y),  # tip
            (arrow_center_x + arrow_size/2, arrow_center_y - arrow_size/2),  # top right
            (arrow_center_x + arrow_size/2, arrow_center_y + arrow_size/2)   # bottom right
        ]
    
    # Draw triangle
    triangle = patches.Polygon(triangle_points, facecolor='black', alpha=0.9)
    ax.add_patch(triangle)
    
    # Set up the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match grid orientation
    
    # Remove ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    orientation_names = ['North', 'East', 'South', 'West']
    ax.set_title(f'{env_id} - Seed {seed} - Agent at ({agent_x},{agent_y}) facing {orientation_names[agent_theta]}', 
                fontsize=12, fontweight='bold')
    
    # Determine save path
    if save_path is None:
        # Save to partial subfolder in Visualisation_Tools
        save_dir = os.path.join(os.path.dirname(__file__), "partial")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lavacrossing_s11n5_seed_{seed}_agent_{agent_x}_{agent_y}_{agent_theta}.png")
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close environment
    env.close()
    
    print(f"Environment with agent visualization saved to: {save_path}")
    return save_path


def plot_lavacrossing_environment(seed, env_id="MiniGrid-LavaCrossingS11N5-v0", save_path=None):
    """
    Create a custom visualization of a lavacrossing environment with specific styling.
    
    Args:
        seed (int): Seed for environment generation
        env_id (str): Environment ID (default: MiniGrid-LavaCrossingS11N5-v0)
        save_path (str): Path to save the plot. If None, saves to raw_envs/
        
    Returns:
        str: Path where the plot was saved
    """
    # Create environment and reset with seed
    env = gym.make(env_id, render_mode='rgb_array')
    obs, info = env.reset(seed=seed)
    
    # Extract grid structure
    if 'log_data' in info and 'new_image' in info['log_data']:
        grid_data = info['log_data']['new_image'][1]
        grid_array = np.array(grid_data)
    else:
        # Fallback: extract from environment grid directly
        grid_array = extract_grid_from_env(env)
    
    height, width = grid_array.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Define colors
    wall_color = '#5d5d5d'      # Slightly darker grey
    floor_color = '#f0f0f0'     # Slightly lighter gray
    lava_color = '#a0a0a0'      # Grey between floor and walls
    grid_color = '#000000'      # Faint black for borders
    goal_color = '#000000'      # Black for goal circle
    
    # Draw each cell
    for y in range(height):
        for x in range(width):
            cell_value = grid_array[y, x]
            
            # Determine cell type (mapping from extract_grid.py)
            # 0: wall, 1: floor, 2: lava, 3: goal
            if cell_value == 0:  # Wall
                cell_color = wall_color
            elif cell_value == 1:  # Floor
                cell_color = floor_color
            elif cell_value == 2:  # Lava
                cell_color = lava_color
            elif cell_value == 3:  # Goal
                cell_color = floor_color  # Base floor color, will add circle
            else:
                cell_color = floor_color  # Default to floor
            
            # Draw base cell
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, 
                                   edgecolor=grid_color, facecolor=cell_color, alpha=0.8)
            ax.add_patch(rect)
            
            # Add goal circle
            if cell_value == 3:  # Goal - add black circle
                circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                                      facecolor=goal_color, alpha=0.9)
                ax.add_patch(circle)
    
    # Set up the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match grid orientation
    
    # Remove ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title(f'{env_id} - Seed {seed}', fontsize=14, fontweight='bold')
    
    # Determine save path
    if save_path is None:
        # Save to raw_envs subfolder in Visualisation_Tools
        save_dir = os.path.join(os.path.dirname(__file__), "raw_envs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lavacrossing_s11n5_seed_{seed}.png")
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close environment
    env.close()
    
    print(f"Environment visualization saved to: {save_path}")
    return save_path


def extract_grid_from_env(env):
    """
    Fallback method to extract grid structure directly from environment.
    
    Args:
        env: MiniGrid environment
        
    Returns:
        np.ndarray: 2D array with cell type values
    """
    grid = env.unwrapped.grid
    height, width = grid.height, grid.width
    
    # Create array to store cell types
    grid_array = np.zeros((height, width), dtype=int)
    
    # Map MiniGrid objects to our cell type values
    for y in range(height):
        for x in range(width):
            cell = grid.get(x, y)
            
            if cell is None:
                grid_array[y, x] = 1  # Floor
            elif cell.type == 'wall':
                grid_array[y, x] = 0  # Wall
            elif cell.type == 'lava':
                grid_array[y, x] = 2  # Lava
            elif cell.type == 'goal':
                grid_array[y, x] = 3  # Goal
            else:
                grid_array[y, x] = 1  # Default to floor
    
    return grid_array


if __name__ == "__main__":
    # Test the function with seed 81102
    plot_lavacrossing_environment(81102)

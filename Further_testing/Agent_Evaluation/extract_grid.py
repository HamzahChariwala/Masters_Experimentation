import numpy as np
import matplotlib.pyplot as plt
import os

def extract_grid_from_env(env):
    """
    Extract grid layout from a MiniGrid environment.
    
    Args:
        env: A MiniGrid environment instance
        
    Returns:
        np.ndarray: A 2D array with string cell types ("floor", "wall", "lava", "goal")
        with shape (height, width) where [0,0] is the top-left corner
    """
    # Reset the environment
    obs, _ = env.reset()
    
    # Try to access grid directly through environment unwrapping
    grid = None
    current_env = env
    max_depth = 3
    
    # Try to find the grid by unwrapping the environment
    for i in range(max_depth):
        # Check current level
        if hasattr(current_env, 'grid'):
            grid = current_env.grid
            break
            
        # Try env attribute
        if hasattr(current_env, 'env'):
            current_env = current_env.env
            if hasattr(current_env, 'grid'):
                grid = current_env.grid
                break
        # Try unwrapped attribute
        elif hasattr(current_env, 'unwrapped'):
            current_env = current_env.unwrapped
            if hasattr(current_env, 'grid'):
                grid = current_env.grid
                break
        else:
            break
    
    # If grid was found, create a tensor from it
    if grid is not None:
        # Get grid dimensions
        width, height = grid.width, grid.height
        
        # Create our output tensor
        env_tensor = np.full((height, width), "floor", dtype=object)
        
        # Fill the tensor based on the grid content
        for y in range(height):
            for x in range(width):
                cell = grid.get(x, y)
                
                if cell is None:
                    env_tensor[y, x] = "floor"
                else:
                    # Map cell types
                    if hasattr(cell, 'type'):
                        cell_type = cell.type
                        
                        # Map based on string type
                        if cell_type == "wall":
                            env_tensor[y, x] = "wall"
                        elif cell_type == "lava":
                            env_tensor[y, x] = "lava"
                        elif cell_type == "goal":
                            env_tensor[y, x] = "goal"
                        elif cell_type == "empty":
                            env_tensor[y, x] = "floor"
                        else:
                            env_tensor[y, x] = "floor"
                            
                    # If object has a numeric 'type_idx' attribute
                    elif hasattr(cell, 'type_idx'):
                        type_idx = cell.type_idx
                        if type_idx == 0:  # Usually empty/floor
                            env_tensor[y, x] = "floor"
                        elif type_idx == 1:  # Usually wall
                            env_tensor[y, x] = "wall"
                        elif type_idx == 2:  # Usually door
                            env_tensor[y, x] = "wall"
                        elif type_idx == 8:  # Usually goal
                            env_tensor[y, x] = "goal"
                        elif type_idx == 9:  # Usually lava
                            env_tensor[y, x] = "lava"
                        else:
                            env_tensor[y, x] = "floor"
        
        return env_tensor
    
    # If grid wasn't accessible, check if new_image exists in log_data
    next_obs, reward, terminated, truncated, info = env.step(0)  # Take a no-op action
    
    if 'log_data' in info and 'new_image' in info['log_data']:
        new_image = info['log_data']['new_image']
        if len(new_image.shape) == 3 and new_image.shape[0] == 2:
            # Extract the grid from new_image
            grid_array = new_image[1]
            height, width = grid_array.shape
            
            # Create tensor and map cell types
            env_tensor = np.full((height, width), "floor", dtype=object)
            
            # Corrected type mapping based on observed values
            type_map = {
                0: "wall",    # Walls
                1: "floor",   # Floors
                2: "lava",    # Lava
                3: "goal",    # Goal
                8: "goal",    # Alternative goal code
                9: "lava"     # Alternative lava code
            }
            
            for y in range(height):
                for x in range(width):
                    cell_value = grid_array[y, x]
                    env_tensor[y, x] = type_map.get(cell_value, "unknown")
            
            return env_tensor
    
    # If all else fails, create default grid based on environment ID
    env_id = str(env)
    
    # Default layout for LavaCrossing environments
    if "LavaCrossing" in env_id:
        width, height = 11, 11
        env_tensor = np.full((height, width), "floor", dtype=object)
        
        # Add walls around the perimeter
        env_tensor[0, :] = "wall"
        env_tensor[-1, :] = "wall"
        env_tensor[:, 0] = "wall"
        env_tensor[:, -1] = "wall"
        
        # Add vertical lava strips
        columns_with_lava = np.linspace(2, width-3, 5).astype(int)
        for col in columns_with_lava:
            env_tensor[1:-1, col] = "lava"
        
        # Add goal at bottom right
        env_tensor[-2, -2] = "goal"
    else:
        # Generic grid for other environments
        width, height = 11, 11
        env_tensor = np.full((height, width), "floor", dtype=object)
        
        # Add walls around the perimeter
        env_tensor[0, :] = "wall"
        env_tensor[-1, :] = "wall" 
        env_tensor[:, 0] = "wall"
        env_tensor[:, -1] = "wall"
        
        # Add goal at bottom right
        env_tensor[-2, -2] = "goal"
    
    return env_tensor

def visualize_env_tensor(env_tensor, save_path="visualizations/env_layout.png", generate_plot=True):
    """
    Visualize the environment tensor using matplotlib.
    
    Args:
        env_tensor: A 2D numpy array with string cell types
        save_path: Path to save the visualization
        generate_plot: If False, no matplotlib plot will be generated (default: True)
    """
    if env_tensor is None:
        print("Cannot visualize empty environment tensor")
        return
    
    # If plot generation is disabled, skip the rest of the function
    if not generate_plot:
        print("Plot generation is disabled")
        return
    
    # Create a color map for cell types
    color_map = {"floor": 0, "wall": 1, "lava": 2, "goal": 3, "unknown": 4}
    colors = ['lightgray', 'darkgray', 'red', 'green', 'purple']
    
    # Get dimensions and create numeric representation
    height, width = env_tensor.shape
    numeric_tensor = np.zeros(env_tensor.shape, dtype=int)
    
    # Convert string types to numeric values
    for y in range(height):
        for x in range(width):
            numeric_tensor[y, x] = color_map.get(env_tensor[y, x], 4)
    
    # Plot the tensor
    plt.figure(figsize=(8, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    plt.imshow(numeric_tensor, cmap=cmap, origin='upper')
    
    # Add grid lines and labels
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    for y in range(height):
        for x in range(width):
            plt.text(x, y, env_tensor[y, x][0].upper(), 
                     ha='center', va='center', color='black', fontweight='bold')
    
    # Add coordinate labels and legend
    plt.xticks(np.arange(width), np.arange(width))
    plt.yticks(np.arange(height), np.arange(height))
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f"{label} ({label[0].upper()})")
        for i, label in enumerate(["floor", "wall", "lava", "goal", "unknown"])
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Environment Layout')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Environment visualization saved to {save_path}")

def print_env_tensor(env_tensor):
    """
    Print a text representation of the environment tensor to the terminal.
    
    Args:
        env_tensor: A 2D numpy array with string cell types
    """
    if env_tensor is None:
        print("Cannot print empty environment tensor")
        return
    
    print("\n===== Environment Tensor (for state.py) =====")
    
    # Print compact grid representation
    height, width = env_tensor.shape
    for y in range(height):
        row = [env_tensor[y, x][0].upper() for x in range(width)]
        print(' '.join(row))
    
    # Print detailed cell types
    print("\nDetailed cell types:")
    for y in range(height):
        for x in range(width):
            print(f"Position ({y}, {x}): {env_tensor[y, x]}") 
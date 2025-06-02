import numpy as np
import matplotlib.pyplot as plt
import os
import gymnasium as gym
import minigrid


def minigrid_render_simple(env_id, seed):

    env = gym.make(env_id, render_mode='human')
    obs, info = env.reset(seed=seed)

    print(f"Environment: {env_id}, Seed: {seed}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    done = False
    while not done:
        env.render()
        action = env.action_space.sample()  # Take random actions for demonstration
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    

def extract_env_structure(env, seed):
    """
    Extract the grid structure from the environment using log_data.
    
    Args:
        env: The MiniGrid environment
        seed: Seed to reset the environment with
        
    Returns:
        np.ndarray: A 2D numpy array with string cell types
    """
    obs, info = env.reset(seed=seed)
    
    # Check if log_data and new_image exist in info
    if 'log_data' in info and 'new_image' in info['log_data']:
        original = info['log_data']['new_image'][1]
        
        # Convert numerical values to string cell types
        mapping = {0: 'wall', 1: 'floor', 2: 'lava', 3: 'goal'}
        result = np.array([[mapping.get(int(cell), '?') for cell in row] for row in original])
        
        return result
    else:
        print("Warning: log_data or new_image not found in environment info")
        # Return an empty array or placeholder
        return np.array([['?']])


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
    
    # Ensure the directory exists, defaulting to EnvVisualisations inside Agent_Evaluation
    if save_path.startswith("visualizations/"):
        # Replace default visualization path with EnvVisualisations
        dirname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EnvVisualisations")
        filename = os.path.basename(save_path)
        save_path = os.path.join(dirname, filename)
    
    # Create the directory if it doesn't exist
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
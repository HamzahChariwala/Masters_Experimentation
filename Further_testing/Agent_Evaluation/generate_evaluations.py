import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import signal

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import from our import_vars.py file
from Agent_Evaluation.import_vars import (
    load_config,
    extract_env_config,
    create_evaluation_env,
    extract_and_visualize_env,
    load_agent,
    DEFAULT_RANK,
    DEFAULT_NUM_EPISODES
)

# Define a timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """
    Run a function with a timeout.
    
    Args:
        func: The function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_duration: Timeout in seconds
        default: Default value to return if function times out
        
    Returns:
        The result of func(*args, **kwargs) or default if it times out
    """
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result
    except TimeoutError:
        print(f"Function {func.__name__} timed out after {timeout_duration} seconds")
        return default
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled

def unwrap_environment(env, max_depth=20):
    """
    Safely unwrap an environment to find its grid attribute,
    with a maximum depth to prevent infinite loops.
    
    Args:
        env: The environment to unwrap
        max_depth: Maximum unwrapping depth
        
    Returns:
        tuple: (grid, agent_pos)
    """
    current_env = env
    unwrap_count = 0
    grid = None
    agent_pos = None
    
    print("Starting environment unwrapping process...")
    
    while hasattr(current_env, 'unwrapped') and unwrap_count < max_depth:
        # Check if this layer has grid
        if hasattr(current_env, 'grid'):
            grid = current_env.grid
            if hasattr(current_env, 'agent_pos'):
                agent_pos = current_env.agent_pos
            print(f"Found grid at unwrap depth {unwrap_count}")
            break
        
        # Move to next layer
        current_env = current_env.unwrapped
        unwrap_count += 1
        
        # Check again at this layer
        if hasattr(current_env, 'grid'):
            grid = current_env.grid
            if hasattr(current_env, 'agent_pos'):
                agent_pos = current_env.agent_pos
            print(f"Found grid at unwrap depth {unwrap_count}")
            break
    
    if unwrap_count >= max_depth:
        print(f"WARNING: Reached maximum unwrap depth ({max_depth}). Unable to find grid attribute.")
    
    return grid, agent_pos

def visualize_env_tensor(env_tensor):
    """
    Visualize the environment tensor using matplotlib.
    
    Args:
        env_tensor: A 2D numpy array with string cell types
    """
    # Create a numeric representation for plotting
    color_map = {
        "floor": 0,
        "wall": 1,
        "lava": 2,
        "goal": 3
    }
    
    # Get the dimensions
    height, width = env_tensor.shape
    
    # Convert string types to numeric values
    numeric_tensor = np.zeros(env_tensor.shape, dtype=int)
    for y in range(height):
        for x in range(width):
            numeric_tensor[y, x] = color_map.get(env_tensor[y, x], 0)
    
    # Create a custom colormap
    from matplotlib.colors import ListedColormap
    colors = ['lightgray', 'darkgray', 'red', 'green']
    cmap = ListedColormap(colors)
    
    # Plot the tensor with correct orientation
    plt.figure(figsize=(10, 10))
    
    # The origin should be at the top-left corner (like a normal grid)
    # and we need to display the tensor directly without transposing
    plt.imshow(numeric_tensor, cmap=cmap, origin='upper')
    
    # Add grid lines
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, width, 1), [])
    plt.yticks(np.arange(-0.5, height, 1), [])
    
    # Add labels for each cell
    for y in range(height):
        for x in range(width):
            plt.text(x, y, env_tensor[y, x][0].upper(), ha='center', va='center', 
                    color='black', fontweight='bold')
    
    # Add coordinate ticks
    plt.xticks(np.arange(width), np.arange(width))
    plt.yticks(np.arange(height), np.arange(height))
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', label='Floor (F)'),
        Patch(facecolor='darkgray', label='Wall (W)'),
        Patch(facecolor='red', label='Lava (L)'),
        Patch(facecolor='green', label='Goal (G)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Environment Layout')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/env_layout.png')
    plt.close()
    print("Environment visualization saved to visualizations/env_layout.png")


def print_env_tensor(env_tensor):
    """
    Print a text representation of the environment tensor to the terminal.
    
    Args:
        env_tensor: A 2D numpy array with string cell types
    """
    print("\n===== Environment Tensor (for state.py) =====")
    
    # Print the content in a grid format
    height, width = env_tensor.shape
    for y in range(height):
        row = []
        for x in range(width):
            # Use the first letter of the cell type for compact display
            cell_type = env_tensor[y, x][0].upper()
            row.append(cell_type)
        print(' '.join(row))
    
    # Also print a more detailed representation for verification
    print("\nDetailed cell types:")
    for y in range(height):
        for x in range(width):
            cell_type = env_tensor[y, x]
            print(f"Position ({y}, {x}): {cell_type}")


def single_env_evals(agent_path: str, env_id: str, seed: int):
    """
    Evaluate an agent in a single environment configuration
    
    Args:
        agent_path (str): Path to the agent folder in Agent_Storage
        env_id (str): Environment ID to use for evaluation
        seed (int): Random seed for reproducibility
    """
    print(f"\nEvaluating agent: {agent_path}")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    
    try:
        # Load config from agent folder
        config = load_config(agent_path)
        
        # Extract environment settings with provided env_id
        env_settings = extract_env_config(config, override_env_id=env_id)
        
        # Create evaluation environment first so we can extract the actual layout
        print("Creating environment...")
        env = create_evaluation_env(env_settings, seed=seed, override_rank=DEFAULT_RANK)
        
        try:
            # Extract and visualize the environment tensor using the new method
            print("Extracting environment layout...")
            env_tensor = extract_and_visualize_env(env, env_id=env_id)
            
            # Load agent
            print("Loading agent...")
            agent = load_agent(agent_path, config)
            
            # Here you would add the code to evaluate the agent and generate visualizations/metrics
            print(f"Ready to evaluate agent in {agent_path} for {DEFAULT_NUM_EPISODES} episodes")
            print(f"Using environment ID: {env_settings['env_id']}")
            print(f"Using seed: {seed}")
            print(f"Using rank: {DEFAULT_RANK}")
        except Exception as e:
            print(f"Error during environment tensor generation or agent evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            env.close()
    except Exception as e:
        print(f"Error setting up evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments - only path is configurable via command line
    parser = argparse.ArgumentParser(description="Generate evaluations for trained agents")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to the agent folder in Agent_Storage")
    args = parser.parse_args()
    
    # Call the evaluation function with default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 42
    single_env_evals(args.path, ENV_ID, SEED)

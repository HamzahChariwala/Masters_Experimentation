"""
Utility to inspect and visualize actual spawn distributions.

This can be imported and used to inspect distribution arrays from a FlexibleSpawnWrapper.
"""
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os
import time  # For timestamp generation
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLOR_TO_IDX, IDX_TO_COLOR
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper


def print_numeric_distribution(grid, goal_pos=None, lava_positions=None, title=None, summary=True):
    """
    Print the actual numeric probability distribution with formatting.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    title : str or None
        Optional title to display 
    summary : bool
        Whether to print summary statistics
    """
    if title:
        print(f"\n=== {title} ===")
    
    height, width = grid.shape
    
    if lava_positions is None:
        lava_positions = []
    
    print("\n  Probability Distribution Array:")
    
    # Print column headers
    header = "    "
    for x in range(width):
        header += f"{x:7d} "
    print(header)
    
    # Print horizontal line
    print("    " + "-" * (8 * width))
    
    # Format the probabilities with special symbols for goal and lava
    for y in range(height):
        row = f"{y:2d} |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position
                row += "  GOAL  "
            elif (x, y) in lava_positions:
                # Lava position
                row += "  LAVA  "
            else:
                # Regular cell with probability value
                prob = grid[y, x]
                row += f" {prob:.5f}"
        print(row)
    
    # Print legend
    print("\n  Legend:")
    print("  - GOAL: Goal position (zero probability)")
    print("  - LAVA: Lava cell (zero probability)")
    print("  - Values represent spawn probabilities (sum to 1.0)")
    
    # Print summary statistics
    if summary:
        nonzero_cells = np.count_nonzero(grid)
        total_cells = width * height
        print(f"\n  Summary:")
        print(f"  - Valid spawn cells: {nonzero_cells} of {total_cells} ({nonzero_cells/total_cells:.1%})")
        if nonzero_cells > 0:
            print(f"  - Highest probability: {np.max(grid):.5f}")
            print(f"  - Average probability (non-zero cells): {np.sum(grid)/nonzero_cells:.5f}")
        else:
            print("  - No valid spawn cells found!")


def plot_distribution(grid, goal_pos=None, lava_positions=None, title="Spawn Distribution", 
                      save_path=None, show=True):
    """
    Create a visual plot of the probability distribution.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    title : str
        Plot title
    save_path : str or None
        Path to save the visualization
    show : bool
        Whether to display the plot
    """
    height, width = grid.shape
    
    # Create a copy for plotting
    plot_grid = grid.copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(plot_grid, origin='upper', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Show coordinates
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    
    # Add markers for goal and lava
    if goal_pos:
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    
    if lava_positions and len(lava_positions) > 0:
        lava_x, lava_y = zip(*lava_positions)
        ax.scatter(lava_x, lava_y, c='red', marker='x', s=100, label='Lava')
    
    # Annotate cells with probability values
    for y in range(height):
        for x in range(width):
            # Skip cells with zero probability
            if plot_grid[y, x] > 0:
                text = ax.text(x, y, f"{plot_grid[y, x]:.4f}",
                            ha="center", va="center", color="w", fontsize=8)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Add legend if we have special markers
    if goal_pos or (lava_positions and len(lava_positions) > 0):
        ax.legend(loc='upper right')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def visualize_wrapper_distribution(wrapper, title="Current Spawn Distribution", 
                                   save_path=None, show=True, print_values=True):
    """
    Visualize the distribution from a FlexibleSpawnWrapper.
    
    Parameters:
    ----------
    wrapper : FlexibleSpawnWrapper
        The wrapper instance to visualize
    title : str
        Plot title
    save_path : str or None
        Path to save the visualization
    show : bool
        Whether to display the plot
    print_values : bool
        Whether to also print the numeric values
    """
    if not hasattr(wrapper, 'current_distribution') or wrapper.current_distribution is None:
        print("Error: Wrapper has no current_distribution")
        return
    
    # Get distribution grid
    grid = wrapper.current_distribution.probabilities
    
    # Get environment information
    goal_pos = getattr(wrapper, 'goal_pos', None)
    
    # Find lava positions by scanning the grid
    lava_positions = []
    if hasattr(wrapper, 'unwrapped') and hasattr(wrapper.unwrapped, 'grid'):
        grid_obj = wrapper.unwrapped.grid
        for i in range(grid_obj.width):
            for j in range(grid_obj.height):
                cell = grid_obj.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
    
    # Print numeric distribution if requested
    if print_values:
        print_numeric_distribution(grid, goal_pos, lava_positions, title=title)
    
    # Plot the distribution
    plot_distribution(grid, goal_pos, lava_positions, title=title, 
                     save_path=save_path, show=show)


def inspect_env_for_spawn_wrapper(env):
    """
    Recursively search for FlexibleSpawnWrapper in environment.
    
    Parameters:
    ----------
    env : gym.Env
        Environment to inspect
    
    Returns:
    -------
    wrapper : FlexibleSpawnWrapper or None
        Found wrapper or None if not found
    """
    # Try to import FlexibleSpawnWrapper
    try:
        from SpawnDistributions.spawn_distributions import FlexibleSpawnWrapper
    except ImportError:
        print("Warning: Cannot import FlexibleSpawnWrapper - visualization may be limited")
        return None
    
    # Check if current env is the wrapper
    if isinstance(env, FlexibleSpawnWrapper):
        return env
    
    # Check if env.env exists and recurse
    if hasattr(env, 'env'):
        return inspect_env_for_spawn_wrapper(env.env)
    
    # Try other common attributes
    if hasattr(env, 'venv'):
        return inspect_env_for_spawn_wrapper(env.venv)
    
    if hasattr(env, 'envs') and len(getattr(env, 'envs', [])) > 0:
        return inspect_env_for_spawn_wrapper(env.envs[0])
    
    # More aggressive search for deeply nested environments
    # Check for other possible attributes that might contain the environment
    for attr_name in dir(env):
        if attr_name.startswith('_'):
            continue  # Skip private attributes
            
        try:
            attr = getattr(env, attr_name)
            # Check if this attribute is an environment-like object
            if hasattr(attr, 'step') and hasattr(attr, 'reset'):
                # Recurse into this attribute
                result = inspect_env_for_spawn_wrapper(attr)
                if result is not None:
                    return result
                    
            # Check if it's a list or tuple of environments
            elif isinstance(attr, (list, tuple)) and len(attr) > 0:
                for item in attr:
                    if hasattr(item, 'step') and hasattr(item, 'reset'):
                        result = inspect_env_for_spawn_wrapper(item)
                        if result is not None:
                            return result
        except Exception:
            # Ignore any errors from accessing attributes
            pass
    
    # Not found
    return None


def visualize_env_spawn_distribution(env, title=None, save_dir=None, show=True):
    """
    Visualize spawn distribution for an environment.
    
    Parameters:
    ----------
    env : gym.Env
        Environment to visualize
    title : str or None
        Optional title for the visualization
    save_dir : str or None
        Directory to save visualization
    show : bool
        Whether to display the plot
    
    Returns:
    -------
    bool
        True if visualization was successful, False otherwise
    """
    # Find the spawn wrapper
    wrapper = inspect_env_for_spawn_wrapper(env)
    
    if wrapper is None:
        print("Could not find FlexibleSpawnWrapper in environment.")
        return False
    
    # Get current timestamp for unique filenames
    timestamp = int(time.time())
    
    # Create save path if a directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if title:
            # Create a filename-safe version of the title
            safe_title = "".join([c if c.isalnum() else "_" for c in title])
            filename = f"{safe_title}_{timestamp}.png"
        else:
            filename = f"spawn_distribution_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Visualize the distribution
    visualize_wrapper_distribution(
        wrapper, 
        title=title or "Current Spawn Distribution",
        save_path=save_path,
        show=show,
        print_values=True  # Always print numerical values
    )
    
    return True


def extract_distribution_from_env(env):
    """
    Extract the probability distribution array from an environment.
    
    Parameters:
    ----------
    env : gym.Env
        Environment to extract distribution from
    
    Returns:
    -------
    tuple
        (grid, goal_pos, lava_positions) or (None, None, None) if not found
    """
    # Find the spawn wrapper
    wrapper = inspect_env_for_spawn_wrapper(env)
    
    if wrapper is None:
        return None, None, None
    
    # Get distribution grid
    if not hasattr(wrapper, 'current_distribution') or wrapper.current_distribution is None:
        return None, None, None
        
    grid = wrapper.current_distribution.probabilities
    
    # Get environment information
    goal_pos = getattr(wrapper, 'goal_pos', None)
    
    # Find lava positions by scanning the grid
    lava_positions = []
    if hasattr(wrapper, 'unwrapped') and hasattr(wrapper.unwrapped, 'grid'):
        grid_obj = wrapper.unwrapped.grid
        for i in range(grid_obj.width):
            for j in range(grid_obj.height):
                cell = grid_obj.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
    
    return grid, goal_pos, lava_positions


class EnhancedSpawnDistributionCallback(BaseCallback):
    """
    Callback for visualizing spawn distributions during training.
    
    This enhanced version has improved ability to find the FlexibleSpawnWrapper
    in complex environment chains.
    """
    
    def __init__(self, verbose=0, vis_frequency=10000, vis_dir='./spawn_vis'):
        """
        Initialize the callback.
        
        Parameters:
        ----------
        verbose : int
            Verbosity level
        vis_frequency : int
            How often to visualize the distribution (in timesteps)
        vis_dir : str
            Directory to save visualizations
        """
        super().__init__(verbose)
        self.vis_frequency = vis_frequency
        self.vis_dir = vis_dir
        os.makedirs(vis_dir, exist_ok=True)
    
    def _on_step(self):
        """
        Called at each step of training.
        
        Returns:
        -------
        bool
            Whether training should continue
        """
        # Check if it's time to visualize
        if self.num_timesteps % self.vis_frequency == 0:
            # Get a reference to the actual environment
            # Try different attributes that might contain the environment
            env = None
            
            # For vectorized environments
            if hasattr(self.model, 'get_env'):
                env = self.model.get_env()
                
                # For vectorized envs, get the first sub-env
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    env = env.envs[0]
                elif hasattr(env, 'venv') and hasattr(env.venv, 'envs') and len(env.venv.envs) > 0:
                    env = env.venv.envs[0]
            
            # If we found an environment, visualize its distribution
            if env:
                title = f"Spawn Distribution at Step {self.num_timesteps:,}"
                success = visualize_env_spawn_distribution(
                    env, 
                    title=title,
                    save_dir=self.vis_dir,
                    show=False  # Don't show during training to avoid blocking
                )
                
                if success and self.verbose > 0:
                    print(f"\nSaved spawn distribution visualization at step {self.num_timesteps}")
        
        return True


def generate_ascii_visualization(grid, goal_pos=None, lava_positions=None, title=None):
    """
    Generate an ASCII visualization of a spawn distribution.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    title : str or None
        Optional title for the visualization
    
    Returns:
    -------
    str
        ASCII visualization
    """
    if lava_positions is None:
        lava_positions = []
    
    height, width = grid.shape
    
    # Define characters for different probability ranges
    # From lowest to highest probability
    chars = " .,:;+*#@"
    max_prob = np.max(grid) if np.max(grid) > 0 else 1.0
    
    # Build the ASCII visualization
    result = []
    
    if title:
        result.append(f"\n=== {title} ===\n")
    
    # Header row with column indices
    header = "    "
    for x in range(width):
        header += f"{x}"
    result.append(header)
    
    # Horizontal line
    result.append("   " + "-" * width)
    
    # Grid rows
    for y in range(height):
        row = f"{y:2d} |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position
                row += "G"
            elif (x, y) in lava_positions:
                # Lava position
                row += "L"
            else:
                # Regular cell with probability indicator
                prob = grid[y, x]
                if prob <= 0:
                    row += " "  # Zero probability
                else:
                    # Map probability to character
                    char_idx = min(int(prob / max_prob * (len(chars) - 1)), len(chars) - 1)
                    row += chars[char_idx]
        result.append(row)
    
    # Legend
    result.append("\nLegend:")
    result.append("G = Goal position (zero probability)")
    result.append("L = Lava cell (zero probability)")
    result.append(f"{chars} = Increasing probability (left to right)")
    
    return "\n".join(result)


if __name__ == "__main__":
    print("\n=== Spawn Distribution Inspector ===")
    print("This module is meant to be imported and used with actual environments.")
    print("Import and use the 'visualize_env_spawn_distribution' function to analyze any environment.")
    print("\nExample usage:")
    print("  from spawn_distribution_inspector import visualize_env_spawn_distribution")
    print("  # Create your environment")
    print("  env = make_env(...)")
    print("  # Visualize its spawn distribution")
    print("  visualize_env_spawn_distribution(env)") 
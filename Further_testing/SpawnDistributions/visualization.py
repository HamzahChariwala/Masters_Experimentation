"""
Visualization utilities for spawn distributions.

This module provides tools to visualize spawn distributions during training,
including enhanced callbacks that can find FlexibleSpawnWrapper in complex
environment chains.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import time

# Import the wrapper and DistributionMap from the new location
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper, DistributionMap

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
            Frequency of visualization (in timesteps)
        vis_dir : str
            Directory to save visualizations
        """
        super().__init__(verbose)
        self.vis_frequency = vis_frequency
        self.vis_dir = vis_dir
        
        # Create directory if it doesn't exist
        os.makedirs(vis_dir, exist_ok=True)
    
    def _on_step(self):
        """
        Called at each step during training.
        Only visualize at specified frequency.
        """
        if self.n_calls % self.vis_frequency == 0:
            self._visualize_spawn_distribution()
        return True
    
    def _recursively_find_wrapper(self, env, depth=0, max_depth=10):
        """
        Recursively search for FlexibleSpawnWrapper in a potentially nested environment.
        
        Parameters:
        ----------
        env : gym.Env
            Environment to search in
        depth : int
            Current recursion depth
        max_depth : int
            Maximum recursion depth to prevent infinite loops
            
        Returns:
        --------
        wrapper : FlexibleSpawnWrapper or None
            Found wrapper or None if not found
        """
        if depth > max_depth:
            return None
        
        # Check if this is the wrapper we're looking for
        if isinstance(env, FlexibleSpawnWrapper):
            return env
        
        # Try common ways to access sub-environments
        
        # 1. Standard env attribute (most gym wrappers)
        if hasattr(env, 'env'):
            wrapper = self._recursively_find_wrapper(env.env, depth + 1, max_depth)
            if wrapper is not None:
                return wrapper
                
        # 2. Vector environment's envs attribute (list of envs)
        if hasattr(env, 'envs') and isinstance(env.envs, (list, tuple)) and len(env.envs) > 0:
            # First try the first environment
            wrapper = self._recursively_find_wrapper(env.envs[0], depth + 1, max_depth)
            if wrapper is not None:
                return wrapper
                
            # If not found, try all other environments
            for sub_env in env.envs[1:]:
                wrapper = self._recursively_find_wrapper(sub_env, depth + 1, max_depth)
                if wrapper is not None:
                    return wrapper
                    
        # 3. Vector environment's venv attribute
        if hasattr(env, 'venv'):
            wrapper = self._recursively_find_wrapper(env.venv, depth + 1, max_depth)
            if wrapper is not None:
                return wrapper
                
        # 4. Special case for SB3 DummyVecEnv with unwrapped
        if hasattr(env, 'unwrapped'):
            wrapper = self._recursively_find_wrapper(env.unwrapped, depth + 1, max_depth)
            if wrapper is not None:
                return wrapper
                
        # 5. For SubprocVecEnv, we need to try a different approach
        if hasattr(env, 'remotes') and hasattr(env, 'waiting'):
            # This is likely a SubprocVecEnv, but we can't access its environments directly
            # We'll have to rely on the model's wrapped single env if available
            if self.verbose > 0:
                print("SubprocVecEnv detected - cannot directly access environments")
                
        return None
    
    def _deep_find_wrapper(self, env):
        """
        Comprehensive search for FlexibleSpawnWrapper with additional debugging.
        
        Parameters:
        ----------
        env : gym.Env
            Environment to search
            
        Returns:
        --------
        wrapper : FlexibleSpawnWrapper or None
            Found wrapper or None if not found
        """
        # First try the standard recursive search
        wrapper = self._recursively_find_wrapper(env)
        if wrapper is not None:
            return wrapper
            
        # If still not found, and we have a model with a single env attribute,
        # try to access that (some SB3 models have this)
        if hasattr(self.model, 'env'):
            if self.verbose > 0:
                print("Trying to find wrapper through model.env...")
            wrapper = self._recursively_find_wrapper(self.model.env)
            if wrapper is not None:
                return wrapper
                
        # If we have a model with get_env method (some SB3 models have this)
        if hasattr(self.model, 'get_env'):
            try:
                model_env = self.model.get_env()
                if self.verbose > 0:
                    print("Trying to find wrapper through model.get_env()...")
                wrapper = self._recursively_find_wrapper(model_env)
                if wrapper is not None:
                    return wrapper
            except:
                pass
                
        if self.verbose > 0:
            print("Could not find FlexibleSpawnWrapper in environment.")
        return None
    
    def _visualize_spawn_distribution(self):
        """
        Create and save a visualization of the current spawn distribution.
        """
        # Try to find the FlexibleSpawnWrapper in the environment
        wrapper = self._deep_find_wrapper(self.training_env)
        
        if wrapper is None:
            if self.verbose > 0:
                print("Could not find FlexibleSpawnWrapper in environment.")
            return
            
        # Extract spawn distribution data
        try:
            # Make sure we have a valid current_distribution
            if not hasattr(wrapper, 'current_distribution') or wrapper.current_distribution is None:
                if self.verbose > 0:
                    print("No current_distribution found in wrapper.")
                return
                
            # Get the probability grid
            grid = wrapper.current_distribution.probabilities
            
            # Get goal position if available
            goal_pos = None
            if hasattr(wrapper, 'goal_pos'):
                goal_pos = wrapper.goal_pos
                
            # Try to get lava positions if available
            lava_positions = []
            if hasattr(wrapper, 'unwrapped') and hasattr(wrapper.unwrapped, 'grid'):
                grid_obj = wrapper.unwrapped.grid
                for i in range(grid_obj.width):
                    for j in range(grid_obj.height):
                        cell = grid_obj.get(i, j)
                        if cell and cell.type == 'lava':
                            lava_positions.append((i, j))
                            
            # Create the visualization
            self._plot_distribution(
                grid, 
                goal_pos=goal_pos,
                lava_positions=lava_positions,
                timestep=self.n_calls
            )
            
            if self.verbose > 0:
                print(f"Created spawn distribution visualization at timestep {self.n_calls}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Error visualizing spawn distribution: {e}")
    
    def _plot_distribution(self, grid, goal_pos=None, lava_positions=None, timestep=0):
        """
        Plot and save spawn distribution visualization.
        
        Parameters:
        ----------
        grid : numpy.ndarray
            The probability distribution grid
        goal_pos : tuple or None
            Position of the goal (x, y), if available
        lava_positions : list or None
            List of lava positions [(x1, y1), (x2, y2), ...], if available
        timestep : int
            Current training timestep
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the distribution as a heatmap
        im = ax.imshow(grid, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Add grid
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add markers for goal and lava
        if goal_pos:
            ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=12, label='Goal')
            
        if lava_positions and len(lava_positions) > 0:
            lava_x, lava_y = zip(*lava_positions)
            ax.scatter(lava_x, lava_y, c='red', marker='x', s=80, label='Lava')
            
        # Add labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add distribution name and stage info if available
        title = f"Spawn Distribution (timestep: {timestep})"
        if hasattr(wrapper, 'distribution_type'):
            title = f"{wrapper.distribution_type.capitalize()} " + title
            
        if hasattr(wrapper, 'current_stage'):
            title += f" - Stage {wrapper.current_stage + 1}"
            
        ax.set_title(title)
        
        # Add a legend if we have markers
        if goal_pos or (lava_positions and len(lava_positions) > 0):
            ax.legend(loc='upper right')
            
        # Save the figure
        filename = os.path.join(self.vis_dir, f"spawn_dist_{timestep}.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)


class SpawnDistributionCallback(EnhancedSpawnDistributionCallback):
    """
    Legacy callback class for backward compatibility.
    Just an alias for the enhanced version.
    """
    pass


def generate_final_visualizations(model=None, spawn_vis_dir='./spawn_vis', use_stage_based_training=False, show=False):
    """
    Generate a set of final visualizations after training.
    This function creates more polished visualizations to summarize
    the training progress.
    
    Parameters:
    ----------
    model : stable_baselines3 model or None
        Trained model (for accessing the environment)
    spawn_vis_dir : str
        Directory containing spawn distribution visualizations
    use_stage_based_training : bool
        Whether stage-based training was used
    show : bool
        Whether to display the plots (True) or just save them (False)
    """
    print("\n====== GENERATING SPAWN DISTRIBUTION VISUALIZATIONS ======")
    
    # Find all visualization files
    if not os.path.exists(spawn_vis_dir):
        print(f"Visualization directory {spawn_vis_dir} not found.")
        print("==========================================================\n")
        return
        
    # Try to find the wrapper
    if model is not None:
        # Create a callback to reuse its wrapper finding logic
        callback = EnhancedSpawnDistributionCallback(verbose=1, vis_dir=spawn_vis_dir)
        callback.model = model  # Need to set this manually
        
        # Find the wrapper
        if hasattr(model, 'env'):
            wrapper = callback._deep_find_wrapper(model.env)
            
            if wrapper is not None:
                # Create final visualization
                try:
                    grid = wrapper.current_distribution.probabilities
                    
                    # Get goal position if available
                    goal_pos = None
                    if hasattr(wrapper, 'goal_pos'):
                        goal_pos = wrapper.goal_pos
                        
                    # Try to get lava positions if available
                    lava_positions = []
                    if hasattr(wrapper, 'unwrapped') and hasattr(wrapper.unwrapped, 'grid'):
                        grid_obj = wrapper.unwrapped.grid
                        for i in range(grid_obj.width):
                            for j in range(grid_obj.height):
                                cell = grid_obj.get(i, j)
                                if cell and cell.type == 'lava':
                                    lava_positions.append((i, j))
                                    
                    # Create a special final visualization
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot the distribution as a heatmap
                    im = ax.imshow(grid, cmap='viridis')
                    plt.colorbar(im, ax=ax, label='Probability')
                    
                    # Add grid
                    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
                    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.2)
                    
                    # Add markers for goal and lava
                    if goal_pos:
                        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
                        
                    if lava_positions and len(lava_positions) > 0:
                        lava_x, lava_y = zip(*lava_positions)
                        ax.scatter(lava_x, lava_y, c='red', marker='x', s=100, label='Lava')
                        
                    # Add text annotations for probabilities
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            if grid[i, j] > 0:
                                ax.text(j, i, f"{grid[i, j]:.4f}", 
                                        ha="center", va="center", 
                                        color="w", fontsize=8)
                    
                    # Add labels and title
                    ax.set_xlabel('X Coordinate')
                    ax.set_ylabel('Y Coordinate')
                    
                    # Create a descriptive title
                    title = "Final Spawn Distribution"
                    if hasattr(wrapper, 'distribution_type'):
                        title = f"{wrapper.distribution_type.capitalize()} {title}"
                        
                    if use_stage_based_training and hasattr(wrapper, 'current_stage'):
                        title += f" (Stage {wrapper.current_stage + 1})"
                        
                    ax.set_title(title, fontsize=14)
                    
                    # Add a legend if we have markers
                    if goal_pos or (lava_positions and len(lava_positions) > 0):
                        ax.legend(loc='upper right')
                        
                    # Save the figure
                    filename = os.path.join(spawn_vis_dir, "final_spawn_distribution.png")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
                        
                    print(f"Final visualization saved to {filename}")
                except Exception as e:
                    print(f"Error creating final visualization: {e}")
    
    print("==========================================================\n") 
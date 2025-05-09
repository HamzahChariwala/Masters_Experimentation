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

class SpawnDistributionCallback(BaseCallback):
    def __init__(self, env, save_dir=None, freq=10000, verbose=0):
        super(SpawnDistributionCallback, self).__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.freq = freq
        
        # Create the save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Record distribution at specific frequencies
        if self.num_timesteps % self.freq == 0:
            # Get the unwrapped environment
            if hasattr(self.env, 'venv'):
                # For vectorized environments, we need to get to a single env
                if hasattr(self.env.venv, 'envs'):
                    # DummyVecEnv case
                    env = self.env.venv.envs[0]
                else:
                    # Abstract VecEnv case - may need to be adapted
                    env = self.env.venv
            else:
                env = self.env
            
            # Try to find the FlexibleSpawnWrapper
            while env and not isinstance(env, FlexibleSpawnWrapper):
                if hasattr(env, 'env'):
                    env = env.env
                else:
                    # Couldn't find the wrapper
                    return True
            
            # If we found the wrapper, visualize the distribution
            if isinstance(env, FlexibleSpawnWrapper):
                if self.save_dir:
                    # Save the visualization to file
                    filename = f"spawn_dist_{self.num_timesteps}.png"
                    save_path = os.path.join(self.save_dir, filename)
                    env.visualize_distribution(
                        title=f"Spawn Distribution at Timestep {self.num_timesteps}",
                        save_path=save_path
                    )
                else:
                    # Just display the visualization (not ideal for headless operation)
                    env.visualize_distribution(
                        title=f"Spawn Distribution at Timestep {self.num_timesteps}"
                    )
        
        return True


class EnhancedSpawnDistributionCallback(BaseCallback):
    def __init__(self, env, save_dir=None, freq=10000, verbose=0):
        super(EnhancedSpawnDistributionCallback, self).__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.freq = freq
        
        # Create the save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.last_visualization = 0
    
    def _on_step(self) -> bool:
        # Only check at the specified frequency to save computation
        if self.num_timesteps - self.last_visualization >= self.freq:
            self.last_visualization = self.num_timesteps
            
            # Find all FlexibleSpawnWrapper instances in vectorized environments
            if hasattr(self.env, 'venv'):
                # For vectorized environments
                spawn_wrappers = []
                
                if hasattr(self.env.venv, 'envs'):
                    # DummyVecEnv case
                    for env in self.env.venv.envs:
                        wrapper = self._find_spawn_wrapper(env)
                        if wrapper:
                            spawn_wrappers.append(wrapper)
                else:
                    # Abstract VecEnv case
                    wrapper = self._find_spawn_wrapper(self.env.venv)
                    if wrapper:
                        spawn_wrappers.append(wrapper)
            else:
                # Non-vectorized environment
                wrapper = self._find_spawn_wrapper(self.env)
                spawn_wrappers = [wrapper] if wrapper else []
            
            # Visualize distributions for all found wrappers
            for i, wrapper in enumerate(spawn_wrappers):
                if self.save_dir:
                    # Save the visualization to file
                    filename = f"spawn_dist_{self.num_timesteps}_env{i}.png"
                    save_path = os.path.join(self.save_dir, filename)
                    wrapper.visualize_distribution(
                        title=f"Env {i} Spawn Distribution at Timestep {self.num_timesteps}",
                        save_path=save_path
                    )
                else:
                    # Just display the visualization
                    wrapper.visualize_distribution(
                        title=f"Env {i} Spawn Distribution at Timestep {self.num_timesteps}"
                    )
        
        return True
    
    def _find_spawn_wrapper(self, env):
        """Recursively find the FlexibleSpawnWrapper in a nested environment."""
        if isinstance(env, FlexibleSpawnWrapper):
            return env
        
        if hasattr(env, 'env'):
            return self._find_spawn_wrapper(env.env)
        
        return None


def generate_final_visualizations(env, save_dir=None):
    """
    Generate summary visualizations of spawn distribution evolution.
    
    Parameters:
    ----------
    env : gym.Env
        The environment containing FlexibleSpawnWrapper
    save_dir : str
        Directory to save visualizations
    """
    # Create the save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Find the FlexibleSpawnWrapper
    wrapper = None
    while env and not isinstance(env, FlexibleSpawnWrapper):
        if hasattr(env, 'env'):
            env = env.env
        else:
            # Couldn't find the wrapper
            if wrapper is None:
                print("Warning: Could not find FlexibleSpawnWrapper")
            return
    
    if not isinstance(env, FlexibleSpawnWrapper):
        print("Warning: Could not find FlexibleSpawnWrapper")
        return
    
    wrapper = env
    
    # Check if history is available (may be disabled for performance)
    history = wrapper.get_distribution_history()
    if history is None or not history:
        print("Warning: Distribution history is not available (disabled for performance)")
        
        # Just visualize the current distribution
        if save_dir:
            save_path = os.path.join(save_dir, "final_distribution.png")
            wrapper.visualize_distribution(
                title="Final Spawn Distribution",
                save_path=save_path
            )
        else:
            wrapper.visualize_distribution(title="Final Spawn Distribution")
        return
    
    # Create a figure for the visualization timeline
    fig, axes = plt.subplots(1, len(history), figsize=(15, 5))
    if len(history) == 1:
        axes = [axes]  # Make it indexable if only one subplot
    
    # Plot each distribution
    for i, (timestep, dist) in enumerate(history):
        ax = axes[i]
        im = ax.imshow(dist, cmap='hot', interpolation='nearest')
        ax.set_title(f"t={timestep}")
        ax.axis('off')
    
    # Add a colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label='Probability')
    
    # Set the main title
    fig.suptitle("Spawn Distribution Evolution", fontsize=16)
    
    # Save the figure if a directory is provided
    if save_dir:
        save_path = os.path.join(save_dir, "distribution_evolution.png")
        plt.savefig(save_path)
        print(f"Saved distribution evolution visualization to {save_path}")
    
    # Show the figure
    plt.tight_layout()
    plt.show() 
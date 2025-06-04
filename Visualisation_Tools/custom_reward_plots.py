#!/usr/bin/env python3
"""
Generate plots for the three custom reward functions implemented in the reward modification wrapper.
Each function gets its own plot with dotted reference lines at x=150 and y=1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma

def create_custom_reward_plots():
    """Create separate plots for each of the three custom reward functions."""
    
    # Parameters from the specific reward function config files
    x_intercept = 150  # Corrected from 100 to 150
    y_intercept = 1.0
    transition_width = 50  # Corrected from 10 to 50 for sigmoid function
    
    # Generate step values from 0 to 200 for plotting
    steps = np.linspace(0, 200, 1000)
    
    # Define the three reward functions as implemented in RewardModifications.py
    def linear_reward(num_steps):
        """Linear: reward = max(0, y_intercept - (y_intercept/x_intercept) * num_steps)"""
        slope = y_intercept / x_intercept
        return np.maximum(0, y_intercept - slope * num_steps)
    
    def exponential_reward(num_steps):
        """Exponential: reward = y_intercept / exp((-ln(0.01)/x_intercept) * num_steps)"""
        slope = -np.log(0.01) / x_intercept
        return y_intercept / np.exp(slope * num_steps)
    
    def sigmoid_reward(num_steps):
        """Sigmoid: reward = y_intercept / (1 + exp((4/transition_width) * (num_steps - x_intercept/2)))"""
        slope = 4 / transition_width
        shift = x_intercept / 2
        return y_intercept / (1 + np.exp(slope * (num_steps - shift)))
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "custom_rewards")
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot configurations
    reward_functions = [
        ('linear', linear_reward, magma(0.2)),
        ('exponential', exponential_reward, magma(0.5)),
        ('sigmoid', sigmoid_reward, magma(0.8))
    ]
    
    for func_name, func, color in reward_functions:
        # Calculate reward values
        rewards = func(steps)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot the reward function
        ax.plot(steps, rewards, color=color, linewidth=3)
        
        # Add dotted reference lines at x=150 and y=1
        ax.axvline(x=150, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Customize plot
        ax.set_xlabel('Number of Steps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
        ax.set_title(f'{func_name.capitalize()} Reward Function\n'
                    f'(x_intercept={x_intercept}, y_intercept={y_intercept}' +
                    (f', transition_width={transition_width}' if func_name == 'sigmoid' else '') + ')', 
                    fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        ax.set_xlim(0, 200)
        ax.set_ylim(0, max(1.2, np.max(rewards) * 1.1))
        
        # Save the plot
        save_path = os.path.join(save_dir, f"{func_name}_reward_function.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {func_name} reward function plot: {save_path}")
        
        # Print some key values for verification
        print(f"  At step 0: {func(0):.4f}")
        print(f"  At step {x_intercept}: {func(x_intercept):.4f}")
        print(f"  At step 150: {func(150):.4f}")
        print(f"  At step 200: {func(200):.4f}")
    
    print(f"\nAll custom reward function plots saved to: {save_dir}")
    print(f"Parameters used:")
    print(f"  x_intercept = {x_intercept}")
    print(f"  y_intercept = {y_intercept}")
    print(f"  transition_width = {transition_width} (sigmoid only)")


if __name__ == "__main__":
    print("Generating custom reward function plots...")
    print("Functions: Linear, Exponential, Sigmoid")
    print("Reference lines: x=150, y=1")
    print("Colors: magma colormap\n")
    
    create_custom_reward_plots()
    
    print("\nCustom reward function plots completed!") 
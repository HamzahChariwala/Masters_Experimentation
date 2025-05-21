import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from robust_mask_generation import RobustMaskGenerator
from minigrid.core.world_object import Wall, Goal, Lava

def create_test_environment():
    """Create a test environment with a specific layout"""
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RobustMaskGenerator(env, view_size=5)
    
    # Reset to initialize
    env.reset()
    base_env = env.unwrapped
    grid = base_env.grid
    
    # Clear the grid
    for i in range(base_env.width):
        for j in range(base_env.height):
            if (i == 0 or j == 0 or i == base_env.width-1 or j == base_env.height-1):
                # Keep the outer walls
                continue
            grid.set(i, j, None)
    
    # Create a test pattern
    # Create an L shape with walls
    for i in range(3):
        grid.set(2, i+2, Wall())  # Vertical wall
    for i in range(3):
        grid.set(i+2, 4, Wall())  # Horizontal wall
    
    # Add a goal
    grid.set(5, 1, Goal())
    
    # Add lava
    grid.set(1, 5, Lava())
    
    return env

def visualize_masks_for_all_directions():
    """Visualize masks for all 4 agent directions"""
    env = create_test_environment()
    base_env = env.unwrapped
    
    # Create a figure to compare masks in different directions
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    # Direction names
    directions = ["Right (0)", "Down (1)", "Left (2)", "Up (3)"]
    
    # Position the agent at the center
    agent_pos = np.array([3, 3])
    
    # Test each direction
    for dir_idx, direction_name in enumerate(directions):
        # Position agent
        base_env.agent_pos = agent_pos.copy()
        base_env.agent_dir = dir_idx
        
        # Get observation
        obs, _, _, _, _ = env.step(0)  # Take a no-op action to refresh observation
        
        # Extract masks
        wall_mask = obs['wall_mask']
        lava_mask = obs['lava_mask']
        goal_mask = obs['goal_mask']
        
        # Draw the world grid
        grid_vis = np.ones((base_env.height, base_env.width, 3))  # White background
        
        # Draw objects on grid
        for i in range(base_env.width):
            for j in range(base_env.height):
                cell = base_env.grid.get(i, j)
                if cell:
                    if cell.type == 'wall':
                        grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray for walls
                    elif cell.type == 'goal':
                        grid_vis[j, i] = [0, 1, 0]  # Green for goal
                    elif cell.type == 'lava':
                        grid_vis[j, i] = [1, 0, 0]  # Red for lava
        
        # Mark agent position and direction
        agent_x, agent_y = base_env.agent_pos
        grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue for agent
        
        # Add direction arrow
        dir_arrows = ["→", "↓", "←", "↑"]
        
        # Plot grid
        axes[dir_idx, 0].imshow(grid_vis)
        axes[dir_idx, 0].set_title(f'Grid - Agent Direction: {direction_name}')
        
        # Add agent direction arrow
        axes[dir_idx, 0].text(agent_x, agent_y, dir_arrows[dir_idx], 
                             ha='center', va='center', color='white',
                             fontsize=20, fontweight='bold')
        
        # Plot combined mask
        combined_mask = np.zeros((5, 5, 3))
        combined_mask[:, :, 0] = lava_mask  # Red channel
        combined_mask[:, :, 1] = goal_mask  # Green channel
        combined_mask[:, :, 2] = wall_mask * 0.5  # Blue channel
        
        # Mark agent position in mask (always at center)
        center = 2
        combined_mask[center, center, :] = [0, 0, 1]  # Blue for agent
        
        axes[dir_idx, 1].imshow(combined_mask)
        axes[dir_idx, 1].set_title(f'Mask - Agent Direction: {direction_name}')
        
        # Plot mask with annotations
        annotated_mask = np.zeros((5, 5, 3))
        annotated_mask[:, :, 0] = lava_mask  # Red channel
        annotated_mask[:, :, 1] = goal_mask  # Green channel
        annotated_mask[:, :, 2] = wall_mask * 0.5  # Blue channel
        
        axes[dir_idx, 2].imshow(annotated_mask)
        axes[dir_idx, 2].set_title(f'Annotated Mask - Direction: {direction_name}')
        
        # Add agent at center
        axes[dir_idx, 2].text(center, center, "A", 
                             ha='center', va='center', color='white',
                             fontsize=15, fontweight='bold')
        
        # Add annotations for walls, goals, lava
        for i in range(5):
            for j in range(5):
                if wall_mask[i, j] > 0:
                    axes[dir_idx, 2].text(j, i, "W", 
                                        ha='center', va='center', color='white',
                                        fontsize=10)
                if goal_mask[i, j] > 0:
                    axes[dir_idx, 2].text(j, i, "G", 
                                        ha='center', va='center', color='black',
                                        fontsize=10, fontweight='bold')
                if lava_mask[i, j] > 0:
                    axes[dir_idx, 2].text(j, i, "L", 
                                        ha='center', va='center', color='white',
                                        fontsize=10, fontweight='bold')
        
        # Add coordinate grid labels
        for i in range(5):
            for j in range(5):
                if not (i == center and j == center):  # Skip agent position
                    axes[dir_idx, 2].text(j, i+0.3, f"({j-center},{i-center})", 
                                        ha='center', va='center', color='gray',
                                        fontsize=7)
    
    # Clean up plot appearance
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('robust_mask_visualization.png')
    plt.close()
    
    print("Visualization saved to robust_mask_visualization.png")

if __name__ == "__main__":
    visualize_masks_for_all_directions() 
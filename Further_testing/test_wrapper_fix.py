import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from Environment_Tooling.BespokeEdits.CustomWrappers import PartialObsWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from minigrid.core.world_object import Wall, Goal, Lava

def test_fixed_rotation():
    """Test that the fixed rotation in PartialObsWrapper works correctly"""
    # Create environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = CustomActionWrapper(env)
    env = PartialObsWrapper(env, 5)
    
    # Reset and setup test grid
    env.reset()
    base_env = env.unwrapped
    grid = base_env.grid
    
    # Clear most of the grid except outer walls
    for i in range(base_env.width):
        for j in range(base_env.height):
            if (i == 0 or j == 0 or i == base_env.width-1 or j == base_env.height-1):
                continue
            grid.set(i, j, None)
    
    # Create recognizable patterns for each direction
    # For right (dir 0): create a horizontal line in front of agent
    for i in range(3):
        grid.set(4+i, 3, Wall())
    
    # Add corner objects for reference
    grid.set(6, 1, Goal())  # Top-right
    grid.set(6, 6, Lava())  # Bottom-right
    grid.set(1, 6, Lava())  # Bottom-left
    grid.set(1, 1, Goal())  # Top-left
    
    # Set up figure for visualization
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    directions = ["Right (0)", "Down (1)", "Left (2)", "Up (3)"]
    dir_arrows = ["→", "↓", "←", "↑"]
    
    # Test each direction
    for dir_idx in range(4):
        # Set agent position and direction
        base_env.agent_pos = np.array([3, 3])
        base_env.agent_dir = dir_idx
        
        # Get observation
        obs, _, _, _, _ = env.step(0)  # No-op to refresh
        
        # Extract masks
        wall_mask = obs['wall_mask']
        goal_mask = obs['goal_mask']
        lava_mask = obs['lava_mask']
        
        # Create grid visualization
        grid_vis = np.ones((base_env.height, base_env.width, 3))  # White
        
        # Draw grid objects
        for i in range(base_env.width):
            for j in range(base_env.height):
                cell = base_env.grid.get(i, j)
                if cell:
                    if cell.type == 'wall':
                        grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray
                    elif cell.type == 'goal':
                        grid_vis[j, i] = [0, 1, 0]  # Green
                    elif cell.type == 'lava':
                        grid_vis[j, i] = [1, 0, 0]  # Red
        
        # Mark agent position and direction
        agent_x, agent_y = base_env.agent_pos
        grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue for agent
        
        # Draw grid
        axes[dir_idx, 0].imshow(grid_vis)
        axes[dir_idx, 0].set_title(f'Grid - Direction: {directions[dir_idx]}')
        
        # Add direction arrow
        axes[dir_idx, 0].text(agent_x, agent_y, dir_arrows[dir_idx], 
                            ha='center', va='center', color='white',
                            fontsize=20, fontweight='bold')
                            
        # Add grid coordinates for reference
        for i in range(base_env.width):
            for j in range(base_env.height):
                axes[dir_idx, 0].text(i, j, f"({i},{j})", 
                                    ha='center', va='center', color='black',
                                    fontsize=6)
        
        # Create combined mask visualization
        combined = np.zeros((wall_mask.shape[0], wall_mask.shape[1], 3))
        combined[:, :, 0] = lava_mask  # Red channel
        combined[:, :, 1] = goal_mask  # Green channel
        combined[:, :, 2] = wall_mask * 0.5  # Blue channel
        
        # Draw agent view position
        combined[env.agent_view_pos[0], env.agent_view_pos[1], :] = [1, 1, 1]  # White for agent view pos
        
        axes[dir_idx, 1].imshow(combined)
        axes[dir_idx, 1].set_title(f'Mask - Direction: {directions[dir_idx]}')
        
        # Add mask coordinates
        for i in range(wall_mask.shape[0]):
            for j in range(wall_mask.shape[1]):
                axes[dir_idx, 1].text(j, i, f"({j},{i})", 
                                    ha='center', va='center', color='white',
                                    fontsize=6)
                
                # Add labels for visible objects
                if wall_mask[i, j] > 0:
                    axes[dir_idx, 1].text(j, i+0.3, "W", 
                                        ha='center', va='center', color='cyan',
                                        fontsize=8, fontweight='bold')
                if goal_mask[i, j] > 0:
                    axes[dir_idx, 1].text(j, i+0.3, "G", 
                                        ha='center', va='center', color='lime',
                                        fontsize=8, fontweight='bold')
                if lava_mask[i, j] > 0:
                    axes[dir_idx, 1].text(j, i+0.3, "L", 
                                        ha='center', va='center', color='red',
                                        fontsize=8, fontweight='bold')
    
    # Clean up plot
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('fixed_rotation_test.png')
    plt.close()
    print("Test completed. Check fixed_rotation_test.png to verify correct rotation.")

if __name__ == "__main__":
    test_fixed_rotation() 
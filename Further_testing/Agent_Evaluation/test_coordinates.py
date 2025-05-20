import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

# Import from our Environment_Tooling modules - using proper paths
from Environment_Tooling.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                          PartialObsWrapper, 
                                          ExtractAbstractGrid, 
                                          PartialRGBObsWrapper, 
                                          PartialGrayObsWrapper, 
                                          ForceFloat32,
                                          RandomSpawnWrapper,
                                          DiagonalMoveMonitor)
from Environment_Tooling.BespokeEdits.RewardModifications import EpisodeCompletionRewardWrapper, LavaStepCounterWrapper
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from Environment_Tooling.BespokeEdits.GymCompatibility import OldGymCompatibility
from Environment_Tooling.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper

# Import local position override functionality
from Agent_Evaluation.EnvironmentTooling.position_override import ForceStartState
from Agent_Evaluation.EnvironmentTooling.import_vars import extract_env_config, load_config
from Agent_Evaluation.EnvironmentTooling.extract_grid import print_env_tensor

def test_coordinates(env_id="MiniGrid-LavaCrossingS11N5-v0", seed=81102):
    """
    Test the coordinate system in MiniGrid environments.
    
    Args:
        env_id (str): Environment ID to use for testing
        seed (int): Random seed for reproducibility
    """
    print(f"\nTesting coordinate system for environment: {env_id}")
    print(f"Using seed: {seed}")
    
    # Create environment directly with gym.make
    env = gym.make(env_id, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    
    # Apply partial observation for windowed view
    env = PartialObsWrapper(env, 7)
    env = ExtractAbstractGrid(env)
    
    # Add selective observation wrapper to enable log_data
    env = SelectiveObservationWrapper(env, cnn_keys=[], mlp_keys=['lava_mask', 'new_image'])
    print("Added SelectiveObservationWrapper to include log_data")
    
    # Add custom action wrapper for diagonal moves
    env = CustomActionWrapper(env)
    print("Added CustomActionWrapper for diagonal moves")
    
    # Wrap with ForceStartState
    env = ForceStartState(env)
    
    # Get unwrapped environment for direct access
    base = env.unwrapped
    
    # Print grid dimensions
    print(f"Grid dimensions: width={base.width}, height={base.height}")
    
    # Test multiple positions
    test_positions = [
        (4, 3, 0),  # (x, y, orientation)
        (3, 4, 0),
        (5, 2, 1),
        (2, 5, 2),
    ]
    
    for pos in test_positions:
        x, y, ori = pos
        
        # Skip if position is outside grid boundaries
        if x >= base.width or y >= base.height:
            print(f"Position {pos} is outside grid boundaries, skipping")
            continue
            
        print(f"\n------- Testing position ({x}, {y}, {ori}) -------")
        
        # Force agent to position
        env.force((x, y), ori)
        
        # Reset environment with seed
        obs, info = env.reset(seed=seed)
        
        print("\nInfo dictionary contents:")
        print(f"Info type: {type(info)}")
        
        if not info:
            print("Info is empty")
        else:
            print(f"Info has {len(info)} keys")
            
            # Print all keys in info dictionary
            print("Keys in info:", list(info.keys()))
            
            # Try to access and print items safely
            for k, v in info.items():
                print(f"Key: {k}")
                print(f"Value type: {type(v)}")
                try:
                    if isinstance(v, dict):
                        print(f"Nested keys: {list(v.keys())}")
                    elif isinstance(v, (list, tuple)):
                        print(f"Length: {len(v)}")
                    else:
                        print(f"Value: {v}")
                except Exception as e:
                    print(f"Error displaying value: {e}")
        
        # Check if log_data exists and print its contents
        if 'log_data' in info:
            print("\nlog_data contents:")
            for k, v in info['log_data'].items():
                print(f"  {k}: {type(v)}")
                if k == 'lava_mask' and hasattr(v, 'shape'):
                    print(f"  lava_mask shape: {v.shape}")
                    
                    # Visualize the lava_mask alongside the coordinate system
                    print("\nLava mask visualization (centered on agent):")
                    lava_mask = info['log_data']['lava_mask']
                    
                    # Print x-coordinates for reference
                    print("  " + " ".join([f"{j}" for j in range(lava_mask.shape[1])]))
                    
                    # Convert mask to ASCII representation
                    for i in range(lava_mask.shape[0]):
                        row = ""
                        for j in range(lava_mask.shape[1]):
                            if i == lava_mask.shape[0]//2 and j == lava_mask.shape[1]//2:
                                row += "A "  # Agent position
                            elif lava_mask[i, j] > 0:
                                row += "L "  # Lava
                            else:
                                row += ". "  # Empty space
                        print(f"{i} {row}")
                    
                    print("^ y-coordinate in lava_mask")
                    print("  x-coordinate in lava_mask ->")
                    print(f"Agent is at center position ({lava_mask.shape[1]//2}, {lava_mask.shape[0]//2})")
        
        # Get and print agent position from environment
        agent_x, agent_y = base.agent_pos
        agent_dir = base.agent_dir
        print(f"Environment reports agent at: ({agent_x}, {agent_y}, {agent_dir})")
        
        # Compare with expected position
        if (agent_x, agent_y, agent_dir) == (x, y, ori):
            print("✅ Position matches expected coordinates")
        else:
            print("❌ Position DOES NOT match expected coordinates")
            print(f"Expected: ({x}, {y}, {ori})")
            print(f"Actual: ({agent_x}, {agent_y}, {agent_dir})")
        
        # Test all possible actions (including diagonals)
        print("\nTesting actions from this position:")
        actions = [0, 1, 2, 3, 4]  # All possible actions with CustomActionWrapper
        action_names = ["right", "down", "left", "diagonal_left", "diagonal_right"]
        
        for action in [3, 4]:  # Focus on testing diagonal actions
            # Force agent back to starting position
            env.force((x, y), ori)
            obs, _ = env.reset(seed=seed)
            
            # Get position before action
            before_x, before_y = base.agent_pos
            before_dir = base.agent_dir
            
            # Take action and get new position
            obs, reward, term, trunc, info = env.step(action)
            
            # Get position after action
            after_x, after_y = base.agent_pos
            after_dir = base.agent_dir
            
            print(f"  Action {action} ({action_names[action]}): ({before_x}, {before_y}, {before_dir}) → ({after_x}, {after_y}, {after_dir})")
            
            # Save visualization of the result
            img = env.get_frame()
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"After action {action} ({action_names[action]})")
            plt.savefig(f"action_{action}_from_{x}_{y}_{ori}.png")
        
        # Render environment
        env.force((x, y), ori)
        obs, _ = env.reset(seed=seed)
        img = env.get_frame()
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Agent at reported position: ({agent_x}, {agent_y}, {agent_dir})")
        plt.savefig(f"agent_at_{agent_x}_{agent_y}_{agent_dir}.png")
        print(f"Saved visualization to agent_at_{agent_x}_{agent_y}_{agent_dir}.png")
        
        # Print a simple ASCII grid for immediate verification
        # Create a grid representation
        grid = np.full((base.height, base.width), ' ')
        
        # Mark walls
        for i in range(base.height):
            for j in range(base.width):
                if base.grid.get(j, i) is not None:
                    obj = base.grid.get(j, i)
                    if obj.type == 'wall':
                        grid[i, j] = '#'
                    elif obj.type == 'lava':
                        grid[i, j] = 'L'
                    elif obj.type == 'goal':
                        grid[i, j] = 'G'
        
        # Mark agent
        agent_symbol = ['→', '↓', '←', '↑'][agent_dir]  # Direction symbols
        grid[agent_y, agent_x] = agent_symbol
        
        # Mark coordinate system axes
        print("\nASCII Grid Representation (with coordinates):")
        print("  " + " ".join([f"{j}" for j in range(min(10, base.width))]))  # x-coordinates
        for i in range(base.height):
            row = "".join(grid[i, :])
            print(f"{i} {row}")
        print("^ y-coordinate")
        print("  x-coordinate ->")
    
    # Close environment
    env.close()
    print("\nCoordinate testing complete")

if __name__ == "__main__":
    test_coordinates() 
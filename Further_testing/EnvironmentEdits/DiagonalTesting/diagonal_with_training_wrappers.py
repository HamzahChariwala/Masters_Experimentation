import numpy as np
import gymnasium as gym
import time
import os
import sys
import matplotlib.pyplot as plt
import json

# Add the parent directory to the system path for imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import (
    GoalAngleDistanceWrapper, 
    PartialObsWrapper, 
    ExtractAbstractGrid, 
    PartialRGBObsWrapper, 
    PartialGrayObsWrapper, 
    ForceFloat32,
    DiagonalMoveMonitor
)
from EnvironmentEdits.BespokeEdits.FeatureExtractor import SelectiveObservationWrapper
from EnvironmentEdits.BespokeEdits.GymCompatibility import OldGymCompatibility
from minigrid.wrappers import FullyObsWrapper
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_agent_info(env):
    """
    Helper function to safely get agent position and direction by unwrapping.
    This is similar to the implementation in Tooling.py.
    """
    # Start with the environment
    current_env = env
    
    # Keep unwrapping until we find agent_pos or run out of unwrapped envs
    while hasattr(current_env, 'unwrapped'):
        # Check if current level has the attributes
        if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
            return current_env.agent_pos, current_env.agent_dir
        
        # Move to the next unwrapped level
        current_env = current_env.unwrapped
        
        # Check again at this level
        if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
            return current_env.agent_pos, current_env.agent_dir
    
    # If we got here, we couldn't find the attributes
    return None, None

def test_diagonal_with_training_wrappers():
    """
    Test diagonal movement using the EXACT SAME wrappers as used in training.
    This ensures we're testing under the same conditions as the actual training environment.
    """
    # Create results directory
    results_dir = "training_wrappers_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Use the same observation parameters as in training
    observation_params = {
        "window_size": 7,
        "cnn_keys": [],
        "mlp_keys": ["four_way_goal_direction",
                    "four_way_angle_alignment",
                    "barrier_mask",
                    "lava_mask"],
        "diagonal_success_reward": 0.5,
        "diagonal_failure_penalty": 0.1
    }
    
    # Action mappings
    actions = {
        "turn_left": 0,
        "turn_right": 1,
        "forward": 2,
        "diagonal_left": 3,
        "diagonal_right": 4
    }
    
    # Direction names
    dir_names = {
        0: "Right",
        1: "Down",
        2: "Left",
        3: "Up"
    }
    
    # Create environment with the EXACT SAME wrapper stack as in training
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    
    # Apply the training wrappers in the same order
    env = CustomActionWrapper(env, 
                             diagonal_success_reward=observation_params["diagonal_success_reward"],
                             diagonal_failure_penalty=observation_params["diagonal_failure_penalty"])
    env = DiagonalMoveMonitor(env)
    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    env = PartialObsWrapper(env, observation_params.get("window_size", 7))
    env = ExtractAbstractGrid(env)
    env = PartialRGBObsWrapper(env, observation_params.get("window_size", 7))
    env = PartialGrayObsWrapper(env, observation_params.get("window_size", 7))
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=observation_params.get("cnn_keys", []),
        mlp_keys=observation_params.get("mlp_keys", [])
    )
    env = ForceFloat32(env)
    env = OldGymCompatibility(env)
    env = Monitor(env)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Print observation structure
    print("Observation structure with training wrappers:")
    print(f"Observation type: {type(obs)}")
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dictionary'}")
    for key in obs.keys() if isinstance(obs, dict) else []:
        if isinstance(obs[key], np.ndarray):
            print(f"  - {key}: {type(obs[key])} shape={obs[key].shape}, dtype={obs[key].dtype}")
        else:
            print(f"  - {key}: {type(obs[key])}")
    
    # Get initial agent position using helper function
    agent_pos, agent_dir = get_agent_info(env)
    print(f"Initial agent position: {agent_pos}, direction: {dir_names[agent_dir] if agent_dir is not None else 'Unknown'}")
    
    # Save observation structure to a file
    with open(os.path.join(results_dir, "observation_structure.txt"), "w") as f:
        f.write("Observation structure with training wrappers:\n")
        f.write(f"Observation type: {type(obs)}\n")
        f.write(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dictionary'}\n")
        for key in obs.keys() if isinstance(obs, dict) else []:
            if isinstance(obs[key], np.ndarray):
                f.write(f"  - {key}: {type(obs[key])} shape={obs[key].shape}, dtype={obs[key].dtype}\n")
            else:
                f.write(f"  - {key}: {type(obs[key])}\n")
        
        if agent_pos is not None and agent_dir is not None:
            f.write(f"\nInitial agent position: {agent_pos}, direction: {dir_names[agent_dir]}\n")
    
    # Store test results
    all_results = {}
    
    # Test sequence: move forward, then diagonal
    test_sequence = [
        {"name": "initial", "action": None},
        {"name": "forward_1", "action": "forward"},
        {"name": "forward_2", "action": "forward"},
        {"name": "diagonal_left", "action": "diagonal_left"},
        {"name": "diagonal_right", "action": "diagonal_right"}
    ]
    
    print("\n=== Testing diagonal moves with training wrappers ===")
    
    # Reset for fresh state
    obs, info = env.reset(seed=42)
    sequence_results = []
    
    for i, step in enumerate(test_sequence):
        # Get agent position using helper function
        agent_pos, agent_dir = get_agent_info(env)
        
        if step["action"] is None:
            # Initial state
            print(f"\nStep {i}: Initial state")
            print(f"  Position: {agent_pos}, Direction: {dir_names[agent_dir] if agent_dir is not None else 'Unknown'}")
            action_taken = "initial"
        else:
            # Take action
            print(f"\nStep {i}: Taking action {step['action']}")
            old_pos = [int(p) for p in agent_pos] if agent_pos is not None else None
            old_dir = int(agent_dir) if agent_dir is not None else None
            
            action_taken = step["action"]
            obs, reward, terminated, truncated, info = env.step(actions[action_taken])
            
            # Get updated position
            agent_pos, agent_dir = get_agent_info(env)
            new_pos = [int(p) for p in agent_pos] if agent_pos is not None else None
            new_dir = int(agent_dir) if agent_dir is not None else None
            
            print(f"  Position: {old_pos} → {new_pos}")
            print(f"  Direction: {dir_names[old_dir] if old_dir is not None else 'Unknown'} → {dir_names[new_dir] if new_dir is not None else 'Unknown'}")
            print(f"  Reward: {reward}")
            
            if action_taken in ["diagonal_left", "diagonal_right"]:
                success = "failed" not in info if "action" in info and info["action"] == "diagonal" else False
                result = "SUCCESS" if success else "FAILED"
                print(f"  Diagonal move {result}")
                if "action" in info:
                    print(f"  Info action: {info['action']}")
                if success and new_pos is not None and old_pos is not None:
                    delta = [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]]
                    print(f"  Position delta: {delta}")
        
        # Record data for this step
        step_data = {
            "step": i,
            "action": action_taken,
        }
        
        # Add agent position and direction if available
        if agent_pos is not None and agent_dir is not None:
            step_data["agent_pos"] = [int(p) for p in agent_pos]
            step_data["agent_dir"] = int(agent_dir)
            step_data["dir_name"] = dir_names[agent_dir]
        
        # Add info data for diagonal moves
        if "action" in info and info["action"] == "diagonal":
            step_data["diagonal_info"] = {
                "action": info["action"],
                "diag_direction": info.get("diag_direction", "unknown"),
                "failed": info.get("failed", False)
            }
        
        # Save each observation key's content
        if isinstance(obs, dict):
            step_data["obs_keys"] = list(obs.keys())
            
            for key in obs.keys():
                if isinstance(obs[key], np.ndarray):
                    # Save stats for large arrays instead of full content
                    if obs[key].size > 100:
                        step_data[f"obs_{key}_shape"] = list(obs[key].shape)
                        step_data[f"obs_{key}_min"] = float(np.min(obs[key]))
                        step_data[f"obs_{key}_max"] = float(np.max(obs[key]))
                        step_data[f"obs_{key}_mean"] = float(np.mean(obs[key]))
                        
                        # Save visualization for certain observation types
                        if key in ["gray_partial_obs", "rgb_partial_obs", "abstract_grid"]:
                            plt.figure(figsize=(8, 8))
                            if len(obs[key].shape) == 3 and obs[key].shape[2] == 3:  # RGB
                                plt.imshow(obs[key])
                            else:  # Grayscale or 2D
                                if len(obs[key].shape) == 3:
                                    # Use first channel if multiple channels
                                    plt.imshow(obs[key][:,:,0], cmap='gray')
                                else:
                                    plt.imshow(obs[key], cmap='gray')
                            plt.colorbar()
                            plt.title(f"Step {i}: {action_taken} - {key}")
                            plt.savefig(os.path.join(results_dir, f"step{i}_{action_taken}_{key}.png"))
                            plt.close()
                    else:
                        # Save small arrays directly
                        step_data[f"obs_{key}"] = obs[key].tolist()
        
        sequence_results.append(step_data)
        time.sleep(0.5)
    
    # Store results
    all_results["training_wrappers_test"] = sequence_results
    
    # Save all results to a JSON file
    with open(os.path.join(results_dir, "training_wrappers_test.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nTest complete! All results saved to training_wrappers_test_results/")
    env.close()

if __name__ == "__main__":
    test_diagonal_with_training_wrappers() 
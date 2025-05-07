import numpy as np
import gymnasium as gym
import time
import os
import sys
import matplotlib.pyplot as plt
import json

# Add the parent directory to the system path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import DiagonalMoveMonitor
from minigrid.wrappers import FullyObsWrapper

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

def test_diagonal_grid_state():
    """
    Test diagonal movement and inspect the internal grid state to verify
    if diagonal moves are properly represented in the agent's observations.
    """
    # Create results directory
    result_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(result_dir, "grid_states"), exist_ok=True)
    
    # Create environment with all standard wrappers
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    env = FullyObsWrapper(env)
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    
    # Dictionary to map direction indices to names
    dir_names = {
        0: "Right",
        1: "Down",
        2: "Left",
        3: "Up"
    }
    
    # Action mappings
    actions = {
        "turn_left": 0,
        "turn_right": 1,
        "forward": 2,
        "diagonal_left": 3,
        "diagonal_right": 4
    }
    
    # Dictionary to store all observation data
    all_observations = {}
    
    # Dump detailed observation information to a file
    with open(os.path.join(result_dir, "observation_keys.txt"), "w") as f:
        f.write("Initial observation keys and structures:\n\n")
    
    # Test each direction with its diagonal moves
    for direction_name, direction in [("right", 0), ("down", 1), ("left", 2), ("up", 3)]:
        print(f"\n=== Testing diagonal moves when facing {direction_name} ===")
        
        # Reset environment and get initial observation
        obs, info = env.reset(seed=42)
        
        # Save observation structure information
        with open(os.path.join(result_dir, "observation_keys.txt"), "a") as f:
            f.write(f"\n\nDirection: {direction_name}\n")
            f.write(f"Observation keys: {list(obs.keys())}\n")
            for key in obs.keys():
                f.write(f"  - {key}: {type(obs[key])} {obs[key].shape if hasattr(obs[key], 'shape') else ''}\n")
        
        # Save the full observation details
        test_data = {
            "direction": direction_name,
            "steps": []
        }
        
        # Record initial state
        step_data = {
            "step": 0,
            "action": "initial",
            "agent_pos": [int(p) for p in env.agent_pos],  # Convert to standard Python int
            "agent_dir": int(env.agent_dir),
            "dir_name": dir_names[env.agent_dir],
        }
        
        # Save grid if available
        if hasattr(env.unwrapped, "grid"):
            grid_str = str(env.unwrapped.grid)
            step_data["grid"] = grid_str
            print(f"Initial grid state:\n{grid_str}")
        
        # Save observation details
        step_data["has_image"] = "image" in obs
        if "image" in obs:
            # Save agent position in the image (non-zero values)
            image = obs["image"]
            agent_positions = np.where(image > 0)
            if len(agent_positions[0]) > 0:
                step_data["agent_in_image"] = [
                    int(agent_positions[0][0]),
                    int(agent_positions[1][0]),
                    int(image[agent_positions][0])
                ]
                print(f"Agent in image: position={step_data['agent_in_image'][:2]}, value={step_data['agent_in_image'][2]}")
            else:
                step_data["agent_in_image"] = [-1, -1, -1]
                print("Agent not found in the image")
            
            # Create visualization of the image
            plt.figure(figsize=(8, 8))
            plt.imshow(image[:,:,0], cmap='tab20')
            plt.colorbar(label='Object/Agent ID')
            plt.title(f"Grid state - {direction_name} - Step {step_data['step']}")
            plt.savefig(os.path.join(result_dir, f"grid_states/{direction_name}_step{step_data['step']}.png"))
            plt.close()
            
        test_data["steps"].append(step_data)
        
        # Turn to face the correct direction
        step_count = 1
        while env.agent_dir != direction:
            print(f"Turning right to face {dir_names[direction]}")
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            
            # Record step data
            step_data = {
                "step": step_count,
                "action": "turn_right",
                "agent_pos": [int(p) for p in env.agent_pos],
                "agent_dir": int(env.agent_dir),
                "dir_name": dir_names[env.agent_dir],
            }
            
            # Process grid and image as before
            if hasattr(env.unwrapped, "grid"):
                grid_str = str(env.unwrapped.grid)
                step_data["grid"] = grid_str
            
            step_data["has_image"] = "image" in obs
            if "image" in obs:
                image = obs["image"]
                agent_positions = np.where(image > 0)
                if len(agent_positions[0]) > 0:
                    step_data["agent_in_image"] = [
                        int(agent_positions[0][0]),
                        int(agent_positions[1][0]),
                        int(image[agent_positions][0])
                    ]
                else:
                    step_data["agent_in_image"] = [-1, -1, -1]
                
                # Create visualization
                plt.figure(figsize=(8, 8))
                plt.imshow(image[:,:,0], cmap='tab20')
                plt.colorbar(label='Object/Agent ID')
                plt.title(f"Grid state - {direction_name} - Step {step_data['step']}")
                plt.savefig(os.path.join(result_dir, f"grid_states/{direction_name}_step{step_data['step']}.png"))
                plt.close()
                
            test_data["steps"].append(step_data)
            step_count += 1
            time.sleep(0.3)
        
        # Now try diagonal moves
        for move_type in ["diagonal_left", "diagonal_right"]:
            print(f"\nTesting {move_type} movement when facing {dir_names[direction]}")
            
            # Try the diagonal move
            old_pos = [int(p) for p in env.agent_pos]
            obs, reward, terminated, truncated, info = env.step(actions[move_type])
            
            # Record step data
            step_data = {
                "step": step_count,
                "action": move_type,
                "prev_pos": old_pos,
                "agent_pos": [int(p) for p in env.agent_pos],
                "agent_dir": int(env.agent_dir),
                "dir_name": dir_names[env.agent_dir],
                "delta": [int(env.agent_pos[0] - old_pos[0]), int(env.agent_pos[1] - old_pos[1])],
                "success": "failed" not in info if "action" in info and info["action"] == "diagonal" else False
            }
            
            # Save grid information
            if hasattr(env.unwrapped, "grid"):
                grid_str = str(env.unwrapped.grid)
                step_data["grid"] = grid_str
                print(f"Grid after {move_type}:\n{grid_str}")
            
            # Process image observation
            step_data["has_image"] = "image" in obs
            if "image" in obs:
                image = obs["image"]
                agent_positions = np.where(image > 0)
                if len(agent_positions[0]) > 0:
                    step_data["agent_in_image"] = [
                        int(agent_positions[0][0]),
                        int(agent_positions[1][0]),
                        int(image[agent_positions][0])
                    ]
                    print(f"Agent in image: position={step_data['agent_in_image'][:2]}, value={step_data['agent_in_image'][2]}")
                else:
                    step_data["agent_in_image"] = [-1, -1, -1]
                    print("Agent not found in the image")
                
                # Create visualization
                plt.figure(figsize=(8, 8))
                plt.imshow(image[:,:,0], cmap='tab20')
                plt.colorbar(label='Object/Agent ID')
                plt.title(f"Grid state - {direction_name} - {move_type} - Step {step_data['step']}")
                plt.savefig(os.path.join(result_dir, f"grid_states/{direction_name}_{move_type}_step{step_data['step']}.png"))
                plt.close()
                
            test_data["steps"].append(step_data)
            step_count += 1
            
            print(f"  From {old_pos} to {[int(p) for p in env.agent_pos]}, delta={step_data['delta']}")
            result = "SUCCESS" if step_data["success"] else "FAILED"
            print(f"  Diagonal move {result}")
            
            time.sleep(0.5)
        
        # Store all data for this direction
        all_observations[direction_name] = test_data
        
        # Reset for next test
        obs, info = env.reset(seed=42)
    
    # Save all observation data to a JSON file
    with open(os.path.join(result_dir, "diagonal_move_observations.json"), "w") as f:
        json.dump(all_observations, f, indent=2, cls=NumpyEncoder)
    
    print("\nTest complete! All grid states and observations saved to diagonal_move_tests/")
    env.close()

if __name__ == "__main__":
    test_diagonal_grid_state() 
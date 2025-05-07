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

def test_all_movement_types():
    """
    Comprehensive test of all movement types to isolate the issue with diagonal moves.
    Tests regular moves (turn left, turn right, forward) and diagonal moves from
    different starting positions to verify whether the agent position updates correctly
    in the observation.
    """
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comprehensive_test_results")
    os.makedirs(results_dir, exist_ok=True)
    
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
    
    # Store test results
    all_results = {
        "regular_moves": {},
        "diagonal_moves": {}
    }
    
    # Create environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    env = FullyObsWrapper(env)
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    
    # Reset environment with fixed seed
    obs, info = env.reset(seed=42)
    print(f"Initial state: Position={env.agent_pos}, Direction={dir_names[env.agent_dir]}")
    
    # First, test regular moves to see if they update the observation correctly
    print("\n=== TESTING REGULAR MOVES ===")
    regular_move_tests = [
        {"name": "turn_left", "action": "turn_left", "count": 1},
        {"name": "turn_right", "action": "turn_right", "count": 1},
        {"name": "forward", "action": "forward", "count": 1},
        {"name": "forward_multiple", "action": "forward", "count": 3}
    ]
    
    for test in regular_move_tests:
        # Reset for this test
        obs, info = env.reset(seed=42)
        initial_pos = [int(p) for p in env.agent_pos]
        initial_dir = int(env.agent_dir)
        test_results = []
        
        print(f"\nTesting {test['name']}: {test['count']} times")
        print(f"  Initial position: {initial_pos}, direction: {dir_names[initial_dir]}")
        
        # Save initial state
        step_data = {
            "step": 0,
            "action": "initial",
            "agent_pos": initial_pos,
            "agent_dir": initial_dir,
            "dir_name": dir_names[initial_dir]
        }
        
        # Extract agent position from observation image
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
                
            # Save image for visualization
            plt.figure(figsize=(8, 8))
            plt.imshow(image[:,:,0], cmap='tab20')
            plt.title(f"Initial state - {test['name']}")
            plt.savefig(os.path.join(results_dir, f"regular_{test['name']}_step0.png"))
            plt.close()
        
        test_results.append(step_data)
        
        # Execute the action multiple times
        for i in range(test['count']):
            action_name = test['action']
            old_pos = [int(p) for p in env.agent_pos]
            old_dir = int(env.agent_dir)
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(actions[action_name])
            
            new_pos = [int(p) for p in env.agent_pos]
            new_dir = int(env.agent_dir)
            
            # Record the results
            step_data = {
                "step": i+1,
                "action": action_name,
                "prev_pos": old_pos,
                "prev_dir": old_dir,
                "agent_pos": new_pos,
                "agent_dir": new_dir,
                "dir_name": dir_names[new_dir],
                "delta_pos": [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]],
                "delta_dir": (new_dir - old_dir) % 4
            }
            
            # Extract agent position from observation image
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
                    
                # Save image for visualization
                plt.figure(figsize=(8, 8))
                plt.imshow(image[:,:,0], cmap='tab20')
                plt.title(f"{test['name']} - Step {i+1}")
                plt.savefig(os.path.join(results_dir, f"regular_{test['name']}_step{i+1}.png"))
                plt.close()
            
            test_results.append(step_data)
            print(f"  Step {i+1}: Position {old_pos} → {new_pos}, Direction {dir_names[old_dir]} → {dir_names[new_dir]}")
            print(f"    Agent in image: {step_data.get('agent_in_image', 'Not found')}")
            
            time.sleep(0.3)
        
        # Store results for this test
        all_results["regular_moves"][test["name"]] = test_results
    
    # Now test diagonal moves from different positions
    print("\n=== TESTING DIAGONAL MOVES FROM DIFFERENT POSITIONS ===")
    
    # First, move to the center of the grid to avoid walls
    obs, info = env.reset(seed=42)
    print("Moving to center of grid...")
    
    # Turn to face right (direction 0)
    while env.agent_dir != 0:
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        time.sleep(0.2)
    
    # Move to the center (around position [4, 4])
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.2)
    
    # Turn to face down (direction 1)
    obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
    time.sleep(0.2)
    
    # Move down to reach center
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.2)
    
    center_pos = [int(p) for p in env.agent_pos]
    print(f"Reached center position: {center_pos}")
    
    # Test diagonal moves in all four directions from center
    diagonal_tests = []
    for direction in range(4):
        # For each direction, test both diagonal left and right
        diagonal_tests.append({
            "name": f"{dir_names[direction]}_diagonal_left",
            "direction": direction,
            "action": "diagonal_left"
        })
        diagonal_tests.append({
            "name": f"{dir_names[direction]}_diagonal_right",
            "direction": direction,
            "action": "diagonal_right"
        })
    
    for test in diagonal_tests:
        # Reset to center position
        obs, info = env.reset(seed=42)
        # Move to center again
        while env.agent_dir != 0:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.1)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions["forward"])
            time.sleep(0.1)
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions["forward"])
            time.sleep(0.1)
        
        # Turn to face the specified direction
        while env.agent_dir != test["direction"]:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.2)
        
        initial_pos = [int(p) for p in env.agent_pos]
        initial_dir = int(env.agent_dir)
        
        print(f"\nTesting {test['name']} from position {initial_pos}")
        print(f"  Initial direction: {dir_names[initial_dir]}")
        
        # Save initial state
        test_results = []
        step_data = {
            "step": 0,
            "action": "initial",
            "agent_pos": initial_pos,
            "agent_dir": initial_dir,
            "dir_name": dir_names[initial_dir]
        }
        
        # Extract agent position from observation image
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
                
            # Save image for visualization
            plt.figure(figsize=(8, 8))
            plt.imshow(image[:,:,0], cmap='tab20')
            plt.title(f"Initial state - {test['name']}")
            plt.savefig(os.path.join(results_dir, f"diagonal_{test['name']}_step0.png"))
            plt.close()
        
        test_results.append(step_data)
        
        # Execute the diagonal move
        action_name = test["action"]
        old_pos = [int(p) for p in env.agent_pos]
        old_dir = int(env.agent_dir)
        
        # Take the action
        obs, reward, terminated, truncated, info = env.step(actions[action_name])
        
        new_pos = [int(p) for p in env.agent_pos]
        new_dir = int(env.agent_dir)
        
        # Record the results
        step_data = {
            "step": 1,
            "action": action_name,
            "prev_pos": old_pos,
            "prev_dir": old_dir,
            "agent_pos": new_pos,
            "agent_dir": new_dir,
            "dir_name": dir_names[new_dir],
            "delta_pos": [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]],
            "success": "failed" not in info if "action" in info and info["action"] == "diagonal" else False
        }
        
        # Extract agent position from observation image
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
                
            # Save image for visualization
            plt.figure(figsize=(8, 8))
            plt.imshow(image[:,:,0], cmap='tab20')
            plt.title(f"{test['name']} - Step 1")
            plt.savefig(os.path.join(results_dir, f"diagonal_{test['name']}_step1.png"))
            plt.close()
        
        test_results.append(step_data)
        
        result = "SUCCESS" if step_data["success"] else "FAILED"
        print(f"  Result: Position {old_pos} → {new_pos} ({result})")
        print(f"    Delta position: {step_data['delta_pos']}")
        print(f"    Agent in image: {step_data.get('agent_in_image', 'Not found')}")
        
        time.sleep(0.5)
        
        # Store results for this test
        all_results["diagonal_moves"][test["name"]] = test_results
    
    # Save all results to a JSON file
    with open(os.path.join(results_dir, "movement_test_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nTest complete! All results saved to comprehensive_test_results/")
    env.close()

if __name__ == "__main__":
    test_all_movement_types() 
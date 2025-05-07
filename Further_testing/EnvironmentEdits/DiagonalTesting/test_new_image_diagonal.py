import numpy as np
import gymnasium as gym
import os
import time
import json

# Add the parent directory to the system path for imports
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import ExtractAbstractGrid, DiagonalMoveMonitor
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

def test_diagonal_center():
    """
    Test diagonal movement starting from the center of the grid
    to clearly show changes in new_image arrays.
    """
    print("Creating environment and moving agent to center...")
    
    # Create results directory
    results_dir = "center_diagonal_test_results"
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
    
    # Create environment with minimal wrappers needed for new_image
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    env = FullyObsWrapper(env)
    env = ExtractAbstractGrid(env)  # This wrapper creates the new_image
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Move agent to center of grid (approximately position [4,4])
    # First make sure we're facing right
    while env.agent_dir != 0:  # 0 = Right
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        time.sleep(0.2)
    
    # Move forward to center
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.2)
    
    # Turn down and move down to center
    obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.2)
    
    # Store the agent's starting position
    center_pos = list(env.agent_pos)
    print(f"Agent positioned at center: {center_pos}, Direction: {dir_names[env.agent_dir]}")
    
    # Test diagonal moves from different directions
    test_directions = [
        {"name": "Right", "dir": 0, "moves": ["diagonal_left", "diagonal_right"]},
        {"name": "Down", "dir": 1, "moves": ["diagonal_left", "diagonal_right"]},
        {"name": "Left", "dir": 2, "moves": ["diagonal_left", "diagonal_right"]},
        {"name": "Up", "dir": 3, "moves": ["diagonal_left", "diagonal_right"]}
    ]
    
    # Store all results
    all_results = {}
    
    for direction in test_directions:
        # Reset to center
        obs, info = env.reset(seed=42)
        
        # Need to move to center again
        while env.agent_dir != 0:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.2)
        
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions["forward"])
            time.sleep(0.2)
        
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions["forward"])
            time.sleep(0.2)
        
        # Turn to face the test direction
        while env.agent_dir != direction["dir"]:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.2)
        
        print(f"\n=== Testing diagonal moves when facing {direction['name']} ===")
        
        # Record the initial state
        direction_results = []
        
        start_data = {
            "step": "initial",
            "agent_pos": list(map(int, env.agent_pos)),
            "agent_dir": int(env.agent_dir),
            "dir_name": dir_names[env.agent_dir]
        }
        
        # Record new_image
        if 'new_image' in obs:
            start_data["new_image"] = obs["new_image"].tolist()
            
            # Display the agent layer from new_image
            agent_layer = obs["new_image"][0]
            print(f"Initial Agent layer (new_image[0]):")
            print(np.array2string(agent_layer, separator=', '))
        
        direction_results.append(start_data)
        
        # Test each diagonal move
        for move in direction["moves"]:
            # Reset to center for each diagonal move
            obs, info = env.reset(seed=42)
            
            # Need to move to center again
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
            
            # Turn to face the test direction
            while env.agent_dir != direction["dir"]:
                obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
                time.sleep(0.1)
                
            # Get position before diagonal move
            before_pos = list(map(int, env.agent_pos))
            
            print(f"\nTesting {move} when facing {direction['name']}")
            
            # Take diagonal move
            obs, reward, terminated, truncated, info = env.step(actions[move])
            time.sleep(0.3)
            
            # Record the result
            after_pos = list(map(int, env.agent_pos))
            delta = [after_pos[0] - before_pos[0], after_pos[1] - before_pos[1]]
            
            move_data = {
                "step": move,
                "before_pos": before_pos,
                "after_pos": after_pos,
                "delta": delta,
                "agent_dir": int(env.agent_dir),
                "dir_name": dir_names[env.agent_dir]
            }
            
            # Record new_image after the move
            if 'new_image' in obs:
                move_data["new_image"] = obs["new_image"].tolist()
                
                # Display the agent layer from new_image
                agent_layer = obs["new_image"][0]
                print(f"After {move} Agent layer (new_image[0]):")
                print(np.array2string(agent_layer, separator=', '))
                
                print(f"Position change: {before_pos} → {after_pos}, delta: {delta}")
            
            direction_results.append(move_data)
        
        all_results[direction["name"]] = direction_results
    
    # Save all results to a JSON file
    with open(os.path.join(results_dir, "diagonal_new_image_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nTest complete! Results saved to center_diagonal_test_results/")
    env.close()

if __name__ == "__main__":
    test_diagonal_center() 
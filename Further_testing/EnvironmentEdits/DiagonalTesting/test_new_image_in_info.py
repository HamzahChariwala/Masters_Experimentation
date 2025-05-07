import numpy as np
import gymnasium as gym
import os
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

def test_new_image_extraction():
    """
    Simplified test to extract new_image arrays before and after diagonal moves.
    """
    print("Creating environment with minimal wrapper stack...")
    
    # Create results directory
    results_dir = "new_image_test_results"
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
    
    # Check if new_image is in the observation
    print("Initial observation keys:", list(obs.keys()))
    
    # Test sequence of moves
    test_sequence = [
        {"name": "initial", "action": None},
        {"name": "forward", "action": "forward"},
        {"name": "diagonal_left", "action": "diagonal_left"},
        {"name": "diagonal_right", "action": "diagonal_right"}
    ]
    
    # Results storage
    all_results = []
    
    print("\n=== Testing moves and capturing new_image ===")
    
    # Take each action in sequence
    for i, step in enumerate(test_sequence):
        # Get current position and direction
        agent_pos = env.agent_pos if hasattr(env, 'agent_pos') else None
        agent_dir = env.agent_dir if hasattr(env, 'agent_dir') else None
        
        # Record data for this step
        step_data = {
            "step": i,
            "action": step["name"],
            "agent_pos": [int(p) for p in agent_pos] if agent_pos is not None else None,
            "agent_dir": int(agent_dir) if agent_dir is not None else None,
            "dir_name": dir_names.get(agent_dir, "Unknown") if agent_dir is not None else "Unknown"
        }
        
        # Print current state
        print(f"\nStep {i}: {step['name']}")
        print(f"  Position: {step_data['agent_pos']}, Direction: {step_data['dir_name']}")
        
        # Extract and print new_image
        if 'new_image' in obs:
            step_data["new_image"] = obs["new_image"].tolist()
            
            # Display the agent layer from new_image
            agent_layer = obs["new_image"][0]
            print(f"  Agent layer (new_image[0]):")
            print(np.array2string(agent_layer, separator=', '))
            
            # Display the object layer from new_image
            object_layer = obs["new_image"][1]
            print(f"  Object layer (new_image[1]):")
            print(np.array2string(object_layer, separator=', '))
        else:
            print("  new_image not found in observation")
        
        all_results.append(step_data)
        
        # Take action if needed
        if step["action"] is not None:
            obs, reward, terminated, truncated, info = env.step(actions[step["action"]])
    
    # Save results to a JSON file
    with open(os.path.join(results_dir, "new_image_test_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nTest complete! Results saved to new_image_test_results/")
    env.close()

if __name__ == "__main__":
    test_new_image_extraction() 
import os
import sys
import numpy as np
import gymnasium as gym

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

# Import necessary modules
from Agent_Evaluation.EnvironmentTooling.import_vars import (
    load_config,
    extract_env_config,
    create_evaluation_env,
    extract_and_visualize_env,
    load_agent
)
from Agent_Evaluation.EnvironmentTooling.extract_grid import print_env_tensor

def test_reset_options():
    """Test if reset options are properly propagating through all wrappers"""
    # Configuration
    agent_path = "Agent_Storage/LavaTests/Standard2"
    env_id = "MiniGrid-LavaCrossingS9N1-v0"
    seed = 81102
    
    print(f"\n\n{'='*50}")
    print(f"TESTING RESET OPTIONS PROPAGATION")
    print(f"{'='*50}")
    print(f"Agent: {agent_path}")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    
    # Load the config
    config = load_config(agent_path)
    
    # Extract environment settings
    env_settings = extract_env_config(config, override_env_id=env_id)
    
    # Create environment for testing using the correct function signature
    env = create_evaluation_env(env_settings=env_settings, seed=seed, override_rank=0)
    
    # Extract the grid for visualization
    env_tensor = extract_and_visualize_env(env, env_id, seed, False)
    print("\nEnvironment layout:")
    print_env_tensor(env_tensor)
    
    # Test position
    test_position = (7, 2, 3)  # (x, y, orientation)
    x, y, orientation = test_position
    
    print(f"\nTesting reset with position: {test_position}")
    
    # Create reset options
    options = {
        'agent_pos': (x, y),
        'agent_dir': orientation
    }
    
    # Reset with options
    print(f"\nCalling env.reset with options = {options}")
    obs, info = env.reset(options=options)
    
    print(f"\nTaking a step to see if new position is preserved")
    next_obs, reward, terminated, truncated, next_info = env.step(0)  # Forward action
    
    print(f"\nResetting again with default position")
    obs, info = env.reset()
    
    print(f"\nResetting with options again to see if it still works")
    options = {
        'agent_pos': (1, 1),
        'agent_dir': 0
    }
    obs, info = env.reset(options=options)
    
    print(f"\nTest completed")
    env.close()

if __name__ == "__main__":
    test_reset_options() 
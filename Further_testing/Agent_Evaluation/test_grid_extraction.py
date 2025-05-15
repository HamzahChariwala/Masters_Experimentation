import os
import sys
import argparse

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import from our import_vars.py file
from Agent_Evaluation.import_vars import (
    extract_env_config,
    create_evaluation_env,
    extract_and_visualize_env,
)

def test_grid_extraction(env_id="MiniGrid-LavaCrossingS11N5-v0", seed=42):
    """
    Test the grid extraction functionality without loading an agent.
    
    Args:
        env_id (str): Environment ID to use for testing
        seed (int): Random seed for reproducibility
    """
    print(f"Testing grid extraction for environment: {env_id}")
    print(f"Seed: {seed}")
    
    try:
        # Create minimal config for this environment
        env_config = {
            'env_id': env_id,
            'window_size': 7,
            'cnn_keys': [],
            'mlp_keys': ["four_way_goal_direction", "four_way_angle_alignment", 
                         "barrier_mask", "lava_mask"],
            'max_episode_steps': 150,
            'use_no_death': True,
            'no_death_types': ('lava',),
            'death_cost': 0.0,
        }
        
        # Create evaluation environment
        print("Creating environment...")
        env = create_evaluation_env(env_config, seed=seed, override_rank=0)
        
        try:
            # Extract and visualize the environment tensor
            print("Extracting environment layout...")
            env_tensor = extract_and_visualize_env(env, env_id=env_id)
            
            print("Grid extraction test completed successfully!")
            
        except Exception as e:
            print(f"Error during environment tensor generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            env.close()
    except Exception as e:
        print(f"Error setting up environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test grid extraction functionality")
    parser.add_argument("--env_id", type=str, default="MiniGrid-LavaCrossingS11N5-v0", 
                        help="Environment ID to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Run the test
    test_grid_extraction(args.env_id, args.seed) 
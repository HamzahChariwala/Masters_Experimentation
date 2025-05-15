import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import signal

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import from our import_vars.py file
from Agent_Evaluation.import_vars import (
    load_config,
    extract_env_config,
    create_evaluation_env,
    extract_and_visualize_env,
    load_agent,
    DEFAULT_RANK,
    DEFAULT_NUM_EPISODES
)


def single_env_evals(agent_path: str, env_id: str, seed: int, generate_plot: bool = True):
    """
    Evaluate an agent in a single environment configuration
    
    Args:
        agent_path (str): Path to the agent folder in Agent_Storage
        env_id (str): Environment ID to use for evaluation
        seed (int): Random seed for reproducibility
        generate_plot (bool): Whether to generate matplotlib plots (default: True)
    """
    print(f"\nEvaluating agent: {agent_path}")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Plot generation: {'enabled' if generate_plot else 'disabled'}")
    
    try:
        # Load config from agent folder
        config = load_config(agent_path)
        
        # Extract environment settings with provided env_id
        env_settings = extract_env_config(config, override_env_id=env_id)
        
        # Create evaluation environment first so we can extract the actual layout
        print("Creating environment...")
        env = create_evaluation_env(env_settings, seed=seed, override_rank=DEFAULT_RANK)
        
        try:
            # Extract and visualize the environment tensor using the new method
            print("Extracting environment layout...")
            env_tensor = extract_and_visualize_env(env, env_id=env_id, generate_plot=generate_plot)
            
            # Load agent
            print("Loading agent...")
            agent = load_agent(agent_path, config)
            
            # Here you would add the code to evaluate the agent and generate visualizations/metrics
            print(f"Ready to evaluate agent in {agent_path} for {DEFAULT_NUM_EPISODES} episodes")
            print(f"Using environment ID: {env_settings['env_id']}")
            print(f"Using seed: {seed}")
        except Exception as e:
            print(f"Error during environment tensor generation or agent evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            env.close()
    except Exception as e:
        print(f"Error setting up evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments - only path is configurable via command line
    parser = argparse.ArgumentParser(description="Generate evaluations for trained agents")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to the agent folder in Agent_Storage")
    args = parser.parse_args()
    
    # Call the evaluation function with default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 42
    single_env_evals(args.path, ENV_ID, SEED, False)
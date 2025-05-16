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
print(f"Added to Python path: {project_root}")

# Import from Behaviour_Specification
from Behaviour_Specification.graph_analysis import analyze_navigation_graphs

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


def single_env_evals(agent_path: str, env_id: str, seed: int, generate_plot: bool = True, lava_penalty: int = 1, debug: bool = False):
    """
    Evaluate an agent in a single environment configuration
    
    Args:
        agent_path (str): Path to the agent folder in Agent_Storage
        env_id (str): Environment ID to use for evaluation
        seed (int): Random seed for reproducibility
        generate_plot (bool): Whether to generate matplotlib plots (default: True)
        lava_penalty (int): Penalty for lava cells in dangerous graph (default: 1)
        debug (bool): Whether to print detailed diagnostic information (default: False)
    """
    print(f"\nEvaluating agent: {agent_path}")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Plot generation: {'enabled' if generate_plot else 'disabled'}")
    print(f"Lava penalty: {lava_penalty}x")
    print(f"Debug output: {'enabled' if debug else 'disabled'}")
    
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
            
            # Analyze navigation graphs - this now includes state node generation, graph creation,
            # and path analysis, all in one function
            analysis_results = analyze_navigation_graphs(
                env_tensor=env_tensor, 
                lava_penalty_multiplier=lava_penalty,
                print_output=True,
                debug=debug
            )
            
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate evaluations for trained agents")
    parser.add_argument("--path", type=str, 
                        help="Path to the agent folder in Agent_Storage")
    parser.add_argument("--no-plot", action="store_true", 
                        help="Disable matplotlib plot generation (useful for headless environments)")
    parser.add_argument("--lava-penalty", type=int, default=1,
                        help="Penalty multiplier for lava cells in the dangerous graph (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug output including diagonal safety checks")
    args = parser.parse_args()
    
    # Path is required
    if not args.path:
        parser.error("the --path argument is required")
        
    # Call the evaluation function with default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 12345
    single_env_evals(args.path, ENV_ID, SEED, not args.no_plot, args.lava_penalty, args.debug)
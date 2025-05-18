import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import json

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

# Import from Behaviour_Specification
from Behaviour_Specification.log import analyze_navigation_graphs, export_path_data_to_json

# Import from our import_vars.py file - updated to reflect new location
from Agent_Evaluation.EnvironmentTooling.import_vars import (
    load_config,
    extract_env_config,
    create_evaluation_env,
    extract_and_visualize_env,
    load_agent,
    DEFAULT_RANK,
    DEFAULT_NUM_EPISODES
)

# Import our agent logging functionality
from Agent_Evaluation.AgentTooling.agent_logger import run_agent_evaluation


def single_env_evals(agent_path: str, env_id: str, seed: int, generate_plot: bool = True, debug: bool = False, force_dijkstra: bool = False):
    """
    Evaluate an agent in a single environment configuration
    
    Args:
        agent_path (str): Path to the agent folder in Agent_Storage
        env_id (str): Environment ID to use for evaluation
        seed (int): Random seed for reproducibility
        generate_plot (bool): Whether to generate matplotlib plots (default: True)
        debug (bool): Whether to print detailed diagnostic information (default: False)
        force_dijkstra (bool): Whether to force recalculation of Dijkstra's paths (default: False)
    """
    print(f"\nEvaluating agent: {agent_path}")
    print(f"Environment: {env_id}")
    print(f"Seed: {seed}")
    print(f"Plot generation: {'enabled' if generate_plot else 'disabled'}")
    print(f"Debug output: {'enabled' if debug else 'disabled'}")
    
    try:
        # We'll let load_config handle the path resolution
        # Just capture the original path for reference
        original_agent_path = agent_path
        
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
            env_tensor = extract_and_visualize_env(env, env_id=env_id, seed=seed, generate_plot=generate_plot)
            
            # Check if Dijkstra's analysis has already been performed
            evaluations_dir = os.path.join(project_root, "Behaviour_Specification", "Evaluations")
            dijkstra_output_path = os.path.join(evaluations_dir, f"{env_id}-{seed}.json")
            if os.path.exists(dijkstra_output_path) and not force_dijkstra:
                print(f"Dijkstra's path data already exists at {dijkstra_output_path}. Skipping path analysis.")
            else:
                # Analyze navigation graphs - this now includes state node generation, graph creation,
                # and path analysis, all in one function
                print("Running Dijkstra's path analysis...")
                analysis_results = analyze_navigation_graphs(
                    env_tensor=env_tensor, 
                    print_output=True,
                    debug=debug
                )
                
                # Export path data to JSON
                output_path = export_path_data_to_json(
                    analysis_results=analysis_results,
                    env_tensor=env_tensor,
                    env_id=env_id,
                    seed=seed
                )
                print(f"Path data exported to: {output_path}")
            
            # Load agent - using the original path since config loading was successful
            print("Loading agent...")
            agent = load_agent(original_agent_path, config)
            
            # Run agent evaluation and log its behavior
            print(f"Evaluating agent in {original_agent_path} for {DEFAULT_NUM_EPISODES} episodes")
            agent_log_path = run_agent_evaluation(
                agent=agent,
                env=env,
                env_id=env_id,
                seed=seed,
                num_episodes=DEFAULT_NUM_EPISODES
            )
            print(f"Agent behavior data exported to: {agent_log_path}")
            
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


def multiple_env_evals(agent_path: str, env_id: str, seed: int, num_envs: int, generate_plot: bool = True, debug: bool = False, force_dijkstra: bool = False):
    """
    Evaluate an agent in multiple environments
    """
    for i in range(num_envs):
        current_seed = seed + i
        single_env_evals(agent_path, env_id, current_seed, generate_plot, debug, force_dijkstra)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate evaluations for trained agents")
    parser.add_argument("--path", type=str, 
                        help="Path to the agent folder in Agent_Storage")
    parser.add_argument("--no-plot", action="store_true", 
                        help="Disable matplotlib plot generation (useful for headless environments)")
    parser.add_argument("--force-dijkstra", action="store_true",
                        help="Force recalculation of Dijkstra's paths even if they exist")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug output including diagonal safety checks")
    args = parser.parse_args()
    
    # Path is required
    if not args.path:
        parser.error("the --path argument is required")
        
    # Call the evaluation function with default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 42
    NUM = 10
    multiple_env_evals(args.path, ENV_ID, SEED, NUM, not args.no_plot, args.debug, args.force_dijkstra)
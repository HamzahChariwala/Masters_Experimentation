import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import json
import yaml
import gymnasium as gym

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
        
        # IMPORTANT: Use exactly the same seed for environment creation and visualization
        # This ensures the environment layout will match between Dijkstra and agent evaluation
        print(f"Creating environment with seed: {seed}")
        
        # Create evaluation environment first so we can extract the actual layout
        print("Creating environment...")
        env = create_evaluation_env(env_settings, seed=seed, override_rank=DEFAULT_RANK)
        
        try:
            # Extract and visualize the environment tensor using the new method
            # Pass the same seed here to ensure filename consistency
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
            # IMPORTANT: Use the same environment instance and seed we used for visualization
            print(f"Evaluating agent in {original_agent_path} for {DEFAULT_NUM_EPISODES} episodes")
            
            # Check if the path is absolute or already includes Agent_Storage
            if os.path.isabs(original_agent_path) or original_agent_path.startswith("Agent_Storage/"):
                full_agent_dir = original_agent_path
            else:
                # Construct the full agent directory path including Agent_Storage
                full_agent_dir = os.path.join(project_root, "Agent_Storage", original_agent_path)
            
            agent_log_path = run_agent_evaluation(
                agent=agent,
                env=env,  # Use the same environment instance
                env_id=env_id,
                seed=seed,  # Use the same seed
                num_episodes=DEFAULT_NUM_EPISODES,
                agent_dir=full_agent_dir
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


def generate_complete_summary(agent_path: str, env_id: str, seed: int, num_envs: int, generate_plot: bool = True, debug: bool = False, force_dijkstra: bool = False):
    """
    Generate summary statistics for an agent in multiple environments.
    First runs evaluations for each environment, then generates performance summaries.
    
    Args:
        agent_path (str): Path to the agent folder in Agent_Storage
        env_id (str): Environment ID to use for evaluation
        seed (int): Random seed for reproducibility (starting seed for multiple environments)
        num_envs (int): Number of environments to evaluate (with sequential seeds)
        generate_plot (bool): Whether to generate matplotlib plots (default: True)
        debug (bool): Whether to print detailed diagnostic information (default: False)
        force_dijkstra (bool): Whether to force recalculation of Dijkstra's paths (default: False)
    """
    # First, run evaluations for all environments if needed
    for i in range(num_envs):
        current_seed = seed + i
        single_env_evals(agent_path, env_id, current_seed, generate_plot, debug, force_dijkstra)
    
    # Then, generate performance summaries
    print("\nGenerating performance summaries...")
    
    # Import summary functions - use the original agent evaluation summary processor
    from Agent_Evaluation.SummaryTooling.evaluation_summary import process_evaluation_logs
    
    # Determine the full agent path
    if os.path.isabs(agent_path) or agent_path.startswith("Agent_Storage/"):
        full_agent_dir = agent_path
    else:
        # Construct the full agent directory path including Agent_Storage
        full_agent_dir = os.path.join(project_root, "Agent_Storage", agent_path)
    
    # Process the evaluation logs
    summaries = process_evaluation_logs(
        agent_dirs=[full_agent_dir],
        save_results=True
    )
    
    # Print summary of results
    agent_name = os.path.basename(full_agent_dir)
    if agent_name in summaries:
        agent_data = summaries[agent_name]
        total_logs = len(agent_data)
        print(f"\nSummary: Processed {total_logs} evaluation logs for agent {agent_path}")
        
        # If a performance_summary.json was created, check for agent statistics
        summary_path = os.path.join(full_agent_dir, "evaluation_logs", "performance_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                full_summary = json.load(f)
                if "agent_statistics" in full_summary:
                    stats = full_summary["agent_statistics"]
                    print(f"  Success rate: {stats['success_rate']:.2f}%")
                    print(f"  Lava avoidance rate: {stats['lava_avoidance_rate']:.2f}%")
                    print(f"  Safe diagonal rate: {stats['safe_diagonal_rate']:.2f}%")
                    print(f"  Average path length: {stats['avg_path_length']:.2f}")
                    print(f"  Total states analyzed: {stats['total_states']}")
    else:
        print(f"No evaluation logs processed for agent {agent_path}")
    
    # Separately process the Dijkstra logs (if the new module is available)
    try:
        print("\nProcessing Dijkstra evaluation logs...")
        # Try to import the new Dijkstra log processor
        from Behaviour_Specification.SummaryTooling import process_dijkstra_logs
        
        # Process Dijkstra logs but don't overwrite existing agent performance data
        evaluations_dir = os.path.join(project_root, "Behaviour_Specification", "Evaluations")
        dijkstra_mode_summaries = process_dijkstra_logs(
            logs_dir=evaluations_dir,
            save_results=True,
            output_dir=evaluations_dir
        )
        
        # Print summary of Dijkstra results
        print(f"\nDijkstra log analysis results:")
        for mode, env_summaries in dijkstra_mode_summaries.items():
            if env_summaries:
                env_count = len(env_summaries)
                print(f"  Mode '{mode}': {env_count} environments processed")
    except (ImportError, ModuleNotFoundError):
        print("Note: Behaviour_Specification.SummaryTooling not available - skipping Dijkstra log processing")
    
    print(f"\nEvaluation and summary generation complete for agent {agent_path}")


def check_environment_seed(config, env_id, seed_override=None):
    """
    Log a warning if the environment is being created with a different seed from the one used for Dijkstra.
    This is a key fix for Issue 1 - environment layout mismatch.
    
    Args:
        config: The agent config dictionary
        env_id: The environment ID
        seed_override: Optional seed override value
    
    Returns:
        int: The correct seed to use
    """
    # If the seed_override is provided, use it
    if seed_override is not None:
        print(f"Using seed override value: {seed_override}")
        return seed_override
    
    # Check if a specific evaluation seed was set in the config
    if 'seeds' in config and 'evaluation' in config['seeds']:
        eval_seed = config['seeds']['evaluation']
        print(f"Using evaluation seed from config: {eval_seed}")
        return eval_seed
    
    # Otherwise, fall back to a default seed
    print("WARNING: No evaluation seed specified in config or override, using default seed 81102")
    return 81102


def generate_agent_evaluations_in_dir(agent_dir, base_env, env_id, num_episodes=1, seed=81102, verbose=False):
    """
    Generate evaluations for all agents in the specified directory.
    
    Args:
        agent_dir: Path to the directory containing agent models
        base_env: Base environment object (will be reset with proper seed for each evaluation)
        env_id: Environment ID string
        num_episodes: Number of episodes to run per evaluation
        seed: Random seed to use for the evaluation
        verbose: Whether to print verbose output
    """
    # Get agent paths
    agent_subdirs = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d))]
    
    for subdir in agent_subdirs:
        agent_path = os.path.join(agent_dir, subdir)
        
        # Check if the agent file exists
        model_file = os.path.join(agent_path, "agent.zip")
        if not os.path.exists(model_file):
            if verbose:
                print(f"No agent file found at {model_file}")
            continue
        
        # Check if config file exists to get the right evaluation seed
        config_file = os.path.join(agent_path, "config.yaml")
        evaluation_seed = seed  # Default
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                evaluation_seed = check_environment_seed(config, env_id, seed)
            except Exception as e:
                print(f"Failed to read config file: {e}")
        
        # Create environment with the correct seed from the config
        env = create_evaluation_env(base_env, env_id, seed=evaluation_seed)
        
        # Generate evaluation
        print(f"\nGenerating evaluation for agent: {agent_path}")
        print(f"Using evaluation seed: {evaluation_seed}")
        generate_agent_evaluation(agent_path, env, env_id, evaluation_seed, num_episodes, verbose)


def generate_agent_evaluation(agent_path, env, env_id, seed, num_episodes=1, verbose=False):
    """
    Generate an evaluation for a single agent and save it to a JSON file.
    
    Args:
        agent_path: Path to the agent directory
        env: Environment to evaluate in
        env_id: Environment ID string
        seed: Random seed to use for the evaluation
        num_episodes: Number of episodes to run
        verbose: Whether to print verbose output
    
    Returns:
        str: Path to the saved JSON file
    """
    print(f"Running agent evaluation for {agent_path} with seed {seed}...")
    
    try:
        # Import agent model
        from stable_baselines3 import DQN
        from stable_baselines3.common.evaluation import evaluate_policy
        
        # Load the agent model from the agent.zip file
        agent_file = os.path.join(agent_path, "agent.zip")
        agent = DQN.load(agent_file)
        
        # Create agent evaluation logs directory
        eval_logs_dir = os.path.join(agent_path, "evaluation_logs")
        os.makedirs(eval_logs_dir, exist_ok=True)
        
        # Run agent evaluation and export to JSON
        from Agent_Evaluation.AgentTooling.agent_logger import run_agent_evaluation
        
        # Run evaluation with real agent path generation
        json_path = run_agent_evaluation(
            agent, env, env_id, seed, num_episodes=num_episodes,
            agent_dir=agent_path
        )
        
        print(f"Agent evaluation saved to {json_path}")
        return json_path
        
    except Exception as e:
        print(f"Error generating agent evaluation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    from Agent_Evaluation.EnvironmentTooling.import_vars import create_evaluation_env
    
    parser = argparse.ArgumentParser(description="Generate agent evaluations")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to the agent directory or parent directory containing multiple agents")
    
    args = parser.parse_args()
    
    # Default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 81102
    NUM = 15  # Number of different seeds to evaluate on
    NUM_EPISODES = 1  # Number of episodes per seed (usually 1 since evaluation is deterministic)
    
    # Generate complete summary which runs evaluations on multiple seeds
    generate_complete_summary(
        args.path, ENV_ID, SEED, NUM, 
        generate_plot=True, 
        debug=False, 
        force_dijkstra=False
    )
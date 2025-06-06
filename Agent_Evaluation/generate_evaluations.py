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

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from Behaviour_Specification
from Behaviour_Specification.log import analyze_navigation_graphs, export_path_data_to_json

# Import from Agent_Evaluation
from Agent_Evaluation.AgentTooling.agent_functionality import evauate_agent_on_single_env, load_agent_from_path
from Agent_Evaluation.AgentTooling.results_processing import export_agent_eval_data_to_json, add_performance_summary_to_agent_logs, create_agent_performance_summary
from Agent_Evaluation.SummaryTooling.comparison_evaluation import load_dijkstra_results, calculate_performance_difference, generate_comparison_evaluation, save_comparison_results
from Agent_Evaluation.SummaryTooling.reachable_path_evaluation import generate_reachable_path_summary
from Agent_Evaluation.SummaryTooling.unreachable_path_evaluation import generate_unreachable_path_summary
from Agent_Evaluation.SummaryTooling.lava_only_evaluation import generate_lava_only_summary
from Agent_Evaluation.add_metrics import add_metrics_to_log

# Import from our import_vars.py file - updated to reflect new location
from Agent_Evaluation.EnvironmentTooling.import_vars import (
    load_config,
    extract_env_config,
    extract_and_visualize_env,
    load_agent,
    DEFAULT_RANK,
    DEFAULT_NUM_EPISODES
)

# Import from Agent_Evaluation - only import the function that actually exists
from Agent_Evaluation.add_metrics import add_metrics_to_log

def generate_complete_summary(agent_path: str, env_id: str, seed: int, num_envs: int, generate_plot: bool = True, debug: bool = False, force_dijkstra: bool = False):

    # First, run evaluations for all environments if needed
    for i in range(num_envs):
        current_seed = seed + i
        single_env_evals(agent_path, env_id, current_seed, generate_plot, debug, force_dijkstra)
    
    # Then, generate performance summaries
    print("\nGenerating performance summaries...")

    try:
        print("\nProcessing Dijkstra evaluation logs...")
        # Try to import the Dijkstra summary processors
        # Use proper imports with direct path to ensure modules are found
        sys.path.insert(0, os.path.join(project_root, "Behaviour_Specification"))
        from Behaviour_Specification.SummaryTooling.evaluation_summary import process_dijkstra_logs, create_dijkstra_performance_summary
        
        # Process Dijkstra logs and generate per-ruleset summary files with compact formatting
        evaluations_dir = os.path.join(project_root, "Behaviour_Specification", "Evaluations")
        print(f"Looking for Dijkstra logs in {evaluations_dir}")
        
        # Use process_dijkstra_logs with generate_summary_files=False to:
        # 1. Process the Dijkstra logs but don't create separate per-ruleset summary files
        # 2. Still add performance data to each log file
        print("Processing Dijkstra logs (without per-ruleset summary files)...")
        dijkstra_mode_summaries = process_dijkstra_logs(
            logs_dir=evaluations_dir,
            save_results=True,
            output_dir=evaluations_dir,
            generate_summary_files=False  # Do NOT generate separate per-ruleset summary files
        )
        
        # Then use create_dijkstra_performance_summary to create the comprehensive performance summary
        print("Creating comprehensive Dijkstra performance summary...")
        dijkstra_summary_file = create_dijkstra_performance_summary(
            logs_dir=evaluations_dir,
            overall_only=False  # Include per-state statistics
        )
        
        if dijkstra_summary_file:
            print(f"Dijkstra performance summary created: {dijkstra_summary_file}")
        
        # Print summary of Dijkstra results
        print(f"\nDijkstra log analysis results:")
        for mode, env_summaries in dijkstra_mode_summaries.items():
            if env_summaries:
                env_count = len(env_summaries)
                print(f"  Mode '{mode}': {env_count} environments processed")
    except Exception as e:
        print(f"Error processing Dijkstra evaluation logs: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate agent performance summaries
    try:
        print("\nProcessing Agent evaluation logs...")
        
        # Path to the agent's evaluation_logs directory
        agent_logs_dir = os.path.join(agent_path, "evaluation_logs")
        
        # Step 1: Add performance data to each log file if not already present
        print("Adding performance data to agent logs...")
        add_performance_summary_to_agent_logs(
            logs_dir=agent_logs_dir,
            save_results=True
        )
        
        # Step 1.5: Add behavioral metrics to agent logs and update summaries
        print("Adding behavioral metrics to agent logs...")
        try:
            add_metrics_to_log(agent_path)
            # print("Updating evaluation summaries with new metrics...")
            # update_evaluation_summaries(agent_path)
            # print("Calculating aggregate statistics...")
            # calculate_aggregate_statistics(agent_path)
        except Exception as e:
            print(f"Warning: Error adding behavioral metrics: {e}")
            # Continue with the rest of the evaluation process even if metrics fail
        
        # Step 2: Create the overall summary file for the agent
        print("Creating agent performance summary...")
        summary_file = create_agent_performance_summary(
            agent_dir=agent_path,
            logs_dir=agent_logs_dir,
            overall_only=False
        )
        
        if summary_file:
            print(f"Agent performance summary created: {summary_file}")
        else:
            print("No agent performance summary created - check for errors or missing data")
            
        # Step 3: Create the lava-only summary for states that start on lava
        print("Creating lava-only summary...")
        lava_summary_file = generate_lava_only_summary(agent_path)
        
        if lava_summary_file:
            print(f"Lava-only summary created: {lava_summary_file}")
        else:
            print("No lava-only summary created - check for errors or missing data")
            
        # Step 4: Create the reachable-path summary for states that can reach goal without lava
        print("Creating reachable-path summary...")
        reachable_summary_file = generate_reachable_path_summary(agent_path)
        
        if reachable_summary_file:
            print(f"Reachable-path summary created: {reachable_summary_file}")
        else:
            print("No reachable-path summary created - check for errors or missing data")
            
        # Step 5: Create the unreachable-path summary for states that cannot reach goal without lava
        print("Creating unreachable-path summary...")
        unreachable_summary_file = generate_unreachable_path_summary(agent_path)
        
        if unreachable_summary_file:
            print(f"Unreachable-path summary created: {unreachable_summary_file}")
        else:
            print("No unreachable-path summary created - check for errors or missing data")
            
    except Exception as e:
        print(f"Error generating agent performance summaries: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate comparison evaluation between agent and Dijkstra results
    try:
        print("\nGenerating comparison evaluation between agent and Dijkstra results...")
        
        # Generate and save comparison evaluation
        final_eval = generate_comparison_evaluation(agent_path)
        if final_eval:
            save_comparison_results(agent_path, final_eval)
            print(f"Comparison evaluation generated and saved to {os.path.join(agent_path, 'final_eval.json')}")
        else:
            print("No comparison evaluation generated - check for errors or missing data")
            
    except Exception as e:
        print(f"Error generating comparison evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nEvaluation and summary generation complete for agent {agent_path}")


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
        env = create_evaluation_env(env_settings, seed=seed, new_approach_bool=True)
        
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
            
            # Load the agent for evaluation
            print("Loading agent for evaluation...")
            model = load_agent_from_path(agent_path)
            
            # Run agent evaluation
            print("Running agent evaluation on various starting positions...")
            eval_results = evauate_agent_on_single_env(
                env=env,
                model=model,
                seed=seed,
                env_tensor=env_tensor
            )
            
            # Export agent evaluation results to JSON
            agent_output_path = export_agent_eval_data_to_json(
                results_dict=eval_results,
                env_tensor=env_tensor,
                env_id=env_id,
                seed=seed,
                agent_folder=agent_path
            )
            print(f"Agent evaluation data exported to: {agent_output_path}")
            
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
    import argparse
    from Agent_Evaluation.EnvironmentTooling.import_vars import create_evaluation_env
    
    parser = argparse.ArgumentParser(description="Generate agent evaluations")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to the agent directory or parent directory containing multiple agents")
    
    args = parser.parse_args()
    
    # Default values
    ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    SEED = 81102
    NUM = 10  # Number of different seeds to evaluate on
    NUM_EPISODES = 1  # Number of episodes per seed (usually 1 since evaluation is deterministic)
    
    # Generate complete summary which runs evaluations on multiple seeds
    generate_complete_summary(
        args.path, ENV_ID, SEED, NUM, 
        generate_plot=True, 
        debug=False, 
        force_dijkstra=False
    ) 
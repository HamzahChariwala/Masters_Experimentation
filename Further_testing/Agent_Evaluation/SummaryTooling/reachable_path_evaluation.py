import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import copy

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import from Agent_Evaluation module
from Agent_Evaluation.AgentTooling.results_processing import format_json_with_compact_arrays

def load_dijkstra_results() -> Dict[str, Any]:
    """
    Load Dijkstra performance results from the standard file location.
    
    Returns:
        Dict[str, Any]: The Dijkstra performance data organized by ruleset
    """
    dijkstra_file = os.path.join(project_root, "Behaviour_Specification", "Evaluations", "performance_summary.json")
    
    if not os.path.exists(dijkstra_file):
        print(f"Error: Dijkstra performance summary file not found at {dijkstra_file}")
        return {}
    
    with open(dijkstra_file, 'r') as f:
        return json.load(f)

def check_reachable_without_lava(state_key: str, dijkstra_data: Dict[str, Any]) -> bool:
    """
    Check if a state can reach the goal without entering lava according to Dijkstra data.
    Uses the standard ruleset which models entering lava as impossible.
    
    Args:
        state_key (str): The state key in format "x,y,orientation"
        dijkstra_data (Dict[str, Any]): The Dijkstra performance data
        
    Returns:
        bool: True if the state can reach the goal without entering lava, False otherwise
    """
    # Get standard ruleset which doesn't allow entering lava
    standard_ruleset = dijkstra_data.get("rulesets", {}).get("standard", {})
    
    # Get per-state statistics
    statistics = standard_ruleset.get("statistics", {})
    
    # Check if this state is in the statistics
    if state_key in statistics:
        # Check if goal is reachable from this state
        # In standard ruleset, goal_reached_proportion > 0 means a path exists without entering lava
        return statistics[state_key].get("goal_reached_proportion", 0) > 0
    
    return False

def generate_reachable_path_summary(agent_dir: str) -> str:
    """
    Generate a performance summary for states that can reach the goal without entering lava.
    Only includes states where the agent starts on a floor tile (not lava).
    
    Args:
        agent_dir (str): Path to the agent directory
        
    Returns:
        str: Path to the generated summary file
    """
    print(f"Generating reachable-path summary for agent: {os.path.basename(agent_dir)}")
    
    # Define logs directory
    logs_dir = os.path.join(agent_dir, "evaluation_logs")
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory not found at {logs_dir}")
        return ""
    
    # Load Dijkstra results - but we'll also load individual environment files
    dijkstra_summary = load_dijkstra_results()
    if not dijkstra_summary:
        print("Error: Failed to load Dijkstra summary results. Aborting.")
        return ""
    
    # Find all JSON files in the logs directory
    json_files = glob.glob(os.path.join(logs_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"Error: No JSON files found in {logs_dir}")
        return ""
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    # Initialize data structures to store statistics for reachable-without-lava states
    reachable_state_stats = defaultdict(lambda: {
        "count": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    })
    
    # Track overall statistics for reachable-without-lava states
    reachable_overall_stats = {
        "total_states": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    }
    
    # Counter for filtered states
    filtered_lava_start_states = 0
    filtered_unreachable_states = 0
    
    # Additional debugging info
    env_counts = {}
    
    # Process each JSON file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        env_name = os.path.splitext(file_name)[0]  # Get environment name without extension
        env_counts[env_name] = {"floor_states": 0, "reachable_states": 0}
        
        print(f"Processing {file_name}")
        
        # Skip performance summary files
        if file_name.startswith("performance_"):
            print(f"  Skipping performance summary file")
            continue
        
        # Load agent performance data
        try:
            with open(json_file, 'r') as f:
                agent_data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Invalid JSON in {file_name}")
            continue
        
        # Skip if no performance data
        if "performance" not in agent_data:
            print(f"  No performance data found, skipping")
            continue
        
        # Get agent performance data - it could be directly or under 'agent' key
        if "agent" in agent_data["performance"]:
            agent_perf_data = agent_data["performance"]["agent"]
        else:
            agent_perf_data = agent_data["performance"]
        
        # Load corresponding Dijkstra data file
        # Extract environment ID and seed from file name
        # Expected format: ENV_ID-SEED.json
        env_id_seed = file_name
        dijkstra_file = os.path.join(project_root, "Behaviour_Specification", "Evaluations", env_id_seed)
        
        if not os.path.exists(dijkstra_file):
            print(f"  Warning: Dijkstra data file not found at {dijkstra_file}")
            continue
        
        # Load Dijkstra data for this specific environment
        try:
            with open(dijkstra_file, 'r') as f:
                dijkstra_data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Invalid JSON in Dijkstra file {dijkstra_file}")
            continue
        
        # Get the standard ruleset data
        if "performance" in dijkstra_data and "standard" in dijkstra_data["performance"]:
            dijkstra_standard = dijkstra_data["performance"]["standard"]
        else:
            print(f"  Warning: Standard ruleset not found in Dijkstra data")
            continue
        
        # Create a set of reachable states according to Dijkstra
        reachable_states = set()
        for state_key, metrics in dijkstra_standard.items():
            # Skip comment and mode keys
            if state_key.startswith("__"):
                continue
                
            # Check if goal is reachable - metrics[3] is reaches_goal
            if isinstance(metrics, list) and len(metrics) > 3 and metrics[3] == 1:
                reachable_states.add(state_key)
        
        print(f"  Found {len(reachable_states)} reachable states in Dijkstra data")
        
        # Process each state in the agent performance data
        for state_key, metrics in agent_perf_data.items():
            # Skip comment and mode keys
            if state_key.startswith("__"):
                continue
            
            # Extract metrics from the array
            # [cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken]
            if len(metrics) >= 6:  # Make sure we have at least the first 6 metrics
                cell_type = metrics[0]
                
                # Skip states where the agent starts on lava
                if cell_type == "lava":
                    filtered_lava_start_states += 1
                    continue
                
                # Count this as a floor state
                env_counts[env_name]["floor_states"] += 1
                
                # Check if this state can reach the goal without entering lava
                # If it's in the reachable_states set, it can reach the goal
                if state_key not in reachable_states:
                    filtered_unreachable_states += 1
                    continue
                
                # Count this as a reachable state
                env_counts[env_name]["reachable_states"] += 1
                
                # Now extract the remaining metrics for states that pass both filters
                path_length = metrics[1]
                lava_steps = metrics[2]
                reaches_goal = metrics[3]
                next_cell_is_lava = metrics[4]
                risky_diagonal = metrics[5]
                
                # Update per-state statistics
                reachable_state_stats[state_key]["count"] += 1
                reachable_state_stats[state_key]["lava_cell_count"] += 1 if cell_type == "lava" else 0
                reachable_state_stats[state_key]["path_length_sum"] += path_length
                reachable_state_stats[state_key]["lava_steps_sum"] += lava_steps
                reachable_state_stats[state_key]["goal_reached_count"] += 1 if reaches_goal else 0
                reachable_state_stats[state_key]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                reachable_state_stats[state_key]["risky_diagonal_count"] += 1 if risky_diagonal else 0
                
                # Update overall statistics
                reachable_overall_stats["total_states"] += 1
                reachable_overall_stats["lava_cell_count"] += 1 if cell_type == "lava" else 0
                reachable_overall_stats["path_length_sum"] += path_length
                reachable_overall_stats["lava_steps_sum"] += lava_steps
                reachable_overall_stats["goal_reached_count"] += 1 if reaches_goal else 0
                reachable_overall_stats["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                reachable_overall_stats["risky_diagonal_count"] += 1 if risky_diagonal else 0
    
    # Print detailed information about each environment
    print("\nEnvironment Breakdown:")
    for env_name, counts in env_counts.items():
        floor_states = counts["floor_states"]
        reachable_states = counts["reachable_states"]
        if floor_states > 0:
            print(f"  {env_name}: {reachable_states}/{floor_states} floor states can reach goal without lava ({reachable_states/floor_states:.1%})")
    
    print(f"\nTotal states filtered out: {filtered_lava_start_states} starting on lava, {filtered_unreachable_states} unable to reach goal without lava")
    print(f"Total reachable states: {reachable_overall_stats['total_states']}")
    
    # Process the collected statistics
    
    # Create summary data for each state
    summary_data = {}
    for state_key, stats in reachable_state_stats.items():
        # Skip states with no data
        if stats["count"] == 0:
            continue
        
        # Calculate averages and proportions
        summary_data[state_key] = {
            "lava_cell_proportion": stats["lava_cell_count"] / stats["count"],
            "avg_path_length": stats["path_length_sum"] / stats["count"],
            "avg_lava_steps": stats["lava_steps_sum"] / stats["count"],
            "goal_reached_proportion": stats["goal_reached_count"] / stats["count"],
            "next_cell_lava_proportion": stats["next_cell_lava_count"] / stats["count"],
            "risky_diagonal_proportion": stats["risky_diagonal_count"] / stats["count"]
        }
    
    # Calculate overall summary
    total_states = reachable_overall_stats["total_states"]
    
    if total_states == 0:
        print("No states found that meet filtering criteria (start on floor AND can reach goal without entering lava)")
        return ""
    
    overall_summary = {
        "total_state_instances": total_states,
        "unique_states": len(summary_data),
        "lava_cell_proportion": reachable_overall_stats["lava_cell_count"] / total_states,
        "avg_path_length": reachable_overall_stats["path_length_sum"] / total_states,
        "avg_lava_steps": reachable_overall_stats["lava_steps_sum"] / total_states,
        "goal_reached_proportion": reachable_overall_stats["goal_reached_count"] / total_states,
        "next_cell_lava_proportion": reachable_overall_stats["next_cell_lava_count"] / total_states,
        "risky_diagonal_proportion": reachable_overall_stats["risky_diagonal_count"] / total_states
    }
    
    # Create a dedicated summary directory
    summary_dir = os.path.join(agent_dir, "evaluation_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create the output filename
    filename = "performance_reachable_paths.json"
    output_file = os.path.join(summary_dir, filename)
    
    # Build the data to write
    output_data = {
        "summary_description": "This file contains summary statistics for agent performance across states that start on floor tiles AND can reach the goal without entering lava",
        "overall_summary": overall_summary,
        "statistics": summary_data
    }
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(output_data))
    
    print(f"Created reachable-path summary file: {output_file}")
    print(f"Total states included: {total_states}")
    print(f"Unique states included: {len(summary_data)}")
    
    return output_file

def process_specific_agent(agent_name: str) -> None:
    """
    Process a specific agent directory and generate a reachable-path summary.
    
    Args:
        agent_name (str): Name of the agent directory within Agent_Storage
    """
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    agent_dir = os.path.join(agent_storage_dir, agent_name)
    
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory '{agent_name}' not found in Agent_Storage")
        return
    
    # Generate reachable-path summary
    generate_reachable_path_summary(agent_dir)

def process_all_agents() -> None:
    """
    Process all agent directories in Agent_Storage and generate reachable-path summaries.
    """
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    
    # List all directories in Agent_Storage
    agent_dirs = [os.path.join(agent_storage_dir, d) for d in os.listdir(agent_storage_dir) 
                 if os.path.isdir(os.path.join(agent_storage_dir, d))]
    
    for agent_dir in agent_dirs:
        # Skip directories that don't look like agent directories (e.g., Hyperparameters)
        evaluation_dir = os.path.join(agent_dir, "evaluation_logs")
        if not os.path.exists(evaluation_dir):
            print(f"Skipping {os.path.basename(agent_dir)} - no evaluation logs directory")
            continue
        
        # Generate reachable-path summary
        generate_reachable_path_summary(agent_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate performance summaries for states that can reach the goal without entering lava")
    parser.add_argument("--agent", type=str, help="Name of a specific agent to process (within Agent_Storage)")
    args = parser.parse_args()
    
    if args.agent:
        process_specific_agent(args.agent)
    else:
        process_all_agents() 
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
from Agent_Evaluation.add_metrics import calculate_behavioral_metrics_for_states

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
    
    # If the state isn't in the Dijkstra data, check if we can parse the state coordinates
    try:
        # Extract coordinates and orientation
        x, y, orientation = map(int, state_key.split(','))
        
        # Look for states with the same coordinates but different orientations
        # If any orientation at this position can reach the goal, consider the position reachable
        for o in range(4):  # Check all 4 orientations
            test_key = f"{x},{y},{o}"
            if test_key in statistics and statistics[test_key].get("goal_reached_proportion", 0) > 0:
                return True
    except:
        pass  # If we can't parse the state, assume it's not reachable
    
    # If we couldn't find evidence that the state can reach the goal, assume it cannot
    return False

def generate_unreachable_path_summary(agent_dir: str) -> str:
    """
    Generate a performance summary for states that cannot reach the goal without entering lava.
    Only includes states where the agent starts on a floor tile (not lava).
    
    Args:
        agent_dir (str): Path to the agent directory
        
    Returns:
        str: Path to the generated summary file
    """
    print(f"Generating unreachable-path summary for agent: {os.path.basename(agent_dir)}")
    
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
    
    # Initialize data structures to store statistics for unreachable-without-lava states
    unreachable_state_stats = defaultdict(lambda: {
        "count": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    })
    
    # Track overall statistics for unreachable-without-lava states
    unreachable_overall_stats = {
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
    filtered_reachable_states = 0
    
    # Additional debugging info
    env_counts = {}
    
    # Process each JSON file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        env_name = os.path.splitext(file_name)[0]  # Get environment name without extension
        env_counts[env_name] = {"floor_states": 0, "unreachable_states": 0}
        
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
        
        # Create a set of unreachable states according to Dijkstra
        unreachable_states = set()
        for state_key, metrics in dijkstra_standard.items():
            # Skip comment and mode keys
            if state_key.startswith("__"):
                continue
                
            # Check if goal is reachable - metrics[3] is reaches_goal
            if isinstance(metrics, list) and len(metrics) > 3 and metrics[3] == 0:
                unreachable_states.add(state_key)
        
        print(f"  Found {len(unreachable_states)} unreachable states in Dijkstra data")
        
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
                # If it's in the unreachable_states set, it cannot reach the goal
                if state_key not in unreachable_states:
                    filtered_reachable_states += 1
                    continue
                
                # Count this as an unreachable state
                env_counts[env_name]["unreachable_states"] += 1
                
                # Now extract the remaining metrics for states that pass both filters
                path_length = metrics[1]
                lava_steps = metrics[2]
                reaches_goal = metrics[3]
                next_cell_is_lava = metrics[4]
                risky_diagonal = metrics[5]
                
                # Update per-state statistics
                unreachable_state_stats[state_key]["count"] += 1
                unreachable_state_stats[state_key]["lava_cell_count"] += 1 if cell_type == "lava" else 0
                unreachable_state_stats[state_key]["path_length_sum"] += path_length
                unreachable_state_stats[state_key]["lava_steps_sum"] += lava_steps
                unreachable_state_stats[state_key]["goal_reached_count"] += 1 if reaches_goal else 0
                unreachable_state_stats[state_key]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                unreachable_state_stats[state_key]["risky_diagonal_count"] += 1 if risky_diagonal else 0
                
                # Update overall statistics
                unreachable_overall_stats["total_states"] += 1
                unreachable_overall_stats["lava_cell_count"] += 1 if cell_type == "lava" else 0
                unreachable_overall_stats["path_length_sum"] += path_length
                unreachable_overall_stats["lava_steps_sum"] += lava_steps
                unreachable_overall_stats["goal_reached_count"] += 1 if reaches_goal else 0
                unreachable_overall_stats["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                unreachable_overall_stats["risky_diagonal_count"] += 1 if risky_diagonal else 0
    
    # Print detailed information about each environment
    print("\nEnvironment Breakdown:")
    for env_name, counts in env_counts.items():
        floor_states = counts["floor_states"]
        unreachable_states = counts["unreachable_states"]
        if floor_states > 0:
            print(f"  {env_name}: {unreachable_states}/{floor_states} floor states cannot reach goal without lava ({unreachable_states/floor_states:.1%})")
    
    print(f"\nTotal states filtered out: {filtered_lava_start_states} starting on lava, {filtered_reachable_states} can reach goal without lava")
    print(f"Total unreachable states: {unreachable_overall_stats['total_states']}")
    
    # Process the collected statistics
    
    # Create summary data for each state
    summary_data = {}
    for state_key, stats in unreachable_state_stats.items():
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
    
    # Calculate behavioral metrics for unreachable states
    print("Calculating behavioral metrics for unreachable path states...")
    
    behavioral_overall_counts = {
        "total_states": 0,
        "blocked_count": 0,
        "chose_safety_count": 0,
        "chose_safety_optimally_count": 0,
        "into_wall_count": 0
    }
    
    behavioral_per_state_counts = defaultdict(lambda: {
        "count": 0,
        "blocked_count": 0,
        "chose_safety_count": 0,
        "chose_safety_optimally_count": 0,
        "into_wall_count": 0
    })
    
    # Process each environment's unreachable states
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        env_name = os.path.splitext(file_name)[0]
        
        if file_name.startswith("performance_"):
            continue
        
        # Load agent performance data
        try:
            with open(json_file, 'r') as f:
                agent_data = json.load(f)
        except json.JSONDecodeError:
            continue
        
        if "performance" not in agent_data:
            continue
        
        # Get agent performance data
        if "agent" in agent_data["performance"]:
            agent_perf_data = agent_data["performance"]["agent"]
        else:
            agent_perf_data = agent_data["performance"]
        
        # Load corresponding Dijkstra data
        dijkstra_file = os.path.join(project_root, "Behaviour_Specification", "Evaluations", file_name)
        
        if not os.path.exists(dijkstra_file):
            continue
        
        try:
            with open(dijkstra_file, 'r') as f:
                env_dijkstra_data = json.load(f)
        except json.JSONDecodeError:
            continue
        
        if "performance" not in env_dijkstra_data:
            continue
            
        # Get the dijkstra performance data in the format expected by calculate_behavioral_metrics_for_states
        dijkstra_perf_data = env_dijkstra_data["performance"]
        
        # Define filter for unreachable states only (using same logic as main summary)
        def unreachable_filter(state_key: str, metrics: List) -> bool:
            # Skip states where agent starts on lava
            if len(metrics) >= 1 and metrics[0] == "lava":
                return False
                
            # Check if this state is unreachable (same logic as main summary)
            # Use dijkstra standard ruleset to check if goal is reachable
            if "standard" in dijkstra_perf_data and state_key in dijkstra_perf_data["standard"]:
                dijkstra_metrics = dijkstra_perf_data["standard"][state_key]
                # metrics[3] is reaches_goal - unreachable if it's 0
                if isinstance(dijkstra_metrics, list) and len(dijkstra_metrics) > 3:
                    return dijkstra_metrics[3] == 0
            
            return False
        
        # Calculate behavioral metrics for this environment (unreachable states only)
        env_overall, env_per_state = calculate_behavioral_metrics_for_states(
            agent_perf_data, dijkstra_perf_data, unreachable_filter
        )
        
        # Aggregate overall counts
        behavioral_overall_counts["total_states"] += env_overall["total_states"]
        behavioral_overall_counts["blocked_count"] += env_overall["blocked_count"]
        behavioral_overall_counts["chose_safety_count"] += env_overall["chose_safety_count"]
        behavioral_overall_counts["chose_safety_optimally_count"] += env_overall["chose_safety_optimally_count"]
        behavioral_overall_counts["into_wall_count"] += env_overall["into_wall_count"]
        
        # Aggregate per-state counts
        for state_key, state_counts in env_per_state.items():
            behavioral_per_state_counts[state_key]["count"] += state_counts["count"]
            behavioral_per_state_counts[state_key]["blocked_count"] += state_counts["blocked_count"]
            behavioral_per_state_counts[state_key]["chose_safety_count"] += state_counts["chose_safety_count"]
            behavioral_per_state_counts[state_key]["chose_safety_optimally_count"] += state_counts["chose_safety_optimally_count"]
            behavioral_per_state_counts[state_key]["into_wall_count"] += state_counts["into_wall_count"]
    
    print(f"Calculated behavioral metrics for {behavioral_overall_counts['total_states']} unreachable state instances")

    # Calculate overall summary
    total_states = unreachable_overall_stats["total_states"]
    
    if total_states == 0:
        print("No states found that meet filtering criteria (start on floor AND cannot reach goal without entering lava)")
        return ""
    
    overall_summary = {
        "total_state_instances": total_states,
        "unique_states": len(summary_data),
        "lava_cell_proportion": unreachable_overall_stats["lava_cell_count"] / total_states,
        "avg_path_length": unreachable_overall_stats["path_length_sum"] / total_states,
        "avg_lava_steps": unreachable_overall_stats["lava_steps_sum"] / total_states,
        "goal_reached_proportion": unreachable_overall_stats["goal_reached_count"] / total_states,
        "next_cell_lava_proportion": unreachable_overall_stats["next_cell_lava_count"] / total_states,
        "risky_diagonal_proportion": unreachable_overall_stats["risky_diagonal_count"] / total_states
    }
    
    # Add behavioral metrics to overall summary
    if behavioral_overall_counts["total_states"] > 0:
        overall_summary["blocked_proportion"] = behavioral_overall_counts["blocked_count"] / behavioral_overall_counts["total_states"]
        overall_summary["chose_safety_proportion"] = behavioral_overall_counts["chose_safety_count"] / behavioral_overall_counts["total_states"]
        overall_summary["chose_safety_optimally_proportion"] = behavioral_overall_counts["chose_safety_optimally_count"] / behavioral_overall_counts["total_states"]
        overall_summary["into_wall_proportion"] = behavioral_overall_counts["into_wall_count"] / behavioral_overall_counts["total_states"]

    # Add behavioral metrics to individual state statistics
    for state_key in summary_data:
        if state_key in behavioral_per_state_counts and behavioral_per_state_counts[state_key]["count"] > 0:
            count = behavioral_per_state_counts[state_key]["count"]
            summary_data[state_key]["blocked_proportion"] = behavioral_per_state_counts[state_key]["blocked_count"] / count
            summary_data[state_key]["chose_safety_proportion"] = behavioral_per_state_counts[state_key]["chose_safety_count"] / count
            summary_data[state_key]["chose_safety_optimally_proportion"] = behavioral_per_state_counts[state_key]["chose_safety_optimally_count"] / count
            summary_data[state_key]["into_wall_proportion"] = behavioral_per_state_counts[state_key]["into_wall_count"] / count

    # Create a dedicated summary directory
    summary_dir = os.path.join(agent_dir, "evaluation_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create the output filename
    filename = "performance_unreachable_paths.json"
    output_file = os.path.join(summary_dir, filename)
    
    # Build the data to write
    output_data = {
        "summary_description": "This file contains summary statistics for agent performance across states that start on floor tiles but CANNOT reach the goal without entering lava",
        "overall_summary": overall_summary,
        "statistics": summary_data
    }
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(output_data))
    
    print(f"Created unreachable-path summary file: {output_file}")
    print(f"Total states included: {total_states}")
    print(f"Unique states included: {len(summary_data)}")
    
    return output_file

def process_specific_agent(agent_name: str) -> None:
    """
    Process a specific agent directory and generate a unreachable-path summary.
    
    Args:
        agent_name (str): Name of the agent directory within Agent_Storage
    """
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    agent_dir = os.path.join(agent_storage_dir, agent_name)
    
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory '{agent_name}' not found in Agent_Storage")
        return
    
    # Generate unreachable-path summary
    generate_unreachable_path_summary(agent_dir)

def process_all_agents() -> None:
    """
    Process all agent directories in Agent_Storage and generate unreachable-path summaries.
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
        
        # Generate unreachable-path summary
        generate_unreachable_path_summary(agent_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate performance summaries for states that cannot reach the goal without entering lava")
    parser.add_argument("--agent", type=str, help="Name of a specific agent to process (within Agent_Storage)")
    args = parser.parse_args()
    
    if args.agent:
        process_specific_agent(args.agent)
    else:
        process_all_agents() 
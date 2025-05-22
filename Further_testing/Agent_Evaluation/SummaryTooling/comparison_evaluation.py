import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional, Tuple, Union
import copy

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Default number of steps to use when a goal is unreachable
# This is used instead of 0 for metrics when the agent can't reach the goal
DEFAULT_UNREACHABLE_STEPS = 150


def format_json_with_compact_arrays(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    Custom JSON formatter that keeps arrays on a single line while maintaining readability.
    
    Args:
        data (Union[Dict, List, Any]): The data to format
        indent (int): Indentation level
        
    Returns:
        str: Formatted JSON string
    """
    import json
    
    return json.dumps(data, indent=indent)


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


def get_agent_evaluation_summaries(agent_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Find and load evaluation summaries for a given agent.
    
    Args:
        agent_path (str): Path to the agent directory
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of evaluation summary data for the agent,
        with keys derived from the file names
    """
    summary_path = os.path.join(agent_path, "evaluation_summary")
    
    if not os.path.exists(summary_path):
        print(f"Warning: No evaluation summary directory found at {summary_path}")
        return {}
    
    # Find all JSON files in the summary directory
    json_files = glob.glob(os.path.join(summary_path, "*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {summary_path}")
        return {}
    
    # Load each JSON file with a descriptive key based on the filename
    summaries = {}
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        file_key = os.path.splitext(file_name)[0]  # Remove the extension
        
        # Create a more descriptive key based on the file name
        if "no_lava" in file_key.lower():
            summary_key = "without_lava"
        elif "all_states" in file_key.lower() or "with_lava" in file_key.lower():
            summary_key = "with_lava"
        else:
            # If we can't determine a descriptive key, use the file name without extension
            summary_key = file_key
        
        print(f"  Loading summary from {file_name} as '{summary_key}'")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                summaries[summary_key] = data
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON file {file_name}")
            continue
    
    return summaries


def calculate_performance_difference(
    agent_summary: Dict[str, Any],
    dijkstra_results: Dict[str, Any],
    ruleset_name: str
) -> Dict[str, Any]:
    """
    Calculate the difference between agent performance and Dijkstra baseline.
    Only includes the overall summary differences, not the state-by-state analysis.
    
    Args:
        agent_summary (Dict[str, Any]): Agent performance summary
        dijkstra_results (Dict[str, Any]): Dijkstra performance results
        ruleset_name (str): Name of the ruleset to compare
        
    Returns:
        Dict[str, Any]: Difference between agent and Dijkstra performance (overall summary only)
    """
    # Check if the ruleset exists in Dijkstra results
    if ruleset_name not in dijkstra_results.get("rulesets", {}):
        print(f"Warning: Ruleset '{ruleset_name}' not found in Dijkstra results")
        return {}
    
    # Get Dijkstra results for the specified ruleset
    dijkstra_ruleset = dijkstra_results["rulesets"][ruleset_name]
    
    # Initialize result dictionary - ONLY including overall_summary, not statistics
    difference = {
        "description": f"Performance difference (Agent minus Dijkstra) for ruleset {ruleset_name}",
        "overall_summary": {}
    }
    
    # Calculate difference for overall summary if both have it
    if "overall_summary" in agent_summary and "overall_summary" in dijkstra_ruleset:
        difference["overall_summary"] = calculate_metric_differences(
            agent_summary["overall_summary"],
            dijkstra_ruleset["overall_summary"]
        )
    
    # Note: Removed state-by-state analysis (statistics) as requested
    
    return difference


def calculate_metric_differences(agent_metrics: Dict[str, Any], dijkstra_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the difference between agent and Dijkstra metrics.
    Uses DEFAULT_UNREACHABLE_STEPS for metrics when the goal is unreachable.
    
    Args:
        agent_metrics (Dict[str, Any]): Agent performance metrics
        dijkstra_metrics (Dict[str, Any]): Dijkstra performance metrics
        
    Returns:
        Dict[str, Any]: Differences between metrics
    """
    differences = {}
    
    # Common metrics we expect to see
    expected_metrics = [
        "avg_path_length", 
        "avg_lava_steps",
        "goal_reached_proportion", 
        "next_cell_lava_proportion",
        "risky_diagonal_proportion"
    ]
    
    # Calculate differences for all metrics that appear in both dictionaries
    for metric in expected_metrics:
        if metric in agent_metrics and metric in dijkstra_metrics:
            # If this is a path length metric and the value is 0 (unreachable goal),
            # use the default unreachable steps value instead
            if metric == "avg_path_length" and dijkstra_metrics[metric] == 0:
                # If the Dijkstra value is 0, use the default value for the calculation
                differences[metric] = agent_metrics[metric] - DEFAULT_UNREACHABLE_STEPS
            else:
                differences[metric] = agent_metrics[metric] - dijkstra_metrics[metric]
    
    # Also include any other metrics that might be in both
    for metric in agent_metrics:
        if metric not in differences and metric in dijkstra_metrics:
            # Make sure they're numeric values before attempting to subtract
            if isinstance(agent_metrics[metric], (int, float)) and isinstance(dijkstra_metrics[metric], (int, float)):
                # If this is a path-related metric and the Dijkstra value is 0, use the default value
                if "path" in metric.lower() and dijkstra_metrics[metric] == 0:
                    differences[metric] = agent_metrics[metric] - DEFAULT_UNREACHABLE_STEPS
                else:
                    differences[metric] = agent_metrics[metric] - dijkstra_metrics[metric]
    
    return differences


def generate_comparison_evaluation(agent_path: str) -> Dict[str, Any]:
    """
    Generate comparison evaluation for an agent, comparing its performance
    against Dijkstra baselines for all rulesets.
    
    Args:
        agent_path (str): Path to the agent directory
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    agent_name = os.path.basename(agent_path)
    print(f"Generating comparison evaluation for agent: {agent_name}")
    
    # Load Dijkstra results
    dijkstra_results = load_dijkstra_results()
    if not dijkstra_results:
        print("Error: Failed to load Dijkstra results. Aborting.")
        return {}
    
    # Get agent evaluation summaries - now returns a dictionary with descriptive keys
    agent_summaries = get_agent_evaluation_summaries(agent_path)
    if not agent_summaries:
        print(f"Error: No evaluation summaries found for {agent_name}. Aborting.")
        return {}
    
    # Get rulesets from Dijkstra results
    rulesets = dijkstra_results.get("rulesets", {}).keys()
    if not rulesets:
        print("Error: No rulesets found in Dijkstra results. Aborting.")
        return {}
    
    print(f"Found {len(agent_summaries)} summaries and {len(rulesets)} rulesets")
    
    # Final evaluation dictionary
    final_eval = {
        "agent_name": agent_name,
        "description": f"Comparison of agent performance against Dijkstra baselines (overall summaries only, using {DEFAULT_UNREACHABLE_STEPS} steps for unreachable goals)",
        "comparisons": {}
    }
    
    # Generate comparisons for each agent summary and ruleset combination
    # Using the descriptive keys from the summaries dictionary
    for summary_key, agent_summary in agent_summaries.items():
        final_eval["comparisons"][summary_key] = {}
        
        for ruleset in rulesets:
            # Calculate difference between agent and Dijkstra for this ruleset
            difference = calculate_performance_difference(agent_summary, dijkstra_results, ruleset)
            
            if difference:
                final_eval["comparisons"][summary_key][ruleset] = difference
    
    return final_eval


def save_comparison_results(agent_path: str, final_eval: Dict[str, Any]) -> None:
    """
    Save comparison results to a JSON file in the agent's directory.
    
    Args:
        agent_path (str): Path to the agent directory
        final_eval (Dict[str, Any]): Comparison results to save
    """
    output_file = os.path.join(agent_path, "final_eval.json")
    
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(final_eval))
    
    print(f"Saved comparison results to {output_file}")


def process_all_agents() -> None:
    """
    Process all agent directories in Agent_Storage and generate comparison evaluations.
    """
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    
    # List all directories in Agent_Storage
    agent_dirs = [os.path.join(agent_storage_dir, d) for d in os.listdir(agent_storage_dir) 
                 if os.path.isdir(os.path.join(agent_storage_dir, d))]
    
    for agent_dir in agent_dirs:
        # Skip directories that don't look like agent directories (e.g., Hyperparameters)
        evaluation_dir = os.path.join(agent_dir, "evaluation_summary")
        if not os.path.exists(evaluation_dir):
            print(f"Skipping {os.path.basename(agent_dir)} - no evaluation summary directory")
            continue
        
        # Generate and save comparison evaluation
        final_eval = generate_comparison_evaluation(agent_dir)
        if final_eval:
            save_comparison_results(agent_dir, final_eval)


def process_specific_agent(agent_name: str) -> None:
    """
    Process a specific agent directory and generate a comparison evaluation.
    
    Args:
        agent_name (str): Name of the agent directory within Agent_Storage
    """
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    agent_dir = os.path.join(agent_storage_dir, agent_name)
    
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory '{agent_name}' not found in Agent_Storage")
        return
    
    # Generate and save comparison evaluation
    final_eval = generate_comparison_evaluation(agent_dir)
    if final_eval:
        save_comparison_results(agent_dir, final_eval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison evaluations between agent and Dijkstra performance")
    parser.add_argument("--agent", type=str, help="Name of a specific agent to process (within Agent_Storage)")
    args = parser.parse_args()
    
    if args.agent:
        process_specific_agent(args.agent)
    else:
        process_all_agents() 
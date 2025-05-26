#!/usr/bin/env python3
import os
import sys
import json
import glob
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import the JSON formatting function from results_processing
from Agent_Evaluation.AgentTooling.results_processing import format_json_with_compact_arrays

def load_evaluation_summary(agent_dir: str, summary_type: str) -> Optional[Dict[str, Any]]:
    """
    Load a specific evaluation summary file for an agent.
    
    Args:
        agent_dir (str): Path to the agent directory
        summary_type (str): Type of summary to load (e.g., "lava_only", "reachable_paths")
        
    Returns:
        Optional[Dict[str, Any]]: The loaded summary data, or None if the file doesn't exist
    """
    # Map summary_type to actual filename
    filename_map = {
        "lava_only": "performance_lava_only.json",
        "reachable_paths": "performance_reachable_paths.json",
        "unreachable_paths": "performance_unreachable_paths.json",
        "all_states": "performance_all_states.json"
    }
    
    if summary_type not in filename_map:
        print(f"Warning: Unknown summary type '{summary_type}'")
        return None
    
    summary_file = os.path.join(agent_dir, "evaluation_summary", filename_map[summary_type])
    
    if not os.path.exists(summary_file):
        # print(f"Warning: Summary file '{summary_file}' does not exist")
        return None
    
    try:
        with open(summary_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file '{summary_file}'")
        return None
    except Exception as e:
        print(f"Error loading summary file '{summary_file}': {e}")
        return None

def collect_agent_versions(base_dir: str, agent_type: str) -> List[str]:
    """
    Find all versions of a specific agent type.
    
    Args:
        base_dir (str): Base directory containing agent directories
        agent_type (str): Type of agent to find versions for (e.g., "Standard")
        
    Returns:
        List[str]: List of paths to agent version directories
    """
    # Look for directories matching "{agent_type}-v*"
    pattern = os.path.join(base_dir, f"{agent_type}-v*")
    agent_dirs = glob.glob(pattern)
    
    # Filter to ensure they're actually directories
    agent_dirs = [d for d in agent_dirs if os.path.isdir(d)]
    
    # Sort by version number
    agent_dirs.sort(key=lambda d: int(d.split("-v")[-1]) if d.split("-v")[-1].isdigit() else float('inf'))
    
    return agent_dirs

def extract_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract the metrics from a summary's overall_summary section.
    
    Args:
        summary (Dict[str, Any]): The summary data
        
    Returns:
        Dict[str, float]: Dictionary of metric names to values
    """
    if not summary or "overall_summary" not in summary:
        return {}
    
    metrics = {}
    for key, value in summary["overall_summary"].items():
        # Only include numeric metrics
        if isinstance(value, (int, float)) and key != "total_state_instances" and key != "unique_states":
            metrics[key] = value
    
    return metrics

def calculate_statistics(metric_values: List[float]) -> Dict[str, float]:
    """
    Calculate mean and standard deviation for a list of metric values.
    
    Args:
        metric_values (List[float]): List of values for a specific metric
        
    Returns:
        Dict[str, float]: Dictionary with mean and std_dev
    """
    if not metric_values:
        return {"mean": 0.0, "std_dev": 0.0, "count": 0}
    
    return {
        "mean": float(np.mean(metric_values)),
        "std_dev": float(np.std(metric_values)),
        "count": len(metric_values)
    }

def generate_agent_category_summary(agent_type_dir: str) -> Optional[str]:
    """
    Generate a summary of performance metrics across all versions of an agent type.
    
    Args:
        agent_type_dir (str): Path to the agent type directory
        
    Returns:
        Optional[str]: Path to the generated summary file, or None if failed
    """
    agent_type = os.path.basename(agent_type_dir)
    print(f"Generating category summary for agent type: {agent_type}")
    
    # Find all versions of this agent
    agent_version_dirs = collect_agent_versions(agent_type_dir, agent_type)
    if not agent_version_dirs:
        print(f"No agent versions found for type '{agent_type}'")
        return None
    
    print(f"Found {len(agent_version_dirs)} agent versions")
    
    # Dictionary to store metrics for each summary type
    metrics_by_summary = defaultdict(lambda: defaultdict(list))
    
    # Dictionary to keep track of how many agents had each summary type
    summary_counts = defaultdict(int)
    
    # Process each agent version
    for agent_dir in agent_version_dirs:
        agent_version = os.path.basename(agent_dir)
        print(f"Processing agent version: {agent_version}")
        
        # Try to load each summary type
        for summary_type in ["lava_only", "reachable_paths", "unreachable_paths", "all_states"]:
            summary = load_evaluation_summary(agent_dir, summary_type)
            if summary:
                summary_counts[summary_type] += 1
                metrics = extract_metrics_from_summary(summary)
                
                # Add each metric value to the list for this summary type
                for metric_name, metric_value in metrics.items():
                    metrics_by_summary[summary_type][metric_name].append(metric_value)
    
    # Calculate statistics for each metric in each summary type
    summary_stats = {}
    for summary_type, metrics in metrics_by_summary.items():
        summary_stats[summary_type] = {
            "metrics": {},
            "agent_count": summary_counts[summary_type]
        }
        
        for metric_name, values in metrics.items():
            summary_stats[summary_type]["metrics"][metric_name] = calculate_statistics(values)
    
    # Generate the output file
    output_file = os.path.join(agent_type_dir, "average_performance.json")
    
    # Create the output data
    output_data = {
        "agent_type": agent_type,
        "total_versions": len(agent_version_dirs),
        "description": f"Average performance metrics across all versions of {agent_type}",
        "summaries": summary_stats
    }
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(output_data))
    
    print(f"Created category summary file: {output_file}")
    return output_file

def process_agent_category(agent_category_path: str) -> None:
    """
    Process a specific agent category (e.g., LavaTests/Standard) and generate a summary.
    
    Args:
        agent_category_path (str): Path to the agent category directory
    """
    agent_type = os.path.basename(agent_category_path)
    print(f"Processing agent category: {agent_type}")
    
    # Check if this is actually a directory containing agent versions
    if not os.path.isdir(agent_category_path):
        print(f"Error: '{agent_category_path}' is not a directory")
        return
    
    # Check if there are any version directories inside
    has_versions = any(os.path.isdir(os.path.join(agent_category_path, d)) and d.startswith(f"{agent_type}-v") 
                      for d in os.listdir(agent_category_path))
    
    if not has_versions:
        print(f"No agent versions found in {agent_category_path}")
        return
    
    # Generate the category summary
    generate_agent_category_summary(agent_category_path)

def process_agent_types(base_dir: str) -> None:
    """
    Process all agent types in a base directory (e.g., LavaTests).
    
    Args:
        base_dir (str): Path to the base directory containing agent type directories
    """
    print(f"Processing agent types in: {base_dir}")
    
    # Find all directories in the base directory
    agent_type_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
    
    for agent_type_dir in agent_type_dirs:
        agent_type = os.path.basename(agent_type_dir)
        
        # Skip directories that don't look like agent types (e.g., "logs")
        if agent_type.lower() in ["logs", "hyperparameters"]:
            continue
        
        # Process this agent type
        process_agent_category(agent_type_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate category summaries for agent types")
    parser.add_argument("--base", type=str, help="Base directory containing agent types (e.g., Agent_Storage/LavaTests)")
    parser.add_argument("--agent", type=str, help="Specific agent type to process (e.g., Standard)")
    args = parser.parse_args()
    
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    
    if args.agent and args.base:
        # Process a specific agent type in a specific base directory
        agent_category_path = os.path.join(agent_storage_dir, args.base, args.agent)
        process_agent_category(agent_category_path)
    elif args.agent:
        # Look for the agent type in various directories
        found = False
        for base_dir in os.listdir(agent_storage_dir):
            base_path = os.path.join(agent_storage_dir, base_dir)
            if os.path.isdir(base_path):
                agent_path = os.path.join(base_path, args.agent)
                if os.path.isdir(agent_path):
                    process_agent_category(agent_path)
                    found = True
        
        if not found:
            print(f"Error: Agent type '{args.agent}' not found in Agent_Storage")
    elif args.base:
        # Process all agent types in a specific base directory
        base_path = os.path.join(agent_storage_dir, args.base)
        if os.path.isdir(base_path):
            process_agent_types(base_path)
        else:
            print(f"Error: Base directory '{args.base}' not found in Agent_Storage")
    else:
        # Process all agent types in all directories
        for base_dir in os.listdir(agent_storage_dir):
            base_path = os.path.join(agent_storage_dir, base_dir)
            if os.path.isdir(base_path) and base_dir not in ["Hyperparameters"]:
                process_agent_types(base_path)

import os
import json
from collections import defaultdict
from typing import Dict, List, Any
import pandas as pd

# Directory to search for average_performance.json files
base_dir = "Agent_Storage"
# Directory to save leaderboard files
output_dir = "Leaderboard_Status/new"
# Folders to exclude
exclude_folders = ["Hyperparameters"]
# Summary types to generate leaderboards for
summary_types = ["lava_only", "reachable_paths", "unreachable_paths", "all_states"]
# Metrics to include in leaderboards
metrics = [
    "lava_cell_proportion",
    "avg_path_length",
    "avg_lava_steps",
    "goal_reached_proportion",
    "next_cell_lava_proportion",
    "risky_diagonal_proportion"
]

def find_average_performance_files() -> List[str]:
    """Recursively find all average_performance.json files."""
    json_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip excluded folders
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        
        if "average_performance.json" in files:
            json_files.append(os.path.join(root, "average_performance.json"))
    
    return json_files

def extract_agent_class(file_path: str) -> str:
    """Extract the agent class from the file path."""
    # Remove base directory and average_performance.json from path
    relative_path = file_path.replace(base_dir + "/", "").replace("/average_performance.json", "")
    return relative_path

def load_performance_data(json_files: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load performance data from JSON files."""
    data = {}
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                perf_data = json.load(f)
                
            agent_class = extract_agent_class(file_path)
            data[agent_class] = perf_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def generate_leaderboards(performance_data: Dict[str, Dict[str, Dict[str, Any]]]):
    """Generate leaderboards for each summary type and metric."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For each summary type, create a leaderboard file
    for summary_type in summary_types:
        # Collect data for each metric
        metric_data = defaultdict(list)
        
        for agent_class, agent_data in performance_data.items():
            if "summaries" in agent_data and summary_type in agent_data["summaries"]:
                summary = agent_data["summaries"][summary_type]
                
                if "metrics" in summary:
                    for metric in metrics:
                        if metric in summary["metrics"]:
                            metric_info = summary["metrics"][metric]
                            metric_data[metric].append({
                                "agent_class": agent_class,
                                "mean": metric_info.get("mean", 0),
                                "std_dev": metric_info.get("std_dev", 0)
                            })
        
        # Create leaderboard file for this summary type
        with open(f"{output_dir}/{summary_type}_leaderboard.md", 'w') as f:
            f.write(f"# {summary_type.replace('_', ' ').title()} Leaderboard\n\n")
            
            for metric in metrics:
                f.write(f"## {metric.replace('_', ' ').title()}\n\n")
                
                # Sort by mean value (ascending or descending depending on metric)
                # For most metrics, higher is better, except for path length and lava steps
                reverse = metric not in ["avg_path_length", "avg_lava_steps"]
                sorted_data = sorted(metric_data[metric], key=lambda x: x["mean"], reverse=reverse)
                
                # Format as a table
                f.write("| Rank | Agent Class | Mean | Std Dev |\n")
                f.write("|------|------------|------|--------|\n")
                
                for i, entry in enumerate(sorted_data, 1):
                    f.write(f"| {i} | {entry['agent_class']} | {entry['mean']:.4f} | {entry['std_dev']:.4f} |\n")
                
                f.write("\n")

def main():
    print("Finding average_performance.json files...")
    json_files = find_average_performance_files()
    print(f"Found {len(json_files)} files")
    
    print("Loading performance data...")
    performance_data = load_performance_data(json_files)
    print(f"Loaded data for {len(performance_data)} agent classes")
    
    print("Generating leaderboards...")
    generate_leaderboards(performance_data)
    print("Leaderboards generated successfully!")

if __name__ == "__main__":
    main() 
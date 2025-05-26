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
    """Recursively find all average_performance.json files"""
    performance_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip excluded folders
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        
        for file in files:
            if file == "average_performance.json":
                file_path = os.path.join(root, file)
                performance_files.append(file_path)
    
    return performance_files

def parse_performance_file(file_path: str) -> Dict[str, Any]:
    """Parse an average_performance.json file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract agent class (folder path relative to Agent_Storage)
    agent_class = os.path.dirname(file_path).replace(base_dir + os.sep, '')
    
    return {
        "agent_class": agent_class,
        "agent_type": data.get("agent_type", "unknown"),
        "summaries": data.get("summaries", {})
    }

def create_leaderboards():
    """Generate leaderboards for each summary type and metric"""
    performance_files = find_average_performance_files()
    
    if not performance_files:
        print("No average_performance.json files found.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse all performance files
    parsed_data = [parse_performance_file(file) for file in performance_files]
    
    # Generate leaderboards for each summary type
    for summary_type in summary_types:
        # Create a leaderboard file for this summary type
        leaderboard_path = os.path.join(output_dir, f"{summary_type}_leaderboard.md")
        
        with open(leaderboard_path, 'w') as f:
            f.write(f"# Leaderboard for {summary_type}\n\n")
            
            # Generate a table for each metric
            for metric in metrics:
                f.write(f"## {metric}\n\n")
                
                # Collect data for this metric
                metric_data = []
                for agent_data in parsed_data:
                    agent_class = agent_data["agent_class"]
                    summaries = agent_data.get("summaries", {})
                    
                    if summary_type in summaries and "metrics" in summaries[summary_type]:
                        if metric in summaries[summary_type]["metrics"]:
                            metric_info = summaries[summary_type]["metrics"][metric]
                            metric_data.append({
                                "Agent Class": agent_class,
                                "Mean": metric_info.get("mean", 0),
                                "Std Dev": metric_info.get("std_dev", 0),
                                "Count": metric_info.get("count", 0)
                            })
                
                if not metric_data:
                    f.write(f"No data available for {metric}.\n\n")
                    continue
                
                # Convert to DataFrame for easier sorting and formatting
                df = pd.DataFrame(metric_data)
                # Sort by mean value (descending for most metrics, but some might need ascending)
                if metric in ["avg_lava_steps"]:
                    # For these metrics, lower is better
                    df = df.sort_values(by="Mean", ascending=True)
                else:
                    # For most metrics, higher is better
                    df = df.sort_values(by="Mean", ascending=False)
                
                # Format table
                table = "| Agent Class | Mean | Std Dev | Count |\n"
                table += "|------------|------|---------|-------|\n"
                
                for _, row in df.iterrows():
                    agent_class = row["Agent Class"]
                    mean = f"{row['Mean']:.4f}"
                    std_dev = f"{row['Std Dev']:.4f}"
                    count = str(int(row["Count"]))
                    
                    table += f"| {agent_class} | {mean} | {std_dev} | {count} |\n"
                
                f.write(table + "\n\n")
    
    print(f"Leaderboards generated in {output_dir}")

if __name__ == "__main__":
    create_leaderboards() 
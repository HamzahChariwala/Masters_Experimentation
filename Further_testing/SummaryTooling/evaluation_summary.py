import os
import sys
import json
import glob
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def format_json_with_compact_arrays(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    Custom JSON formatter that keeps arrays on a single line,
    except for nested arrays in the environment layout which are kept on separate lines.
    
    Args:
        data (Union[Dict, List, Any]): The data to format
        indent (int): Indentation level
        
    Returns:
        str: Formatted JSON string
    """
    # First, convert to a JSON string with regular indentation
    json_str = json.dumps(data, indent=indent)
    
    # Function to replace array content with single-line format
    def compact_array(match):
        content = match.group(1).strip()
        
        # Skip empty arrays
        if not content:
            return "[]"
        
        # Check if this is a nested array (contains at least one '[')
        if '[' in content:
            # This might be the environment layout - handle nested arrays specially
            if '"layout"' in json_str[max(0, match.start()-20):match.start()]:
                # For layout, keep each nested array on its own line
                lines = content.split('\n')
                processed_lines = []
                
                # Process each line - might be a nested array or just part of one
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('[') and stripped.endswith(','):
                        # This is a complete nested array with a trailing comma
                        nested_content = stripped[1:-1]  # Remove [ and ,
                        # Compact the nested array's contents
                        processed_lines.append(f"[{compact_nested_array(nested_content)}],")
                    elif stripped.startswith('[') and stripped.endswith(']'):
                        # This is a complete nested array without a trailing comma
                        nested_content = stripped[1:-1]  # Remove [ and ]
                        # Compact the nested array's contents
                        processed_lines.append(f"[{compact_nested_array(nested_content)}]")
                    else:
                        # This is part of an array - keep as is
                        processed_lines.append(line)
                
                # Join with newlines to keep the layout structure
                indentation = ' ' * indent
                joined = '\n'.join([f"{indentation}{line}" for line in processed_lines])
                return f"[\n{joined}\n{' ' * (indent-2)}]"
            else:
                # Handle any other nested arrays normally - compact everything
                return f"[{compact_nested_array(content)}]"
        else:
            # Regular (non-nested) array - put on a single line
            return f"[{compact_nested_array(content)}]"
    
    # Function to compact the content of a nested array
    def compact_nested_array(content):
        # Remove all newlines and extra spaces
        compacted = re.sub(r'\s+', ' ', content).strip()
        # Remove spaces after commas for further compacting
        compacted = re.sub(r',\s+', ', ', compacted)
        return compacted
    
    # Replace all arrays with their compacted versions
    # This regex matches arrays with their content: [...content...]
    pattern = r'\[\n([\s\S]*?)\n\s*\]'
    result = re.sub(pattern, compact_array, json_str)
    
    return result


def process_evaluation_logs(agent_dirs: Optional[List[str]] = None, 
                          save_results: bool = True,
                          output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Process all evaluation logs in the specified agent directories.
    For each JSON file that doesn't have a 'performance' key, generate summary statistics.
    
    Args:
        agent_dirs (Optional[List[str]]): List of agent directory paths to process.
            If None, process all directories in Agent_Storage.
        save_results (bool): Whether to save the results to a JSON file.
        output_dir (Optional[str]): Directory to save the results.
            If None, save to the same directory as the evaluation logs.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of summary statistics for each agent.
    """
    # Dictionary to store summary statistics for each agent
    all_summaries = {}
    
    # If no agent directories specified, process all directories in Agent_Storage
    if agent_dirs is None:
        agent_storage_dir = os.path.join(project_root, "Agent_Storage")
        # List all directories in Agent_Storage
        agent_dirs = [os.path.join(agent_storage_dir, d) for d in os.listdir(agent_storage_dir) 
                     if os.path.isdir(os.path.join(agent_storage_dir, d))]
    
    # Process each agent directory
    for agent_dir in agent_dirs:
        agent_name = os.path.basename(agent_dir)
        print(f"Processing agent: {agent_name}")
        
        # Path to evaluation_logs directory
        eval_logs_dir = os.path.join(agent_dir, "evaluation_logs")
        
        # Skip if evaluation_logs directory doesn't exist
        if not os.path.exists(eval_logs_dir):
            print(f"  No evaluation logs found for {agent_name}")
            continue
        
        # Find all JSON files in evaluation_logs directory
        json_files = glob.glob(os.path.join(eval_logs_dir, "*.json"))
        
        # Process each JSON file
        agent_summaries = {}
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            print(f"  Processing {file_name}")
            
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Skip if the file already has a 'performance' key
            if 'performance' in data:
                print(f"    File already has performance metrics")
                continue
            
            # Generate summary statistics for this log
            performance_data = generate_summary_stats(data, file_name)
            
            # Add to agent summaries
            if performance_data:
                agent_summaries[file_name] = performance_data
                
                # Write performance data back to the original JSON file
                # Create a new dictionary with performance at the top
                updated_data = {
                    "performance": performance_data
                }
                # Copy the environment data
                updated_data["environment"] = data["environment"]
                
                # Write back to the file with custom JSON formatting
                with open(json_file, 'w') as f:
                    f.write(format_json_with_compact_arrays(updated_data))
                
                print(f"    Added performance data to {file_name}")
        
        # Store summaries for this agent
        if agent_summaries:
            all_summaries[agent_name] = agent_summaries
            
            # Save results if requested
            if save_results:
                save_summary_results(agent_dir, agent_summaries, output_dir)
    
    return all_summaries


def generate_summary_stats(data: Dict[str, Any], file_name: str) -> Dict[str, Any]:
    """
    Generate summary statistics for a given evaluation log.
    
    Args:
        data (Dict[str, Any]): The evaluation log data.
        file_name (str): The name of the JSON file.
        
    Returns:
        Dict[str, Any]: Dictionary of summary statistics.
    """
    # Extract environment data
    if "environment" not in data:
        print(f"    Error: No environment data found in {file_name}")
        return None
    
    env_data = data["environment"]
    
    # Extract layout (usually a 2D grid)
    if "layout" not in env_data:
        print(f"    Error: No layout data found in {file_name}")
        return None
    
    # Create performance dictionary with a comment about the array order
    performance = {
        "__comment": "Each state maps to an array with the following values in order: " +
                    "[cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, " +
                    "risky_diagonal, target_state, action_taken]"
    }
    
    # Process each state in the environment data
    for state_key, state_data in env_data.items():
        # Skip the layout key
        if state_key == "layout":
            continue
        
        # Parse state coordinates
        try:
            x, y, orientation = map(int, state_key.split(","))
        except ValueError:
            # Skip keys that aren't in the expected format
            continue
        
        # Get relevant data for this state
        cell_type = "unknown"
        path_length = 0
        lava_steps = 0
        reaches_goal = False
        next_cell_is_lava = False
        risky_diagonal = False
        target_state = None
        action_taken = None
        
        # Check if the state has path_taken and next_step data
        if "path_taken" in state_data:
            path = state_data["path_taken"]
            path_length = len(path) - 1 if len(path) > 0 else 0
            
            # Count lava steps
            lava_steps = 0
            if path_length > 0:
                for i in range(1, len(path)):
                    step_coords = path[i].split(",")
                    step_x, step_y = int(step_coords[0]), int(step_coords[1])
                    
                    # Get the layout
                    layout = env_data["layout"]
                    
                    # Check dimensions to avoid index errors
                    if 0 <= step_y < len(layout) and 0 <= step_x < len(layout[0]):
                        # Remember that the layout is transposed (x,y in state vs y,x in layout)
                        if layout[step_x][step_y] == "lava":
                            lava_steps += 1
        
        # Get cell type
        layout = env_data["layout"]
        if 0 <= y < len(layout[0]) and 0 <= x < len(layout):
            # Remember that the layout is transposed
            cell_type = layout[x][y]
        
        # Check if the cell is a goal
        reaches_goal = (cell_type == "goal")
        
        # Extract next_step data if available
        if "next_step" in state_data:
            next_step = state_data["next_step"]
            
            # Check if next cell is lava
            next_cell_is_lava = next_step.get("type") == "lava"
            
            # Check if the move is a risky diagonal
            risky_diagonal = next_step.get("risky_diagonal", False)
            
            # Get target state
            target_state = next_step.get("target_state")
            
            # Get action taken
            action_taken = next_step.get("action")
        
        # If we couldn't extract target_state and action_taken from next_step,
        # try to extract from summary_stats
        if (target_state is None or action_taken is None) and "summary_stats" in state_data:
            # This is a fallback and may not be available
            pass
        
        # Package the metrics into an array in the specified order
        state_metrics = [
            cell_type,
            path_length,
            lava_steps,
            reaches_goal,
            next_cell_is_lava,
            risky_diagonal,
            target_state,
            action_taken
        ]
        
        # Add to performance dictionary
        performance[state_key] = state_metrics
    
    return performance


def save_summary_results(agent_dir: str, 
                       agent_summaries: Dict[str, Dict[str, Any]],
                       output_dir: Optional[str] = None) -> None:
    """
    Save summary results to a JSON file and compute agent-wide statistics.
    
    Args:
        agent_dir (str): Path to the agent directory.
        agent_summaries (Dict[str, Dict[str, Any]]): Summary statistics for this agent.
        output_dir (Optional[str]): Directory to save the results.
            If None, save to the agent's evaluation_logs directory.
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(agent_dir, "evaluation_logs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute agent-wide statistics across all environments
    agent_stats = compute_agent_stats(agent_summaries)
    
    # Combine the summaries and agent-wide stats
    combined_summary = {
        "agent_statistics": agent_stats,
        "environment_summaries": agent_summaries
    }
    
    # Output file path
    output_file = os.path.join(output_dir, "performance_summary.json")
    
    # Save to JSON file with custom JSON formatting
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(combined_summary))
    
    print(f"Saved summary results to {output_file}")
    
    # Print some key stats
    print(f"  Success rate: {agent_stats['success_rate']:.2f}%")
    print(f"  Lava avoidance rate: {agent_stats['lava_avoidance_rate']:.2f}%")
    print(f"  Average path length: {agent_stats['avg_path_length']:.2f}")
    print(f"  Environments analyzed: {agent_stats['env_count']}")

def compute_agent_stats(agent_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute agent-wide statistics based on performance across all environments.
    
    Args:
        agent_summaries (Dict[str, Dict[str, Any]]): Summary statistics for this agent.
        
    Returns:
        Dict[str, Any]: Agent-wide statistics.
    """
    # Initialize counters and accumulators
    total_states = 0
    goal_reached_count = 0
    lava_avoidance_count = 0
    lava_decisions_count = 0
    path_length_sum = 0
    risky_diagonal_count = 0
    diagonal_decisions_count = 0
    
    # Process each environment
    for env_file, perf_data in agent_summaries.items():
        # Skip any non-state entries (like __comment)
        state_keys = [k for k in perf_data.keys() if k != "__comment"]
        
        # Process each state
        for state_key in state_keys:
            state_metrics = perf_data[state_key]
            
            # Skip if metrics are missing
            if len(state_metrics) < 8:
                continue
                
            total_states += 1
            
            # Extract metrics
            cell_type = state_metrics[0]
            path_length = state_metrics[1]
            lava_steps = state_metrics[2]
            reaches_goal = state_metrics[3]
            next_cell_is_lava = state_metrics[4]
            risky_diagonal = state_metrics[5]
            target_state = state_metrics[6]
            action = state_metrics[7]
            
            # Count goal reached
            if reaches_goal:
                goal_reached_count += 1
            
            # Count lava avoidance
            if action is not None:
                if next_cell_is_lava is not None:
                    lava_decisions_count += 1
                    if not next_cell_is_lava:
                        lava_avoidance_count += 1
            
            # Count risky diagonals
            if action in [3, 4]:  # Diagonal moves (left or right)
                diagonal_decisions_count += 1
                if risky_diagonal:
                    risky_diagonal_count += 1
            
            # Sum path lengths
            if path_length is not None:
                path_length_sum += path_length
    
    # Calculate statistics
    success_rate = (goal_reached_count / total_states * 100) if total_states > 0 else 0
    lava_avoidance_rate = (lava_avoidance_count / lava_decisions_count * 100) if lava_decisions_count > 0 else 100
    safe_diagonal_rate = ((diagonal_decisions_count - risky_diagonal_count) / diagonal_decisions_count * 100) if diagonal_decisions_count > 0 else 100
    avg_path_length = path_length_sum / total_states if total_states > 0 else 0
    
    # Return the statistics
    return {
        "success_rate": success_rate,
        "lava_avoidance_rate": lava_avoidance_rate,
        "safe_diagonal_rate": safe_diagonal_rate,
        "avg_path_length": avg_path_length,
        "total_states": total_states,
        "goal_reached_count": goal_reached_count,
        "risky_diagonal_count": risky_diagonal_count,
        "diagonal_decisions_count": diagonal_decisions_count,
        "env_count": len(agent_summaries)
    }


if __name__ == "__main__":
    # Example usage
    process_evaluation_logs() 
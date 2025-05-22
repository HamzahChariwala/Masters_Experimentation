import os
import json
import numpy as np
from typing import Dict, Any, Union, List, Optional
from collections import defaultdict


def format_json_with_compact_arrays(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    Custom JSON formatter that keeps arrays on a single line,
    except for nested arrays in the environment layout which
    are kept with each row on its own line.
    
    Args:
        data (Union[Dict, List, Any]): The data to format
        indent (int): Indentation level
        
    Returns:
        str: Formatted JSON string
    """
    # Helper function to format arrays with special handling for layout and paths
    def custom_format(obj, level=0):
        if isinstance(obj, dict):
            # Format dictionary
            result = "{\n"
            items = list(obj.items())
            
            for i, (key, value) in enumerate(items):
                indent_str = " " * (level + 2)
                result += f'{indent_str}"{key}": {custom_format(value, level + 2)}'
                if i < len(items) - 1:
                    result += ","
                result += "\n"
            
            return result + (" " * level) + "}"
            
        elif isinstance(obj, list):
            # Check if this is a matrix (2D array) - layout or mask
            is_matrix = all(isinstance(x, list) for x in obj) and obj
            is_layout = False
            
            # Check if this is a layout array
            if is_matrix:
                # We need to look up in the call stack to determine context
                # This is not perfect but works for our specific use case
                import inspect
                stack = inspect.stack()
                for frame in stack:
                    if 'key' in frame.frame.f_locals:
                        key = frame.frame.f_locals['key']
                        if key in ["layout"]:
                            is_layout = True
                            break
            
            if is_matrix and is_layout:
                # Format layout with each row on its own line
                result = "[\n"
                indent_str = " " * (level + 2)
                
                for i, row in enumerate(obj):
                    # Format each row compactly
                    row_str = json.dumps(row)
                    result += indent_str + row_str
                    if i < len(obj) - 1:
                        result += ","
                    result += "\n"
                
                return result + (" " * level) + "]"
            else:
                # Format regular arrays compactly on one line
                return json.dumps(obj)
                
        else:
            # Format primitives normally
            return json.dumps(obj)
    
    # Apply our custom formatter
    return custom_format(data)


def export_agent_eval_data_to_json(
    results_dict: Dict[str, Any],
    env_tensor: np.ndarray,
    env_id: str,
    seed: int,
    agent_folder: str
) -> str:
    """
    Export agent evaluation data to JSON in the specified format.
    
    Args:
        results_dict (Dict[str, Any]): Results from evauate_agent_on_single_env
        env_tensor (np.ndarray): Environment tensor containing cell types
        env_id (str): Environment ID
        seed (int): Random seed used in the evaluation
        agent_folder (str): Path to the agent folder
        
    Returns:
        str: Path to the generated JSON file
    """
    print("\nExporting agent evaluation data to JSON...")
    
    # Ensure the evaluation_logs directory exists inside the agent folder
    evaluation_logs_dir = os.path.join(agent_folder, "evaluation_logs")
    os.makedirs(evaluation_logs_dir, exist_ok=True)
    
    # Create output filename with env_id and seed
    output_path = os.path.join(evaluation_logs_dir, f"{env_id}-{seed}.json")
    
    # Process the results to ensure tuples are converted to lists for JSON serialization
    processed_results = {}
    for state_key, state_data in results_dict.items():
        processed_state = {}
        
        # Process path_taken: convert each tuple to a list
        processed_state["path_taken"] = [list(pos) for pos in state_data["path_taken"]]
        
        # Process next_step
        processed_state["next_step"] = {
            "action": state_data["next_step"]["action"],
            "target_state": list(state_data["next_step"]["target_state"]) if state_data["next_step"]["target_state"] is not None else None,
            "type": state_data["next_step"]["type"],
            "risky_diagonal": state_data["next_step"]["risky_diagonal"]
        }
        
        # Copy summary_stats as is
        processed_state["summary_stats"] = state_data["summary_stats"]
        
        # Include model_inputs if present, converting numpy arrays to JSON serializable format
        if "model_inputs" in state_data:
            processed_state["model_inputs"] = {}
            processed_state["model_inputs"]['lava_mask'] = numpy_to_json_serializable(state_data["model_inputs"]["lava_mask"])
            processed_state["model_inputs"]['raw_input'] = numpy_to_json_serializable(state_data["model_inputs"]["raw_input"])
        
        # Add to processed results
        processed_results[state_key] = processed_state
    
    # Create the main dictionary structure
    json_data = {
        "environment": {
            "layout": env_tensor.tolist(),  # Add environment layout to the output
            "legend": {
                "wall": "Wall cell - impassable",
                "floor": "Floor cell - normal traversal",
                "lava": "Lava cell - avoided in standard path, penalized in dangerous paths",
                "goal": "Goal cell - destination"
            }
        },
        "states": processed_results
    }
    
    # Write to file using custom formatter for compact JSON
    with open(output_path, "w") as json_file:
        json_file.write(format_json_with_compact_arrays(json_data))
    
    print(f"Agent evaluation data exported to {output_path}")
    return output_path


def numpy_to_json_serializable(arr: np.ndarray) -> Union[List, Dict, Any]:
    """
    Convert a NumPy ndarray to a JSON-serializable format.
    
    Args:
        arr (np.ndarray): The NumPy array to convert
        
    Returns:
        Union[List, Dict, Any]: A JSON-serializable representation of the array
    """
    # Handle None values
    if arr is None:
        return None
    
    # Handle scalar values
    if np.isscalar(arr) or arr.ndim == 0:
        # Convert numpy types to Python native types
        if isinstance(arr, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(arr)
        elif isinstance(arr, (np.floating, np.float64, np.float32, np.float16)):
            return float(arr)
        elif isinstance(arr, (np.bool_)):
            return bool(arr)
        else:
            return arr.item()
    
    # Convert to Python list (handles both 1D and multi-dimensional arrays)
    arr_list = arr.tolist()
    
    # For masked arrays, convert mask to regular list
    if isinstance(arr, np.ma.MaskedArray):
        # Return dict with data and mask for more complex handling if needed
        return {
            "data": arr.data.tolist(),
            "mask": arr.mask.tolist() if isinstance(arr.mask, np.ndarray) else bool(arr.mask)
        }
    
    return arr_list

def add_performance_summary_to_agent_logs(logs_dir: Optional[str] = None, save_results: bool = True) -> Dict[str, Any]:
    """
    Add a performance summary to agent log files, similar to the format used in Dijkstra logs.
    This function processes each JSON file in the logs_dir, extracts performance metrics from
    the agent paths, and adds a performance key at the top of each file.
    
    Args:
        logs_dir (Optional[str]): Directory containing agent log files. If None, uses default path.
        save_results (bool): Whether to save the updated files or just return the data.
        
    Returns:
        Dict[str, Any]: Dictionary of performance summaries by file.
    """
    import glob
    
    # Dictionary to store performance summaries
    performance_summaries = {}
    
    # If no logs directory specified, use default
    if logs_dir is None:
        # Look for agent logs in a standard location
        possible_locations = [
            "Agent_Storage/*/evaluation_logs",
            "Agent_Evaluation/*/evaluation_logs"
        ]
        
        for pattern in possible_locations:
            log_dirs = glob.glob(pattern)
            if log_dirs:
                logs_dir = log_dirs[0]
                print(f"Using log directory: {logs_dir}")
                break
    
    if logs_dir is None or not os.path.exists(logs_dir):
        print("No valid logs directory found")
        return performance_summaries
    
    # Find all JSON files in the logs directory
    json_files = glob.glob(os.path.join(logs_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return performance_summaries
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Initialize data structures to store statistics
    state_stats = defaultdict(lambda: {
        "count": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    })
    
    # Track overall statistics as well
    overall_stats = {
        "total_states": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    }
    
    # Track floor-only statistics
    floor_only_stats = {
        "total_states": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    }
    
    # Process each JSON file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        print(f"Processing {file_name}")
        
        # Skip performance summary files
        if file_name.startswith("performance_summary"):
            print(f"  Skipping performance summary file")
            continue
        
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Invalid JSON in {file_name}")
            continue
        
        # Skip if performance data already exists
        if "performance" in data:
            print(f"  File already has performance metrics")
            performance_summaries[file_name] = data["performance"]
            continue
        
        # Skip if no environment data
        if "environment" not in data or "states" not in data:
            print(f"  Error: Missing required data in {file_name}")
            continue
        
        # Extract environment layout
        env_layout = data["environment"]["layout"]
        
        # Initialize performance data
        performance_data = {
            "__comment": "Each state maps to an array with the following values in order: " +
                        "[cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, " +
                        "risky_diagonal, target_state, action_taken]",
            "__mode": "agent"  # Only one mode for agent, unlike Dijkstra's 7 rule sets
        }
        
        # Process each state
        for state_key, state_data in data["states"].items():
            # Default values
            cell_type = "unknown"
            path_length = 0
            lava_steps = 0
            reaches_goal = False
            next_cell_is_lava = False
            risky_diagonal = False
            target_state = None
            action_taken = None
            
            # Parse state coordinates
            try:
                x, y, orientation = map(int, state_key.split(","))
                
                # Get cell type from environment layout
                if 0 <= x < len(env_layout[0]) and 0 <= y < len(env_layout):
                    # IMPORTANT: Fix for coordinate system mismatch
                    # Environment layout is stored as rows first, so access using [y][x]
                    cell_type = env_layout[y][x]
            except (ValueError, IndexError):
                # Skip keys that aren't in the expected format
                continue
            
            # Extract performance metrics from the state data
            
            # Try to get cell type directly from the state data first
            if "cell_type" in state_data:
                cell_type = state_data["cell_type"]
            # We should not use next_step.type as it refers to the target cell, not the current cell
            
            # Get path length and lava steps
            if "path_taken" in state_data:
                path = state_data["path_taken"]
                path_length = len(path) - 1 if len(path) > 0 else 0
                
                # Count lava steps
                if path_length > 0:
                    for step in path:
                        step_x, step_y = step[0], step[1]
                        if 0 <= step_x < len(env_layout[0]) and 0 <= step_y < len(env_layout):
                            # Correct coordinate access: env_layout[y][x]
                            if env_layout[step_y][step_x] == "lava":
                                lava_steps += 1
            
            # Check if goal was reached
            if "summary_stats" in state_data:
                # Use the summary stats if available
                if "reachable" in state_data["summary_stats"]:
                    reaches_goal = state_data["summary_stats"]["reachable"]
            
            # Also check the cell type and path for goal reaching
            # If current cell is goal, it's reached
            if cell_type == "goal":
                reaches_goal = True
            # Check if the path includes the goal
            elif "path_taken" in state_data and len(state_data["path_taken"]) > 0:
                path = state_data["path_taken"]
                
                # Check if the last step in the path is the goal
                if len(path) > 0:
                    last_step = path[-1]
                    last_x, last_y = last_step[0], last_step[1]
                    
                    # Check if the last position is the goal
                    if 0 <= last_x < len(env_layout[0]) and 0 <= last_y < len(env_layout):
                        if env_layout[last_y][last_x] == "goal":
                            reaches_goal = True
            
            # Extract next step information
            if "next_step" in state_data:
                next_step = state_data["next_step"]
                
                # Check if next cell is lava
                next_cell_is_lava = next_step.get("type") == "lava"
                
                # Check if the move is a risky diagonal
                risky_diagonal = next_step.get("risky_diagonal", False)
                
                # Get target state and action
                if next_step.get("target_state") is not None:
                    target_x, target_y, target_o = next_step["target_state"]
                    target_state = f"{target_x},{target_y},{target_o}"
                
                action_taken = next_step.get("action")
            
            # Package metrics into the format used by Dijkstra logs
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
            
            # Add to performance data
            performance_data[state_key] = state_metrics
        
        # Store the performance data
        performance_summaries[file_name] = performance_data
        
        # Save the updated file if requested
        if save_results:
            # Create a new data structure with performance first
            updated_data = {
                "performance": {
                    "agent": performance_data  # Use "agent" key to match the structure of Dijkstra logs
                }
            }
            
            # Add the rest of the data
            for key, value in data.items():
                updated_data[key] = value
            
            # Write the updated file
            with open(json_file, 'w') as f:
                f.write(format_json_with_compact_arrays(updated_data))
            
            print(f"  Added performance data to {file_name}")
    
    return performance_summaries

def create_agent_performance_summary(agent_dir: Optional[str] = None, logs_dir: Optional[str] = None, overall_only: bool = False) -> str:
    """
    Create a summary of agent performance across all environments.
    This function analyzes all JSON files in an agent's evaluation_logs directory
    and creates a summary file with statistics for each state.
    
    Args:
        agent_dir (Optional[str]): Path to agent directory. If None, tries to detect automatically.
        logs_dir (Optional[str]): Path to evaluation logs directory. If None, uses agent_dir/evaluation_logs.
        overall_only (bool): If True, only include the overall summary without per-state statistics.
        
    Returns:
        str: Path to the generated summary file
    """
    import glob
    from collections import defaultdict
    
    # If no agent directory is provided, try to detect it
    if agent_dir is None:
        # Try to infer from logs_dir if provided
        if logs_dir is not None:
            agent_dir = os.path.dirname(logs_dir)
        else:
            # Try to find agent directories
            possible_dirs = glob.glob("Agent_Storage/*") + glob.glob("../Agent_Storage/*") + glob.glob("../../Agent_Storage/*")
            if possible_dirs:
                agent_dir = possible_dirs[0]
                print(f"Using agent directory: {agent_dir}")
    
    # If no logs directory is provided, use agent_dir/evaluation_logs
    if logs_dir is None:
        if agent_dir is None:
            print("No agent directory found and no logs directory provided")
            return ""
        logs_dir = os.path.join(agent_dir, "evaluation_logs")
    
    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return ""
    
    # Find all JSON files in the logs directory
    json_files = glob.glob(os.path.join(logs_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return ""
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    # Initialize data structures to store statistics
    state_stats = defaultdict(lambda: {
        "count": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    })
    
    # Initialize a separate dictionary to store floor-only state statistics
    floor_state_stats = defaultdict(lambda: {
        "count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    })
    
    # Track overall statistics as well
    overall_stats = {
        "total_states": 0,
        "lava_cell_count": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    }
    
    # Track floor-only statistics
    floor_only_stats = {
        "total_states": 0,
        "path_length_sum": 0,
        "lava_steps_sum": 0,
        "goal_reached_count": 0,
        "next_cell_lava_count": 0,
        "risky_diagonal_count": 0
    }
    
    # Process each JSON file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        print(f"Processing {file_name}")
        
        # Skip performance summary files
        if file_name.startswith("performance_summary"):
            print(f"  Skipping performance summary file")
            continue
        
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Invalid JSON in {file_name}")
            continue
        
        # Skip if no performance data
        if "performance" not in data:
            print(f"  No performance data found, skipping")
            continue
        
        # Get performance data - it could be directly or under 'agent' key
        if "agent" in data["performance"]:
            perf_data = data["performance"]["agent"]
        else:
            perf_data = data["performance"]
        
        # Process each state in the performance data
        for state_key, metrics in perf_data.items():
            # Skip comment and mode keys
            if state_key.startswith("__"):
                continue
                
            # Extract metrics from the array
            # [cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken]
            if len(metrics) >= 6:  # Make sure we have at least the first 6 metrics
                cell_type = metrics[0]
                path_length = metrics[1]
                lava_steps = metrics[2]
                reaches_goal = metrics[3]
                next_cell_is_lava = metrics[4]
                risky_diagonal = metrics[5]
                
                # Update per-state statistics
                state_stats[state_key]["count"] += 1
                state_stats[state_key]["lava_cell_count"] += 1 if cell_type == "lava" else 0
                state_stats[state_key]["path_length_sum"] += path_length
                state_stats[state_key]["lava_steps_sum"] += lava_steps
                state_stats[state_key]["goal_reached_count"] += 1 if reaches_goal else 0
                state_stats[state_key]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                state_stats[state_key]["risky_diagonal_count"] += 1 if risky_diagonal else 0
                
                # Update overall statistics
                overall_stats["total_states"] += 1
                overall_stats["lava_cell_count"] += 1 if cell_type == "lava" else 0
                overall_stats["path_length_sum"] += path_length
                overall_stats["lava_steps_sum"] += lava_steps
                overall_stats["goal_reached_count"] += 1 if reaches_goal else 0
                overall_stats["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                overall_stats["risky_diagonal_count"] += 1 if risky_diagonal else 0
                
                # Update floor-only statistics if this is a floor tile
                if cell_type != "lava":
                    floor_only_stats["total_states"] += 1
                    floor_only_stats["path_length_sum"] += path_length
                    floor_only_stats["lava_steps_sum"] += lava_steps
                    floor_only_stats["goal_reached_count"] += 1 if reaches_goal else 0
                    floor_only_stats["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                    floor_only_stats["risky_diagonal_count"] += 1 if risky_diagonal else 0
                    
                    # Update floor-only per-state statistics
                    floor_state_stats[state_key]["count"] += 1
                    floor_state_stats[state_key]["path_length_sum"] += path_length
                    floor_state_stats[state_key]["lava_steps_sum"] += lava_steps
                    floor_state_stats[state_key]["goal_reached_count"] += 1 if reaches_goal else 0
                    floor_state_stats[state_key]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                    floor_state_stats[state_key]["risky_diagonal_count"] += 1 if risky_diagonal else 0
    
    # Create summary data
    summary_data = {}
    if not overall_only:
        for state_key, stats in state_stats.items():
            count = stats["count"]
            if count > 0:
                summary_data[state_key] = {
                    "lava_cell_proportion": stats["lava_cell_count"] / count,
                    "avg_path_length": stats["path_length_sum"] / count,
                    "avg_lava_steps": stats["lava_steps_sum"] / count,
                    "goal_reached_proportion": stats["goal_reached_count"] / count,
                    "next_cell_lava_proportion": stats["next_cell_lava_count"] / count,
                    "risky_diagonal_proportion": stats["risky_diagonal_count"] / count
                }
    
    # Create floor-only summary data
    floor_summary_data = {}
    if not overall_only:
        for state_key, stats in floor_state_stats.items():
            count = stats["count"]
            if count > 0:
                floor_summary_data[state_key] = {
                    "avg_path_length": stats["path_length_sum"] / count,
                    "avg_lava_steps": stats["lava_steps_sum"] / count,
                    "goal_reached_proportion": stats["goal_reached_count"] / count,
                    "next_cell_lava_proportion": stats["next_cell_lava_count"] / count,
                    "risky_diagonal_proportion": stats["risky_diagonal_count"] / count
                }
    
    # Calculate overall summary
    total_states = overall_stats["total_states"]
    overall_summary = {}
    if total_states > 0:
        overall_summary = {
            "total_state_instances": total_states,
            "unique_states": len(state_stats),
            "lava_cell_proportion": overall_stats["lava_cell_count"] / total_states,
            "avg_path_length": overall_stats["path_length_sum"] / total_states,
            "avg_lava_steps": overall_stats["lava_steps_sum"] / total_states,
            "goal_reached_proportion": overall_stats["goal_reached_count"] / total_states,
            "next_cell_lava_proportion": overall_stats["next_cell_lava_count"] / total_states,
            "risky_diagonal_proportion": overall_stats["risky_diagonal_count"] / total_states
        }
    
    # Calculate floor-only summary
    floor_states = floor_only_stats["total_states"]
    floor_only_summary = {}
    if floor_states > 0:
        floor_only_summary = {
            "total_state_instances": floor_states,
            "unique_states": len(floor_state_stats),
            "avg_path_length": floor_only_stats["path_length_sum"] / floor_states,
            "avg_lava_steps": floor_only_stats["lava_steps_sum"] / floor_states,
            "goal_reached_proportion": floor_only_stats["goal_reached_count"] / floor_states,
            "next_cell_lava_proportion": floor_only_stats["next_cell_lava_count"] / floor_states,
            "risky_diagonal_proportion": floor_only_stats["risky_diagonal_count"] / floor_states
        }
    
    # Create a dedicated summary directory
    base_dir = agent_dir or os.path.dirname(logs_dir)
    summary_dir = os.path.join(base_dir, "evaluation_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create the output filename based on the mode
    filename = "performance_all_states.json"
    if overall_only:
        filename = "performance_all_states_overall.json"
    
    # Create summary file
    output_file = os.path.join(summary_dir, filename)
    
    # Build the data to write
    output_data = {
        "summary_description": "This file contains summary statistics for agent performance across all environments",
        "overall_summary": overall_summary
    }
    
    # Add per-state statistics if requested
    if not overall_only:
        output_data["statistics"] = summary_data
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(output_data))
    
    print(f"Created performance summary file: {output_file}")
    
    # Create floor-only summary file
    floor_filename = "performance_no_lava.json"
    floor_output_file = os.path.join(summary_dir, floor_filename)
    
    # Build the floor-only data to write
    floor_output_data = {
        "summary_description": "This file contains summary statistics for agent performance only for floor tiles (non-lava cells)",
        "overall_summary": floor_only_summary
    }
    
    # Add per-state statistics if requested
    if not overall_only:
        floor_output_data["statistics"] = floor_summary_data
    
    # Write the floor-only file
    with open(floor_output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(floor_output_data))
    
    print(f"Created floor-only summary file: {floor_output_file}")
    
    return output_file

def compare_performance_summaries(agent_summary_file: str, dijkstra_summary_file: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Compare agent and Dijkstra performance summaries to identify discrepancies.
    
    Args:
        agent_summary_file (str): Path to agent performance summary file
        dijkstra_summary_file (str): Path to Dijkstra performance summary file
        verbose (bool): Whether to print detailed comparison information
        
    Returns:
        Dict[str, Any]: Dictionary of discrepancies by metric
    """
    try:
        # Load agent summary
        with open(agent_summary_file, 'r') as f:
            agent_data = json.load(f)
        
        # Load Dijkstra summary
        with open(dijkstra_summary_file, 'r') as f:
            dijkstra_data = json.load(f)
        
        # Extract overall summaries
        agent_overall = agent_data.get("overall_summary", {})
        
        # Get Dijkstra rulesets
        dijkstra_rulesets = dijkstra_data.get("rulesets", {})
        
        # Check if we have data to compare
        if not agent_overall:
            print(f"Error: No overall summary found in agent file {agent_summary_file}")
            return {}
        
        if not dijkstra_rulesets:
            print(f"Error: No rulesets found in Dijkstra file {dijkstra_summary_file}")
            return {}
        
        # Print header
        print("\nCOMPARING AGENT AND DIJKSTRA PERFORMANCE SUMMARIES\n")
        print(f"Agent summary: {agent_summary_file}")
        print(f"Dijkstra summary: {dijkstra_summary_file}")
        print("\nOVERALL COMPARISON:\n")
        
        # Track discrepancies
        discrepancies = {}
        
        # Compare overall metrics
        metrics = ["total_state_instances", "unique_states", "lava_cell_proportion", 
                  "avg_path_length", "avg_lava_steps", "goal_reached_proportion", 
                  "next_cell_lava_proportion", "risky_diagonal_proportion"]
        
        # For each Dijkstra ruleset
        for ruleset, ruleset_data in dijkstra_rulesets.items():
            ruleset_overall = ruleset_data.get("overall_summary", {})
            
            if not ruleset_overall:
                continue
            
            print(f"\nComparing Agent vs Dijkstra ({ruleset}):")
            
            # Compare each metric
            ruleset_discrepancies = {}
            for metric in metrics:
                agent_value = agent_overall.get(metric, "N/A")
                dijkstra_value = ruleset_overall.get(metric, "N/A")
                
                # Skip if either value is missing
                if agent_value == "N/A" or dijkstra_value == "N/A":
                    continue
                
                # Calculate difference
                if isinstance(agent_value, (int, float)) and isinstance(dijkstra_value, (int, float)):
                    diff = abs(agent_value - dijkstra_value)
                    rel_diff = diff / max(abs(agent_value), abs(dijkstra_value), 1e-10) * 100
                    
                    # Record discrepancy if difference is significant
                    if rel_diff > 1.0:  # More than 1% difference
                        ruleset_discrepancies[metric] = {
                            "agent_value": agent_value,
                            "dijkstra_value": dijkstra_value,
                            "absolute_diff": diff,
                            "relative_diff_percent": rel_diff
                        }
                        
                        # Print discrepancy
                        print(f"  - {metric}: Agent={agent_value}, Dijkstra={dijkstra_value}, Diff={diff:.4f} ({rel_diff:.2f}%)")
                    elif verbose:
                        # Print all comparisons in verbose mode
                        print(f"  - {metric}: Agent={agent_value}, Dijkstra={dijkstra_value} (similar)")
            
            # Add to overall discrepancies
            if ruleset_discrepancies:
                discrepancies[ruleset] = ruleset_discrepancies
        
        # Compare per-state metrics if available and verbose is True
        if verbose and "statistics" in agent_data and any("statistics" in rd for rs, rd in dijkstra_rulesets.items()):
            print("\nPER-STATE COMPARISON:\n")
            
            agent_stats = agent_data.get("statistics", {})
            
            # For each Dijkstra ruleset
            for ruleset, ruleset_data in dijkstra_rulesets.items():
                if "statistics" not in ruleset_data:
                    continue
                
                dijkstra_stats = ruleset_data.get("statistics", {})
                
                # Find common states
                common_states = set(agent_stats.keys()) & set(dijkstra_stats.keys())
                
                print(f"\nFound {len(common_states)} common states between Agent and Dijkstra ({ruleset})")
                
                # Track states with lava_cell_proportion discrepancies
                lava_discrepancies = []
                
                # Check each common state
                for state in common_states:
                    agent_state = agent_stats[state]
                    dijkstra_state = dijkstra_stats[state]
                    
                    # Check lava_cell_proportion specifically
                    if "lava_cell_proportion" in agent_state and "lava_cell_proportion" in dijkstra_state:
                        agent_lava = agent_state["lava_cell_proportion"]
                        dijkstra_lava = dijkstra_state["lava_cell_proportion"]
                        
                        if abs(agent_lava - dijkstra_lava) > 0.01:  # More than 1% difference
                            lava_discrepancies.append({
                                "state": state,
                                "agent_value": agent_lava,
                                "dijkstra_value": dijkstra_lava,
                                "diff": abs(agent_lava - dijkstra_lava)
                            })
                
                # Print lava proportion discrepancies
                if lava_discrepancies:
                    print(f"\nFound {len(lava_discrepancies)} states with lava_cell_proportion discrepancies:")
                    
                    # Sort by difference
                    lava_discrepancies.sort(key=lambda x: x["diff"], reverse=True)
                    
                    # Print top 10
                    for i, disc in enumerate(lava_discrepancies[:10]):
                        print(f"  {i+1}. State {disc['state']}: Agent={disc['agent_value']}, Dijkstra={disc['dijkstra_value']}, Diff={disc['diff']:.4f}")
                    
                    # Add to discrepancies
                    if "state_lava_discrepancies" not in discrepancies:
                        discrepancies["state_lava_discrepancies"] = {}
                    
                    discrepancies["state_lava_discrepancies"][ruleset] = lava_discrepancies
        
        return discrepancies
    
    except Exception as e:
        print(f"Error comparing summaries: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process agent evaluation logs and add performance summaries.")
    parser.add_argument("--logs_dir", type=str, help="Directory containing agent logs.")
    parser.add_argument("--no_save", action="store_true", help="Don't save results, just print summary.")
    parser.add_argument("--create_summary", action="store_true", help="Create a performance summary file.")
    parser.add_argument("--agent_dir", type=str, help="Agent directory for creating summary file.")
    parser.add_argument("--overall_only", action="store_true", help="Only include overall summary without per-state statistics.")
    parser.add_argument("--compare", action="store_true", help="Compare agent and Dijkstra summaries.")
    parser.add_argument("--agent_summary", type=str, help="Path to agent summary file for comparison.")
    parser.add_argument("--dijkstra_summary", type=str, help="Path to Dijkstra summary file for comparison.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed comparison information.")
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.agent_summary or not args.dijkstra_summary:
            print("Error: --agent_summary and --dijkstra_summary must be provided for comparison")
        else:
            compare_performance_summaries(args.agent_summary, args.dijkstra_summary, args.verbose)
    elif args.create_summary:
        create_agent_performance_summary(
            agent_dir=args.agent_dir, 
            logs_dir=args.logs_dir,
            overall_only=args.overall_only
        )
    else:
        add_performance_summary_to_agent_logs(
            logs_dir=args.logs_dir,
            save_results=not args.no_save
        )

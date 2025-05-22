"""
SummaryTooling for Dijkstra's algorithm evaluation logs.
Processes logs and generates performance summaries for different evaluation modes.
"""

import os
import sys
import json
import glob
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


def format_json_with_compact_arrays(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    Custom JSON formatter that keeps arrays on a single line,
    except for nested arrays in the environment layout, barrier_mask, and lava_mask
    which are kept with each row on its own line.
    
    Args:
        data (Union[Dict, List, Any]): The data to format
        indent (int): Indentation level
        
    Returns:
        str: Formatted JSON string
    """
    # Preprocess mask arrays at the object level before converting to JSON
    if isinstance(data, dict) and "environment" in data and isinstance(data["environment"], dict):
        env = data["environment"]
        
        # First clean up any empty rows in the layout if it exists
        if "layout" in env and isinstance(env["layout"], list):
            layout = env["layout"]
            clean_layout = []
            for row in layout:
                if row and isinstance(row, list):
                    clean_layout.append(row)
            env["layout"] = clean_layout
    
    # Helper function to format arrays with special handling for layout and masks
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
            is_mask_or_layout = False
            
            # Check if this is a barrier_mask, lava_mask, or layout array
            if is_matrix:
                # We need to look up in the call stack to determine context
                # This is not perfect but works for our specific use case
                import inspect
                stack = inspect.stack()
                for frame in stack:
                    if 'key' in frame.frame.f_locals:
                        key = frame.frame.f_locals['key']
                        if key in ["barrier_mask", "lava_mask", "layout"]:
                            is_mask_or_layout = True
                            break
            
            if is_matrix and is_mask_or_layout:
                # Format layout or mask with each row on its own line
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


def generate_summary_stats(data: Dict[str, Any], file_name: str, mode: str) -> Dict[str, Any]:
    """
    Generate summary statistics for a given evaluation log based on mode.
    
    Args:
        data (Dict[str, Any]): The evaluation log data.
        file_name (str): The name of the JSON file.
        mode (str): The evaluation mode (standard, conservative, or one of the dangerous modes).
        
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
                    "risky_diagonal, target_state, action_taken]",
        "__mode": mode
    }
    
    # Check if we have a "states" section or if state data is directly in environment
    if "states" in data:
        # Process states from the "states" section
        for state_key, state_data in data["states"].items():
            # Skip if this state doesn't have the requested mode
            if mode not in state_data:
                continue
                
            # Get the mode-specific data
            mode_data = state_data[mode]
            
            # Process each state
            cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken = process_state(
                state_key, mode_data, env_data["layout"]
            )
            
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
    else:
        # Process states directly from environment data
        for state_key, state_data in env_data.items():
            # Skip the layout key and legend key
            if state_key == "layout" or state_key == "legend":
                continue
            
            # Parse state coordinates
            try:
                x, y, orientation = map(int, state_key.split(","))
            except ValueError:
                # Skip keys that aren't in the expected format
                continue
            
            # Mode-specific data to use
            mode_data = None
            
            # For Dijkstra logs, check if we have the specified mode
            if mode in state_data:
                mode_data = state_data[mode]
            elif "standard" in state_data and mode == "standard":
                mode_data = state_data["standard"]
            elif "conservative" in state_data and mode == "conservative":
                mode_data = state_data["conservative"]
            else:
                # Check for dangerous_1 through dangerous_5 modes
                for i in range(1, 6):
                    danger_mode = f"dangerous_{i}"
                    if danger_mode in state_data and mode == danger_mode:
                        mode_data = state_data[danger_mode]
                        break
            
            # Skip if we couldn't find appropriate mode data
            if mode_data is None:
                continue
                
            # Process each state
            cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken = process_state(
                state_key, mode_data, env_data["layout"]
            )
            
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


def process_state(state_key: str, mode_data: Dict[str, Any], layout: List[List[str]]) -> Tuple:
    """
    Process a single state and extract metrics.
    
    Args:
        state_key (str): The state key (e.g., "1,2,3")
        mode_data (Dict[str, Any]): The mode-specific data for this state
        layout (List[List[str]]): The environment layout
        
    Returns:
        Tuple: A tuple containing (cell_type, path_length, lava_steps, reaches_goal, 
               next_cell_is_lava, risky_diagonal, target_state, action_taken)
    """
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
    except ValueError:
        # Skip keys that aren't in the expected format
        return (cell_type, path_length, lava_steps, reaches_goal, 
                next_cell_is_lava, risky_diagonal, target_state, action_taken)
    
    # Get cell type
    if 0 <= y < len(layout[0]) and 0 <= x < len(layout):
        # Remember that the layout is transposed
        cell_type = layout[x][y]
    
    # Check if the cell is a goal
    reaches_goal = (cell_type == "goal")
    
    # Check if the state has path_taken data
    if "path_taken" in mode_data:
        path = mode_data["path_taken"]
        path_length = len(path) - 1 if len(path) > 0 else 0
        
        # Count lava steps
        lava_steps = 0
        if path_length > 0:
            for i in range(1, len(path)):
                # Handle both string-based paths ("x,y,orientation") and array-based paths ([x,y,orientation])
                if isinstance(path[i], str):
                    # Legacy format: "x,y,orientation"
                    step_coords = path[i].split(",")
                    step_x, step_y = int(step_coords[0]), int(step_coords[1])
                else:
                    # New format: [x, y, orientation]
                    step_x, step_y = path[i][0], path[i][1]
                
                # Check dimensions to avoid index errors
                if 0 <= step_y < len(layout[0]) and 0 <= step_x < len(layout):
                    # Remember that the layout is transposed
                    if layout[step_x][step_y] == "lava":
                        lava_steps += 1
    
    # Extract next_step data if available
    if "next_step" in mode_data:
        next_step = mode_data["next_step"]
        
        # Check if next cell is lava
        next_cell_is_lava = next_step.get("type") == "lava"
        
        # Check if the move is a risky diagonal
        risky_diagonal = next_step.get("risky_diagonal", False)
        
        # Get target state
        target_state = next_step.get("target_state")
        
        # Convert array target_state to string if needed for backwards compatibility
        if isinstance(target_state, list):
            target_state = f"{target_state[0]},{target_state[1]},{target_state[2]}"
        
        # Get action taken
        action_taken = next_step.get("action")
    
    # If we couldn't extract target_state and action_taken from next_step,
    # try to extract from summary_stats
    if (target_state is None or action_taken is None) and "summary_stats" in mode_data:
        # This is a fallback and may not be available
        pass
    
    return (cell_type, path_length, lava_steps, reaches_goal, 
            next_cell_is_lava, risky_diagonal, target_state, action_taken)


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
        state_keys = [k for k in perf_data.keys() if not k.startswith("__")]
        
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


def save_summary_results(output_dir: str, mode: str, 
                        mode_summaries: Dict[str, Dict[str, Any]]) -> None:
    """
    Save summary results to a JSON file for a specific mode.
    
    Args:
        output_dir (str): Directory to save the results.
        mode (str): The evaluation mode.
        mode_summaries (Dict[str, Dict[str, Any]]): Summary statistics for this mode.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute mode-wide statistics across all environments
    mode_stats = compute_agent_stats(mode_summaries)
    
    # Combine the summaries and mode-wide stats
    mode_summary = {
        "mode_statistics": mode_stats,
        "environment_summaries": mode_summaries
    }
    
    # Output file path
    output_file = os.path.join(output_dir, f"performance_summary_{mode}.json")
    
    # Save to JSON file with custom JSON formatting
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(mode_summary))
    
    print(f"Saved {mode} summary results to {output_file}")
    
    # Print some key stats
    print(f"  Success rate: {mode_stats['success_rate']:.2f}%")
    print(f"  Lava avoidance rate: {mode_stats['lava_avoidance_rate']:.2f}%")
    print(f"  Average path length: {mode_stats['avg_path_length']:.2f}")
    print(f"  Environments analyzed: {mode_stats['env_count']}")

    return mode_summary


def process_dijkstra_logs(logs_dir: Optional[str] = None,
                         save_results: bool = True,
                         output_dir: Optional[str] = None,
                         generate_summary_files: bool = False) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process all Dijkstra evaluation logs in the specified directory.
    For each JSON file, generate summary statistics for each mode.
    
    Args:
        logs_dir (Optional[str]): Directory containing Dijkstra logs.
            If None, process all logs in Behaviour_Specification/Evaluations.
        save_results (bool): Whether to save the results to JSON files.
        output_dir (Optional[str]): Directory to save the results.
            If None, save to the same directory as the logs.
        generate_summary_files (bool): Whether to generate separate summary JSON files.
            If False, only adds summaries to the top of each main JSON file.
    
    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Dictionary of summary statistics for each mode and environment.
    """
    # Dictionary to store summary statistics for each mode
    all_mode_summaries = {
        "standard": {},
        "conservative": {},
        "dangerous_1": {},
        "dangerous_2": {},
        "dangerous_3": {},
        "dangerous_4": {},
        "dangerous_5": {}
    }
    
    # If no logs directory specified, use default
    if logs_dir is None:
        logs_dir = os.path.join(project_root, "Behaviour_Specification", "Evaluations")
    
    # If no output directory specified, use logs directory
    if output_dir is None:
        output_dir = logs_dir
    
    # Find all JSON files in the logs directory
    json_files = glob.glob(os.path.join(logs_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return all_mode_summaries
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        print(f"Processing {file_name}")
        
        # Skip performance summary files
        if file_name.startswith("performance_summary"):
            print(f"  Skipping performance summary file")
            continue
        
        # Skip backup or example files
        if "_before_overwriting" in file_name or "_example" in file_name or "_standard.json" in file_name or "_conservative.json" in file_name or "_dangerous_" in file_name:
            print(f"  Skipping backup/example/mode file: {file_name}")
            continue
            
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  Error: Invalid JSON in {file_name}")
            continue
            
        # Skip if no environment data
        if "environment" not in data:
            print(f"  Error: No environment data in {file_name}")
            continue
        
        # Initialize a dictionary to store performance data for all modes in this file
        all_performances = {}
        needs_reordering = False
        
        # Check if the file already has performance metrics
        if "performance" in data:
            print(f"  File already has performance metrics")
            
            # Check if performance is not the first key
            first_key = next(iter(data))
            if first_key != "performance":
                print(f"  Performance needs to be moved to the top")
                needs_reordering = True
            
            # Check if it has the proper structure with mode keys
            performance = data["performance"]
            if any(mode in performance for mode in all_mode_summaries.keys()):
                # It already has the proper structure
                all_performances = performance
            else:
                # It has the old structure with a single mode
                mode = performance.get("__mode", "standard")
                all_performances[mode] = performance
                
            # Add to mode summaries
            for mode, perf_data in all_performances.items():
                if mode.startswith("__"):
                    continue
                all_mode_summaries[mode][file_name] = perf_data
            
            # If we don't need to reorder and already have performance data, skip to next file
            if not needs_reordering:
                continue
        else:
            # No performance section yet, we'll need to detect modes and add it
            # Detect available modes in the file
            detected_modes = []
            
            # Check for "states" section
            if "states" in data:
                # Look through states to find available modes
                for state_key, state_data in data["states"].items():
                    if isinstance(state_data, dict):
                        for mode in ["standard", "conservative"] + [f"dangerous_{i}" for i in range(1, 6)]:
                            if mode in state_data:
                                if mode not in detected_modes:
                                    detected_modes.append(mode)
            else:
                # Check the environment data directly for the first state
                env_data = data["environment"]
                for state_key, state_data in env_data.items():
                    if state_key != "layout" and state_key != "legend":
                        # Check if the state has nested mode data
                        if isinstance(state_data, dict):
                            if "standard" in state_data:
                                detected_modes.append("standard")
                            if "conservative" in state_data:
                                detected_modes.append("conservative")
                            for i in range(1, 6):
                                danger_mode = f"dangerous_{i}"
                                if danger_mode in state_data:
                                    detected_modes.append(danger_mode)
                        break
            
            # If no nested modes detected, assume all data is standard mode
            if not detected_modes:
                detected_modes = ["standard"]
            
            # Process each detected mode
            for mode in detected_modes:
                print(f"  Processing mode: {mode}")
                
                # Generate summary statistics for this mode
                performance_data = generate_summary_stats(data, file_name, mode)
                
                if performance_data:
                    # Add to all performances for this file
                    all_performances[mode] = performance_data
                    
                    # Add to mode summaries
                    all_mode_summaries[mode][file_name] = performance_data
                    
            needs_reordering = True
            
        # Save combined performance data to a single file if there are any modes
        if save_results and all_performances and needs_reordering:
            # Create a new ordered dictionary with performance first
            reordered_data = {}
            
            # Add performance data first
            reordered_data["performance"] = all_performances
            
            # Add all other keys from the original data
            for key, value in data.items():
                if key != "performance":  # Skip if it's already the performance key
                    reordered_data[key] = value
            
            # Write back to the file with custom JSON formatting
            output_file = os.path.join(output_dir, file_name)
            with open(output_file, 'w') as f:
                f.write(format_json_with_compact_arrays(reordered_data))
            
            if "performance" not in data:
                print(f"  Added combined performance data for {len(all_performances)} modes to {file_name}")
            else:
                print(f"  Reordered file with performance data first")
    
    # Print detection results
    for mode in ["standard", "conservative", "dangerous_1", "dangerous_2", "dangerous_3", "dangerous_4", "dangerous_5"]:
        if mode in all_mode_summaries and all_mode_summaries[mode]:
            print(f"  Mode '{mode}': {len(all_mode_summaries[mode])} environments processed")
    
    # Only generate separate summary files if explicitly requested
    if save_results and generate_summary_files:
        # Create a combined summary of all modes
        combined_summaries = {}
        
        # Process each mode that has data
        for mode, mode_summaries in all_mode_summaries.items():
            if mode_summaries:
                mode_summary = save_summary_results(output_dir, mode, mode_summaries)
                combined_summaries[mode] = mode_summary
        
        # Save combined summary to a single file
        if combined_summaries:
            combined_file = os.path.join(output_dir, "performance_summary.json")
            with open(combined_file, 'w') as f:
                f.write(format_json_with_compact_arrays(combined_summaries))
            print(f"Saved combined performance summary to {combined_file}")
    
    return all_mode_summaries


def create_dijkstra_performance_summary(logs_dir: Optional[str] = None, overall_only: bool = False) -> str:
    """
    Create a summary of Dijkstra performance across all environments for each ruleset.
    This function analyzes all JSON files in the Evaluations directory and creates a
    summary file with statistics for each state and ruleset.
    
    Args:
        logs_dir (Optional[str]): Path to evaluation logs directory. If None, uses default path.
        overall_only (bool): If True, only include the overall summary without per-state statistics.
        
    Returns:
        str: Path to the generated summary file
    """
    import glob
    from collections import defaultdict
    
    # If no logs directory is provided, use default
    if logs_dir is None:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Evaluations")
    
    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return ""
    
    # Find all JSON files in the logs directory
    json_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return ""
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    # List of rulesets to process
    rulesets = ["standard", "conservative", "dangerous_1", "dangerous_2", "dangerous_3", "dangerous_4", "dangerous_5"]
    
    # Initialize data structures to store statistics for each ruleset
    ruleset_stats = {}
    for ruleset in rulesets:
        ruleset_stats[ruleset] = {
            "state_stats": defaultdict(lambda: {
                "count": 0,
                "lava_cell_count": 0,
                "path_length_sum": 0,
                "lava_steps_sum": 0,
                "goal_reached_count": 0,
                "next_cell_lava_count": 0,
                "risky_diagonal_count": 0
            }),
            "overall_stats": {
                "total_states": 0,
                "lava_cell_count": 0,
                "path_length_sum": 0,
                "lava_steps_sum": 0,
                "goal_reached_count": 0,
                "next_cell_lava_count": 0,
                "risky_diagonal_count": 0
            }
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
        
        # Process each ruleset
        for ruleset in rulesets:
            if ruleset not in data["performance"]:
                print(f"  No {ruleset} data found, skipping this ruleset")
                continue
            
            perf_data = data["performance"][ruleset]
            
            # Process each state in the ruleset
            for state_key, metrics in perf_data.items():
                # Skip comment and mode keys
                if state_key.startswith("__"):
                    continue
                
                # Make sure metrics is an array and has enough elements
                if not isinstance(metrics, list) or len(metrics) < 6:
                    continue
                
                # Extract metrics from the array
                # [cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken]
                cell_type = metrics[0]
                path_length = metrics[1]
                lava_steps = metrics[2]
                reaches_goal = metrics[3]
                next_cell_is_lava = metrics[4]
                risky_diagonal = metrics[5]
                
                # Update per-state statistics
                ruleset_stats[ruleset]["state_stats"][state_key]["count"] += 1
                ruleset_stats[ruleset]["state_stats"][state_key]["lava_cell_count"] += 1 if cell_type == "lava" else 0
                ruleset_stats[ruleset]["state_stats"][state_key]["path_length_sum"] += path_length
                ruleset_stats[ruleset]["state_stats"][state_key]["lava_steps_sum"] += lava_steps
                ruleset_stats[ruleset]["state_stats"][state_key]["goal_reached_count"] += 1 if reaches_goal else 0
                ruleset_stats[ruleset]["state_stats"][state_key]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                ruleset_stats[ruleset]["state_stats"][state_key]["risky_diagonal_count"] += 1 if risky_diagonal else 0
                
                # Update overall statistics
                ruleset_stats[ruleset]["overall_stats"]["total_states"] += 1
                ruleset_stats[ruleset]["overall_stats"]["lava_cell_count"] += 1 if cell_type == "lava" else 0
                ruleset_stats[ruleset]["overall_stats"]["path_length_sum"] += path_length
                ruleset_stats[ruleset]["overall_stats"]["lava_steps_sum"] += lava_steps
                ruleset_stats[ruleset]["overall_stats"]["goal_reached_count"] += 1 if reaches_goal else 0
                ruleset_stats[ruleset]["overall_stats"]["next_cell_lava_count"] += 1 if next_cell_is_lava else 0
                ruleset_stats[ruleset]["overall_stats"]["risky_diagonal_count"] += 1 if risky_diagonal else 0
    
    # Create summary data for each ruleset
    summary_data = {}
    
    for ruleset in rulesets:
        # Skip rulesets with no data
        if ruleset_stats[ruleset]["overall_stats"]["total_states"] == 0:
            continue
        
        # Calculate overall summary for this ruleset
        total_states = ruleset_stats[ruleset]["overall_stats"]["total_states"]
        overall_summary = {
            "total_state_instances": total_states,
            "unique_states": len(ruleset_stats[ruleset]["state_stats"]),
            "lava_cell_proportion": ruleset_stats[ruleset]["overall_stats"]["lava_cell_count"] / total_states,
            "avg_path_length": ruleset_stats[ruleset]["overall_stats"]["path_length_sum"] / total_states,
            "avg_lava_steps": ruleset_stats[ruleset]["overall_stats"]["lava_steps_sum"] / total_states,
            "goal_reached_proportion": ruleset_stats[ruleset]["overall_stats"]["goal_reached_count"] / total_states,
            "next_cell_lava_proportion": ruleset_stats[ruleset]["overall_stats"]["next_cell_lava_count"] / total_states,
            "risky_diagonal_proportion": ruleset_stats[ruleset]["overall_stats"]["risky_diagonal_count"] / total_states
        }
        
        ruleset_data = {"overall_summary": overall_summary}
        
        # Add per-state statistics if requested
        if not overall_only:
            state_summary = {}
            for state_key, stats in ruleset_stats[ruleset]["state_stats"].items():
                count = stats["count"]
                if count > 0:
                    state_summary[state_key] = {
                        "lava_cell_proportion": stats["lava_cell_count"] / count,
                        "avg_path_length": stats["path_length_sum"] / count,
                        "avg_lava_steps": stats["lava_steps_sum"] / count,
                        "goal_reached_proportion": stats["goal_reached_count"] / count,
                        "next_cell_lava_proportion": stats["next_cell_lava_count"] / count,
                        "risky_diagonal_proportion": stats["risky_diagonal_count"] / count
                    }
            ruleset_data["statistics"] = state_summary
            
        summary_data[ruleset] = ruleset_data
    
    # Create the output filename
    filename = "performance_summary_overall.json" if overall_only else "performance_summary.json"
    output_file = os.path.join(logs_dir, filename)
    
    # Build the data to write
    output_data = {
        "summary_description": "This file contains summary statistics for Dijkstra performance across all environments and rulesets",
        "rulesets": summary_data
    }
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_arrays(output_data))
    
    print(f"Created Dijkstra performance summary file: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Dijkstra evaluation logs and generate summaries.")
    parser.add_argument("--logs_dir", type=str, help="Directory containing Dijkstra evaluation logs.")
    parser.add_argument("--no_save", action="store_true", help="Don't save results, just print summary.")
    parser.add_argument("--create_summary", action="store_true", help="Create a performance summary file.")
    parser.add_argument("--overall_only", action="store_true", help="Only include overall summary without per-state statistics.")
    
    args = parser.parse_args()
    
    if args.create_summary:
        create_dijkstra_performance_summary(
            logs_dir=args.logs_dir,
            overall_only=args.overall_only
        )
    else:
        process_dijkstra_logs(
            logs_dir=args.logs_dir,
            save_results=not args.no_save,
            generate_summary_files=False
        ) 
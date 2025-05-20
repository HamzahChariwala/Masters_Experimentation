import os
import json
import numpy as np
from typing import Dict, Any, Union, List


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

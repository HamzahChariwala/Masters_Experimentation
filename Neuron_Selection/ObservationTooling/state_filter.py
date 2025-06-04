"""
Module for filtering and collecting state data from evaluation logs based on specific criteria.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Callable, Union
import re
import argparse
from operator import gt, lt, ge, le, eq, ne

# Available properties from state array
STATE_ARRAY_PROPERTIES = {
    'cell_type': 0,
    'path_length': 1,
    'lava_steps': 2,
    'reaches_goal': 3,
    'next_cell_is_lava': 4,
    'risky_diagonal': 5,
    'target_state': 6,
    'action_taken': 7,
    'blocked': 8,
    'chose_safety': 9,
    'chose_safety_optimally': 10,
    'into_wall': 11
}

# Default properties to extract - modify this list to change what gets extracted
DEFAULT_PROPERTIES = [
    'model_inputs.raw_input',  # Raw input array
    'risky_diagonal',         # Boolean indicating if the move is a risky diagonal
    'next_cell_is_lava',      # Boolean indicating if the next cell is lava
    'action_taken',           # Integer representing the action (0-4)
    'blocked',                # Boolean indicating if path to goal exists without stepping into lava
    'chose_safety',           # Boolean indicating if agent chose floor over optimal lava move
    'chose_safety_optimally', # Boolean indicating if the safe move was the most optimal safe move
    'into_wall'               # Boolean indicating if the action results in hitting a wall
]

def load_json_file(file_path: str) -> Dict:
    """Load and return contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Get a value from nested dictionaries using dot notation."""
    current = data
    for part in path.split('.'):
        if isinstance(current, dict):
            current = current.get(part, {})
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None
    return current

def create_comparison_function(operator: str, value: str) -> Callable[[Any], bool]:
    """Create a comparison function based on operator and value."""
    operators = {
        'eq': eq,
        'ne': ne,
        'gt': gt,
        'lt': lt,
        'gte': ge,
        'lte': le,
        'true': lambda x: bool(x) is True,
        'false': lambda x: bool(x) is False,
        'in': lambda x, val_set: x in val_set
    }
    
    if operator in ('true', 'false'):
        return operators[operator]
    
    if operator == 'in':
        # Parse comma-separated values into a set
        val_set = set()
        for val in value.split(','):
            val = val.strip()
            try:
                if '.' in val:
                    val_set.add(float(val))
                else:
                    val_set.add(int(val))
            except ValueError:
                val_set.add(val)  # Keep as string if not a number
        return lambda x: x in val_set
    
    op_func = operators.get(operator)
    if op_func is None:
        raise ValueError(f"Unknown operator: {operator}")
    
    # Try to convert value to number if possible
    try:
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass
    
    return lambda x: op_func(x, value)

def create_state_filter(criteria_list: List[str]) -> Callable[[List, Dict[str, Any]], bool]:
    """
    Create a filter function based on provided criteria strings.
    
    Args:
        criteria_list: List of criteria strings in format "property:operator:value"
                      Example: ["cell_type:eq:floor", "path_length:lt:150", "risky_diagonal:is:true"]
                      For boolean values, use "property:is:true" or "property:is:false"
    """
    criteria = {}
    for criterion in criteria_list:
        try:
            if ':is:' in criterion:
                path, _, value = criterion.partition(':is:')
                if value.lower() not in ('true', 'false'):
                    raise ValueError(f"Boolean value must be 'true' or 'false', got {value}")
                criteria[path] = lambda x, v=value.lower(): bool(x) is (v == 'true')
            else:
                path, operator, value = criterion.split(':')
                criteria[path] = create_comparison_function(operator, value)
        except ValueError as e:
            raise ValueError(f"Invalid criterion format: {criterion}. Expected format: property:operator:value or property:is:true/false") from e
    
    # Map property names to state array indices
    state_props = {
        'cell_type': 0,
        'path_length': 1,
        'lava_steps': 2,
        'reaches_goal': 3,
        'next_cell_is_lava': 4,
        'risky_diagonal': 5,
        'target_state': 6,
        'action_taken': 7,
        'blocked': 8,
        'chose_safety': 9,
        'chose_safety_optimally': 10,
        'into_wall': 11
    }
    
    def filter_func(state_data: List, raw_state_data: Dict[str, Any]) -> bool:
        """Check if a state meets all criteria."""
        for prop, check_func in criteria.items():
            if prop in state_props:
                idx = state_props[prop]
                if idx >= len(state_data):
                    return False
                if not check_func(state_data[idx]):
                    return False
            else:
                # Get value from raw state data
                value = get_nested_value(raw_state_data, prop)
                if value is None or not check_func(value):
                    return False
        return True
    
    return filter_func

def get_property_value(state_array: List, state_data: Dict[str, Any], prop: str) -> Any:
    """Get a property value from either state array or nested state data."""
    if prop == 'model_inputs.raw_input':
        # Return raw input array under 'input' key
        return {'input': state_data['model_inputs']['raw_input']}
    elif prop in STATE_ARRAY_PROPERTIES:
        return state_array[STATE_ARRAY_PROPERTIES[prop]]
    else:
        # Handle nested properties
        obj = state_data
        for key in prop.split('.'):
            obj = obj[key]
        return obj

def process_evaluation_file(
    file_path: str,
    state_filter: Callable[[List, Dict[str, Any]], bool],
    properties: List[str],
    counter: int = 1
) -> Tuple[Dict[str, Any], int]:
    """Process a single evaluation log file and return matching states."""
    data = load_json_file(file_path)
    results = {}
    
    # Get environment name from filename (keeping seed)
    env_name = os.path.basename(file_path).replace('.json', '')
    
    # Process each state
    for state_key, state_data in data['states'].items():
        # Get the state array from performance data
        state_array = data['performance']['agent'][state_key]
        
        # Check if state matches filter criteria
        if state_filter(state_array, state_data):
            # Create a result object with the input array and additional properties
            key = f"{env_name}-{state_key}-{counter:04d}"
            result = {"input": state_data['model_inputs']['raw_input']}
            
            # Add additional properties if they are in the default properties
            if 'risky_diagonal' in properties or 'all' in properties:
                result["risky_diagonal"] = bool(state_array[STATE_ARRAY_PROPERTIES['risky_diagonal']])
            
            if 'next_cell_is_lava' in properties or 'all' in properties:
                result["next_cell_is_lava"] = bool(state_array[STATE_ARRAY_PROPERTIES['next_cell_is_lava']])
            
            if 'action_taken' in properties or 'all' in properties:
                result["action_taken"] = int(state_array[STATE_ARRAY_PROPERTIES['action_taken']])
            
            # Add new behavioral metrics
            if 'blocked' in properties or 'all' in properties:
                result["blocked"] = bool(state_array[STATE_ARRAY_PROPERTIES['blocked']])
            
            if 'chose_safety' in properties or 'all' in properties:
                result["chose_safety"] = bool(state_array[STATE_ARRAY_PROPERTIES['chose_safety']])
            
            if 'chose_safety_optimally' in properties or 'all' in properties:
                result["chose_safety_optimally"] = bool(state_array[STATE_ARRAY_PROPERTIES['chose_safety_optimally']])
            
            if 'into_wall' in properties or 'all' in properties:
                result["into_wall"] = bool(state_array[STATE_ARRAY_PROPERTIES['into_wall']])
            
            results[key] = result
            counter += 1
    
    return results, counter

def collect_matching_states(
    path: str,
    state_filter: Callable[[List, Dict[str, Any]], bool],
    properties: List[str]
) -> Dict[str, Any]:
    """Collect all matching states from an agent's evaluation logs."""
    eval_logs_path = os.path.join(path, 'evaluation_logs')
    if not os.path.exists(eval_logs_path):
        raise FileNotFoundError(f"Evaluation logs directory not found at {eval_logs_path}")
    
    results = {}
    # Global counter for all states across all environments
    counter = 1
    
    for filename in os.listdir(eval_logs_path):
        if filename.endswith('.json'):
            file_path = os.path.join(eval_logs_path, filename)
            file_results, counter = process_evaluation_file(file_path, state_filter, properties, counter)
            results.update(file_results)
    
    return results

def format_json_with_compact_arrays(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """Format JSON with proper indentation but keep arrays compact."""
    if isinstance(data, (list, tuple)):
        # Keep arrays on one line
        items = [json.dumps(item) for item in data]
        return f"[{', '.join(items)}]"
    elif isinstance(data, dict):
        # Format dictionaries with each entry on its own line
        lines = []
        for key, value in data.items():
            formatted_value = format_json_with_compact_arrays(value, indent + 2)
            lines.append(f'{" " * indent}"{key}": {formatted_value}')
        return "{\n" + ",\n".join(lines) + "\n" + " " * (indent - 2) + "}"
    else:
        # Use standard JSON formatting for other types
        return json.dumps(data)

def save_results(results: Dict[str, Any], output_path: str):
    """Save the filtered results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Format JSON with proper indentation
    formatted_json = "{\n"
    entries = []
    
    for key, value in results.items():
        # Format each entry with input array on a single line
        entry = f'  "{key}": {json.dumps(value)}'
        entries.append(entry)
    
    formatted_json += ",\n".join(entries)
    formatted_json += "\n}"
    
    with open(output_path, 'w') as f:
        f.write(formatted_json)

if __name__ == "__main__":

    # ===================================================
    # CRITERIA CONFIGURATION SECTION
    # ===================================================
    # To set filtering criteria, modify this list with one or more conditions
    # Format: ["property:operator:value", ...] or ["property:is:true/false", ...]
    # - 'property' can be a state array property name or a dot-notation path to nested data
    # - 'operator' can be: eq, gt, lt, gte, lte
    # - For boolean values, use 'is:true' or 'is:false'
    # 
    # Examples:
    # CRITERIA = ["reaches_goal:is:true"]  # Only get states where the agent reaches the goal
    # CRITERIA = ["cell_type:eq:floor", "path_length:lt:150"]  # Get states on floor with short paths
    # CRITERIA = ["next_cell_is_lava:is:true", "risky_diagonal:is:true"]  # Get risky states near lava
    # CRITERIA = ["reaches_goal:is:true", "lava_steps:eq:0"]  # Get successful states with no lava steps
    # ===================================================

    # Set your criteria here
    CRITERIA = ['chose_safety_optimally:is:true', 'action_taken:in:2,3,4', 'blocked:is:false']

    parser = argparse.ArgumentParser(description='Filter states from agent evaluation logs.')
    parser.add_argument('--path', help='Path to the agent directory')
    parser.add_argument('--output', help='Custom output file path (default: {path}/filtered_states.json)')
    
    args = parser.parse_args()
    
    # Create a filter function from the hardcoded criteria
    state_filter = create_state_filter(CRITERIA)
    
    # Process all evaluation logs in the agent directory
    try:
        results = collect_matching_states(args.path, state_filter, DEFAULT_PROPERTIES)
        print(f"Found {len(results)} matching states")
        
        # Save results to the specified output path or the default location
        output_path = args.output or os.path.join(args.path, 'filtered_states.json')
        save_results(results, output_path)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}") 
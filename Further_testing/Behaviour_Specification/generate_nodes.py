from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from Behaviour_Specification.state_class import State

def generate_state_nodes(
    env_tensor: np.ndarray, 
    max_size: Tuple[int, int],
    orientations: List[int] = [0, 1, 2, 3],
    valid_types: Optional[Dict[str, List[str]]] = None,
    debug: bool = False
) -> Dict[Tuple[int, int, int], State]:
    """
    Generate State objects for all combinations of x, y, and orientations.
    
    Args:
        env_tensor (np.ndarray): Environment tensor containing cell types (floor, wall, lava, etc.).
                                 The tensor is indexed as env_tensor[y, x] (row, col).
        max_size (Tuple[int, int]): Maximum dimensions of the environment (width, height).
        orientations (List[int]): List of orientations to generate states for.
                                 Default is [0, 1, 2, 3] (up, right, down, left).
        valid_types (Dict[str, List[str]], optional): Dictionary of valid types for different
                                                      neighbor categories.
        debug (bool): Whether to print debug information during generation (default: False)
    
    Returns:
        Dict[Tuple[int, int, int], State]: Dictionary mapping (x, y, orientation) tuples to State objects.
    """
    width, height = max_size
    states = {}
    
    # Generate states for all combinations of x, y, and orientation
    for x in range(width):
        for y in range(height):
            for theta in orientations:
                # Create state tuple (x, y, orientation)
                state_tuple = (x, y, theta)
                
                # Create State object
                state_obj = State(state=state_tuple)
                
                # Populate all state variables (type, neighbors, etc.)
                state_obj.populate_object(env_tensor, valid_types, debug)
                
                # Add to dictionary with state tuple as key
                states[state_tuple] = state_obj
    
    return states

def filter_state_nodes(
    state_nodes: Dict[Tuple[int, int, int], State], 
    filter_condition: callable
) -> Dict[Tuple[int, int, int], State]:
    """
    Filter state nodes based on a condition function.
    
    Args:
        state_nodes (Dict[Tuple[int, int, int], State]): Dictionary of state nodes.
        filter_condition (callable): Function that takes a State object and returns 
                                     True if it should be included, False otherwise.
    
    Returns:
        Dict[Tuple[int, int, int], State]: Filtered dictionary of state nodes.
    """
    filtered_nodes = {}
    
    for state_tuple, state_obj in state_nodes.items():
        if filter_condition(state_obj):
            filtered_nodes[state_tuple] = state_obj
    
    return filtered_nodes

def analyze_state_nodes(
    env_tensor: np.ndarray, 
    print_output: bool = True,
    debug: bool = False
) -> Dict[Tuple[int, int, int], State]:
    """
    Generate and analyze state nodes from an environment tensor.
    This function creates state nodes for all positions and orientations,
    counts the distribution of node types, and filters out invalid nodes.
    
    Args:
        env_tensor (np.ndarray): Environment tensor containing cell types.
        print_output (bool): Whether to print analysis information.
        debug (bool): Whether to print debug information during state generation (default: False)
        
    Returns:
        Dict[Tuple[int, int, int], State]: Dictionary of generated state nodes.
    """
    # Get environment dimensions
    height, width = env_tensor.shape
    orientations = [0, 1, 2, 3]  # All possible orientations
    
    if print_output:
        print(f"Grid dimensions: {width}x{height}")
        print(f"Total number of possible states: {width * height * len(orientations)}")
    
    # Generate the nodes
    nodes = generate_state_nodes(env_tensor, (width, height), orientations, debug=debug)
    
    # Count nodes by type
    type_counts = {}
    for state_tuple, state_obj in nodes.items():
        state_type = state_obj.type
        type_counts[state_type] = type_counts.get(state_type, 0) + 1
    
    if print_output:
        print(f"Generated {len(nodes)} state nodes")
        print("State types distribution:")
        for state_type, count in type_counts.items():
            print(f"  - {state_type}: {count} states")
    
    # Filter to include only nodes with at least one standard neighbor
    valid_nodes = filter_state_nodes(nodes, lambda s: len(s.valid_standard) > 0)
    
    if print_output:
        print(f"Nodes with valid standard neighbors: {len(valid_nodes)} states")
        print("===== Node Generation Complete =====\n")
    
    return nodes
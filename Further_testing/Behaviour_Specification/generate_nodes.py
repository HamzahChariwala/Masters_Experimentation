from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from Behaviour_Specification.state_class import State

def generate_state_nodes(
    env_tensor: np.ndarray, 
    max_size: Tuple[int, int],
    orientations: List[int] = [0, 1, 2, 3],
    valid_types: Optional[Dict[str, List[str]]] = None
) -> Dict[Tuple[int, int, int], State]:
    """
    Generate State objects for all combinations of x, y, and orientations.
    
    Args:
        env_tensor (np.ndarray): Environment tensor containing cell types (floor, wall, lava, etc.).
                                 The tensor should be indexed as env_tensor[x, y].
        max_size (Tuple[int, int]): Maximum dimensions of the environment (width, height).
        orientations (List[int]): List of orientations to generate states for.
                                 Default is [0, 1, 2, 3] (up, right, down, left).
        valid_types (Dict[str, List[str]], optional): Dictionary of valid types for different
                                                      neighbor categories.
    
    Returns:
        Dict[Tuple[int, int, int], State]: Dictionary mapping (x, y, orientation) tuples to State objects.
    """
    max_x, max_y = max_size
    states = {}
    
    # Generate states for all combinations of x, y, and orientation
    for x in range(max_x):
        for y in range(max_y):
            for theta in orientations:
                # Create state tuple (x, y, orientation)
                state_tuple = (x, y, theta)
                
                # Create State object
                state_obj = State(state=state_tuple)
                
                # Populate all state variables (type, neighbors, etc.)
                state_obj.populate_object(env_tensor, valid_types)
                
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
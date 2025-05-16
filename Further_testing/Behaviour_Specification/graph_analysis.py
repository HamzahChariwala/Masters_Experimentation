import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from Behaviour_Specification.generate_nodes import filter_state_nodes, analyze_state_nodes
from Behaviour_Specification.dijkstras_algorithm import create_graphs_from_nodes, compute_shortest_paths

def analyze_navigation_graphs(
    env_tensor: np.ndarray,
    lava_penalty_multiplier: int = 10,
    print_output: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate state nodes, create navigation graphs, and analyze paths.
    
    Args:
        env_tensor (np.ndarray): Environment tensor containing cell types
        lava_penalty_multiplier (int): Penalty multiplier for lava cells
        print_output (bool): Whether to print analysis information
        debug (bool): Whether to print detailed diagnostic information (default: False)
        
    Returns:
        Dict[str, Any]: Dictionary containing nodes, graphs, and path analysis results
    """
    # Generate and analyze state nodes
    if print_output:
        print("\n===== Generating State Nodes =====")
    nodes = analyze_state_nodes(env_tensor, print_output=print_output, debug=debug)
    
    # Create graphs from nodes
    if print_output:
        print("\n===== Creating Navigation Graphs =====")
    graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier)
    
    # Find the goal state for path testing
    goal_states = filter_state_nodes(nodes, lambda s: s.type == "goal")
    start_states = filter_state_nodes(nodes, lambda s: s.type == "floor" and s.state[0] == 1 and s.state[1] == 1)
    
    # Results to return
    results = {
        "nodes": nodes,
        "graphs": graphs,
        "goal_states": goal_states,
        "start_states": start_states,
        "paths": {}
    }
    
    # Analyze paths if start and goal states exist
    if goal_states and start_states:
        start_state = next(iter(start_states.keys()))
        goal_state = next(iter(goal_states.keys()))
        
        # Extract just the position (x, y) from the goal state
        goal_x, goal_y, _ = goal_state
        goal_position = (goal_x, goal_y)
        
        if print_output:
            print(f"\nTesting path from {start_state} to goal position {goal_position}:")
        
        # Create node index mappings
        node_indices = {}
        for graph_name, graph in graphs.items():
            node_indices[graph_name] = {}
            for idx, node_data in enumerate(graph.nodes()):
                node_indices[graph_name][node_data.state] = idx
        
        # Test paths in each graph
        for graph_name, graph in graphs.items():
            if print_output:
                print(f"\nGraph: {graph_name}")
            
            try:
                # Use the goal position instead of specific goal state
                paths = compute_shortest_paths(
                    graph, start_state, goal_position, node_indices[graph_name]
                )
                
                path_result = {
                    "success": False,
                    "path_states": [],
                    "path_indices": [],
                    "path_cost": 0
                }
                
                if not paths:
                    if print_output:
                        print(f"No path found from {start_state} to goal position {goal_position}")
                    results["paths"][graph_name] = path_result
                    continue
                
                # Get the target index and path
                target_idx = next(iter(paths.keys()))
                path_indices = paths[target_idx]
                
                if not path_indices:
                    if print_output:
                        print("Path is empty")
                    results["paths"][graph_name] = path_result
                    continue
                    
                # Convert to states
                path_states = [graph.nodes()[idx].state for idx in path_indices]
                
                # Calculate path cost (accounting for lava penalty in dangerous graph)
                path_cost = 0
                for i in range(len(path_indices) - 1):
                    src_idx = path_indices[i]
                    dst_idx = path_indices[i + 1]
                    edge_weight = graph.get_edge_data(src_idx, dst_idx)
                    path_cost += edge_weight if edge_weight is not None else 1
                
                # Update result for this graph
                path_result = {
                    "success": True,
                    "path_states": path_states,
                    "path_indices": path_indices,
                    "path_cost": path_cost
                }
                results["paths"][graph_name] = path_result
                
                if print_output:
                    print(f"Path length: {len(path_states)}")
                    print(f"Start: {path_states[0]}")
                    print(f"End: {path_states[-1]}")
                    print(f"Path cost: {path_cost}")
                    
            except Exception as e:
                if print_output:
                    print(f"Error computing path: {e}")
                results["paths"][graph_name] = {
                    "success": False,
                    "error": str(e)
                }
    else:
        if print_output:
            print("Could not find suitable start and goal states for path testing")
    
    if print_output:
        print("===== Navigation Graph Analysis Complete =====\n")
        
    return results 
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union

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
        Dict[str, Any]: Dictionary containing:
            - nodes: All state nodes
            - graphs: All navigation graphs
            - goal_states: Dictionary of goal state nodes
            - paths_from_position: Dictionary mapping each valid start position (x, y) to a dict 
              containing path results for each graph type (dangerous, standard, conservative)
              with format: {(x,y): {"dangerous": (success, cost), "standard": (success, cost), ...}}
    """
    # Generate and analyze state nodes
    if print_output:
        print("\n===== Generating State Nodes =====")
    nodes = analyze_state_nodes(env_tensor, print_output=print_output, debug=debug)
    
    # Create graphs from nodes
    if print_output:
        print("\n===== Creating Navigation Graphs =====")
    graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier)
    
    # Find the goal states and all potential starting states
    goal_states = filter_state_nodes(nodes, lambda s: s.type == "goal")
    floor_states = filter_state_nodes(nodes, lambda s: s.type == "floor")
    lava_states = filter_state_nodes(nodes, lambda s: s.type == "lava")
    
    # Get environment dimensions
    height, width = env_tensor.shape
    
    # Results to return
    results = {
        "nodes": nodes,
        "graphs": graphs,
        "goal_states": goal_states,
        "paths_from_position": {}
    }
    
    # Check if we have goal states
    if not goal_states:
        if print_output:
            print("No goal states found in the environment")
        return results
    
    # Extract the goal position
    goal_state = next(iter(goal_states.keys()))
    goal_x, goal_y, _ = goal_state
    goal_position = (goal_x, goal_y)
    
    if print_output:
        print(f"\n===== Computing Paths to Goal at {goal_position} =====")
    
    # Create node index mappings for all graphs
    node_indices = {}
    for graph_name, graph in graphs.items():
        node_indices[graph_name] = {}
        for idx, node_data in enumerate(graph.nodes()):
            node_indices[graph_name][node_data.state] = idx
    
    # Initialize paths_from_position dictionary
    paths_from_position = {}
    
    # All possible orientations
    orientations = [0, 1, 2, 3]  # Up, right, down, left
    
    # Analyze paths from every valid start position 
    # Skip the outer border (rows/cols 0 and height-1/width-1)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            # Skip if this position is a wall or goal
            if env_tensor[y, x] == "wall" or env_tensor[y, x] == "goal":
                continue
            
            # Store the cell type for reference
            cell_type = env_tensor[y, x]
            
            # Initialize best path for each graph type
            best_paths = {
                "dangerous": (False, float('inf')),
                "standard": (False, float('inf')),
                "conservative": (False, float('inf'))
            }
            
            # For standard and conservative graphs, skip lava cells as starting positions
            if cell_type == "lava":
                if debug and print_output:
                    print(f"Skipping lava cell at ({x}, {y}) for standard/conservative path analysis")
                # For dangerous graph, we'll still test from lava cells
                test_graph_types = ["dangerous"]
            else:
                # For floor cells, test all graph types
                test_graph_types = ["dangerous", "standard", "conservative"]
            
            # Test all orientations at this position
            for orientation in orientations:
                start_state = (x, y, orientation)
                
                # Skip if this state doesn't exist in our nodes
                if start_state not in nodes:
                    continue
                    
                if print_output and debug and (x == 1 and y == 1):  # Only print for a sample position
                    print(f"\nTesting paths from position ({x}, {y}, {orientation}) to goal at {goal_position}")
                
                # Test paths in each applicable graph type
                for graph_name in test_graph_types:
                    graph = graphs[graph_name]
                    try:
                        # Use goal position rather than specific goal state to find shortest path to any orientation
                        paths = compute_shortest_paths(
                            graph, start_state, goal_position, node_indices[graph_name]
                        )
                        
                        # If no path was found
                        if not paths:
                            continue
                        
                        # Get the target index and path
                        target_idx = next(iter(paths.keys()))
                        path_indices = paths[target_idx]
                        
                        # If path is empty
                        if not path_indices:
                            continue
                        
                        # Calculate path cost (accounting for lava penalty in dangerous graph)
                        path_cost = 0
                        for i in range(len(path_indices) - 1):
                            src_idx = path_indices[i]
                            dst_idx = path_indices[i + 1]
                            edge_weight = graph.get_edge_data(src_idx, dst_idx)
                            path_cost += edge_weight if edge_weight is not None else 1
                        
                        # Update if this is a better path
                        if path_cost < best_paths[graph_name][1]:
                            best_paths[graph_name] = (True, path_cost)
                            
                        if print_output and debug and (x == 1 and y == 1):  # Only print for a sample position
                            print(f"  {graph_name.capitalize()} graph (orientation {orientation}): " 
                                  f"Path found, cost: {path_cost}")
                        
                    except Exception as e:
                        if print_output and debug and (x == 1 and y == 1):  # Only print for a sample position
                            print(f"  Error computing {graph_name} path from ({x}, {y}, {orientation}): {e}")
            
            # Only include this position in the results if at least one path was found
            any_path_found = any(success for success, _ in best_paths.values())
            if any_path_found:
                # Store best paths for this position
                paths_from_position[(x, y)] = best_paths
    
    # Store all path results
    results["paths_from_position"] = paths_from_position
    
    # Summarize the findings
    if print_output:
        # Count reachable positions by type
        reachable_positions = {
            "dangerous": {
                "floor": 0,
                "lava": 0,
                "total": 0
            },
            "standard": {
                "floor": 0,
                "lava": 0,
                "total": 0
            },
            "conservative": {
                "floor": 0,
                "lava": 0,
                "total": 0
            }
        }
        
        # Count reachable positions by graph type and cell type
        for pos, path_results in paths_from_position.items():
            x, y = pos
            pos_type = env_tensor[y, x]
            
            for graph_name, (success, _) in path_results.items():
                if success:
                    reachable_positions[graph_name][pos_type if pos_type in ["floor", "lava"] else "floor"] += 1
                    reachable_positions[graph_name]["total"] += 1
        
        total_floor_positions = sum(1 for y in range(1, height - 1) for x in range(1, width - 1) 
                                  if env_tensor[y, x] == "floor")
        total_lava_positions = sum(1 for y in range(1, height - 1) for x in range(1, width - 1) 
                                 if env_tensor[y, x] == "lava")
        total_positions = total_floor_positions + total_lava_positions
        
        print(f"\n===== Path Analysis Summary =====")
        print(f"Total floor positions: {total_floor_positions}")
        print(f"Total lava positions: {total_lava_positions}")
        print(f"Total positions analyzed: {total_positions}")
        
        for graph_name, counts in reachable_positions.items():
            print(f"  {graph_name.capitalize()} graph: {counts['total']}/{total_positions} positions can reach the goal " 
                  f"({counts['total']/total_positions*100:.1f}%)")
            print(f"    - Floor: {counts['floor']}/{total_floor_positions} " 
                  f"({counts['floor']/total_floor_positions*100:.1f}%)")
            if graph_name == "dangerous":
                print(f"    - Lava: {counts['lava']}/{total_lava_positions} " 
                      f"({counts['lava']/total_lava_positions*100:.1f}%)")
        
        print("===== Navigation Graph Analysis Complete =====\n")
    
    # Make absolutely sure we return the full results dictionary with all necessary data
    # Fix the issue where sometimes only the graphs are returned
    if not isinstance(results, dict) or "nodes" not in results or "graphs" not in results:
        # Create a properly structured results dictionary
        fixed_results = {
            "nodes": nodes,
            "graphs": graphs,
            "goal_states": goal_states,
            "paths_from_position": paths_from_position
        }
        return fixed_results
        
    return results 
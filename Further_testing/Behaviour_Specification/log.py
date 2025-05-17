import os
import json
import numpy as np
import argparse
from typing import Dict, Tuple, List, Optional, Any, Union

# Import from reorganized modules
from Behaviour_Specification.StateGeneration.generate_nodes import filter_state_nodes, analyze_state_nodes
from Behaviour_Specification.DijkstrasAlgorithm.dijkstras_algorithm import create_graphs_from_nodes, compute_shortest_paths


def analyze_navigation_graphs(
    env_tensor: np.ndarray,
    print_output: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate state nodes, create navigation graphs, and analyze paths.
    
    Args:
        env_tensor (np.ndarray): Environment tensor containing cell types
        print_output (bool): Whether to print analysis information
        debug (bool): Whether to print detailed diagnostic information (default: False)
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - nodes: All state nodes
            - graphs: Dictionary mapping graph types to graphs
              * standard: Standard graph avoiding lava
              * conservative: Conservative graph avoiding lava and risky diagonals
              * dangerous_1: Dangerous graph with lava penalty of 1
              * dangerous_2: Dangerous graph with lava penalty of 2
              * dangerous_3: Dangerous graph with lava penalty of 3
              * dangerous_4: Dangerous graph with lava penalty of 4
              * dangerous_5: Dangerous graph with lava penalty of 5
            - goal_states: Dictionary of goal state nodes
            - paths_from_position: Dictionary mapping each valid start position (x, y) to a dict 
              containing path results for each graph type
    """
    # Generate and analyze state nodes
    if print_output:
        print("\n===== Generating State Nodes =====")
    nodes = analyze_state_nodes(env_tensor, print_output=print_output, debug=debug)
    
    # Create standard and conservative graphs
    if print_output:
        print("\n===== Creating Navigation Graphs =====")
    
    # Store all graphs in this dictionary
    all_graphs = {}
    
    # Create standard and conservative graphs (no lava penalty needed)
    base_graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier=1)
    all_graphs['standard'] = base_graphs['standard']
    all_graphs['conservative'] = base_graphs['conservative']
    
    # Create dangerous graphs with different lava penalties
    for multiplier in range(1, 6):
        if print_output:
            print(f"Creating dangerous graph with lava penalty {multiplier}x...")
        dangerous_graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier=multiplier)
        all_graphs[f'dangerous_{multiplier}'] = dangerous_graphs['dangerous']
    
    # Find the goal states and all potential starting states
    goal_states = filter_state_nodes(nodes, lambda s: s.type == "goal")
    floor_states = filter_state_nodes(nodes, lambda s: s.type == "floor")
    lava_states = filter_state_nodes(nodes, lambda s: s.type == "lava")
    
    # Get environment dimensions
    height, width = env_tensor.shape
    
    # Results to return
    results = {
        "nodes": nodes,
        "graphs": all_graphs,
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
    for graph_name, graph in all_graphs.items():
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
                "standard": (False, float('inf')),
                "conservative": (False, float('inf'))
            }
            # Add entries for each dangerous graph
            for multiplier in range(1, 6):
                best_paths[f"dangerous_{multiplier}"] = (False, float('inf'))
            
            # For standard and conservative graphs, skip lava cells as starting positions
            if cell_type == "lava":
                if debug and print_output:
                    print(f"Skipping lava cell at ({x}, {y}) for standard/conservative path analysis")
                # For dangerous graph, we'll still test from lava cells
                test_graph_types = [f"dangerous_{i}" for i in range(1, 6)]
            else:
                # For floor cells, test all graph types
                test_graph_types = ["standard", "conservative"] + [f"dangerous_{i}" for i in range(1, 6)]
            
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
                    graph = all_graphs[graph_name]
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
        # Add entries for each dangerous graph
        for multiplier in range(1, 6):
            reachable_positions[f"dangerous_{multiplier}"] = {
                "floor": 0,
                "lava": 0,
                "total": 0
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
            if "dangerous" in graph_name:
                print(f"    - Lava: {counts['lava']}/{total_lava_positions} " 
                      f"({counts['lava']/total_lava_positions*100:.1f}%)")
        
        print("===== Navigation Graph Analysis Complete =====\n")
    
    # Make absolutely sure we return the full results dictionary with all necessary data
    # Fix the issue where sometimes only the graphs are returned
    if not isinstance(results, dict) or "nodes" not in results or "graphs" not in results:
        # Create a properly structured results dictionary
        fixed_results = {
            "nodes": nodes,
            "graphs": all_graphs,
            "goal_states": goal_states,
            "paths_from_position": paths_from_position
        }
        return fixed_results
        
    return results


def export_path_data_to_json(
    analysis_results: Dict[str, Any], 
    env_tensor: np.ndarray, 
    env_id: str,
    seed: int = 0
) -> str:
    """
    Export Dijkstra's path data to JSON in the specified format.
    
    Args:
        analysis_results (Dict[str, Any]): Results from analyze_navigation_graphs
        env_tensor (np.ndarray): Environment tensor containing cell types
        env_id (str): Environment ID
        seed (int): Random seed used to generate the environment
        
    Returns:
        str: Path to the generated JSON file
    """
    print("\nExporting Dijkstra's path data to JSON...")
    
    # Ensure the Evaluations directory exists inside Behaviour_Specification
    evaluations_dir = os.path.join(os.path.dirname(__file__), "Evaluations")
    os.makedirs(evaluations_dir, exist_ok=True)
    
    # Create output filename with env_id and seed
    output_path = os.path.join(evaluations_dir, f"{env_id}-{seed}.json")
    
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
        "states": {}
    }
    
    # Get data from analysis results
    nodes = analysis_results.get("nodes", {})
    graphs = analysis_results.get("graphs", {})
    goal_states = analysis_results.get("goal_states", {})
    
    # Get goal state 
    goal_state = next(iter(goal_states.keys()), None)
    if not goal_state:
        print("  Warning: No goal state found for path calculations")
        goal_x, goal_y = None, None
    else:
        goal_x, goal_y, _ = goal_state
    
    goal_position = (goal_x, goal_y) if goal_x is not None else None
    
    # Create node index mappings for all graphs
    node_indices = {}
    for graph_name, graph in graphs.items():
        node_indices[graph_name] = {}
        for idx, node_data in enumerate(graph.nodes()):
            node_indices[graph_name][node_data.state] = idx
    
    # Get environment dimensions
    height, width = env_tensor.shape
    
    # All possible orientations
    orientations = [0, 1, 2, 3]  # Up, right, down, left
    
    # Define action mapping for next steps
    action_mapping = {
        "rotate_left": 0,   # Rotation anti-clockwise
        "rotate_right": 1,  # Rotation clockwise
        "forward": 2,       # Move forward
        "diagonal_left": 3, # Move diagonally left
        "diagonal_right": 4 # Move diagonally right
    }
    
    # Process each valid state
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            # Skip walls and goal positions
            if env_tensor[y, x] == "wall" or env_tensor[y, x] == "goal":
                continue
                
            cell_type = env_tensor[y, x]
            
            for orientation in orientations:
                state_key = f"{x},{y},{orientation}"
                start_state = (x, y, orientation)
                
                # Skip if state doesn't exist in nodes
                if start_state not in nodes:
                    continue
                
                # Initialize state data in the json structure
                json_data["states"][state_key] = {}
                
                # Determine which graph types to test based on cell type
                if cell_type == "lava":
                    test_graph_types = [f"dangerous_{i}" for i in range(1, 6)]
                else:
                    test_graph_types = ["standard", "conservative"] + [f"dangerous_{i}" for i in range(1, 6)]
                
                # Get path information for each graph type
                for graph_name in test_graph_types:
                    graph = graphs[graph_name]
                    
                    # Initialize with default values (not reachable)
                    path_data = {
                        "path_taken": [],        # Renamed from "path" to "path_taken"
                        "next_step": {           # New field for next step information
                            "action": None,      # Action type (0-4)
                            "target_state": None, # Next state
                            "type": None,        # Type of next state cell
                            "risky_diagonal": False # Whether diagonal move is risky (near lava)
                        },
                        "summary_stats": {       # New field for summary statistics
                            "path_cost": 0,      # Cost of optimal path
                            "path_length": 0,    # Number of steps in path
                            "lava_steps": 0,     # Steps that traverse lava cells
                            "reachable": False   # Whether goal is reachable
                        }
                    }
                    
                    try:
                        # Compute the shortest path
                        paths = compute_shortest_paths(
                            graph, start_state, goal_position, node_indices[graph_name]
                        )
                        
                        # If a path was found, get the details
                        if paths:
                            target_idx = next(iter(paths.keys()))
                            path_indices = paths[target_idx]
                            
                            if path_indices:  # If path is not empty
                                # Calculate the path cost and populate path states
                                path_cost = 0
                                visited_states = []
                                lava_steps = 0
                                
                                # First state (start)
                                node_data = graph.nodes()[path_indices[0]]
                                first_state = f"{node_data.state[0]},{node_data.state[1]},{node_data.state[2]}"
                                visited_states.append(first_state)
                                
                                # Calculate cost, extract states, and count lava steps
                                for i in range(len(path_indices) - 1):
                                    src_idx = path_indices[i]
                                    dst_idx = path_indices[i + 1]
                                    
                                    # Get edge weight and add to cost
                                    edge_weight = graph.get_edge_data(src_idx, dst_idx)
                                    step_cost = edge_weight if edge_weight is not None else 1
                                    path_cost += step_cost
                                    
                                    # Get destination node data
                                    dst_node_data = graph.nodes()[dst_idx]
                                    dst_state = dst_node_data.state
                                    dst_x, dst_y, dst_orientation = dst_state
                                    
                                    # Check if this step is on lava
                                    if env_tensor[dst_y, dst_x] == "lava":
                                        lava_steps += 1
                                    
                                    # Add to visited states
                                    visited_states.append(f"{dst_x},{dst_y},{dst_orientation}")
                                
                                # Update path data
                                path_data["path_taken"] = visited_states
                                path_data["summary_stats"]["path_cost"] = path_cost
                                path_data["summary_stats"]["path_length"] = len(visited_states) - 1  # Subtract 1 since first state isn't a step
                                path_data["summary_stats"]["lava_steps"] = lava_steps
                                path_data["summary_stats"]["reachable"] = True
                                
                                # Get information for next_step if path has at least one step
                                if len(path_indices) > 1:
                                    # Source and destination for first step
                                    src_idx = path_indices[0]
                                    dst_idx = path_indices[1]
                                    
                                    src_node_data = graph.nodes()[src_idx]
                                    dst_node_data = graph.nodes()[dst_idx]
                                    
                                    src_state = src_node_data.state
                                    dst_state = dst_node_data.state
                                    
                                    # Format the target state as a string
                                    dst_x, dst_y, dst_orientation = dst_state
                                    target_state_str = f"{dst_x},{dst_y},{dst_orientation}"
                                    
                                    # Determine action type by comparing source and destination
                                    sx, sy, sori = src_state
                                    dx, dy, dori = dst_state
                                    
                                    # Check for rotation (orientation change without position change)
                                    if (sx, sy) == (dx, dy) and sori != dori:
                                        # Determine rotation direction
                                        if (sori + 1) % 4 == dori:
                                            action = action_mapping["rotate_right"]
                                        else:
                                            action = action_mapping["rotate_left"]
                                    else:
                                        # Position change - determine move type
                                        node = nodes[src_state]
                                        move_type = node.identify_move_type(src_state, dst_state, sori)
                                        
                                        if move_type == "forward":
                                            action = action_mapping["forward"]
                                        elif move_type == "diagonal-left":
                                            action = action_mapping["diagonal_left"]
                                        elif move_type == "diagonal-right":
                                            action = action_mapping["diagonal_right"]
                                        else:
                                            # Default to forward for any other move types
                                            action = action_mapping["forward"]
                                    
                                    # Check if diagonal move is risky (near lava)
                                    risky_diagonal = False
                                    if action in [action_mapping["diagonal_left"], action_mapping["diagonal_right"]]:
                                        # For diagonals, check if any adjacent cells are lava
                                        move_type = "diagonal-left" if action == action_mapping["diagonal_left"] else "diagonal-right"
                                        adjacent_cells = node.get_adjacent_cells_for_diagonal(src_state, move_type)
                                        
                                        # Check each adjacent cell for lava
                                        for adj_x, adj_y in adjacent_cells:
                                            if (0 <= adj_y < height and 0 <= adj_x < width and 
                                                env_tensor[adj_y, adj_x] == "lava"):
                                                risky_diagonal = True
                                                break
                                    
                                    # Update next_step information
                                    path_data["next_step"]["action"] = action
                                    path_data["next_step"]["target_state"] = target_state_str
                                    path_data["next_step"]["type"] = env_tensor[dst_y, dst_x]
                                    path_data["next_step"]["risky_diagonal"] = risky_diagonal
                    except Exception as e:
                        # Just leave as default values if there was an error
                        print(f"  Error computing path for {state_key}, {graph_name}: {e}")
                    
                    # Store path data for this graph type
                    json_data["states"][state_key][graph_name] = path_data
    
    # Write to file
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    
    print(f"Dijkstra's path data exported to {output_path}")
    return output_path


def generate_env_tensor_from_file(env_file: str) -> np.ndarray:
    """
    Generate an environment tensor from a file (placeholder function).
    In a real implementation, this would parse a file format to create an environment tensor.
    
    Args:
        env_file (str): Path to environment file
        
    Returns:
        np.ndarray: Environment tensor
    """
    # This is a placeholder. In a real implementation, this would parse a file format.
    # For now, just create a simple 11x11 grid with a lava crossing
    env_tensor = np.full((11, 11), "wall", dtype=object)
    
    # Set interior to floor
    env_tensor[1:-1, 1:-1] = "floor"
    
    # Create a lava crossing
    env_tensor[5, 1:-1] = "lava"
    
    # Create a path
    env_tensor[5, 5] = "floor"
    
    # Set goal
    env_tensor[1, 9] = "goal"
    
    return env_tensor


def main():
    """Main function for command-line usage"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Dijkstra's path analysis logs")
    parser.add_argument("--env-id", type=str, default="MiniGrid-LavaCrossingS11N5-v0",
                        help="Environment ID")
    parser.add_argument("--env-file", type=str, 
                        help="Path to environment file (if not using gym environment)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug output")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of path data even if file exists")
    args = parser.parse_args()
    
    # Get path to project root directory
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Check if file already exists and we're not forcing regeneration
    evaluations_dir = os.path.join(project_root, "Evaluations")
    output_path = os.path.join(evaluations_dir, f"{args.env_id}-{args.seed}.json")
    if os.path.exists(output_path) and not args.force:
        print(f"Path data file {output_path} already exists. Use --force to regenerate.")
        return

    # Generate environment tensor
    if args.env_file:
        env_tensor = generate_env_tensor_from_file(args.env_file)
    else:
        # This would normally create a gym environment and extract the tensor
        # For now, just use the placeholder function
        print(f"No environment file provided. Using placeholder for {args.env_id}")
        env_tensor = generate_env_tensor_from_file(None)
    
    # Analyze navigation graphs
    analysis_results = analyze_navigation_graphs(
        env_tensor=env_tensor,
        print_output=True,
        debug=args.debug
    )
    
    # Export to JSON
    output_path = export_path_data_to_json(
        analysis_results=analysis_results,
        env_tensor=env_tensor,
        env_id=args.env_id,
        seed=args.seed
    )
    
    print(f"Dijkstra's path analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    main() 
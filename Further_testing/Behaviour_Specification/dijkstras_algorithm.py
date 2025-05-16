import os
import sys
import rustworkx as rx
from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np
import argparse

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

from Behaviour_Specification.state_class import State
from Behaviour_Specification.generate_nodes import generate_state_nodes

def create_graphs_from_nodes(
    nodes: Dict[Tuple[int, int, int], State],
    lava_penalty_multiplier: int = 1
) -> Dict[str, rx.PyDiGraph]:
    """
    Create directed graphs from state nodes for different movement policies.
    
    Args:
        nodes: Dictionary mapping state tuples (x, y, orientation) to State objects,
              as returned by generate_state_nodes.
        lava_penalty_multiplier: Penalty multiplier for moving into lava cells in the 'dangerous' graph.
                      Default is 10 (10x weight compared to regular moves).
    
    Returns:
        Dictionary mapping policy names to corresponding directed graphs:
        - 'dangerous': Graph including lava cells with higher weights
        - 'standard': Graph avoiding lava cells
        - 'conservative': Graph avoiding lava cells and risky diagonals
    """
    # Create three separate directed graphs
    graphs = {
        'dangerous': rx.PyDiGraph(),
        'standard': rx.PyDiGraph(),
        'conservative': rx.PyDiGraph()
    }
    
    # Create a mapping from state tuples to node indices
    node_indices = {}
    
    # First, add all nodes to each graph
    print(f"Adding {len(nodes)} nodes to each graph...")
    for state_tuple, state_obj in nodes.items():
        # Add the node to each graph with the State object as data
        # The indices returned by add_node() are used to create edges later
        for graph_name, graph in graphs.items():
            idx = graph.add_node(state_obj)
            # Store the mapping for this graph
            if graph_name not in node_indices:
                node_indices[graph_name] = {}
            node_indices[graph_name][state_tuple] = idx
    
    # Statistics for tracking edge counts
    edge_counts = {graph_name: 0 for graph_name in graphs}
    lava_edge_count = 0
    
    # Debug info
    neighbor_counts = {
        'dangerous': 0,
        'standard': 0,
        'conservative': 0
    }
    
    # Now add edges based on neighbor relationships
    print("Adding edges based on neighbor relationships...")
    for state_tuple, state_obj in nodes.items():
        # Keep track of total neighbors
        neighbor_counts['dangerous'] += len(state_obj.valid_dangerous)
        neighbor_counts['standard'] += len(state_obj.valid_standard)
        neighbor_counts['conservative'] += len(state_obj.valid_conservative)
        
        # Add edges for each graph type
        neighbors = {
            'dangerous': state_obj.valid_dangerous,
            'standard': state_obj.valid_standard,
            'conservative': state_obj.valid_conservative
        }
        
        for graph_name, graph in graphs.items():
            # Use the appropriate neighbor list for this graph
            for neighbor_tuple in neighbors[graph_name]:
                if neighbor_tuple in node_indices[graph_name]:
                    src_idx = node_indices[graph_name][state_tuple]
                    dst_idx = node_indices[graph_name][neighbor_tuple]
                    
                    # For the dangerous graph, assign weights based on cell type
                    weight = 1
                    if graph_name == 'dangerous':
                        # Check if this is a lava cell
                        if neighbor_tuple in nodes and nodes[neighbor_tuple].type == 'lava':
                            weight = lava_penalty_multiplier
                            lava_edge_count += 1
                    
                    # Add edge with the appropriate weight
                    graph.add_edge(src_idx, dst_idx, weight)
                    edge_counts[graph_name] += 1
    
    # Print statistics
    print("Graph construction complete:")
    for graph_name, edge_count in edge_counts.items():
        print(f"  - {graph_name.capitalize()} graph: {len(nodes)} nodes, {edge_count} edges")
    print(f"  - Lava edges with penalty {lava_penalty_multiplier}x: {lava_edge_count}")
    
    # Print debug info
    print("\nTotal valid neighbors:")
    for graph_name, count in neighbor_counts.items():
        print(f"  - {graph_name.capitalize()}: {count} neighbors")
    
    return graphs

def compute_shortest_paths(
    graph: rx.PyDiGraph,
    source_state: Tuple[int, int, int],
    target_state: Optional[Union[Tuple[int, int, int], Tuple[int, int]]] = None,
    node_indices: Optional[Dict[Tuple[int, int, int], int]] = None
) -> Dict[int, List[int]]:
    """
    Compute shortest paths from source state to all other states (or a specific target).
    
    Args:
        graph: Directed graph created by create_graphs_from_nodes
        source_state: Starting state tuple (x, y, orientation)
        target_state: Optional target state tuple. Can be:
                     - A full state (x, y, orientation): finds path to that specific state
                     - A position (x, y): finds path to any orientation at that position
                     - None: compute paths to all states
        node_indices: Mapping of state tuples to node indices. If None, will be computed
    
    Returns:
        Dictionary mapping target node indices to paths (lists of node indices)
    """
    # If node_indices wasn't provided, compute it
    if node_indices is None:
        node_indices = {}
        for idx, node_data in enumerate(graph.nodes()):
            state = node_data.state
            node_indices[state] = idx
    
    # Get the source node index
    if source_state not in node_indices:
        raise ValueError(f"Source state {source_state} not found in graph")
    source_idx = node_indices[source_state]
    
    # If target specified, get its index or indices
    target_idx = None
    target_indices = []
    
    if target_state is not None:
        # Check if target_state is a position (x, y) or a full state (x, y, orientation)
        if len(target_state) == 2:
            # It's a position (x, y), find all states at this position with any orientation
            target_x, target_y = target_state
            target_indices = [
                node_indices[state] for state in node_indices.keys()
                if state[0] == target_x and state[1] == target_y
            ]
            
            if not target_indices:
                raise ValueError(f"No states found at position {target_state}")
        else:
            # It's a full state (x, y, orientation)
            if target_state not in node_indices:
                raise ValueError(f"Target state {target_state} not found in graph")
            target_idx = node_indices[target_state]
            target_indices = [target_idx]
    
    # Run Dijkstra's algorithm
    try:
        if target_indices:
            # Compute paths to all states
            paths = rx.dijkstra_shortest_paths(
                graph, source_idx, weight_fn=lambda x: x
            )
            
            # If we have target indices, only return paths to those targets
            if len(target_indices) == 1:
                # Single target case
                idx = target_indices[0]
                return {idx: paths[idx] if idx in paths else []}
            else:
                # Multiple targets case (different orientations at same position)
                # Find the shortest path among all orientations at the target position
                result = {}
                shortest_path = None
                shortest_idx = None
                
                for idx in target_indices:
                    if idx in paths and (shortest_path is None or len(paths[idx]) < len(shortest_path)):
                        shortest_path = paths[idx]
                        shortest_idx = idx
                
                if shortest_idx is not None:
                    result[shortest_idx] = shortest_path
                return result
        else:
            # All-targets shortest paths
            paths = rx.dijkstra_shortest_paths(
                graph, source_idx, weight_fn=lambda x: x
            )
            return paths
    except Exception as e:
        print(f"Error in Dijkstra's algorithm: {e}")
        # Let's try an alternative approach using single_source_dijkstra_path
        try:
            result = {}
            
            if target_indices:
                # Try to find paths to each target
                for idx in target_indices:
                    try:
                        path = rx.dijkstra_shortest_path(
                            graph, source_idx, target=idx, weight_fn=lambda x: x
                        )
                        result[idx] = path
                    except:
                        # No path to this target
                        pass
                return result
            else:
                # Get path to each target separately
                for idx in range(graph.num_nodes()):
                    try:
                        path = rx.dijkstra_shortest_path(
                            graph, source_idx, target=idx, weight_fn=lambda x: x
                        )
                        result[idx] = path
                    except:
                        # No path to this target
                        pass
                return result
        except Exception as e2:
            print(f"Error in alternative approach: {e2}")
            return {}

def test_graph_creation(lava_penalty_multiplier: int = 10, debug: bool = False):
    """
    Test the graph creation functionality with a simple environment.
    
    Args:
        lava_penalty_multiplier: Penalty for lava edges in the dangerous graph
        debug: Whether to print detailed diagnostic information (default: False)
    """
    print("\n===== Testing Graph Creation =====")
    
    # Create a simple test environment tensor (11x11 grid with walls, floors, lava, and a goal)
    print("Creating test environment tensor...")
    test_env = np.full((11, 11), "floor", dtype=object)
    # Add walls around the perimeter
    test_env[0, :] = "wall"
    test_env[-1, :] = "wall"
    test_env[:, 0] = "wall"
    test_env[:, -1] = "wall"
    # Add some lava
    test_env[3, 3:8] = "lava"
    test_env[7, 3:8] = "lava"
    # Add a goal
    test_env[-2, -2] = "goal"
    
    # Print the environment layout
    print("\n===== Test Environment Layout =====")
    # Create a mapping for simplified display
    display_map = {
        "wall": "W",
        "floor": "F",
        "lava": "L",
        "goal": "G",
        "unknown": "?"
    }
    
    # Print the grid
    for y in range(test_env.shape[0]):
        row = [display_map.get(test_env[y, x], "?") for x in range(test_env.shape[1])]
        print(" ".join(row))
    
    # Print a legend
    print("\nLegend:")
    print("W = Wall, F = Floor, L = Lava, G = Goal")
    print("===== End Environment Layout =====\n")
    
    # Generate state nodes
    print("Generating state nodes...")
    width, height = test_env.shape
    orientations = [0, 1, 2, 3]  # All possible orientations
    nodes = generate_state_nodes(test_env, (width, height), orientations, debug=debug)
    print(f"Generated {len(nodes)} state nodes")
    
    # Check neighbor assignments for a few key positions
    print("\n===== DEBUG: Neighbor Classification =====")
    test_positions = [(1, 1, 0), (3, 2, 1), (3, 3, 1)]  # Floor, floor near lava, lava positions
    
    for pos in test_positions:
        if pos in nodes:
            node = nodes[pos]
            pos_type = node.type
            print(f"\nPosition {pos} ({pos_type}):")
            print(f"  - Feasible neighbors: {len(node.feasible_neighbors)}")
            print("    " + ", ".join(str(n) for n in node.feasible_neighbors[:5]) + ("..." if len(node.feasible_neighbors) > 5 else ""))
            
            print(f"  - Dangerous neighbors: {len(node.valid_dangerous)}")
            print("    " + ", ".join(str(n) for n in node.valid_dangerous[:5]) + ("..." if len(node.valid_dangerous) > 5 else ""))
            
            print(f"  - Standard neighbors: {len(node.valid_standard)}")
            print("    " + ", ".join(str(n) for n in node.valid_standard[:5]) + ("..." if len(node.valid_standard) > 5 else ""))
            
            print(f"  - Conservative neighbors: {len(node.valid_conservative)}")
            print("    " + ", ".join(str(n) for n in node.valid_conservative[:5]) + ("..." if len(node.valid_conservative) > 5 else ""))
            
            # Check if lava is included in the correct neighbor sets
            lava_in_dangerous = any(nodes[n].type == "lava" for n in node.valid_dangerous if n in nodes)
            lava_in_standard = any(nodes[n].type == "lava" for n in node.valid_standard if n in nodes)
            lava_in_conservative = any(nodes[n].type == "lava" for n in node.valid_conservative if n in nodes)
            
            print(f"  - Contains lava: dangerous={lava_in_dangerous}, standard={lava_in_standard}, conservative={lava_in_conservative}")
        else:
            print(f"Position {pos} not found in nodes")
    
    # Look at a lava position directly
    lava_pos = (3, 3, 0)  # A lava cell
    if lava_pos in nodes:
        lava_node = nodes[lava_pos]
        print(f"\nLava node at {lava_pos}:")
        print(f"  - Type: {lava_node.type}")
        print(f"  - Included in dangerous graphs: {lava_pos in [k for node in nodes.values() for k in node.valid_dangerous]}")
        print(f"  - Included in standard graphs: {lava_pos in [k for node in nodes.values() for k in node.valid_standard]}")
    
    print("===== End DEBUG =====\n")
    
    # Create graphs
    print(f"Creating graphs with lava penalty {lava_penalty_multiplier}x...")
    graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier)
    
    # Test shortest paths in each graph
    print("\nTesting shortest path computation...")
    
    # Find floor and goal positions for testing
    floor_pos = (1, 1, 0)  # Top-left floor cell, facing up
    goal_x, goal_y = 9, 9  # Goal cell coordinates
    
    # Find all goal states (with any orientation)
    goal_states = [(goal_x, goal_y, ori) for ori in range(4) if (goal_x, goal_y, ori) in nodes]
    if not goal_states:
        print(f"No goal states found at position ({goal_x}, {goal_y})")
        return
    
    # Create node index mappings for each graph
    node_indices = {}
    for graph_name, graph in graphs.items():
        node_indices[graph_name] = {}
        for idx, node_data in enumerate(graph.nodes()):
            state = node_data.state
            node_indices[graph_name][state] = idx
    
    # Debug function to check cell types along paths
    def check_path_validity(graph_name, path_states, nodes):
        print(f"\n===== Path Validity Check for {graph_name} =====")
        
        # Check for lava in path
        lava_count = sum(1 for state in path_states if nodes[state].type == "lava")
        
        print(f"Path contains {lava_count} lava cells")
        if lava_count > 0:
            if graph_name == "standard" or graph_name == "conservative":
                print("ERROR: Standard or conservative path contains lava!")
                print("Lava cells in path:")
                for i, state in enumerate(path_states):
                    if nodes[state].type == "lava":
                        print(f"  Step {i}: {state}")
                        # Check if this lava cell is correctly identified in the environment
                        x, y, _ = state
                        print(f"  Environment tensor at this position: {test_env[y, x]}")
                        if test_env[y, x] != "lava":
                            print(f"  COORDINATE MISMATCH: Environment says {test_env[y, x]}, node says 'lava'")
                            # Check the neighbor for debugging
                            print(f"  Checking environment around position:")
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= ny < height and 0 <= nx < width:
                                        print(f"    ({nx}, {ny}): {test_env[ny, nx]}")
        
        # Check for diagonal moves
        diagonal_count = 0
        for i in range(len(path_states) - 1):
            if i > 0:  # Skip the first step which might be a rotation
                current = path_states[i]
                next_state = path_states[i+1]
                
                # Skip rotations (same position)
                if current[0] == next_state[0] and current[1] == next_state[1]:
                    continue
                
                # Identify move type
                move_type = nodes[current].identify_move_type(current, next_state, current[2])
                if move_type in ["diagonal-left", "diagonal-right"]:
                    diagonal_count += 1
                    
                    print(f"\nDiagonal move at step {i}: {current} -> {next_state} (type: {move_type})")
                    
                    # CRITICAL: Specific check for the problematic move
                    if current == (2, 7, 2) and next_state == (3, 8, 2):
                        print("*** FOUND PROBLEMATIC MOVE: (2,7,2) to (3,8,2) ***")
                        print(f"  Current state type: {nodes[current].type}")
                        print(f"  Next state type: {nodes[next_state].type}")
                        
                        # Check lava position
                        lava_pos = (3, 7)
                        print(f"  Lava position (3,7) in environment: {test_env[7, 3]}")
                        
                        # Check if the lava position is considered in safety check
                        cx, cy, orientation = current
                        forward = nodes[current].movement_vectors[orientation][0]
                        if move_type == "diagonal-left":
                            left_vec = nodes[current].left_vectors[orientation]
                            cells_to_check = [
                                (cx + forward[0], cy + forward[1]),  # Forward cell
                                (cx + left_vec[0], cy + left_vec[1])  # Left cell
                            ]
                        else:  # diagonal-right
                            right_vec = nodes[current].right_vectors[orientation]
                            cells_to_check = [
                                (cx + forward[0], cy + forward[1]),  # Forward cell
                                (cx + right_vec[0], cy + right_vec[1])  # Right cell
                            ]
                        
                        print(f"  Cells checked in safety function: {cells_to_check}")
                        print(f"  Should contain (3,7)? Let's check movement vectors:")
                        print(f"  Current orientation: {orientation}")
                        print(f"  Movement vectors for this orientation: {nodes[current].movement_vectors[orientation]}")
                        print(f"  Forward vector: {forward}")
                        if move_type == "diagonal-left":
                            print(f"  Left vector: {left_vec}")
                        else:
                            print(f"  Right vector: {right_vec}")
                            
                        # Mock the safety check calculation
                        print("  Recalculating cells to check for safety:")
                        forward_cell = (cx + forward[0], cy + forward[1])
                        print(f"    Forward cell: {forward_cell}")
                        if move_type == "diagonal-left":
                            direction_cell = (cx + left_vec[0], cy + left_vec[1])
                            print(f"    Left cell: {direction_cell}")
                        else:
                            direction_cell = (cx + right_vec[0], cy + right_vec[1])
                            print(f"    Right cell: {direction_cell}")
                        
                        # Check actual cell types
                        print("  Checking actual cell types in environment tensor:")
                        forward_x, forward_y = forward_cell
                        print(f"    Forward cell ({forward_x}, {forward_y}) type: {test_env[forward_y, forward_x]}")
                        dir_x, dir_y = direction_cell
                        print(f"    Direction cell ({dir_x}, {dir_y}) type: {test_env[dir_y, dir_x]}")
                    
                    # For conservative path, check if it's near lava
                    if graph_name == "conservative":
                        # Check surrounding cells for lava
                        cx, cy, orientation = current
                        
                        # Get the cells that would be checked by is_diagonal_safe
                        forward = nodes[current].movement_vectors[orientation][0]
                        if move_type == "diagonal-left":
                            left_vec = nodes[current].left_vectors[orientation]
                            adjacent_cells = [(cx + forward[0], cy + forward[1]), (cx + left_vec[0], cy + left_vec[1])]
                        else:  # diagonal-right
                            right_vec = nodes[current].right_vectors[orientation]
                            adjacent_cells = [(cx + forward[0], cy + forward[1]), (cx + right_vec[0], cy + right_vec[1])]
                        
                        # Check if any adjacent cell is lava
                        print(f"  Checking cells around diagonal move: {adjacent_cells}")
                        for adj_x, adj_y in adjacent_cells:
                            if (0 <= adj_x < width and 0 <= adj_y < height):
                                cell_type = test_env[adj_y, adj_x]
                                print(f"  Cell ({adj_x}, {adj_y}) has type: {cell_type}")
                                if cell_type == "lava":
                                    print(f"ERROR: Conservative path takes diagonal move near lava at step {i}: {current} -> {next_state}")
                                    print(f"Adjacent lava at: ({adj_x}, {adj_y}), type: {cell_type}")
                                    
                                    # Check why the diagonal safety check didn't catch this
                                    print(f"SAFETY CHECK FAILURE ANALYSIS:")
                                    print(f"  Lava at ({adj_x}, {adj_y}) should have been detected in:")
                                    print(f"  is_diagonal_safe({current}, {next_state}, {orientation}, {move_type})")
        
        print(f"Path contains {diagonal_count} diagonal moves")
        print("===== End Validity Check =====")
        
        return lava_count, diagonal_count
    
    # For visualizing paths in the grid
    def visualize_path(graph_name, path_states):
        print(f"\n===== Path Visualization for {graph_name} =====")
        
        # Create a display matrix for visualization
        display = np.full((height, width), ".", dtype=str)
        
        # Fill in walls, lava, and goal for context
        for y in range(height):
            for x in range(width):
                if test_env[y, x] == "wall":
                    display[y, x] = "W"
                elif test_env[y, x] == "lava":
                    display[y, x] = "L"
                elif test_env[y, x] == "goal":
                    display[y, x] = "G"
        
        # Mark the path with numbers or symbols
        for i, state_tuple in enumerate(path_states):
            x, y, _ = state_tuple
            
            # Use special symbols for start and end
            if i == 0:
                display[y, x] = "S"  # Start
            elif i == len(path_states) - 1:
                display[y, x] = "E"  # End
            else:
                # For the path, use a directional symbol based on orientation
                ori = state_tuple[2]
                symbols = {0: "^", 1: ">", 2: "v", 3: "<"}
                display[y, x] = symbols[ori]
        
        # Print the visualization
        for row in display:
            print(" ".join(row))
        
        print("\nLegend:")
        print("W = Wall, L = Lava, G = Goal")
        print("S = Start, E = End")
        print("^ = Facing Up, > = Facing Right, v = Facing Down, < = Facing Left")
        print("===== End Path Visualization =====")
    
    # Compute and show paths
    for graph_name, graph in graphs.items():
        print(f"\nGraph: {graph_name}")
        
        try:
            # Check graph connectivity
            print(f"Graph connectivity: {rx.is_weakly_connected(graph)}")
            
            # Test if source is in the graph
            if floor_pos not in node_indices[graph_name]:
                print(f"Start position {floor_pos} not found in graph")
                continue
            
            # Find shortest path to the goal position with any orientation
            print(f"Testing path to goal at position ({goal_x}, {goal_y}) with any orientation...")
            
            try:
                # Pass just the position (x, y) instead of a full state
                paths = compute_shortest_paths(
                    graph, floor_pos, (goal_x, goal_y), node_indices[graph_name]
                )
                
                if not paths:
                    print(f"No path found from {floor_pos} to goal position ({goal_x}, {goal_y})")
                    continue
                
                # Get the target index and path
                target_idx = next(iter(paths.keys()))
                shortest_path = paths[target_idx]
                
                if not shortest_path:
                    print(f"Path is empty")
                    continue
                
                # Convert indices to states
                path_states = []
                for idx in shortest_path:
                    try:
                        state_obj = graph.nodes()[idx]
                        path_states.append(state_obj.state)
                    except Exception as e:
                        print(f"Error retrieving state at index {idx}: {e}")
                
                # Check if the final state is a goal
                if path_states:
                    final_x, final_y, _ = path_states[-1]
                    is_goal = (test_env[final_y, final_x] == "goal")
                    if is_goal:
                        print(f"Path successfully reaches goal at position ({final_x}, {final_y})")
                    else:
                        print(f"WARNING: Final state {path_states[-1]} is not a goal!")
                
                # Print path info
                # Note: We distinguish between "states" and "steps":
                # - The path contains N states (including starting and ending states)
                # - But there are N-1 steps (transitions between states)
                # This ensures consistency with "GOAL REACHED after X steps" message
                num_states = len(path_states)
                num_steps = num_states - 1
                print(f"Path states: {num_states} (including starting state)")
                print(f"Path steps: {num_steps}")
                print(f"Start: {path_states[0] if path_states else 'N/A'}")
                print(f"End: {path_states[-1] if path_states else 'N/A'}")
                
                # Check path validity
                check_path_validity(graph_name, path_states, nodes)
                
                # Show state types along the path
                print("\nState types along path:")
                for i, state_tuple in enumerate(path_states):
                    state_obj = nodes[state_tuple]
                    print(f"  {i:2d}: {state_tuple} - {state_obj.type}")
                    
                    # Stop when we reach any goal state
                    x, y, _ = state_tuple
                    if test_env[y, x] == "goal":
                        # Note: We report steps as (len(path_states) - 1) because the first state (i=0)
                        # is the starting position, not a step. This ensures consistency with "Number of steps"
                        # reported above. We're counting state transitions (moves), not states themselves.
                        steps_taken = i
                        print(f"  GOAL REACHED after {steps_taken} steps (at state {i})")
                
                # Visualize the path in the grid
                visualize_path(graph_name, path_states)
                
            except Exception as e:
                print(f"Error computing path: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error computing path: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n===== Graph Creation Test Complete =====")
    
    return graphs, nodes

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Test graph creation and pathfinding")
    parser.add_argument("--lava-penalty", type=int, default=1,
                        help="Penalty multiplier for lava cells (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed debug output including diagonal safety checks")
    
    args = parser.parse_args()
    
    # Test the graph creation with command line arguments
    test_graph_creation(lava_penalty_multiplier=args.lava_penalty, debug=args.debug)

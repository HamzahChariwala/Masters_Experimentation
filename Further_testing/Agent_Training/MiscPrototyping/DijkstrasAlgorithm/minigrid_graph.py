"""
MiniGrid to Graph Converter

This module provides functionality to convert MiniGrid environments to graph representations
that can be used with Dijkstra's algorithm to find optimal paths.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from collections import defaultdict
import os

try:
    import minigrid
    from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper
except ImportError:
    print("Warning: minigrid not found. Please install with: pip install minigrid")

# Define the directions and their corresponding vectors
# 0: right, 1: down, 2: left, 3: up
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRECTION_NAMES = ['right', 'down', 'left', 'up']

# Define diagonal directions
# These are the forward diagonal movements from each orientation
# For orientation 0 (right): (1,1) = right-down, (1,-1) = right-up
# For orientation 1 (down): (1,1) = down-right, (-1,1) = down-left
# For orientation 2 (left): (-1,1) = left-down, (-1,-1) = left-up
# For orientation 3 (up): (-1,-1) = up-left, (1,-1) = up-right
DIAGONAL_DIRECTIONS = {
    0: [(1, 1), (1, -1)],   # right: down-right, up-right
    1: [(1, 1), (-1, 1)],   # down: down-right, down-left
    2: [(-1, 1), (-1, -1)], # left: left-down, left-up
    3: [(-1, -1), (1, -1)]  # up: up-left, up-right
}

# Define actions
# 0: turn left, 1: turn right, 2: move forward, 3: move left-diagonal, 4: move right-diagonal
ACTION_NAMES = ['left', 'right', 'forward', 'left_diagonal', 'right_diagonal', 'toggle', 'pickup', 'drop', 'done']

class MiniGridGraphConverter:
    """
    Converts MiniGrid environments to graph representations suitable for Dijkstra's algorithm.
    
    This class handles:
    1. Converting a MiniGrid environment to a directed graph
    2. Supporting different lava handling modes
    3. Visualizing the graph
    4. Computing optimal paths using Dijkstra's algorithm
    """
    
    def __init__(self):
        """Initialize the converter"""
        self.env = None
        self.graph = None
        self.optimal_costs = None
        self.env_id = None
    
    def create_graph_from_env(self, env_id, seed=None, lava_mode="blocked", lava_cost_multiplier=5.0):
        """
        Create a graph representation of a MiniGrid environment.
        
        Args:
            env_id (str): The environment ID to create
            seed (int, optional): Random seed for environment
            lava_mode (str): How to handle lava tiles ('normal', 'costly', 'blocked')
            lava_cost_multiplier (float): Cost multiplier for lava tiles when lava_mode is 'costly'
            
        Returns:
            nx.DiGraph: A directed graph representation of the environment
        """
        # Store the original env_id for later reference
        self.env_id = env_id
        
        # Create the environment
        try:
            self.env = gym.make(env_id)
            if seed is not None:
                self.env.reset(seed=seed)
            else:
                self.env.reset()
        except (gym.error.NameNotFound, gym.error.NamespaceNotFound, ValueError) as e:
            print(f"Warning: {e}")
            print(f"Attempting to use minigrid directly...")
            try:
                # Try importing from minigrid directly
                import importlib
                import re
                
                # Extract environment name from the ID
                if ':' in env_id:
                    env_name = env_id.split(':')[-1]
                else:
                    env_name = env_id
                
                # Handle special cases where MiniGrid- prefix might be missing
                if not env_name.startswith('MiniGrid-'):
                    env_name = f"MiniGrid-{env_name}"
                
                print(f"Using environment: {env_name}")
                
                # Import the environment class directly
                try:
                    # First, try importing from minigrid.envs
                    module_name = "minigrid.envs"
                    module = importlib.import_module(module_name)
                    env_classes = [cls for name, cls in module.__dict__.items() 
                               if isinstance(cls, type) and name.endswith('Env')]
                    
                    # Find the right environment class based on name pattern
                    env_class = None
                    for cls in env_classes:
                        # Remove 'Env' suffix and compare
                        cls_name = cls.__name__
                        if cls_name.endswith('Env'):
                            cls_name = cls_name[:-3]
                        
                        # Convert CamelCase to kebab-case for comparison
                        cls_name_kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', cls_name)
                        if cls_name_kebab.lower() in env_name.lower():
                            env_class = cls
                            break
                    
                    if env_class:
                        # Create the environment instance
                        self.env = env_class()
                    else:
                        # If no matching class found, try another approach
                        raise ImportError("Environment class not found")
                        
                except (ImportError, AttributeError):
                    # If the above direct import failed, try a more standard approach
                    self.env = gym.make(env_name)
                
                # Apply FullyObsWrapper for better graph creation
                self.env = FullyObsWrapper(self.env)
                
                # Reset the environment with the seed
                if seed is not None:
                    self.env.reset(seed=seed)
                else:
                    self.env.reset()
                    
            except Exception as e2:
                raise ValueError(f"Failed to create environment: {e2}")
        
        # Make sure the environment is fully observable
        if not hasattr(self.env, 'grid'):
            # If we don't have direct access to the grid, try to apply FullyObsWrapper
            try:
                self.env = FullyObsWrapper(self.env)
                self.env.reset()
            except Exception as e:
                raise ValueError(f"Environment does not provide grid access and cannot be wrapped: {e}")
        
        # Create a directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes and edges for each cell and orientation
        width, height = self.env.width, self.env.height
        
        # Iterate through all positions and orientations
        for x in range(width):
            for y in range(height):
                # Check if the cell is a wall or goal (those are excluded)
                cell = self.env.grid.get(x, y)
                
                # Skip walls and goals for node creation
                if cell is not None and cell.type in ['wall', 'goal']:
                    continue
                
                # Skip lava if it's blocked mode
                if lava_mode == "blocked" and cell is not None and cell.type == 'lava':
                    continue
                
                # For each valid cell, create nodes for all 4 orientations
                for dir_idx in range(4):
                    node = (x, y, dir_idx)
                    self.graph.add_node(node)  # Simplified - we'll extract position from the node itself
                    
                    # Add edges for available actions from this node
                    
                    # 1. Add turn left/right edges (these are always valid actions)
                    # Turn left: 0 -> 3, 1 -> 0, 2 -> 1, 3 -> 2
                    left_dir = (dir_idx - 1) % 4
                    self.graph.add_edge(node, (x, y, left_dir), action=0, cost=1)
                    
                    # Turn right: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0
                    right_dir = (dir_idx + 1) % 4
                    self.graph.add_edge(node, (x, y, right_dir), action=1, cost=1)
                    
                    # 2. Add forward movement edge if possible
                    dx, dy = DIRECTIONS[dir_idx]
                    new_x, new_y = x + dx, y + dy
                    
                    # Check if the new position is valid
                    if 0 <= new_x < width and 0 <= new_y < height:
                        next_cell = self.env.grid.get(new_x, new_y)
                        
                        # Forward movement is only valid if:
                        # - The target cell is empty, lava (depending on lava_mode), or a goal
                        is_valid_move = (
                            next_cell is None or  # Empty cell
                            (next_cell.type == 'lava' and lava_mode != "blocked") or  # Lava cell (if not blocked)
                            next_cell.type == 'empty' or  # Empty cell
                            next_cell.type == 'goal'  # Goal cell
                        )
                        
                        if is_valid_move:
                            # Add forward movement edge
                            # Set cost based on cell type
                            cost = 1  # Default cost
                            if next_cell is not None and next_cell.type == 'lava' and lava_mode == "costly":
                                cost = lava_cost_multiplier
                            
                            # Only add edge if the move is optimal
                            # Don't add edges into lava in costly mode unless absolutely necessary
                            should_add_edge = True
                            if next_cell is not None and next_cell.type == 'lava' and lava_mode == "costly":
                                # Check if there's a path around the lava
                                # This is a simple check - if any adjacent cell is walkable, prefer that
                                has_alternative = False
                                for alt_dx, alt_dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    if (alt_dx, alt_dy) != (dx, dy):  # Skip the direction we're checking
                                        alt_x, alt_y = x + alt_dx, y + alt_dy
                                        if 0 <= alt_x < width and 0 <= alt_y < height:
                                            alt_cell = self.env.grid.get(alt_x, alt_y)
                                            if (alt_cell is None or 
                                                (alt_cell.type != 'wall' and 
                                                 alt_cell.type != 'lava')):
                                                has_alternative = True
                                                break
                                
                                # If there's an alternative, only add the lava edge if it's actually better
                                # For costly mode, we'll assume lava is never optimal if there's an alternative
                                if has_alternative:
                                    should_add_edge = False
                            
                            if should_add_edge:
                                self.graph.add_edge(node, (new_x, new_y, dir_idx), action=2, cost=cost)
                    
                    # 3. Add diagonal movement edges (if possible)
                    # For each orientation, there are two possible diagonal moves
                    for diagonal_idx, (diag_dx, diag_dy) in enumerate(DIAGONAL_DIRECTIONS[dir_idx]):
                        diag_x, diag_y = x + diag_dx, y + diag_dy
                        
                        # Check if diagonal position is valid
                        if 0 <= diag_x < width and 0 <= diag_y < height:
                            diag_cell = self.env.grid.get(diag_x, diag_y)
                            
                            # Diagonal movement is only valid if:
                            # - The target cell is empty, lava (depending on lava_mode), or a goal
                            is_valid_diagonal = (
                                diag_cell is None or  # Empty cell
                                (diag_cell.type == 'lava' and lava_mode != "blocked") or  # Lava cell (if not blocked)
                                diag_cell.type == 'empty' or  # Empty cell
                                diag_cell.type == 'goal'  # Goal cell
                            )
                            
                            # Check the adjacent cells for the diagonal move
                            # These are the cells we would pass through diagonally
                            adj1_x, adj1_y = x + diag_dx, y
                            adj2_x, adj2_y = x, y + diag_dy
                            
                            adj1_cell = self.env.grid.get(adj1_x, adj1_y)
                            adj2_cell = self.env.grid.get(adj2_x, adj2_y)
                            
                            # Check if adjacent cells are blocked by wall or lava (in blocked mode)
                            adj1_blocked = False
                            if adj1_cell is not None:
                                if adj1_cell.type == 'wall':
                                    adj1_blocked = True
                                elif adj1_cell.type == 'lava' and lava_mode == "blocked":
                                    adj1_blocked = True
                                    
                            adj2_blocked = False
                            if adj2_cell is not None:
                                if adj2_cell.type == 'wall':
                                    adj2_blocked = True
                                elif adj2_cell.type == 'lava' and lava_mode == "blocked":
                                    adj2_blocked = True
                            
                            # For blocked mode, we need both adjacent cells to be passable
                            # For other modes, we'll be more lenient - allow diagonals as long as the target is valid
                            should_add_diagonal = is_valid_diagonal
                            if lava_mode == "blocked":
                                should_add_diagonal = is_valid_diagonal and not adj1_blocked and not adj2_blocked
                            else:
                                # In normal and costly modes, we'll allow diagonal movement as long as:
                                # 1. The target diagonal cell is valid
                                # 2. At least one of the adjacent cells is not blocked
                                should_add_diagonal = is_valid_diagonal and not (adj1_blocked and adj2_blocked)
                            
                                # Additional check for costly mode to prevent diagonal moves into lava
                                if lava_mode == "costly" and diag_cell is not None and diag_cell.type == 'lava':
                                    # Check if either adjacent cell is lava
                                    adj1_is_lava = adj1_cell is not None and adj1_cell.type == 'lava'
                                    adj2_is_lava = adj2_cell is not None and adj2_cell.type == 'lava'
                                    
                                    # If diagonal target is lava and at least one adjacent is lava,
                                    # it's usually better to go around
                                    if adj1_is_lava or adj2_is_lava:
                                        should_add_diagonal = False
                            
                            if should_add_diagonal:
                                # Add diagonal movement edge
                                # Set cost based on cell type
                                diag_cost = 1.4  # Slightly higher cost for diagonal moves (√2 ≈ 1.414)
                                if diag_cell is not None and diag_cell.type == 'lava' and lava_mode == "costly":
                                    diag_cost = lava_cost_multiplier * 1.4
                                
                                # Action 3 for left diagonal, 4 for right diagonal
                                action = 3 if diagonal_idx == 0 else 4
                                self.graph.add_edge(node, (diag_x, diag_y, dir_idx), action=action, cost=diag_cost)

        return self.graph
    
    def run_dijkstra(self, goal_pos=None):
        """
        Run Dijkstra's algorithm to find optimal paths to a goal.
        
        Args:
            goal_pos (tuple, optional): The goal position (x, y). If None, auto-detect.
            
        Returns:
            dict: A dictionary mapping each node to its optimal cost to the goal
        """
        if self.graph is None:
            raise ValueError("Graph not created. Call create_graph_from_env first.")
        
        # Find the goal position if not provided
        if goal_pos is None:
            # Search for the goal in the grid
            width, height = self.env.width, self.env.height
            for x in range(width):
                for y in range(height):
                    cell = self.env.grid.get(x, y)
                    if cell is not None and cell.type == 'goal':
                        goal_pos = (x, y)
                        break
                if goal_pos is not None:
                    break
            
            if goal_pos is None:
                raise ValueError("Goal position not found in the environment.")
        
        # Since we want the optimal cost from each cell+orientation to the goal, 
        # and Dijkstra gives us costs from a source to all other nodes,
        # we need to invert our graph
        inverse_graph = self.graph.reverse()
        
        # For the goal, we need to calculate for all possible orientations
        # since the agent can reach the goal with any orientation
        goal_nodes = [(goal_pos[0], goal_pos[1], dir_idx) for dir_idx in range(4)]
        
        # Create a virtual goal node and connect it to all goal orientation nodes
        virtual_goal = "virtual_goal"
        inverse_graph.add_node(virtual_goal)
        for goal_node in goal_nodes:
            if goal_node in inverse_graph.nodes:
                inverse_graph.add_edge(virtual_goal, goal_node, cost=0)
        
        # Run Dijkstra's algorithm from the virtual goal node
        # This will find the shortest path from the goal to every other node
        costs = nx.single_source_dijkstra_path_length(
            inverse_graph, virtual_goal, weight='cost'
        )
        
        # Remove the virtual goal from the results
        costs.pop(virtual_goal, None)
        
        # Additional step: For cell positions that have diagonal access,
        # update the costs if diagonal movement is more efficient
        # This is a safeguard to ensure we're properly considering all movement options
        width, height = self.env.width, self.env.height
        updated = True
        max_iterations = 5  # Limit iterations to prevent infinite loops
        iterations = 0
        
        while updated and iterations < max_iterations:
            updated = False
            iterations += 1
            
            # Create a copy of current costs for this iteration
            new_costs = costs.copy()
            
            # For each node
            for node in costs:
                if isinstance(node, tuple) and len(node) == 3:
                    x, y, dir_idx = node
                    
                    # Check all possible movements from this position
                    # 1. Regular forward movement
                    dx, dy = DIRECTIONS[dir_idx]
                    new_x, new_y = x + dx, y + dy
                    if (new_x, new_y, dir_idx) in costs:
                        forward_cost = costs[(new_x, new_y, dir_idx)] + 1
                        if forward_cost < costs[node]:
                            new_costs[node] = forward_cost
                            updated = True
                    
                    # 2. Diagonal movements
                    for diag_dx, diag_dy in DIAGONAL_DIRECTIONS[dir_idx]:
                        diag_x, diag_y = x + diag_dx, y + diag_dy
                        if (diag_x, diag_y, dir_idx) in costs:
                            diag_cost = costs[(diag_x, diag_y, dir_idx)] + 1.4  # √2 ≈ 1.414
                            if diag_cost < costs[node]:
                                new_costs[node] = diag_cost
                                updated = True
            
            # Update costs with the new values
            if updated:
                costs = new_costs
        
        self.optimal_costs = costs
        return costs
    
    def export_optimal_costs(self, output_file):
        """
        Export the optimal costs to a JSON file.
        
        Args:
            output_file (str): The path to the output file
            
        Returns:
            dict: The exported optimal costs
        """
        if self.optimal_costs is None:
            raise ValueError("Optimal costs not calculated. Call run_dijkstra first.")
        
        # Create a 2D grid of optimal costs
        width, height = self.env.width, self.env.height
        
        # Initialize a 2D array of arrays, where each cell contains costs for 4 orientations
        cost_grid = []
        for y in range(height):
            row = []
            for x in range(width):
                cell_costs = [float('inf')] * 4  # Initialize with infinity for all 4 orientations
                
                # Check if this cell has valid costs
                cell = self.env.grid.get(x, y)
                if cell is None or (cell.type != 'wall' and cell.type != 'goal'):
                    # For each orientation, get the cost if it exists
                    for dir_idx in range(4):
                        node = (x, y, dir_idx)
                        if node in self.optimal_costs:
                            cell_costs[dir_idx] = self.optimal_costs[node]
                row.append(cell_costs)
            cost_grid.append(row)
        
        # Create result dictionary with proper formatting
        # Use original env_id without the gymnasium prefix for better readability
        env_name = self.env_id if self.env_id else "Unknown"
        # Clean up the environment name
        if ':' in env_name:
            env_name = env_name.split(':')[-1]
        
        if hasattr(self.env, 'np_random'):
            seed_value = self.env.np_random.integers(0, 2**31-1)  # Get a seed-like value
        else:
            seed_value = "unknown"
            
        result_key = f"{env_name}_{seed_value}"
        
        # Create the result dictionary
        result = {
            result_key: {
                "grid": cost_grid,
                "width": width,
                "height": height,
                "walls": [],  # Store wall positions
                "lava": [],   # Store lava positions
                "goal": []    # Store goal position
            }
        }
        
        # Add wall, lava, and goal positions
        for y in range(height):
            for x in range(width):
                cell = self.env.grid.get(x, y)
                if cell is not None:
                    if cell.type == 'wall':
                        result[result_key]["walls"].append([x, y])
                    elif cell.type == 'lava':
                        result[result_key]["lava"].append([x, y])
                    elif cell.type == 'goal':
                        result[result_key]["goal"].append([x, y])
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def visualize_graph(self, output_file=None, show_costs=True):
        """
        Visualize the graph and optionally the optimal costs.
        
        Args:
            output_file (str, optional): The path to save the visualization
            show_costs (bool): Whether to show the optimal costs
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.graph is None:
            raise ValueError("Graph not created. Call create_graph_from_env first.")
        
        # Create a figure with a larger size for better visibility
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get the height from the environment for correct y-coordinate inversion
        height = self.env.height
        
        # Get node positions
        # Use the node itself to extract position information (since it's a tuple (x, y, dir_idx))
        pos = {}
        for node in self.graph.nodes():
            if isinstance(node, tuple) and len(node) == 3:
                x, y, _ = node
                pos[node] = (x, height - y - 1)  # Invert y for visualization (bottom-right as goal)
            else:
                # Handle unexpected node types
                pos[node] = (0, 0)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_size=200, alpha=0.7)
        
        # Draw edges with different colors based on action type
        action_colors = ['red', 'green', 'blue', 'orange', 'purple']  # Added colors for diagonal moves
        for action, color in enumerate(action_colors):
            action_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('action') == action]
            nx.draw_networkx_edges(self.graph, pos, edgelist=action_edges, edge_color=color, alpha=0.4, width=1.5, arrows=True)
        
        # Add a legend for action colors
        ax.plot([], [], color='red', label='Turn Left')
        ax.plot([], [], color='green', label='Turn Right')
        ax.plot([], [], color='blue', label='Move Forward')
        ax.plot([], [], color='orange', label='Move Left Diagonal')
        ax.plot([], [], color='purple', label='Move Right Diagonal')
        ax.legend()
        
        # Add costmap visualization if costs are available and requested
        if show_costs and self.optimal_costs is not None:
            # Create a 2D array of costs for visualization
            width, height = self.env.width, self.env.height
            cost_map = np.full((height, width), np.inf)
            
            # For each position, find the minimum cost across all orientations
            for (x, y, _), cost in self.optimal_costs.items():
                if 0 <= x < width and 0 <= y < height:
                    cost_map[y, x] = min(cost_map[y, x], cost)
            
            # Replace inf with NaN for better visualization
            cost_map = np.where(cost_map == np.inf, np.nan, cost_map)
            
            # Create a separate axes for the costmap
            ax_cost = ax.twinx().twiny()
            im = ax_cost.imshow(cost_map, cmap='viridis', alpha=0.6, origin='lower')  # origin='lower' for bottom-right goal
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_cost)
            cbar.set_label('Minimum Cost to Goal')
            
            # Hide the axes of the costmap
            ax_cost.set_xticks([])
            ax_cost.set_yticks([])
        
        # Add title and labels
        plt.title('MiniGrid Environment as a Graph')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Save or show the visualization
        if output_file:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig

def generate_optimal_paths_for_env(env_id, seed=None, lava_modes=None, output_dir="./", visualize=True):
    """
    Generate optimal paths for a MiniGrid environment using Dijkstra's algorithm.
    
    Args:
        env_id (str): The environment ID to analyze
        seed (int, optional): Random seed for environment
        lava_modes (list, optional): List of lava modes to analyze. Defaults to all modes.
        output_dir (str): Directory to save outputs
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: A dictionary mapping each lava mode to its optimal costs
    """
    if lava_modes is None:
        lava_modes = ["normal", "costly", "blocked"]
    
    results = {}
    converter = MiniGridGraphConverter()
    
    for lava_mode in lava_modes:
        print(f"  Analyzing with lava mode: {lava_mode}")
        # Create the graph with the specified lava mode
        graph = converter.create_graph_from_env(
            env_id, seed=seed, lava_mode=lava_mode, 
            lava_cost_multiplier=5.0 if lava_mode == "costly" else 1.0
        )
        
        # Run Dijkstra's algorithm
        costs = converter.run_dijkstra()
        
        # Export the optimal costs
        env_seed_str = str(seed) if seed is not None else "random"
        # Extract the environment name without the gymnasium prefix
        env_name = env_id.split(':')[-1] if ':' in env_id else env_id
        env_name = env_name.replace('-', '_')
        
        output_file = f"{output_dir}/{env_name}_{env_seed_str}_{lava_mode}.json"
        result = converter.export_optimal_costs(output_file)
        
        # Visualize if requested
        if visualize:
            vis_file = f"{output_dir}/{env_name}_{env_seed_str}_{lava_mode}.png"
            converter.visualize_graph(vis_file, show_costs=True)
        
        results[lava_mode] = result
    
    return results

if __name__ == "__main__":
    # Example usage
    env_id = "MiniGrid-Empty-8x8-v0"
    seed = 12345
    
    # Generate optimal paths for all lava modes
    generate_optimal_paths_for_env(
        env_id, 
        seed=seed,
        lava_modes=["normal", "costly", "blocked"],
        output_dir="./results",
        visualize=True
    ) 
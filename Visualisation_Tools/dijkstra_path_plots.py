#!/usr/bin/env python3
"""
Generate path visualizations showing routes taken by different Dijkstra rulesets
overlaid on environment plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import minigrid
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from matplotlib.cm import magma
from Behaviour_Specification.StateGeneration.generate_nodes import generate_state_nodes
from Behaviour_Specification.DijkstrasAlgorithm.dijkstras_algorithm import create_graphs_from_nodes, compute_shortest_paths
from Visualisation_Tools.env_plots import extract_grid_from_env


def get_dijkstra_path(seed, source_state, ruleset, lava_penalty=1):
    """
    Get the path taken by a specific dijkstra ruleset.
    
    Args:
        seed (int): Environment seed
        source_state (tuple): Starting state (x, y, orientation)
        ruleset (str): Ruleset name ('standard', 'conservative', 'dangerous_1', 'dangerous_5')
        lava_penalty (int): Penalty for lava cells in dangerous rulesets
        
    Returns:
        list: Path as list of (x, y, orientation) tuples
    """
    # Create environment
    env = gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode='rgb_array')
    obs, info = env.reset(seed=seed)
    
    # Get grid data and convert to environment tensor
    if 'log_data' in info and 'new_image' in info['log_data']:
        grid_data = info['log_data']['new_image'][1]
        grid_array = np.array(grid_data)
    else:
        grid_array = extract_grid_from_env(env)
    
    height, width = grid_array.shape
    
    # Convert grid_array to the format expected by generate_state_nodes
    # The state generation expects string types
    env_tensor = np.empty((height, width), dtype=object)
    for y in range(height):
        for x in range(width):
            cell_value = grid_array[y, x]
            if cell_value == 0:  # Wall
                env_tensor[y, x] = 'wall'
            elif cell_value == 1:  # Floor
                env_tensor[y, x] = 'floor'
            elif cell_value == 2:  # Lava
                env_tensor[y, x] = 'lava'
            elif cell_value == 3:  # Goal
                env_tensor[y, x] = 'goal'
            else:
                env_tensor[y, x] = 'floor'  # Default
    
    # Generate state nodes with correct parameters
    nodes = generate_state_nodes(env_tensor, (width, height))
    
    # Create graphs with appropriate penalty
    if ruleset in ['dangerous_1', 'dangerous_3']:
        penalty = int(ruleset.split('_')[1])
        graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier=penalty)
        graph_key = 'dangerous'
    else:
        graphs = create_graphs_from_nodes(nodes, lava_penalty_multiplier=lava_penalty)
        graph_key = ruleset
    
    graph = graphs[graph_key]
    
    # Find goal position 
    goal_pos = None
    for y in range(height):
        for x in range(width):
            if grid_array[y, x] == 3:  # Goal
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    
    if not goal_pos:
        raise ValueError(f"No goal found in environment seed {seed}")
    
    # Create node indices mapping
    node_indices = {}
    for idx, node_data in enumerate(graph.nodes()):
        state = node_data.state
        node_indices[state] = idx
    
    # Compute shortest path to goal position (any orientation)
    try:
        paths = compute_shortest_paths(graph, source_state, goal_pos, node_indices)
        
        if not paths:
            return []
        
        # Get the path
        target_idx = next(iter(paths.keys()))
        shortest_path = paths[target_idx]
        
        # Convert indices to states
        path_states = []
        for idx in shortest_path:
            state_obj = graph.nodes()[idx]
            path_states.append(state_obj.state)
        
        env.close()
        return path_states
        
    except Exception as e:
        print(f"Error computing path for {ruleset} on seed {seed}: {e}")
        env.close()
        return []


def plot_dijkstra_paths(seed, env_id="MiniGrid-LavaCrossingS11N5-v0", save_path=None):
    """
    Create separate visualizations showing paths from different Dijkstra rulesets.
    
    Args:
        seed (int): Environment seed
        env_id (str): Environment ID
        save_path (str): Path to save the plot. If None, saves to dijkstras/
        
    Returns:
        list: Paths where the plots were saved
    """
    # Create environment and get grid
    env = gym.make(env_id, render_mode='rgb_array')
    obs, info = env.reset(seed=seed)
    
    if 'log_data' in info and 'new_image' in info['log_data']:
        grid_data = info['log_data']['new_image'][1]
        grid_array = np.array(grid_data)
    else:
        grid_array = extract_grid_from_env(env)
    
    height, width = grid_array.shape
    
    # Define rulesets and colors (conservative darkest to dangerous_1 lightest)
    # Using more distinct values from magma colormap
    rulesets = ['conservative', 'standard', 'dangerous_3', 'dangerous_1']
    colors = [magma(0.1), magma(0.35), magma(0.65), magma(0.8)]  # Darkened dangerous_1 from 0.9 to 0.8
    
    # Starting state: (1, 1, 1) - position (1,1) facing right (East)
    source_state = (1, 1, 1)
    
    # Get paths for each ruleset
    paths = {}
    for ruleset in rulesets:
        try:
            path = get_dijkstra_path(seed, source_state, ruleset)
            paths[ruleset] = path
            print(f"Got path for {ruleset}: {len(path)} states")
        except Exception as e:
            print(f"Failed to get path for {ruleset}: {e}")
            paths[ruleset] = []
    
    # Define base colors
    wall_color = '#5d5d5d'
    floor_color = '#f0f0f0'
    lava_color = '#a0a0a0'
    grid_color = '#000000'
    goal_color = '#000000'
    
    # Create separate plots for each ruleset
    save_paths = []
    
    for ruleset, color in zip(rulesets, colors):
        path = paths.get(ruleset, [])
        if not path:
            print(f"Skipping {ruleset} - no path found")
            continue
            
        # Create figure and axis for this ruleset
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw base environment
        for y in range(height):
            for x in range(width):
                cell_value = grid_array[y, x]
                
                if cell_value == 0:  # Wall
                    cell_color = wall_color
                elif cell_value == 1:  # Floor
                    cell_color = floor_color
                elif cell_value == 2:  # Lava
                    cell_color = lava_color
                elif cell_value == 3:  # Goal
                    cell_color = floor_color
                else:
                    cell_color = floor_color
                
                # Draw base cell
                rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5,
                                       edgecolor=grid_color, facecolor=cell_color, alpha=0.8)
                ax.add_patch(rect)
                
                # Add goal circle
                if cell_value == 3:
                    circle = patches.Circle((x + 0.5, y + 0.5), 0.3,
                                          facecolor=goal_color, alpha=0.9)
                    ax.add_patch(circle)
        
        # Overlay path with low opacity colored squares
        visited_positions = set()
        
        for state in path:
            x, y, orientation = state
            pos = (x, y)
            
            # Only add square if we haven't been to this position in this path
            if pos not in visited_positions:
                visited_positions.add(pos)
                
                # Add colored square with higher opacity
                path_rect = patches.Rectangle((x, y), 1, 1, linewidth=0,
                                            facecolor=color, alpha=0.6)  # Increased from 0.25 to 0.6
                ax.add_patch(path_rect)
        
        # Mark starting position with a special marker
        start_x, start_y, start_ori = source_state
        start_marker = patches.Circle((start_x + 0.5, start_y + 0.5), 0.2,
                                    facecolor='white', edgecolor='black', alpha=0.9, linewidth=2)
        ax.add_patch(start_marker)
        
        # Set up the plot
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title with path length info
        path_length = len(path) - 1  # Steps = states - 1
        ax.set_title(f'{env_id} - Seed {seed} - {ruleset.replace("_", " ").title()} Path ({path_length} steps)', 
                    fontsize=14, fontweight='bold')
        
        # Determine save path
        if save_path is None:
            save_dir = os.path.join(os.path.dirname(__file__), "dijkstras")
            os.makedirs(save_dir, exist_ok=True)
            plot_save_path = os.path.join(save_dir, f"dijkstra_{ruleset}_seed_{seed}.png")
        else:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            plot_save_path = os.path.join(save_dir, f"dijkstra_{ruleset}_seed_{seed}.png")
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        save_paths.append(plot_save_path)
        print(f"{ruleset.replace('_', ' ').title()} path visualization saved to: {plot_save_path}")
    
    # Close environment
    env.close()
    
    return save_paths


def generate_dijkstra_visualizations(seeds=None):
    """Generate dijkstra path visualizations for multiple seeds."""
    
    if seeds is None:
        # Use some training and evaluation seeds
        seeds = [12345, 12346, 81102, 81103]
    
    print("Generating Dijkstra path visualizations...")
    print("Rulesets: conservative, standard, dangerous_3, dangerous_1")
    print("Colors: More distinct magma colormap values with higher opacity")
    print("Each ruleset will be plotted separately")
    print()
    
    total_plots = 0
    for i, seed in enumerate(seeds):
        print(f"Processing seed {i+1}/{len(seeds)}: {seed}")
        try:
            save_paths = plot_dijkstra_paths(seed)
            total_plots += len(save_paths)
            print(f"Generated {len(save_paths)} plots for seed {seed}")
        except Exception as e:
            print(f"Failed to process seed {seed}: {e}")
        print()
    
    print("Dijkstra path visualizations completed!")
    print(f"Generated {total_plots} total plots in the dijkstras/ folder.")
    print("Each plot shows one ruleset's path with:")
    print("- Higher opacity (0.6) for better visibility")
    print("- More distinct magma colors")
    print("- Path length information in title")


if __name__ == "__main__":
    # Test with a few seeds
    generate_dijkstra_visualizations() 
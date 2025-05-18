#!/usr/bin/env python3
"""
Test script to compare environment layouts between Dijkstra logs and agent evaluation logs
to verify if there's a seed mismatch or layout issue.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def load_json_file(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)

def display_layout(layout, title, save_path=None):
    """
    Visualize the environment layout using matplotlib.
    
    Args:
        layout: A 2D array with string cell types
        title: Title for the plot
        save_path: Path to save the visualization (optional)
    """
    # Create a color map for cell types
    color_map = {"floor": 0, "wall": 1, "lava": 2, "goal": 3, "unknown": 4}
    colors = ['lightgray', 'darkgray', 'red', 'green', 'purple']
    
    # Convert layout to numpy array
    layout_array = np.array(layout)
    height, width = layout_array.shape
    
    # Create numeric representation
    numeric_tensor = np.zeros((height, width), dtype=int)
    
    # Convert string types to numeric values
    for y in range(height):
        for x in range(width):
            numeric_tensor[y, x] = color_map.get(layout_array[y, x], 4)
    
    # Plot the tensor
    plt.figure(figsize=(8, 8))
    cmap = ListedColormap(colors)
    plt.imshow(numeric_tensor, cmap=cmap, origin='upper')
    
    # Add grid lines and labels
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    for y in range(height):
        for x in range(width):
            plt.text(x, y, layout_array[y, x][0].upper(), 
                     ha='center', va='center', color='black', fontweight='bold')
    
    # Add coordinate labels and legend
    plt.xticks(np.arange(width), np.arange(width))
    plt.yticks(np.arange(height), np.arange(height))
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f"{label} ({label[0].upper()})")
        for i, label in enumerate(["floor", "wall", "lava", "goal", "unknown"])
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def display_diff_layout(dijkstra_layout, agent_layout, title, save_path=None):
    """
    Create a visualization showing the differences between layouts.
    
    Args:
        dijkstra_layout: Layout from Dijkstra logs
        agent_layout: Layout from agent logs
        title: Title for the plot
        save_path: Path to save the visualization (optional)
    """
    # Convert layouts to numpy arrays
    dijkstra_array = np.array(dijkstra_layout)
    agent_array = np.array(agent_layout)
    
    # Create a difference array
    diff_array = np.zeros_like(dijkstra_array, dtype=object)
    
    height, width = dijkstra_array.shape
    for y in range(height):
        for x in range(width):
            if dijkstra_array[y, x] != agent_array[y, x]:
                diff_array[y, x] = f"D:{dijkstra_array[y, x][0]}/A:{agent_array[y, x][0]}"
            else:
                diff_array[y, x] = dijkstra_array[y, x]
    
    # Create a custom color map for the difference visualization
    color_map = {
        "floor": 0, 
        "wall": 1, 
        "lava": 2, 
        "goal": 3, 
        "unknown": 4,
        "diff": 5
    }
    colors = ['lightgray', 'darkgray', 'red', 'green', 'purple', 'orange']
    
    # Create numeric representation
    numeric_tensor = np.zeros((height, width), dtype=int)
    
    # Convert string types to numeric values
    for y in range(height):
        for x in range(width):
            if dijkstra_array[y, x] != agent_array[y, x]:
                numeric_tensor[y, x] = color_map["diff"]
            else:
                numeric_tensor[y, x] = color_map.get(dijkstra_array[y, x], 4)
    
    # Plot the tensor
    plt.figure(figsize=(10, 10))
    cmap = ListedColormap(colors)
    plt.imshow(numeric_tensor, cmap=cmap, origin='upper')
    
    # Add grid lines
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    
    # Add text labels for differences
    for y in range(height):
        for x in range(width):
            if dijkstra_array[y, x] != agent_array[y, x]:
                plt.text(x, y, f"D:{dijkstra_array[y, x][0]}/A:{agent_array[y, x][0]}", 
                        ha='center', va='center', color='black', fontsize=6, fontweight='bold')
            else:
                plt.text(x, y, dijkstra_array[y, x][0], 
                        ha='center', va='center', color='black', fontweight='bold')
    
    # Add coordinate labels and legend
    plt.xticks(np.arange(width), np.arange(width))
    plt.yticks(np.arange(height), np.arange(height))
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f"{label}")
        for i, label in enumerate(["floor", "wall", "lava", "goal", "unknown", "difference"])
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def print_layout_differences(dijkstra_layout, agent_layout):
    """
    Print the specific differences between two layouts.
    
    Args:
        dijkstra_layout: Layout from Dijkstra logs
        agent_layout: Layout from agent logs
    """
    dijkstra_array = np.array(dijkstra_layout)
    agent_array = np.array(agent_layout)
    
    print("\nSpecific differences (y, x, Dijkstra, Agent):")
    diff_count = 0
    
    for y in range(dijkstra_array.shape[0]):
        for x in range(dijkstra_array.shape[1]):
            if dijkstra_array[y, x] != agent_array[y, x]:
                diff_count += 1
                print(f"  Position ({y}, {x}): {dijkstra_array[y, x]} vs {agent_array[y, x]}")
    
    return diff_count

def check_other_seeds(env_id, seed, agent_layout):
    """
    Check if the agent layout matches any other Dijkstra seed layout.
    
    Args:
        env_id: Environment ID
        seed: Original seed
        agent_layout: Layout from agent logs
    """
    print("\nChecking if agent layout matches other Dijkstra seeds...")
    
    # Find all Dijkstra logs for this environment
    dijkstra_dir = os.path.join(project_root, "Behaviour_Specification", "Evaluations")
    dijkstra_files = [f for f in os.listdir(dijkstra_dir) if f.startswith(env_id) and f.endswith(".json")]
    
    agent_array = np.array(agent_layout)
    
    for dijkstra_file in dijkstra_files:
        # Extract seed from filename
        try:
            file_seed = int(dijkstra_file.split("-")[-1].split(".")[0])
        except:
            continue
        
        if file_seed == seed:
            continue  # Skip the original seed
        
        dijkstra_path = os.path.join(dijkstra_dir, dijkstra_file)
        try:
            dijkstra_data = load_json_file(dijkstra_path)
            dijkstra_layout = dijkstra_data.get("environment", {}).get("layout")
            if not dijkstra_layout:
                continue
                
            dijkstra_array = np.array(dijkstra_layout)
            
            if np.array_equal(dijkstra_array, agent_array):
                print(f"  Match found! Agent layout matches Dijkstra seed {file_seed}")
                return file_seed
        except Exception as e:
            print(f"  Error checking {dijkstra_file}: {e}")
    
    print("  No matching seed found")
    return None

def compare_layouts(env_id, seed):
    """
    Compare layouts between Dijkstra logs and agent evaluation logs for the same env_id and seed.
    
    Args:
        env_id: Environment ID (e.g., "MiniGrid-LavaCrossingS11N5-v0")
        seed: Random seed
    """
    # Construct file paths
    dijkstra_path = os.path.join(project_root, "Behaviour_Specification", "Evaluations", f"{env_id}-{seed}.json")
    agent_path = os.path.join(project_root, "Agent_Storage", "LavaTests", "Standard", "evaluation_logs", f"{env_id}-{seed}.json")
    
    # Check if files exist
    if not os.path.exists(dijkstra_path):
        print(f"Dijkstra log not found: {dijkstra_path}")
        return False
    
    if not os.path.exists(agent_path):
        print(f"Agent log not found: {agent_path}")
        return False
    
    # Load JSON data
    dijkstra_data = load_json_file(dijkstra_path)
    agent_data = load_json_file(agent_path)
    
    # Extract layouts
    dijkstra_layout = dijkstra_data.get("environment", {}).get("layout")
    agent_layout = agent_data.get("environment", {}).get("layout")
    
    if not dijkstra_layout or not agent_layout:
        print("Layout data not found in one or both files")
        return False
    
    # Convert to numpy arrays for comparison
    dijkstra_layout_array = np.array(dijkstra_layout)
    agent_layout_array = np.array(agent_layout)
    
    # Check dimensions
    if dijkstra_layout_array.shape != agent_layout_array.shape:
        print(f"Layout dimensions don't match: Dijkstra {dijkstra_layout_array.shape} vs Agent {agent_layout_array.shape}")
    
    # Compare layouts
    match = np.array_equal(dijkstra_layout_array, agent_layout_array)
    
    print(f"Layouts match: {match}")
    if not match:
        # Count differences
        diff_count = print_layout_differences(dijkstra_layout, agent_layout)
        print(f"Total differences: {diff_count} out of {dijkstra_layout_array.size}")
        
        # Check if agent layout matches any other Dijkstra seed
        matching_seed = check_other_seeds(env_id, seed, agent_layout)
        
        # Display layouts
        save_dir = os.path.join(os.path.dirname(__file__), "layout_comparison")
        
        display_layout(dijkstra_layout, f"Dijkstra Layout: {env_id}-{seed}", 
                      save_path=os.path.join(save_dir, f"dijkstra_{env_id}-{seed}.png"))
        
        display_layout(agent_layout, f"Agent Layout: {env_id}-{seed}" + 
                      (f" (matches seed {matching_seed})" if matching_seed else ""),
                      save_path=os.path.join(save_dir, f"agent_{env_id}-{seed}.png"))
        
        # Display difference layout
        display_diff_layout(dijkstra_layout, agent_layout, 
                         f"Layout Differences: {env_id}-{seed}",
                         save_path=os.path.join(save_dir, f"diff_{env_id}-{seed}.png"))
    
    return match

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare layouts between Dijkstra and agent logs")
    parser.add_argument("--env_id", default="MiniGrid-LavaCrossingS11N5-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=81102, help="Random seed")
    
    args = parser.parse_args()
    
    compare_layouts(args.env_id, args.seed) 